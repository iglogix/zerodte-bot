"""
EOD Debrief — single LLM call at market close to summarise the day's activity.

Reads today's decision cycle JSON files + trade outcomes from DB and produces
a human-readable markdown summary with observations and config suggestions.
Written to logs/summaries/YYYY-MM-DD.md and sent via the notifier.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a trading performance analyst for a 0DTE SPX iron condor bot.

You will receive a JSON summary of today's trading activity. Write a concise EOD debrief in markdown.

Format exactly as:
## {DATE} EOD Summary
- Entries attempted: N | Executed: N | Skipped: N
- Skip breakdown: <reason: count, reason: count, ...>
- <one line per executed trade with strikes, credit collected, and outcome>
- Net P&L: $X
- Observations: <2-3 bullet points about what drove today's results>
- Suggestions: <1-3 specific, actionable config changes to consider, with reasoning>

Be concise. Focus on actionable observations. If no trades were executed, say why and suggest fixes.
Do not invent data — only use what is provided."""


class EodDebrief:
    """
    Generates an end-of-day performance summary using an LLM.
    Called once at EOD close by PositionMonitor.
    """

    def __init__(
        self,
        llm_client,
        db=None,
        notifier=None,
        decisions_dir: str = "logs/decisions",
        summaries_dir: str = "logs/summaries",
    ):
        self.llm_client = llm_client
        self.db = db
        self.notifier = notifier
        self.decisions_dir = Path(decisions_dir)
        self.summaries_dir = Path(summaries_dir)

    async def run(self, session_date: Optional[date] = None) -> Optional[str]:
        """
        Generate and save the EOD debrief for session_date (defaults to today).
        Returns the markdown content, or None on failure.
        """
        if session_date is None:
            session_date = date.today()

        try:
            summary = await self._build_summary(session_date)
            markdown = await self._call_llm(summary, session_date)
            await self._save(markdown, session_date)
            await self._ingest_brain(markdown, session_date)
            await self._notify(markdown, session_date)
            return markdown
        except Exception:
            logger.exception("EodDebrief failed for %s", session_date)
            return None

    async def _build_summary(self, session_date: date) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "date": session_date.isoformat(),
            "cycles": [],
            "trades": [],
        }

        # Read decision cycle JSON files for today
        day_dir = self.decisions_dir / session_date.isoformat()
        if day_dir.exists():
            for json_file in sorted(day_dir.glob("*.json")):
                try:
                    with open(json_file) as f:
                        doc = json.load(f)
                    cycle_summary = {
                        "time": doc.get("started_at", "")[:16],
                        "outcome": doc.get("outcome"),
                        "reason": None,
                    }
                    # Extract skip reason from primary_response if present
                    primary = doc.get("primary_response")
                    if isinstance(primary, list):
                        primary = " ".join(primary)
                    if primary:
                        cycle_summary["reason"] = primary[:120]

                    # Extract candidates_found if present
                    for tc in doc.get("tool_calls", []):
                        if tc.get("name") == "get_premium_balance_candidates":
                            cycle_summary["candidates_found"] = tc.get("result", {}).get("candidates_found")
                        if tc.get("name") == "get_skew_snapshot":
                            r = tc.get("result", {})
                            cycle_summary["iv_ratio"] = r.get("iv_ratio")
                            cycle_summary["credit_ratio"] = r.get("credit_ratio")

                    summary["cycles"].append(cycle_summary)
                except Exception:
                    logger.error("Could not read decision file %s", json_file)

        # Read trade outcomes from DB
        if self.db:
            try:
                from agentic_ic.services.db_service import DatabaseService
                all_trades = await self.db.get_all_trades_for_date(session_date)
                for t in all_trades:
                    summary["trades"].append({
                        "entry_time": str(t.get("entry_time", ""))[:16],
                        "short_call": t.get("short_call_strike"),
                        "long_call": t.get("long_call_strike"),
                        "short_put": t.get("short_put_strike"),
                        "long_put": t.get("long_put_strike"),
                        "call_credit": t.get("call_spread_premium"),
                        "put_credit": t.get("put_spread_premium"),
                        "total_credit": t.get("total_premium"),
                        "realized_pnl": t.get("realized_pnl"),
                        "close_reason": t.get("close_reason"),
                        "status": t.get("status"),
                    })
            except Exception:
                logger.error("Could not fetch trades from DB for EOD debrief")

        # Compute aggregates
        executed = [c for c in summary["cycles"] if c["outcome"] == "executed"]
        skipped = [c for c in summary["cycles"] if c["outcome"] != "executed"]
        skip_reasons: Dict[str, int] = {}
        for c in skipped:
            reason = (c.get("reason") or c.get("outcome") or "unknown").split(":")[0].strip()
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        total_pnl = sum(
            t.get("realized_pnl") or 0
            for t in summary["trades"]
            if t.get("realized_pnl") is not None
        )

        summary["aggregates"] = {
            "cycles_total": len(summary["cycles"]),
            "executed": len(executed),
            "skipped": len(skipped),
            "skip_reasons": skip_reasons,
            "net_pnl": round(total_pnl, 2),
        }

        return summary

    async def _call_llm(self, summary: Dict[str, Any], session_date: date) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Today's trading summary for {session_date.isoformat()}:\n\n"
                    + json.dumps(summary, indent=2, default=str)
                ),
            },
        ]
        response = await self.llm_client.chat(messages, tools=None)
        content = (response.get("content") or "").strip()
        if not content:
            return f"## {session_date.isoformat()} EOD Summary\n\nNo summary generated."
        return content

    async def _save(self, markdown: str, session_date: date):
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.summaries_dir / f"{session_date.isoformat()}.md"
        with open(out_path, "w") as f:
            f.write(markdown)
        logger.info("EOD debrief written: %s", out_path)

    async def _ingest_brain(self, markdown: str, session_date: date):
        """Best-effort POST of the debrief to the LightRAG brain. Silent on failure."""
        url = os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621").rstrip("/")
        api_key = os.environ.get("LIGHTRAG_API_KEY")
        if not api_key:
            logger.debug("LIGHTRAG_API_KEY not set; skipping brain ingest")
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        source = f"eod-debrief:0dte-bot:{session_date.isoformat()}:{ts}"
        payload = {"text": markdown, "file_source": source}
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
                resp = await client.post(
                    f"{url}/documents/text",
                    headers={"X-API-Key": api_key, "Content-Type": "application/json"},
                    json=payload,
                )
                if 200 <= resp.status_code < 300:
                    logger.info("EOD debrief ingested to brain (source=%s)", source)
                else:
                    logger.warning("Brain ingest returned %s: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Brain ingest failed: %s", e)

    async def _notify(self, markdown: str, session_date: date):
        if self.notifier:
            try:
                await self.notifier.notify_eod_summary({
                    "date": session_date.isoformat(),
                    "summary": markdown,
                })
            except Exception:
                logger.error("Failed to send EOD debrief notification")
