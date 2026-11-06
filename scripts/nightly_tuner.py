"""
Nightly Config Tuner — learns from N days of trade data and suggests config changes.

Run at 5pm via cron:
  python -m scripts.nightly_tuner

Reads last 14 days of:
  - logs/decisions/YYYY-MM-DD/*.json  (cycle outcomes, skip reasons, candidate credits)
  - SQLite IronCondorTrade records     (entry strikes, credits, PnL, close_reason)

Writes suggested config changes to logs/tuning/YYYY-MM-DD.md.
Changes are NEVER applied automatically — human review required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a quantitative trading strategy tuner for a 0DTE SPX iron condor bot.

You will receive a JSON summary of the last N days of trading activity.
Your job is to suggest specific, evidence-based changes to the bot's config thresholds.

Write your output in markdown with this structure:

## Config Tuning Suggestions — {DATE}
### Data summary
<2-3 sentences on the key patterns you see>

### Suggested changes
For each suggestion:
**`config_key`**: current_value → **new_value**
Reason: <one sentence with supporting evidence from the data>

### No-change items
List any thresholds that appear correctly tuned with brief justification.

### Caveats
Note any data quality issues or insufficient sample sizes.

Rules:
- Only suggest changes backed by clear evidence in the data
- Minimum 10 trade samples before suggesting credit threshold changes
- Minimum 5 samples for timing/stability threshold changes
- If data is insufficient, say so clearly and suggest what to track
- Never suggest disabling safety guards (anomaly detector, large-move guard)
- Be conservative: suggest small incremental changes, not large jumps
"""


async def _fetch_trades(db_path: str, since: date) -> List[Dict[str, Any]]:
    """Read trade records from SQLite."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM iron_condor_trades
               WHERE entry_time >= ?
               ORDER BY entry_time""",
            (since.isoformat(),),
        )
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Could not fetch trades from DB: %s", exc)
        return []


def _read_decision_cycles(decisions_dir: Path, since: date) -> List[Dict[str, Any]]:
    """Read decision cycle JSON files for the period."""
    cycles = []
    for day_offset in range((date.today() - since).days + 1):
        day = since + timedelta(days=day_offset)
        day_dir = decisions_dir / day.isoformat()
        if not day_dir.exists():
            continue
        for json_file in sorted(day_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    doc = json.load(f)
                cycle = {
                    "date": day.isoformat(),
                    "time": doc.get("started_at", "")[:16],
                    "outcome": doc.get("outcome"),
                    "tool_results": {},
                }
                for tc in doc.get("tool_calls", []):
                    name = tc.get("name", "")
                    result = tc.get("result", {})
                    if name == "get_premium_balance_candidates":
                        cycle["tool_results"]["candidates"] = {
                            "found": result.get("candidates_found", 0),
                            "top_credit_diff": (
                                result.get("top_candidate", {}) or {}
                            ).get("credit_diff"),
                            "best_credit_diff_available": min(
                                (c.get("credit_diff", 99) for c in result.get("candidates", [])),
                                default=None,
                            ),
                        }
                    if name == "get_skew_snapshot":
                        cycle["tool_results"]["skew"] = {
                            "iv_ratio": result.get("iv_ratio"),
                            "credit_ratio": result.get("credit_ratio"),
                            "verdict": result.get("verdict"),
                        }
                    if name == "regime_classifier":
                        cycle["tool_results"]["regime"] = result.get("verdict")
                    if name == "anomaly_detector":
                        cycle["tool_results"]["anomaly"] = result.get("status")
                cycles.append(cycle)
            except Exception:
                logger.warning("Could not read %s", json_file)
    return cycles


def _build_summary(
    cycles: List[Dict],
    trades: List[Dict],
    current_config: Dict[str, Any],
    lookback_days: int,
) -> Dict[str, Any]:
    """Build a compact summary for the LLM."""
    executed = [c for c in cycles if c["outcome"] == "executed"]
    waited = [c for c in cycles if c["outcome"] == "waited"]
    skip_session = [c for c in cycles if c["outcome"] == "skip_session"]
    skip_trade = [c for c in cycles if c["outcome"] == "skip_trade"]

    # Cycles where no candidates were found — collect best available credit_diff
    no_candidates = [
        c for c in cycles
        if c.get("tool_results", {}).get("candidates", {}).get("found", -1) == 0
    ]
    best_diffs_when_no_candidate = [
        c["tool_results"]["candidates"]["best_credit_diff_available"]
        for c in no_candidates
        if c["tool_results"]["candidates"].get("best_credit_diff_available") is not None
    ]

    # Trade outcomes
    wins = [t for t in trades if (t.get("realized_pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("realized_pnl") or 0) < 0]
    stop_triggered = [t for t in trades if t.get("close_reason") == "stop_triggered"]
    take_profit = [t for t in trades if t.get("close_reason") == "take_profit"]

    return {
        "period_days": lookback_days,
        "total_cycles": len(cycles),
        "executed": len(executed),
        "waited": len(waited),
        "skip_session": len(skip_session),
        "skip_trade": len(skip_trade),
        "no_candidate_cycles": len(no_candidates),
        "best_credit_diffs_when_no_candidate": sorted(best_diffs_when_no_candidate)[:20],
        "trades_total": len(trades),
        "trades_win": len(wins),
        "trades_loss": len(losses),
        "trades_stop_triggered": len(stop_triggered),
        "trades_take_profit": len(take_profit),
        "avg_pnl": (
            round(sum(t.get("realized_pnl") or 0 for t in trades) / len(trades), 2)
            if trades else None
        ),
        "avg_call_credit": (
            round(sum(t.get("call_spread_premium") or 0 for t in trades) / len(trades), 3)
            if trades else None
        ),
        "avg_put_credit": (
            round(sum(t.get("put_spread_premium") or 0 for t in trades) / len(trades), 3)
            if trades else None
        ),
        "current_config": current_config,
    }


async def run_tuner(
    decisions_dir: str = "logs/decisions",
    tuning_dir: str = "logs/tuning",
    db_path: str = "0dte_bot.db",
    config_path: str = "config/config.yaml",
    lookback_days: int = 14,
):
    import yaml
    from agentic_ic.services.llm_client import LLMClient

    # Load config for current thresholds
    current_config: Dict[str, Any] = {}
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        current_config = {
            **cfg.get("strategy", {}),
            **cfg.get("entry", {}),
        }
    except Exception as exc:
        logger.warning("Could not load config: %s", exc)

    since = date.today() - timedelta(days=lookback_days)

    cycles = _read_decision_cycles(Path(decisions_dir), since)
    trades = await _fetch_trades(db_path, since)
    summary = _build_summary(cycles, trades, current_config, lookback_days)

    logger.info(
        "Tuner summary: %d cycles, %d trades, %d no-candidate cycles",
        len(cycles), len(trades), summary["no_candidate_cycles"],
    )

    llm_model = os.environ.get("TUNER_MODEL", "anthropic/claude-sonnet-4-5")
    llm = LLMClient(model=llm_model)

    today = date.today().isoformat()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.replace("{DATE}", today)},
        {
            "role": "user",
            "content": (
                f"Trading data for the last {lookback_days} days (since {since.isoformat()}):\n\n"
                + json.dumps(summary, indent=2, default=str)
            ),
        },
    ]

    response = await llm.chat(messages, tools=None)
    markdown = (response.get("content") or "").strip()
    if not markdown:
        logger.error("LLM returned empty response")
        return

    out_dir = Path(tuning_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}.md"
    with open(out_path, "w") as f:
        f.write(markdown)

    logger.info("Tuning suggestions written to %s", out_path)
    print(f"\n{markdown}\n")


def main():
    parser = argparse.ArgumentParser(description="Nightly config tuner")
    parser.add_argument("--decisions-dir", default="logs/decisions")
    parser.add_argument("--tuning-dir", default="logs/tuning")
    parser.add_argument("--db", default="0dte_bot.db")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--lookback-days", type=int, default=14)
    args = parser.parse_args()

    asyncio.run(run_tuner(
        decisions_dir=args.decisions_dir,
        tuning_dir=args.tuning_dir,
        db_path=args.db,
        config_path=args.config,
        lookback_days=args.lookback_days,
    ))


if __name__ == "__main__":
    main()
