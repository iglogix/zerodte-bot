"""
Anomaly Detector — pre-flight AI safety check before each entry cycle.

Makes a single fast LLM call with a compact market snapshot and returns
{"status": "ok"} or {"status": "anomaly", "reason": "..."}.

Catches things rules can't enumerate:
- Streamer frozen (all IVs identical or zero)
- Bid > Ask on multiple strikes
- SPX moved abnormally in one candle
- Options market data missing / all zeros
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a market data quality checker for an options trading bot.
You will receive a compact market snapshot and must decide if the data looks structurally valid.

Return ONLY valid JSON in exactly this format:
{"status": "ok"}
or
{"status": "anomaly", "reason": "<concise one-line description>"}

Flag as anomaly if you see ANY of:
- bid > ask on 2 or more option strikes
- all implied volatilities are identical (streamer likely frozen)
- any implied volatility is 0 or missing across the whole chain
- SPX moved more than 2% in a single 5-minute candle
- option chain has fewer than 4 strikes total
- all option bids are 0 (no quotes)

Do NOT flag as anomaly for normal conditions like high skew, low premiums, or wide spreads.
Respond with ONLY the JSON — no explanation, no markdown."""


class AnomalyDetector:
    """
    Pre-flight market data sanity check using a fast/cheap LLM.
    Fails open (returns ok) if the LLM call fails, to avoid blocking on outages.
    """

    def __init__(self, llm_client, tool_handlers=None):
        self.llm_client = llm_client
        self.tool_handlers = tool_handlers

    async def check(self, candles: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run anomaly check. Returns {"status": "ok"} or {"status": "anomaly", "reason": "..."}.
        Fails open on LLM errors.
        """
        try:
            snapshot = await self._build_snapshot(candles)
            return await self._call_llm(snapshot)
        except Exception:
            logger.exception("AnomalyDetector LLM call failed — failing open")
            return {"status": "ok", "note": "detector_failed_open"}

    async def _build_snapshot(self, candles: Optional[List[Dict]]) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}

        if candles:
            snap["last_3_candles"] = [
                {
                    "open": c.get("open"),
                    "high": c.get("high"),
                    "low": c.get("low"),
                    "close": c.get("close"),
                }
                for c in candles[-3:]
            ]

        if self.tool_handlers:
            try:
                spx = await self.tool_handlers.get_spx_price()
                snap["spx_price"] = spx.get("price")
                snap["spx_bid"] = spx.get("bid")
                snap["spx_ask"] = spx.get("ask")
            except Exception:
                pass

            try:
                chain = await self.tool_handlers.get_option_chain(
                    min_delta=0.05,
                    max_delta=0.25,
                )
                snap["chain_count"] = len(chain)
                snap["sample_strikes"] = [
                    {
                        "strike": o.get("strike"),
                        "type": o.get("option_type"),
                        "bid": o.get("bid"),
                        "ask": o.get("ask"),
                        "iv": o.get("iv"),
                    }
                    for o in chain[:10]
                ]
            except Exception:
                pass

        return snap

    async def _call_llm(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(snapshot, default=str)},
        ]
        response = await self.llm_client.chat(messages, tools=None)
        content = (response.get("content") or "").strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(l for l in lines if not l.startswith("```")).strip()

        try:
            result = json.loads(content)
            if result.get("status") in ("ok", "anomaly"):
                return result
        except (json.JSONDecodeError, AttributeError):
            pass

        logger.warning("AnomalyDetector: unparseable LLM response: %r", content[:200])
        return {"status": "ok", "note": "unparseable_response_failing_open"}
