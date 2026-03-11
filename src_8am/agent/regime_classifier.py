"""
ORB Regime Classifier — AI judgment before entering a trade.

Evaluates the post-09:30 breakout quality and market conditions.
Returns one of: enter | skip_trade | skip_session
Fails open to 'enter' if the LLM call fails.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a market regime classifier for an 8AM Opening Range Breakout (ORB) futures trading bot.

The strategy trades ES/MES futures: the 08:00–08:15 ET candle defines the range. After 09:30, a breakout
above the high establishes a long bias; below the low establishes a short bias. We then wait for price
to retest the midline before entering a directional trade.

Return ONLY valid JSON in exactly this format:
{"verdict": "enter", "confidence": 0.85, "reasoning": "one concise sentence"}
or
{"verdict": "skip_trade", "reasoning": "one concise sentence"}
or
{"verdict": "skip_session", "reasoning": "one concise sentence"}

Verdict definitions:
- "enter": breakout is clean and directional, midline retest likely, good setup
- "skip_trade": conditions are temporarily poor — choppy price action around ORB levels
- "skip_session": conditions are broadly poor for the day — do not trade at all

Use "skip_session" only for:
- ORB range is too narrow (< 2 pts) or too wide (> 30 pts) — unreliable range
- Major economic event (FOMC, CPI, NFP) clearly driving erratic moves
- Breakout reversed immediately and price is whipsawing across the ORB range

Use "skip_trade" for:
- Breakout looks real but price is not pulling back cleanly to midline
- Multiple failed breakout attempts (both high and low tested)

Use "enter" when:
- ORB range is meaningful (2–30 pts)
- Breakout above high or below low is clean (not immediately reversed)
- Post-breakout candles show directional momentum consistent with bias
- No major economic event driving irrational moves

Respond with ONLY the JSON — no explanation, no markdown."""


class OrbRegimeClassifier:
    """
    AI-based regime classifier for ORB setups.
    Fails open to 'enter' on LLM errors.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def classify(
        self,
        orb_levels: Dict[str, Any],
        candles: Optional[List[Dict]] = None,
        bias: Optional[str] = None,
        breakout_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classify whether conditions are good for entry.
        Returns dict with 'verdict' key.
        """
        try:
            snapshot = self._build_snapshot(orb_levels, candles, bias, breakout_price)
            return await self._call_llm(snapshot)
        except Exception:
            logger.exception("OrbRegimeClassifier LLM call failed — failing open (enter)")
            return {"verdict": "enter", "reasoning": "classifier_failed_open", "confidence": 0.0}

    def _build_snapshot(
        self,
        orb_levels: Dict[str, Any],
        candles: Optional[List[Dict]],
        bias: Optional[str],
        breakout_price: Optional[float],
    ) -> Dict[str, Any]:
        from datetime import datetime
        now = datetime.now()

        snap: Dict[str, Any] = {
            "time_of_day": now.strftime("%H:%M"),
            "day_of_week": now.strftime("%A"),
            "orb_high": orb_levels.get("high"),
            "orb_low": orb_levels.get("low"),
            "orb_midline": orb_levels.get("midline"),
            "orb_range_pts": orb_levels.get("range_pts"),
            "bias": bias,
            "breakout_price": breakout_price,
        }

        if candles:
            snap["recent_5min_candles"] = [
                {
                    "open": round(c.get("open", 0), 2),
                    "high": round(c.get("high", 0), 2),
                    "low": round(c.get("low", 0), 2),
                    "close": round(c.get("close", 0), 2),
                    "volume": c.get("volume"),
                }
                for c in candles[-10:]
            ]
            closes = [c.get("close", 0) for c in candles]
            if len(closes) >= 2:
                snap["post_breakout_range_pts"] = round(max(closes) - min(closes), 2)
                snap["last_candle_move_pts"] = round(closes[-1] - closes[-2], 2)

        return snap

    async def _call_llm(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(snapshot, default=str)},
        ]
        response = await self.llm_client.chat(messages, tools=None)
        content = (response.get("content") or "").strip()

        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()

        try:
            result = json.loads(content)
            if result.get("verdict") in ("enter", "skip_trade", "skip_session"):
                logger.info(
                    "OrbRegimeClassifier: verdict=%s confidence=%s reasoning=%s",
                    result.get("verdict"),
                    result.get("confidence", "n/a"),
                    result.get("reasoning", ""),
                )
                return result
        except (json.JSONDecodeError, AttributeError):
            pass

        logger.error("OrbRegimeClassifier: unparseable response: %r", content[:200])
        return {"verdict": "enter", "reasoning": "unparseable_response_failing_open", "confidence": 0.0}
