"""
Regime Classifier — AI market environment judgment before each entry cycle.

Makes a single fast LLM call and returns one of three verdicts:
  enter        → proceed with trade entry
  skip_trade   → skip this cycle only; re-evaluate next scheduled window
  skip_session → skip all remaining entry windows today

This replaces the simple candle-range rule with genuine pattern recognition:
the LLM can weigh candle shape sequence, intraday momentum, IV context,
and time-of-day in combination — something a single threshold can't do.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a market regime classifier for a 0DTE SPX iron condor trading bot.

The strategy sells short-volatility (neutral) iron condors. It profits when SPX stays range-bound.
You must classify whether the current market environment is appropriate for entry.

Return ONLY valid JSON in exactly this format:
{"verdict": "enter", "confidence": 0.85, "reasoning": "one concise sentence"}
or
{"verdict": "skip_trade", "reasoning": "one concise sentence"}
or
{"verdict": "skip_session", "reasoning": "one concise sentence"}

Verdict definitions:
- "enter": market is stable/range-bound, appropriate for short-vol entry
- "skip_trade": conditions are temporarily poor — likely to improve within 1-2 hours (skip this window only)
- "skip_session": conditions are broadly poor for the whole day — do not trade today at all

Use "skip_session" only for serious session-wide conditions:
- VIX extremely elevated (>30) suggesting persistent high volatility
- Strong directional trend sustained all morning (not just one spike)
- Major news event clearly driving sustained one-directional movement
- Erratic/whipsaw candle sequence with no stabilisation over 3+ hours

Use "skip_trade" for temporary, localised conditions:
- One spike followed by stabilisation starting (wait for next window)
- Candles tightening but momentum not fully exhausted
- Unusual single candle in otherwise stable session

Use "enter" when:
- Last 3 candles have similar closing prices (flat / slight drift)
- No strong directional momentum
- Normal intraday conditions for the time of day

Do NOT consider premium levels, strikes, or credit amounts — only market dynamics.
Respond with ONLY the JSON — no explanation, no markdown."""


class RegimeClassifier:
    """
    AI-based market regime classifier.
    Fails open to 'enter' if the LLM call fails, so a classifier outage
    does not silently block all trading.
    """

    def __init__(self, llm_client, tool_handlers=None):
        self.llm_client = llm_client
        self.tool_handlers = tool_handlers

    async def classify(self, candles: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Classify market regime. Returns dict with 'verdict' key.
        Fails open (returns 'enter') on LLM errors.
        """
        try:
            snapshot = await self._build_snapshot(candles)
            return await self._call_llm(snapshot)
        except Exception:
            logger.exception("RegimeClassifier LLM call failed — failing open (enter)")
            return {"verdict": "enter", "reasoning": "classifier_failed_open", "confidence": 0.0}

    async def _build_snapshot(self, candles: Optional[List[Dict]]) -> Dict[str, Any]:
        from datetime import datetime
        now = datetime.now()

        snap: Dict[str, Any] = {
            "time_of_day": now.strftime("%H:%M"),
            "day_of_week": now.strftime("%A"),
        }

        if candles:
            snap["candles"] = [
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
                snap["intraday_range_pts"] = round(max(closes) - min(closes), 2)
                snap["last_candle_move_pts"] = round(closes[-1] - closes[-2], 2)

        if self.tool_handlers:
            try:
                skew = await self.tool_handlers.get_skew_snapshot()
                snap["iv_ratio"] = skew.get("iv_ratio")
                snap["credit_ratio"] = skew.get("credit_ratio")
                snap["skew_verdict"] = skew.get("verdict")
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
            if result.get("verdict") in ("enter", "skip_trade", "skip_session"):
                logger.info(
                    "RegimeClassifier: verdict=%s confidence=%s reasoning=%s",
                    result.get("verdict"),
                    result.get("confidence", "n/a"),
                    result.get("reasoning", ""),
                )
                return result
        except (json.JSONDecodeError, AttributeError):
            pass

        logger.warning("RegimeClassifier: unparseable LLM response: %r", content[:200])
        return {"verdict": "enter", "reasoning": "unparseable_response_failing_open", "confidence": 0.0}
