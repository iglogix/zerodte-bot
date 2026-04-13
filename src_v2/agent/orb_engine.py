"""Three-phase ORB entry engine.

Phase A: Fetch ORB candle, compute high/low/midline.
Phase B: Poll for breakout → AI preflight → poll for midline retest → enter.
Phase C: Hand off to PositionMonitor (via orchestrator).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.agent.decision_engine import CycleResult

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class OrbEngine(BaseEngine):

    def __init__(self, deps: EngineDeps, strategy):
        super().__init__(deps, strategy)
        from src_v2.backtest.sim_clock import WallClock
        self._clock = deps.clock or WallClock()
        strat_cfg = deps.config.get("strategy", {})
        self._orb_start = strat_cfg.get("orb_window_start", "08:00")
        self._orb_end = strat_cfg.get("orb_window_end", "08:15")
        self._eod = time(*[int(x) for x in strat_cfg.get("eod_force_close", "15:50").split(":")])
        self._midline_tol = strat_cfg.get("midline_tolerance_pts", 0.25)
        range_cfg = strat_cfg.get("orb_range_filter", {})
        self._min_range = range_cfg.get("min_pts", 2.0)
        self._max_range = range_cfg.get("max_pts", 30.0)
        self._orb_symbol = deps.config.get("bot", {}).get("symbol", "ES")
        self._poll_interval = 1.0  # seconds between breakout/midline polls

        self._orb_levels: Optional[Dict] = None
        self._preflight_done = False
        self._preflight_result: Optional[Dict] = None

    async def run(self) -> CycleResult:
        """Full ORB session: setup → entry watch."""
        self.deps.tool_handlers.reset_orb_state()

        if not await self._setup_phase():
            return CycleResult(outcome="waited", reason="orb_setup_failed").finish()

        return await self._entry_watch()

    async def _setup_phase(self) -> bool:
        """Phase A: fetch ORB candle, compute levels."""
        result = await self.deps.tool_handlers.get_orb_levels(
            self._orb_symbol, self._orb_start, self._orb_end,
        )
        if "error" in result:
            logger.error("ORB setup failed: %s", result["error"])
            return False

        range_pts = result.get("range_pts", 0)
        if range_pts < self._min_range:
            logger.info("ORB range %.1f < min %.1f — skipping", range_pts, self._min_range)
            return False
        if range_pts > self._max_range:
            logger.info("ORB range %.1f > max %.1f — skipping", range_pts, self._max_range)
            return False

        self._orb_levels = result
        logger.info(
            "ORB levels set: high=%.2f low=%.2f mid=%.2f range=%.1f",
            result["high"], result["low"], result["midline"], range_pts,
        )
        return True

    async def _entry_watch(self) -> CycleResult:
        """Phase B: poll for breakout → preflight → midline → enter."""
        consecutive_errors = 0

        while True:
            try:
                now_et = self._clock.now(ET)
                if now_et.time() >= self._eod:
                    logger.info("EOD reached — no entry today")
                    return CycleResult(outcome="waited", reason="eod_no_entry").finish()

                # Get current ES futures price (ORB levels are ES-based)
                spot_data = await self.deps.tool_handlers.get_futures_price(self._orb_symbol)
                current_price = spot_data.get("price", 0)
                if not current_price:
                    await self._clock.sleep(self._poll_interval)
                    continue

                # Check breakout
                breakout = await self.deps.tool_handlers.get_breakout_status(current_price)
                if breakout.get("status") != "breakout":
                    await self._clock.sleep(self._poll_interval)
                    continue

                bias = breakout["bias"]

                # Run AI preflight (once per breakout)
                if not self._preflight_done:
                    block_reason = await self._run_preflight(current_price)
                    self._preflight_done = True
                    if block_reason:
                        logger.info("Preflight blocked entry: %s", block_reason)
                        return CycleResult(
                            outcome="skip_session" if "skip_session" in block_reason else "waited",
                            reason=block_reason,
                        ).finish()

                # Check midline touch
                midline = self._orb_levels["midline"]
                if self.deps.tool_handlers.check_midline_touch(
                    current_price, midline, self._midline_tol
                ):
                    logger.info(
                        "Midline touch: price=%.2f mid=%.2f bias=%s → entering",
                        current_price, midline, bias,
                    )
                    return await self._execute_entry(current_price, bias)

                consecutive_errors = 0
                await self._clock.sleep(self._poll_interval)

            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                logger.exception("Entry watch error (%d consecutive)", consecutive_errors)
                if consecutive_errors >= 10:
                    logger.error("Too many consecutive errors — aborting")
                    return CycleResult(outcome="error", reason="too_many_errors").finish()
                await self._clock.sleep(5)

    async def _run_preflight(self, price: float) -> Optional[str]:
        """AI anomaly + regime check. Run once per breakout. Returns block reason or None."""
        # Anomaly check
        if self.deps.anomaly_detector:
            try:
                candles = await self.deps.tool_handlers.get_5min_candles(count=10)
                result = await self.deps.anomaly_detector.check(candles=candles)
                if result.get("status") == "anomaly":
                    return f"anomaly: {result.get('reason', 'unknown')}"
            except Exception:
                logger.exception("Anomaly detector failed — proceeding")

        # Regime check
        if self.deps.regime_classifier:
            try:
                candles = await self.deps.tool_handlers.get_5min_candles(count=10)
                result = await self.deps.regime_classifier.classify(candles=candles)
                verdict = result.get("verdict", "enter")
                if verdict == "skip_session":
                    return f"regime_skip_session: {result.get('reasoning', '')}"
                if verdict == "skip_trade":
                    return f"regime_skip_trade: {result.get('reasoning', '')}"
            except Exception:
                logger.exception("Regime classifier failed — proceeding")

        return None

    async def _execute_entry(self, price: float, bias: str) -> CycleResult:
        """Fetch chain + spot, set strategy._orb_bias, search → validate → place."""
        try:
            chain = await self.deps.tool_handlers.get_option_chain(expiration="0DTE")
            spot_result = await self.deps.tool_handlers.get_orb_spot(chain, price)
            spot = spot_result.get("spot", price)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"chain_fetch_failed: {exc}").finish()

        # Set bias on strategy so search_candidates uses correct EM multipliers
        self.strategy._orb_bias = bias

        try:
            candidates = await self.strategy.search_candidates(chain, spot)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"candidates_failed: {exc}").finish()

        if candidates.get("candidates_found", 0) == 0:
            return CycleResult(outcome="waited", reason="no_candidates").finish()

        top = candidates["top_candidate"]

        try:
            validation = await self.strategy.validate(top, chain)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"validation_error: {exc}").finish()

        if not validation.get("passed", False):
            return CycleResult(
                outcome="waited",
                reason=f"validation_failed: {validation.get('failures', [])}",
            ).finish()

        try:
            order_payload = self.strategy.build_order(top)
            trade_result = await self.deps.tool_handlers.place_order(order_payload)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"place_order_failed: {exc}").finish()

        logger.info("ORB trade placed: bias=%s strikes=%s", bias, top.strikes)
        return CycleResult(
            outcome="executed",
            reason="orb_trade_placed",
            trade=trade_result,
            candidates_found=candidates.get("candidates_found", 0),
        ).finish()
