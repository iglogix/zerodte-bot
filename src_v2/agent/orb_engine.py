"""Three-phase ORB entry engine.

Phase A: Fetch ORB candle, compute high/low/midline.
Phase B: Poll for breakout → AI preflight → poll for midline retest → enter.
Phase C: Hand off to PositionMonitor (via orchestrator).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.agent.decision_engine import CycleResult

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class OrbEngine(BaseEngine):

    # Entry-attempt failures that happen BEFORE any order is submitted — safe
    # to retry when rearm_after_failed_entry is on. place_order_failed is
    # deliberately absent (a timed-out submit may have reached the broker).
    _REARMABLE_REASONS = frozenset({
        "no_candidates",
        "validation_failed",
        "validation_error",
        "chain_fetch_failed",
        "candidates_failed",
    })

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
        # retest_mode: where to wait for entry trigger after breakout
        #   "midline"        — wait for return to midline (default)
        #   "breakout_level" — wait for retest of broken level (continuation):
        #                      long bias retests ORB high, short bias retests ORB low
        #   "opposite_level" — wait for touch of OPPOSITE level (failed breakout):
        #                      long bias touches ORB low → reverses to short bias entry
        self._retest_mode = strat_cfg.get("retest_mode", "midline")
        # Research flag (default off): keep the same retest TRIGGER but INVERT the
        # entry bias, i.e. fade the breakout instead of trading with it. Used to
        # test the mean-reversion hypothesis on the exact same entry set.
        self._fade_breakout = bool(strat_cfg.get("fade_breakout", False))
        self._poll_interval = 1.0  # seconds between breakout/midline polls
        # On a regime skip_trade (vs skip_session), the classifier means "skip
        # this window, conditions likely improve within 1-2 hours" — so we defer
        # entry and re-evaluate the regime after this many minutes rather than
        # ending the session. Only skip_session (and EOD) terminate the watch.
        self._regime_recheck_min = strat_cfg.get("regime_recheck_minutes", 30)
        # Re-arm after a failed entry attempt (default off = legacy: any entry
        # attempt, even one that finds no candidates, ends the session). When
        # on, PRE-ORDER failures keep the watch alive and a later retest may
        # enter after a cooldown. place_order failures are never re-armed — a
        # timed-out submission may have reached the broker, and retrying would
        # stack a second spread on a position the operator thinks is one.
        self._rearm_after_failed_entry = bool(strat_cfg.get("rearm_after_failed_entry", False))
        self._rearm_cooldown_min = strat_cfg.get("rearm_cooldown_minutes", 5)
        self._entry_retry_at: Optional[datetime] = None

        self._orb_levels: Optional[Dict] = None
        self._preflight_passed = False
        self._preflight_retry_at: Optional[datetime] = None  # None = check now
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

                # AI preflight. skip_session is terminal; skip_trade / transient
                # anomaly only DEFER entry — we re-check the regime after a
                # cooldown and keep watching, instead of ending the session.
                if not self._preflight_passed and (
                    self._preflight_retry_at is None or now_et >= self._preflight_retry_at
                ):
                    block_reason = await self._run_preflight(current_price)
                    if block_reason:
                        if "skip_session" in block_reason:
                            logger.info("Preflight blocked entry (session): %s", block_reason)
                            return CycleResult(
                                outcome="skip_session", reason=block_reason,
                            ).finish()
                        # Temporary skip — defer and re-evaluate later.
                        self._preflight_retry_at = now_et + timedelta(
                            minutes=self._regime_recheck_min
                        )
                        logger.info(
                            "Preflight skip_trade — deferring entry, re-checking regime at "
                            "%s: %s", self._preflight_retry_at.strftime("%H:%M"), block_reason,
                        )
                    else:
                        self._preflight_passed = True

                # Hold off on entry until preflight has cleared (still in a
                # skip_trade cooldown → keep watching but don't enter yet).
                if not self._preflight_passed:
                    await self._clock.sleep(self._poll_interval)
                    continue

                # Determine retest level based on mode
                retest_level, entry_bias = self._resolve_retest_target(bias)
                if self.deps.tool_handlers.check_midline_touch(
                    current_price, retest_level, self._midline_tol
                ):
                    # Re-arm cooldown after a failed entry attempt — keep
                    # watching but don't hammer another attempt every poll.
                    if self._entry_retry_at and now_et < self._entry_retry_at:
                        await self._clock.sleep(self._poll_interval)
                        continue
                    logger.info(
                        "Retest touch (%s): price=%.2f level=%.2f orb_bias=%s entry_bias=%s → entering",
                        self._retest_mode, current_price, retest_level, bias, entry_bias,
                    )
                    result = await self._execute_entry(current_price, entry_bias)
                    if (
                        self._rearm_after_failed_entry
                        and result.outcome != "executed"
                        and result.reason.split(":", 1)[0] in self._REARMABLE_REASONS
                    ):
                        self._entry_retry_at = now_et + timedelta(
                            minutes=self._rearm_cooldown_min
                        )
                        logger.warning(
                            "Entry attempt failed (%s) — re-armed, next attempt "
                            "allowed at %s",
                            result.reason, self._entry_retry_at.strftime("%H:%M:%S"),
                        )
                        continue
                    return result

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

    def _resolve_retest_target(self, breakout_bias: str) -> tuple:
        """Return (retest_level, entry_bias) based on configured retest_mode.

        - midline: wait for midline retest, trade in breakout direction
        - breakout_level: wait for retest of broken level (continuation)
        - opposite_level: wait for touch of opposite level (failed breakout) —
                          INVERTS entry bias (long breakout + low touch → short entry)
        """
        levels = self._orb_levels
        inv = {"long": "short", "short": "long"}
        if self._retest_mode == "breakout_level":
            level = levels["high"] if breakout_bias == "long" else levels["low"]
            entry_bias = inv[breakout_bias] if self._fade_breakout else breakout_bias
            return level, entry_bias
        if self._retest_mode == "opposite_level":
            level = levels["low"] if breakout_bias == "long" else levels["high"]
            inverted = inv[breakout_bias]
            return level, inverted
        # default: midline
        entry_bias = inv[breakout_bias] if self._fade_breakout else breakout_bias
        return levels["midline"], entry_bias

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
        # Concurrency guard. This path never consulted the RiskManager, so
        # risk.max_concurrent_trades went unenforced. Within one process the
        # engine is single-shot, but the unit runs Restart=on-failure — so a
        # mid-session crash re-runs the entry watch and would stack a second
        # spread on top of the live one.
        if await self._at_trade_limit():
            logger.warning(
                "Open trade already at the concurrency limit — not entering; "
                "holding the session open so PositionMonitor keeps managing it"
            )
            # Must NOT return straight away: run_scheduled() stops the
            # PositionMonitor the moment engine.run() returns, which would
            # abandon the very position this guard is protecting (no
            # take-profit, no EOD close). Idle to EOD exactly as a live entry does.
            await self._idle_until_eod()
            return CycleResult(outcome="waited", reason="max_concurrent_trades").finish()

        try:
            chain = await self.deps.tool_handlers.get_option_chain(expiration="0DTE")
            spot_result = await self.deps.tool_handlers.get_orb_spot(chain, price)
            spot = spot_result.get("spot", price)
        except Exception as exc:
            return self._abort_entry(
                "error", f"chain_fetch_failed: {exc}", bias, price,
            )

        # Set bias on strategy so search_candidates uses correct EM multipliers
        self.strategy._orb_bias = bias

        try:
            candidates = await self.strategy.search_candidates(chain, spot)
        except Exception as exc:
            return self._abort_entry(
                "error", f"candidates_failed: {exc}", bias, spot,
            )

        if candidates.get("candidates_found", 0) == 0:
            return self._abort_entry(
                "waited", "no_candidates", bias, spot,
                detail={"candidates_found": 0},
            )

        top = candidates["top_candidate"]

        try:
            validation = await self.strategy.validate(top, chain)
        except Exception as exc:
            return self._abort_entry(
                "error", f"validation_error: {exc}", bias, spot,
                detail={"strikes": top.strikes, "credit": top.total_credit},
            )

        if not validation.get("passed", False):
            return self._abort_entry(
                "waited", f"validation_failed: {validation.get('failures', [])}",
                bias, spot,
                detail={"strikes": top.strikes, "credit": top.total_credit,
                        "validation": validation},
            )

        try:
            order_payload = self.strategy.build_order(top)
            trade_result = await self.deps.tool_handlers.place_order(order_payload)
        except Exception as exc:
            return self._abort_entry(
                "error", f"place_order_failed: {exc}", bias, spot,
                detail={"strikes": top.strikes, "credit": top.total_credit},
            )

        logger.info(
            "ORB trade placed: bias=%s strikes=%s — holding session open until EOD "
            "so PositionMonitor manages the position", bias, top.strikes,
        )
        # Write a per-order decision file (logs/decisions/<date>/) capturing the
        # full entry context (ORB levels, AI preflight, chosen spread, order+fill).
        self._write_order_decision(bias, spot, top, order_payload, trade_result)

        # Single-shot entry: keep the engine task alive until just past EOD.
        # run_scheduled() tears down the PositionMonitor as soon as engine.run()
        # returns, so returning here would kill monitoring ~1s after entry.
        await self._idle_until_eod()
        return CycleResult(
            outcome="executed",
            reason="orb_trade_placed",
            trade=trade_result,
            candidates_found=candidates.get("candidates_found", 0),
        ).finish()

    async def _at_trade_limit(self) -> bool:
        """True if open trades already meet risk.max_concurrent_trades.

        Fails SAFE: if the open trades cannot be read, assume the limit is hit.
        Skipping an entry costs one signal; stacking a duplicate doubles live
        risk on a position the operator thinks is a single spread.
        """
        db = getattr(self.deps, "db", None)
        if db is None:
            return False
        limit = (self.deps.config or {}).get("risk", {}).get("max_concurrent_trades", 1)
        try:
            return len(await db.get_open_trades() or []) >= limit
        except Exception:
            logger.exception("Could not read open trades — refusing to enter")
            return True

    def _abort_entry(
        self, outcome: str, reason: str, bias: str, spot: float,
        detail: Optional[Dict] = None,
    ) -> CycleResult:
        """Terminate an entry attempt WITHOUT an order: log the reason and write
        a decision file so a non-entry is as auditable as an entry. Before this
        existed, every early return here was silent — on 2026-08-21 both live
        bots exited 0 one second after a retest trigger with no trace of why."""
        log = logger.error if outcome == "error" else logger.warning
        log("Entry attempt not executed (%s): %s", outcome, reason)
        self._write_skip_decision(outcome, reason, bias, spot, detail)
        return CycleResult(outcome=outcome, reason=reason).finish()

    def _write_skip_decision(
        self, outcome: str, reason: str, bias: str, spot: float,
        detail: Optional[Dict] = None,
    ) -> None:
        """Write one JSON per NON-executed entry attempt — same folder and shape
        as order decisions, so "why didn't it trade?" is always answerable."""
        dfl = getattr(self.deps, "decision_file_logger", None)
        if not dfl:
            return
        try:
            strat_name = (self.deps.config or {}).get("bot", {}).get("strategy", "orb")
            cycle = dfl.start_cycle(
                mode=getattr(self.deps, "mode", "unknown"),
                market_data={
                    "strategy": strat_name,
                    "orb_levels": self._orb_levels,
                    "spot_at_entry": spot,
                    "bias": bias,
                },
            )
            if self._preflight_result:
                cycle.add_tool_call("ai_preflight", {}, self._preflight_result)
            cycle.add_tool_call(
                "entry_attempt", {"bias": bias},
                {"reason": reason, **(detail or {})},
            )
            cycle.finish(
                outcome=outcome,
                response=f"ORB entry not executed: {reason}",
            )
        except Exception:
            logger.exception("Failed to write ORB skip decision file")

    def _write_order_decision(self, bias, spot, setup, order_payload, trade_result) -> None:
        """Write one JSON per order to logs/decisions/<date>/ — the full entry
        context: ORB levels, AI preflight verdict, chosen spread, order + fill."""
        dfl = getattr(self.deps, "decision_file_logger", None)
        if not dfl:
            return
        try:
            strat_name = (self.deps.config or {}).get("bot", {}).get("strategy", "orb")
            cycle = dfl.start_cycle(
                mode=getattr(self.deps, "mode", "unknown"),
                market_data={
                    "strategy": strat_name,
                    "orb_levels": self._orb_levels,
                    "spot_at_entry": spot,
                    "bias": bias,
                },
            )
            if self._preflight_result:
                cycle.add_tool_call("ai_preflight", {}, self._preflight_result)
            cycle.add_tool_call(
                "selected_spread", {"bias": bias},
                {"strikes": setup.strikes, "credit": setup.total_credit,
                 **(setup.extra or {})},
            )
            cycle.add_tool_call("place_order", order_payload, trade_result)
            cycle.finish(
                outcome="executed",
                response=f"ORB {bias} entry {setup.strikes} credit={setup.total_credit}",
            )
        except Exception:
            logger.exception("Failed to write ORB order decision file")

    async def _idle_until_eod(self) -> None:
        """Block until ~5min past EOD force-close so the background PositionMonitor
        owns the full exit lifecycle (TP / breach / EOD close) after a single entry.
        The grace lets the monitor's EOD close (at self._eod) finish before
        run_scheduled() tears it down.

        NOT in backtest. That idle exists purely to outlive run_scheduled(), which
        stops the monitor the moment engine.run() returns — a live-only concern.
        BacktestOrchestrator instead runs the engine to completion and THEN walks
        the remaining bars through the monitor, so idling here consumes every one
        of those bars (each sleep advances the sim clock) and the monitor loop
        never executes: no take-profit, no breach exit, everything booked at the
        final bar. That silently inflated the survivor config from $13,751 to
        $31,207 and made tp_pct look inert.
        """
        if getattr(self.deps, "mode", "") == "backtest":
            return

        from datetime import datetime, timedelta
        while True:
            now = self._clock.now(ET)
            eod_dt = datetime.combine(now.date(), self._eod, tzinfo=ET) + timedelta(minutes=5)
            remaining = (eod_dt - now).total_seconds()
            if remaining <= 0:
                return
            await self._clock.sleep(max(1.0, min(30.0, remaining)))
