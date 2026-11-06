"""
ScheduledEngine — runs decision cycles at scheduled entry windows.

Extracted from DecisionEngine. The scheduling loop (run()) and the
per-cycle logic (run_cycle / _run / _get_market_snapshot) live here.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from zerodte.agent.base_engine import BaseEngine, EngineDeps
from zerodte.agent.decision_engine import CycleResult

logger = logging.getLogger(__name__)

_NYSE_CAL = None


def _nyse_calendar():
    """Lazily build and cache the NYSE market calendar (pandas is heavy)."""
    global _NYSE_CAL
    if _NYSE_CAL is None:
        import pandas_market_calendars as mcal
        _NYSE_CAL = mcal.get_calendar("NYSE")
    return _NYSE_CAL


def is_trading_day(d: date) -> bool:
    """True if `d` is a US equity-market trading day (weekday, not a holiday).

    Weekends short-circuit without touching the calendar. Holidays are checked
    against the NYSE calendar; if that lookup fails for any reason, a weekday
    degrades to True (the entry pipeline's anomaly guard still backstops an
    actual holiday) so the scheduler never crashes on a calendar error.
    """
    if d.weekday() >= 5:  # Saturday / Sunday
        return False
    try:
        sched = _nyse_calendar().schedule(start_date=d.isoformat(), end_date=d.isoformat())
        return not sched.empty
    except Exception:
        logger.debug("is_trading_day: NYSE calendar unavailable — weekday fallback")
        return True


class ScheduledEngine(BaseEngine):
    """
    Deterministic entry engine with AI advisors, running on a schedule.

    The engine calls ToolHandlers methods directly — no LLM tool loop.
    AI is used only for anomaly detection and regime classification.
    """

    def __init__(self, deps: EngineDeps, strategy: Any):
        super().__init__(deps, strategy)
        self.tool_handlers = deps.tool_handlers
        self.risk_manager = deps.risk_manager
        self.anomaly_detector = deps.anomaly_detector
        self.regime_classifier = deps.regime_classifier
        self.decision_file_logger = deps.decision_file_logger
        self.mode = deps.mode
        entry_cfg = deps.config.get("entry", {})
        self.stability_fraction = entry_cfg.get("stability_fraction", 0.15)
        self.single_candle_fraction = entry_cfg.get("single_candle_fraction", 0.10)
        self.fallback_stability_pts = entry_cfg.get("fallback_stability_pts", 8.0)
        gate_cfg = entry_cfg.get("markov_gate") or {}
        self.markov_gate = None
        if gate_cfg.get("enabled"):
            from zerodte.agent.markov_gate import MarkovGate
            self.markov_gate = MarkovGate(gate_cfg)
        # (date, decision) — the gate uses prior closes only, so one evaluation
        # holds for every entry window of the day
        self._markov_decision: Optional[tuple] = None

    async def run(self):
        """Run the scheduled entry loop."""
        entry_cfg = self.deps.config.get("entry", {})
        schedule = entry_cfg.get("schedule",
            ["09:45", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"])
        skip_non_trading_days = entry_cfg.get("skip_non_trading_days", True)

        session_skipped = False
        session_skip_date = None
        non_trading_logged = None

        while True:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            today = now.date()

            if session_skip_date and session_skip_date < today:
                session_skipped = False
                session_skip_date = None

            if skip_non_trading_days and not is_trading_day(today):
                if non_trading_logged != today:
                    logger.info("[SCHEDULER] %s is not a trading day — entry cycles suspended",
                                today.isoformat())
                    non_trading_logged = today
                await asyncio.sleep(60)
                continue

            if current_time in schedule:
                if session_skipped:
                    logger.info("[SCHEDULER] Skipping %s — session skip active", current_time)
                else:
                    result = await self.run_cycle()
                    if result.outcome == "skip_session":
                        session_skipped = True
                        session_skip_date = date.today()
                await asyncio.sleep(61)
            else:
                await asyncio.sleep(10)

    async def run_cycle(self) -> CycleResult:
        """Run one entry decision cycle. Returns a CycleResult."""
        cycle = None
        if self.decision_file_logger:
            market_snap = await self._get_market_snapshot()
            cycle = self.decision_file_logger.start_cycle(
                mode=self.mode,
                market_data=market_snap,
            )

        result = await self._run(cycle)
        result.finish()

        if cycle:
            cycle.finish(outcome=result.outcome, response=result.reason)

        return result

    async def _run(self, cycle) -> CycleResult:
        # ------------------------------------------------------------------
        # Step 0: Markov regime gate — causal day-level pre-check
        # ------------------------------------------------------------------
        if self.markov_gate:
            decision = await self._check_markov_gate()
            if not decision["allow"]:
                return CycleResult(
                    outcome="skip_session",
                    reason=f"markov_gate: {decision['reason']}",
                )

        # ------------------------------------------------------------------
        # Step 1: fetch candles (needed by guards + regime classifier)
        # ------------------------------------------------------------------
        try:
            candles: List[Dict] = await self.tool_handlers.get_5min_candles(count=10)
        except Exception as exc:
            logger.exception("Failed to fetch candles")
            return CycleResult(outcome="error", reason=f"candle_fetch_failed: {exc}")

        if len(candles) < 3:
            return CycleResult(outcome="waited", reason="insufficient_candle_data")

        # ------------------------------------------------------------------
        # Step 2: Fetch expected daily move for volatility-relative thresholds
        # ------------------------------------------------------------------
        expected_move: Optional[float] = None
        iv_source: str = "unavailable"
        try:
            iv_data = await self.tool_handlers.get_atm_iv()
            expected_move = iv_data.get("expected_daily_move")
            iv_source = iv_data.get("source", "unavailable")
            if expected_move:
                if iv_data.get("atm_call_mid") and iv_data.get("atm_put_mid"):
                    logger.info(
                        "Expected daily move: %.1f pts (straddle: call=%.2f + put=%.2f, source=%s)",
                        expected_move,
                        iv_data["atm_call_mid"], iv_data["atm_put_mid"],
                        iv_source,
                    )
                else:
                    logger.info(
                        "Expected daily move: %.1f pts (IV=%.1f%%, source=%s)",
                        expected_move,
                        (iv_data.get("atm_iv") or 0) * 100,
                        iv_source,
                    )
        except Exception:
            logger.error("get_atm_iv failed — using fallback stability threshold")

        # Resolve thresholds: IV-relative when available, absolute fallback otherwise
        if expected_move and expected_move > 0:
            single_candle_threshold = expected_move * self.single_candle_fraction
            stability_threshold = expected_move * self.stability_fraction
        else:
            single_candle_threshold = self.fallback_stability_pts * (self.single_candle_fraction / self.stability_fraction)
            stability_threshold = self.fallback_stability_pts

        # ------------------------------------------------------------------
        # Step 3: Large-move guard — last candle > single_candle_fraction of expected move
        # ------------------------------------------------------------------
        prev_close = candles[-2].get("close", 0) if len(candles) >= 2 else 0
        last_close = candles[-1].get("close", 0)
        if prev_close > 0:
            last_candle_move = abs(last_close - prev_close)
            if last_candle_move > single_candle_threshold:
                logger.warning(
                    "Large-move guard triggered: candle_move=%.2f pts > threshold=%.2f pts (%.0f%% of expected_move=%.1f)",
                    last_candle_move, single_candle_threshold,
                    self.single_candle_fraction * 100, expected_move or 0,
                )
                return CycleResult(
                    outcome="waited",
                    reason=(
                        f"large_move_guard: candle={last_candle_move:.1f}pts > "
                        f"{self.single_candle_fraction:.0%} of expected_move={expected_move or 'n/a'}pts"
                    ),
                )

        # ------------------------------------------------------------------
        # Step 4a: Stability check — last 3 closes within stability_fraction of expected move
        # ------------------------------------------------------------------
        closes = [c.get("close", 0) for c in candles[-3:]]
        candle_range = max(closes) - min(closes)
        if candle_range > stability_threshold:
            logger.info(
                "Stability check failed: range=%.2f pts > threshold=%.2f pts (%.0f%% of expected_move=%.1f)",
                candle_range, stability_threshold,
                self.stability_fraction * 100, expected_move or 0,
            )
            return CycleResult(
                outcome="waited",
                reason=(
                    f"market_unstable: range={candle_range:.1f}pts > "
                    f"{self.stability_fraction:.0%} of expected_move={expected_move or 'n/a'}pts"
                ),
            )

        # ------------------------------------------------------------------
        # Step 4: Anomaly detector (AI pre-flight)
        # ------------------------------------------------------------------
        anomaly_reason: Optional[str] = None
        if self.anomaly_detector:
            try:
                anomaly_result = await self.anomaly_detector.check(candles=candles)
                if anomaly_result.get("status") == "anomaly":
                    anomaly_reason = anomaly_result.get("reason", "unknown_anomaly")
                    logger.warning("Anomaly detected: %s", anomaly_reason)
                    return CycleResult(
                        outcome="waited",
                        reason=f"anomaly: {anomaly_reason}",
                        anomaly=anomaly_reason,
                    )
            except Exception:
                logger.exception("AnomalyDetector failed — proceeding without check")

        # ------------------------------------------------------------------
        # Step 5: Regime classifier (AI judgment) — hard gate
        # ------------------------------------------------------------------
        regime: Optional[str] = None
        if self.regime_classifier:
            try:
                regime_result = await self.regime_classifier.classify(candles=candles)
                regime = regime_result.get("verdict", "enter")
                reasoning = regime_result.get("reasoning", "")
                logger.info("Regime classifier verdict: %s — %s", regime, reasoning)

                if regime == "skip_session":
                    return CycleResult(
                        outcome="skip_session",
                        reason=f"regime_classifier: {reasoning}",
                        regime=regime,
                    )
                if regime == "skip_trade":
                    return CycleResult(
                        outcome="skip_trade",
                        reason=f"regime_classifier: {reasoning}",
                        regime=regime,
                    )
            except Exception:
                logger.exception("RegimeClassifier failed — proceeding with entry")
                regime = "enter_fallback"

        # ------------------------------------------------------------------
        # Step 6: Risk check
        # ------------------------------------------------------------------
        if self.risk_manager:
            try:
                check = await self.risk_manager.can_open_new_trade()
                if not check.get("allowed", True):
                    return CycleResult(
                        outcome="waited",
                        reason=f"risk_limits_exceeded: {check.get('reason', '')}",
                    )
            except Exception:
                logger.exception("RiskManager check failed — blocking entry")
                return CycleResult(outcome="waited", reason="risk_check_error")

        # ------------------------------------------------------------------
        # Step 7: Strategy — search candidates
        # ------------------------------------------------------------------
        try:
            chain = await self.tool_handlers.get_option_chain(expiration="0DTE")
            spot_data = await self.tool_handlers.get_spx_price()
            spot = spot_data.get("price", 0)
            candidates = await self.strategy.search_candidates(chain, spot)
        except Exception as exc:
            return CycleResult(outcome="waited", reason=f"candidates_fetch_failed: {exc}")

        if cycle:
            cycle.add_tool_call("search_candidates", {}, candidates)

        candidates_found: int = candidates.get("candidates_found", 0)
        if candidates_found == 0:
            logger.info("No candidates found — skipping entry")
            return CycleResult(
                outcome="waited",
                reason="no_candidates",
                candidates_found=0,
            )

        top = candidates["top_candidate"]

        # ------------------------------------------------------------------
        # Step 8: Strategy — validate
        # ------------------------------------------------------------------
        try:
            validation = await self.strategy.validate(top, chain)
        except Exception as exc:
            return CycleResult(outcome="waited", reason=f"validation_error: {exc}")

        if cycle:
            cycle.add_tool_call("validate", {"setup": str(top.strikes)}, validation)

        if not validation.get("passed", False):
            failures = validation.get("failures", [])
            logger.info("Validation failed: %s", failures)
            return CycleResult(
                outcome="waited",
                reason=f"validation_failed: {failures}",
                candidates_found=candidates_found,
            )

        # ------------------------------------------------------------------
        # Step 9: Strategy — build and place order
        # ------------------------------------------------------------------
        try:
            order_payload = self.strategy.build_order(top)
            trade_result = await self.tool_handlers.place_order(order_payload)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"place_order_failed: {exc}")

        if cycle:
            cycle.add_tool_call("place_order", order_payload, trade_result)

        logger.info("Trade placed: %s | result=%s", top.strikes, trade_result)

        return CycleResult(
            outcome="executed",
            reason="trade_placed",
            trade=trade_result,
            regime=regime,
            candidates_found=candidates_found,
        )

    async def _check_markov_gate(self) -> Dict[str, Any]:
        """Evaluate the Markov regime gate, once per session day.

        Fails OPEN on any data problem: a broken daily-candle feed must
        degrade to ungated behaviour, never to a silent permanent skip.
        """
        assert self.markov_gate is not None  # only called when configured
        today = date.today()
        if self._markov_decision and self._markov_decision[0] == today:
            return self._markov_decision[1]
        try:
            closes = await self.tool_handlers.get_daily_closes(
                count=self.markov_gate.min_closes + 5)
            decision = self.markov_gate.evaluate(closes)
        except Exception as exc:
            logger.warning("Markov gate data fetch failed — failing open: %r", exc)
            decision = {"allow": True, "reason": f"data_error_fail_open: {exc}"}
        (logger.warning if not decision["allow"] else logger.info)(
            "Markov gate: %s — %s",
            "BLOCKED (skip session)" if not decision["allow"] else "allowed",
            decision["reason"],
        )
        self._markov_decision = (today, decision)
        return decision

    async def _get_market_snapshot(self) -> Dict[str, Any]:
        """Build a compact market snapshot for the decision log."""
        snap: Dict[str, Any] = {"timestamp": datetime.now().isoformat(), "mode": self.mode}
        try:
            quote = await self.tool_handlers.get_spx_price()
            snap["spx_price"] = quote.get("price")
        except Exception:
            pass
        try:
            snap["recent_candles"] = await self.tool_handlers.get_5min_candles(count=6)
        except Exception:
            pass
        return snap
