"""
Deterministic Decision Engine — replaces the LLM tool loop.

Execution flow per cycle:
  1. Large-move guard (hard rule, no LLM)
  2. Stability check (hard rule, no LLM)
  3. AnomalyDetector (AI pre-flight)
  4. RegimeClassifier (AI market judgment) → enter / skip_trade / skip_session
  5. Risk check (RiskManager)
  6. get_premium_balance_candidates
  7. validate_ic_params
  8. place_iron_condor
  9. Log outcome
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    outcome: str          # executed | waited | skip_trade | skip_session | error
    reason: str = ""
    trade: Optional[Dict[str, Any]] = None
    regime: Optional[str] = None
    anomaly: Optional[str] = None
    candidates_found: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str = ""

    def finish(self) -> "CycleResult":
        self.finished_at = datetime.now().isoformat()
        return self


class DecisionEngine:
    """
    Deterministic entry engine with AI advisors.

    The engine calls ToolHandlers methods directly — no LLM tool loop.
    AI is used only for anomaly detection and regime classification.
    """

    def __init__(
        self,
        tool_handlers=None,
        risk_manager=None,
        anomaly_detector=None,
        regime_classifier=None,
        decision_file_logger=None,
        mode: str = "dry_run",
        stability_fraction: float = 0.15,
        single_candle_fraction: float = 0.10,
        fallback_stability_pts: float = 8.0,
    ):
        self.tool_handlers = tool_handlers
        self.risk_manager = risk_manager
        self.anomaly_detector = anomaly_detector
        self.regime_classifier = regime_classifier
        self.decision_file_logger = decision_file_logger
        self.mode = mode
        self.stability_fraction = stability_fraction
        self.single_candle_fraction = single_candle_fraction
        self.fallback_stability_pts = fallback_stability_pts

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
            logger.warning("get_atm_iv failed — using fallback stability threshold")

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
        # Step 7: Strike selection
        # ------------------------------------------------------------------
        try:
            candidates = await self.tool_handlers.get_premium_balance_candidates()
        except Exception as exc:
            return CycleResult(outcome="waited", reason=f"candidates_fetch_failed: {exc}")

        if cycle:
            cycle.add_tool_call("get_premium_balance_candidates", {}, candidates)

        candidates_found: int = candidates.get("candidates_found", 0)
        if candidates_found == 0:
            logger.info("No balanced candidates found — skipping entry")
            return CycleResult(
                outcome="waited",
                reason="no_candidates",
                candidates_found=0,
            )

        top = candidates["top_candidate"]

        # ------------------------------------------------------------------
        # Step 8: Validate
        # ------------------------------------------------------------------
        try:
            validation = await self.tool_handlers.validate_ic_params(
                short_call_strike=top["short_call_strike"],
                long_call_strike=top["long_call_strike"],
                short_put_strike=top["short_put_strike"],
                long_put_strike=top["long_put_strike"],
            )
        except Exception as exc:
            return CycleResult(outcome="waited", reason=f"validation_error: {exc}")

        if cycle:
            cycle.add_tool_call("validate_ic_params", top, validation)

        if not validation.get("passed", False):
            failures = validation.get("failures", [])
            logger.info("Validation failed: %s", failures)
            return CycleResult(
                outcome="waited",
                reason=f"validation_failed: {failures}",
                candidates_found=candidates_found,
            )

        # ------------------------------------------------------------------
        # Step 9: Place trade
        # ------------------------------------------------------------------
        try:
            trade_result = await self.tool_handlers.place_iron_condor(
                short_call_strike=top["short_call_strike"],
                long_call_strike=top["long_call_strike"],
                short_put_strike=top["short_put_strike"],
                long_put_strike=top["long_put_strike"],
            )
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"place_order_failed: {exc}")

        if cycle:
            cycle.add_tool_call("place_iron_condor", top, trade_result)

        logger.info(
            "Trade placed: sc=%s lc=%s sp=%s lp=%s | result=%s",
            top["short_call_strike"], top["long_call_strike"],
            top["short_put_strike"], top["long_put_strike"],
            trade_result,
        )

        return CycleResult(
            outcome="executed",
            reason="trade_placed",
            trade=trade_result,
            regime=regime,
            candidates_found=candidates_found,
        )

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
