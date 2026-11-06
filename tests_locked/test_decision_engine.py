"""
Unit tests for DecisionEngine, AnomalyDetector, and RegimeClassifier.
All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentic_ic.agent.decision_engine import DecisionEngine, CycleResult
from agentic_ic.agent.anomaly_detector import AnomalyDetector
from agentic_ic.agent.regime_classifier import RegimeClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(closes, base=6900.0):
    """Build minimal candle dicts from a list of close prices."""
    candles = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c
        candles.append({
            "open": prev, "high": max(prev, c) + 0.5,
            "low": min(prev, c) - 0.5, "close": c, "volume": 100000,
        })
    return candles


# ATM straddle: call_mid=39.5 + put_mid=38.7 = 78.2 pts expected move
# stability_fraction=0.15 → stability_threshold ≈ 11.7 pts
# single_candle_fraction=0.10 → single_candle_threshold ≈ 7.8 pts
_MOCK_IV_DATA = {
    "expected_daily_move": 78.2,
    "atm_call_mid": 39.5,
    "atm_put_mid": 38.7,
    "atm_iv": 0.18,
    "spx_price": 6900.0,
    "source": "straddle",
}


def _make_engine(
    candles=None,
    candidates=None,
    validation=None,
    anomaly_status="ok",
    regime_verdict="enter",
    risk_allowed=True,
    iv_data=None,
):
    flat_closes = [6900.0, 6900.5, 6901.0, 6900.8, 6901.2,
                   6900.9, 6901.1, 6900.7, 6901.0, 6901.3]
    tool_handlers = AsyncMock()
    tool_handlers.get_5min_candles = AsyncMock(return_value=candles or _make_candles(flat_closes))
    tool_handlers.get_spx_price = AsyncMock(return_value={"price": 6901.0})
    tool_handlers.get_atm_iv = AsyncMock(return_value=iv_data or _MOCK_IV_DATA)
    tool_handlers.get_premium_balance_candidates = AsyncMock(return_value=candidates or {
        "candidates_found": 1,
        "top_candidate": {
            "short_call_strike": 6930.0, "long_call_strike": 6960.0,
            "short_put_strike": 6870.0, "long_put_strike": 6840.0,
            "call_credit": 0.75, "put_credit": 0.70,
        },
    })
    tool_handlers.validate_ic_params = AsyncMock(return_value=validation or {
        "passed": True, "failures": [],
    })
    tool_handlers.place_iron_condor = AsyncMock(return_value={
        "status": "filled", "trade_id": "T001",
    })

    risk_manager = AsyncMock()
    risk_manager.can_open_new_trade = AsyncMock(
        return_value={"allowed": risk_allowed, "reason": "max_concurrent_trades"}
        if not risk_allowed
        else {"allowed": True}
    )

    anomaly_detector = AsyncMock()
    anomaly_detector.check = AsyncMock(return_value={"status": anomaly_status})

    regime_classifier = AsyncMock()
    regime_classifier.classify = AsyncMock(return_value={
        "verdict": regime_verdict,
        "reasoning": f"test_{regime_verdict}",
        "confidence": 0.9,
    })

    engine = DecisionEngine(
        tool_handlers=tool_handlers,
        risk_manager=risk_manager,
        anomaly_detector=anomaly_detector,
        regime_classifier=regime_classifier,
        stability_fraction=0.15,
        single_candle_fraction=0.10,
        fallback_stability_pts=8.0,
    )
    return engine


# ---------------------------------------------------------------------------
# DecisionEngine tests
# ---------------------------------------------------------------------------

class TestDecisionEngineHappyPath:
    @pytest.mark.asyncio
    async def test_full_happy_path_executes(self):
        engine = _make_engine()
        result = await engine.run_cycle()
        assert result.outcome == "executed"
        assert result.trade is not None

    @pytest.mark.asyncio
    async def test_executed_result_has_trade_id(self):
        engine = _make_engine()
        result = await engine.run_cycle()
        assert result.trade.get("trade_id") == "T001"

    @pytest.mark.asyncio
    async def test_calls_place_iron_condor_with_top_candidate(self):
        engine = _make_engine()
        await engine.run_cycle()
        engine.tool_handlers.place_iron_condor.assert_called_once_with(
            short_call_strike=6930.0, long_call_strike=6960.0,
            short_put_strike=6870.0, long_put_strike=6840.0,
        )


class TestDecisionEngineLargeMoveGuard:
    @pytest.mark.asyncio
    async def test_large_move_blocks_entry(self):
        # expected_move=78.2pts, single_candle_threshold=7.82pts
        # Last candle moves 20pts → blocked
        closes = [6900.0] * 9 + [6920.0]
        engine = _make_engine(candles=_make_candles(closes))
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "large_move_guard" in result.reason

    @pytest.mark.asyncio
    async def test_small_move_allows_entry(self):
        # Last candle moves 3pts — below 7.82pt single_candle_threshold
        # Last 3 closes: 6901.0, 6901.5, 6902.0 — range 1pt, well within stability threshold
        closes = [6900.0] * 7 + [6901.0, 6901.5, 6902.0]
        engine = _make_engine(candles=_make_candles(closes))
        result = await engine.run_cycle()
        assert result.outcome == "executed"


class TestDecisionEngineStabilityCheck:
    @pytest.mark.asyncio
    async def test_unstable_candles_blocks_entry(self):
        # expected_move=78.2pts, stability_threshold=11.73pts
        # Last 3 closes span 20pts → blocked
        # But last candle (6900→6890) moves 10pts > single_candle_threshold (7.82pts) too
        # Use last candle within guard, but last 3 span 20pts
        closes = [6900.0] * 7 + [6880.0, 6900.0, 6882.0]  # range=20pts, last candle move=18pts
        engine = _make_engine(candles=_make_candles(closes))
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        # either large_move_guard or market_unstable triggered
        assert "waited" == result.outcome

    @pytest.mark.asyncio
    async def test_stability_threshold_triggered_when_candle_guard_passes(self):
        # Last candle move: 5pts (< 7.82 single_candle threshold, passes guard)
        # But last 3 range: 6880→6895→6890 = 15pts > 11.73 stability threshold → blocked
        closes = [6900.0] * 7 + [6880.0, 6895.0, 6890.0]
        engine = _make_engine(candles=_make_candles(closes))
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "market_unstable" in result.reason

    @pytest.mark.asyncio
    async def test_stable_candles_allows_entry(self):
        # Last 3 closes: range=0.5pts — well within 11.73pt threshold
        closes = [6900.0] * 7 + [6900.5, 6901.0, 6900.8]
        engine = _make_engine(candles=_make_candles(closes))
        result = await engine.run_cycle()
        assert result.outcome == "executed"

    @pytest.mark.asyncio
    async def test_fallback_threshold_used_when_iv_unavailable(self):
        # IV unavailable → fallback_stability_pts=8.0 used
        # single_candle_threshold = 8.0 * (0.10/0.15) = 5.33pts
        # stability_threshold = 8.0pts
        # Last 3: 6890, 6900, 6896 → range=10pts > 8.0 stability threshold → blocked
        # Last candle move: 6900→6896 = 4pts < 5.33 single_candle threshold → passes guard
        closes = [6900.0] * 7 + [6890.0, 6900.0, 6896.0]
        no_iv = {"atm_iv": None, "spx_price": None, "expected_daily_move": None, "source": "unavailable"}
        engine = _make_engine(candles=_make_candles(closes), iv_data=no_iv)
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "market_unstable" in result.reason


class TestDecisionEngineAnomalyDetector:
    @pytest.mark.asyncio
    async def test_anomaly_blocks_entry(self):
        engine = _make_engine(anomaly_status="anomaly")
        engine.anomaly_detector.check = AsyncMock(return_value={
            "status": "anomaly", "reason": "bid > ask on 5 strikes",
        })
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "anomaly" in result.reason
        assert result.anomaly == "bid > ask on 5 strikes"

    @pytest.mark.asyncio
    async def test_anomaly_detector_failure_fails_open(self):
        engine = _make_engine()
        engine.anomaly_detector.check = AsyncMock(side_effect=Exception("timeout"))
        result = await engine.run_cycle()
        assert result.outcome == "executed"  # fails open


class TestDecisionEngineRegimeClassifier:
    @pytest.mark.asyncio
    async def test_skip_trade_blocks_single_cycle(self):
        engine = _make_engine(regime_verdict="skip_trade")
        result = await engine.run_cycle()
        assert result.outcome == "skip_trade"
        assert "regime_classifier" in result.reason

    @pytest.mark.asyncio
    async def test_skip_session_returns_skip_session(self):
        engine = _make_engine(regime_verdict="skip_session")
        result = await engine.run_cycle()
        assert result.outcome == "skip_session"

    @pytest.mark.asyncio
    async def test_regime_classifier_failure_fails_open(self):
        engine = _make_engine()
        engine.regime_classifier.classify = AsyncMock(side_effect=Exception("timeout"))
        result = await engine.run_cycle()
        assert result.outcome == "executed"  # fails open

    @pytest.mark.asyncio
    async def test_enter_verdict_proceeds(self):
        engine = _make_engine(regime_verdict="enter")
        result = await engine.run_cycle()
        assert result.outcome == "executed"


class TestDecisionEngineRiskCheck:
    @pytest.mark.asyncio
    async def test_risk_limits_exceeded_blocks_entry(self):
        engine = _make_engine(risk_allowed=False)
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "risk_limits" in result.reason


class TestDecisionEngineCandidates:
    @pytest.mark.asyncio
    async def test_no_candidates_skips_entry(self):
        engine = _make_engine(candidates={
            "candidates_found": 0,
            "top_candidate": None,
            "place_instruction": "No balanced configuration found.",
        })
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert result.reason == "no_candidates"
        assert result.candidates_found == 0

    @pytest.mark.asyncio
    async def test_candidates_found_logged(self):
        engine = _make_engine()
        result = await engine.run_cycle()
        assert result.candidates_found == 1


class TestDecisionEngineValidation:
    @pytest.mark.asyncio
    async def test_validation_failure_blocks_entry(self):
        engine = _make_engine(validation={
            "passed": False,
            "failures": ["delta_out_of_range", "credit_too_low"],
        })
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "validation_failed" in result.reason

    @pytest.mark.asyncio
    async def test_validation_pass_proceeds_to_placement(self):
        engine = _make_engine(validation={"passed": True, "failures": []})
        result = await engine.run_cycle()
        assert result.outcome == "executed"


class TestDecisionEngineCycleResult:
    @pytest.mark.asyncio
    async def test_result_has_timestamps(self):
        engine = _make_engine()
        result = await engine.run_cycle()
        assert result.started_at
        assert result.finished_at
        assert result.finished_at >= result.started_at


# ---------------------------------------------------------------------------
# AnomalyDetector tests
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def _make_detector(self, llm_response: str):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value={"content": llm_response})
        return AnomalyDetector(llm_client=llm)

    @pytest.mark.asyncio
    async def test_ok_response_parsed(self):
        detector = self._make_detector('{"status": "ok"}')
        result = await detector.check(candles=_make_candles([6900.0] * 5))
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_anomaly_response_parsed(self):
        detector = self._make_detector('{"status": "anomaly", "reason": "frozen streamer"}')
        result = await detector.check()
        assert result["status"] == "anomaly"
        assert result["reason"] == "frozen streamer"

    @pytest.mark.asyncio
    async def test_llm_failure_fails_open(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=Exception("connection error"))
        detector = AnomalyDetector(llm_client=llm)
        result = await detector.check()
        assert result["status"] == "ok"
        assert "note" in result

    @pytest.mark.asyncio
    async def test_unparseable_response_fails_open(self):
        detector = self._make_detector("Sorry, I cannot determine this.")
        result = await detector.check()
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# RegimeClassifier tests
# ---------------------------------------------------------------------------

class TestRegimeClassifier:
    def _make_classifier(self, llm_response: str):
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value={"content": llm_response})
        return RegimeClassifier(llm_client=llm)

    @pytest.mark.asyncio
    async def test_enter_verdict_parsed(self):
        clf = self._make_classifier('{"verdict": "enter", "confidence": 0.9, "reasoning": "flat candles"}')
        result = await clf.classify(candles=_make_candles([6900.0] * 10))
        assert result["verdict"] == "enter"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_skip_trade_verdict_parsed(self):
        clf = self._make_classifier('{"verdict": "skip_trade", "reasoning": "spike recovering"}')
        result = await clf.classify()
        assert result["verdict"] == "skip_trade"

    @pytest.mark.asyncio
    async def test_skip_session_verdict_parsed(self):
        clf = self._make_classifier('{"verdict": "skip_session", "reasoning": "VIX > 30, trend day"}')
        result = await clf.classify()
        assert result["verdict"] == "skip_session"

    @pytest.mark.asyncio
    async def test_llm_failure_fails_open_to_enter(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=Exception("timeout"))
        clf = RegimeClassifier(llm_client=llm)
        result = await clf.classify()
        assert result["verdict"] == "enter"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_invalid_verdict_fails_open(self):
        clf = self._make_classifier('{"verdict": "maybe", "reasoning": "unsure"}')
        result = await clf.classify()
        assert result["verdict"] == "enter"

    @pytest.mark.asyncio
    async def test_unparseable_response_fails_open(self):
        clf = self._make_classifier("I think the market looks okay today.")
        result = await clf.classify()
        assert result["verdict"] == "enter"
