"""Phase 4 TDD tests: DecisionEngine calls strategy interface, not IC methods."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from zerodte.agent.decision_engine import DecisionEngine, CycleResult
from zerodte.strategies.base import LiveStrategy, LiveTradeSetup, OrderLeg


def _make_mock_strategy(candidates_found=1, validation_passed=True):
    """Build a mock LiveStrategy with configurable behavior."""
    strategy = AsyncMock(spec=LiveStrategy)

    setup = LiveTradeSetup(
        strategy_type="test",
        legs=[OrderLeg("", 5850.0, "call", "sell", 1)],
        total_credit=1.50,
        strikes={"short_call": 5890.0},
    )

    strategy.search_candidates.return_value = {
        "candidates_found": candidates_found,
        "top_candidate": setup if candidates_found > 0 else None,
        "candidates": [setup] if candidates_found > 0 else [],
    }
    strategy.validate.return_value = {
        "passed": validation_passed,
        "checks": [],
        "failures": [] if validation_passed else ["test_failure"],
    }
    strategy.build_order.return_value = {
        "order_type": "test",
        "order_id": "TEST-001",
    }
    return strategy


def _make_engine(strategy, candles=None, risk_ok=True):
    """Build DecisionEngine with mocked dependencies."""
    tool_handlers = AsyncMock()
    tool_handlers.get_5min_candles.return_value = candles or [
        {"close": 5850, "high": 5852, "low": 5848, "open": 5849},
        {"close": 5851, "high": 5853, "low": 5849, "open": 5850},
        {"close": 5852, "high": 5854, "low": 5850, "open": 5851},
    ]
    tool_handlers.get_atm_iv.return_value = {
        "expected_daily_move": 30.0,
        "source": "test",
    }
    tool_handlers.get_option_chain.return_value = []
    tool_handlers.get_spx_price.return_value = {"price": 5850.0}
    tool_handlers.place_order.return_value = {"order_id": "PLACED-001"}

    risk_manager = AsyncMock()
    risk_manager.can_open_new_trade.return_value = {"allowed": risk_ok}

    engine = DecisionEngine(
        tool_handlers=tool_handlers,
        risk_manager=risk_manager,
        strategy=strategy,
    )
    return engine


class TestEngineCallsStrategy:
    @pytest.mark.asyncio
    async def test_calls_search_candidates(self):
        strategy = _make_mock_strategy(candidates_found=1)
        engine = _make_engine(strategy)
        await engine.run_cycle()
        strategy.search_candidates.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_validate_on_top_candidate(self):
        strategy = _make_mock_strategy(candidates_found=1)
        engine = _make_engine(strategy)
        await engine.run_cycle()
        strategy.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_build_order_on_valid(self):
        strategy = _make_mock_strategy(candidates_found=1, validation_passed=True)
        engine = _make_engine(strategy)
        result = await engine.run_cycle()
        strategy.build_order.assert_called_once()
        assert result.outcome == "executed"

    @pytest.mark.asyncio
    async def test_skips_on_no_candidates(self):
        strategy = _make_mock_strategy(candidates_found=0)
        engine = _make_engine(strategy)
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "no_candidates" in result.reason
        strategy.validate.assert_not_called()
        strategy.build_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_on_validation_failure(self):
        strategy = _make_mock_strategy(candidates_found=1, validation_passed=False)
        engine = _make_engine(strategy)
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "validation_failed" in result.reason
        strategy.build_order.assert_not_called()


class TestEngineGuardsStillWork:
    @pytest.mark.asyncio
    async def test_insufficient_candles(self):
        strategy = _make_mock_strategy()
        engine = _make_engine(strategy, candles=[{"close": 5850}])
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "insufficient" in result.reason
        strategy.search_candidates.assert_not_called()

    @pytest.mark.asyncio
    async def test_risk_limit_blocks_entry(self):
        strategy = _make_mock_strategy()
        engine = _make_engine(strategy, risk_ok=False)
        engine.risk_manager.can_open_new_trade.return_value = {
            "allowed": False, "reason": "max_concurrent",
        }
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "risk" in result.reason
        strategy.search_candidates.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_move_guard(self):
        strategy = _make_mock_strategy()
        candles = [
            {"close": 5850, "high": 5852, "low": 5848, "open": 5849},
            {"close": 5851, "high": 5853, "low": 5849, "open": 5850},
            {"close": 5880, "high": 5882, "low": 5878, "open": 5879},  # +29pt move
        ]
        engine = _make_engine(strategy, candles=candles)
        result = await engine.run_cycle()
        assert result.outcome == "waited"
        assert "large_move" in result.reason


class TestEngineOrderPlacement:
    @pytest.mark.asyncio
    async def test_places_order_via_tool_handlers(self):
        strategy = _make_mock_strategy()
        engine = _make_engine(strategy)
        result = await engine.run_cycle()
        assert result.outcome == "executed"
        engine.tool_handlers.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_failure_returns_error(self):
        strategy = _make_mock_strategy()
        engine = _make_engine(strategy)
        engine.tool_handlers.place_order.side_effect = Exception("API error")
        result = await engine.run_cycle()
        assert result.outcome == "error"
        assert "place_order_failed" in result.reason
