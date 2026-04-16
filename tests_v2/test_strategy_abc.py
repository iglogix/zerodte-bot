"""Phase 1 tests: LiveStrategy ABC cannot be instantiated directly."""

import pytest
from src_v2.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)


class TestLiveStrategyABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LiveStrategy()

    def test_partial_implementation_raises(self):
        """Implementing only some methods should still raise TypeError."""
        class Partial(LiveStrategy):
            def configure(self, config): pass
            async def search_candidates(self, chain, spot): pass
            # Missing all other methods
        with pytest.raises(TypeError):
            Partial()

    def test_order_leg_creation(self):
        leg = OrderLeg(
            symbol="SPXW250406C5890",
            strike=5890.0,
            option_type="call",
            side="sell",
            quantity=1,
        )
        assert leg.strike == 5890.0
        assert leg.side == "sell"

    def test_live_trade_setup_creation(self):
        setup = LiveTradeSetup(
            strategy_type="iron_condor",
            legs=[],
            total_credit=1.50,
            strikes={"short_call": 5890.0},
        )
        assert setup.strategy_type == "iron_condor"
        assert setup.extra == {}

    def test_exit_action_defaults(self):
        action = ExitAction(action="hold")
        assert action.legs_to_close == []
        assert action.reason == ""
        assert action.new_stop_price is None

    def test_exit_action_take_profit(self):
        action = ExitAction(
            action="take_profit",
            legs_to_close=["SPXW250406C5890"],
            reason="short call at $0.05",
        )
        assert action.action == "take_profit"
        assert len(action.legs_to_close) == 1
