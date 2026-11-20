"""Phase 2 TDD tests: IronCondorStrategy."""

import pytest

from zerodte.strategies.base import LiveStrategy, LiveTradeSetup
from zerodte.strategies.iron_condor import IronCondorStrategy


class TestIronCondorConfigure:
    def test_is_live_strategy(self):
        assert issubclass(IronCondorStrategy, LiveStrategy)

    def test_configure_reads_params(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        assert s._credit_equality_threshold == 0.20
        assert s._min_short_delta == 0.10
        assert s._max_short_delta == 0.15
        assert s._min_side_credit == 0.50
        assert s._max_side_credit == 2.50
        assert s._wing_width == 30
        assert s._stop_limit_gap == 40
        assert s._stop_market_gap == 30
        assert s._take_profit_threshold == 0.05


class TestIronCondorSearchCandidates:
    @pytest.mark.asyncio
    async def test_finds_balanced_candidates(self, ic_config, sample_ic_chain):
        s = IronCondorStrategy()
        s.configure(ic_config)
        result = await s.search_candidates(sample_ic_chain, spot=5850.0)
        assert result["candidates_found"] > 0
        top = result["top_candidate"]
        assert isinstance(top, LiveTradeSetup)
        assert top.strategy_type == "iron_condor"
        assert "short_call" in top.strikes
        assert "short_put" in top.strikes
        assert top.total_credit > 0

    @pytest.mark.asyncio
    async def test_empty_chain_returns_zero(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        result = await s.search_candidates([], spot=5850.0)
        assert result["candidates_found"] == 0

    @pytest.mark.asyncio
    async def test_candidates_sorted_by_credit_diff(self, ic_config, sample_ic_chain):
        s = IronCondorStrategy()
        s.configure(ic_config)
        result = await s.search_candidates(sample_ic_chain, spot=5850.0)
        if result["candidates_found"] > 1:
            diffs = [c.extra["credit_diff"] for c in result["candidates"]]
            assert diffs == sorted(diffs)


class TestIronCondorValidate:
    @pytest.mark.asyncio
    async def test_valid_setup_passes(self, ic_config, sample_ic_chain):
        s = IronCondorStrategy()
        s.configure(ic_config)
        result = await s.search_candidates(sample_ic_chain, spot=5850.0)
        if result["candidates_found"] == 0:
            pytest.skip("No candidates to validate")
        top = result["top_candidate"]
        val = await s.validate(top, sample_ic_chain)
        assert val["passed"] is True

    @pytest.mark.asyncio
    async def test_invalid_strike_order_fails(self, ic_config, sample_ic_chain):
        s = IronCondorStrategy()
        s.configure(ic_config)
        bad_setup = LiveTradeSetup(
            strategy_type="iron_condor",
            legs=[],
            total_credit=1.0,
            strikes={
                "short_call": 5890.0,
                "long_call": 5880.0,  # wrong: must be > short_call
                "short_put": 5810.0,
                "long_put": 5780.0,
            },
        )
        val = await s.validate(bad_setup, sample_ic_chain)
        assert val["passed"] is False
        assert "strike_order" in str(val["failures"])


class TestIronCondorBuildOrder:
    def test_builds_4_leg_order(self, ic_config, sample_ic_chain):
        s = IronCondorStrategy()
        s.configure(ic_config)
        setup = LiveTradeSetup(
            strategy_type="iron_condor",
            legs=[],  # will be filled by build_order
            total_credit=1.50,
            strikes={
                "short_call": 5890.0,
                "long_call": 5920.0,
                "short_put": 5810.0,
                "long_put": 5780.0,
            },
        )
        order = s.build_order(setup)
        assert order["order_type"] == "iron_condor"
        assert order["short_call_strike"] == 5890.0
        assert order["long_call_strike"] == 5920.0
        assert order["short_put_strike"] == 5810.0
        assert order["long_put_strike"] == 5780.0


class TestIronCondorStopLoss:
    def test_stop_equals_premium_times_100(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        setup = LiveTradeSetup(
            strategy_type="iron_condor", legs=[], total_credit=1.50,
            strikes={}, extra={"total_premium": 1.50},
        )
        stop = s.compute_stop_loss(setup)
        assert stop["stop_dollars"] == 150.0

    def test_build_stop_orders_oco(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        setup = LiveTradeSetup(
            strategy_type="iron_condor", legs=[], total_credit=1.50,
            strikes={}, extra={"total_premium": 1.50},
        )
        stop_info = s.compute_stop_loss(setup)
        orders = s.build_stop_orders(setup, stop_info)
        assert len(orders) >= 1  # at least one stop order


class TestIronCondorCheckExit:
    def test_take_profit_at_005(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        trade = {
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
        }
        # Uses ask price for TP (cost to buy back)
        quotes = {
            "SC": {"bid": 0.03, "ask": 0.05},
            "SP": {"bid": 0.50, "ask": 0.60},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "take_profit"
        assert "SC" in action.legs_to_close

    def test_hold_when_above_threshold(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        trade = {
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
        }
        quotes = {
            "SC": {"bid": 0.50, "ask": 0.60},
            "SP": {"bid": 0.50, "ask": 0.60},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "hold"

    def test_tp_uses_ask_not_mid(self, ic_config):
        """TP should trigger at ask=0.05, not mid=(0.03+0.05)/2=0.04."""
        s = IronCondorStrategy()
        s.configure(ic_config)
        trade = {
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
        }
        # ask=0.06 > threshold → should hold even though mid=0.04
        quotes = {
            "SC": {"bid": 0.02, "ask": 0.06},
            "SP": {"bid": 0.50, "ask": 0.60},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "hold"


class TestIronCondorTrailingStop:
    def test_tightens_on_decay(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        trade = {
            "original_stop_loss": 2.00,
            "original_long_call_value": 0.50,
            "original_long_put_value": 0.50,
            "long_call_symbol": "LC_SYM",
            "long_put_symbol": "LP_SYM",
        }
        quotes = {
            "LC_SYM": {"bid": 0.20, "ask": 0.30},  # decayed from 0.50
            "LP_SYM": {"bid": 0.50, "ask": 0.60},  # unchanged
        }
        new_stop = s.compute_trailing_stop(trade, quotes)
        assert new_stop is not None
        assert new_stop < trade["original_stop_loss"]


class TestIronCondorEdgeCases:
    @pytest.mark.asyncio
    async def test_search_zero_bid_chain(self, ic_config):
        """Chain with all zero bids should return 0 candidates."""
        s = IronCondorStrategy()
        s.configure(ic_config)
        chain = [
            {"strike": 5890.0, "option_type": "CALL", "delta": 0.12, "bid": 0, "ask": 1.60},
            {"strike": 5810.0, "option_type": "PUT", "delta": -0.12, "bid": 0, "ask": 1.70},
        ]
        result = await s.search_candidates(chain, spot=5850.0)
        assert result["candidates_found"] == 0

    def test_trailing_stop_full_decay(self, ic_config):
        """When decay exceeds original stop, floor at 0.01."""
        s = IronCondorStrategy()
        s.configure(ic_config)
        trade = {
            "original_stop_loss": 0.50,
            "original_long_call_value": 1.00,
            "original_long_put_value": 1.00,
            "long_call_symbol": "LC",
            "long_put_symbol": "LP",
        }
        quotes = {
            "LC": {"bid": 0.01, "ask": 0.02},
            "LP": {"bid": 0.01, "ask": 0.02},
        }
        new_stop = s.compute_trailing_stop(trade, quotes)
        assert new_stop == 0.01


class TestIronCondorLegPairs:
    def test_returns_ic_pairs(self, ic_config):
        s = IronCondorStrategy()
        s.configure(ic_config)
        pairs = s.leg_pairs()
        assert len(pairs) == 2
        shorts = [p[0] for p in pairs]
        assert "short_call_symbol" in shorts
        assert "short_put_symbol" in shorts
