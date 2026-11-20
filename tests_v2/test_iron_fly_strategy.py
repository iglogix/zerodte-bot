"""Phase 3 TDD tests: IronFlyStrategy."""

import pytest

from zerodte.strategies.base import LiveStrategy, LiveTradeSetup
from zerodte.strategies.iron_fly import IronFlyStrategy


class TestIronFlyConfigure:
    def test_is_live_strategy(self):
        assert issubclass(IronFlyStrategy, LiveStrategy)

    def test_configure_reads_params(self, if_config):
        s = IronFlyStrategy()
        s.configure(if_config)
        assert s._strike_step == 5
        assert s._em_wing_fraction == 0.50
        assert s._tp_fixed_dollars == 200
        assert s._multiplier == 100
        assert s._slippage_per_leg == 0.05


class TestIronFlySearchCandidates:
    @pytest.mark.asyncio
    async def test_center_strike_above_spot(self, if_config, sample_if_chain):
        s = IronFlyStrategy()
        s.configure(if_config)
        result = await s.search_candidates(sample_if_chain, spot=5847.0)
        assert result["candidates_found"] > 0
        top = result["top_candidate"]
        assert isinstance(top, LiveTradeSetup)
        assert top.strategy_type == "iron_fly"
        # Center must be first strike strictly above 5847 → 5850
        assert top.strikes["center"] == 5850.0

    @pytest.mark.asyncio
    async def test_center_strike_when_on_strike(self, if_config, sample_if_chain):
        """If spot is exactly on a strike, center = next strike up."""
        s = IronFlyStrategy()
        s.configure(if_config)
        result = await s.search_candidates(sample_if_chain, spot=5850.0)
        if result["candidates_found"] > 0:
            assert result["top_candidate"].strikes["center"] == 5855.0

    @pytest.mark.asyncio
    async def test_wings_at_em_fraction(self, if_config, sample_if_chain):
        s = IronFlyStrategy()
        s.configure(if_config)
        result = await s.search_candidates(sample_if_chain, spot=5847.0)
        if result["candidates_found"] == 0:
            pytest.skip("No candidates")
        top = result["top_candidate"]
        center = top.strikes["center"]
        long_put = top.strikes["long_put"]
        long_call = top.strikes["long_call"]
        # Wings should be on both sides of center
        assert long_put < center
        assert long_call > center

    @pytest.mark.asyncio
    async def test_short_legs_same_strike(self, if_config, sample_if_chain):
        """Iron fly: both shorts at the same center strike."""
        s = IronFlyStrategy()
        s.configure(if_config)
        result = await s.search_candidates(sample_if_chain, spot=5847.0)
        if result["candidates_found"] == 0:
            pytest.skip("No candidates")
        top = result["top_candidate"]
        sell_legs = [l for l in top.legs if l.side == "sell"]
        assert len(sell_legs) == 2
        assert sell_legs[0].strike == sell_legs[1].strike

    @pytest.mark.asyncio
    async def test_total_credit_positive(self, if_config, sample_if_chain):
        s = IronFlyStrategy()
        s.configure(if_config)
        result = await s.search_candidates(sample_if_chain, spot=5847.0)
        if result["candidates_found"] == 0:
            pytest.skip("No candidates")
        assert result["top_candidate"].total_credit > 0

    @pytest.mark.asyncio
    async def test_empty_chain(self, if_config):
        s = IronFlyStrategy()
        s.configure(if_config)
        result = await s.search_candidates([], spot=5850.0)
        assert result["candidates_found"] == 0


class TestIronFlyValidate:
    @pytest.mark.asyncio
    async def test_valid_setup_passes(self, if_config, sample_if_chain):
        s = IronFlyStrategy()
        s.configure(if_config)
        result = await s.search_candidates(sample_if_chain, spot=5847.0)
        if result["candidates_found"] == 0:
            pytest.skip("No candidates")
        val = await s.validate(result["top_candidate"], sample_if_chain)
        assert val["passed"] is True

    @pytest.mark.asyncio
    async def test_invalid_wings_fails(self, if_config, sample_if_chain):
        s = IronFlyStrategy()
        s.configure(if_config)
        bad_setup = LiveTradeSetup(
            strategy_type="iron_fly",
            legs=[],
            total_credit=5.0,
            strikes={
                "center": 5850.0,
                "long_put": 5860.0,   # wrong: put wing above center
                "long_call": 5870.0,
            },
        )
        val = await s.validate(bad_setup, sample_if_chain)
        assert val["passed"] is False


class TestIronFlyBuildOrder:
    def test_builds_4_leg_same_short_strike(self, if_config):
        s = IronFlyStrategy()
        s.configure(if_config)
        setup = LiveTradeSetup(
            strategy_type="iron_fly",
            legs=[],
            total_credit=15.0,
            strikes={"center": 5850.0, "long_put": 5835.0, "long_call": 5865.0},
        )
        order = s.build_order(setup)
        # order_type is the submission route; an iron fly is an IC with equal short strikes (asserted below).
        assert order["order_type"] == "iron_condor"
        assert order["short_call_strike"] == 5850.0
        assert order["short_put_strike"] == 5850.0
        assert order["long_call_strike"] == 5865.0
        assert order["long_put_strike"] == 5835.0


class TestIronFlyCheckExit:
    def test_tp_at_fixed_dollars(self, if_config):
        """TP when mark-to-market drops below credit - tp_dollars/multiplier."""
        s = IronFlyStrategy()
        s.configure(if_config)
        # credit=15.0, tp=$200, multiplier=100 → tp_debit = 15.0 - 2.0 = 13.0
        trade = {
            "credit_received": 15.0,
            "center_strike": 5850.0,
            "long_put": 5835.0,
            "long_call": 5865.0,
            "short_call_symbol": "SC",
            "short_put_symbol": "SP",
            "long_call_symbol": "LC",
            "long_put_symbol": "LP",
        }
        # Current mark = 12.0 (below tp_debit of 13.0) → take profit
        quotes = {
            "SC": {"bid": 6.00, "ask": 6.20},   # short call
            "LC": {"bid": 0.80, "ask": 1.00},   # long call
            "SP": {"bid": 5.80, "ask": 6.00},   # short put
            "LP": {"bid": 0.70, "ask": 0.90},   # long put
        }
        # Mark = (SC_mid - LC_mid) + (SP_mid - LP_mid) = (6.10-0.90) + (5.90-0.80) = 5.20+5.10 = 10.30
        action = s.check_exit(trade, quotes)
        assert action.action == "take_profit"

    def test_hold_when_mark_above_tp(self, if_config):
        s = IronFlyStrategy()
        s.configure(if_config)
        trade = {
            "credit_received": 15.0,
            "center_strike": 5850.0,
            "long_put": 5835.0,
            "long_call": 5865.0,
            "short_call_symbol": "SC",
            "short_put_symbol": "SP",
            "long_call_symbol": "LC",
            "long_put_symbol": "LP",
        }
        # Mark still high → hold
        quotes = {
            "SC": {"bid": 7.50, "ask": 7.70},
            "LC": {"bid": 0.50, "ask": 0.70},
            "SP": {"bid": 7.40, "ask": 7.60},
            "LP": {"bid": 0.50, "ask": 0.70},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "hold"

    def test_sl_on_wing_breach(self, if_config):
        """SL when underlying breaches wing strike (spot in quotes)."""
        s = IronFlyStrategy()
        s.configure(if_config)
        trade = {
            "credit_received": 15.0,
            "center_strike": 5850.0,
            "long_put": 5835.0,
            "long_call": 5865.0,
            "short_call_symbol": "SC",
            "short_put_symbol": "SP",
            "long_call_symbol": "LC",
            "long_put_symbol": "LP",
        }
        quotes = {
            "SC": {"bid": 7.00, "ask": 7.20},
            "LC": {"bid": 0.50, "ask": 0.70},
            "SP": {"bid": 7.00, "ask": 7.20},
            "LP": {"bid": 0.50, "ask": 0.70},
            "_underlying": {"price": 5870.0},  # above long_call 5865
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "stop_loss"
        assert "wing" in action.reason


class TestIronFlyEdgeCases:
    @pytest.mark.asyncio
    async def test_tiny_em_produces_minimum_wings(self, if_config, sample_if_chain):
        """When EM is very small, wings should still be at least 1 step away."""
        s = IronFlyStrategy()
        # Force em_wing_fraction very small
        if_config["options"]["em_wing_fraction"] = 0.01
        s.configure(if_config)
        result = await s.search_candidates(sample_if_chain, spot=5847.0)
        if result["candidates_found"] > 0:
            top = result["top_candidate"]
            assert top.strikes["long_call"] > top.strikes["center"]
            assert top.strikes["long_put"] < top.strikes["center"]

    def test_check_exit_no_underlying_falls_through_to_tp(self, if_config):
        """When _underlying is missing, SL can't trigger but TP still can."""
        s = IronFlyStrategy()
        s.configure(if_config)
        trade = {
            "credit_received": 15.0,
            "center_strike": 5850.0,
            "long_put": 5835.0,
            "long_call": 5865.0,
            "short_call_symbol": "SC",
            "short_put_symbol": "SP",
            "long_call_symbol": "LC",
            "long_put_symbol": "LP",
        }
        # No _underlying key, but mark is below TP threshold
        quotes = {
            "SC": {"bid": 6.00, "ask": 6.20},
            "LC": {"bid": 0.80, "ask": 1.00},
            "SP": {"bid": 5.80, "ask": 6.00},
            "LP": {"bid": 0.70, "ask": 0.90},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "take_profit"

    def test_check_exit_zero_credit_holds(self, if_config):
        """If credit_received is 0, TP math won't trigger."""
        s = IronFlyStrategy()
        s.configure(if_config)
        trade = {
            "credit_received": 0,
            "center_strike": 5850.0,
            "long_put": 5835.0,
            "long_call": 5865.0,
            "short_call_symbol": "SC",
            "short_put_symbol": "SP",
            "long_call_symbol": "LC",
            "long_put_symbol": "LP",
        }
        quotes = {
            "SC": {"bid": 7.00, "ask": 7.20},
            "LC": {"bid": 0.50, "ask": 0.70},
            "SP": {"bid": 7.00, "ask": 7.20},
            "LP": {"bid": 0.50, "ask": 0.70},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "hold"


class TestIronFlyStopLoss:
    def test_wing_based_stop(self, if_config):
        s = IronFlyStrategy()
        s.configure(if_config)
        setup = LiveTradeSetup(
            strategy_type="iron_fly", legs=[], total_credit=15.0,
            strikes={"center": 5850.0, "long_put": 5835.0, "long_call": 5865.0},
        )
        stop = s.compute_stop_loss(setup)
        # Wing-based: max loss = wing_width × multiplier
        assert stop["long_call_stop_price"] == 5865.0
        assert stop["long_put_stop_price"] == 5835.0


class TestIronFlyLegPairs:
    def test_returns_if_pairs(self, if_config):
        s = IronFlyStrategy()
        s.configure(if_config)
        pairs = s.leg_pairs()
        assert len(pairs) == 2
        shorts = [p[0] for p in pairs]
        assert "short_call_symbol" in shorts
        assert "short_put_symbol" in shorts
