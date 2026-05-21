"""TDD tests: MeicStrategy."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src_v2.strategies.base import ExitAction

ET = ZoneInfo("America/New_York")


class TestExitActionWidening:
    def test_exit_action_has_partial_fields_with_defaults(self):
        a = ExitAction(action="hold")
        assert a.leg_quantities == {}
        assert a.quantity_fraction == 1.0

    def test_exit_action_accepts_leg_quantities(self):
        a = ExitAction(
            action="take_profit",
            legs_to_close=["SPXW_C"],
            leg_quantities={"SPXW_C": 2},
            quantity_fraction=0.25,
        )
        assert a.leg_quantities == {"SPXW_C": 2}
        assert a.quantity_fraction == 0.25


from src_v2.strategies.base import LiveStrategy
from src_v2.strategies.meic import MeicStrategy
from src_v2.agent.scheduled_engine import ScheduledEngine


class TestMeicConfigure:
    def test_is_live_strategy(self):
        assert issubclass(MeicStrategy, LiveStrategy)

    def test_configure_reads_all_keys(self, meic_config):
        s = MeicStrategy()
        s.configure(meic_config)
        assert s._am_min_credit == 1.25
        assert s._am_max_credit == 1.65
        assert s._pm_min_credit == 0.80
        assert s._pm_max_credit == 1.30
        assert s._wing_widths == [80, 90, 100]
        assert s._stop_pct == 0.70
        assert s._am_tiers == [0.55, 0.65, 0.75, 0.85]
        assert s._am_fractions == [0.25, 0.25, 0.25, 0.25]
        assert s._pm_tiers == [0.65, 0.85]
        assert s._pm_fractions == [0.50, 0.50]
        assert s._abandon_long_below == 0.05
        assert s._default_tranche_qty == 3
        assert s._tranche_sizing == {"11:00": 3, "11:40": 4, "12:00": 5, "13:00": 6}
        assert s._am_pm_cutoff.hour == 12 and s._am_pm_cutoff.minute == 30

    def test_configure_rejects_stop_pct_too_low(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.50
        s = MeicStrategy()
        with pytest.raises(ValueError):
            s.configure(meic_config)

    def test_configure_rejects_stop_pct_too_high(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.95
        s = MeicStrategy()
        with pytest.raises(ValueError):
            s.configure(meic_config)


class TestMeicSimpleMethods:
    def test_leg_pairs(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s.leg_pairs() == [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    def test_compute_trailing_stop_returns_none(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s.compute_trailing_stop({}, {}) is None

    def test_create_engine_returns_scheduled_engine(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        engine = s.create_engine(_FakeDeps())
        assert isinstance(engine, ScheduledEngine)


class _FakeDeps:
    """Minimal EngineDeps stand-in for create_engine."""
    tool_handlers = None
    risk_manager = None
    anomaly_detector = None
    regime_classifier = None
    decision_file_logger = None
    db = None
    mode = "dry_run"
    config = {"entry": {}}
    clock = None


from src_v2.strategies.base import LiveTradeSetup


def _at(s: MeicStrategy, hhmm: str):
    """Pin the strategy clock to a fixed ET wall time on an arbitrary date."""
    h, m = (int(x) for x in hhmm.split(":"))
    s._now_fn = lambda: datetime(2026, 5, 20, h, m, tzinfo=ET)


class TestMeicSearchCandidates:
    @pytest.mark.asyncio
    async def test_empty_chain_returns_zero(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:30")
        result = await s.search_candidates([], spot=5850.0)
        assert result["candidates_found"] == 0
        assert result["top_candidate"] is None

    @pytest.mark.asyncio
    async def test_am_band_finds_candidate(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:30")
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        assert result["candidates_found"] > 0
        top = result["top_candidate"]
        assert isinstance(top, LiveTradeSetup)
        assert top.strategy_type == "meic"
        # Credits must be in AM band
        assert 1.25 <= top.extra["call_credit"] <= 1.65
        assert 1.25 <= top.extra["put_credit"] <= 1.65

    @pytest.mark.asyncio
    async def test_pm_band_rejects_am_credits(self, meic_config, sample_meic_chain):
        # At 13:00 the PM band is [0.80, 1.30]; the sample chain's ~1.39
        # credits fall OUTSIDE PM band -> no candidates.
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "13:00")
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        assert result["candidates_found"] == 0

    @pytest.mark.asyncio
    async def test_stamps_tranche_qty(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:40")  # tranche_sizing["11:40"] == 4
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        top = result["top_candidate"]
        assert top.extra["qty"] == 4
        assert all(leg.quantity == 4 for leg in top.legs)

    @pytest.mark.asyncio
    async def test_default_qty_when_slot_missing(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:20")  # not in tranche_sizing -> default 3
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        top = result["top_candidate"]
        assert top.extra["qty"] == 3


class TestMeicValidate:
    def _setup(self, qty=3, cc=1.40, pc=1.40,
               sc=5950.0, lc=6040.0, sp=5750.0, lp=5660.0):
        from src_v2.strategies.base import LiveTradeSetup, OrderLeg
        return LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", sc, "call", "sell", qty)],
            total_credit=cc + pc,
            strikes={"short_call": sc, "long_call": lc,
                     "short_put": sp, "long_put": lp},
            extra={"call_credit": cc, "put_credit": pc, "qty": qty},
        )

    @pytest.mark.asyncio
    async def test_valid_setup_passes(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup()
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is True
        assert res["failures"] == []

    @pytest.mark.asyncio
    async def test_bad_call_strike_order_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup(lc=5940.0)  # long_call < short_call
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "strike_order_call" in res["failures"]

    @pytest.mark.asyncio
    async def test_asymmetric_wings_fail(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        # call wing 90 (5950->6040), put wing 60 (5750->5690): diff 30 > 5
        setup = self._setup(lp=5690.0)
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "wing_symmetry" in res["failures"]

    @pytest.mark.asyncio
    async def test_wing_out_of_range_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        # both wings 50pt -> below 80 minimum
        setup = self._setup(lc=6000.0, lp=5700.0)
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "wing_range" in res["failures"]

    @pytest.mark.asyncio
    async def test_credit_out_of_band_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup(cc=2.50)  # above AM band max 1.65
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "call_credit_band" in res["failures"]

    @pytest.mark.asyncio
    async def test_zero_qty_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup(qty=0)
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "qty" in res["failures"]


class TestMeicBuildOrder:
    def test_two_spread_payload(self, meic_config):
        from src_v2.strategies.base import LiveTradeSetup, OrderLeg
        s = MeicStrategy(); s.configure(meic_config)
        setup = LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", 5950, "call", "sell", 4)],
            total_credit=2.80,
            strikes={"short_call": 5950, "long_call": 6040,
                     "short_put": 5750, "long_put": 5660},
            extra={"call_credit": 1.40, "put_credit": 1.40, "qty": 4},
        )
        order = s.build_order(setup)
        assert order["order_type"] == "meic_two_spreads"
        assert order["quantity"] == 4
        assert order["put_spread"] == {
            "short_strike": 5750, "long_strike": 5660,
            "credit": 1.40, "quantity": 4,
        }
        assert order["call_spread"] == {
            "short_strike": 5950, "long_strike": 6040,
            "credit": 1.40, "quantity": 4,
        }


class TestMeicStops:
    def _setup(self, cc=1.40, pc=1.40, sc_mid=1.45, sp_mid=1.45):
        from src_v2.strategies.base import LiveTradeSetup, OrderLeg
        return LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", 5950, "call", "sell", 3)],
            total_credit=cc + pc,
            strikes={"short_call": 5950, "long_call": 6040,
                     "short_put": 5750, "long_put": 5660},
            extra={"call_credit": cc, "put_credit": pc, "qty": 3,
                   "short_call_mid": sc_mid, "short_put_mid": sp_mid},
        )

    def test_compute_stop_loss_math(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)  # stop_pct 0.70
        info = s.compute_stop_loss(self._setup())
        # total credit 2.80 -> stop amount 1.96
        assert info["total_credit"] == 2.80
        assert info["stop_amount"] == pytest.approx(1.96)
        assert info["short_call_stop_price"] == pytest.approx(1.45 + 1.96)
        assert info["short_put_stop_price"] == pytest.approx(1.45 + 1.96)
        assert info["order_type"] == "stop_market"

    def test_build_stop_orders_stop_market_on_shorts_only(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        setup = self._setup()
        info = s.compute_stop_loss(setup)
        orders = s.build_stop_orders(setup, info)
        assert len(orders) == 2
        sides = {o["side"] for o in orders}
        assert sides == {"short_call", "short_put"}
        for o in orders:
            assert "stop_market" in o
            assert "stop_limit" not in o
            assert o["stop_market"]["trigger_price"] > 0


class TestMeicScaleOut:
    def _trade(self, entry="11:40", qty=4, cc=1.40, pc=1.40):
        h, m = (int(x) for x in entry.split(":"))
        return {
            "trade_id": f"2026-05-20_{entry}",
            "entry_time_et": datetime(2026, 5, 20, h, m, tzinfo=ET),
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
            "call_credit": cc, "put_credit": pc,
            "call_open_qty": qty, "put_open_qty": qty,
            "short_call_stop_price": 99.0, "short_put_stop_price": 99.0,
        }

    def _quotes(self, sc_ask, sp_ask, lc_bid=0.50, lp_bid=0.50):
        return {
            "SC": {"bid": sc_ask - 0.05, "ask": sc_ask},
            "SP": {"bid": sp_ask - 0.05, "ask": sp_ask},
            "LC": {"bid": lc_bid, "ask": lc_bid + 0.05},
            "LP": {"bid": lp_bid, "ask": lp_bid + 0.05},
        }

    def test_am_tier_fires_at_55pct(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)  # AM
        # call side: credit 1.40, ask 0.60 -> pnl 57% -> tier 0 (55%) fires
        # put side: ask 1.40 -> pnl 0% -> nothing
        action = s.check_exit(trade, self._quotes(sc_ask=0.60, sp_ask=1.40))
        assert action.action == "take_profit"
        # 25% of 4 = 1 contract of short call, plus paired long
        assert action.leg_quantities.get("SC") == 1
        assert action.leg_quantities.get("LC") == 1
        assert "SP" not in action.leg_quantities

    def test_am_tier_does_not_refire(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)
        q = self._quotes(sc_ask=0.60, sp_ask=1.40)
        s.check_exit(trade, q)            # fires tier 0
        action2 = s.check_exit(trade, q)  # same pnl -> no new tier
        assert action2.action == "hold"

    def test_am_multiple_tiers_one_bar(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)
        # call ask 0.20 -> pnl ~86% -> tiers 0,1,2,3 all fire
        action = s.check_exit(trade, self._quotes(sc_ask=0.20, sp_ask=1.40))
        assert action.action == "take_profit"
        # 4 tiers x 25% of 4 = 4 contracts total
        assert action.leg_quantities.get("SC") == 4

    def test_pm_uses_pm_tiers(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="13:00", qty=4)  # PM
        # call ask 0.49 -> pnl 65% -> PM tier 0 (65%) fires; 50% of 4 = 2
        action = s.check_exit(trade, self._quotes(sc_ask=0.49, sp_ask=1.40))
        assert action.action == "take_profit"
        assert action.leg_quantities.get("SC") == 2

    def test_hold_when_below_first_tier(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)
        # call ask 1.00 -> pnl ~29% -> below 55% tier
        action = s.check_exit(trade, self._quotes(sc_ask=1.00, sp_ask=1.40))
        assert action.action == "hold"

    def test_missing_ask_quote_holds(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)
        # Short call quote present but has no usable ask -> must NOT fire any tier.
        quotes = {
            "SC": {"bid": 0.10},   # no "ask"/"price"
            "SP": {"bid": 1.35, "ask": 1.40},
            "LC": {"bid": 0.50, "ask": 0.55},
            "LP": {"bid": 0.50, "ask": 0.55},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "hold"


class TestMeicStopAndAbandon:
    def _trade(self, entry="11:40", qty=4):
        h, m = (int(x) for x in entry.split(":"))
        return {
            "trade_id": f"2026-05-20_{entry}_stop",
            "entry_time_et": datetime(2026, 5, 20, h, m, tzinfo=ET),
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
            "call_credit": 1.40, "put_credit": 1.40,
            "call_open_qty": qty, "put_open_qty": qty,
            "short_call_stop_price": 3.41, "short_put_stop_price": 3.41,
        }

    def test_stop_loss_triggers_on_call_breach(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade()
        quotes = {
            "SC": {"bid": 3.40, "ask": 3.50},   # ask >= 3.41 stop
            "SP": {"bid": 1.35, "ask": 1.40},
            "LC": {"bid": 0.80, "ask": 0.90},
            "LP": {"bid": 0.50, "ask": 0.60},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "stop_loss"
        assert action.leg_quantities.get("SC") == 4   # all contracts
        assert action.leg_quantities.get("LC") == 4   # paired long
        assert "SP" not in action.leg_quantities

    def test_abandon_long_when_bidless(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade()
        # call side hits 55% tier (ask 0.60); long call bid 0.03 < 0.05 -> abandon
        quotes = {
            "SC": {"bid": 0.55, "ask": 0.60},
            "SP": {"bid": 1.35, "ask": 1.40},
            "LC": {"bid": 0.03, "ask": 0.08},
            "LP": {"bid": 0.50, "ask": 0.60},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "take_profit"
        assert action.leg_quantities.get("SC") == 1
        assert "LC" not in action.leg_quantities   # long abandoned
        assert "call_abandon_long" in action.reason
