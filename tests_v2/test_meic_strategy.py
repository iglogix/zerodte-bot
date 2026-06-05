"""TDD tests: MeicStrategy."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src_v2.strategies.base import LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction

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


from src_v2.strategies.meic import MeicStrategy
from src_v2.agent.scheduled_engine import ScheduledEngine


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


def _at(s: MeicStrategy, hhmm: str):
    """Pin the strategy clock to a fixed ET wall time on an arbitrary date."""
    h, m = (int(x) for x in hhmm.split(":"))
    s._now_fn = lambda: datetime(2026, 5, 20, h, m, tzinfo=ET)


def _em_chain(spot, em, step=5, span=400, rate=0.10, floor=0.05):
    """Synthetic chain: ATM straddle == em; premium decays linearly at `rate`
    per point from peak em/2, floored at `floor`. spot must be a step multiple
    so the ATM straddle equals em exactly. Far strikes hit the floor (bid 0 ->
    treated as bidless by mid_price)."""
    peak = em / 2.0
    atm = round(spot / step) * step
    chain = []
    k = atm - span
    while k <= atm + span:
        dist = abs(k - spot)
        mid = max(floor, peak - rate * dist)
        bid = round(mid - 0.05, 2)
        ask = round(mid + 0.05, 2)
        for otype in ("put", "call"):
            chain.append({"strike": float(k), "option_type": otype,
                          "bid": bid, "ask": ask})
        k += step
    return chain


class TestMeicConfigure:
    def test_is_live_strategy(self):
        assert issubclass(MeicStrategy, LiveStrategy)

    def test_configure_reads_em_relative_keys(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s._short_em_mult == 1.0
        assert s._wing_em_frac == 0.5
        assert s._min_credit_width_ratio == 0.08
        assert s._strike_step == 5
        assert s._stop_pct == 0.70
        assert s._am_pm_cutoff.hour == 12 and s._am_pm_cutoff.minute == 30
        assert s._tranche_sizing == {"11:00": 3, "11:40": 4, "12:00": 5, "13:00": 6}

    def test_configure_rejects_stop_pct_too_low(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.50
        with pytest.raises(ValueError):
            MeicStrategy().configure(meic_config)

    def test_configure_rejects_stop_pct_too_high(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.95
        with pytest.raises(ValueError):
            MeicStrategy().configure(meic_config)


class TestMeicSearchCandidates:
    @pytest.mark.asyncio
    async def test_empty_chain_returns_zero(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates([], spot=6700.0)
        assert result["candidates_found"] == 0
        assert result["top_candidate"] is None

    @pytest.mark.asyncio
    async def test_high_vol_regime_candidate(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates(_em_chain(6700, 120.0), spot=6700.0)
        assert result["candidates_found"] == 1
        top = result["top_candidate"]
        assert top.strategy_type == "meic"
        assert top.strikes["short_call"] == 6820.0
        assert top.strikes["short_put"] == 6580.0
        assert top.extra["wing_pts"] == 60
        assert top.extra["em_pts"] == pytest.approx(120.0)
        assert top.extra["call_credit"] == pytest.approx(6.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_low_vol_regime_candidate_scales_down(self, meic_config):
        # SAME config; spot 4285, EM 24 -> short_call 4310, short_put 4260, wing 10.
        # This is the regime the absolute-band version produced 0 trades for.
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates(_em_chain(4285, 24.0), spot=4285.0)
        assert result["candidates_found"] == 1
        top = result["top_candidate"]
        assert top.strikes["short_call"] == 4310.0
        assert top.strikes["short_put"] == 4260.0
        assert top.extra["wing_pts"] == 10
        assert top.extra["call_credit"] == pytest.approx(1.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_stamps_tranche_qty(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:40")  # qty 4
        top = (await s.search_candidates(_em_chain(6700, 120.0), 6700.0))["top_candidate"]
        assert top.extra["qty"] == 4
        assert all(leg.quantity == 4 for leg in top.legs)

    @pytest.mark.asyncio
    async def test_credit_to_width_gate_rejects_thin_credit(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates(_em_chain(6700, 120.0, rate=0.05), 6700.0)
        assert result["candidates_found"] == 0

    @pytest.mark.asyncio
    async def test_skips_when_wing_leg_bidless(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        chain = [o for o in _em_chain(6700, 120.0)
                 if not (o["strike"] == 6880.0 and o["option_type"] == "call")]
        result = await s.search_candidates(chain, 6700.0)
        assert result["candidates_found"] == 0


class TestMeicDislocationGuard:
    """Reject entries priced off a dislocated leg quote (e.g. 2025-04-09 13:20
    short put 0.05/45.00). The spread mid of such a quote is a fantasy fill."""

    def test_configure_reads_dislocation_params(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s._disloc_min_spread_pts == 1.0
        assert s._disloc_spread_mult == 1.5

    @pytest.mark.asyncio
    async def test_rejects_dislocated_short_leg(self, meic_config):
        # Both put legs wide so the put credit stays POSITIVE (11.9) and passes
        # the credit/width gate -> the dislocation guard is the ONLY rejecter.
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        chain = _em_chain(6700, 120.0)
        for o in chain:
            if o["option_type"] == "put" and o["strike"] == 6580.0:
                o["bid"], o["ask"] = 0.05, 45.00     # short put: spread 44.95, mid 22.5
            if o["option_type"] == "put" and o["strike"] == 6520.0:
                o["bid"], o["ask"] = 0.10, 21.10     # long put: spread 21.0, mid 10.6
        result = await s.search_candidates(chain, 6700.0)
        assert result["candidates_found"] == 0

    @pytest.mark.asyncio
    async def test_allows_wide_but_sane_quote(self, meic_config):
        # spread 4.0 is wide in absolute terms but < mult*mid (1.5*48=72) -> NOT
        # dislocated; credit unchanged (mid still 48) -> candidate must survive.
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        chain = _em_chain(6700, 120.0)
        for o in chain:
            if o["option_type"] == "put" and o["strike"] == 6580.0:
                o["bid"], o["ask"] = 46.0, 50.0
        result = await s.search_candidates(chain, 6700.0)
        assert result["candidates_found"] == 1


class TestMeicValidate:
    def _setup(self, sc=6820.0, lc=6880.0, sp=6580.0, lp=6520.0,
               ccw=0.10, pcw=0.10, qty=3):
        return LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", sc, "call", "sell", qty)],
            total_credit=12.0,
            strikes={"short_call": sc, "long_call": lc,
                     "short_put": sp, "long_put": lp},
            extra={"call_credit": 6.0, "put_credit": 6.0, "qty": qty,
                   "call_credit_width": ccw, "put_credit_width": pcw,
                   "wing_pts": int(lc - sc)},
        )

    @pytest.mark.asyncio
    async def test_valid_setup_passes(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(), [])
        assert res["passed"] is True and res["failures"] == []

    @pytest.mark.asyncio
    async def test_bad_call_strike_order_fails(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(lc=6810.0), [])
        assert "strike_order_call" in res["failures"]

    @pytest.mark.asyncio
    async def test_asymmetric_wings_fail(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(lp=6540.0), [])  # put wing 40 vs call 60
        assert "wing_symmetry" in res["failures"]

    @pytest.mark.asyncio
    async def test_credit_to_width_below_gate_fails(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(ccw=0.05), [])
        assert "credit_to_width_call" in res["failures"]

    @pytest.mark.asyncio
    async def test_zero_qty_fails(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(qty=0), [])
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
            extra={"call_credit": 1.40, "put_credit": 1.40, "qty": 4,
                   "short_call_mid": 1.45, "short_put_mid": 1.45},
        )
        order = s.build_order(setup)
        assert order["order_type"] == "meic_two_spreads"
        assert order["quantity"] == 4
        # stop_amount = 0.70 * total_credit 2.80 = 1.96; trigger = short_mid + 1.96
        assert order["stop_amount"] == pytest.approx(1.96)
        assert order["put_spread"]["short_strike"] == 5750
        assert order["put_spread"]["long_strike"] == 5660
        assert order["put_spread"]["credit"] == 1.40
        assert order["put_spread"]["quantity"] == 4
        assert order["put_spread"]["stop_trigger"] == pytest.approx(1.45 + 1.96)
        assert order["call_spread"]["short_strike"] == 5950
        assert order["call_spread"]["stop_trigger"] == pytest.approx(1.45 + 1.96)


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


class TestMeicComputeEm:
    def _chain(self, spot, em, step=5):
        # ATM straddle (call_mid + put_mid at the strike nearest spot) == em.
        atm = round(spot / step) * step
        half = em / 2.0
        bid, ask = round(half - 0.05, 2), round(half + 0.05, 2)
        return [
            {"strike": float(atm), "option_type": "call", "bid": bid, "ask": ask},
            {"strike": float(atm), "option_type": "put", "bid": bid, "ask": ask},
        ]

    def test_em_is_atm_straddle(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s._compute_em(self._chain(6700, 120.0), 6700.0) == pytest.approx(120.0)

    def test_em_uses_nearest_strike_to_spot(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        # spot 6702 -> nearest listed strike 6700; straddle still 120
        assert s._compute_em(self._chain(6700, 120.0), 6702.0) == pytest.approx(120.0)

    def test_em_none_when_atm_legs_missing(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        calls_only = [{"strike": 6700.0, "option_type": "call", "bid": 59.95, "ask": 60.05}]
        assert s._compute_em(calls_only, 6700.0) is None

    def test_em_none_for_empty_chain(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s._compute_em([], 6700.0) is None
