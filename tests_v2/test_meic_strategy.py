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
