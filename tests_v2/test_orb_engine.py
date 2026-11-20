"""Tests for OrbEngine three-phase entry."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from zerodte.agent.orb_engine import OrbEngine
from zerodte.agent.base_engine import EngineDeps
from zerodte.agent.decision_engine import CycleResult
from zerodte.backtest.sim_clock import WallClock

ET = ZoneInfo("America/New_York")


class _MockClock:
    """Test clock returning a fixed time."""
    def __init__(self, hour=10, minute=0):
        self._time = datetime(2025, 10, 1, hour, minute, tzinfo=ET)

    def now(self, tz=None):
        if tz:
            return self._time.astimezone(tz)
        return self._time

    async def sleep(self, seconds):
        # Instant, but advance the clock so _idle_until_eod() reaches EOD and returns.
        self._time += timedelta(seconds=seconds)


def _make_deps(config=None, clock=None):
    cfg = config or {
        "bot": {"symbol": "ES"},
        "strategy": {
            "orb_window_start": "08:00", "orb_window_end": "08:15",
            "eod_force_close": "15:50",
            "midline_tolerance_pts": 0.25,
            "orb_range_filter": {"min_pts": 2.0, "max_pts": 30.0},
        },
    }
    deps = EngineDeps(
        tool_handlers=MagicMock(),
        risk_manager=MagicMock(),
        anomaly_detector=None,
        regime_classifier=None,
        decision_file_logger=MagicMock(),
        db=MagicMock(),
        mode="manual",
        config=cfg,
        clock=clock,
    )
    # Make tool_handler methods async
    deps.tool_handlers.get_orb_levels = AsyncMock()
    deps.tool_handlers.get_breakout_status = AsyncMock()
    deps.tool_handlers.get_futures_price = AsyncMock()
    deps.tool_handlers.get_spx_price = AsyncMock()
    deps.tool_handlers.get_option_chain = AsyncMock(return_value=[])
    deps.tool_handlers.get_orb_spot = AsyncMock(return_value={"spot": 5850.0})
    deps.tool_handlers.get_5min_candles = AsyncMock(return_value=[])
    deps.tool_handlers.place_order = AsyncMock(return_value={"status": "filled"})
    deps.tool_handlers.reset_orb_state = MagicMock()
    deps.tool_handlers.check_midline_touch = MagicMock(return_value=False)
    # Concurrency guard reads open trades; a bare MagicMock fails safe ("limit hit").
    deps.db.get_open_trades = AsyncMock(return_value=[])
    return deps


def _make_strategy():
    s = MagicMock()
    s._orb_bias = None
    s.search_candidates = AsyncMock(return_value={
        "candidates_found": 1, "top_candidate": MagicMock(strikes={"short_put": 5830})
    })
    s.validate = AsyncMock(return_value={"passed": True})
    s.build_order = MagicMock(return_value={"order_type": "iron_condor"})
    return s


class TestSetupPhase:
    @pytest.mark.asyncio
    async def test_stores_orb_levels(self):
        deps = _make_deps()
        deps.tool_handlers.get_orb_levels.return_value = {
            "high": 5860, "low": 5840, "midline": 5850, "range_pts": 20,
        }
        engine = OrbEngine(deps, _make_strategy())
        assert await engine._setup_phase() is True
        assert engine._orb_levels["high"] == 5860

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self):
        deps = _make_deps()
        deps.tool_handlers.get_orb_levels.return_value = {"error": "no_candles"}
        engine = OrbEngine(deps, _make_strategy())
        assert await engine._setup_phase() is False

    @pytest.mark.asyncio
    async def test_skips_narrow_range(self):
        deps = _make_deps()
        deps.tool_handlers.get_orb_levels.return_value = {
            "high": 5851, "low": 5850, "midline": 5850.5, "range_pts": 1.0,
        }
        engine = OrbEngine(deps, _make_strategy())
        assert await engine._setup_phase() is False

    @pytest.mark.asyncio
    async def test_skips_wide_range(self):
        deps = _make_deps()
        deps.tool_handlers.get_orb_levels.return_value = {
            "high": 5900, "low": 5840, "midline": 5870, "range_pts": 60,
        }
        engine = OrbEngine(deps, _make_strategy())
        assert await engine._setup_phase() is False


class TestEntryWatch:
    @pytest.mark.asyncio
    async def test_exits_at_eod(self):
        clock = _MockClock(hour=16, minute=0)  # after EOD
        deps = _make_deps(clock=clock)
        engine = OrbEngine(deps, _make_strategy())
        engine._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
        result = await engine._entry_watch()
        assert result.outcome == "waited"
        assert "eod" in result.reason

    @pytest.mark.asyncio
    async def test_detects_breakout_and_enters_on_midline(self):
        """Full flow: breakout → midline touch → entry."""
        clock = _MockClock(hour=10, minute=0)
        deps = _make_deps(clock=clock)
        strategy = _make_strategy()
        engine = OrbEngine(deps, strategy)
        engine._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
        engine._poll_interval = 0  # no wait

        # First poll: breakout detected
        deps.tool_handlers.get_futures_price.return_value = {"price": 101}
        deps.tool_handlers.get_breakout_status.return_value = {"status": "breakout", "bias": "long", "price": 101}
        # Midline touch on first poll after breakout
        deps.tool_handlers.check_midline_touch.return_value = True

        result = await engine._entry_watch()

        assert result.outcome == "executed"
        assert strategy._orb_bias == "long"

    @pytest.mark.asyncio
    async def test_sets_orb_bias_on_strategy(self):
        clock = _MockClock(hour=10, minute=0)
        deps = _make_deps(clock=clock)
        strategy = _make_strategy()
        engine = OrbEngine(deps, strategy)
        engine._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
        engine._poll_interval = 0

        deps.tool_handlers.get_futures_price.return_value = {"price": 89}
        deps.tool_handlers.get_breakout_status.return_value = {"status": "breakout", "bias": "short", "price": 89}
        deps.tool_handlers.check_midline_touch.return_value = True

        await engine._entry_watch()

        assert strategy._orb_bias == "short"


class TestPreflight:
    @pytest.mark.asyncio
    async def test_anomaly_blocks(self):
        deps = _make_deps()
        deps.anomaly_detector = AsyncMock()
        deps.anomaly_detector.check = AsyncMock(return_value={"status": "anomaly", "reason": "spike"})
        engine = OrbEngine(deps, _make_strategy())
        result = await engine._run_preflight(5850)
        assert "anomaly" in result

    @pytest.mark.asyncio
    async def test_regime_skip_session(self):
        deps = _make_deps()
        deps.regime_classifier = AsyncMock()
        deps.regime_classifier.classify = AsyncMock(return_value={"verdict": "skip_session", "reasoning": "volatile"})
        engine = OrbEngine(deps, _make_strategy())
        result = await engine._run_preflight(5850)
        assert "skip_session" in result

    @pytest.mark.asyncio
    async def test_passes_when_no_advisors(self):
        deps = _make_deps()
        engine = OrbEngine(deps, _make_strategy())
        result = await engine._run_preflight(5850)
        assert result is None

    @pytest.mark.asyncio
    async def test_preflight_runs_once(self):
        """After first breakout, preflight should only run once."""
        clock = _MockClock(hour=10, minute=0)
        deps = _make_deps(clock=clock)
        strategy = _make_strategy()
        engine = OrbEngine(deps, strategy)
        engine._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
        engine._poll_interval = 0

        call_count = 0

        async def counting_price(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"price": 101}

        deps.tool_handlers.get_futures_price = counting_price
        deps.tool_handlers.get_breakout_status.return_value = {"status": "breakout", "bias": "long", "price": 101}
        # First poll: midline not touched. Second poll: touched.
        deps.tool_handlers.check_midline_touch.side_effect = [False, True]

        result = await engine._entry_watch()

        assert result.outcome == "executed"
        assert engine._preflight_passed is True


class TestExecuteEntry:
    @pytest.mark.asyncio
    async def test_validates_and_places(self):
        # An injected clock: the post-entry idle must never sleep on the wall clock.
        deps = _make_deps(clock=_MockClock(hour=10, minute=0))
        strategy = _make_strategy()
        engine = OrbEngine(deps, strategy)

        result = await engine._execute_entry(5850, "long")

        assert result.outcome == "executed"
        strategy.search_candidates.assert_awaited_once()
        strategy.validate.assert_awaited_once()
        strategy.build_order.assert_called_once()
        deps.tool_handlers.place_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_candidates(self):
        deps = _make_deps()
        strategy = _make_strategy()
        strategy.search_candidates.return_value = {"candidates_found": 0, "top_candidate": None}
        engine = OrbEngine(deps, strategy)

        result = await engine._execute_entry(5850, "long")
        assert result.outcome == "waited"
        assert "no_candidates" in result.reason

    @pytest.mark.asyncio
    async def test_validation_fails(self):
        deps = _make_deps()
        strategy = _make_strategy()
        strategy.validate.return_value = {"passed": False, "failures": ["strike_order"]}
        engine = OrbEngine(deps, strategy)

        result = await engine._execute_entry(5850, "long")
        assert result.outcome == "waited"
        assert "validation_failed" in result.reason
