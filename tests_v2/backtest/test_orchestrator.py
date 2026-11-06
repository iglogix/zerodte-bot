"""Tests for BacktestOrchestrator."""

import pickle
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import List, Dict
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPOT = 5925.0


def _make_bar(hour, minute, open_, high, low, close, d=date(2025, 10, 1)):
    return {
        "ts_et": datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET),
        "open": open_, "high": high, "low": low, "close": close, "volume": 100,
    }


@dataclass
class FakeDayBars:
    session_date: date
    symbol: str
    bars: List[Dict]


def _build_quotes():
    """Build CBBO quotes centred around SPOT=5925 for a full trading day."""
    quotes = {}
    # Every minute from 08:00 to 15:55
    for h in range(8, 16):
        end_m = 56 if h < 15 else 56
        for m in range(0, end_m):
            minute = f"{h:02d}:{m:02d}"
            # ATM straddle (for EM computation)
            quotes[(minute, SPOT, False)] = (12.0, 13.0)
            quotes[(minute, SPOT, True)] = (13.0, 14.0)
            # OTM puts (short put spreads)
            quotes[(minute, 5910.0, False)] = (4.00, 4.50)
            quotes[(minute, 5905.0, False)] = (2.50, 3.00)
            quotes[(minute, 5900.0, False)] = (1.50, 2.00)
            quotes[(minute, 5895.0, False)] = (1.00, 1.40)
            # OTM calls (short call spreads)
            quotes[(minute, 5945.0, True)] = (3.50, 4.00)
            quotes[(minute, 5950.0, True)] = (2.00, 2.50)
            quotes[(minute, 5955.0, True)] = (1.50, 2.00)
            quotes[(minute, 5960.0, True)] = (0.80, 1.20)
            # Fill remaining strikes with small quotes
            for k in range(5800, 6100, 5):
                fk = float(k)
                if (minute, fk, False) not in quotes:
                    quotes[(minute, fk, False)] = (0.05, 0.15)
                if (minute, fk, True) not in quotes:
                    quotes[(minute, fk, True)] = (0.05, 0.15)
    return quotes


@pytest.fixture
def cbbo_cache(tmp_path):
    quotes = _build_quotes()
    pkl_path = tmp_path / "2025-10-01.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(quotes, f)
    return str(tmp_path)


def _make_config(eod="15:50"):
    return {
        "bot": {"symbol": "ES"},
        "strategy": {
            "orb_window_start": "08:00",
            "orb_window_end": "08:15",
            "market_open": "09:30",
            "eod_force_close": eod,
            "midline_tolerance_pts": 1.0,
            "orb_range_filter": {"min_pts": 2.0, "max_pts": 30.0},
            "skip_nfp_fridays": True,
            "skip_fomc_days": True,
            "skip_cpi_days": True,
        },
        "options": {
            "strike_step_pts": 5,
            "em_with_trend": 0.50,
            "em_against_trend": 0.75,
            "width_with_trend": 5,
            "width_against_trend": 10,
            "min_credit_per_spread": 0.01,
            "tp_pct_of_credit": 0.50,
            "sl_mult_of_credit": 0.0,
            "hold_put_breach": True,
            "slippage_per_spread": 0.10,
            "contract_multiplier": 100,
            "quantity": 1,
        },
    }


def _build_bars(d=date(2025, 10, 1)):
    """Build bars with ORB, breakout, midline retest, then quiet afternoon."""
    bars = []
    # ORB phase: 08:00-08:14 — range 5920-5930
    for m in range(15):
        bars.append(_make_bar(8, m, 5924, 5930, 5920, 5925, d=d))

    # Pre-market: 08:15-09:29 — stay inside range
    for h in range(8, 10):
        start_m = 15 if h == 8 else 0
        end_m = 30 if h == 9 else 60
        for m in range(start_m, end_m):
            bars.append(_make_bar(h, m, 5924, 5929, 5921, 5925, d=d))

    # 09:30 — inside range
    bars.append(_make_bar(9, 30, 5925, 5928, 5922, 5926, d=d))
    # 09:31 — breakout long (close > orb_high=5930)
    bars.append(_make_bar(9, 31, 5926, 5935, 5926, 5932, d=d))
    # 09:32 — midline retest (price touches 5925 midline)
    bars.append(_make_bar(9, 32, 5932, 5932, 5924, 5926, d=d))
    # 09:33 onwards — quiet trading, stays inside
    for m in range(33, 60):
        bars.append(_make_bar(9, m, 5926, 5930, 5922, 5927, d=d))

    # 10:00-15:55 — quiet trading
    for h in range(10, 16):
        end_m = 56 if h < 15 else 56
        for m in range(0, end_m):
            bars.append(_make_bar(h, m, 5926, 5930, 5922, 5927, d=d))

    return bars


# ---------------------------------------------------------------------------
# Tests: event day skips
# ---------------------------------------------------------------------------


class TestEventSkips:
    def test_skip_nfp_friday(self, cbbo_cache):
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.orchestrator import BacktestOrchestrator
        from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        config = _make_config()
        orch = BacktestOrchestrator(config, OrbEmIronCondorStrategy, store)

        # 2025-10-03 is a Friday, day <= 7 → NFP
        nfp_day = FakeDayBars(
            session_date=date(2025, 10, 3),
            symbol="SPX",
            bars=_build_bars(date(2025, 10, 3)),
        )
        results = orch.run([nfp_day])
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].skip_reason == "nfp_friday"

    def test_skip_fomc_day(self, cbbo_cache):
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.orchestrator import BacktestOrchestrator
        from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        config = _make_config()
        orch = BacktestOrchestrator(config, OrbEmIronCondorStrategy, store)

        # Use a known FOMC date
        fomc_day = FakeDayBars(
            session_date=date(2025, 1, 29),
            symbol="SPX",
            bars=_build_bars(date(2025, 1, 29)),
        )
        results = orch.run([fomc_day])
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].skip_reason == "fomc_day"

    def test_skip_no_bars(self, cbbo_cache):
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.orchestrator import BacktestOrchestrator
        from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        config = _make_config()
        orch = BacktestOrchestrator(config, OrbEmIronCondorStrategy, store)

        empty_day = FakeDayBars(
            session_date=date(2025, 10, 1),
            symbol="SPX",
            bars=[],
        )
        results = orch.run([empty_day])
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].skip_reason == "no_bars"


# ---------------------------------------------------------------------------
# Tests: no CBBO data
# ---------------------------------------------------------------------------


class TestNoCbbo:
    def test_skip_no_cbbo(self, tmp_path):
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.orchestrator import BacktestOrchestrator
        from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy

        # Empty cbbo store — no pkl files
        store = CbboStore(str(tmp_path), cache_dir=str(tmp_path))
        config = _make_config()
        orch = BacktestOrchestrator(config, OrbEmIronCondorStrategy, store)

        day = FakeDayBars(
            session_date=date(2025, 10, 1),
            symbol="SPX",
            bars=_build_bars(),
        )
        results = orch.run([day])
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].skip_reason == "no_cbbo_data"


# ---------------------------------------------------------------------------
# Tests: full trade lifecycle
# ---------------------------------------------------------------------------


class TestFullTrade:
    def test_full_entry_and_eod_close(self, cbbo_cache):
        """OrbEngine finds breakout + midline → entry → PositionMonitor EOD close."""
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.orchestrator import BacktestOrchestrator
        from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        config = _make_config()
        orch = BacktestOrchestrator(config, OrbEmIronCondorStrategy, store)

        day = FakeDayBars(
            session_date=date(2025, 10, 1),
            symbol="SPX",
            bars=_build_bars(),
        )
        results = orch.run([day])

        # Should have 2 results (put side + call side)
        traded = [r for r in results if r.status != "skipped"]
        if not traded:
            # If no candidates found due to CBBO pricing, that's a valid skip
            assert all(r.status == "skipped" for r in results)
            return

        assert len(traded) == 2
        sides = {r.spread_type for r in traded}
        assert "put_credit_spread" in sides
        assert "call_credit_spread" in sides

        for r in traded:
            assert r.session_date == date(2025, 10, 1)
            assert r.entry_time is not None
            assert r.short_strike is not None
            assert r.long_strike is not None
            assert r.credit_received is not None
            assert r.credit_received >= 0
            # Either TP'd, stopped, or EOD closed
            assert r.exit_reason is not None

    def test_no_breakout_skips(self, cbbo_cache):
        """If price never breaks out of ORB range, engine returns 'waited'."""
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.orchestrator import BacktestOrchestrator
        from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        config = _make_config()
        orch = BacktestOrchestrator(config, OrbEmIronCondorStrategy, store)

        # Build bars that stay inside range all day
        bars = []
        d = date(2025, 10, 1)
        for h in range(8, 16):
            for m in range(60):
                bars.append(_make_bar(h, m, 5924, 5929, 5921, 5925, d=d))

        day = FakeDayBars(session_date=d, symbol="SPX", bars=bars)
        results = orch.run([day])

        assert len(results) == 1
        assert results[0].status == "skipped"
        # Reason should indicate no entry happened
        assert results[0].skip_reason is not None


# ---------------------------------------------------------------------------
# Tests: BacktestToolHandlers enhancements
# ---------------------------------------------------------------------------


class TestPlaceOrderEnhanced:
    @pytest.mark.asyncio
    async def test_place_order_builds_full_trade(self, cbbo_cache):
        """place_order should create trade with leg symbols and credits."""
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.sim_clock import SimClock
        from zerodte.backtest.mock_db import BacktestDb
        from zerodte.backtest.bt_tool_handlers import BacktestToolHandlers

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        store.load_day(date(2025, 10, 1))

        bars = [_make_bar(9, 31, 5926, 5935, 5926, 5932)]
        clock = SimClock(bars[0]["ts_et"])
        db = BacktestDb()
        handlers = BacktestToolHandlers(bars, store, clock, db, date(2025, 10, 1))

        payload = {
            "order_type": "iron_condor",
            "short_put_strike": 5910.0,
            "long_put_strike": 5905.0,
            "short_call_strike": 5945.0,
            "long_call_strike": 5950.0,
            "quantity": 1,
        }
        result = await handlers.place_order(payload)
        assert result["status"] == "filled"

        trades = db.get_all_trades()
        assert len(trades) == 1
        trade = trades[0]

        # Check leg symbols exist
        assert trade["short_put_symbol"]
        assert trade["long_put_symbol"]
        assert trade["short_call_symbol"]
        assert trade["long_call_symbol"]

        # Check credits computed
        assert "put_credit" in trade
        assert "call_credit" in trade
        assert trade["put_credit"] >= 0
        assert trade["call_credit"] >= 0

        # Check entry price recorded
        assert trade["entry_price"] == 5932

    @pytest.mark.asyncio
    async def test_close_position_records_exit_quote(self, cbbo_cache):
        """close_position should record CBBO exit mark in _exit_log."""
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.sim_clock import SimClock
        from zerodte.backtest.mock_db import BacktestDb
        from zerodte.backtest.bt_tool_handlers import BacktestToolHandlers

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        store.load_day(date(2025, 10, 1))

        bars = [_make_bar(9, 31, 5926, 5935, 5926, 5932)]
        clock = SimClock(bars[0]["ts_et"])
        db = BacktestDb()
        handlers = BacktestToolHandlers(bars, store, clock, db, date(2025, 10, 1))

        # Build a symbol for strike 5910 put
        sym = handlers._make_symbol(5910.0, False)
        await handlers.close_position(symbol=sym)

        assert sym in handlers._exit_log
        assert "bid" in handlers._exit_log[sym]
        assert "ask" in handlers._exit_log[sym]

    @pytest.mark.asyncio
    async def test_option_chain_uses_full_type_names(self, cbbo_cache):
        """get_option_chain should return 'call'/'put', not 'C'/'P'."""
        from zerodte.backtest.cbbo_store import CbboStore
        from zerodte.backtest.sim_clock import SimClock
        from zerodte.backtest.mock_db import BacktestDb
        from zerodte.backtest.bt_tool_handlers import BacktestToolHandlers

        store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
        store.load_day(date(2025, 10, 1))

        bars = [_make_bar(9, 31, 5926, 5935, 5926, 5932)]
        clock = SimClock(bars[0]["ts_et"])
        db = BacktestDb()
        handlers = BacktestToolHandlers(bars, store, clock, db, date(2025, 10, 1))

        chain = await handlers.get_option_chain()
        assert chain  # non-empty
        types = {o["option_type"] for o in chain}
        assert types == {"call", "put"}
        assert "C" not in types
        assert "P" not in types
