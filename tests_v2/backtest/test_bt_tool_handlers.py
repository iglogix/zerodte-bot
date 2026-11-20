"""Tests for BacktestToolHandlers."""

import pickle
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def _make_bar(hour, minute, open_, high, low, close, d=date(2025, 10, 1)):
    return {
        "ts_et": datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET),
        "open": open_, "high": high, "low": low, "close": close, "volume": 100,
    }


@pytest.fixture
def cbbo_cache(tmp_path):
    quotes = {}
    for minute in ["08:00", "08:05", "08:10", "09:30", "09:31", "09:32", "09:33"]:
        quotes[(minute, 5925.0, False)] = (12.0, 13.0)
        quotes[(minute, 5925.0, True)] = (13.0, 14.0)
        quotes[(minute, 5910.0, False)] = (4.00, 4.50)
        quotes[(minute, 5905.0, False)] = (2.50, 3.00)
        quotes[(minute, 5945.0, True)] = (3.50, 4.00)
        quotes[(minute, 5955.0, True)] = (1.50, 2.00)
        for k in range(5800, 6100, 5):
            if (minute, float(k), False) not in quotes:
                quotes[(minute, float(k), False)] = (0.05, 0.15)
            if (minute, float(k), True) not in quotes:
                quotes[(minute, float(k), True)] = (0.05, 0.15)
    pkl_path = tmp_path / "2025-10-01.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(quotes, f)
    return str(tmp_path)


@pytest.fixture
def handlers(cbbo_cache):
    from zerodte.backtest.bt_tool_handlers import BacktestToolHandlers
    from zerodte.backtest.cbbo_store import CbboStore
    from zerodte.backtest.mock_db import BacktestDb
    from zerodte.backtest.sim_clock import SimClock

    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))

    bars = (
        [_make_bar(8, m, 5924, 5930, 5920, 5925) for m in range(15)] +  # ORB
        [_make_bar(9, 30, 5925, 5928, 5922, 5926)] +  # inside range
        [_make_bar(9, 31, 5926, 5935, 5926, 5932)] +  # breakout long
        [_make_bar(9, 32, 5932, 5932, 5924, 5926)] +  # midline retest
        [_make_bar(9, 33, 5927, 5935, 5926, 5933)]    # fill bar
    )

    clock = SimClock(bars[0]["ts_et"])
    db = BacktestDb()
    return BacktestToolHandlers(bars, store, clock, db, date(2025, 10, 1))


@pytest.mark.asyncio
async def test_get_orb_levels(handlers):
    result = await handlers.get_orb_levels("ES", "08:00", "08:15")
    assert result["high"] == 5930
    assert result["low"] == 5920
    assert result["midline"] == 5925
    assert result["range_pts"] == 10


@pytest.mark.asyncio
async def test_get_futures_price(handlers):
    result = await handlers.get_futures_price("ES")
    assert result["price"] == 5925  # first bar close


@pytest.mark.asyncio
async def test_breakout_latches(handlers):
    await handlers.get_orb_levels("ES", "08:00", "08:15")
    # Inside range
    result = await handlers.get_breakout_status(5925)
    assert result["status"] == "inside_range"
    # Breakout long
    result = await handlers.get_breakout_status(5931)
    assert result["status"] == "breakout"
    assert result["bias"] == "long"
    # Should stay latched
    result = await handlers.get_breakout_status(5920)
    assert result["bias"] == "long"


def test_midline_touch(handlers):
    # Backtest uses bar range (low/high) for midline detection.
    # Handlers fixture's first bar: low=5920, high=5930.
    # Midline 5925 is within [5920, 5930] → True regardless of `price` arg.
    assert handlers.check_midline_touch(5925.0, 5925.0, 0.5) is True
    # Midline outside bar range → False
    assert handlers.check_midline_touch(9999.0, 9999.0, 0.5) is False


@pytest.mark.asyncio
async def test_get_option_chain(handlers):
    # Set bar to 09:33 (has CBBO data)
    handlers.set_bar_index(18)  # 15 ORB bars + 4 post-open bars, last is 09:33
    chain = await handlers.get_option_chain()
    assert len(chain) > 0
    assert all("strike" in o and "bid" in o for o in chain)


@pytest.mark.asyncio
async def test_place_order(handlers):
    result = await handlers.place_order({"order_type": "iron_condor"})
    assert "trade_id" in result
    trades = handlers._db.get_all_trades()
    assert len(trades) == 1


@pytest.mark.asyncio
async def test_advance_bar(handlers):
    assert handlers.current_bar["ts_et"].hour == 8
    assert handlers.current_bar["ts_et"].minute == 0
    handlers.advance_bar()
    assert handlers.current_bar["ts_et"].minute == 1


def test_reset_orb_state(handlers):
    handlers._orb_levels = {"high": 100}
    handlers._breakout_bias = "long"
    handlers.reset_orb_state()
    assert handlers._orb_levels is None
    assert handlers._breakout_bias is None
