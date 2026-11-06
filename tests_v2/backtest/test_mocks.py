"""Tests for BacktestDb and BacktestCache."""

import pickle
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


# --- BacktestDb tests ---

@pytest.mark.asyncio
async def test_db_add_and_get_open():
    from zerodte.backtest.mock_db import BacktestDb
    db = BacktestDb()
    await db.add_trade({"trade_id": "T1", "status": "open", "credit": 1.5})
    trades = await db.get_open_trades()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "T1"


@pytest.mark.asyncio
async def test_db_update_trade():
    from zerodte.backtest.mock_db import BacktestDb
    db = BacktestDb()
    await db.add_trade({"trade_id": "T1", "status": "open"})
    await db.update_trade("T1", {"status": "closed", "exit_reason": "tp"})
    trades = await db.get_open_trades()
    assert len(trades) == 0
    all_trades = db.get_all_trades()
    assert all_trades[0]["status"] == "closed"


@pytest.mark.asyncio
async def test_db_get_open_excludes_closed():
    from zerodte.backtest.mock_db import BacktestDb
    db = BacktestDb()
    await db.add_trade({"trade_id": "T1", "status": "open"})
    await db.add_trade({"trade_id": "T2", "status": "closed"})
    await db.add_trade({"trade_id": "T3", "status": "expired"})
    trades = await db.get_open_trades()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "T1"


# --- BacktestCache tests ---

@pytest.fixture
def cbbo_cache(tmp_path):
    quotes = {
        ("09:45", 5900.0, False): (3.50, 4.00),
        ("09:45", 5900.0, True): (2.00, 2.50),
    }
    pkl_path = tmp_path / "2025-10-01.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(quotes, f)
    return str(tmp_path)


@pytest.mark.asyncio
async def test_cache_returns_quote(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    from zerodte.backtest.mock_cache import BacktestCache
    from zerodte.backtest.sim_clock import SimClock

    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    clock = SimClock(datetime(2025, 10, 1, 9, 45, tzinfo=ET))
    cache = BacktestCache(store, clock)

    # OCC symbol: SPXW  251001P05900000 (put at 5900)
    quote = await cache.get_cached_quote("SPXW  251001P05900000")
    assert quote is not None
    assert quote["bid"] == 3.50
    assert quote["ask"] == 4.00
    assert quote["price"] == pytest.approx(3.75)


@pytest.mark.asyncio
async def test_cache_returns_none_for_missing(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    from zerodte.backtest.mock_cache import BacktestCache
    from zerodte.backtest.sim_clock import SimClock

    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    clock = SimClock(datetime(2025, 10, 1, 9, 45, tzinfo=ET))
    cache = BacktestCache(store, clock)

    quote = await cache.get_cached_quote("SPXW  251001P09999000")
    assert quote is None
