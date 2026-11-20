"""Tests for zerodte.backtest.cbbo_store."""

import pickle
from datetime import date

import pytest


@pytest.fixture
def cbbo_cache(tmp_path):
    """Create a minimal pickle cache with 1 day of CBBO quotes."""
    quotes = {
        ("09:45", 5900.0, False): (3.50, 4.00),
        ("09:45", 5895.0, False): (1.50, 2.00),
        ("09:45", 5950.0, True): (3.00, 3.50),
        ("09:45", 5955.0, True): (1.00, 1.50),
        ("09:45", 5925.0, False): (12.00, 13.00),
        ("09:45", 5925.0, True): (13.00, 14.00),
        ("10:00", 5900.0, False): (2.00, 2.50),
        ("10:00", 5895.0, False): (0.80, 1.20),
        ("10:00", 5950.0, True): (2.00, 2.50),
        ("10:00", 5955.0, True): (0.50, 1.00),
        ("10:00", 5925.0, False): (10.00, 11.00),
        ("10:00", 5925.0, True): (11.00, 12.00),
    }
    pkl_path = tmp_path / "2025-10-01.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(quotes, f)
    return str(tmp_path)


def test_load_day(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    n = store.load_day(date(2025, 10, 1))
    assert n == 12
    assert store.loaded_date == date(2025, 10, 1)


def test_lookup(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    bid, ask = store.lookup("09:45", 5900.0, False)
    assert bid == 3.50
    assert ask == 4.00
    assert store.lookup("09:45", 9999.0, False) is None


def test_lookup_stale_fallback(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    result = store.lookup("09:46", 5900.0, False, max_stale_minutes=5)
    assert result == (3.50, 4.00)


def test_atm_straddle(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    em = store.atm_straddle("09:45", spot=5925.0)
    assert em == pytest.approx(26.0)


def test_derive_spot(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    spot = store.derive_spot("09:45", bar_price=5925.0)
    assert spot == pytest.approx(5926.0)


def test_available_strikes(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    assert 5900.0 in store.available_strikes
    assert 5925.0 in store.available_strikes


def test_preload_cache(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    mem = CbboStore.preload_cache(cbbo_cache)
    assert date(2025, 10, 1) in mem
    store = CbboStore(cbbo_cache, preloaded=mem)
    n = store.load_day(date(2025, 10, 1))
    assert n == 12


def test_load_missing_day(cbbo_cache):
    from zerodte.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    n = store.load_day(date(2099, 1, 1))
    assert n == 0
