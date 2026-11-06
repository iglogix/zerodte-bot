from datetime import date

from zerodte.flux.cache import ChainCache
from zerodte.flux.sources.marketdata import MarketDataSource

CANNED = {
    "side": ["call", "put"],
    "strike": [7000.0, 7000.0],
    "bid": [40.0, 38.0],
    "ask": [41.0, 39.0],
    "iv": [0.180, 0.190],
    "delta": [0.52, -0.48],
    "updated": [1780940400, 1780940400],
}


def test_eod_fetch_is_cached(tmp_path):
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        return CANNED

    src = MarketDataSource(token="x", fetch_json=fetch, cache=ChainCache(tmp_path / "c.db"))
    a = src.chain(date(2026, 6, 12), on_date=date(2026, 4, 9))
    b = src.chain(date(2026, 6, 12), on_date=date(2026, 4, 9))   # second call → cache hit
    assert calls["n"] == 1                                       # fetched only once
    assert a.rows[0].iv == b.rows[0].iv == 0.180


def test_live_fetch_is_not_cached(tmp_path):
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        return CANNED

    src = MarketDataSource(token="x", fetch_json=fetch, cache=ChainCache(tmp_path / "c.db"))
    src.chain(date(2026, 6, 12))   # on_date=None → live, never cached
    src.chain(date(2026, 6, 12))
    assert calls["n"] == 2
