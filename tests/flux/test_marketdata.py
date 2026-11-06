# tests/flux/test_marketdata.py
from datetime import date

from zerodte.flux.sources.marketdata import MarketDataSource

# Columnar response like marketdata.app returns (columns= set, so NO "s" key)
CANNED = {
    "side": ["call", "put", "call"],
    "strike": [7000.0, 7000.0, 7010.0],
    "bid": [40.0, 38.0, 35.0],
    "ask": [41.0, 39.0, 36.0],
    "iv": [0.180, 0.190, None],          # last row has null iv → dropped
    "delta": [0.52, -0.48, 0.47],
    "updated": [1780940400, 1780940400, 1780940400],
}


def test_parse_columnar_into_chainsnapshot():
    src = MarketDataSource(token="x", fetch_json=lambda url: CANNED)
    snap = src.chain(date(2026, 6, 12))
    assert snap.underlying == "SPX" and snap.expiry == date(2026, 6, 12)
    assert len(snap.rows) == 2                       # null-iv row dropped
    call = [r for r in snap.rows if r.is_call][0]
    assert call.strike == 7000.0 and call.iv == 0.180 and call.delta == 0.52
    put = [r for r in snap.rows if not r.is_call][0]
    assert put.delta == -0.48


def test_url_includes_strikelimit_columns_and_date():
    seen = {}
    def fake(url):
        seen["url"] = url
        return CANNED
    src = MarketDataSource(token="tok", fetch_json=fake)
    src.chain(date(2026, 6, 12), on_date=date(2026, 4, 9))
    u = seen["url"]
    assert "/options/chain/SPX/" in u
    assert "expiration=2026-06-12" in u
    assert "date=2026-04-09" in u
    assert "strikeLimit=25" in u
    assert "columns=" in u


def test_empty_response_is_empty_chain_not_error():
    src = MarketDataSource(token="x", fetch_json=lambda url: {})
    snap = src.chain(date(2026, 6, 12))
    assert snap.rows == ()
