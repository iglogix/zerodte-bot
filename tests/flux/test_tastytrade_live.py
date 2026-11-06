from datetime import date

from zerodte.backtest.osi_symbol import parse_osi
from zerodte.flux.sources.tastytrade_live import TastyTradeLiveSource

SPOT = 7000.0


def _quote_fn(symbols):
    """Fake DXLink: ATM-ish strikes have quotes; others don't (illiquid wings omitted)."""
    out = {}
    for s in symbols:
        strike, _is_call = parse_osi(s)
        if abs(strike - SPOT) <= 10:
            out[s] = (39.9, 40.1)
    return out


def test_live_source_builds_chain_with_computed_iv():
    src = TastyTradeLiveSource(quote_fn=_quote_fn, spot_fn=lambda: SPOT, ref_date=date(2026, 6, 8))
    snap = src.chain(date(2026, 6, 17))
    assert snap.underlying == "SPX" and snap.expiry == date(2026, 6, 17)
    assert len(snap.rows) >= 2
    assert all(0.0 < r.iv < 2.0 for r in snap.rows)
    assert any(r.is_call for r in snap.rows) and any(not r.is_call for r in snap.rows)


def test_live_source_empty_when_no_spot():
    src = TastyTradeLiveSource(quote_fn=lambda s: {}, spot_fn=lambda: None)
    assert src.chain(date(2026, 6, 17)).rows == ()


def test_live_source_subscribes_atm_band_symbols():
    seen = {}
    def qf(symbols):
        seen["symbols"] = symbols
        return _quote_fn(symbols)
    TastyTradeLiveSource(quote_fn=qf, spot_fn=lambda: SPOT,
                         half_width_pts=20, step_pts=5).chain(date(2026, 6, 17))
    # ±20 / 5 = 9 strikes × 2 sides = 18 OSI symbols, all SPXW for the right expiry
    assert len(seen["symbols"]) == 18
    assert all("260617" in s for s in seen["symbols"])
