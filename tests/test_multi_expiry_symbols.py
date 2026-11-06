# tests/test_multi_expiry_symbols.py
from datetime import date
from zerodte.backtest.meic_backfill import multi_expiry_symbols, parse_osi_multi

def test_multi_expiry_symbols_cover_each_expiry():
    exps = [date(2024, 1, 5), date(2024, 1, 19)]
    syms = multi_expiry_symbols(exps, 5000.0, 5000.0, buffer_pts=10, step_pts=5)
    # 5 strikes (4990..5010) x 2 cp x 2 expiries = 20 symbols
    assert len(syms) == 20
    assert any("240105" in s and s.rstrip().endswith("C05000000") for s in syms)
    assert any("240119" in s for s in syms)

def test_multi_expiry_symbols_asymmetric_down():
    # wider downside so the put-driven gamma flip is never truncated
    syms = multi_expiry_symbols([date(2024, 1, 5)], 5000.0, 5000.0,
                                buffer_down_pts=400, buffer_up_pts=200, step_pts=5)
    strikes = sorted({int(s[-8:]) / 1000.0 for s in syms})
    assert min(strikes) == 4600.0 and max(strikes) == 5200.0   # -400 / +200

def test_parse_osi_multi_roundtrip():
    assert parse_osi_multi("SPXW  240119P04700000") == (date(2024, 1, 19), 4700.0, False)
    assert parse_osi_multi("garbage") is None
