# tests/test_jl_backfill.py
from datetime import date, datetime

from scripts.backfill_jl_cbbo import jl_center_and_symbols


def _bars_0930(open_spot):
    return [{"ts_et": datetime(2025, 1, 2, 9, 30), "open": open_spot,
             "high": open_spot + 1, "low": open_spot - 1, "close": open_spot}]


def test_center_and_symbol_window():
    center, symbols = jl_center_and_symbols(
        date(2025, 1, 2), _bars_0930(5000.0), half_width_pts=100, step_pts=5)
    assert center == 5000.0
    # symbols span 4900..5100 (center +/- 100), both call+put at every 5 pts
    assert len(symbols) == ((5100 - 4900) // 5 + 1) * 2
    assert all("SPXW" in s for s in symbols)


def test_center_none_when_no_0930_bar():
    center, symbols = jl_center_and_symbols(
        date(2025, 1, 2), [], half_width_pts=100, step_pts=5)
    assert center is None and symbols == []
