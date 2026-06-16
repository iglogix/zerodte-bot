# tests/test_gex_regime.py
from datetime import date
from src_v2.backtest import gex_regime as gr

def test_slice_and_expiries():
    moi = {
        (date(2022, 10, 5), 5000.0, True): 100,
        (date(2022, 10, 5), 5000.0, False): 80,
        (date(2022, 10, 6), 5000.0, True): 40,
    }
    assert gr.expiries(moi) == [date(2022, 10, 5), date(2022, 10, 6)]
    assert gr.slice_expiry(moi, date(2022, 10, 5)) == {(5000.0, True): 100, (5000.0, False): 80}

def test_years_to_expiry_0dte_matches_butterfly():
    # entry 09:30 -> 6.5 trading hours to 16:00 -> 1.0 trading day -> 1/252 yr
    assert abs(gr.years_to_expiry("09:30", 0) - (1.0 / 252.0)) < 1e-12

def test_years_to_expiry_1dte_adds_one_session():
    assert abs(gr.years_to_expiry("09:30", 1) - (2.0 / 252.0)) < 1e-12

def test_intraday_fraction_floor():
    # after close the fraction floors at 0.25h/6.5h, never zero/negative
    assert gr.years_to_expiry("16:30", 0) > 0
