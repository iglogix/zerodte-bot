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

def test_standing_regime_sign_and_distflip():
    # put-heavy below spot -> net short dealer gamma at spot (negative), flip below spot
    moi = {}
    for k in range(4800, 5201, 25):
        moi[(date(2022,10,5), float(k), True)] = 500 if k >= 5000 else 100
        moi[(date(2022,10,5), float(k), False)] = 500 if k <= 5000 else 100
    t = {date(2022,10,5): gr.years_to_expiry("09:30", 0)}
    feat = gr.standing_regime(moi, S=5000.0, em=40.0, sigma=0.20, t_by_expiry=t)
    assert feat.sign in (-1, 0, 1)
    assert feat.total_gex == gr.total_gex_multi(moi, 5000.0, 0.20, t)
    if feat.flip is not None:
        assert abs(feat.dist_flip_em - (5000.0 - feat.flip) / 40.0) < 1e-9

def test_proxy_uses_0dte_kernel_on_1dte_oi():
    # proxy = run single-expiry gex on the 1DTE OI but at the 0DTE time-to-expiry
    moi = {
        (date(2022,10,6), 5000.0, True): 1000,   # 1DTE expiry
        (date(2022,10,6), 5000.0, False): 200,
        (date(2022,10,5), 5000.0, True): 9,      # 0DTE expiry (ignored by proxy)
    }
    feat = gr.proxy_regime(moi, expiry_1dte=date(2022,10,6), S=5000.0, em=40.0,
                           sigma=0.20, entry_anchor="09:30")
    # call-heavy 1DTE -> positive dealer gamma
    assert feat.sign == 1
