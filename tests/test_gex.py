import math

from zerodte.backtest import gex


def test_bs_gamma_peaks_atm_and_nonneg():
    S, T, sig = 5000.0, 0.01, 0.20
    g_atm = gex.bs_gamma(S, 5000.0, T, sig)
    g_otm = gex.bs_gamma(S, 5100.0, T, sig)
    assert g_atm > 0
    assert g_atm > g_otm            # gamma is highest at the money
    assert gex.bs_gamma(S, 5000.0, 0.0, sig) == 0.0   # T=0 guard
    assert gex.bs_gamma(S, 5000.0, T, 0.0) == 0.0     # sigma=0 guard


def test_iv_from_em_roundtrips_atm_straddle():
    S, T = 5000.0, 0.01
    sigma = 0.20
    em = math.sqrt(2.0 / math.pi) * S * sigma * math.sqrt(T)   # straddle approx
    assert abs(gex.iv_from_em(S, em, T) - sigma) < 1e-9


def test_total_gex_sign_follows_call_minus_put_oi():
    S, T, sig = 5000.0, 0.01, 0.20
    call_heavy = {(5000.0, True): 10000, (5000.0, False): 1000}
    put_heavy = {(5000.0, True): 1000, (5000.0, False): 10000}
    assert gex.total_gex(call_heavy, S, sig, T) > 0     # net long dealer gamma
    assert gex.total_gex(put_heavy, S, sig, T) < 0      # net short dealer gamma


def test_magnet_strike_is_highest_gamma_oi_concentration():
    S, T, sig = 5000.0, 0.01, 0.20
    # 5000 has modest OI but max gamma; 5200 has huge OI but low gamma.
    oi = {
        (4900.0, True): 500, (4900.0, False): 500,
        (5000.0, True): 3000, (5000.0, False): 3000,
        (5200.0, True): 8000, (5200.0, False): 8000,
    }
    # near-ATM gamma dominates the far OTM despite larger OI there
    assert gex.magnet_strike(oi, S, sig, T) == 5000.0


def test_magnet_pulls_toward_oi_when_gamma_comparable():
    S, T, sig = 5000.0, 0.05, 0.20   # higher T/sigma flattens gamma across nearby strikes
    oi = {
        (4995.0, True): 100, (4995.0, False): 100,
        (5005.0, True): 9000, (5005.0, False): 9000,
    }
    assert gex.magnet_strike(oi, S, sig, T) == 5005.0


def test_gamma_flip_returns_level_between_sign_change():
    S, T, sig = 5000.0, 0.01, 0.20
    # put-heavy below, call-heavy above -> total GEX flips sign as the eval spot rises
    oi = {
        (4900.0, True): 200, (4900.0, False): 9000,
        (5100.0, True): 9000, (5100.0, False): 200,
    }
    flip = gex.gamma_flip(oi, S, sig, T)
    assert flip is not None
    assert 4900.0 <= flip <= 5100.0
