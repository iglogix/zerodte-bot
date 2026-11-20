# tests/dc_time_machine/test_iv.py
import math

import pytest

from zerodte.backtest.dc_time_machine.iv import (
    bs_delta,
    bs_price,
    front_back_ratio,
    implied_vol,
)


def test_bs_price_round_trips_through_implied_vol():
    S, K, T, r, sigma = 5000.0, 5000.0, 9 / 365, 0.04, 0.18
    price = bs_price(S, K, T, sigma, r, is_call=True)
    iv = implied_vol(price, S, K, T, r, is_call=True)
    assert iv == pytest.approx(0.18, abs=1e-3)


def test_atm_call_delta_near_half():
    d = bs_delta(5000.0, 5000.0, 9 / 365, 0.18, 0.04, is_call=True)
    assert 0.5 < d < 0.6  # slightly above 0.5 due to drift/vega skew of d1


def test_put_delta_is_negative():
    d = bs_delta(5000.0, 5000.0, 9 / 365, 0.18, 0.04, is_call=False)
    assert -0.6 < d < -0.4


def test_implied_vol_nan_when_price_below_intrinsic():
    iv = implied_vol(price=0.01, S=5100.0, K=5000.0, T=9 / 365, r=0.04, is_call=True)
    assert math.isnan(iv)


def test_front_back_ratio():
    assert front_back_ratio(0.22, 0.20) == pytest.approx(1.1)
    assert math.isnan(front_back_ratio(0.22, 0.0))
