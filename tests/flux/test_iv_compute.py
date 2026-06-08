import pytest

from src_v2.flux.iv_compute import parity_spot, compute_iv_rows


def test_parity_spot_from_atm_call_put():
    # call_mid == put_mid at 7000 → forward ≈ 7000
    rows = [(7000.0, True, 39.9, 40.1), (7000.0, False, 39.9, 40.1),
            (6900.0, True, 110.0, 111.0)]
    assert parity_spot(rows) == pytest.approx(7000.0, abs=0.5)


def test_parity_spot_none_without_matched_strike():
    assert parity_spot([(7000.0, True, 40, 41)]) is None   # no put at any strike


def test_compute_iv_rows_inverts_bs_and_keeps_atm():
    # spot ~7000, ~9 DTE; an ATM call/put with real time value → invertible IV & delta
    rows = [(7000.0, True, 39.9, 40.1), (7000.0, False, 39.9, 40.1)]
    out, spot = compute_iv_rows(rows, T=9 / 365.0)
    assert spot == pytest.approx(7000.0, abs=0.5)
    assert len(out) == 2
    call = [r for r in out if r.is_call][0]
    assert 0.05 < call.iv < 1.0          # a sane IV
    assert 0.3 < call.delta < 0.7        # ATM call delta ≈ 0.5


def test_compute_iv_rows_drops_below_intrinsic_or_zero_mid():
    rows = [(7000.0, True, 0.0, 0.0),                 # zero mid → dropped
            (6000.0, True, 0.0, 0.05)]                # mid 0.025 < intrinsic 1000 → NaN → dropped
    out, _ = compute_iv_rows(rows, T=9 / 365.0, spot=7000.0)
    assert out == []
