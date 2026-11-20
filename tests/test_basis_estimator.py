import pickle
from datetime import date
from pathlib import Path

import pytest

from zerodte.backtest.basis_estimator import (
    build_monthly_basis_table,
    compute_daily_basis,
    estimate_spx_from_es,
)


def _fake_cbbo_pickle(tmp_path: Path, day: date, atm_strike: float, call_mid: float, put_mid: float) -> Path:
    """Write a minimal pickle that derive_spot can read at 09:30."""
    quotes = {
        ("09:30", atm_strike, True): (call_mid - 0.5, call_mid + 0.5),
        ("09:30", atm_strike, False): (put_mid - 0.5, put_mid + 0.5),
    }
    p = tmp_path / f"{day.isoformat()}.pkl"
    with open(p, "wb") as f:
        pickle.dump(quotes, f)
    return p


def test_compute_daily_basis_uses_put_call_parity(tmp_path):
    # SPX_spot = K + C - P. With K=5000, C=10, P=4 -> spot=5006. ES=5010 -> basis=4.
    _fake_cbbo_pickle(tmp_path, date(2025, 9, 22), atm_strike=5000.0, call_mid=10.0, put_mid=4.0)
    basis = compute_daily_basis(tmp_path, date(2025, 9, 22), es_open=5010.0, anchor_minute="09:30")
    assert basis == pytest.approx(4.0, abs=0.01)


def test_build_monthly_basis_table_takes_median(tmp_path):
    # Two days in Sep 2025: bases 4.0 and 6.0 -> median 5.0
    _fake_cbbo_pickle(tmp_path, date(2025, 9, 22), 5000.0, 10.0, 4.0)  # spot=5006, basis 4 vs ES 5010
    _fake_cbbo_pickle(tmp_path, date(2025, 9, 23), 5000.0, 10.0, 6.0)  # spot=5004, basis 6 vs ES 5010
    es_opens = {date(2025, 9, 22): 5010.0, date(2025, 9, 23): 5010.0}
    table = build_monthly_basis_table(tmp_path, es_opens, anchor_minute="09:30")
    assert table[(2025, 9)] == pytest.approx(5.0, abs=0.01)


def test_estimate_spx_from_es_uses_table_or_fallback():
    table = {(2025, 9): 4.0}
    assert estimate_spx_from_es(date(2025, 9, 15), 5010.0, table, default_basis=10.0) == 5006.0
    # Month not in table → fallback
    assert estimate_spx_from_es(date(2024, 1, 15), 5010.0, table, default_basis=10.0) == 5000.0
