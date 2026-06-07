# tests/dc_time_machine/test_wf_cv.py
from datetime import date, timedelta

import pytest

from src_v2.backtest.dc_time_machine.wf_cv import split_folds, annualized_sharpe


def test_split_folds_ratio_and_count():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    folds = split_folds(days, ratio=5, n_folds=4)
    assert len(folds) == 4
    for train, val in folds:
        # each fold: train ≈ 5× val, train precedes val, no overlap
        assert len(train) >= 4 * len(val)
        assert max(train) < min(val)


def test_annualized_sharpe_positive_for_steady_wins():
    pnls = [100.0, 120.0, 90.0, 110.0, 105.0]
    assert annualized_sharpe(pnls) > 0


def test_annualized_sharpe_zero_for_empty_or_flat():
    assert annualized_sharpe([]) == 0.0
    assert annualized_sharpe([50.0, 50.0]) == 0.0
