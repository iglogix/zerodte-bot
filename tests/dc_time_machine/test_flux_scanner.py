# tests/dc_time_machine/test_flux_scanner.py
from datetime import date

import pytest

from zerodte.backtest.dc_time_machine.flux_scanner import ratio_zscore, rank_candidates


def test_ratio_zscore_positive_when_today_above_history():
    z = ratio_zscore(today=1.20, history=[1.00, 1.02, 0.98, 1.01])
    assert z > 1.0


def test_ratio_zscore_zero_for_flat_history():
    import math
    assert math.isnan(ratio_zscore(1.0, [1.0, 1.0])) or ratio_zscore(1.0, [1.0, 1.0]) == 0.0


def test_rank_candidates_orders_by_ratio_drop_desc():
    cands = [
        {"session": date(2026, 2, 2), "ratio": 1.30, "ratio_prev": 1.40},  # dropped 0.10
        {"session": date(2026, 2, 3), "ratio": 1.10, "ratio_prev": 1.12},  # dropped 0.02
    ]
    ranked = rank_candidates(cands, by="drop")
    assert ranked[0]["session"] == date(2026, 2, 2)
