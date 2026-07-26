"""Skyline structure must reproduce the video's stated payoff economics.

Video anchor (SPX 7580 ATM, 1 lot): net debit 1255, max loss 755, max profit 1245.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from skyline_core import LEGS, N_CONTRACTS, payoff_at

ATM = 7580.0
VIDEO_DEBIT = 1255.0


def _pnl(s):
    return payoff_at(s, ATM) - VIDEO_DEBIT


def test_twelve_distinct_strikes_sixteen_contracts():
    assert len({(off, c) for off, _, c in LEGS}) == 12
    assert len({off for off, _, c in LEGS}) == 12
    assert N_CONTRACTS == 16


def test_video_max_loss_at_atm_valley():
    # 7580 is the "deepest valley" the trade is centred on
    assert _pnl(7580.0) == -755.0


def test_video_max_profit_at_peaks():
    for peak in (7520.0, 7560.0, 7605.0):
        assert _pnl(peak) == 1245.0
    # rightmost fly is 20-wide -> lower peak
    assert _pnl(7640.0) == 745.0


def test_outer_floors_cap_loss_at_max_loss():
    # broken wings leave +5pt residual on each tail
    assert _pnl(7000.0) == -755.0
    assert _pnl(8000.0) == -755.0


def test_side_valleys_are_shallow():
    assert _pnl(7540.0) == -255.0
    assert _pnl(7625.0) == -255.0


def test_max_loss_is_global_minimum():
    lo = min(_pnl(s) for s in [x / 2 for x in range(2 * 7000, 2 * 8000)])
    assert lo == -755.0
