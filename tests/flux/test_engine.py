# tests/flux/test_engine.py
from datetime import date, datetime, timezone

import pytest

from zerodte.flux.engine import atm_iv, ratio_point
from zerodte.flux.models import ChainSnapshot, OptionRow


def _chain(expiry, rows):
    return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), tuple(rows))


def _rows():
    # calls at delta .60/.50/.35 ; puts at -.60/-.50/-.35
    return [
        OptionRow(6990, True, 1, 1, 0.205, 0.60), OptionRow(7000, True, 1, 1, 0.200, 0.50),
        OptionRow(7030, True, 1, 1, 0.230, 0.35),
        OptionRow(7010, False, 1, 1, 0.215, -0.60), OptionRow(7000, False, 1, 1, 0.210, -0.50),
        OptionRow(6970, False, 1, 1, 0.250, -0.35),
    ]


def test_atm_iv_at_50_delta_averages_call_and_put():
    c = _chain(date(2026, 6, 12), _rows())
    # nearest |delta|→0.50: call .200, put .210 → avg .205
    assert atm_iv(c, ref_delta=0.50) == pytest.approx(0.205)


def test_atm_iv_at_35_delta_picks_wing_strikes():
    c = _chain(date(2026, 6, 12), _rows())
    # nearest to 0.35: call .230, put .250 → avg .240
    assert atm_iv(c, ref_delta=0.35) == pytest.approx(0.240)


def test_atm_iv_none_when_empty():
    assert atm_iv(_chain(date(2026, 6, 12), []), 0.35) is None


def test_ratio_point():
    front = _chain(date(2026, 6, 12), _rows())
    back = _chain(date(2026, 6, 15), [OptionRow(7000, True, 1, 1, 0.18, 0.50),
                                      OptionRow(7000, False, 1, 1, 0.18, -0.50)])
    rp = ratio_point(front, back, ref_delta=0.50)
    assert rp.front_iv == pytest.approx(0.205) and rp.back_iv == pytest.approx(0.18)
    assert rp.ratio == pytest.approx(0.205 / 0.18)
    assert rp.front_expiry == date(2026, 6, 12) and rp.back_expiry == date(2026, 6, 15)
