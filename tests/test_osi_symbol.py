from datetime import date
import pytest
from src_v2.backtest.osi_symbol import format_osi, build_strike_window


def test_format_osi_put_integer_strike():
    assert format_osi(date(2025, 9, 22), 6445.0, is_call=False) == "SPXW  250922P06445000"


def test_format_osi_call_integer_strike():
    assert format_osi(date(2024, 3, 20), 5235.0, is_call=True) == "SPXW  240320C05235000"


def test_format_osi_5pt_strike_step():
    assert format_osi(date(2026, 1, 15), 4750.0, is_call=False) == "SPXW  260115P04750000"


def test_format_osi_strike_with_half_point():
    assert format_osi(date(2025, 9, 22), 6442.5, is_call=True) == "SPXW  250922C06442500"


def test_build_strike_window_default():
    syms = build_strike_window(date(2025, 9, 22), atm_strike=6445.0, half_width_pts=50, step_pts=5)
    assert "SPXW  250922P06395000" in syms
    assert "SPXW  250922C06495000" in syms
    assert "SPXW  250922P06445000" in syms
    assert "SPXW  250922C06445000" in syms
    assert len(syms) == 21 * 2  # 21 strikes × call+put


def test_build_strike_window_strikes_are_step_aligned():
    syms = build_strike_window(date(2025, 9, 22), atm_strike=6443.0, half_width_pts=10, step_pts=5)
    parsed_strikes = sorted({int(s[-8:]) // 1000 for s in syms})
    assert parsed_strikes == [6435, 6440, 6445, 6450, 6455]
