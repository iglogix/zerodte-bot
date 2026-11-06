# tests/dc_time_machine/test_strategy.py
from zerodte.backtest.dc_time_machine.strategy import DcParams


def test_defaults_match_video_favorite():
    p = DcParams()
    assert p.entry_minute == "09:40"
    assert p.target_delta == 0.35
    assert p.front_dte == 9
    assert p.gap == 1
    assert p.wing_width == 5.0
    assert p.transform_epsilon == 0.0
    assert p.stop_loss_frac == 0.20
    assert p.fill_k == 1.0
    assert p.entry_filter == "always"
    assert p.overnight_policy == "close"


def test_params_are_frozen_and_overridable():
    p = DcParams(target_delta=0.20, wing_width=10.0, entry_filter="fri_mon")
    assert (p.target_delta, p.wing_width, p.entry_filter) == (0.20, 10.0, "fri_mon")
