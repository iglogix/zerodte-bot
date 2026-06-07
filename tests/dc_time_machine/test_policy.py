# tests/dc_time_machine/test_policy.py
import pytest

from src_v2.backtest.dc_time_machine.policy import entry_allowed, overnight_decision


def test_always_filter_admits_everything():
    assert entry_allowed("always", {}) is True


def test_backwardation_filter_needs_ratio_above_one():
    assert entry_allowed("backwardation", {"ratio": 1.05}) is True
    assert entry_allowed("backwardation", {"ratio": 0.98}) is False


def test_fri_mon_filter_keys_on_flag():
    assert entry_allowed("fri_mon", {"is_fri_mon": True}) is True
    assert entry_allowed("fri_mon", {"is_fri_mon": False}) is False


def test_spike_filter_needs_elevated_and_reverting():
    assert entry_allowed("spike", {"ratio_z": 1.5, "ratio_falling": True}) is True
    assert entry_allowed("spike", {"ratio_z": 1.5, "ratio_falling": False}) is False
    assert entry_allowed("spike", {"ratio_z": 0.2, "ratio_falling": True}) is False


def test_overnight_close_policy_always_closes():
    assert overnight_decision("close", {"dc_pnl_frac": 0.05, "ratio": 1.2}) == "close"


def test_overnight_hold_policy_holds_only_when_shallow_loss_and_backwardated():
    assert overnight_decision("hold", {"dc_pnl_frac": -0.05, "ratio": 1.1}) == "hold"
    assert overnight_decision("hold", {"dc_pnl_frac": -0.15, "ratio": 1.1}) == "close"
    assert overnight_decision("hold", {"dc_pnl_frac": -0.05, "ratio": 0.9}) == "close"


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        entry_allowed("nope", {})
    with pytest.raises(ValueError):
        overnight_decision("nope", {})
