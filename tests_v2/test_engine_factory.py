"""Tests that strategies create the correct engine type."""
import pytest
from unittest.mock import MagicMock
from src_v2.agent.base_engine import BaseEngine, EngineDeps


def _make_deps():
    return EngineDeps(
        tool_handlers=MagicMock(),
        risk_manager=MagicMock(),
        anomaly_detector=MagicMock(),
        regime_classifier=MagicMock(),
        decision_file_logger=MagicMock(),
        db=MagicMock(),
        mode="dry_run",
        config={},
    )


def test_engine_deps_has_all_fields():
    deps = _make_deps()
    assert deps.tool_handlers is not None
    assert deps.mode == "dry_run"


def test_base_engine_is_abstract():
    with pytest.raises(TypeError):
        BaseEngine(_make_deps(), MagicMock())


def test_iron_condor_creates_scheduled_engine():
    from src_v2.strategies.iron_condor import IronCondorStrategy
    from src_v2.agent.scheduled_engine import ScheduledEngine

    s = IronCondorStrategy()
    s.configure({"strategy": {"min_short_delta": 0.10, "max_short_delta": 0.15,
                              "min_side_credit": 0.50, "max_side_credit": 2.50,
                              "wing_width": 30},
                 "stop_loss": {}, "risk": {}})
    engine = s.create_engine(_make_deps())
    assert isinstance(engine, ScheduledEngine)
    assert isinstance(engine, BaseEngine)


def test_iron_fly_creates_scheduled_engine():
    from src_v2.strategies.iron_fly import IronFlyStrategy
    from src_v2.agent.scheduled_engine import ScheduledEngine

    s = IronFlyStrategy()
    s.configure({"options": {"strike_step_pts": 5, "contract_multiplier": 100}})
    engine = s.create_engine(_make_deps())
    assert isinstance(engine, ScheduledEngine)


def test_orb_em_ic_creates_orb_engine():
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy
    from src_v2.agent.orb_engine import OrbEngine

    s = OrbEmIronCondorStrategy()
    s.configure({"options": {"strike_step_pts": 5, "contract_multiplier": 100}})
    engine = s.create_engine(_make_deps())
    assert isinstance(engine, OrbEngine)
    assert isinstance(engine, BaseEngine)


def test_decision_engine_backward_compat():
    """DecisionEngine import still works and is a wrapper around ScheduledEngine."""
    from src_v2.agent.decision_engine import DecisionEngine
    from src_v2.agent.scheduled_engine import ScheduledEngine

    engine = DecisionEngine(
        tool_handlers=MagicMock(),
        strategy=MagicMock(),
    )
    assert isinstance(engine._inner, ScheduledEngine)
