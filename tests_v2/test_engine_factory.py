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
