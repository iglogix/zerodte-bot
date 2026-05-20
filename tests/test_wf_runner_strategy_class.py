"""Verify the WF-CV runner can resolve strategy class from config."""

import pytest

from src_v2.strategies.open_ic import OpenIcStrategy


def _resolve_strategy_class(dotted_path: str | None):
    """Mirror the runner's resolution logic for direct testing."""
    from scripts.run_v2_wf_cv_open_ic import resolve_strategy_class

    return resolve_strategy_class(dotted_path)


def test_default_resolves_to_open_ic():
    cls = _resolve_strategy_class(None)
    assert cls is OpenIcStrategy


def test_explicit_open_ic_path():
    cls = _resolve_strategy_class("src_v2.strategies.open_ic.OpenIcStrategy")
    assert cls is OpenIcStrategy


def test_unknown_module_raises():
    with pytest.raises((ImportError, ModuleNotFoundError)):
        _resolve_strategy_class("src_v2.strategies.no_such.NoSuchStrategy")


def test_unknown_attr_raises():
    with pytest.raises(AttributeError):
        _resolve_strategy_class("src_v2.strategies.open_ic.NoSuchClass")
