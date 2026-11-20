"""Strategy selection: every configured name resolves to its own class."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from zerodte.main import STRATEGY_REGISTRY, load_strategy_class

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@pytest.mark.parametrize("name, class_name", [
    ("iron_condor", "IronCondorStrategy"),
    ("iron_fly", "IronFlyStrategy"),
    ("orb_em_ic", "OrbEmIronCondorStrategy"),
    ("orb_em_directional", "OrbEmDirectionalStrategy"),
    ("meic", "MeicStrategy"),
    ("power_hour", "PowerHourStrategy"),
    ("reverse_ic", "ReverseIcStrategy"),
    ("open_ic", "OpenIcStrategy"),
    ("open_strangle", "OpenStrangleStrategy"),
])
def test_each_name_resolves_to_its_own_class(name, class_name):
    assert load_strategy_class(name).__name__ == class_name


def test_unknown_name_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError, match="no_such_strategy"):
        load_strategy_class("no_such_strategy")


def test_every_zerodte_config_names_a_registered_strategy():
    """A config must never silently fall through to a different strategy."""
    unknown = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        bot = cfg.get("bot")
        name = bot.get("strategy") if isinstance(bot, dict) else None
        if isinstance(name, str) and name not in STRATEGY_REGISTRY:
            unknown[path.name] = name
    assert not unknown, f"configs naming unregistered strategies: {unknown}"
