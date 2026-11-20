"""Integration tests for BotOrchestrator wiring — verifies config → strategy → engine → monitor."""

import os
from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from zerodte.main import BotOrchestrator, _resolve_mode


class TestModeResolution:
    def test_cli_flag_wins(self):
        args = Namespace(live=True, sandbox=False, manual=False)
        config = {"bot": {"mode": "dry_run"}}
        assert _resolve_mode(args, config) == "live"

    def test_env_var_overrides_config(self):
        args = Namespace(live=False, sandbox=False, manual=False)
        config = {"bot": {"mode": "dry_run"}}
        with patch.dict(os.environ, {"BOT_MODE": "sandbox"}):
            assert _resolve_mode(args, config) == "sandbox"

    def test_config_used_when_no_env(self):
        args = Namespace(live=False, sandbox=False, manual=False)
        config = {"bot": {"mode": "dry_run"}}
        with patch.dict(os.environ, {}, clear=True):
            # Remove BOT_MODE if present
            os.environ.pop("BOT_MODE", None)
            assert _resolve_mode(args, config) == "dry_run"

    def test_default_dry_run(self):
        args = Namespace(live=False, sandbox=False, manual=False)
        config = {"bot": {}}
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BOT_MODE", None)
            assert _resolve_mode(args, config) == "dry_run"


class TestStrategyCreation:
    def test_creates_iron_condor(self):
        from zerodte.strategies.iron_condor import IronCondorStrategy
        config = {"bot": {"strategy": "iron_condor"}, "strategy": {}}
        bot = BotOrchestrator(config, mode="dry_run")
        s = bot._create_strategy("iron_condor")
        assert isinstance(s, IronCondorStrategy)

    def test_creates_iron_fly(self):
        from zerodte.strategies.iron_fly import IronFlyStrategy
        config = {"bot": {"strategy": "iron_fly"}, "options": {}}
        bot = BotOrchestrator(config, mode="dry_run")
        s = bot._create_strategy("iron_fly")
        assert isinstance(s, IronFlyStrategy)

    def test_unknown_strategy_is_rejected(self):
        """A typo in bot.strategy must not silently trade an iron condor."""
        import pytest
        config = {"bot": {"strategy": "unknown_thing"}, "strategy": {}}
        bot = BotOrchestrator(config, mode="dry_run")
        with pytest.raises(ValueError, match="unknown_thing"):
            bot._create_strategy("unknown_thing")


class TestConfigSchedule:
    def test_iron_fly_config_has_single_entry(self):
        """Iron fly config should schedule entry at 10:00 only."""
        import yaml
        with open("config/config_iron_fly_spx.yaml") as f:
            config = yaml.safe_load(f)
        schedule = config.get("entry", {}).get("schedule", [])
        assert schedule == ["10:00"]

    def test_iron_fly_config_has_strategy_key(self):
        import yaml
        with open("config/config_iron_fly_spx.yaml") as f:
            config = yaml.safe_load(f)
        assert config["bot"]["strategy"] == "iron_fly"

    def test_iron_fly_config_has_em_wing_fraction(self):
        import yaml
        with open("config/config_iron_fly_spx.yaml") as f:
            config = yaml.safe_load(f)
        assert config["options"]["em_wing_fraction"] == 0.50


class TestOrchestratorUsesCreateEngine:
    def test_orchestrator_uses_strategy_create_engine(self):
        """BotOrchestrator should use strategy.create_engine(deps) instead of hardcoded engine."""
        import inspect
        source = inspect.getsource(BotOrchestrator.initialize)
        assert "create_engine" in source
        assert "DecisionEngine(" not in source  # no hardcoded DecisionEngine construction

    def test_orchestrator_delegates_run_to_engine(self):
        """run_scheduled should delegate to engine.run() not implement its own loop."""
        import inspect
        source = inspect.getsource(BotOrchestrator.run_scheduled)
        assert "self._engine.run()" in source
        assert "current_time in schedule" not in source  # no inline scheduling loop


class TestEngineStrategyWiring:
    @pytest.mark.asyncio
    async def test_engine_receives_strategy(self):
        """DecisionEngine must have a strategy attribute after init."""
        from zerodte.agent.decision_engine import DecisionEngine
        from zerodte.strategies.iron_fly import IronFlyStrategy

        strategy = IronFlyStrategy()
        strategy.configure({"options": {}})

        engine = DecisionEngine(
            tool_handlers=AsyncMock(),
            risk_manager=AsyncMock(),
            strategy=strategy,
        )
        assert engine.strategy is strategy

    @pytest.mark.asyncio
    async def test_monitor_receives_strategy(self):
        """PositionMonitor must have a strategy attribute after init."""
        from zerodte.services.position_monitor import PositionMonitor
        from zerodte.strategies.iron_fly import IronFlyStrategy

        strategy = IronFlyStrategy()
        strategy.configure({"options": {}})

        monitor = PositionMonitor(
            db=AsyncMock(),
            cache=AsyncMock(),
            tool_handlers=AsyncMock(),
            strategy=strategy,
        )
        assert monitor.strategy is strategy
