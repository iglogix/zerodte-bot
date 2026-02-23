"""
0DTE Iron Condor Trading Bot — main entry point.

Usage:
  python -m src.main            # dry-run mode (default)
  python -m src.main --live     # live trading mode
  python -m src.main --once     # run one cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, date, time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: str, level: str = "INFO"):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{date.today().isoformat()}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def _load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_prompt(prompt_path: str = "bot_prompt.md") -> str:
    with open(prompt_path) as f:
        return f.read()


VALID_MODES = {"dry_run", "manual", "sandbox", "live"}


class BotOrchestrator:
    """
    Top-level orchestrator that wires all components together and
    runs the scheduled entry loop.

    mode:
      dry_run - no API calls at all, fully simulated responses
      manual  - live market data + full order generation, orders NOT submitted
      sandbox - real orders to TastyTrade cert environment
      live    - real orders to TastyTrade production
    """

    def __init__(self, config: Dict[str, Any], mode: str = "dry_run"):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: {VALID_MODES}")
        self.config = config
        self.mode = mode
        self.dry_run = mode == "dry_run"
        self._prompt: Optional[str] = None
        self._cache = None
        self._db = None
        self._risk_manager = None
        self._agent = None
        self._notifier = None
        self._monitor = None
        self._tool_handlers = None

    async def initialize(self):
        """Initialize all services."""
        from src.services.market_data import CachedMarketData
        from src.services.db_service import DatabaseService
        from src.services.risk_manager import RiskManager
        from src.services.notifications import NotificationService
        from src.services.llm_client import LLMClient
        from src.agent.tool_handlers import ToolHandlers
        from src.agent.dual_validator import DualLLMValidator
        from src.agent.trading_agent import TradingAgent

        db_cfg = self.config.get("database", {})
        llm_cfg = self.config.get("llm", {})
        risk_cfg = self.config.get("risk", {})
        notif_cfg = self.config.get("notifications", {})
        stop_cfg = self.config.get("stop_loss", {})
        cache_cfg = self.config.get("cache", {})
        monitor_cfg = self.config.get("monitor", {})

        self._db = DatabaseService(db_cfg.get("trading_db", "0dte_bot.db"))
        self._db.initialize()

        self._cache = CachedMarketData(
            cache_db_path=db_cfg.get("cache_db", "data/market_cache.db"),
            option_chain_ttl=cache_cfg.get("option_chain_ttl", 60),
            quote_ttl=cache_cfg.get("quote_ttl", 60),
            account_info_ttl=cache_cfg.get("account_info_ttl", 60),
        )
        await self._cache.initialize()

        self._notifier = NotificationService(
            email_config=notif_cfg.get("email", {}),
            webhook_config=notif_cfg.get("webhook", {}),
        )

        primary_llm = LLMClient(model=llm_cfg.get("primary_model", "claude-sonnet-4.6"))
        validator_llm = LLMClient(model=llm_cfg.get("validator_model", "gpt-5.2"))

        self._validator = DualLLMValidator(
            primary_llm=primary_llm,
            secondary_llm=validator_llm,
            notifier=self._notifier,
            decision_logger=self._db,
            timeout_seconds=llm_cfg.get("timeout_seconds", 30),
        )

        from src.services.tastytrade_client import TastyTradeClient
        from src.services.position_monitor import PositionMonitor

        api_client = TastyTradeClient(
            mode=self.mode,
        )
        self._cache.api_client = api_client

        self._tool_handlers = ToolHandlers(
            cache=self._cache,
            api_client=api_client,
            db=self._db,
            mode=self.mode,
            stop_limit_gap=stop_cfg.get("stop_limit_gap", 40),
            stop_market_gap=stop_cfg.get("stop_market_gap", 30),
        )

        self._risk_manager = RiskManager(
            db=self._db,
            cache=self._cache,
            max_concurrent_trades=risk_cfg.get("max_concurrent_trades", 10),
            max_daily_risk_percent=risk_cfg.get("max_daily_risk_percent", 2.0),
            max_buying_power_percent=risk_cfg.get("max_buying_power_percent", 50.0),
            take_profit_threshold=risk_cfg.get("take_profit_threshold", 0.05),
        )

        eod_str = monitor_cfg.get("eod_close_time", "15:55")
        eod_h, eod_m = (int(x) for x in eod_str.split(":"))
        self._monitor = PositionMonitor(
            db=self._db,
            cache=self._cache,
            tool_handlers=self._tool_handlers,
            risk_manager=self._risk_manager,
            notifier=self._notifier,
            poll_interval_seconds=monitor_cfg.get("poll_interval_seconds", 30),
            take_profit_threshold=risk_cfg.get("take_profit_threshold", 0.05),
            eod_close_time=time(eod_h, eod_m),
            mode=self.mode,
        )

        self._agent = TradingAgent(
            llm_client=primary_llm,
            tool_handlers=self._tool_handlers,
            validator=self._validator,
            db=self._db,
            dry_run=self.dry_run,
            max_iterations=llm_cfg.get("max_iterations", 10),
        )

        self._prompt = _load_prompt()
        logger.info("Bot initialized — mode: %s", self.mode.upper())

    async def run_cycle(self):
        """Run one decision cycle."""
        logger.info("--- Running decision cycle at %s ---", datetime.now().strftime("%H:%M:%S"))

        market_data = await self._gather_market_data()

        result = await self._agent.run_tool_loop(
            prompt=self._prompt,
            market_data=market_data,
        )

        logger.info("Cycle complete. Response: %s", result.get("response", "")[:200])
        return result

    async def _gather_market_data(self) -> Dict[str, Any]:
        """Gather initial market snapshot to pass to the agent."""
        data: Dict[str, Any] = {}

        try:
            quote = await self._cache.get_cached_quote("SPX", max_age_seconds=60)
            if quote:
                data["spx_price"] = quote.get("price")
        except Exception:
            logger.warning("Could not fetch SPX price for market snapshot")

        try:
            candles = await self._cache.get_5min_candles("SPX", count=6)
            data["recent_candles"] = candles
        except Exception:
            logger.warning("Could not fetch candles for market snapshot")

        try:
            open_trades = await self._db.get_open_trades()
            data["open_positions"] = len(open_trades)
        except Exception:
            pass

        data["timestamp"] = datetime.now().isoformat()
        data["mode"] = self.mode
        return data

    async def run_scheduled(self):
        """Run the entry scheduler and position monitor concurrently."""
        entry_cfg = self.config.get("entry", {})
        schedule = entry_cfg.get("schedule", ["09:45", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"])

        logger.info("Starting scheduled bot. Entry times: %s", schedule)

        monitor_task = asyncio.create_task(self._monitor.start())

        try:
            while True:
                now = datetime.now()
                current_time = now.strftime("%H:%M")

                if current_time in schedule:
                    await self.run_cycle()
                    await asyncio.sleep(61)
                else:
                    await asyncio.sleep(10)
        finally:
            await self._monitor.stop()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def shutdown(self):
        if self._monitor:
            await self._monitor.stop()
        if self._cache:
            await self._cache.close()
        logger.info("Bot shutdown complete")


def _resolve_mode(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """
    Resolve bot mode with priority: CLI flag > BOT_MODE env var > config.yaml.
    CLI flags --sandbox and --live are mutually exclusive.
    """
    if args.live:
        return "live"
    if args.sandbox:
        return "sandbox"
    if args.manual:
        return "manual"
    env_mode = os.environ.get("BOT_MODE", "").strip().lower()
    if env_mode in VALID_MODES:
        return env_mode
    cfg_mode = config.get("bot", {}).get("mode", "dry_run")
    if cfg_mode in VALID_MODES:
        return cfg_mode
    return "dry_run"


async def main_async(args: argparse.Namespace):
    config = _load_config()
    _setup_logging(
        config.get("logging", {}).get("log_dir", "logs"),
        config.get("logging", {}).get("level", "INFO"),
    )

    mode = _resolve_mode(args, config)
    orchestrator = BotOrchestrator(config=config, mode=mode)

    try:
        await orchestrator.initialize()

        if args.once:
            await orchestrator.run_cycle()
        else:
            await orchestrator.run_scheduled()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await orchestrator.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="0DTE Iron Condor Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Mode priority: CLI flag > BOT_MODE env var > config.yaml bot.mode

Examples:
  python -m src.main                  # dry-run (default)
  python -m src.main --manual         # live data, orders generated but NOT submitted
  python -m src.main --sandbox        # TastyTrade cert environment
  python -m src.main --live           # TastyTrade production
  python -m src.main --manual --once  # one manual cycle""",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--manual",
        action="store_true",
        help="Live market data, full order generation, but orders NOT submitted",
    )
    mode_group.add_argument(
        "--sandbox",
        action="store_true",
        help="Use TastyTrade sandbox (cert) environment",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Use TastyTrade production environment (real money)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
