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


class BotOrchestrator:
    """
    Top-level orchestrator that wires all components together and
    runs the scheduled entry loop.
    """

    def __init__(self, config: Dict[str, Any], dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self._prompt: Optional[str] = None
        self._cache = None
        self._db = None
        self._risk_manager = None
        self._agent = None
        self._notifier = None

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

        self._db = DatabaseService(db_cfg.get("trading_db", "0dte_bot.db"))
        self._db.initialize()

        self._cache = CachedMarketData(
            cache_db_path=db_cfg.get("cache_db", "data/market_cache.db"),
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

        api_client = TastyTradeClient(
            environment=os.environ.get("TASTYTRADE_ENVIRONMENT", "sandbox"),
        )
        self._cache.api_client = api_client

        tool_handlers = ToolHandlers(
            cache=self._cache,
            api_client=api_client,
            db=self._db,
            dry_run=self.dry_run,
        )

        self._risk_manager = RiskManager(
            db=self._db,
            cache=self._cache,
            max_concurrent_trades=risk_cfg.get("max_concurrent_trades", 10),
            max_daily_risk_percent=risk_cfg.get("max_daily_risk_percent", 2.0),
            max_buying_power_percent=risk_cfg.get("max_buying_power_percent", 50.0),
            take_profit_threshold=risk_cfg.get("take_profit_threshold", 0.05),
        )

        self._agent = TradingAgent(
            llm_client=primary_llm,
            tool_handlers=tool_handlers,
            validator=self._validator,
            db=self._db,
            dry_run=self.dry_run,
            max_iterations=llm_cfg.get("max_iterations", 10),
        )

        self._prompt = _load_prompt()
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        logger.info("Bot initialized — mode: %s", mode)

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
        data["dry_run"] = self.dry_run
        return data

    async def run_scheduled(self):
        """Run the bot on the configured entry schedule."""
        entry_cfg = self.config.get("entry", {})
        schedule = entry_cfg.get("schedule", ["09:45", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"])

        logger.info("Starting scheduled bot. Entry times: %s", schedule)

        while True:
            now = datetime.now()
            current_time = now.strftime("%H:%M")

            if current_time in schedule:
                await self.run_cycle()
                await asyncio.sleep(61)
            else:
                await asyncio.sleep(10)

    async def shutdown(self):
        if self._cache:
            await self._cache.close()
        logger.info("Bot shutdown complete")


async def main_async(args: argparse.Namespace):
    config = _load_config()
    _setup_logging(
        config.get("logging", {}).get("log_dir", "logs"),
        config.get("logging", {}).get("level", "INFO"),
    )

    dry_run = not args.live
    orchestrator = BotOrchestrator(config=config, dry_run=dry_run)

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
    parser = argparse.ArgumentParser(description="0DTE Iron Condor Trading Bot")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: dry-run)",
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
