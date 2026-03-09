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
        self._cache = None
        self._db = None
        self._risk_manager = None
        self._engine = None
        self._notifier = None
        self._monitor = None
        self._tool_handlers = None
        self._fill_listener = None
        self._fill_queue = None
        self._api_client = None
        self._eod_debrief = None
        self._session_skipped: bool = False
        self._session_skip_date: Optional[date] = None

    async def initialize(self):
        """Initialize all services."""
        from src.services.market_data import CachedMarketData
        from src.services.db_service import DatabaseService
        from src.services.risk_manager import RiskManager
        from src.services.notifications import NotificationService
        from src.services.llm_client import LLMClient
        from src.services.decision_logger import DecisionFileLogger
        from src.agent.tool_handlers import ToolHandlers
        from src.agent.decision_engine import DecisionEngine
        from src.agent.anomaly_detector import AnomalyDetector
        from src.agent.regime_classifier import RegimeClassifier
        from src.agent.eod_debrief import EodDebrief

        db_cfg = self.config.get("database", {})
        llm_cfg = self.config.get("llm", {})
        risk_cfg = self.config.get("risk", {})
        notif_cfg = self.config.get("notifications", {})
        stop_cfg = self.config.get("stop_loss", {})
        cache_cfg = self.config.get("cache", {})
        monitor_cfg = self.config.get("monitor", {})
        strategy_cfg = self.config.get("strategy", {})
        entry_cfg = self.config.get("entry", {})
        log_cfg = self.config.get("logging", {})

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

        advisor_llm = LLMClient(model=llm_cfg.get("advisor_model", "anthropic/claude-haiku-4-5"))
        debrief_llm = LLMClient(model=llm_cfg.get("debrief_model", "anthropic/claude-sonnet-4-5"))

        from src.services.tastytrade_client import TastyTradeClient
        from src.services.position_monitor import PositionMonitor
        from src.services.order_fill_listener import OrderFillListener

        api_client = TastyTradeClient(
            mode=self.mode,
        )
        await api_client.connect()
        self._cache.api_client = api_client

        self._tool_handlers = ToolHandlers(
            cache=self._cache,
            api_client=api_client,
            db=self._db,
            mode=self.mode,
            stop_limit_gap=stop_cfg.get("stop_limit_gap", 40),
            stop_market_gap=stop_cfg.get("stop_market_gap", 30),
            credit_equality_threshold=strategy_cfg.get("credit_equality_threshold", 0.20),
            skew_threshold=strategy_cfg.get("skew_threshold", 1.15),
            min_short_delta=strategy_cfg.get("min_short_delta", 0.10),
            max_short_delta=strategy_cfg.get("max_short_delta", 0.15),
            min_side_credit=strategy_cfg.get("min_side_credit", 0.50),
            max_side_credit=strategy_cfg.get("max_side_credit", 2.50),
        )

        self._risk_manager = RiskManager(
            db=self._db,
            cache=self._cache,
            max_concurrent_trades=risk_cfg.get("max_concurrent_trades", 10),
            max_daily_risk_percent=risk_cfg.get("max_daily_risk_percent", 2.0),
            max_buying_power_percent=risk_cfg.get("max_buying_power_percent", 50.0),
            take_profit_threshold=risk_cfg.get("take_profit_threshold", 0.05),
        )

        self._api_client = api_client
        self._fill_queue = asyncio.Queue()

        self._fill_listener = OrderFillListener(
            api_client=api_client,
            fill_queue=self._fill_queue,
            mode=self.mode,
        )

        eod_str = monitor_cfg.get("eod_close_time", "15:55")
        eod_h, eod_m = (int(x) for x in eod_str.split(":"))

        decision_file_logger = DecisionFileLogger(
            base_dir=str(Path(log_cfg.get("log_dir", "logs")) / "decisions")
        )

        anomaly_detector = AnomalyDetector(
            llm_client=advisor_llm,
            tool_handlers=self._tool_handlers,
        )
        regime_classifier = RegimeClassifier(
            llm_client=advisor_llm,
            tool_handlers=self._tool_handlers,
        )

        self._engine = DecisionEngine(
            tool_handlers=self._tool_handlers,
            risk_manager=self._risk_manager,
            anomaly_detector=anomaly_detector,
            regime_classifier=regime_classifier,
            decision_file_logger=decision_file_logger,
            mode=self.mode,
            stability_fraction=entry_cfg.get("stability_fraction", 0.15),
            single_candle_fraction=entry_cfg.get("single_candle_fraction", 0.10),
            fallback_stability_pts=entry_cfg.get("fallback_stability_pts", 8.0),
        )

        self._eod_debrief = EodDebrief(
            llm_client=debrief_llm,
            db=self._db,
            notifier=self._notifier,
            decisions_dir=str(Path(log_cfg.get("log_dir", "logs")) / "decisions"),
            summaries_dir=str(Path(log_cfg.get("log_dir", "logs")) / "summaries"),
        )

        self._monitor = PositionMonitor(
            db=self._db,
            cache=self._cache,
            tool_handlers=self._tool_handlers,
            risk_manager=self._risk_manager,
            notifier=self._notifier,
            fill_queue=self._fill_queue,
            poll_interval_seconds=monitor_cfg.get("poll_interval_seconds", 30),
            take_profit_threshold=risk_cfg.get("take_profit_threshold", 0.05),
            eod_close_time=time(eod_h, eod_m),
            mode=self.mode,
            eod_debrief=self._eod_debrief,
        )

        logger.info("Bot initialized — mode: %s", self.mode.upper())

    async def run_cycle(self):
        """Run one decision cycle. Returns CycleResult."""
        logger.info("--- Running decision cycle at %s ---", datetime.now().strftime("%H:%M:%S"))
        result = await self._engine.run_cycle()
        logger.info(
            "Cycle complete: outcome=%s reason=%s",
            result.outcome, result.reason,
        )

        if result.outcome == "skip_session":
            self._session_skipped = True
            self._session_skip_date = date.today()
            logger.warning(
                "Session skip triggered — all remaining entry windows suppressed for today: %s",
                result.reason,
            )

        return result

    async def run_scheduled(self):
        """Run entry scheduler, position monitor, and fill listener concurrently."""
        entry_cfg = self.config.get("entry", {})
        schedule = entry_cfg.get("schedule", ["09:45", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"])

        logger.info("Starting scheduled bot. Entry times: %s", schedule)

        monitor_task = asyncio.create_task(self._monitor.start())
        listener_task = asyncio.create_task(self._fill_listener.start())

        try:
            while True:
                now = datetime.now()
                current_time = now.strftime("%H:%M")

                # Reset session skip flag at the start of a new day
                if self._session_skip_date and self._session_skip_date < date.today():
                    self._session_skipped = False
                    self._session_skip_date = None

                if current_time in schedule:
                    if self._session_skipped:
                        logger.info(
                            "[SCHEDULER] Skipping %s — session skip active (reason set earlier today)",
                            current_time,
                        )
                    else:
                        await self.run_cycle()
                    await asyncio.sleep(61)
                else:
                    await asyncio.sleep(10)
        finally:
            await self._fill_listener.stop()
            await self._monitor.stop()
            for task in (listener_task, monitor_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def shutdown(self):
        if self._fill_listener:
            await self._fill_listener.stop()
        if self._monitor:
            await self._monitor.stop()
        if self._api_client:
            await self._api_client.disconnect()
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

    loop = asyncio.new_event_loop()

    def _suppress_ssl_shutdown(loop, context):
        msg = context.get("message", "")
        if "SSL" in msg or "Bad file descriptor" in msg or "socket.send" in msg:
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_suppress_ssl_shutdown)
    try:
        loop.run_until_complete(main_async(args))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
