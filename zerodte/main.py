"""
0DTE Iron Condor Trading Bot — main entry point.

Usage:
  python -m zerodte.main            # dry-run mode (default)
  python -m zerodte.main --live     # live trading mode
  python -m zerodte.main --once     # run one cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: str, level: str = "INFO", strategy: str = "bot"):
    log_path = Path(log_dir) / date.today().isoformat()
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{strategy}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    # Third-party libraries flood DEBUG (the tastytrade SDK logs every quote
    # tick — 95% of a 261KB session log on 2026-06-11). The SDK's __init__
    # sets its logger to DEBUG at import time and our client imports it
    # lazily, so import it first and pin afterwards.
    try:
        import tastytrade  # noqa: F401
    except ImportError:
        pass
    for noisy in ("tastytrade", "websockets", "httpx", "httpcore", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.INFO)


def _load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_prompt(prompt_path: str = "bot_prompt.md") -> str:
    with open(prompt_path) as f:
        return f.read()


VALID_MODES = {"dry_run", "manual", "sandbox", "live", "alpaca_paper"}


# Strategy name (config key ``bot.strategy``) -> "module:Class". Modules are
# imported on demand so that starting one strategy does not import the others.
STRATEGY_REGISTRY = {
    "iron_condor": "zerodte.strategies.iron_condor:IronCondorStrategy",
    "iron_fly": "zerodte.strategies.iron_fly:IronFlyStrategy",
    "orb_em_ic": "zerodte.strategies.orb_em_ic:OrbEmIronCondorStrategy",
    "orb_em_directional": "zerodte.strategies.orb_em_directional:OrbEmDirectionalStrategy",
    "meic": "zerodte.strategies.meic:MeicStrategy",
    "power_hour": "zerodte.strategies.power_hour:PowerHourStrategy",
    "reverse_ic": "zerodte.strategies.reverse_ic:ReverseIcStrategy",
    "open_ic": "zerodte.strategies.open_ic:OpenIcStrategy",
    "open_strangle": "zerodte.strategies.open_strangle:OpenStrangleStrategy",
}


def load_strategy_class(strategy_name: str):
    """Resolve a configured strategy name to its class.

    An unknown name is an error: silently falling back to another strategy
    would trade something the operator did not configure.
    """
    try:
        module_name, class_name = STRATEGY_REGISTRY[strategy_name].split(":")
    except KeyError:
        known = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy {strategy_name!r}; expected one of: {known}") from None
    return getattr(importlib.import_module(module_name), class_name)


class BotOrchestrator:
    """
    Top-level orchestrator that wires all components together and
    runs the scheduled entry loop.

    mode:
      dry_run      - no API calls at all, fully simulated responses
      manual       - live market data + full order generation, orders NOT submitted
      sandbox      - real orders to TastyTrade cert environment
      live         - real orders to TastyTrade production
      alpaca_paper - hybrid: live TastyTrade market data, orders submitted to
                     the Alpaca PAPER account. Exists because TT cert cannot
                     trade SPXW 0DTE, so `manual` can only assume fills.
                     Alpaca has no futures, so the ES opening-range signal
                     still comes from TastyTrade/DXLink.
    """

    def __init__(self, config: Dict[str, Any], mode: str = "dry_run"):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: {VALID_MODES}")

        # Refuse to run an Alpaca-execution config against a different venue.
        # BOT_MODE (loaded from .env) outranks the config file, so a stale
        # BOT_MODE=sandbox would otherwise silently send these orders to the
        # TastyTrade cert environment instead of Alpaca.
        if config.get("broker", {}).get("execution") == "alpaca" and mode != "alpaca_paper":
            raise ValueError(
                f"config sets broker.execution=alpaca but the resolved mode is "
                f"{mode!r}. Orders would go to the wrong venue. Pass --alpaca "
                f"(or unset BOT_MODE, which overrides bot.mode in the config)."
            )
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
        self._quote_comparator = None
        self._fill_queue = None
        self._api_client = None
        self._eod_debrief = None
        self._session_skipped: bool = False
        self._session_skip_date: Optional[date] = None

    async def initialize(self):
        """Initialize all services."""
        from trading_core.agent.anomaly_detector import AnomalyDetector
        from trading_core.agent.regime_classifier import RegimeClassifier
        from trading_core.services.db_service import DatabaseService
        from trading_core.services.decision_logger import DecisionFileLogger
        from trading_core.services.market_data import CachedMarketData
        from trading_core.services.notifications import NotificationService
        from zerodte.agent.base_engine import EngineDeps
        from zerodte.agent.eod_debrief import EodDebrief
        from zerodte.agent.tool_handlers import ToolHandlers
        from zerodte.services.llm_client import LLMClient
        from zerodte.services.risk_manager import RiskManager

        db_cfg = self.config.get("database", {})
        llm_cfg = self.config.get("llm", {})
        risk_cfg = self.config.get("risk", {})
        notif_cfg = self.config.get("notifications", {})
        self.config.get("stop_loss", {})
        cache_cfg = self.config.get("cache", {})
        monitor_cfg = self.config.get("monitor", {})
        strategy_cfg = self.config.get("strategy", {})
        self.config.get("entry", {})
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

        from trading_core.services.order_fill_listener import OrderFillListener
        from zerodte.services.position_monitor import PositionMonitor
        from zerodte.services.tastytrade_client import TastyTradeClient

        # alpaca_paper is a HYBRID: TastyTrade supplies market data (Alpaca has
        # no futures, so the ES opening-range signal cannot come from there),
        # and Alpaca receives the orders. The TT client is built in `manual` —
        # live production market data — and is never asked to write, because
        # AlpacaExecutionClient overrides every write method.
        if self.mode == "alpaca_paper":
            from zerodte.services.alpaca_client import AlpacaExecutionClient

            broker_cfg = self.config.get("broker", {})
            entry_fill = dict(broker_cfg.get("entry_fill", {}))
            # Rung 1 must reproduce the backtest's assumed fill, so the ladder
            # is anchored to the same slippage the strategy already subtracts.
            entry_fill.setdefault(
                "base_slippage",
                self.config.get("options", {}).get("slippage_per_spread", 0.10),
            )
            data_client = TastyTradeClient(mode="manual")
            api_client = AlpacaExecutionClient(
                data_client=data_client,
                base_url=os.environ.get("APCA_API_BASE_URL", ""),
                fill_timeout_seconds=broker_cfg.get("fill_timeout_seconds", 30.0),
                entry_fill=entry_fill,
            )
        else:
            api_client = TastyTradeClient(
                mode=self.mode,
            )
        await api_client.connect()
        self._cache.api_client = api_client

        from zerodte.services.execution_log import ExecutionLog
        from zerodte.services.quote_comparator import QuoteComparator
        # Namespaced per strategy: MEIC and ORB both default to log_dir "logs",
        # so a shared file interleaves their records and each strategy's report
        # then reads the other's rows.
        self._execution_log = ExecutionLog(
            base_dir=log_cfg.get("log_dir", "logs"),
            strategy=self.config.get("bot", {}).get("strategy", ""),
        )
        self._quote_comparator = QuoteComparator(
            execution_log=self._execution_log,
            poll_interval_seconds=monitor_cfg.get("poll_interval_seconds", 15),
        )

        self._tool_handlers = ToolHandlers(
            cache=self._cache,
            api_client=api_client,
            db=self._db,
            mode=self.mode,
            execution_log=self._execution_log,
            on_entry=self._quote_comparator.register,
            es_roll_days_before_expiry=strategy_cfg.get("es_roll_days_before_expiry", 2),
        )
        # Comparator reads the same chain source the strategy uses.
        self._quote_comparator._th = self._tool_handlers

        self._risk_manager = RiskManager(
            db=self._db,
            cache=self._cache,
            max_concurrent_trades=risk_cfg.get("max_concurrent_trades", 10),
            max_daily_risk_percent=risk_cfg.get("max_daily_risk_percent", 2.0),
            max_buying_power_percent=risk_cfg.get("max_buying_power_percent", 50.0),
        )

        # Strategy selection from config
        strategy_name = self.config.get("bot", {}).get("strategy", "iron_condor")
        self._strategy = self._create_strategy(strategy_name)

        self._api_client = api_client
        self._fill_queue = asyncio.Queue()

        async def _log_fill(ev):
            self._execution_log.log_fill(
                ev.symbol, ev.fill_price, ev.order_type, ev.is_stop_fill, ev.quantity)

        self._fill_listener = OrderFillListener(
            api_client=api_client,
            fill_queue=self._fill_queue,
            mode=self.mode,
            on_fill=_log_fill,
        )

        eod_str = monitor_cfg.get("eod_close_time", "15:55")
        eod_h, eod_m = (int(x) for x in eod_str.split(":"))

        decision_file_logger = DecisionFileLogger(
            base_dir=str(Path(log_cfg.get("log_dir", "logs")) / "decisions")
        )

        # LLM advisor gates (anomaly + regime) are OPTIONAL. They are NOT part of
        # any backtest/WF-validated config (the backtest engine enters every valid
        # slot), so paper runs that must stay comparable to the backtest disable
        # them via `llm.advisors_enabled: false`. Deterministic guards (Markov
        # gate, dislocation, stability) are unaffected. Default true = legacy.
        if llm_cfg.get("advisors_enabled", True):
            anomaly_detector = AnomalyDetector(
                llm_client=advisor_llm,
                tool_handlers=self._tool_handlers,
            )
            regime_classifier = RegimeClassifier(
                llm_client=advisor_llm,
                tool_handlers=self._tool_handlers,
            )
        else:
            anomaly_detector = None
            regime_classifier = None
            logger.info("LLM advisor gates DISABLED (llm.advisors_enabled=false) "
                        "— entering on deterministic guards only")

        deps = EngineDeps(
            tool_handlers=self._tool_handlers,
            risk_manager=self._risk_manager,
            anomaly_detector=anomaly_detector,
            regime_classifier=regime_classifier,
            decision_file_logger=decision_file_logger,
            db=self._db,
            mode=self.mode,
            config=self.config,
        )
        self._engine = self._strategy.create_engine(deps)

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
            strategy=self._strategy,
            notifier=self._notifier,
            fill_queue=self._fill_queue,
            poll_interval_seconds=monitor_cfg.get("poll_interval_seconds", 30),
            eod_close_time=time(eod_h, eod_m),
            mode=self.mode,
            eod_debrief=self._eod_debrief,
        )

        logger.info(
            "Bot initialized — mode: %s, strategy: %s",
            self.mode.upper(), strategy_name,
        )

    def _create_strategy(self, strategy_name: str):
        """Instantiate and configure the trading strategy from config."""
        strategy = load_strategy_class(strategy_name)()
        strategy.configure(self.config)
        logger.info("Strategy created: %s", strategy_name)
        return strategy

    async def run_cycle(self):
        """Run one decision cycle. Returns CycleResult."""
        logger.info("--- Running decision cycle at %s ---", datetime.now().strftime("%H:%M:%S"))

        # ScheduledEngine has run_cycle(); OrbEngine only has run().
        if hasattr(self._engine, 'run_cycle'):
            result = await self._engine.run_cycle()
        else:
            result = await self._engine.run()

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
        """Run entry engine, position monitor, fill listener, and quote comparator concurrently."""
        monitor_task = asyncio.create_task(self._monitor.start())
        listener_task = asyncio.create_task(self._fill_listener.start())
        comparator_task = asyncio.create_task(self._quote_comparator.start())

        try:
            result = await self._engine.run()
            # The engine's CycleResult is the ONLY record of why a session
            # ended (no_candidates, validation_failed, eod_no_entry, ...).
            # Discarding it made 2026-08-21's silent 09:27 exit undiagnosable.
            logger.info(
                "Engine session complete: outcome=%s reason=%s",
                getattr(result, "outcome", "?"), getattr(result, "reason", ""),
            )
        finally:
            await self._fill_listener.stop()
            await self._monitor.stop()
            await self._quote_comparator.stop()
            for task in (listener_task, monitor_task, comparator_task):
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
        logger.info("Mode: live (from CLI --live)")
        return "live"
    if args.sandbox:
        logger.info("Mode: sandbox (from CLI --sandbox)")
        return "sandbox"
    if args.manual:
        logger.info("Mode: manual (from CLI --manual)")
        return "manual"
    if getattr(args, "alpaca", False):
        logger.info("Mode: alpaca_paper (from CLI --alpaca)")
        return "alpaca_paper"
    env_mode = os.environ.get("BOT_MODE", "").strip().lower()
    if env_mode in VALID_MODES:
        logger.info("Mode: %s (from BOT_MODE env var — overrides config)", env_mode)
        return env_mode
    cfg_mode = config.get("bot", {}).get("mode", "dry_run")
    if cfg_mode in VALID_MODES:
        logger.info("Mode: %s (from config)", cfg_mode)
        return cfg_mode
    return "dry_run"


async def main_async(args: argparse.Namespace):
    config = _load_config(args.config)
    strategy_name = config.get("bot", {}).get("strategy", "bot")
    _setup_logging(
        config.get("logging", {}).get("log_dir", "logs"),
        config.get("logging", {}).get("level", "INFO"),
        strategy=strategy_name,
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
  python -m zerodte.main                  # dry-run (default)
  python -m zerodte.main --manual         # live data, orders generated but NOT submitted
  python -m zerodte.main --sandbox        # TastyTrade cert environment
  python -m zerodte.main --live           # TastyTrade production
  python -m zerodte.main --alpaca         # TT market data, orders to Alpaca paper
  python -m zerodte.main --manual --once  # one manual cycle""",
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
    mode_group.add_argument(
        "--alpaca",
        action="store_true",
        help="Hybrid paper: TastyTrade market data, orders to the Alpaca PAPER account",
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
