"""
8AM ORB Break-and-Retest Bot — main entry point.

Usage:
  python -m es_8am.main                    # dry-run (default)
  python -m es_8am.main --manual           # live data, orders NOT submitted
  python -m es_8am.main --sandbox          # TastyTrade cert environment
  python -m es_8am.main --live             # TastyTrade production
  python -m es_8am.main --backtest         # run backtest and exit
  python -m es_8am.main --once             # run one setup+watch cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

VALID_MODES = {"dry_run", "manual", "sandbox", "live", "backtest"}


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


def _load_config(config_path: str = "config/config_8am.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


class OrbBotOrchestrator:
    """
    Top-level orchestrator for the 8AM ORB bot.

    Modes:
      dry_run  - no API calls, fully simulated
      manual   - live data + order generation, orders NOT submitted
      sandbox  - real orders to TastyTrade cert environment
      live     - real orders to TastyTrade production
      backtest - replay historical Databento data, no API calls
    """

    def __init__(self, config: Dict[str, Any], mode: str = "dry_run"):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: {VALID_MODES}")
        self.config = config
        self.mode = mode
        self._db = None
        self._market_data = None
        self._tt_client = None
        self._tool_handlers = None
        self._decision_engine = None
        self._llm_client = None

    async def initialize(self):
        """Initialize all services."""
        from agentic_ic.services.llm_client import LLMClient
        from es_8am.agent.anomaly_detector import OrbAnomalyDetector
        from es_8am.agent.decision_engine import OrbDecisionEngine
        from es_8am.agent.regime_classifier import OrbRegimeClassifier
        from es_8am.agent.tool_handlers import OrbToolHandlers
        from es_8am.services.db_service import OrbDatabaseService
        from es_8am.services.market_data import OrbMarketData

        db_cfg = self.config.get("database", {})
        llm_cfg = self.config.get("llm", {})
        self.config.get("logging", {})

        self._db = OrbDatabaseService(db_cfg.get("trading_db", "orb_bot.db"))
        self._db.initialize()

        advisor_llm = LLMClient(model=llm_cfg.get("advisor_model", "anthropic/claude-haiku-4-5"))

        if self.mode != "dry_run":
            from agentic_ic.services.tastytrade_client import TastyTradeClient
            self._tt_client = TastyTradeClient(mode=self.mode)
            await self._tt_client.connect()
        else:
            self._tt_client = _DryRunTTClient()

        self._market_data = OrbMarketData(tt_client=self._tt_client, config=self.config)

        self._tool_handlers = OrbToolHandlers(
            market_data=self._market_data,
            tt_client=self._tt_client,
            db_service=self._db,
            config=self.config,
            mode=self.mode,
        )

        regime_classifier = OrbRegimeClassifier(llm_client=advisor_llm)
        anomaly_detector = OrbAnomalyDetector(config=self.config)

        self._decision_engine = OrbDecisionEngine(
            tool_handlers=self._tool_handlers,
            db_service=self._db,
            market_data=self._market_data,
            regime_classifier=regime_classifier,
            anomaly_detector=anomaly_detector,
            config=self.config,
            mode=self.mode,
        )

        logger.info("ORB bot initialized — mode: %s", self.mode.upper())

    async def run_day(self):
        """
        Run a full trading day: wait for 08:15 ET, run setup, then entry watch loop.
        """
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        strategy = self.config.get("strategy", {})

        orb_end_str = strategy.get("orb_window_end", "08:15")
        orb_h, orb_m = [int(x) for x in orb_end_str.split(":")]
        orb_end = time(orb_h, orb_m)

        market_open_str = strategy.get("market_open", "09:30")
        mo_h, mo_m = [int(x) for x in market_open_str.split(":")]
        market_open_t = time(mo_h, mo_m)

        # Phase A: wait for 08:15 then fetch ORB levels
        while True:
            now_et = datetime.now(ET)
            if now_et.time() >= orb_end:
                break
            wait_secs = max(1, (
                datetime.combine(now_et.date(), orb_end, tzinfo=ET) - now_et
            ).seconds)
            logger.info("Waiting for ORB window to close (08:15 ET) — %ds remaining", wait_secs)
            await asyncio.sleep(min(wait_secs, 30))

        ok = await self._decision_engine.run_setup_phase()
        if not ok:
            logger.error("ORB setup failed — aborting day")
            return

        # Phase B: wait for 09:30 then start entry watch loop
        while True:
            now_et = datetime.now(ET)
            if now_et.time() >= market_open_t:
                break
            wait_secs = max(1, (
                datetime.combine(now_et.date(), market_open_t, tzinfo=ET) - now_et
            ).seconds)
            logger.info("Waiting for market open (09:30 ET) — %ds remaining", wait_secs)
            await asyncio.sleep(min(wait_secs, 30))

        await self._decision_engine.run_entry_watch_loop()

    async def run_backtest(self):
        """Run the backtest engine over the configured date range."""
        from datetime import date as dt_date

        from es_8am.backtest.engine import OrbBacktestEngine
        from es_8am.backtest.report import generate_report, print_report

        bt_cfg = self.config.get("backtest", {})
        zip_path = bt_cfg.get("data_file", "GLBX-20260227-7PWD5NKFSF.zip")

        start_str = bt_cfg.get("start_date")
        end_str = bt_cfg.get("end_date")
        start_date = dt_date.fromisoformat(start_str) if start_str else None
        end_date = dt_date.fromisoformat(end_str) if end_str else None

        logger.info(
            "Running backtest: %s → %s on %s",
            start_date or "all", end_date or "all", zip_path,
        )

        engine = OrbBacktestEngine(config=self.config, db_service=self._db)
        results = engine.run(zip_path=zip_path, start_date=start_date, end_date=end_date)

        report = generate_report(results, config=self.config)
        print_report(report)
        return report

    async def shutdown(self):
        if self._tt_client and not isinstance(self._tt_client, _DryRunTTClient):
            try:
                await self._tt_client.disconnect()
            except Exception:
                pass
        logger.info("ORB bot shutdown complete")


class _DryRunTTClient:
    """Stub TastyTrade client for dry-run mode — no real API calls."""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_candles(self, symbol, interval, from_time=None, count=10):
        return []

    async def get_future_quote(self, symbol):
        return {"mark": 5000.0, "bid": 4999.75, "ask": 5000.25}

    async def get_account_info(self):
        return {"net_liq": 100000.0, "buying_power": 50000.0, "cash": 100000.0}

    async def place_futures_bracket(self, **kwargs):
        import uuid
        return {
            "trade_id": f"DRY-{uuid.uuid4().hex[:8].upper()}",
            "status": "simulated",
            **kwargs,
        }

    async def close_futures_position(self, trade_id: str):
        return {"trade_id": trade_id, "status": "simulated_close"}


def _resolve_mode(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    if args.backtest:
        return "backtest"
    if args.live:
        return "live"
    if args.sandbox:
        return "sandbox"
    if args.manual:
        return "manual"
    env_mode = os.environ.get("ORB_BOT_MODE", os.environ.get("BOT_MODE", "")).strip().lower()
    if env_mode in VALID_MODES:
        return env_mode
    cfg_mode = config.get("bot", {}).get("mode", "dry_run")
    if cfg_mode in VALID_MODES:
        return cfg_mode
    return "dry_run"


async def main_async(args: argparse.Namespace):
    config_path = getattr(args, "config", "config/config_8am.yaml")
    config = _load_config(config_path)

    log_cfg = config.get("logging", {})
    _setup_logging(log_cfg.get("log_dir", "logs_8am"), log_cfg.get("level", "INFO"))

    mode = _resolve_mode(args, config)
    orchestrator = OrbBotOrchestrator(config=config, mode=mode)

    try:
        await orchestrator.initialize()

        if mode == "backtest":
            await orchestrator.run_backtest()
        else:
            await orchestrator.run_day()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await orchestrator.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="8AM ORB Break-and-Retest Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Mode priority: CLI flag > ORB_BOT_MODE env var > config_8am.yaml bot.mode

Examples:
  python -m es_8am.main                   # dry-run (default)
  python -m es_8am.main --manual          # live data, orders NOT submitted
  python -m es_8am.main --sandbox         # TastyTrade cert environment
  python -m es_8am.main --live            # TastyTrade production
  python -m es_8am.main --backtest        # run historical backtest""",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--manual", action="store_true",
                            help="Live data, full order generation, orders NOT submitted")
    mode_group.add_argument("--sandbox", action="store_true",
                            help="TastyTrade sandbox (cert) environment")
    mode_group.add_argument("--live", action="store_true",
                            help="TastyTrade production (real money)")
    mode_group.add_argument("--backtest", action="store_true",
                            help="Run historical backtest and print report")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--config", default="config/config_8am.yaml",
                        help="Path to config file")
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
