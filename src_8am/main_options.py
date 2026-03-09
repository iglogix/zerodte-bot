"""
8AM ORB Options Bot — main entry point.

Sells SPX 0DTE credit spreads triggered by the ORB midline retest.

  LONG  bias → sell PUT  credit spread (floor below ORB low)
  SHORT bias → sell CALL credit spread (ceiling above ORB high)

Usage:
  python -m src_8am.main_options                   # dry-run (default)
  python -m src_8am.main_options --manual          # live data, orders NOT submitted
  python -m src_8am.main_options --sandbox         # TastyTrade cert environment
  python -m src_8am.main_options --live            # TastyTrade production
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

VALID_MODES = {"dry_run", "manual", "sandbox", "live"}


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


def _load_config(config_path: str = "config/config_8am_options.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


class OrbOptionsBotOrchestrator:
    """
    Top-level orchestrator for the 8AM ORB Options bot.
    Initializes all services and runs the two-phase trading day.
    """

    def __init__(self, config: Dict[str, Any], mode: str = "dry_run"):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: {VALID_MODES}")
        self.config = config
        self.mode = mode
        self._db = None
        self._market_data = None
        self._options_service = None
        self._tt_client = None
        self._tool_handlers = None
        self._decision_engine = None

    async def initialize(self):
        from src_8am.services.db_service import OrbDatabaseService
        from src_8am.services.market_data import OrbMarketData
        from src_8am.services.orb_options_service import OrbOptionsService
        from src_8am.agent.options_tool_handlers import OrbOptionsToolHandlers
        from src_8am.agent.options_decision_engine import OrbOptionsDecisionEngine
        from src_8am.agent.regime_classifier import OrbRegimeClassifier
        from src_8am.agent.anomaly_detector import OrbAnomalyDetector
        from src.services.llm_client import LLMClient

        db_cfg = self.config.get("database", {})
        llm_cfg = self.config.get("llm", {})

        self._db = OrbDatabaseService(db_cfg.get("trading_db", "orb_options_bot.db"))
        self._db.initialize()

        advisor_llm = LLMClient(model=llm_cfg.get("advisor_model", "anthropic/claude-haiku-4-5"))

        if self.mode != "dry_run":
            from src.services.tastytrade_client import TastyTradeClient
            self._tt_client = TastyTradeClient(mode=self.mode)
            await self._tt_client.connect()
        else:
            self._tt_client = _DryRunTTClient()

        self._market_data = OrbMarketData(tt_client=self._tt_client, config=self.config)
        self._options_service = OrbOptionsService(tt_client=self._tt_client, config=self.config)

        self._tool_handlers = OrbOptionsToolHandlers(
            market_data=self._market_data,
            options_service=self._options_service,
            tt_client=self._tt_client,
            db_service=self._db,
            config=self.config,
            mode=self.mode,
        )

        regime_classifier = OrbRegimeClassifier(llm_client=advisor_llm)
        anomaly_detector = OrbAnomalyDetector(config=self.config)

        self._decision_engine = OrbOptionsDecisionEngine(
            tool_handlers=self._tool_handlers,
            db_service=self._db,
            market_data=self._market_data,
            regime_classifier=regime_classifier,
            anomaly_detector=anomaly_detector,
            config=self.config,
            mode=self.mode,
        )

        logger.info("ORB Options bot initialized — mode: %s", self.mode.upper())

    async def run_day(self):
        """Run a full trading day: wait for 08:15, setup, wait for 09:30, entry watch."""
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        strategy = self.config.get("strategy", {})

        orb_end_str = strategy.get("orb_window_end", "08:15")
        orb_h, orb_m = [int(x) for x in orb_end_str.split(":")]
        orb_end = time(orb_h, orb_m)

        market_open_str = strategy.get("market_open", "09:30")
        mo_h, mo_m = [int(x) for x in market_open_str.split(":")]
        market_open_t = time(mo_h, mo_m)

        # Phase A: wait until 08:15, fetch ORB levels
        while True:
            now_et = datetime.now(ET)
            if now_et.time() >= orb_end:
                break
            wait_secs = max(1, (
                datetime.combine(now_et.date(), orb_end, tzinfo=ET) - now_et
            ).seconds)
            logger.info("Waiting for ORB window close (08:15 ET) — %ds", wait_secs)
            await asyncio.sleep(min(wait_secs, 30))

        ok = await self._decision_engine.run_setup_phase()
        if not ok:
            logger.error("ORB setup failed — aborting day")
            return

        # Phase B: wait until 09:30, start entry watch
        while True:
            now_et = datetime.now(ET)
            if now_et.time() >= market_open_t:
                break
            wait_secs = max(1, (
                datetime.combine(now_et.date(), market_open_t, tzinfo=ET) - now_et
            ).seconds)
            logger.info("Waiting for market open (09:30 ET) — %ds", wait_secs)
            await asyncio.sleep(min(wait_secs, 30))

        await self._decision_engine.run_entry_watch_loop()

    async def shutdown(self):
        if self._tt_client and not isinstance(self._tt_client, _DryRunTTClient):
            try:
                await self._tt_client.disconnect()
            except Exception:
                pass
        logger.info("ORB Options bot shutdown complete")


class _DryRunTTClient:
    """
    Stub TT client for dry-run mode.

    Provides simulated market data for testing without live API connections.
    All prices are synthetic but use realistic Black-Scholes calculations.
    """

    def __init__(self):
        self._base_price = 5500.0  # Base ES/SPX price
        self._price_drift = 0.0    # Accumulated price drift for simulation
        self._last_candle_time = None

    async def connect(self): pass
    async def disconnect(self): pass

    # -------------------------------------------------------------------------
    # Market data methods required by OrbMarketData
    # -------------------------------------------------------------------------

    async def get_candles(
        self,
        symbol: str,
        interval: str = "5m",
        from_time=None,
        count: int = 1,
    ):
        """
        Return simulated OHLC candles for dry-run testing.

        Generates realistic candles with slight random variation around base price.
        """
        import random
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        now = datetime.now(ET)

        # Parse interval to minutes
        interval_mins = 5
        if interval == "15m":
            interval_mins = 15
        elif interval == "1m":
            interval_mins = 1

        candles = []
        base = self._base_price + self._price_drift

        for i in range(count):
            # Add small random drift to simulate price movement
            drift = random.uniform(-1.5, 1.5)
            self._price_drift += drift * 0.1  # Accumulate small drift

            price = base + drift
            high = price + random.uniform(0.5, 2.0)
            low = price - random.uniform(0.5, 2.0)
            open_p = price + random.uniform(-1.0, 1.0)
            close_p = price + random.uniform(-1.0, 1.0)

            # Ensure OHLC consistency
            high = max(high, open_p, close_p)
            low = min(low, open_p, close_p)

            ts = now - timedelta(minutes=interval_mins * (count - 1 - i))

            candles.append({
                "open": round(open_p, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_p, 2),
                "volume": random.randint(1000, 5000),
                "timestamp": ts.isoformat(),
            })

        return candles

    async def get_future_quote(self, symbol: str):
        """
        Return simulated ES futures quote for price polling.

        Adds small random jitter to simulate live price movement.
        """
        import random

        # Add small jitter to simulate live price movement
        jitter = random.uniform(-0.50, 0.50)
        price = self._base_price + self._price_drift + jitter

        return {
            "symbol": symbol,
            "bid": round(price - 0.25, 2),
            "ask": round(price + 0.25, 2),
            "last": round(price, 2),
            "mark": round(price, 2),
        }

    async def get_quote(self, symbol: str):
        """Alias for get_future_quote for compatibility."""
        return await self.get_future_quote(symbol)

    # -------------------------------------------------------------------------
    # Options chain methods required by OrbOptionsService
    # -------------------------------------------------------------------------

    async def get_option_chain(self, symbol: str):
        """Return a synthetic SPX chain for dry-run testing with realistic BS deltas."""
        import math
        from datetime import date as dt_date

        def _ncdf(x: float) -> float:
            """Standard normal CDF via Abramowitz & Stegun approximation."""
            a = abs(x)
            t = 1.0 / (1.0 + 0.2316419 * a)
            poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                   + t * (-1.821255978 + t * 1.330274429))))
            cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
            return cdf if x >= 0 else 1.0 - cdf

        today = dt_date.today().isoformat()
        dt_str = today.replace("-", "")[2:]
        spx = 5500.0
        iv = 0.15        # ~15% IV typical for 0DTE SPX
        t = 2.0 / 252    # ~2h remaining post-09:30 (realistic for entry window)
        sqrt_t = math.sqrt(t)
        chain = []

        for strike in range(5200, 5800, 5):
            k = float(strike)
            # BS d1 for call delta
            d1 = (math.log(spx / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
            call_delta = round(_ncdf(d1), 4)
            put_delta = round(call_delta - 1.0, 4)

            # BS call/put prices: C = S*N(d1) - K*N(d2), P = K*N(-d2) - S*N(-d1)
            d2 = d1 - iv * sqrt_t
            call_mid = max(0.05, spx * _ncdf(d1) - k * _ncdf(d2))
            put_mid = max(0.05, k * _ncdf(-d2) - spx * _ncdf(-d1))
            call_bid = round(call_mid * 0.95, 2)
            put_bid = round(put_mid * 0.95, 2)

            chain.append({
                "strike": k, "option_type": "call",
                "bid": call_bid, "ask": round(call_bid + 0.10, 2),
                "delta": call_delta, "expiration": today,
                "symbol": f"SPXW{dt_str}C{strike:08d}",
                "streamer_symbol": f".SPXW{dt_str}C{strike}",
            })
            chain.append({
                "strike": k, "option_type": "put",
                "bid": put_bid, "ask": round(put_bid + 0.10, 2),
                "delta": put_delta, "expiration": today,
                "symbol": f"SPXW{dt_str}P{strike:08d}",
                "streamer_symbol": f".SPXW{dt_str}P{strike}",
            })
        return chain

    async def get_account_info(self):
        return {"equity": 100000.0, "buying_power": 50000.0, "cash_balance": 100000.0}

    async def place_order(self, order: Dict):
        import uuid
        return {"order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}"}


def _resolve_mode(args: argparse.Namespace, config: Dict[str, Any]) -> str:
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
    config_path = getattr(args, "config", "config/config_8am_options.yaml")
    config = _load_config(config_path)

    log_cfg = config.get("logging", {})
    _setup_logging(log_cfg.get("log_dir", "logs_8am_options"), log_cfg.get("level", "INFO"))

    mode = _resolve_mode(args, config)
    orchestrator = OrbOptionsBotOrchestrator(config=config, mode=mode)

    try:
        await orchestrator.initialize()
        await orchestrator.run_day()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await orchestrator.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="8AM ORB Options Bot — SPX 0DTE credit spread seller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Mode priority: CLI flag > ORB_BOT_MODE env var > config_8am_options.yaml bot.mode

Examples:
  python -m src_8am.main_options                   # dry-run
  python -m src_8am.main_options --manual          # live data, orders NOT submitted
  python -m src_8am.main_options --sandbox         # TastyTrade cert environment
  python -m src_8am.main_options --live            # TastyTrade production""",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--manual", action="store_true",
                            help="Live data, orders NOT submitted")
    mode_group.add_argument("--sandbox", action="store_true",
                            help="TastyTrade sandbox (cert) environment")
    mode_group.add_argument("--live", action="store_true",
                            help="TastyTrade production (real money)")
    parser.add_argument("--config", default="config/config_8am_options.yaml")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()

    def _suppress_ssl(loop, context):
        msg = context.get("message", "")
        if "SSL" in msg or "Bad file descriptor" in msg:
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_suppress_ssl)
    try:
        loop.run_until_complete(main_async(args))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
