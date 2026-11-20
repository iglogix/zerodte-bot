"""
ORB Decision Engine — deterministic two-phase trading cycle.

Phase A (08:15 ET): Fetch ORB candle, compute levels, store for day.
Phase B (09:30+ ET): Entry watch loop — polls every ~1s for midline touch.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from es_8am.agent.anomaly_detector import OrbAnomalyDetector
from es_8am.agent.regime_classifier import OrbRegimeClassifier

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class OrbDecisionEngine:
    """
    Orchestrates the ORB strategy cycle for a single trading day.

    Phase A — Setup (08:15 ET, runs once):
        1. Fetch 15-min ORB candle → compute high / low / midline
        2. Store in DB

    Phase B — Entry watch loop (09:30 ET, polls every ~1s):
        1. Time guard: before 09:30 → wait
        2. One-trade-per-day guard
        3. Poll breakout status
        4. On first confirmed breakout → AI pre-flight (anomaly + regime)
        5. Poll for midline touch
        6. On midline touch → risk check → place bracket order → log → exit loop
    """

    def __init__(
        self,
        tool_handlers,
        db_service,
        market_data,
        regime_classifier: OrbRegimeClassifier,
        anomaly_detector: OrbAnomalyDetector,
        config: Dict[str, Any],
        mode: str = "dry_run",
    ):
        self._tools = tool_handlers
        self._db = db_service
        self._md = market_data
        self._regime = regime_classifier
        self._anomaly = anomaly_detector
        self._config = config
        self.mode = mode

        self._strategy = config.get("strategy", {})
        self._exit = config.get("exit", {})
        self._risk = config.get("risk", {})
        self._bot = config.get("bot", {})

        self._orb_levels: Optional[Dict[str, Any]] = None
        self._session_trade_id: Optional[str] = None
        self._preflight_done: bool = False
        self._preflight_result: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------------
    # Phase A — Setup
    # -------------------------------------------------------------------------

    async def run_setup_phase(self) -> bool:
        """
        Fetch and store the ORB candle. Should be called at or after 08:15 ET.
        Returns True if ORB levels were successfully established.
        """
        logger.info("Phase A: fetching ORB candle")
        levels = await self._tools.get_orb_levels()

        if "error" in levels:
            logger.error("Phase A failed: %s", levels["error"])
            return False

        self._orb_levels = levels
        logger.info(
            "ORB levels: H=%.2f L=%.2f mid=%.2f range=%.2f pts",
            levels["high"],
            levels["low"],
            levels["midline"],
            levels["range_pts"],
        )

        today = date.today()
        trade_id = f"ORB-{today.isoformat()}-{uuid.uuid4().hex[:6].upper()}"
        self._session_trade_id = trade_id

        await self._db.create_trade(
            {
                "trade_id": trade_id,
                "symbol": self._bot.get("symbol", "ES"),
                "session_date": today,
                "orb_high": levels["high"],
                "orb_low": levels["low"],
                "orb_midline": levels["midline"],
                "orb_range_pts": levels["range_pts"],
                "status": "watching",
            }
        )
        return True

    # -------------------------------------------------------------------------
    # Phase B — Entry watch loop
    # -------------------------------------------------------------------------

    async def run_entry_watch_loop(self):
        """
        Main entry watch loop. Runs from 09:30 until trade placed or EOD.
        Polls every ~1s.
        """
        if not self._orb_levels:
            logger.error("Entry watch loop started without ORB levels — aborting")
            return

        eod_str = self._strategy.get("eod_force_close", "15:50")
        eod_h, eod_m = [int(x) for x in eod_str.split(":")]
        tolerance = self._strategy.get("midline_tolerance_pts", 0.25)
        poll_interval = 1.0

        logger.info("Phase B: entry watch loop started")

        while True:
            now_et = datetime.now(ET)
            now_time = now_et.time()

            # EOD guard
            if now_time >= time(eod_h, eod_m):
                logger.info("EOD reached (%s ET) — no trade placed today", eod_str)
                if self._session_trade_id:
                    await self._db.update_trade(
                        self._session_trade_id,
                        {"status": "skipped", "exit_reason": "no_retest"},
                    )
                return

            # One-trade-per-day guard
            if await self._db.traded_today(date.today()):
                logger.info("Trade already placed today — exiting watch loop")
                return

            # Fetch current price
            current_price = await self._md.get_current_price()
            if current_price is None:
                await asyncio.sleep(poll_interval)
                continue

            # Step 1: Check for breakout
            if self._tools._breakout_bias is None:
                status = await self._tools.get_breakout_status()
                if status.get("status") != "breakout":
                    await asyncio.sleep(poll_interval)
                    continue
                # Breakout just detected — fall through to pre-flight

            bias = self._tools._breakout_bias
            if bias is None:
                await asyncio.sleep(poll_interval)
                continue

            # Step 2: AI pre-flight (once per breakout)
            if not self._preflight_done:
                logger.info("Breakout confirmed (%s) — running AI pre-flight", bias)
                ok = await self._run_preflight(current_price)
                self._preflight_done = True
                if not ok:
                    logger.info("Pre-flight blocked entry — exiting watch loop")
                    if self._session_trade_id:
                        await self._db.update_trade(
                            self._session_trade_id,
                            {
                                "status": "skipped",
                                "exit_reason": "preflight_blocked",
                                "bias": bias,
                                "regime_verdict": self._preflight_result.get("regime", {}).get("verdict"),
                                "regime_reasoning": self._preflight_result.get("regime", {}).get("reasoning"),
                            },
                        )
                    return

            # Step 3: Check midline touch
            midline = self._orb_levels["midline"]
            if self._tools.check_midline_touch(current_price, midline, tolerance):
                logger.info(
                    "Midline touch detected: price=%.2f midline=%.2f bias=%s",
                    current_price,
                    midline,
                    bias,
                )
                placed = await self._execute_entry(current_price, bias, midline)
                if placed:
                    return

            await asyncio.sleep(poll_interval)

    # -------------------------------------------------------------------------
    # Pre-flight checks
    # -------------------------------------------------------------------------

    async def _run_preflight(self, current_price: float) -> bool:
        """Run anomaly check + regime classification. Returns True if safe to proceed."""
        account = await self._tools.get_account_info()

        # Anomaly check
        anomaly = await self._anomaly.check(
            current_price=current_price,
            orb_levels=self._orb_levels,
            account_info=account,
        )
        if not anomaly["safe"]:
            logger.warning("Anomaly check failed: %s", anomaly["reasons"])
            self._preflight_result = {"anomaly": anomaly, "regime": {"verdict": "blocked"}}
            return False

        # Regime classification
        candles_result = await self._tools.get_candles(count=10)
        candles = candles_result.get("candles", [])
        regime = await self._regime.classify(
            orb_levels=self._orb_levels,
            candles=candles,
            bias=self._tools._breakout_bias,
            breakout_price=self._tools._breakout_price,
        )

        self._preflight_result = {"anomaly": anomaly, "regime": regime}

        if regime.get("verdict") in ("skip_trade", "skip_session"):
            logger.info("Regime classifier blocked: %s", regime.get("reasoning"))
            return False

        return True

    # -------------------------------------------------------------------------
    # Entry execution
    # -------------------------------------------------------------------------

    async def _execute_entry(
        self,
        entry_price: float,
        bias: str,
        midline: float,
    ) -> bool:
        """Compute TP/SL, run risk check, place bracket order, update DB."""
        tp_pts = self._exit.get("tp_pts", 20)
        sl_pts = self._exit.get("sl_pts", 5)
        qty = self._bot.get("quantity", 1)

        if bias == "long":
            tp_price = round(entry_price + tp_pts, 4)
            sl_price = round(entry_price - sl_pts, 4)
        else:
            tp_price = round(entry_price - tp_pts, 4)
            sl_price = round(entry_price + sl_pts, 4)

        # Risk check
        max_daily_loss = self._risk.get("max_daily_loss_pts", 10)
        closed = await self._db.get_closed_trades(start_date=date.today(), end_date=date.today())
        daily_loss = sum(
            t.get("realized_pnl_pts", 0) or 0
            for t in closed
            if (t.get("realized_pnl_pts") or 0) < 0
        )
        if abs(daily_loss) >= max_daily_loss:
            logger.warning("Daily loss limit reached (%.1f pts) — blocking entry", daily_loss)
            if self._session_trade_id:
                await self._db.update_trade(
                    self._session_trade_id,
                    {"status": "skipped", "exit_reason": "daily_loss_limit"},
                )
            return False

        result = await self._tools.place_futures_bracket(
            direction=bias,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            quantity=qty,
        )

        if "error" in result:
            logger.error("Order placement failed: %s", result["error"])
            return False

        now = datetime.now()
        updates = {
            "bias": bias,
            "breakout_time": self._tools._breakout_time,
            "breakout_price": self._tools._breakout_price,
            "entry_time": now,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "status": "entered",
            "regime_verdict": (self._preflight_result or {}).get("regime", {}).get("verdict"),
            "regime_reasoning": (self._preflight_result or {}).get("regime", {}).get("reasoning"),
        }
        if self._session_trade_id:
            await self._db.update_trade(self._session_trade_id, updates)

        logger.info(
            "Trade entered: %s %s @ %.2f | TP=%.2f SL=%.2f | id=%s",
            bias.upper(),
            self._bot.get("symbol", "ES"),
            entry_price,
            tp_price,
            sl_price,
            result.get("trade_id", "?"),
        )
        return True
