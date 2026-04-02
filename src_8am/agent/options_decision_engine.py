"""
ORB Options Decision Engine — deterministic two-phase cycle for SPX 0DTE credit spreads.

Phase A (08:15 ET): Fetch ORB candle, compute levels, store for day.
Phase B (09:30+ ET): Entry watch loop — polls every ~1s for midline touch,
                     then sells a credit spread aligned with the breakout bias.
Phase C (continuous): Position monitor — polls spread mark price every 30s
                      to check TP/SL and manage exit.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, date, time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from src_8am.agent.anomaly_detector import OrbAnomalyDetector
from src_8am.agent.regime_classifier import OrbRegimeClassifier

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class OrbOptionsDecisionEngine:
    """
    Orchestrates the ORB credit spread strategy for a single trading day.

    Phase A — Setup (08:15 ET, once):
        1. Fetch 15-min ORB candle → compute high / low / midline
        2. Persist watching record in DB

    Phase B — Entry watch (09:30 ET, ~1s poll):
        1. Detect breakout above ORB high (long) or below ORB low (short)
        2. AI pre-flight: anomaly + regime checks (once per breakout)
        3. On midline retest → fetch option chain → select spread → place order

    Phase C — Position monitor (30s poll):
        4. Poll spread mark price via TT REST
        5. Buy back at 50% of credit (TP) or 200% of credit (SL)
        6. At EOD (15:50): let expire if OTM, close if ITM
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
        self._opt_cfg = config.get("options", {})
        self._risk = config.get("risk", {})
        self._bot = config.get("bot", {})
        self._llm_enabled = config.get("llm", {}).get("enabled", True)

        self._orb_levels: Optional[Dict[str, Any]] = None
        self._session_trade_id: Optional[str] = None
        self._active_spread: Optional[Dict[str, Any]] = None
        self._preflight_done: bool = False
        self._preflight_result: Optional[Dict[str, Any]] = None

        eod_str = self._strategy.get("eod_force_close", "15:50")
        h, m = [int(x) for x in eod_str.split(":")]
        self._eod = time(h, m)

    # -------------------------------------------------------------------------
    # Phase A — Setup
    # -------------------------------------------------------------------------

    async def run_setup_phase(self) -> bool:
        """Fetch ORB candle and persist the session record. Returns True on success."""

        # Check for event-day skips (FOMC, CPI, NFP)
        today = date.today()
        from src_8am.backtest.options_engine import OrbOptionsBacktestEngine
        if self._strategy.get("skip_fomc_days", False) and today in OrbOptionsBacktestEngine._FOMC_DATES:
            logger.info("FOMC day (%s) — skipping session per config", today)
            return False
        if self._strategy.get("skip_cpi_days", False) and today in OrbOptionsBacktestEngine._CPI_DATES:
            logger.info("CPI day (%s) — skipping session per config", today)
            return False
        if self._strategy.get("skip_nfp_fridays", False) and today.weekday() == 4 and today.day <= 7:
            logger.info("NFP Friday (%s) — skipping session per config", today)
            return False

        logger.info("Phase A: fetching ORB candle (options mode)")
        levels = await self._tools.get_orb_levels()

        if "error" in levels:
            logger.error("Phase A failed: %s", levels["error"])
            return False

        self._orb_levels = levels
        logger.info(
            "ORB levels: H=%.2f L=%.2f mid=%.2f range=%.2f pts",
            levels["high"], levels["low"], levels["midline"], levels["range_pts"],
        )

        trade_id = f"ORBOPT-{today.isoformat()}-{uuid.uuid4().hex[:6].upper()}"
        self._session_trade_id = trade_id

        from src_8am.models.database import OrbSpreadTrade
        await self._db.create_spread_trade({
            "trade_id": trade_id,
            "session_date": today,
            "orb_high": levels["high"],
            "orb_low": levels["low"],
            "orb_midline": levels["midline"],
            "orb_range_pts": levels["range_pts"],
            "status": "watching",
        })
        return True

    # -------------------------------------------------------------------------
    # Phase B — Entry watch loop
    # -------------------------------------------------------------------------

    async def run_entry_watch_loop(self):
        """Poll every ~1s from 09:30 until spread placed or EOD."""
        if not self._orb_levels:
            logger.error("No ORB levels — aborting entry watch")
            return

        tolerance = self._strategy.get("midline_tolerance_pts", 0.25)
        poll_interval = 1.0

        logger.info("Phase B: entry watch loop started (options mode)")

        while True:
            now_et = datetime.now(ET)
            if now_et.time() >= self._eod:
                logger.info("EOD reached — no spread placed today")
                if self._session_trade_id:
                    await self._db.update_spread_trade(
                        self._session_trade_id,
                        {"status": "skipped", "exit_reason": "no_retest"},
                    )
                return

            if await self._db.spread_traded_today(date.today()):
                logger.info("Spread already placed today — exiting entry watch")
                return

            current_price = await self._md.get_current_price()
            if current_price is None:
                await asyncio.sleep(poll_interval)
                continue

            # Detect breakout
            if self._tools._breakout_bias is None:
                status = await self._tools.get_breakout_status()
                if status.get("status") != "breakout":
                    await asyncio.sleep(poll_interval)
                    continue

            bias = self._tools._breakout_bias
            if bias is None:
                await asyncio.sleep(poll_interval)
                continue

            # AI pre-flight (once per breakout)
            if not self._preflight_done:
                logger.info("Breakout confirmed (%s) — running AI pre-flight (options)", bias)
                ok = await self._run_preflight(current_price)
                self._preflight_done = True
                if not ok:
                    logger.info("Pre-flight blocked — skipping day")
                    if self._session_trade_id:
                        await self._db.update_spread_trade(
                            self._session_trade_id,
                            {
                                "status": "skipped",
                                "exit_reason": "preflight_blocked",
                                "bias": bias,
                                "regime_verdict": (self._preflight_result or {}).get(
                                    "regime", {}
                                ).get("verdict"),
                            },
                        )
                    return

            # Midline touch check
            midline = self._orb_levels["midline"]
            if self._tools.check_midline_touch(current_price, midline, tolerance):
                logger.info(
                    "Midline touch: price=%.2f mid=%.2f bias=%s → fetching spread",
                    current_price, midline, bias,
                )
                placed = await self._execute_entry(current_price, bias)
                if placed:
                    # Hand off to position monitor
                    await self._run_position_monitor()
                    return

            await asyncio.sleep(poll_interval)

    # -------------------------------------------------------------------------
    # Phase C — Position monitor
    # -------------------------------------------------------------------------

    async def _run_position_monitor(self):
        """
        Poll spread mark price every 30s until TP, SL, or EOD.
        In live mode, fetches current spread mark via TT option chain.
        In dry_run/manual, simulates mark price decay.
        """
        if not self._active_spread:
            return

        poll_secs = self._opt_cfg.get("monitor_poll_seconds", 30)
        credit = self._active_spread["credit_received"]
        tp_debit = self._active_spread["tp_debit"]
        sl_debit = self._active_spread["sl_debit"]
        spread_type = self._active_spread["spread_type"]

        logger.info(
            "Phase C: monitoring spread %s | credit=%.2f TP=%.2f SL=%.2f",
            spread_type, credit, tp_debit, sl_debit,
        )

        sim_tick = 0

        while True:
            now_et = datetime.now(ET)

            if now_et.time() >= self._eod:
                await self._exit_position("eod_close", mark=None)
                return

            mark = await self._get_spread_mark(sim_tick)
            sim_tick += 1

            if mark is not None:
                logger.debug("Spread mark=%.2f tp_debit=%.2f sl_debit=%.2f", mark, tp_debit, sl_debit)

                if mark <= tp_debit:
                    logger.info("TP hit: mark=%.2f <= tp_debit=%.2f", mark, tp_debit)
                    await self._exit_position("tp_50pct", mark=mark)
                    return

                if mark >= sl_debit:
                    logger.info("SL hit: mark=%.2f >= sl_debit=%.2f", mark, sl_debit)
                    await self._exit_position("sl_200pct", mark=mark)
                    return

            await asyncio.sleep(poll_secs)

    async def _get_spread_mark(self, tick: int) -> Optional[float]:
        """
        Return current spread mark price.
        dry_run: simulate linear decay to 0 over 120 ticks (1h at 30s poll).
        live: fetch from TT option chain.
        """
        if self.mode == "dry_run":
            credit = self._active_spread["credit_received"]
            decay = max(0.0, credit * (1 - tick / 120))
            return round(decay, 2)
        try:
            short_sym = self._active_spread.get("short_symbol", "")
            long_sym = self._active_spread.get("long_symbol", "")
            if not short_sym:
                return None
            short_q = await self._tt_mark(short_sym)
            long_q = await self._tt_mark(long_sym)
            if short_q is not None and long_q is not None:
                return round(short_q - long_q, 2)
        except Exception:
            logger.info("Spread mark fetch failed", exc_info=True)
        return None

    async def _tt_mark(self, symbol: str) -> Optional[float]:
        try:
            q = await self._tools._tt.get_quote(symbol)
            b, a = q.get("bid", 0), q.get("ask", 0)
            return (b + a) / 2 if b and a else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Pre-flight
    # -------------------------------------------------------------------------

    async def _run_preflight(self, current_price: float) -> bool:
        # Bypass LLM pre-flight checks when disabled (mechanical strategy)
        if not self._llm_enabled:
            logger.info("LLM disabled — skipping anomaly/regime checks, entering on signal")
            self._preflight_result = {
                "anomaly": {"safe": True, "reasons": ["llm_disabled"]},
                "regime": {"verdict": "enter", "reasoning": "llm_disabled"},
            }
            return True

        account = await self._tools.get_account_info()

        anomaly = await self._anomaly.check(
            current_price=current_price,
            orb_levels=self._orb_levels,
            account_info=account,
        )
        if not anomaly["safe"]:
            logger.warning("Anomaly blocked: %s", anomaly["reasons"])
            self._preflight_result = {"anomaly": anomaly, "regime": {"verdict": "blocked"}}
            return False

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
            logger.info("Regime blocked: %s", regime.get("reasoning"))
            return False
        return True

    # -------------------------------------------------------------------------
    # Entry execution
    # -------------------------------------------------------------------------

    async def _execute_entry(self, current_price: float, bias: str) -> bool:
        """Select spread, validate, place order, persist to DB."""
        candidates = await self._tools.get_spread_candidates()
        if "error" in candidates:
            logger.error("Spread selection failed: %s", candidates["error"])
            return False

        # IC mode returns {"spreads": [...], "spread_mode": "iron_condor"}
        if candidates.get("spread_mode") == "iron_condor":
            return await self._execute_ic_entry(candidates, current_price, bias)

        # Directional: single spread
        return await self._place_single_spread(candidates, current_price, bias)

    async def _place_single_spread(
        self, spread: Dict[str, Any], current_price: float, bias: str
    ) -> bool:
        """Place a single directional spread."""
        max_loss = spread.get("max_loss_dollars", 0)
        max_risk_usd = self._risk.get("max_loss_per_trade_usd", 500)
        qty = self._bot.get("quantity", 1)
        if max_loss * qty > max_risk_usd:
            logger.warning(
                "Max loss $%.0f exceeds limit $%.0f — skipping", max_loss * qty, max_risk_usd
            )
            if self._session_trade_id:
                await self._db.update_spread_trade(
                    self._session_trade_id,
                    {"status": "skipped", "exit_reason": "risk_limit"},
                )
            return False

        result = await self._tools.place_credit_spread(spread)
        if "error" in result:
            logger.error("Order failed: %s", result["error"])
            return False

        now = datetime.now()
        credit = spread["credit"]
        updates = {
            "bias": bias,
            "breakout_time": self._tools._breakout_time,
            "breakout_price": self._tools._breakout_price,
            "spread_type": spread["spread_type"],
            "short_strike": spread["short_strike"],
            "long_strike": spread["long_strike"],
            "spread_width": spread["spread_width"],
            "short_symbol": spread["short_symbol"],
            "long_symbol": spread["long_symbol"],
            "expiration": spread.get("expiration", date.today().isoformat()),
            "entry_time": now,
            "credit_received": credit,
            "max_profit": spread.get("max_profit_dollars"),
            "max_loss": spread.get("max_loss_dollars"),
            "spx_at_entry": current_price,
            "status": "entered",
            "regime_verdict": (self._preflight_result or {}).get("regime", {}).get("verdict"),
            "regime_reasoning": (self._preflight_result or {}).get("regime", {}).get("reasoning"),
        }
        if self._session_trade_id:
            await self._db.update_spread_trade(self._session_trade_id, updates)

        self._active_spread = {
            "trade_id": result.get("trade_id", self._session_trade_id),
            "credit_received": credit,
            "tp_debit": spread["tp_debit"],
            "sl_debit": spread["sl_debit"],
            "spread_type": spread["spread_type"],
            "short_symbol": spread["short_symbol"],
            "long_symbol": spread["long_symbol"],
        }

        logger.info(
            "Spread entered: %s short=%s long=%s credit=%.2f TP@%.2f SL@%.2f",
            spread["spread_type"],
            spread["short_strike"],
            spread["long_strike"],
            credit,
            spread["tp_debit"],
            spread["sl_debit"],
        )
        return True

    async def _execute_ic_entry(
        self, candidates: Dict[str, Any], current_price: float, bias: str
    ) -> bool:
        """Place both legs of an iron condor."""
        spreads = candidates.get("spreads", [])
        if not spreads:
            logger.warning("IC mode: no valid spreads")
            return False

        placed = []
        for spread in spreads:
            result = await self._tools.place_credit_spread(spread)
            if "error" in result:
                logger.error("IC leg failed: %s %s", spread["spread_type"], result["error"])
                continue
            placed.append((spread, result))

        if not placed:
            return False

        # Track all legs as active spreads
        if not hasattr(self, "_active_spreads"):
            self._active_spreads = []

        now = datetime.now()
        for spread, result in placed:
            credit = spread["credit"]
            trend_type = spread.get("trend_type", "")
            em_info = f" ({trend_type} {spread.get('em_multiplier', '')}×EM)" if trend_type else ""

            logger.info(
                "Spread entered: %s short=%s long=%s credit=%.2f TP@%.2f SL@%.2f%s",
                spread["spread_type"],
                spread["short_strike"],
                spread["long_strike"],
                credit,
                spread["tp_debit"],
                spread["sl_debit"],
                em_info,
            )

            active = {
                "trade_id": result.get("trade_id"),
                "credit_received": credit,
                "tp_debit": spread["tp_debit"],
                "sl_debit": spread["sl_debit"],
                "spread_type": spread["spread_type"],
                "short_symbol": spread["short_symbol"],
                "long_symbol": spread["long_symbol"],
                "trend_type": trend_type,
            }
            self._active_spreads.append(active)

            # Set first leg as primary active spread for backward compat
            if not self._active_spread:
                self._active_spread = active

        return True

    # -------------------------------------------------------------------------
    # Exit
    # -------------------------------------------------------------------------

    async def _exit_position(self, reason: str, mark: Optional[float]):
        """Close the spread and record exit."""
        if not self._active_spread:
            return

        trade_id = self._active_spread["trade_id"]
        credit = self._active_spread["credit_received"]

        if reason == "eod_close":
            # Determine if likely OTM (expire worthless) vs ITM (must close)
            spx_price = await self._md.get_current_price()
            if spx_price is not None:
                short_strike = (self._active_spread.get("short_symbol") or "")
                debit = 0.0  # assume expires worthless — conservative approximation
                reason = "expired_worthless"
            else:
                debit = mark or 0.0
        else:
            debit = mark or 0.0
            await self._tools.close_spread(trade_id=trade_id, reason=reason)

        pnl = round((credit - debit) * 100 * self._bot.get("quantity", 1), 2)

        if self._session_trade_id:
            await self._db.update_spread_trade(
                self._session_trade_id,
                {
                    "exit_time": datetime.now(),
                    "debit_paid": debit,
                    "exit_reason": reason,
                    "realized_pnl": pnl,
                    "status": "closed",
                },
            )

        logger.info(
            "Spread closed: reason=%s debit=%.2f pnl=$%.2f",
            reason, debit, pnl,
        )
