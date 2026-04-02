"""
ORB Options tool handler implementations.
Adapts the ORB midline-retest entry signal to SPX 0DTE credit spread selling.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OrbOptionsToolHandlers:
    """
    Concrete implementations of the ORB options tools.

    LONG  bias → sell PUT  credit spread (below ORB low — bullish floor)
    SHORT bias → sell CALL credit spread (above ORB high — bearish ceiling)

    Entry timing: same midline retest signal as futures variant.
    Exit management: TP at 50% credit, SL at 200% credit, expire worthless at EOD.
    """

    def __init__(
        self,
        market_data,
        options_service,
        tt_client,
        db_service,
        config: Dict[str, Any],
        mode: str = "dry_run",
    ):
        self._md = market_data
        self._opts = options_service
        self._tt = tt_client
        self._db = db_service
        self._config = config
        self.mode = mode

        self._strategy = config.get("strategy", {})
        self._opt_cfg = config.get("options", {})
        self._bot = config.get("bot", {})

        self._orb_levels: Optional[Dict[str, Any]] = None
        self._breakout_bias: Optional[str] = None
        self._breakout_price: Optional[float] = None
        self._breakout_time: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # get_orb_levels (identical to futures variant)
    # -------------------------------------------------------------------------

    async def get_orb_levels(self) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            candle = await self._md.get_orb_candle()
            if not candle:
                result = {"error": "ORB candle not available yet"}
                await self._log("get_orb_levels", {}, result, start)
                return result

            high = float(candle["high"])
            low = float(candle["low"])
            midline = round((high + low) / 2, 4)
            range_pts = round(high - low, 4)

            self._orb_levels = {
                "high": high, "low": low,
                "midline": midline, "range_pts": range_pts,
            }
            await self._log("get_orb_levels", {}, self._orb_levels, start)
            return self._orb_levels
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log("get_orb_levels", {}, result, start)
            return result

    # -------------------------------------------------------------------------
    # get_breakout_status
    # -------------------------------------------------------------------------

    async def get_breakout_status(self) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            if not self._orb_levels:
                return {"status": "no_orb_levels", "bias": None}

            status = await self._md.get_breakout_status(
                orb_high=self._orb_levels["high"],
                orb_low=self._orb_levels["low"],
            )

            if status["status"] == "breakout" and self._breakout_bias is None:
                self._breakout_bias = status["bias"]
                self._breakout_price = status["price"]
                self._breakout_time = datetime.now()
                logger.info(
                    "Breakout detected: bias=%s price=%.2f",
                    self._breakout_bias, self._breakout_price,
                )

            result = {
                **status,
                "orb_high": self._orb_levels["high"],
                "orb_low": self._orb_levels["low"],
                "midline": self._orb_levels["midline"],
                "established_bias": self._breakout_bias,
            }
            await self._log("get_breakout_status", {}, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log("get_breakout_status", {}, result, start)
            return result

    # -------------------------------------------------------------------------
    # get_spread_candidates
    # -------------------------------------------------------------------------

    async def get_spread_candidates(self) -> Dict[str, Any]:
        """
        Fetch ES futures option chain and compute the spread to sell based on current bias.
        Returns the spread selection dict (strikes, credit, TP/SL levels).
        """
        start = time.monotonic()
        try:
            if not self._orb_levels or not self._breakout_bias:
                result = {"error": "no_orb_levels_or_bias"}
                await self._log("get_spread_candidates", {}, result, start)
                return result

            es_price = await self._md.get_current_price()
            if es_price is None:
                result = {"error": "no_underlying_price"}
                await self._log("get_spread_candidates", {}, result, start)
                return result

            chain = await self._opts.get_0dte_chain(es_price)
            if not chain:
                result = {"error": "empty_option_chain"}
                await self._log("get_spread_candidates", {}, result, start)
                return result

            orb_bias = self._breakout_bias
            use_em = self._opts._use_em
            spread_mode = self._opts._spread_mode

            if spread_mode == "iron_condor":
                # IC mode: place both legs
                spreads = []
                for leg_type in ["put_credit_spread", "call_credit_spread"]:
                    if use_em:
                        s = self._opts.select_em_spread(chain, es_price, leg_type, orb_bias)
                    elif leg_type == "put_credit_spread":
                        s = self._opts.select_put_spread(chain, self._orb_levels["low"], es_price)
                    else:
                        s = self._opts.select_call_spread(chain, self._orb_levels["high"], es_price)
                    if s is not None:
                        spreads.append({**s, "underlying_price": es_price, "bias": orb_bias})

                if not spreads:
                    result = {"error": "no_valid_spread_found", "bias": orb_bias}
                    await self._log("get_spread_candidates", {}, result, start)
                    return result

                # Return as IC with both legs
                result = {
                    "spreads": spreads,
                    "spread_mode": "iron_condor",
                    "underlying_price": es_price,
                    "bias": orb_bias,
                }
                await self._log("get_spread_candidates", {}, result, start)
                return result

            # Directional mode: single leg
            if use_em:
                leg_type = "put_credit_spread" if orb_bias == "long" else "call_credit_spread"
                spread = self._opts.select_em_spread(chain, es_price, leg_type, orb_bias)
            elif orb_bias == "long":
                spread = self._opts.select_put_spread(
                    chain, self._orb_levels["low"], es_price
                )
            else:
                spread = self._opts.select_call_spread(
                    chain, self._orb_levels["high"], es_price
                )

            if spread is None:
                result = {"error": "no_valid_spread_found", "bias": orb_bias}
                await self._log("get_spread_candidates", {}, result, start)
                return result

            result = {**spread, "underlying_price": es_price, "bias": orb_bias}
            await self._log("get_spread_candidates", {}, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log("get_spread_candidates", {}, result, start)
            return result

    # -------------------------------------------------------------------------
    # get_account_info
    # -------------------------------------------------------------------------

    async def get_account_info(self) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            if self.mode == "dry_run":
                result = {"net_liq": 100000.0, "buying_power": 50000.0, "mode": "dry_run"}
            else:
                raw = await self._tt.get_account_info()
                result = {
                    "net_liq": raw.get("equity", 0),
                    "buying_power": raw.get("buying_power", 0),
                    "cash": raw.get("cash_balance", 0),
                }
            await self._log("get_account_info", {}, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log("get_account_info", {}, result, start)
            return result

    # -------------------------------------------------------------------------
    # place_credit_spread
    # -------------------------------------------------------------------------

    async def place_credit_spread(self, spread: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place the selected credit spread. In dry_run/manual: simulate fill.
        Returns result dict with trade_id, order_id, and fill details.
        """
        start = time.monotonic()
        try:
            spread_type = spread["spread_type"]
            qty = self._bot.get("quantity", 1)

            if self.mode in ("dry_run", "manual"):
                trade_id = f"OPT-DRY-{uuid.uuid4().hex[:8].upper()}"
                result = {
                    "trade_id": trade_id,
                    "order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
                    "status": "simulated",
                    "spread_type": spread_type,
                    "short_strike": spread["short_strike"],
                    "long_strike": spread["long_strike"],
                    "credit_received": spread["credit"],
                    "quantity": qty,
                    "mode": self.mode,
                }
                logger.info(
                    "[%s] Would sell %s: short=%s long=%s credit=%.2f max_loss=%.0f",
                    self.mode.upper(),
                    spread_type,
                    spread["short_strike"],
                    spread["long_strike"],
                    spread["credit"],
                    spread.get("max_loss_dollars", 0),
                )
            else:
                order = {
                    "order_type": spread_type,
                    "short_strike": spread["short_strike"],
                    "long_strike": spread["long_strike"],
                    "short_symbol": spread["short_symbol"],
                    "long_symbol": spread["long_symbol"],
                    "instrument_type": "Future Option",
                    "credit": spread.get("credit", 0),
                    "quantity": qty,
                }
                raw = await self._tt.place_order(order)
                trade_id = f"OPT-{uuid.uuid4().hex[:8].upper()}"
                result = {
                    "trade_id": trade_id,
                    "order_id": raw.get("order_id"),
                    "status": "submitted",
                    "spread_type": spread_type,
                    "short_strike": spread["short_strike"],
                    "long_strike": spread["long_strike"],
                    "credit_received": spread["credit"],
                    "quantity": qty,
                }

            await self._log("place_credit_spread", spread, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log("place_credit_spread", spread, result, start)
            return result

    # -------------------------------------------------------------------------
    # close_spread
    # -------------------------------------------------------------------------

    async def close_spread(self, trade_id: str, reason: str = "manual") -> Dict[str, Any]:
        """Close (buy back) an open spread position."""
        start = time.monotonic()
        args = {"trade_id": trade_id, "reason": reason}
        try:
            if self.mode in ("dry_run", "manual"):
                result = {"trade_id": trade_id, "status": "simulated_close", "reason": reason}
            else:
                result = await self._tt.place_order({
                    "order_type": "close_spread",
                    "trade_id": trade_id,
                })
                result["reason"] = reason
            await self._log("close_spread", args, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log("close_spread", args, result, start)
            return result

    # -------------------------------------------------------------------------
    # get_candles (for regime classifier)
    # -------------------------------------------------------------------------

    async def get_candles(self, count: int = 10) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            candles = await self._md.get_recent_5min_candles(count=count)
            result = {"candles": candles, "count": len(candles)}
            await self._log("get_candles", {"count": count}, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log("get_candles", {"count": count}, result, start)
            return result

    # -------------------------------------------------------------------------
    # Midline touch check
    # -------------------------------------------------------------------------

    def check_midline_touch(
        self, current_price: float, midline: float, tolerance_pts: Optional[float] = None
    ) -> bool:
        tol = tolerance_pts if tolerance_pts is not None else self._strategy.get(
            "midline_tolerance_pts", 0.25
        )
        return abs(current_price - midline) <= tol

    # -------------------------------------------------------------------------
    # Internal logging
    # -------------------------------------------------------------------------

    async def _log(self, name: str, args: Dict, result: Dict, start: float):
        elapsed_ms = int((time.monotonic() - start) * 1000)
        try:
            await self._db.log_tool_call({
                "name": name,
                "arguments": args,
                "result": result,
                "session_date": date.today(),
                "duration_ms": elapsed_ms,
            })
        except Exception:
            logger.debug("Tool log failed for %s", name, exc_info=True)
