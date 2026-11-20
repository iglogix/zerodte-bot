"""
ORB tool handler implementations.
Each method corresponds to a tool in tool_definitions.py.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OrbToolHandlers:
    """
    Concrete implementations of the ORB agent tools.
    Wraps OrbMarketData, TastyTradeClient, and OrbDatabaseService.
    """

    def __init__(
        self,
        market_data,
        tt_client,
        db_service,
        config: Dict[str, Any],
        mode: str = "dry_run",
    ):
        self._md = market_data
        self._tt = tt_client
        self._db = db_service
        self._config = config
        self.mode = mode

        self._strategy = config.get("strategy", {})
        self._exit = config.get("exit", {})
        self._bot = config.get("bot", {})

        self._orb_levels: Optional[Dict[str, Any]] = None
        self._breakout_bias: Optional[str] = None
        self._breakout_price: Optional[float] = None
        self._breakout_time: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # get_orb_levels
    # -------------------------------------------------------------------------

    async def get_orb_levels(self) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            candle = await self._md.get_orb_candle()
            if not candle:
                result = {"error": "ORB candle not available yet"}
                await self._log_tool("get_orb_levels", {}, result, start)
                return result

            high = float(candle["high"])
            low = float(candle["low"])
            midline = round((high + low) / 2, 4)
            range_pts = round(high - low, 4)

            self._orb_levels = {
                "high": high,
                "low": low,
                "midline": midline,
                "range_pts": range_pts,
                "candle_open": candle.get("open"),
                "candle_close": candle.get("close"),
                "candle_time": candle.get("time"),
            }
            await self._log_tool("get_orb_levels", {}, self._orb_levels, start)
            return self._orb_levels
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log_tool("get_orb_levels", {}, result, start)
            return result

    # -------------------------------------------------------------------------
    # get_candles
    # -------------------------------------------------------------------------

    async def get_candles(self, count: int = 10) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            candles = await self._md.get_recent_5min_candles(count=count)
            result = {"candles": candles, "count": len(candles)}
            await self._log_tool("get_candles", {"count": count}, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log_tool("get_candles", {"count": count}, result, start)
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
                    self._breakout_bias,
                    self._breakout_price,
                )

            result = {
                **status,
                "orb_high": self._orb_levels["high"],
                "orb_low": self._orb_levels["low"],
                "midline": self._orb_levels["midline"],
                "established_bias": self._breakout_bias,
            }
            await self._log_tool("get_breakout_status", {}, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log_tool("get_breakout_status", {}, result, start)
            return result

    # -------------------------------------------------------------------------
    # get_account_info
    # -------------------------------------------------------------------------

    async def get_account_info(self) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            if self.mode == "dry_run":
                result = {
                    "net_liq": 100000.0,
                    "buying_power": 50000.0,
                    "cash": 100000.0,
                    "mode": "dry_run",
                }
            else:
                result = await self._tt.get_account_info()
            await self._log_tool("get_account_info", {}, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log_tool("get_account_info", {}, result, start)
            return result

    # -------------------------------------------------------------------------
    # place_futures_bracket
    # -------------------------------------------------------------------------

    async def place_futures_bracket(
        self,
        direction: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        args = {
            "direction": direction,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "quantity": quantity,
        }

        try:
            if self.mode in ("dry_run", "manual"):
                trade_id = f"DRY-{uuid.uuid4().hex[:8].upper()}"
                result = {
                    "trade_id": trade_id,
                    "order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
                    "status": "simulated",
                    "direction": direction,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "quantity": quantity,
                    "symbol": self._bot.get("symbol", "ES"),
                    "mode": self.mode,
                }
                logger.info(
                    "[%s] Would place %s bracket: entry=%.2f TP=%.2f SL=%.2f",
                    self.mode.upper(),
                    direction.upper(),
                    entry_price,
                    tp_price,
                    sl_price,
                )
            else:
                result = await self._tt.place_futures_bracket(
                    symbol=self._bot.get("symbol", "ES"),
                    direction=direction,
                    quantity=quantity,
                    entry_price=entry_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                )

            await self._log_tool("place_futures_bracket", args, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log_tool("place_futures_bracket", args, result, start)
            return result

    # -------------------------------------------------------------------------
    # close_position
    # -------------------------------------------------------------------------

    async def close_position(self, trade_id: str) -> Dict[str, Any]:
        start = time.monotonic()
        args = {"trade_id": trade_id}
        try:
            if self.mode in ("dry_run", "manual"):
                result = {"trade_id": trade_id, "status": "simulated_close", "mode": self.mode}
            else:
                result = await self._tt.close_futures_position(trade_id=trade_id)
            await self._log_tool("close_position", args, result, start)
            return result
        except Exception as exc:
            result = {"error": str(exc)}
            await self._log_tool("close_position", args, result, start)
            return result

    # -------------------------------------------------------------------------
    # Midline touch check (used by DecisionEngine — not a tool call)
    # -------------------------------------------------------------------------

    def check_midline_touch(
        self,
        current_price: float,
        midline: float,
        tolerance_pts: Optional[float] = None,
    ) -> bool:
        """Return True if current_price is within tolerance of midline."""
        tol = tolerance_pts if tolerance_pts is not None else self._strategy.get(
            "midline_tolerance_pts", 0.25
        )
        return abs(current_price - midline) <= tol

    # -------------------------------------------------------------------------
    # Internal logging
    # -------------------------------------------------------------------------

    async def _log_tool(
        self,
        name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        start_mono: float,
    ):
        elapsed_ms = int((time.monotonic() - start_mono) * 1000)
        try:
            await self._db.log_tool_call(
                {
                    "name": name,
                    "arguments": args,
                    "result": result,
                    "session_date": date.today(),
                    "duration_ms": elapsed_ms,
                }
            )
        except Exception:
            logger.debug("Tool call log failed for %s", name, exc_info=True)
