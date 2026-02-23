"""
Tool handler implementations for the LLM agent.
Each method corresponds to a tool the LLM can call.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolHandlers:
    """Handles all tool calls dispatched by the trading agent."""

    def __init__(
        self,
        cache=None,
        api_client=None,
        db=None,
        mode: Optional[str] = None,
        dry_run: Optional[bool] = None,
        stop_limit_gap: int = 40,
        stop_market_gap: int = 30,
    ):
        if mode is not None:
            self.mode = mode
        elif dry_run is not None:
            self.mode = "dry_run" if dry_run else "live"
        else:
            self.mode = "dry_run"
        self.cache = cache
        self.api_client = api_client
        self.db = db
        self.dry_run = self.mode == "dry_run"
        self.manual = self.mode == "manual"
        self.stop_limit_gap = stop_limit_gap
        self.stop_market_gap = stop_market_gap

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call by name."""
        handler = getattr(self, tool_name, None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return await handler(**arguments)
        except Exception as exc:
            logger.exception("Tool %s raised an error", tool_name)
            return {"error": str(exc)}

    # -------------------------------------------------------------------------
    # Market data tools
    # -------------------------------------------------------------------------

    def _is_simulated(self) -> bool:
        """True only in dry_run — market data is also simulated."""
        return self.mode == "dry_run"

    def _block_submission(self) -> bool:
        """True in dry_run and manual — orders must not be submitted."""
        return self.mode in ("dry_run", "manual")

    def _manual_order_result(self, tool_name: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Build a manual-mode result: order fully generated but not submitted."""
        logger.info(
            "[MANUAL] Order generated (NOT submitted) — %s: %s",
            tool_name,
            details,
        )
        return {
            **details,
            "manual": True,
            "submitted": False,
            "note": "Order generated from live data but NOT submitted (manual mode)",
        }

    async def get_spx_price(self) -> Dict[str, Any]:
        """Return current SPX price, using cache when valid."""
        try:
            cached = await self.cache.get_cached_quote("SPX", max_age_seconds=60)
            if cached is not None:
                return cached
            data = await self.api_client.get_quote("SPX")
            await self.cache.cache_quote("SPX", data)
            return data
        except Exception as exc:
            return {"error": str(exc)}

    async def get_option_chain(
        self,
        expiration: str = "0DTE",
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return 0DTE option chain, optionally filtered by delta range."""
        chain = await self.cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)
        if min_delta is not None or max_delta is not None:
            filtered = []
            for opt in chain:
                abs_delta = abs(opt.get("delta", 0))
                if min_delta is not None and abs_delta < min_delta:
                    continue
                if max_delta is not None and abs_delta > max_delta:
                    continue
                filtered.append(opt)
            return filtered
        return chain

    async def get_5min_candles(self, count: int = 6) -> List[Dict[str, Any]]:
        """Return the last N 5-minute SPX candles."""
        return await self.cache.get_5min_candles("SPX", count=count)

    async def get_account_info(self) -> Dict[str, Any]:
        """Return current account info."""
        return await self.cache.get_account_info_with_fallback(max_age_seconds=60)

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return all open iron condor positions."""
        trades = await self.db.get_open_trades()
        return trades if trades else []

    async def check_buying_power(
        self,
        required_amount: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check available buying power against 50% account limit."""
        account = await self.cache.get_cached_account_info(max_age_seconds=60)
        if account is None:
            account = await self.cache.get_account_info_with_fallback()

        equity = account.get("equity", 0)
        max_allowed = equity * 0.5
        available = account.get("buying_power") or max_allowed

        result = {
            "available": available,
            "max_allowed": max_allowed,
            "equity": equity,
        }

        if required_amount is not None:
            if required_amount > max_allowed:
                result["allowed"] = False
                result["reason"] = (
                    f"Required ${required_amount:,.0f} exceeds 50% buying power limit "
                    f"(${max_allowed:,.0f})"
                )
            else:
                result["allowed"] = True

        return result

    # -------------------------------------------------------------------------
    # Order execution tools
    # -------------------------------------------------------------------------

    async def place_iron_condor(
        self,
        short_call_strike: float,
        long_call_strike: float,
        short_put_strike: float,
        long_put_strike: float,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Place a full 4-leg iron condor. Validates strike order first."""
        if long_call_strike <= short_call_strike:
            return {
                "error": "Invalid strike order: long_call_strike must be > short_call_strike"
            }
        if short_put_strike >= short_call_strike:
            return {
                "error": "Invalid strike order: short_put_strike must be < short_call_strike"
            }
        if long_put_strike >= short_put_strike:
            return {
                "error": "Invalid strike order: long_put_strike must be < short_put_strike"
            }

        if self._block_submission():
            details = {
                "short_call_strike": short_call_strike,
                "long_call_strike": long_call_strike,
                "short_put_strike": short_put_strike,
                "long_put_strike": long_put_strike,
                "quantity": quantity,
            }
            if self.dry_run:
                order_id = f"DRY-IC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                logger.info("[DRY RUN] Simulated iron condor: %s", order_id)
                return {"order_id": order_id, "dry_run": True, "simulated": True, **details}
            return self._manual_order_result("place_iron_condor", details)

        return await self.api_client.place_order({
            "order_type": "iron_condor",
            "short_call_strike": short_call_strike,
            "long_call_strike": long_call_strike,
            "short_put_strike": short_put_strike,
            "long_put_strike": long_put_strike,
            "quantity": quantity,
        })

    async def place_credit_spread(
        self,
        spread_type: str,
        short_strike: float,
        long_strike: float,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Place a single credit spread (call or put side)."""
        if self._block_submission():
            details = {"spread_type": spread_type, "short_strike": short_strike, "long_strike": long_strike, "quantity": quantity}
            if self.dry_run:
                return {"order_id": f"DRY-CS-{datetime.now().strftime('%Y%m%d%H%M%S')}", "dry_run": True, "simulated": True}
            return self._manual_order_result("place_credit_spread", details)

        return await self.api_client.place_order({
            "order_type": f"{spread_type}_credit_spread",
            "short_strike": short_strike,
            "long_strike": long_strike,
            "quantity": quantity,
        })

    async def place_stop_order(
        self,
        symbol: str,
        stop_loss_amount: float,
        stop_limit_gap: Optional[int] = None,
        stop_market_gap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place OCO stop-loss: stop-limit + stop-market (gaps from config)."""
        stop_limit_gap = stop_limit_gap if stop_limit_gap is not None else self.stop_limit_gap
        stop_market_gap = stop_market_gap if stop_market_gap is not None else self.stop_market_gap
        if self._block_submission():
            details = {
                "symbol": symbol,
                "stop_loss_amount": stop_loss_amount,
                "stop_limit_gap": stop_limit_gap,
                "stop_market_gap": stop_market_gap,
                "stop_limit_trigger": stop_loss_amount,
                "stop_limit_price": stop_loss_amount + stop_limit_gap,
                "stop_market_trigger": stop_loss_amount + stop_market_gap,
            }
            if self.dry_run:
                return {
                    "stop_limit_order_id": f"DRY-SL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "stop_market_order_id": f"DRY-SM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "dry_run": True,
                    "simulated": True,
                }
            return self._manual_order_result("place_stop_order", details)

        return await self.api_client.place_oco_order({
            "symbol": symbol,
            "stop_loss_amount": stop_loss_amount,
            "stop_limit_gap": stop_limit_gap,
            "stop_market_gap": stop_market_gap,
        })

    async def close_position(
        self,
        symbol: Optional[str] = None,
        trade_id: Optional[str] = None,
        quantity: int = 1,
        order_type: str = "market",
        close_all: bool = False,
    ) -> Dict[str, Any]:
        """Close a position or specific leg."""
        if self._block_submission():
            details = {"symbol": symbol, "trade_id": trade_id, "quantity": quantity, "order_type": order_type, "close_all": close_all}
            if self.dry_run:
                return {"order_id": f"DRY-CL-{datetime.now().strftime('%Y%m%d%H%M%S')}", "dry_run": True, "simulated": True}
            return self._manual_order_result("close_position", details)

        if close_all and trade_id:
            return await self.api_client.place_order({
                "order_type": "close_trade",
                "trade_id": trade_id,
                "close_all": True,
            })

        return await self.api_client.place_order({
            "order_type": f"close_{order_type}",
            "symbol": symbol,
            "quantity": quantity,
        })

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order."""
        if self._block_submission():
            if self.dry_run:
                return {"status": "cancelled", "order_id": order_id, "dry_run": True}
            return self._manual_order_result("cancel_order", {"order_id": order_id, "status": "would_cancel"})

        return await self.api_client.cancel_order(order_id)
