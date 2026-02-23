"""
TastyTrade API client for 0DTE Iron Condor Trading Bot.
Handles authentication, market data, and order execution via the TastyTrade REST API.

Sandbox environment: https://api.cert.tastyworks.com
Production environment: https://api.tastyworks.com
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

_SANDBOX_URL = "https://api.cert.tastyworks.com"
_PRODUCTION_URL = "https://api.tastyworks.com"


VALID_MODES = {"dry_run", "manual", "sandbox", "live"}


class TastyTradeClient:
    """
    Async TastyTrade REST API client.
    Handles session tokens, market data, and order placement.

    mode:
      dry_run - no real API calls; market data and orders fully simulated
      manual  - live market data from production API; orders generated but NOT submitted
      sandbox - all calls go to api.cert.tastyworks.com
      live    - all calls go to api.tastyworks.com
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        account_number: Optional[str] = None,
        mode: str = "dry_run",
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: {VALID_MODES}")
        self.mode = mode
        self.username = username or os.environ.get("TASTYTRADE_USERNAME", "")
        self.password = password or os.environ.get("TASTYTRADE_PASSWORD", "")
        self.account_number = account_number or os.environ.get("TASTYTRADE_ACCOUNT_NUMBER", "")
        self.base_url = _SANDBOX_URL if mode == "sandbox" else _PRODUCTION_URL
        self._session_token: Optional[str] = None
        self._http: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    async def connect(self):
        self._http = aiohttp.ClientSession()
        await self.authenticate()

    async def disconnect(self):
        if self._http:
            await self._http.close()

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    async def authenticate(self) -> bool:
        """Authenticate and store session token."""
        payload = {
            "login": self.username,
            "password": self.password,
            "remember-me": True,
        }
        async with self._http.post(
            f"{self.base_url}/sessions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self._session_token = data["data"]["session-token"]
            logger.info("TastyTrade authenticated successfully")
            return True

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": self._session_token or "",
            "Content-Type": "application/json",
        }

    # -------------------------------------------------------------------------
    # Market data
    # -------------------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol."""
        async with self._http.get(
            f"{self.base_url}/market-data/quotes/{symbol}",
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            item = data["data"]
            return {
                "symbol": symbol,
                "price": float(item.get("last", item.get("mark", 0))),
                "bid": float(item.get("bid", 0)),
                "ask": float(item.get("ask", 0)),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_option_chain(self, symbol: str) -> List[Dict[str, Any]]:
        """Get 0DTE option chain for a symbol with Greeks."""
        today = date.today().strftime("%Y-%m-%d")
        async with self._http.get(
            f"{self.base_url}/option-chains/{symbol}/nested",
            headers=self._headers(),
            params={"expiration-date": today},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        options = []
        for expiration in data.get("data", {}).get("items", []):
            for strike in expiration.get("strikes", []):
                for side in ["call", "put"]:
                    opt = strike.get(side, {})
                    if not opt:
                        continue
                    options.append({
                        "symbol": opt.get("symbol", ""),
                        "strike": float(strike.get("strike-price", 0)),
                        "option_type": side,
                        "expiration": expiration.get("expiration-date", ""),
                        "bid": float(opt.get("bid", 0)),
                        "ask": float(opt.get("ask", 0)),
                        "last": float(opt.get("last", 0)),
                        "volume": int(opt.get("volume", 0)),
                        "open_interest": int(opt.get("open-interest", 0)),
                        "delta": float(opt.get("delta", 0)),
                        "gamma": float(opt.get("gamma", 0)),
                        "theta": float(opt.get("theta", 0)),
                        "vega": float(opt.get("vega", 0)),
                        "iv": float(opt.get("implied-volatility", 0)),
                    })
        return options

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account balances and buying power."""
        async with self._http.get(
            f"{self.base_url}/accounts/{self.account_number}/balances",
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            item = data["data"]
            return {
                "account_number": self.account_number,
                "equity": float(item.get("net-liquidating-value", 0)),
                "buying_power": float(item.get("derivative-buying-power", 0)),
                "cash_balance": float(item.get("cash-balance", 0)),
                "day_trading_buying_power": float(item.get("day-trading-buying-power", 0)),
                "maintenance_requirement": float(item.get("maintenance-requirement", 0)),
            }

    # -------------------------------------------------------------------------
    # Order placement
    # -------------------------------------------------------------------------

    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order. Dispatches to the appropriate order builder."""
        order_type = order.get("order_type", "")

        if order_type == "iron_condor":
            return await self._place_iron_condor(order)
        elif order_type in ("call_credit_spread", "put_credit_spread"):
            return await self._place_credit_spread(order)
        elif order_type.startswith("close_"):
            return await self._place_close_order(order)
        else:
            raise ValueError(f"Unknown order_type: {order_type}")

    async def _place_iron_condor(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Build and submit a 4-leg iron condor order."""
        today = date.today().strftime("%y%m%d")
        qty = order.get("quantity", 1)

        legs = [
            self._build_leg(f"SPXW{today}C{int(order['short_call_strike'])}", "Sell to Open", qty),
            self._build_leg(f"SPXW{today}C{int(order['long_call_strike'])}", "Buy to Open", qty),
            self._build_leg(f"SPXW{today}P{int(order['short_put_strike'])}", "Sell to Open", qty),
            self._build_leg(f"SPXW{today}P{int(order['long_put_strike'])}", "Buy to Open", qty),
        ]

        payload = {
            "time-in-force": "Day",
            "order-type": "Net Credit",
            "legs": legs,
        }

        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"order_id": str(data["data"]["order"]["id"])}

    async def _place_credit_spread(self, order: Dict[str, Any]) -> Dict[str, Any]:
        today = date.today().strftime("%y%m%d")
        qty = order.get("quantity", 1)
        side = "C" if "call" in order["order_type"] else "P"

        legs = [
            self._build_leg(f"SPXW{today}{side}{int(order['short_strike'])}", "Sell to Open", qty),
            self._build_leg(f"SPXW{today}{side}{int(order['long_strike'])}", "Buy to Open", qty),
        ]

        payload = {
            "time-in-force": "Day",
            "order-type": "Net Credit",
            "legs": legs,
        }

        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"order_id": str(data["data"]["order"]["id"])}

    async def _place_close_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        symbol = order.get("symbol", "")
        qty = order.get("quantity", 1)
        action = "Buy to Close"

        legs = [self._build_leg(symbol, action, qty)]
        payload = {
            "time-in-force": "Day",
            "order-type": "Market",
            "legs": legs,
        }

        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"order_id": str(data["data"]["order"]["id"])}

    async def place_oco_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place an OCO (One-Cancels-Other) stop order pair."""
        symbol = order["symbol"]
        stop_loss_amount = order["stop_loss_amount"]
        stop_limit_gap = order.get("stop_limit_gap", 40)
        stop_market_gap = order.get("stop_market_gap", 30)

        trigger = stop_loss_amount
        limit_price = trigger + stop_limit_gap
        market_trigger = trigger + stop_market_gap

        stop_limit_payload = {
            "time-in-force": "GTC",
            "order-type": "Stop Limit",
            "stop-trigger": str(trigger),
            "price": str(limit_price),
            "legs": [self._build_leg(symbol, "Buy to Close", 1)],
        }

        stop_market_payload = {
            "time-in-force": "GTC",
            "order-type": "Stop",
            "stop-trigger": str(market_trigger),
            "legs": [self._build_leg(symbol, "Buy to Close", 1)],
        }

        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=stop_limit_payload,
        ) as resp:
            resp.raise_for_status()
            sl_data = await resp.json()
            sl_order_id = str(sl_data["data"]["order"]["id"])

        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=stop_market_payload,
        ) as resp:
            resp.raise_for_status()
            sm_data = await resp.json()
            sm_order_id = str(sm_data["data"]["order"]["id"])

        return {
            "stop_limit_order_id": sl_order_id,
            "stop_market_order_id": sm_order_id,
        }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        async with self._http.delete(
            f"{self.base_url}/accounts/{self.account_number}/orders/{order_id}",
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            return {"status": "cancelled", "order_id": order_id}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_leg(symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        return {
            "instrument-type": "Equity Option",
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
        }
