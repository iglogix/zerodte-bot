"""
TastyTrade API client for 0DTE Iron Condor Trading Bot.

Authentication: OAuth2 via tastytrade SDK Session(client_secret, refresh_token).
  - Sandbox:    Session(client_secret, refresh_token, is_test=True)
  - Production: Session(client_secret, refresh_token)
  - Access tokens last 15 min; SDK auto-refreshes transparently.
  - Refresh tokens never expire.

Credentials loaded from environment variables (see .env.example).
User-Agent header must be <product>/<version> — required by TastyTrade API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

_SANDBOX_URL = "https://api.cert.tastyworks.com"
_PRODUCTION_URL = "https://api.tastyworks.com"
_USER_AGENT = "0dte-iron-condor-bot/1.0"


VALID_MODES = {"dry_run", "manual", "sandbox", "live"}


class TastyTradeClient:
    """
    Async TastyTrade REST API client using OAuth2.

    mode:
      dry_run - no real API calls; market data and orders fully simulated
      manual  - live market data from production API; orders generated but NOT submitted
      sandbox - OAuth2 against api.cert.tastyworks.com (is_test=True)
      live    - OAuth2 against api.tastyworks.com
    """

    def __init__(
        self,
        provider_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        account_number: Optional[str] = None,
        mode: str = "dry_run",
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: {VALID_MODES}")
        self.mode = mode
        is_sandbox = mode == "sandbox"

        pfx = "TASTYTRADE_SANDBOX_" if is_sandbox else "TASTYTRADE_"
        self.provider_secret = provider_secret or os.environ.get(f"{pfx}CLIENT_SECRET", "")
        self.refresh_token = refresh_token or os.environ.get(f"{pfx}REFRESH_TOKEN", "")
        self.account_number = account_number or os.environ.get(
            f"{pfx}ACCOUNT_NUMBER",
            os.environ.get(f"{pfx}ACCOUNT", ""),  # fallback: TASTYTRADE_ACCOUNT
        )

        self.base_url = _SANDBOX_URL if is_sandbox else _PRODUCTION_URL
        self._sdk_session = None
        self._access_token: Optional[str] = None
        self._http: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    async def connect(self):
        """Authenticate and open HTTP session."""
        if self.mode == "dry_run":
            return
        self._http = aiohttp.ClientSession(headers={"User-Agent": _USER_AGENT})
        await self._authenticate()

    async def disconnect(self):
        if self._http:
            await self._http.close()
            self._http = None

    # -------------------------------------------------------------------------
    # Authentication — OAuth2
    # -------------------------------------------------------------------------

    async def _authenticate(self):
        """
        Authenticate using OAuth2. Tries tastytrade SDK first;
        falls back to raw HTTP refresh-token exchange.
        """
        try:
            await self._authenticate_via_sdk()
        except ImportError:
            logger.warning("tastytrade SDK not installed — using raw OAuth2 token exchange")
            await self._authenticate_via_http()

    async def _authenticate_via_sdk(self):
        """Use OAuthSession from the tastytrade SDK (v10+)."""
        from tastytrade import OAuthSession  # type: ignore

        is_test = self.mode == "sandbox"
        loop = asyncio.get_event_loop()
        self._sdk_session = await loop.run_in_executor(
            None,
            lambda: OAuthSession(self.provider_secret, self.refresh_token, is_test=is_test),
        )
        self._access_token = self._sdk_session.session_token
        logger.info(
            "TastyTrade OAuth2 authenticated via OAuthSession (mode=%s, is_test=%s)",
            self.mode, is_test,
        )

    async def _authenticate_via_http(self):
        """
        Raw OAuth2 refresh-token exchange when SDK is not installed.
        POST /oauth/token with grant_type=refresh_token.
        """
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.provider_secret,
        }
        async with self._http.post(
            f"{self.base_url}/oauth/token",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self._access_token = data["access_token"]
            logger.info("TastyTrade OAuth2 authenticated via HTTP token exchange (mode=%s)", self.mode)

    async def get_session(self):
        """Return the SDK session object (for AlertStreamer / Account.get)."""
        return self._sdk_session

    async def get_session_token(self) -> str:
        """Return the current access token (auto-refreshed by SDK)."""
        if self._sdk_session is not None:
            return self._sdk_session.session_token
        return self._access_token or ""

    async def get_account_number(self) -> str:
        return self.account_number

    def _headers(self) -> Dict[str, str]:
        token = self._access_token or ""
        auth = f"Bearer {token}" if token else ""
        return {
            "Authorization": auth,
            "Content-Type": "application/json",
        }

    # -------------------------------------------------------------------------
    # Market data
    # -------------------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol.
        Uses /market-metrics for index symbols (SPX, SPXW);
        falls back to /market-data/quotes/{symbol} for equities/ETFs.
        """
        async with self._http.get(
            f"{self.base_url}/market-metrics",
            headers=self._headers(),
            params={"symbols": symbol},
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                items = data.get("data", {}).get("items", [])
                if items:
                    item = items[0]
                    price = float(
                        item.get("last", None)
                        or item.get("implied-volatility-index", 0)
                    )
                    return {
                        "symbol": symbol,
                        "price": price,
                        "bid": 0.0,
                        "ask": 0.0,
                        "timestamp": datetime.now().isoformat(),
                    }

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
