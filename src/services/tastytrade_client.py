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

        # In sandbox mode, market data (quotes, chains) comes from production
        # since the cert environment has no live data feed.
        self._market_data_url = _PRODUCTION_URL if is_sandbox else self.base_url
        self._market_http: Optional[aiohttp.ClientSession] = None
        self._market_token: Optional[str] = None
        self._market_sdk_session = None  # production OAuthSession for DXLink in sandbox

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

        if self.mode == "sandbox":
            await self._connect_market_data_session()

    async def disconnect(self):
        if self._http:
            await self._http.close()
            self._http = None
        if self._market_http:
            await self._market_http.close()
            self._market_http = None

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

    async def _connect_market_data_session(self):
        """
        Sandbox only: open a second session against production for read-only
        market data (quotes, option chains). Uses production credentials.
        """
        prod_secret = os.environ.get("TASTYTRADE_CLIENT_SECRET", "")
        prod_token = os.environ.get("TASTYTRADE_REFRESH_TOKEN", "")
        if not prod_secret or not prod_token:
            logger.warning(
                "Sandbox market data: no production credentials found — "
                "set TASTYTRADE_CLIENT_SECRET + TASTYTRADE_REFRESH_TOKEN"
            )
            return
        try:
            from tastytrade import OAuthSession  # type: ignore
            loop = asyncio.get_event_loop()
            prod_session = await loop.run_in_executor(
                None,
                lambda: OAuthSession(prod_secret, prod_token, is_test=False),
            )
            self._market_sdk_session = prod_session
            self._market_token = prod_session.session_token
            self._market_http = aiohttp.ClientSession(headers={"User-Agent": _USER_AGENT})
            logger.info("Sandbox market data session opened against production API")
        except Exception as exc:
            logger.warning("Could not open sandbox market data session: %s", exc)

    def _market_headers(self) -> Dict[str, str]:
        """Headers for read-only market data requests (production in sandbox).
        Always reads the live token from the SDK session so auto-refresh is respected.
        """
        if self._market_sdk_session is not None:
            token = self._market_sdk_session.session_token or ""
        elif self._sdk_session is not None:
            token = self._sdk_session.session_token or ""
        else:
            token = self._market_token or self._access_token or ""
        auth = f"Bearer {token}" if token else ""
        return {
            "Authorization": auth,
            "Content-Type": "application/json",
        }

    def _market_http_and_url(self):
        """Return (http_session, base_url) that are guaranteed to match.

        In sandbox mode, if the production market data session opened successfully
        (_market_http is set), return the production http+url pair.
        Otherwise fall back to the sandbox http+url so the token always matches
        the endpoint — preventing a 401 from sending a sandbox token to production.
        """
        if self._market_http is not None:
            return self._market_http, self._market_data_url
        if self.mode == "sandbox" and self._market_data_url != self.base_url:
            logger.warning(
                "Production market data session unavailable — falling back to sandbox URL. "
                "Set TASTYTRADE_CLIENT_SECRET + TASTYTRADE_REFRESH_TOKEN for live market data in sandbox mode."
            )
        return self._http, self.base_url

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
        if self._sdk_session is not None:
            token = self._sdk_session.session_token or ""
        else:
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
        Get real-time quote via DXLinkStreamer (works for SPX and all symbols).
        In sandbox mode uses the production session (cert has no live feed).
        """
        try:
            return await self._get_quote_via_dxlink(symbol)
        except Exception as exc:
            logger.warning("DXLink quote failed for %s: %s — falling back to REST", symbol, exc)
            return await self._get_quote_via_rest(symbol)

    async def _get_quote_via_dxlink(self, symbol: str) -> Dict[str, Any]:
        """Fetch a single quote snapshot via DXLinkStreamer."""
        from tastytrade import DXLinkStreamer  # type: ignore
        from tastytrade.dxfeed import Quote    # type: ignore

        session = self._market_sdk_session or self._sdk_session
        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Quote, [symbol])
            quote = await asyncio.wait_for(streamer.get_event(Quote), timeout=10)
            bid = float(quote.bid_price or 0)
            ask = float(quote.ask_price or 0)
            price = (bid + ask) / 2 if bid and ask else bid or ask
            return {
                "symbol": symbol,
                "price": price,
                "bid": bid,
                "ask": ask,
                "timestamp": datetime.now().isoformat(),
            }

    async def _get_quote_via_rest(self, symbol: str) -> Dict[str, Any]:
        """REST fallback for quotes."""
        http, base = self._market_http_and_url()
        hdrs = self._market_headers()
        async with http.get(
            f"{base}/market-data/quotes/{symbol}",
            headers=hdrs,
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

    async def get_future_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get futures quote with mark/last/bid/ask fields.
        Alias for get_quote with additional fields for compatibility.
        """
        quote = await self.get_quote(symbol)
        # Add mark and last fields expected by OrbMarketData
        quote["mark"] = quote.get("price", 0)
        quote["last"] = quote.get("price", 0)
        return quote

    async def get_candles(
        self,
        symbol: str,
        interval: str = "5m",
        from_time: Optional[datetime] = None,
        count: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV candles for a symbol via DXLink.

        Args:
            symbol: DXLink symbol (e.g., "/ESH5:XCME")
            interval: Candle interval ("1m", "5m", "15m", etc.)
            from_time: Start time for candles (optional)
            count: Number of candles to return

        Returns:
            List of candle dicts with open/high/low/close/volume/timestamp
        """
        try:
            from tastytrade import DXLinkStreamer  # type: ignore
            from tastytrade.dxfeed import Candle   # type: ignore

            session = self._market_sdk_session or self._sdk_session
            candles: List[Dict[str, Any]] = []

            # Calculate from_time if not provided
            if from_time is None:
                # Parse interval to minutes
                interval_mins = 5
                if interval == "1m":
                    interval_mins = 1
                elif interval == "15m":
                    interval_mins = 15
                elif interval == "30m":
                    interval_mins = 30
                elif interval == "1h":
                    interval_mins = 60

                from_time = datetime.now().replace(second=0, microsecond=0)
                from_time = from_time - __import__("datetime").timedelta(
                    minutes=count * interval_mins + 10
                )

            # Collect all candles from the stream, filter by from_time
            from_ts = from_time.timestamp() if from_time else 0
            all_candles: List[Dict[str, Any]] = []

            async with DXLinkStreamer(session) as streamer:
                await streamer.subscribe_candle([symbol], interval, from_time)
                try:
                    # Collect candles for up to 8 seconds
                    end_time = asyncio.get_event_loop().time() + 8
                    while asyncio.get_event_loop().time() < end_time:
                        c = await asyncio.wait_for(streamer.get_event(Candle), timeout=2)
                        if not c.open or not c.time:
                            continue
                        candle_ts = c.time / 1000
                        # Only include candles at or after from_time
                        if candle_ts >= from_ts:
                            all_candles.append({
                                "open": float(c.open),
                                "high": float(c.high),
                                "low": float(c.low),
                                "close": float(c.close),
                                "volume": int(c.volume or 0),
                                "timestamp": datetime.fromtimestamp(candle_ts).isoformat(),
                                "_ts": candle_ts,
                            })
                except asyncio.TimeoutError:
                    pass

            # Sort by timestamp, deduplicate by timestamp, return requested count
            all_candles.sort(key=lambda x: x["_ts"])
            seen = set()
            candles = []
            for c in all_candles:
                if c["_ts"] not in seen:
                    seen.add(c["_ts"])
                    del c["_ts"]
                    candles.append(c)
            return candles[:count]

        except Exception as exc:
            logger.warning("get_candles failed for %s: %s", symbol, exc)
            return []

    async def get_option_chain(self, symbol: str) -> List[Dict[str, Any]]:
        """Get 0DTE option chain for a symbol with Greeks.
        Fetches symbol list from REST, then enriches near-ATM strikes
        with live bid/ask/delta via DXLink.
        In sandbox mode, reads from production API (cert has no live feed).
        """
        today = date.today().strftime("%Y-%m-%d")
        http, base = self._market_http_and_url()
        async with http.get(
            f"{base}/option-chains/{symbol}/nested",
            headers=self._market_headers(),
            params={"expiration-date": today},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        options = []
        for chain in data.get("data", {}).get("items", []):
            for expiration in chain.get("expirations", []):
                exp_date = expiration.get("expiration-date", "")
                if exp_date != today:
                    continue
                for strike in expiration.get("strikes", []):
                    strike_price = float(strike.get("strike-price", 0))
                    for side in ["call", "put"]:
                        sym = strike.get(side, "").strip()
                        streamer_sym = strike.get(f"{side}-streamer-symbol", "")
                        if not sym:
                            continue
                        options.append({
                            "symbol": sym,
                            "streamer_symbol": streamer_sym,
                            "strike": strike_price,
                            "option_type": side,
                            "expiration": exp_date,
                            "bid": 0.0,
                            "ask": 0.0,
                            "last": 0.0,
                            "volume": 0,
                            "open_interest": 0,
                            "delta": 0.0,
                            "gamma": 0.0,
                            "theta": 0.0,
                            "vega": 0.0,
                            "iv": 0.0,
                        })

        if not options:
            return options

        # Get SPX price for near-ATM filtering, then enrich with Greeks
        spx_price = 0.0
        try:
            q = await self._get_quote_via_dxlink("SPX")
            spx_price = q.get("price", 0.0)
        except Exception:
            pass
        try:
            options = await self._enrich_option_chain_greeks(options, spx_price=spx_price)
        except Exception as exc:
            logger.warning("Option chain Greeks enrichment failed: %s", exc)

        return options

    async def _enrich_option_chain_greeks(
        self, options: List[Dict[str, Any]], spx_price: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Fetch live bid/ask/delta for near-ATM options via a single DXLink session.
        Subscribes to both Quote and Greeks simultaneously and collects for 6s.
        """
        from tastytrade import DXLinkStreamer  # type: ignore
        from tastytrade.dxfeed import Quote, Greeks  # type: ignore

        # Filter to near-ATM strikes only (±200 pts) to limit subscription size
        if spx_price:
            near_atm = [o for o in options if abs(o["strike"] - spx_price) <= 200]
        else:
            # Pick middle 100 strikes as a heuristic
            mid = len(options) // 2
            near_atm = options[max(0, mid - 50): mid + 50]

        if not near_atm:
            return options

        streamer_syms = [o["streamer_symbol"] for o in near_atm if o["streamer_symbol"]]
        if not streamer_syms:
            return options

        session = self._market_sdk_session or self._sdk_session
        quotes_map: Dict[str, Any] = {}
        greeks_map: Dict[str, Any] = {}

        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Quote, streamer_syms)
            await streamer.subscribe(Greeks, streamer_syms)

            needed_q = set(streamer_syms)
            needed_g = set(streamer_syms)
            deadline = asyncio.get_event_loop().time() + 8.0

            while (needed_q or needed_g) and asyncio.get_event_loop().time() < deadline:
                left = deadline - asyncio.get_event_loop().time()
                if left <= 0:
                    break
                if needed_q:
                    try:
                        q = await asyncio.wait_for(streamer.get_event(Quote), timeout=min(1.5, left))
                        quotes_map[q.event_symbol] = q
                        needed_q.discard(q.event_symbol)
                        continue
                    except asyncio.TimeoutError:
                        needed_q.clear()
                if needed_g:
                    try:
                        g = await asyncio.wait_for(streamer.get_event(Greeks), timeout=min(1.5, left))
                        greeks_map[g.event_symbol] = g
                        needed_g.discard(g.event_symbol)
                    except asyncio.TimeoutError:
                        needed_g.clear()

        for opt in options:
            ssym = opt.get("streamer_symbol", "")
            q = quotes_map.get(ssym)
            g = greeks_map.get(ssym)
            if q:
                bid = float(q.bid_price or 0)
                ask = float(q.ask_price or 0)
                opt["bid"] = bid
                opt["ask"] = ask
                opt["last"] = (bid + ask) / 2 if bid and ask else bid or ask
            if g:
                opt["delta"] = float(g.delta or 0)
                opt["gamma"] = float(g.gamma or 0)
                opt["theta"] = float(g.theta or 0)
                opt["vega"] = float(g.vega or 0)
                opt["iv"] = float(g.volatility or 0)

        return options

    async def get_5min_candles(self, symbol: str, count: int = 6) -> List[Dict[str, Any]]:
        """
        Fetch the last N completed 5-minute candles for a symbol via DXLink.
        Uses production session in sandbox mode.
        """
        try:
            from tastytrade import DXLinkStreamer  # type: ignore
            from tastytrade.dxfeed import Candle   # type: ignore

            session = self._market_sdk_session or self._sdk_session
            candles: List[Dict[str, Any]] = []

            # Request candles from count*5 minutes ago to cover enough history
            from_dt = datetime.now().replace(second=0, microsecond=0)
            from_dt = from_dt.replace(minute=(from_dt.minute // 5) * 5)
            import datetime as dt_module
            from_dt = from_dt - dt_module.timedelta(minutes=count * 5 + 10)

            async with DXLinkStreamer(session) as streamer:
                await streamer.subscribe_candle([symbol], "5m", from_dt)
                try:
                    while len(candles) < count:
                        c = await asyncio.wait_for(streamer.get_event(Candle), timeout=8)
                        if not c.open:
                            continue
                        candles.append({
                            "timestamp": datetime.fromtimestamp(c.time / 1000).isoformat() if c.time else datetime.now().isoformat(),
                            "open": float(c.open or 0),
                            "high": float(c.high or 0),
                            "low": float(c.low or 0),
                            "close": float(c.close or 0),
                            "volume": float(c.volume or 0),
                        })
                except asyncio.TimeoutError:
                    pass

            candles.sort(key=lambda x: x["timestamp"])
            return candles[-count:]
        except Exception as exc:
            logger.warning("DXLink candles failed for %s: %s", symbol, exc)
            return []

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
