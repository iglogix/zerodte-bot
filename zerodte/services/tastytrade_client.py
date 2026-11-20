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
import json
import logging
import os
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import aiohttp
import httpx

from zerodte.backtest.osi_symbol import format_osi

logger = logging.getLogger(__name__)

_SANDBOX_URL = "https://api.cert.tastyworks.com"
_PRODUCTION_URL = "https://api.tastyworks.com"
_USER_AGENT = "0dte-iron-condor-bot/1.0"
# Plain float, not httpx.Timeout: the tastytrade SDK (>=13.2.2) forwards this
# into its httpx2 fork, which treats a foreign Timeout object as a raw number
# and crashes in anyio. Both libraries expand a float to all four timeout
# categories, so semantics are unchanged.
_TT_HTTP_TIMEOUT = 20.0


VALID_MODES = {"dry_run", "manual", "sandbox", "live"}


def _is_timeout_error(exc: BaseException) -> bool:
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException))


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

        # Persistent DXLink quote streamer (replaces per-call open/close).
        # The streamer is an anyio task group, so it must be hosted by its own
        # dedicated OWNER task — entering it from a caller's task means any
        # internal failure cancels that caller (this killed the scheduled
        # engine mid-session on 2026-06-11). See _quote_streamer_owner().
        self._quote_streamer = None
        self._quote_owner_task: Optional[asyncio.Task] = None
        self._quote_ready: Optional[asyncio.Event] = None
        self._quote_shutdown = False
        self._latest_quotes: Dict[str, Dict[str, Any]] = {}
        self._quote_symbols: set = set()
        self._quote_streamer_lock = asyncio.Lock()
        # Symbols that should bypass DXLink and go straight to REST
        # (futures options with spaces don't stream via DXLink)
        self._dxlink_rest_only: set = set()
        # Adaptive REST fallback: a symbol whose DXLink stream goes silent (e.g.
        # an expiring front-month future the day before roll) would otherwise
        # eat a full 5s timeout on EVERY poll and flood the log. After
        # N consecutive DXLink timeouts we pin the symbol to REST for a cooldown,
        # then re-probe DXLink once the cooldown elapses (self-healing).
        self._dxlink_fail_counts: Dict[str, int] = {}
        self._dxlink_rest_until: Dict[str, float] = {}  # symbol -> monotonic deadline
        self._dxlink_max_consec_fails = 3
        self._dxlink_rest_cooldown_s = 300.0

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

    async def refresh_session(self):
        """Re-authenticate to get a fresh session token (tokens expire after 15 min)."""
        if self.mode == "dry_run":
            return
        try:
            await self._authenticate()
            if self.mode == "sandbox":
                await self._connect_market_data_session()
            logger.info("TastyTrade session refreshed successfully")
        except Exception as exc:
            logger.error("TastyTrade session refresh failed: %s", exc)

    async def disconnect(self):
        await self._close_quote_streamer()
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
        """Use the tastytrade SDK's OAuth2 session flow."""
        from tastytrade import Session as TTSession  # type: ignore

        is_test = self.mode == "sandbox"
        if not self.provider_secret or not self.refresh_token:
            raise RuntimeError(
                "TastyTrade authentication failed — set "
                "TASTYTRADE_CLIENT_SECRET + TASTYTRADE_REFRESH_TOKEN in .env"
            )

        self._sdk_session = TTSession(
            self.provider_secret,
            self.refresh_token,
            is_test=is_test,
            timeout=_TT_HTTP_TIMEOUT,
        )
        await self._sdk_session.refresh(force=True)
        self._access_token = self._sdk_session.session_token
        logger.info("TastyTrade authenticated via SDK OAuth2 (mode=%s)", self.mode)

    async def _connect_market_data_session(self):
        """
        Sandbox only: open a second session against production for read-only
        market data (quotes, option chains). Uses production credentials.
        """
        from tastytrade import Session as TTSession  # type: ignore

        prod_secret = os.environ.get("TASTYTRADE_CLIENT_SECRET", "")
        prod_token = os.environ.get("TASTYTRADE_REFRESH_TOKEN", "")
        if not prod_secret or not prod_token:
            logger.warning(
                "Sandbox market data: no credentials found — "
                "set TASTYTRADE_CLIENT_SECRET + TASTYTRADE_REFRESH_TOKEN"
            )
            return
        try:
            prod_session = TTSession(
                prod_secret,
                prod_token,
                is_test=False,
                timeout=_TT_HTTP_TIMEOUT,
            )
            await prod_session.refresh(force=True)
            self._market_sdk_session = prod_session
            self._market_token = prod_session.session_token
            self._market_http = aiohttp.ClientSession(headers={"User-Agent": _USER_AGENT})
            logger.info("Sandbox market data session opened via SDK OAuth2")
        except Exception as exc:
            logger.error("Could not open sandbox market data session: %s", exc)

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

    async def _fetch_nested_future_option_chain_data(self, product_code: str) -> Dict[str, Any]:
        """Fetch TT futures option chain as raw JSON, bypassing SDK model parsing."""
        endpoint = f"/futures-option-chains/{product_code.lstrip('/')}/nested"
        session = self._market_sdk_session or self._sdk_session

        if session is not None:
            data = await session._get(endpoint)
            return data.get("data", data) if isinstance(data, dict) else data

        http, base = self._market_http_and_url()
        async with http.get(f"{base}{endpoint}", headers=self._market_headers()) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("data", data) if isinstance(data, dict) else data

    async def _authenticate_via_http(self):
        """
        Raw OAuth2 refresh-token exchange when SDK is not installed.
        POST /oauth/token with grant_type=refresh_token.
        """
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_secret": self.provider_secret,
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

    async def _authed_json(self, method: str, url: str, *, market: bool = False,
                           **kwargs) -> Dict[str, Any]:
        """Request returning parsed JSON, with ONE refresh-and-retry on 401. Access tokens
        expire after ~15 min and the SDK does not refresh in the background, so a long-lived
        client (e.g. the Flux server) must re-auth lazily on the first 401 and retry."""
        for attempt in (0, 1):
            if market:
                http, _ = self._market_http_and_url()   # production session in sandbox
                headers = self._market_headers()
            else:
                http, headers = self._http, self._headers()
            async with http.request(method, url, headers=headers, **kwargs) as resp:
                if resp.status == 401 and attempt == 0:
                    logger.info("401 on %s — refreshing session and retrying", url)
                    try:
                        await self.refresh_session()
                    except Exception as exc:
                        logger.error("session refresh failed: %s", exc)
                    continue
                if resp.status >= 400:
                    logger.error("%s %s -> %s: %s", method, url, resp.status, await resp.text())
                resp.raise_for_status()
                return await resp.json()

    # -------------------------------------------------------------------------
    # Market data
    # -------------------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote via persistent DXLinkStreamer.
        In sandbox mode uses the production session (cert has no live feed).
        Falls back to REST if the streamer is unavailable.
        Futures options (symbol contains spaces) skip DXLink entirely.
        """
        # Futures options like "./ESH6 EW2H6 260313P6630" don't stream via DXLink
        if " " in symbol or symbol in self._dxlink_rest_only:
            return await self._get_quote_via_rest(symbol)
        # Symbol pinned to REST after repeated DXLink timeouts? Skip the 5s probe
        # until the cooldown elapses, then allow one re-probe (self-healing).
        if self._dxlink_rest_pinned(symbol):
            return await self._get_quote_via_rest(symbol)
        try:
            quote = await self._get_quote_via_dxlink(symbol)
            self._dxlink_fail_counts.pop(symbol, None)  # healthy — reset streak
            return quote
        except Exception as exc:
            fails = self._dxlink_fail_counts.get(symbol, 0) + 1
            self._dxlink_fail_counts[symbol] = fails
            if fails >= self._dxlink_max_consec_fails:
                # Stream is persistently silent for this symbol — pin to REST for
                # a cooldown and log ONCE instead of warning on every poll.
                self._dxlink_rest_until[symbol] = (
                    time.monotonic() + self._dxlink_rest_cooldown_s
                )
                logger.warning(
                    "DXLink quote for %s failed %d× consecutively (%s) — "
                    "REST-only for %.0fs (likely illiquid/expiring contract)",
                    symbol, fails, exc, self._dxlink_rest_cooldown_s,
                )
            else:
                logger.warning(
                    "DXLink quote failed for %s: %s — falling back to REST", symbol, exc
                )
            return await self._get_quote_via_rest(symbol)

    def _dxlink_rest_pinned(self, symbol: str) -> bool:
        """True while *symbol* is in its post-failure REST cooldown. When the
        cooldown elapses, clear the pin + streak so the next call re-probes
        DXLink once — if the stream recovered, a success resets cleanly."""
        until = self._dxlink_rest_until.get(symbol)
        if until is None:
            return False
        if time.monotonic() >= until:
            self._dxlink_rest_until.pop(symbol, None)
            self._dxlink_fail_counts.pop(symbol, None)
            logger.info("DXLink REST cooldown elapsed for %s — re-probing stream", symbol)
            return False
        return True

    # -- Persistent DXLink quote streamer ------------------------------------

    async def _ensure_quote_streamer(self, symbol: str):
        """Ensure the owner task is alive and *symbol* is subscribed."""
        from tastytrade.dxfeed import Quote  # type: ignore

        async with self._quote_streamer_lock:
            if self._quote_owner_task is None or self._quote_owner_task.done():
                self._quote_shutdown = False
                self._quote_ready = asyncio.Event()
                self._quote_owner_task = asyncio.create_task(
                    self._quote_streamer_owner(), name="dxlink-quote-owner"
                )
            ready = self._quote_ready

        # Wait for the streamer to come up (outside the lock — the owner task
        # needs the lock-free path to set readiness).
        await asyncio.wait_for(ready.wait(), timeout=15)

        async with self._quote_streamer_lock:
            if symbol not in self._quote_symbols and self._quote_streamer is not None:
                await self._quote_streamer.subscribe(Quote, [symbol])
                self._quote_symbols.add(symbol)
                logger.info("DXLink subscribed to %s", symbol)

    async def _quote_streamer_owner(self):
        """Dedicated task that OWNS the DXLinkStreamer context.

        The SDK streamer hosts an anyio task group; whatever task enters it
        becomes the host that anyio cancels on internal failure. By doing the
        `async with` here, a websocket drop cancels only this task, the real
        exception surfaces through __aexit__, and we reconnect with backoff —
        the calling engine/monitor tasks are never touched.
        """
        from tastytrade import DXLinkStreamer  # type: ignore
        from tastytrade.dxfeed import Quote  # type: ignore

        backoff = 1.0
        while not self._quote_shutdown:
            try:
                session = self._market_sdk_session or self._sdk_session
                async with DXLinkStreamer(session) as streamer:
                    self._quote_streamer = streamer
                    resub = sorted(self._quote_symbols)
                    if resub:
                        await streamer.subscribe(Quote, resub)
                        logger.info("DXLink resubscribed to %d symbols", len(resub))
                    self._quote_ready.set()
                    logger.info("Persistent DXLink quote streamer started (owner task)")
                    backoff = 1.0
                    while True:
                        event = await streamer.get_event(Quote)
                        sym = event.event_symbol
                        bid = float(event.bid_price or 0)
                        ask = float(event.ask_price or 0)
                        mid = (bid + ask) / 2 if bid and ask else bid or ask
                        self._latest_quotes[sym] = {
                            "symbol": sym,
                            "price": mid,
                            "bid": bid,
                            "ask": ask,
                            "timestamp": datetime.now().isoformat(),
                        }
            except asyncio.CancelledError:
                if self._quote_shutdown:
                    raise
                logger.warning("DXLink streamer cancelled internally — reconnecting")
            except BaseException as exc:  # anyio surfaces ExceptionGroup here
                if self._quote_shutdown:
                    return
                logger.warning("DXLink streamer died: %r — reconnecting in %.0fs",
                               exc, backoff)
            self._quote_streamer = None
            self._quote_ready = asyncio.Event()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _close_quote_streamer(self):
        """Shut down the owner task (and with it the streamer context)."""
        self._quote_shutdown = True
        if self._quote_owner_task and not self._quote_owner_task.done():
            self._quote_owner_task.cancel()
            try:
                await self._quote_owner_task
            except (asyncio.CancelledError, Exception):
                pass
        self._quote_owner_task = None
        self._quote_streamer = None
        self._quote_symbols.clear()
        self._latest_quotes.clear()

    async def _get_quote_via_dxlink(self, symbol: str, max_stale_s: float = 30.0) -> Dict[str, Any]:
        """Return the latest quote from the persistent DXLink streamer.

        Quotes older than max_stale_s are treated as missing — serving a stale
        price to the stop monitor during a streamer outage is worse than
        falling back to REST.
        """
        await self._ensure_quote_streamer(symbol)

        # Wait up to 5s for a fresh quote after (re)subscribe
        for _ in range(50):
            cached = self._latest_quotes.get(symbol)
            if cached is not None:
                try:
                    age = (datetime.now()
                           - datetime.fromisoformat(cached["timestamp"])).total_seconds()
                except (KeyError, ValueError):
                    age = 0.0
                if age <= max_stale_s:
                    return cached
            await asyncio.sleep(0.1)

        raise TimeoutError(f"No fresh DXLink quote for {symbol} within 5s")

    async def _get_quote_via_rest(self, symbol: str) -> Dict[str, Any]:
        """REST fallback for quotes using TT market-data endpoint."""
        from urllib.parse import quote as url_quote
        http, base = self._market_http_and_url()
        hdrs = self._market_headers()
        raw = symbol.lstrip("/")
        # Detect instrument type: futures start with / or contain exchange suffix
        if symbol.startswith("/") or symbol.startswith("."):
            # Futures option symbols contain spaces (e.g. "./ESH6 EW2H6 ...")
            itype = "Future Option" if " " in symbol else "Future"
        else:
            itype = "Equity"
        encoded = url_quote(raw, safe="")
        url = f"{base}/market-data/{itype}/{encoded}"
        async with http.get(url, headers=hdrs) as resp:
            resp.raise_for_status()
            data = await resp.json()
            item = data.get("data", data)
            # TT returns nested items or flat depending on endpoint
            if isinstance(item, dict) and "items" in item:
                item = item["items"][0] if item["items"] else {}
            elif isinstance(item, list):
                item = item[0] if item else {}
            mark = item.get("mark", item.get("last", 0))
            return {
                "symbol": symbol,
                "price": float(mark) if mark else 0.0,
                "bid": float(item.get("bid", 0) or 0),
                "ask": float(item.get("ask", 0) or 0),
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
            from tastytrade.dxfeed import Candle  # type: ignore

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

            for attempt in range(2):
                session = self._market_sdk_session or self._sdk_session
                if session is None:
                    return []
                try:
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
                    break
                except Exception as exc:
                    if attempt == 0 and _is_timeout_error(exc):
                        logger.warning(
                            "get_candles streamer setup timed out for %s interval=%s; refreshing session and retrying",
                            symbol,
                            interval,
                        )
                        await self.refresh_session()
                        continue
                    raise

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

        except asyncio.TimeoutError as exc:
            logger.warning(
                "get_candles timed out for %s interval=%s count=%s from_time=%s (%s: %r)",
                symbol,
                interval,
                count,
                from_time,
                type(exc).__name__,
                exc,
            )
            return []
        except Exception as exc:
            logger.exception(
                "get_candles failed for %s interval=%s count=%s from_time=%s (%s: %r)",
                symbol,
                interval,
                count,
                from_time,
                type(exc).__name__,
                exc,
            )
            return []

    async def get_option_chain(self, symbol: str,
                               expiration: Optional[date] = None) -> List[Dict[str, Any]]:
        """Get an option chain for a symbol with live Greeks, for `expiration` (default 0DTE).
        Fetches symbol list from REST, then enriches near-ATM strikes with live bid/ask/delta
        via DXLink. In sandbox mode, reads from production API (cert has no live feed).
        """
        target = (expiration or date.today()).strftime("%Y-%m-%d")
        _, base = self._market_http_and_url()
        data = await self._authed_json("GET", f"{base}/option-chains/{symbol}/nested",
                                       market=True, params={"expiration-date": target})

        options = []
        for chain in data.get("data", {}).get("items", []):
            for expiration in chain.get("expirations", []):
                exp_date = expiration.get("expiration-date", "")
                if exp_date != target:
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
            logger.error("Option chain Greeks enrichment failed: %s", exc)

        return options

    async def _enrich_option_chain_greeks(
        self, options: List[Dict[str, Any]], spx_price: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Fetch live bid/ask/delta for near-ATM options via a single DXLink session.
        Subscribes to both Quote and Greeks simultaneously and collects for 6s.
        """
        from tastytrade import DXLinkStreamer  # type: ignore
        from tastytrade.dxfeed import Greeks, Quote  # type: ignore

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

    async def _get_atm_iv_rest(self, symbol: str = "/ES") -> Optional[float]:
        """Get ATM 0DTE IV from /market-metrics REST endpoint."""
        try:
            http, base = self._market_http_and_url()
            async with http.get(
                f"{base}/market-metrics",
                headers=self._market_headers(),
                params={"symbols": symbol},
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                items = data.get("data", {}).get("items", [])
                if not items:
                    return None
                today_str = date.today().strftime("%Y-%m-%d")
                for exp_iv in items[0].get("option-expiration-implied-volatilities", []):
                    if exp_iv.get("expiration-date") == today_str:
                        return float(exp_iv["implied-volatility"])
                # Fall back to index IV
                idx = items[0].get("implied-volatility-index")
                return float(idx) if idx else None
        except Exception as exc:
            logger.warning("ATM IV REST fetch failed: %s", exc)
            return None

    @staticmethod
    def _enrich_chain_bs(
        options: List[Dict[str, Any]], spot: float, atm_iv: float
    ) -> None:
        """Enrich option chain with BS-computed greeks and prices in-place."""
        import math

        def _ncdf(x: float) -> float:
            a = abs(x)
            t = 1.0 / (1.0 + 0.2316419 * a)
            poly = t * (0.319381530 + t * (-0.356563782 + t * (
                1.781477937 + t * (-1.821255978 + t * 1.330274429))))
            cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
            return cdf if x >= 0 else 1.0 - cdf

        def _npdf(x: float) -> float:
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        now_et = datetime.now(et)
        close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        tte_hours = max(0.01, (close_dt - now_et).total_seconds() / 3600)
        t = tte_hours / (252 * 6.5)  # TTE in years
        if t <= 0:
            return
        sqrt_t = math.sqrt(t)

        for opt in options:
            k = opt.get("strike", 0)
            if k <= 0:
                continue
            iv = atm_iv
            try:
                d1 = (math.log(spot / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
                d2 = d1 - iv * sqrt_t
            except (ValueError, ZeroDivisionError):
                continue

            nd1 = _ncdf(d1)
            nd2 = _ncdf(d2)
            pdf_d1 = _npdf(d1)

            if opt.get("option_type") == "call":
                price = spot * nd1 - k * nd2
                delta = round(nd1, 4)
            else:
                price = k * _ncdf(-d2) - spot * _ncdf(-d1)
                delta = round(nd1 - 1.0, 4)

            gamma = round(pdf_d1 / (spot * iv * sqrt_t), 6) if spot > 0 else 0.0
            theta = round(-(spot * pdf_d1 * iv) / (2 * sqrt_t * 252), 4)
            vega = round(spot * pdf_d1 * sqrt_t / 100, 4)  # per 1% IV move

            bid = round(max(0.05, price * 0.95), 2)
            ask = round(price * 1.05 + 0.10, 2)

            opt["delta"] = delta
            opt["gamma"] = gamma
            opt["theta"] = theta
            opt["vega"] = vega
            opt["iv"] = round(iv, 5)
            opt["bid"] = bid
            opt["ask"] = ask
            opt["last"] = round(price, 2)

        logger.info(
            "BS enrichment: %d options, spot=%.2f, IV=%.4f, TTE=%.2fh",
            len(options), spot, atm_iv, tte_hours,
        )

    async def get_active_futures_contract(self, product_code: str) -> Optional[str]:
        """Discover the active futures contract from the option chain API.

        Returns the DXLink symbol (e.g. '/ESH26:XCME') for the underlying
        that has 0DTE options today, or None if unavailable.

        This is the authoritative source — immune to calendar logic bugs
        that can cause contract month mismatch on expiration day.
        """
        try:
            chain = await self._fetch_nested_future_option_chain_data(product_code)
            today_str = date.today().strftime("%Y-%m-%d")

            for subchain in chain.get("option-chains", []):
                for expiration in subchain.get("expirations", []):
                    if str(expiration.get("expiration-date", "")) == today_str:
                        short_sym = expiration.get("underlying-symbol", "")  # e.g. "/ESH6"
                        # Expand short symbol to full DXLink format
                        # /ESH6 → /ESH26:XCME (insert century digit, add exchange)
                        exchange = "XCME"
                        strikes = expiration.get("strikes", [])
                        if strikes:
                            ss = strikes[0].get("call-streamer-symbol", "")
                            if ":" in ss:
                                exchange = ss.split(":")[-1]
                        # Expand 1-digit year to 2-digit: /ESH6 → /ESH26
                        import re
                        m = re.match(r"(/\w+?)(\d)$", short_sym)
                        if m:
                            prefix, digit = m.groups()
                            century = str(date.today().year)[-2]
                            short_sym = f"{prefix}{century}{digit}"
                        return f"{short_sym}:{exchange}"
            return None
        except Exception as exc:
            logger.debug("Could not discover active contract for %s: %s", product_code, exc)
            return None

    async def get_future_option_chain(
        self, product_code: str, underlying_price: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get 0DTE futures option chain (e.g. ES) with Greeks.
        Uses the tastytrade SDK's NestedFutureOptionChain for proper auth.
        In sandbox mode, uses the production session (cert has no live feed).
        """
        today = date.today().strftime("%Y-%m-%d")
        try:
            chain = await self._fetch_nested_future_option_chain_data(product_code)
        except Exception as exc:
            logger.warning(
                "get_future_option_chain fetch failed for %s (%s: %r)",
                product_code,
                type(exc).__name__,
                exc,
            )
            return []

        options = []
        for subchain in chain.get("option-chains", []):
            for expiration in subchain.get("expirations", []):
                exp_date = str(expiration.get("expiration-date", ""))
                if exp_date != today:
                    continue
                for strike in expiration.get("strikes", []):
                    strike_price = float(strike.get("strike-price", 0) or 0)
                    for side in ["call", "put"]:
                        sym = strike.get(side, "")
                        streamer_sym = strike.get(f"{side}-streamer-symbol", "")
                        if not sym:
                            continue
                        options.append({
                            "symbol": sym,
                            "streamer_symbol": streamer_sym or "",
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

        try:
            options = await self._enrich_option_chain_greeks(
                options, spx_price=underlying_price
            )
        except Exception as exc:
            logger.warning(
                "Future option chain Greeks enrichment failed for %s (%s: %r)",
                product_code,
                type(exc).__name__,
                exc,
            )

        # If DXLink enrichment left most deltas at 0, fall back to BS greeks
        n_with_delta = sum(1 for o in options if o.get("delta", 0) != 0)
        if n_with_delta < len(options) * 0.1 and underlying_price > 0:
            logger.info("No DXLink greeks — computing BS greeks from ATM IV")
            atm_iv = await self._get_atm_iv_rest("/ES")
            if atm_iv and atm_iv > 0:
                self._enrich_chain_bs(options, underlying_price, atm_iv)

        return options

    async def get_5min_candles(self, symbol: str, count: int = 6) -> List[Dict[str, Any]]:
        """
        Fetch the last N completed 5-minute candles for a symbol via DXLink.
        Uses production session in sandbox mode.
        """
        try:
            from tastytrade import DXLinkStreamer  # type: ignore
            from tastytrade.dxfeed import Candle  # type: ignore

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
            logger.error("DXLink candles failed for %s: %s", symbol, exc)
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

    async def place_stop_market(
        self, symbol: str, trigger_price: float, quantity: int = 1,
    ) -> Dict[str, Any]:
        """Submit a resting single-leg stop-market (Buy to Close) on a short option."""
        payload = {
            "time-in-force": "GTC",
            "order-type": "Stop",
            "stop-trigger": str(round(float(trigger_price), 2)),
            "legs": [self._build_leg(symbol, "Buy to Close", quantity)],
        }
        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.error("Stop-market order rejected (%s): %s", resp.status, body)
            resp.raise_for_status()
            data = await resp.json()
            order_id = str(data["data"]["order"]["id"])
            logger.info("Stop-market placed: %s trigger=%s order_id=%s",
                        symbol, trigger_price, order_id)
            return {"order_id": order_id, "symbol": symbol,
                    "trigger_price": float(trigger_price)}

    async def _place_iron_condor(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Build and submit a 4-leg iron condor order."""
        exp = date.today()
        qty = order.get("quantity", 1)

        legs = [
            self._build_leg(format_osi(exp, order['short_call_strike'], is_call=True), "Sell to Open", qty),
            self._build_leg(format_osi(exp, order['long_call_strike'], is_call=True), "Buy to Open", qty),
            self._build_leg(format_osi(exp, order['short_put_strike'], is_call=False), "Sell to Open", qty),
            self._build_leg(format_osi(exp, order['long_put_strike'], is_call=False), "Buy to Open", qty),
        ]

        credit = order.get("credit", 0)
        payload = {
            "time-in-force": "Day",
            "order-type": "Limit",
            "price": str(credit) if credit else None,
            "price-effect": "Credit",
            "legs": legs,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.error("Iron condor order rejected (%s): %s", resp.status, body)
            resp.raise_for_status()
            data = await resp.json()
            order_id = str(data["data"]["order"]["id"])
            logger.info("Iron condor order placed successfully: order_id=%s", order_id)
            return {"order_id": order_id}

    async def get_orders(self, working_only: bool = True) -> List[Dict[str, Any]]:
        """Recent orders for the account. working_only keeps non-terminal ones (resting/working).
        Normalized to {id, status, order_type, price, price_effect, legs:[{symbol,action,qty}]}."""
        terminal = {"Filled", "Cancelled", "Canceled", "Rejected", "Expired", "Removed"}
        data = await self._authed_json(
            "GET", f"{self.base_url}/accounts/{self.account_number}/orders")
        out: List[Dict[str, Any]] = []
        for o in data.get("data", {}).get("items", []):
            status = o.get("status", "")
            if working_only and status in terminal:
                continue
            out.append({
                "id": str(o.get("id", "")),
                "status": status,
                "order_type": o.get("order-type", ""),
                "price": float(o.get("price", 0) or 0),
                "price_effect": o.get("price-effect", ""),
                "legs": [{"symbol": l.get("symbol", "").strip(), "action": l.get("action", ""),
                          "qty": int(float(l.get("quantity", 0) or 0))}
                         for l in o.get("legs", [])],
            })
        return out

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Open positions for the trading account, normalized to
        {symbol, quantity, direction, instrument_type}. Quantity is absolute; direction is
        Long/Short/Zero."""
        data = await self._authed_json(
            "GET", f"{self.base_url}/accounts/{self.account_number}/positions")
        out: List[Dict[str, Any]] = []
        for item in data.get("data", {}).get("items", []):
            out.append({
                "symbol": item.get("symbol", ""),
                "quantity": abs(int(float(item.get("quantity", 0) or 0))),
                "direction": item.get("quantity-direction", ""),
                "instrument_type": item.get("instrument-type", ""),
                "avg_open_price": float(item.get("average-open-price", 0) or 0),
            })
        return out

    async def get_listed_chain(self, symbol: str) -> Dict[str, Dict[float, Dict[str, str]]]:
        """All listed expirations/strikes for `symbol` from the TRADING env (so the returned
        broker symbols are tradeable there). {exp_iso: {strike: {"call": sym, "put": sym}}}.
        Instrument metadata is available in cert even though live quotes are not."""
        data = await self._authed_json("GET", f"{self.base_url}/option-chains/{symbol}/nested")
        out: Dict[str, Dict[float, Dict[str, str]]] = {}
        for chain in data.get("data", {}).get("items", []):
            for expn in chain.get("expirations", []):
                exp = expn.get("expiration-date", "")
                strikes: Dict[float, Dict[str, str]] = {}
                for s in expn.get("strikes", []):
                    k = float(s.get("strike-price", 0) or 0)
                    call = (s.get("call") or "").strip()
                    put = (s.get("put") or "").strip()
                    if call or put:
                        strikes[k] = {"call": call, "put": put}
                if exp and strikes:
                    out[exp] = strikes
        return out

    async def place_complex_order(self, leg_specs, price: float, price_effect: str = "Debit",
                                  tif: str = "Day") -> Dict[str, Any]:
        """Submit one complex (multi-leg, possibly multi-expiry) limit order.

        leg_specs: iterable of (osi_symbol, action, quantity). Each leg's OSI carries its own
        expiry, so calendars/diagonals work. price = net limit ($/share, positive)."""
        legs = [self._build_leg(osi, action, int(qty)) for (osi, action, qty) in leg_specs]
        payload = {"time-in-force": tif, "order-type": "Limit",
                   "price": str(round(abs(float(price)), 2)), "price-effect": price_effect,
                   "legs": legs}
        data = await self._authed_json(
            "POST", f"{self.base_url}/accounts/{self.account_number}/orders", json=payload)
        return {"order_id": str(data["data"]["order"]["id"])}

    async def place_option_limit(self, osi: str, action: str, quantity: int, limit: float,
                                 tif: str = "Day") -> Dict[str, Any]:
        """Single-leg option limit order (for leg-in execution). Buy → Debit, Sell → Credit."""
        effect = "Debit" if action.startswith("Buy") else "Credit"
        payload = {"time-in-force": tif, "order-type": "Limit",
                   "price": str(round(abs(float(limit)), 2)), "price-effect": effect,
                   "legs": [self._build_leg(osi, action, int(quantity))]}
        data = await self._authed_json(
            "POST", f"{self.base_url}/accounts/{self.account_number}/orders", json=payload)
        return {"order_id": str(data["data"]["order"]["id"])}

    async def _place_credit_spread(self, order: Dict[str, Any]) -> Dict[str, Any]:
        qty = order.get("quantity", 1)

        # Use explicit symbols if provided (futures options), else build SPXW symbols
        if "short_symbol" in order and "long_symbol" in order:
            short_sym = order["short_symbol"]
            long_sym = order["long_symbol"]
            inst_type = order.get("instrument_type", "Future Option")
        else:
            exp = date.today()
            is_call = "call" in order["order_type"]
            short_sym = format_osi(exp, order['short_strike'], is_call=is_call)
            long_sym = format_osi(exp, order['long_strike'], is_call=is_call)
            inst_type = "Equity Option"

        legs = [
            self._build_leg(short_sym, "Sell to Open", qty, instrument_type=inst_type),
            self._build_leg(long_sym, "Buy to Open", qty, instrument_type=inst_type),
        ]

        credit = order.get("credit", 0) or 0
        # Round the net-credit limit to the 0.05 option tick.
        price = round(round(float(credit) / 0.05) * 0.05, 2) if credit else None
        payload = {
            "time-in-force": "Day",
            "order-type": "Limit",
            "price": str(price) if price else None,
            "price-effect": "Credit",
            "legs": legs,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info("Credit spread order payload: %s", json.dumps(payload, indent=2))

        async with self._http.post(
            f"{self.base_url}/accounts/{self.account_number}/orders",
            headers=self._headers(),
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.error("Credit spread order rejected (%s): %s", resp.status, body)
            resp.raise_for_status()
            data = await resp.json()
            order_id = str(data["data"]["order"]["id"])
            logger.info("Credit spread order placed successfully: order_id=%s", order_id)
            return {"order_id": order_id}

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
    def _build_leg(
        symbol: str, action: str, quantity: int,
        instrument_type: str = "Equity Option",
    ) -> Dict[str, Any]:
        return {
            "instrument-type": instrument_type,
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
        }
