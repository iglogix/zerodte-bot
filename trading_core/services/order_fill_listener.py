"""
Order Fill Listener — websocket-based real-time order fill detection.

Uses TastyTrade's AlertStreamer (account streamer) to receive order fill events
the moment they happen, rather than waiting for the polling cycle.

Architecture:
  - Runs as a separate asyncio task alongside PositionMonitor
  - On stop-order fill for a short leg → puts event on shared asyncio.Queue
  - PositionMonitor drains the queue each cycle and acts immediately
  - If websocket disconnects or is unavailable (dry_run/manual), falls back
    silently — polling in PositionMonitor remains the safety net

TastyTrade AlertStreamer docs:
  https://tastyworks-api.readthedocs.io/en/latest/account-streamer.html
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

_RECONNECT_BASE_SECONDS = 2
_MAX_RECONNECT_ATTEMPTS = 10


@dataclass
class FillEvent:
    """Normalised order fill event from the account streamer."""
    order_id: str
    symbol: str
    side: str                # "Buy" | "Sell"
    quantity: int
    fill_price: float
    status: str              # "Filled" | "Partially Filled"
    order_type: str          # "Stop Limit" | "Stop" | "Market" | etc.
    received_at: datetime = field(default_factory=datetime.now)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_stop_fill(self) -> bool:
        return self.order_type.lower() in ("stop limit", "stop", "stop market")

    @property
    def is_fully_filled(self) -> bool:
        return self.status.lower() == "filled"


class OrderFillListener:
    """
    Listens for order fill events via TastyTrade's AlertStreamer websocket.

    Puts FillEvent objects onto `fill_queue` for PositionMonitor to consume.
    Runs with exponential-backoff auto-reconnect.

    In dry_run / manual mode: does not connect; fill_queue remains empty
    (PositionMonitor polling handles everything).
    """

    def __init__(
        self,
        api_client=None,
        fill_queue: Optional[asyncio.Queue] = None,
        mode: str = "dry_run",
        on_fill: Optional[Callable[[FillEvent], Any]] = None,
    ):
        self.api_client = api_client
        self.fill_queue: asyncio.Queue = fill_queue or asyncio.Queue()
        self.mode = mode
        self.on_fill = on_fill
        self._running = False
        self._connected = False
        self._seen_order_ids: Set[str] = set()

    @property
    def is_active(self) -> bool:
        """True when the websocket is connected and listening."""
        return self._connected

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self):
        """Start the listener loop. No-op in dry_run / manual mode."""
        # alpaca_paper: fills come back from the Alpaca order poll, not the
        # TastyTrade websocket — the TT account holds no position to report on.
        if self.mode in ("dry_run", "manual", "alpaca_paper"):
            logger.info(
                "OrderFillListener: mode=%s — websocket disabled, polling is sole mechanism",
                self.mode.upper(),
            )
            return

        self._running = True
        logger.info("OrderFillListener starting (mode=%s)", self.mode.upper())
        await self._connect_with_retry()

    async def stop(self):
        self._running = False
        self._connected = False
        logger.info("OrderFillListener stopped")

    # -------------------------------------------------------------------------
    # Connection with exponential backoff
    # -------------------------------------------------------------------------

    async def _connect_with_retry(self):
        attempt = 0
        while self._running and attempt < _MAX_RECONNECT_ATTEMPTS:
            try:
                logger.info(
                    "OrderFillListener: connecting (attempt %d/%d)",
                    attempt + 1, _MAX_RECONNECT_ATTEMPTS,
                )
                await self._listen_loop()
            except Exception as exc:
                self._connected = False
                if not self._running:
                    break
                wait = min(_RECONNECT_BASE_SECONDS ** attempt, 60)
                logger.warning(
                    "OrderFillListener: disconnected (%s) — retrying in %ds",
                    exc, wait,
                )
                await asyncio.sleep(wait)
                attempt += 1

        if attempt >= _MAX_RECONNECT_ATTEMPTS:
            logger.error(
                "OrderFillListener: max reconnect attempts reached — "
                "falling back to polling-only mode"
            )
            self._connected = False

    # -------------------------------------------------------------------------
    # Core listen loop
    # -------------------------------------------------------------------------

    async def _listen_loop(self):
        """
        Connect to TastyTrade AlertStreamer and consume order events.

        Uses the `tastytrade` SDK if available; falls back to raw aiohttp
        websocket if the SDK is not installed.
        """
        try:
            await self._listen_via_sdk()
        except ImportError:
            logger.warning(
                "tastytrade SDK not installed — falling back to raw websocket"
            )
            await self._listen_via_raw_ws()

    async def _listen_via_sdk(self):
        """Use the official tastytrade Python SDK AlertStreamer (v10+)."""
        from tastytrade import Account, AlertStreamer  # type: ignore
        from tastytrade.order import PlacedOrder  # type: ignore

        session = await self.api_client.get_session()

        accounts = await Account.get(session)
        if not isinstance(accounts, list):
            accounts = [accounts]

        self._connected = True
        logger.info(
            "OrderFillListener: connected via SDK AlertStreamer (%d account(s))",
            len(accounts),
        )

        async with AlertStreamer(session) as streamer:
            await streamer.subscribe_accounts(accounts)
            async for order in streamer.listen(PlacedOrder):
                if not self._running:
                    break
                await self._handle_sdk_order(order)

    async def _handle_sdk_order(self, order: Any):
        """Convert SDK PlacedOrder to FillEvent and enqueue if it's a fill."""
        status = getattr(order, "status", "") or ""
        if status.lower() not in ("filled", "partially filled"):
            return

        legs = getattr(order, "legs", []) or []
        order_type = getattr(order, "order_type", "Unknown") or "Unknown"

        for leg in legs:
            symbol = getattr(leg, "symbol", "") or ""
            side = getattr(leg, "action", "") or ""
            quantity = int(getattr(leg, "quantity", 1) or 1)
            fill_price = float(getattr(leg, "average_open_price", 0.0) or 0.0)
            order_id = str(getattr(order, "id", "")) or ""

            if order_id in self._seen_order_ids:
                continue
            self._seen_order_ids.add(order_id)

            event = FillEvent(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                fill_price=fill_price,
                status=status,
                order_type=order_type,
                raw={"order_id": order_id, "status": status},
            )
            await self._dispatch(event)

    async def _listen_via_raw_ws(self):
        """
        Fallback: raw aiohttp websocket to TastyTrade account streamer.

        TastyTrade account streamer endpoint:
          wss://streamer.tastyworks.com  (production)
          wss://streamer.cert.tastyworks.com  (sandbox)
        """
        import aiohttp

        ws_url = (
            "wss://streamer.cert.tastyworks.com"
            if self.mode == "sandbox"
            else "wss://streamer.tastyworks.com"
        )
        token = await self.api_client.get_session_token()
        headers = {"Authorization": f"Bearer {token}"}

        self._connected = True
        logger.info("OrderFillListener: connected via raw websocket to %s", ws_url)

        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(ws_url, headers=headers) as ws:
                # Subscribe to account-level events
                await ws.send_json({
                    "action": "connect",
                    "value": [await self.api_client.get_account_number()],
                    "auth-token": token,
                })

                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_raw_message(msg.json())
                    elif msg.type in (
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSED,
                    ):
                        raise ConnectionError("Websocket closed/error")

    async def _handle_raw_message(self, data: Dict[str, Any]):
        """Parse a raw account streamer message for order fill events."""
        msg_type = data.get("type", "")
        if msg_type != "OrderFill":
            return

        order_id = str(data.get("order-id", ""))
        if order_id in self._seen_order_ids:
            return
        self._seen_order_ids.add(order_id)

        event = FillEvent(
            order_id=order_id,
            symbol=data.get("symbol", ""),
            side=data.get("action", ""),
            quantity=int(data.get("quantity", 1)),
            fill_price=float(data.get("fill-price", 0.0)),
            status=data.get("status", "Filled"),
            order_type=data.get("order-type", "Unknown"),
            raw=data,
        )
        await self._dispatch(event)

    # -------------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------------

    async def _dispatch(self, event: FillEvent):
        """Enqueue fill event and call optional callback."""
        logger.info(
            "[FILL] order_id=%s symbol=%s side=%s qty=%d price=%.2f type=%s",
            event.order_id, event.symbol, event.side,
            event.quantity, event.fill_price, event.order_type,
        )
        await self.fill_queue.put(event)
        if self.on_fill:
            try:
                await self.on_fill(event)
            except Exception:
                logger.exception("on_fill callback raised an error")
