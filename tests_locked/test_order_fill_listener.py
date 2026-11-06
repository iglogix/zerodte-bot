"""
Tests for OrderFillListener — websocket-based order fill detection.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_ic.services.order_fill_listener import FillEvent, OrderFillListener


# ---------------------------------------------------------------------------
# FillEvent unit tests
# ---------------------------------------------------------------------------

class TestFillEvent:
    def _make(self, order_type="Stop Limit", status="Filled"):
        return FillEvent(
            order_id="ORD-001",
            symbol="SPXW  250117C05900000",
            side="Sell",
            quantity=1,
            fill_price=42.50,
            status=status,
            order_type=order_type,
        )

    def test_is_stop_fill_stop_limit(self):
        assert self._make(order_type="Stop Limit").is_stop_fill is True

    def test_is_stop_fill_stop(self):
        assert self._make(order_type="Stop").is_stop_fill is True

    def test_is_stop_fill_stop_market(self):
        assert self._make(order_type="Stop Market").is_stop_fill is True

    def test_is_not_stop_fill_market(self):
        assert self._make(order_type="Market").is_stop_fill is False

    def test_is_not_stop_fill_limit(self):
        assert self._make(order_type="Limit").is_stop_fill is False

    def test_is_fully_filled(self):
        assert self._make(status="Filled").is_fully_filled is True

    def test_is_not_fully_filled_partial(self):
        assert self._make(status="Partially Filled").is_fully_filled is False

    def test_received_at_set_automatically(self):
        event = self._make()
        assert isinstance(event.received_at, datetime)

    def test_raw_defaults_to_empty_dict(self):
        event = self._make()
        assert event.raw == {}


# ---------------------------------------------------------------------------
# OrderFillListener — mode gating
# ---------------------------------------------------------------------------

class TestOrderFillListenerModeGating:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_connect(self):
        listener = OrderFillListener(mode="dry_run")
        # start() should return immediately without connecting
        await asyncio.wait_for(listener.start(), timeout=1.0)
        assert listener.is_active is False

    @pytest.mark.asyncio
    async def test_manual_does_not_connect(self):
        listener = OrderFillListener(mode="manual")
        await asyncio.wait_for(listener.start(), timeout=1.0)
        assert listener.is_active is False

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        listener = OrderFillListener(mode="dry_run")
        await listener.start()
        await listener.stop()
        assert listener._running is False


# ---------------------------------------------------------------------------
# OrderFillListener — fill queue
# ---------------------------------------------------------------------------

class TestOrderFillListenerQueue:
    def _make_listener(self, mode="sandbox"):
        q = asyncio.Queue()
        listener = OrderFillListener(
            api_client=MagicMock(),
            fill_queue=q,
            mode=mode,
        )
        return listener, q

    @pytest.mark.asyncio
    async def test_dispatch_puts_event_on_queue(self):
        listener, q = self._make_listener()
        event = FillEvent(
            order_id="ORD-001",
            symbol="SPXW  250117C05900000",
            side="Sell",
            quantity=1,
            fill_price=42.50,
            status="Filled",
            order_type="Stop Limit",
        )
        await listener._dispatch(event)
        assert q.qsize() == 1
        queued = q.get_nowait()
        assert queued.order_id == "ORD-001"

    @pytest.mark.asyncio
    async def test_dispatch_calls_on_fill_callback(self):
        callback = AsyncMock()
        q = asyncio.Queue()
        listener = OrderFillListener(
            fill_queue=q,
            mode="sandbox",
            on_fill=callback,
        )
        event = FillEvent(
            order_id="ORD-002",
            symbol="SPXW  250117P05800000",
            side="Buy",
            quantity=1,
            fill_price=10.00,
            status="Filled",
            order_type="Stop",
        )
        await listener._dispatch(event)
        callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_duplicate_order_id_not_enqueued_twice(self):
        listener, q = self._make_listener()
        event = FillEvent(
            order_id="ORD-DUP",
            symbol="SPXW  250117C05900000",
            side="Sell",
            quantity=1,
            fill_price=42.50,
            status="Filled",
            order_type="Stop Limit",
        )
        # Simulate raw message handler dedup
        listener._seen_order_ids.add("ORD-DUP")
        # _dispatch is called after dedup check in real handlers; test dedup guard
        raw_msg = {
            "type": "OrderFill",
            "order-id": "ORD-DUP",
            "symbol": "SPXW  250117C05900000",
            "action": "Sell",
            "quantity": 1,
            "fill-price": 42.50,
            "status": "Filled",
            "order-type": "Stop Limit",
        }
        await listener._handle_raw_message(raw_msg)
        assert q.empty()  # dedup blocked it

    @pytest.mark.asyncio
    async def test_non_fill_raw_message_ignored(self):
        listener, q = self._make_listener()
        await listener._handle_raw_message({"type": "HeartBeat"})
        assert q.empty()

    @pytest.mark.asyncio
    async def test_raw_message_order_fill_enqueued(self):
        listener, q = self._make_listener()
        raw_msg = {
            "type": "OrderFill",
            "order-id": "ORD-003",
            "symbol": "SPXW  250117C05900000",
            "action": "Sell",
            "quantity": 1,
            "fill-price": 55.00,
            "status": "Filled",
            "order-type": "Stop",
        }
        await listener._handle_raw_message(raw_msg)
        assert q.qsize() == 1
        event = q.get_nowait()
        assert event.symbol == "SPXW  250117C05900000"
        assert event.is_stop_fill is True
        assert event.fill_price == 55.00


# ---------------------------------------------------------------------------
# OrderFillListener — SDK handler
# ---------------------------------------------------------------------------

class TestHandleSdkOrder:
    @pytest.mark.asyncio
    async def test_sdk_order_filled_stop_enqueued(self):
        q = asyncio.Queue()
        listener = OrderFillListener(fill_queue=q, mode="sandbox")

        leg = MagicMock()
        leg.symbol = "SPXW  250117C05900000"
        leg.action = "Sell"
        leg.quantity = 1
        leg.average_open_price = 42.50

        order = MagicMock()
        order.id = "SDK-ORD-001"
        order.status = "Filled"
        order.order_type = "Stop Limit"
        order.legs = [leg]

        await listener._handle_sdk_order(order)
        assert q.qsize() == 1
        event = q.get_nowait()
        assert event.order_id == "SDK-ORD-001"
        assert event.is_stop_fill is True

    @pytest.mark.asyncio
    async def test_sdk_order_non_fill_status_ignored(self):
        q = asyncio.Queue()
        listener = OrderFillListener(fill_queue=q, mode="sandbox")

        order = MagicMock()
        order.id = "SDK-ORD-002"
        order.status = "Working"
        order.order_type = "Stop Limit"
        order.legs = []

        await listener._handle_sdk_order(order)
        assert q.empty()

    @pytest.mark.asyncio
    async def test_sdk_order_duplicate_not_enqueued(self):
        q = asyncio.Queue()
        listener = OrderFillListener(fill_queue=q, mode="sandbox")
        listener._seen_order_ids.add("SDK-DUP")

        leg = MagicMock()
        leg.symbol = "SPXW  250117C05900000"
        leg.action = "Sell"
        leg.quantity = 1
        leg.average_open_price = 42.50

        order = MagicMock()
        order.id = "SDK-DUP"
        order.status = "Filled"
        order.order_type = "Stop Limit"
        order.legs = [leg]

        await listener._handle_sdk_order(order)
        assert q.empty()
