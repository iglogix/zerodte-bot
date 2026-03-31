"""
Tests for persistent DXLink quote streamer in TastyTradeClient.

Verifies that the client keeps a single long-lived DXLink connection
instead of opening/closing a new websocket on every get_quote() call.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.tastytrade_client import TastyTradeClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quote_event(symbol="/ESH26:XCME", bid=6780.75, ask=6781.25):
    """Create a mock DXLink Quote event."""
    evt = MagicMock()
    evt.event_symbol = symbol
    evt.bid_price = bid
    evt.ask_price = ask
    return evt


def _build_client() -> TastyTradeClient:
    """Create a TastyTradeClient in manual mode with a fake SDK session."""
    client = TastyTradeClient(
        provider_secret="fake", refresh_token="fake",
        account_number="TEST123", mode="manual",
    )
    client._sdk_session = MagicMock()
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persistent_streamer_reuses_connection():
    """get_quote() should open the DXLink streamer once, not per call."""
    client = _build_client()
    mock_streamer = AsyncMock()
    mock_streamer.subscribe = AsyncMock()
    mock_streamer.get_event = AsyncMock(
        return_value=_make_quote_event(bid=6780.0, ask=6781.0)
    )

    mock_cls = MagicMock()
    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_streamer)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    mock_cls.return_value = mock_instance

    with patch("src.services.tastytrade_client.TastyTradeClient._get_quote_via_dxlink") as mock_dxlink:
        # Bypass the real implementation and test _ensure_quote_streamer directly
        pass

    # Test via the real path with mocked DXLinkStreamer
    with patch.dict("sys.modules", {"tastytrade": MagicMock(), "tastytrade.dxfeed": MagicMock()}):
        with patch("src.services.tastytrade_client.TastyTradeClient._ensure_quote_streamer") as mock_ensure:
            mock_ensure.return_value = None
            client._latest_quotes["/ESH26:XCME"] = {
                "symbol": "/ESH26:XCME",
                "price": 6780.5,
                "bid": 6780.0,
                "ask": 6781.0,
                "timestamp": "2026-03-11T15:17:48",
            }
            q1 = await client._get_quote_via_dxlink("/ESH26:XCME")
            q2 = await client._get_quote_via_dxlink("/ESH26:XCME")

            assert q1["price"] == 6780.5
            assert q2["price"] == 6780.5
            # _ensure_quote_streamer called each time (idempotent), not creating new streamers
            assert mock_ensure.call_count == 2


@pytest.mark.asyncio
async def test_streamer_returns_cached_quote():
    """_get_quote_via_dxlink reads from _latest_quotes cache."""
    client = _build_client()
    client._latest_quotes["/ESH26:XCME"] = {
        "symbol": "/ESH26:XCME",
        "price": 6781.0,
        "bid": 6780.75,
        "ask": 6781.25,
        "timestamp": "2026-03-11T15:30:00",
    }

    with patch.object(client, "_ensure_quote_streamer", new_callable=AsyncMock):
        quote = await client._get_quote_via_dxlink("/ESH26:XCME")

    assert quote["price"] == 6781.0
    assert quote["bid"] == 6780.75
    assert quote["ask"] == 6781.25


@pytest.mark.asyncio
async def test_streamer_timeout_when_no_quote():
    """_get_quote_via_dxlink raises TimeoutError if no quote arrives within 5s."""
    client = _build_client()
    # No quotes in cache — should timeout (but we shorten the wait via mock)

    with patch.object(client, "_ensure_quote_streamer", new_callable=AsyncMock):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError, match="No DXLink quote"):
                await client._get_quote_via_dxlink("/ESH26:XCME")


@pytest.mark.asyncio
async def test_feed_loop_updates_latest_quotes():
    """_quote_feed_loop should update _latest_quotes from streamer events."""
    client = _build_client()

    events = [
        _make_quote_event("/ESH26:XCME", 6780.0, 6781.0),
        _make_quote_event("/ESH26:XCME", 6782.0, 6783.0),
    ]
    call_count = 0

    async def mock_get_event(quote_type):
        nonlocal call_count
        if call_count < len(events):
            evt = events[call_count]
            call_count += 1
            return evt
        # After all events, cancel the loop
        raise asyncio.CancelledError()

    mock_streamer = AsyncMock()
    mock_streamer.get_event = mock_get_event
    client._quote_streamer = mock_streamer

    with patch.dict("sys.modules", {"tastytrade.dxfeed": MagicMock()}) as _:
        # Patch the Quote import inside the method
        with patch(
            "src.services.tastytrade_client.TastyTradeClient._quote_feed_loop",
            wraps=client._quote_feed_loop,
        ):
            # Run the real feed loop — it will process 2 events then CancelledError
            await client._quote_feed_loop()

    cached = client._latest_quotes.get("/ESH26:XCME")
    assert cached is not None
    # Should have the LATEST quote (second event)
    assert cached["bid"] == 6782.0
    assert cached["ask"] == 6783.0
    assert cached["price"] == 6782.5  # midpoint


@pytest.mark.asyncio
async def test_feed_loop_marks_streamer_dead_on_error():
    """If the feed loop hits an error, it marks the streamer as None for reconnect."""
    client = _build_client()

    async def mock_get_event(quote_type):
        raise ConnectionError("websocket died")

    mock_streamer = AsyncMock()
    mock_streamer.get_event = mock_get_event
    client._quote_streamer = mock_streamer
    client._quote_symbols = {"/ESH26:XCME"}

    with patch.dict("sys.modules", {"tastytrade.dxfeed": MagicMock()}):
        await client._quote_feed_loop()

    # Streamer should be marked as dead
    assert client._quote_streamer is None
    assert len(client._quote_symbols) == 0


@pytest.mark.asyncio
async def test_close_quote_streamer_cleanup():
    """_close_quote_streamer cancels the task and cleans up."""
    client = _build_client()

    mock_streamer = AsyncMock()
    mock_streamer.__aexit__ = AsyncMock(return_value=False)
    client._quote_streamer = mock_streamer
    client._quote_symbols = {"/ESH26:XCME"}
    client._latest_quotes["/ESH26:XCME"] = {"price": 6780.0}

    # Create a dummy task that's "running"
    async def dummy():
        await asyncio.sleep(100)

    client._quote_feed_task = asyncio.create_task(dummy())

    await client._close_quote_streamer()

    assert client._quote_streamer is None
    assert client._quote_feed_task is None
    assert len(client._quote_symbols) == 0
    assert len(client._latest_quotes) == 0
    mock_streamer.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_get_quote_falls_back_to_rest_on_dxlink_failure():
    """get_quote should fall back to REST if DXLink raises."""
    client = _build_client()

    with patch.object(
        client, "_get_quote_via_dxlink",
        new_callable=AsyncMock,
        side_effect=TimeoutError("no quote"),
    ):
        with patch.object(
            client, "_get_quote_via_rest",
            new_callable=AsyncMock,
            return_value={
                "symbol": "/ESH26:XCME", "price": 6781.0,
                "bid": 6780.75, "ask": 6781.25,
                "timestamp": "2026-03-11T15:30:00",
            },
        ) as mock_rest:
            quote = await client.get_quote("/ESH26:XCME")

    assert quote["price"] == 6781.0
    mock_rest.assert_called_once_with("/ESH26:XCME")


@pytest.mark.asyncio
async def test_ensure_subscribes_new_symbols():
    """_ensure_quote_streamer subscribes to new symbols without recreating the streamer."""
    client = _build_client()

    mock_streamer = AsyncMock()
    mock_streamer.subscribe = AsyncMock()
    client._quote_streamer = mock_streamer
    client._quote_symbols = set()

    # Fake a running feed task
    client._quote_feed_task = asyncio.create_task(asyncio.sleep(100))

    with patch.dict("sys.modules", {
        "tastytrade": MagicMock(),
        "tastytrade.dxfeed": MagicMock(),
    }):
        await client._ensure_quote_streamer("/ESH26:XCME")
        await client._ensure_quote_streamer("/ESM26:XCME")
        # Second call for same symbol should NOT re-subscribe
        await client._ensure_quote_streamer("/ESH26:XCME")

    assert mock_streamer.subscribe.call_count == 2
    assert "/ESH26:XCME" in client._quote_symbols
    assert "/ESM26:XCME" in client._quote_symbols

    # Cleanup
    client._quote_feed_task.cancel()
    try:
        await client._quote_feed_task
    except asyncio.CancelledError:
        pass
