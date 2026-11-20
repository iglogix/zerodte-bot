"""Tests for WebSocket hub."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from web.ws_hub import WSHub


@pytest.fixture
def hub() -> WSHub:
    return WSHub()


def _mock_ws() -> MagicMock:
    ws = MagicMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect_and_disconnect(hub: WSHub) -> None:
    ws = _mock_ws()
    await hub.connect(ws)
    assert ws in hub.connections

    hub.disconnect(ws)
    assert ws not in hub.connections


@pytest.mark.asyncio
async def test_broadcast_sends_to_all(hub: WSHub) -> None:
    ws1 = _mock_ws()
    ws2 = _mock_ws()
    await hub.connect(ws1)
    await hub.connect(ws2)

    data = {"balance": 1000}
    await hub.broadcast("positions", data)

    expected = {"type": "positions", "data": data}
    ws1.send_json.assert_awaited_once_with(expected)
    ws2.send_json.assert_awaited_once_with(expected)


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections(hub: WSHub) -> None:
    alive = _mock_ws()
    dead = _mock_ws()
    dead.send_json = AsyncMock(side_effect=Exception("connection closed"))

    await hub.connect(alive)
    await hub.connect(dead)

    await hub.broadcast("heartbeat", {})

    assert alive in hub.connections
    assert dead not in hub.connections
    alive.send_json.assert_awaited_once()
