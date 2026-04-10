"""WebSocket hub for multiplexed broadcasts to connected clients."""

from __future__ import annotations

from typing import Any, List

from starlette.websockets import WebSocket


class WSHub:
    def __init__(self) -> None:
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg_type: str, data: Any) -> None:
        message = {"type": msg_type, "data": data}
        dead: list[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, msg_type: str, data: Any) -> None:
        await ws.send_json({"type": msg_type, "data": data})
