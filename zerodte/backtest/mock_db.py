"""In-memory trade store for backtesting."""

from __future__ import annotations

from typing import Any, Dict, List


class BacktestDb:
    """Minimal in-memory database satisfying PositionMonitor's interface."""

    def __init__(self):
        self._trades: Dict[str, Dict[str, Any]] = {}  # trade_id -> trade dict

    async def add_trade(self, trade: Dict[str, Any]) -> None:
        trade_id = trade.get("trade_id", "")
        self._trades[trade_id] = dict(trade)

    async def get_open_trades(self) -> List[Dict[str, Any]]:
        return [t for t in self._trades.values()
                if t.get("status") not in ("closed", "expired")]

    async def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> None:
        if trade_id in self._trades:
            self._trades[trade_id].update(updates)

    def get_all_trades(self) -> List[Dict[str, Any]]:
        return list(self._trades.values())

    async def get_open_trade_count(self) -> int:
        return len(await self.get_open_trades())
