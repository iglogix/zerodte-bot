"""Periodically polls TastyTrade for positions/account data, broadcasts via callback."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List

logger = logging.getLogger(__name__)


class MarketPoller:
    """Periodically polls TastyTrade for positions/account data, broadcasts via callback."""

    def __init__(
        self,
        api_client: Any,
        db: Any,
        on_positions: Callable[[List[Dict[str, Any]]], Coroutine],
        on_account: Callable[[Dict[str, Any]], Coroutine],
        position_interval: float = 10.0,
        account_interval: float = 30.0,
    ) -> None:
        self._api_client = api_client
        self._db = db
        self._on_positions = on_positions
        self._on_account = on_account
        self._position_interval = position_interval
        self._account_interval = account_interval
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start polling tasks."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._poll_positions()),
            asyncio.create_task(self._poll_account()),
        ]
        logger.info(
            "MarketPoller started (positions=%.1fs, account=%.1fs)",
            self._position_interval,
            self._account_interval,
        )

    async def stop(self) -> None:
        """Cancel polling tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("MarketPoller stopped")

    async def _poll_positions(self) -> None:
        """Loop: call db.get_open_trades(), enrich, call on_positions callback."""
        while self._running:
            try:
                trades = await self._db.get_open_trades()
                enriched = [self._enrich_trade(t) for t in trades]
                await self._on_positions(enriched)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("MarketPoller: position poll failed")
            await asyncio.sleep(self._position_interval)

    async def _poll_account(self) -> None:
        """Loop: call api_client.get_account_info(), call on_account callback."""
        while self._running:
            try:
                info = await self._api_client.get_account_info()
                await self._on_account(info)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("MarketPoller: account poll failed")
            await asyncio.sleep(self._account_interval)

    @staticmethod
    def _enrich_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dashboard-relevant fields from a trade dict."""
        strategy_data: Dict[str, Any] = {}
        for key in (
            "short_call_strike",
            "long_call_strike",
            "short_put_strike",
            "long_put_strike",
            "call_spread_premium",
            "put_spread_premium",
            "call_wing_width",
            "put_wing_width",
            "underlying_price_at_entry",
            "expiration_date",
            "buying_power_used",
            "stop_loss_amount",
        ):
            if key in trade:
                strategy_data[key] = trade[key]

        return {
            "trade_id": trade.get("trade_id"),
            "strategy_type": "iron_condor",
            "underlying": trade.get("underlying"),
            "total_credit": trade.get("total_premium"),
            "unrealized_pnl": trade.get("unrealized_pnl"),
            "status": trade.get("status"),
            "entry_time": str(trade["entry_time"]) if trade.get("entry_time") else None,
            "strategy_data": strategy_data,
        }
