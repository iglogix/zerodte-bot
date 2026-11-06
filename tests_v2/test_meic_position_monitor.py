"""PositionMonitor partial-close behaviour for MEIC."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from zerodte.services.position_monitor import PositionMonitor
from zerodte.strategies.base import ExitAction


@pytest.mark.asyncio
async def test_monitor_honours_leg_quantities():
    trade = {"trade_id": "T1", "short_call_symbol": "SC", "long_call_symbol": "LC"}

    db = MagicMock()
    db.get_open_trades = AsyncMock(return_value=[trade])
    db.get_trade = AsyncMock(return_value=None)
    db.update_trade = AsyncMock(return_value=True)

    cache = MagicMock()
    cache.get_cached_quote = AsyncMock(return_value={"bid": 0.5, "ask": 0.6})

    tool_handlers = MagicMock()
    tool_handlers.close_position = AsyncMock(return_value={"order_id": "X"})
    tool_handlers.get_spx_price = AsyncMock(return_value={"price": 5850.0})
    tool_handlers.get_open_positions = AsyncMock(return_value=[{"symbol": "SC"}])

    strategy = MagicMock()
    strategy.leg_pairs.return_value = [("short_call_symbol", "long_call_symbol", "call")]
    strategy.check_exit.return_value = ExitAction(
        action="take_profit",
        legs_to_close=["SC"],
        leg_quantities={"SC": 2},
    )
    strategy.compute_trailing_stop.return_value = None

    mon = PositionMonitor(
        db=db, cache=cache, tool_handlers=tool_handlers,
        strategy=strategy, mode="dry_run",
    )
    await mon._check_trade(trade)

    tool_handlers.close_position.assert_awaited()
    call_kwargs = tool_handlers.close_position.await_args.kwargs
    assert call_kwargs["symbol"] == "SC"
    assert call_kwargs["quantity"] == 2
