"""Phase 5 TDD tests: PositionMonitor delegates exit logic to strategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import time
from zerodte.strategies.base import ExitAction


def _make_monitor(strategy_action="hold", trades=None):
    """Build PositionMonitor with mocked deps and strategy."""
    from zerodte.services.position_monitor import PositionMonitor

    strategy = MagicMock()
    strategy.check_exit.return_value = ExitAction(action=strategy_action)
    strategy.compute_trailing_stop.return_value = None
    strategy.leg_pairs.return_value = [
        ("short_call_symbol", "long_call_symbol", "call"),
        ("short_put_symbol", "long_put_symbol", "put"),
    ]

    db = AsyncMock()
    db.get_open_trades.return_value = trades or []
    db.update_trade = AsyncMock()

    cache = AsyncMock()
    cache.get_cached_quote = AsyncMock(return_value={"bid": 0.50, "ask": 0.60, "price": 0.55})

    tool_handlers = AsyncMock()
    tool_handlers.close_position = AsyncMock(return_value={"status": "closed"})
    tool_handlers.get_open_positions = AsyncMock(return_value=[])

    monitor = PositionMonitor(
        db=db,
        cache=cache,
        tool_handlers=tool_handlers,
        strategy=strategy,
        mode="dry_run",
    )
    return monitor


class TestMonitorDelegatesToStrategy:
    @pytest.mark.asyncio
    async def test_calls_strategy_check_exit(self):
        trade = {"trade_id": "T1", "short_call_symbol": "SC", "short_put_symbol": "SP"}
        monitor = _make_monitor(strategy_action="hold", trades=[trade])
        await monitor._check_trade(trade)
        monitor.strategy.check_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_legs_on_take_profit(self):
        trade = {
            "trade_id": "T1",
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
        }
        monitor = _make_monitor(trades=[trade])
        monitor.strategy.check_exit.return_value = ExitAction(
            action="take_profit",
            legs_to_close=["SC"],
            reason="short call at $0.05",
        )
        await monitor._check_trade(trade)
        monitor.tool_handlers.close_position.assert_called()

    @pytest.mark.asyncio
    async def test_closes_all_on_stop_loss(self):
        trade = {
            "trade_id": "T1",
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
        }
        monitor = _make_monitor(trades=[trade])
        monitor.strategy.check_exit.return_value = ExitAction(
            action="stop_loss",
            legs_to_close=["SC", "LC", "SP", "LP"],
            reason="wing breach",
        )
        await monitor._check_trade(trade)
        assert monitor.tool_handlers.close_position.call_count == 4

    @pytest.mark.asyncio
    async def test_hold_does_nothing(self):
        trade = {"trade_id": "T1", "short_call_symbol": "SC", "short_put_symbol": "SP"}
        monitor = _make_monitor(strategy_action="hold", trades=[trade])
        await monitor._check_trade(trade)
        monitor.tool_handlers.close_position.assert_not_called()


class TestMonitorTrailingStop:
    @pytest.mark.asyncio
    async def test_calls_strategy_trailing_stop(self):
        trade = {"trade_id": "T1", "short_call_symbol": "SC", "short_put_symbol": "SP"}
        monitor = _make_monitor(trades=[trade])
        monitor.strategy.compute_trailing_stop.return_value = None
        await monitor._check_trade(trade)
        monitor.strategy.compute_trailing_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_update_when_none_returned(self):
        trade = {"trade_id": "T1", "short_call_symbol": "SC", "short_put_symbol": "SP"}
        monitor = _make_monitor(trades=[trade])
        monitor.strategy.compute_trailing_stop.return_value = None
        await monitor._check_trade(trade)
        monitor.tool_handlers.cancel_order.assert_not_called()


class TestMonitorStopDetection:
    @pytest.mark.asyncio
    async def test_uses_strategy_leg_pairs(self):
        trade = {
            "trade_id": "T1",
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
        }
        monitor = _make_monitor(trades=[trade])
        # Simulate SC stopped out (not in open positions)
        monitor.tool_handlers.get_open_positions.return_value = [
            {"symbol": "LC"}, {"symbol": "SP"}, {"symbol": "LP"},
        ]
        await monitor._check_stop_triggered(trade)
        monitor.strategy.leg_pairs.assert_called()


class TestMonitorEodClose:
    @pytest.mark.asyncio
    async def test_eod_close_is_generic(self):
        """EOD close doesn't depend on strategy."""
        trade = {"trade_id": "T1"}
        monitor = _make_monitor(trades=[trade])
        await monitor._close_all_eod([trade])
        monitor.tool_handlers.close_position.assert_called_once()
        monitor.db.update_trade.assert_called_once()
