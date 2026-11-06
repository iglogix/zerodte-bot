"""
Tests for PositionMonitor — fill queue drain, stop detection, take-profit,
trailing stop, EOD close, and polling fallback deduplication.
"""

import asyncio
import pytest
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch, call

from agentic_ic.services.order_fill_listener import FillEvent
from agentic_ic.services.position_monitor import PositionMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    trade_id="T-001",
    short_call="SPXW  250117C05900000",
    long_call="SPXW  250117C05930000",
    short_put="SPXW  250117P05800000",
    long_put="SPXW  250117P05770000",
    stop_loss_amount=300.0,
    total_premium=300.0,
):
    return {
        "trade_id": trade_id,
        "short_call_symbol": short_call,
        "long_call_symbol": long_call,
        "short_put_symbol": short_put,
        "long_put_symbol": long_put,
        "stop_loss_amount": stop_loss_amount,
        "total_premium": total_premium,
        "original_call_long_value": 1.50,
        "original_put_long_value": 1.50,
        "short_call_stop_order_id": "SL-CALL-001",
        "short_put_stop_order_id": "SL-PUT-001",
    }


def _make_monitor(fill_queue=None, mode="dry_run", trades=None, risk_manager=None):
    db = AsyncMock()
    db.get_open_trades = AsyncMock(return_value=trades or [])
    db.update_trade = AsyncMock()

    cache = AsyncMock()
    cache.get_cached_quote = AsyncMock(return_value=None)

    tool_handlers = AsyncMock()
    tool_handlers.close_position = AsyncMock(return_value={"order_id": "CL-001"})
    tool_handlers.cancel_order = AsyncMock(return_value={"status": "cancelled"})
    tool_handlers.get_open_positions = AsyncMock(return_value=[])

    notifier = AsyncMock()
    notifier.notify_trade_closed = AsyncMock()

    monitor = PositionMonitor(
        db=db,
        cache=cache,
        tool_handlers=tool_handlers,
        risk_manager=risk_manager,
        notifier=notifier,
        fill_queue=fill_queue,
        poll_interval_seconds=30,
        take_profit_threshold=0.05,
        eod_close_time=time(15, 55),
        mode=mode,
    )
    return monitor


def _make_stop_fill_event(symbol, order_id="ORD-STOP-001"):
    return FillEvent(
        order_id=order_id,
        symbol=symbol,
        side="Buy",
        quantity=1,
        fill_price=42.50,
        status="Filled",
        order_type="Stop Limit",
    )


# ---------------------------------------------------------------------------
# Fill queue drain — primary path
# ---------------------------------------------------------------------------

class TestFillQueueDrain:
    @pytest.mark.asyncio
    async def test_drain_empty_queue_is_noop(self):
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q)
        await monitor._drain_fill_queue()  # should not raise

    @pytest.mark.asyncio
    async def test_drain_ignores_non_stop_fills(self):
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[_make_trade()])
        event = FillEvent(
            order_id="ORD-MKT",
            symbol="SPXW  250117C05900000",
            side="Sell",
            quantity=1,
            fill_price=10.0,
            status="Filled",
            order_type="Market",  # not a stop
        )
        await q.put(event)
        await monitor._drain_fill_queue()
        # No long leg close should have been called
        monitor.tool_handlers.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_ignores_partial_fills(self):
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[_make_trade()])
        event = FillEvent(
            order_id="ORD-PART",
            symbol="SPXW  250117C05900000",
            side="Sell",
            quantity=1,
            fill_price=42.50,
            status="Partially Filled",
            order_type="Stop Limit",
        )
        await q.put(event)
        await monitor._drain_fill_queue()
        monitor.tool_handlers.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_fill_closes_paired_long_call(self):
        trade = _make_trade()
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[trade])

        event = _make_stop_fill_event(symbol=trade["short_call_symbol"])
        await q.put(event)
        await monitor._drain_fill_queue()

        monitor.tool_handlers.close_position.assert_awaited_once()
        call_kwargs = monitor.tool_handlers.close_position.call_args
        assert trade["long_call_symbol"] in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_stop_fill_closes_paired_long_put(self):
        trade = _make_trade()
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[trade])

        event = _make_stop_fill_event(symbol=trade["short_put_symbol"])
        await q.put(event)
        await monitor._drain_fill_queue()

        monitor.tool_handlers.close_position.assert_awaited_once()
        call_kwargs = monitor.tool_handlers.close_position.call_args
        assert trade["long_put_symbol"] in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_stop_fill_records_symbol_in_filled_set(self):
        trade = _make_trade()
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[trade])

        event = _make_stop_fill_event(symbol=trade["short_call_symbol"])
        await q.put(event)
        await monitor._drain_fill_queue()

        assert trade["short_call_symbol"] in monitor._stop_filled_symbols

    @pytest.mark.asyncio
    async def test_multiple_fills_drained_in_one_cycle(self):
        trade = _make_trade()
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[trade])

        await q.put(_make_stop_fill_event(trade["short_call_symbol"], "ORD-001"))
        await q.put(_make_stop_fill_event(trade["short_put_symbol"], "ORD-002"))
        await monitor._drain_fill_queue()

        assert monitor.tool_handlers.close_position.await_count == 2

    @pytest.mark.asyncio
    async def test_fill_sends_notification(self):
        trade = _make_trade()
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[trade])

        event = _make_stop_fill_event(symbol=trade["short_call_symbol"])
        await q.put(event)
        await monitor._drain_fill_queue()

        monitor.notifier.notify_trade_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_fill_queue_drain_is_noop(self):
        monitor = _make_monitor(fill_queue=None)
        await monitor._drain_fill_queue()  # should not raise


# ---------------------------------------------------------------------------
# Polling fallback deduplication
# ---------------------------------------------------------------------------

class TestPollingFallbackDeduplication:
    @pytest.mark.asyncio
    async def test_polling_skips_ws_handled_symbol(self):
        trade = _make_trade()
        monitor = _make_monitor(trades=[trade])

        # Mark BOTH short legs as already handled by websocket
        monitor._stop_filled_symbols.add(trade["short_call_symbol"])
        monitor._stop_filled_symbols.add(trade["short_put_symbol"])

        # Polling would detect both as closed (positions gone)
        monitor.tool_handlers.get_open_positions = AsyncMock(return_value=[])

        await monitor._check_stop_triggered(trade)

        # close_position should NOT be called again for either leg
        monitor.tool_handlers.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_polling_acts_when_symbol_not_in_ws_set(self):
        trade = _make_trade()
        monitor = _make_monitor(trades=[trade])

        # Position is gone (stop triggered), not in ws set
        monitor.tool_handlers.get_open_positions = AsyncMock(return_value=[])

        await monitor._check_stop_triggered(trade)

        monitor.tool_handlers.close_position.assert_awaited()

    @pytest.mark.asyncio
    async def test_polling_adds_symbol_to_filled_set(self):
        trade = _make_trade()
        monitor = _make_monitor(trades=[trade])
        monitor.tool_handlers.get_open_positions = AsyncMock(return_value=[])

        await monitor._check_stop_triggered(trade)

        assert trade["short_call_symbol"] in monitor._stop_filled_symbols

    @pytest.mark.asyncio
    async def test_polling_no_action_when_position_still_open(self):
        trade = _make_trade()
        monitor = _make_monitor(trades=[trade])

        # Position still exists
        monitor.tool_handlers.get_open_positions = AsyncMock(return_value=[
            {"symbol": trade["short_call_symbol"]},
            {"symbol": trade["short_put_symbol"]},
        ])

        await monitor._check_stop_triggered(trade)
        monitor.tool_handlers.close_position.assert_not_called()


# ---------------------------------------------------------------------------
# Take-profit
# ---------------------------------------------------------------------------

class TestTakeProfit:
    @pytest.mark.asyncio
    async def test_take_profit_closes_short_leg_at_threshold(self):
        trade = _make_trade()
        risk_manager = MagicMock()
        risk_manager.should_take_profit = MagicMock(return_value=True)
        monitor = _make_monitor(trades=[trade], risk_manager=risk_manager)

        monitor.cache.get_cached_quote = AsyncMock(
            return_value={"price": 0.04}
        )

        await monitor._check_take_profit(trade)
        monitor.tool_handlers.close_position.assert_awaited()

    @pytest.mark.asyncio
    async def test_take_profit_no_action_above_threshold(self):
        trade = _make_trade()
        risk_manager = MagicMock()
        risk_manager.should_take_profit = MagicMock(return_value=False)
        monitor = _make_monitor(trades=[trade], risk_manager=risk_manager)

        monitor.cache.get_cached_quote = AsyncMock(
            return_value={"price": 0.50}
        )

        await monitor._check_take_profit(trade)
        monitor.tool_handlers.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_take_profit_skips_when_no_quote(self):
        trade = _make_trade()
        risk_manager = MagicMock()
        risk_manager.should_take_profit = MagicMock(return_value=True)
        monitor = _make_monitor(trades=[trade], risk_manager=risk_manager)

        monitor.cache.get_cached_quote = AsyncMock(return_value=None)

        await monitor._check_take_profit(trade)
        monitor.tool_handlers.close_position.assert_not_called()


# ---------------------------------------------------------------------------
# EOD forced close
# ---------------------------------------------------------------------------

class TestEodClose:
    @pytest.mark.asyncio
    async def test_eod_closes_all_trades(self):
        trades = [_make_trade("T-001"), _make_trade("T-002")]
        monitor = _make_monitor(trades=trades)

        await monitor._close_all_eod(trades)

        assert monitor.tool_handlers.close_position.await_count == 2
        assert monitor.db.update_trade.await_count == 2

    @pytest.mark.asyncio
    async def test_eod_sets_close_reason(self):
        trade = _make_trade()
        monitor = _make_monitor(trades=[trade])

        await monitor._close_all_eod([trade])

        update_call = monitor.db.update_trade.call_args
        assert update_call[0][1]["close_reason"] == "eod_forced_close"

    @pytest.mark.asyncio
    async def test_eod_notifies_for_each_trade(self):
        trades = [_make_trade("T-001"), _make_trade("T-002")]
        monitor = _make_monitor(trades=trades)

        await monitor._close_all_eod(trades)

        assert monitor.notifier.notify_trade_closed.await_count == 2


# ---------------------------------------------------------------------------
# Expiry cleanup
# ---------------------------------------------------------------------------

class TestExpiryCleanup:
    @pytest.mark.asyncio
    async def test_mark_expired_updates_status(self):
        trades = [_make_trade("T-001"), _make_trade("T-002")]
        monitor = _make_monitor(trades=trades)

        await monitor._mark_expired(trades)

        assert monitor.db.update_trade.await_count == 2
        for c in monitor.db.update_trade.call_args_list:
            assert c[0][1]["status"] == "expired"
            assert c[0][1]["close_reason"] == "option_expiry"


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

class TestTrailingStop:
    @pytest.mark.asyncio
    async def test_trailing_stop_tightens_when_long_decays(self):
        trade = _make_trade()
        risk_manager = MagicMock()
        # New stop is 10% tighter than original (triggers the 5% threshold)
        risk_manager.calculate_trailing_stop = MagicMock(return_value=270.0)
        monitor = _make_monitor(trades=[trade], risk_manager=risk_manager)

        monitor.cache.get_cached_quote = AsyncMock(
            return_value={"price": 0.50}  # long has decayed from 1.50
        )

        await monitor._update_trailing_stop(trade)

        monitor.tool_handlers.cancel_order.assert_awaited()

    @pytest.mark.asyncio
    async def test_trailing_stop_no_action_when_not_tighter_enough(self):
        trade = _make_trade()
        risk_manager = MagicMock()
        # New stop is only 2% tighter — below the 5% threshold
        risk_manager.calculate_trailing_stop = MagicMock(return_value=294.0)
        monitor = _make_monitor(trades=[trade], risk_manager=risk_manager)

        monitor.cache.get_cached_quote = AsyncMock(
            return_value={"price": 1.45}
        )

        await monitor._update_trailing_stop(trade)

        monitor.tool_handlers.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_trailing_stop_skipped_without_risk_manager(self):
        trade = _make_trade()
        monitor = _make_monitor(trades=[trade], risk_manager=None)

        await monitor._update_trailing_stop(trade)
        monitor.tool_handlers.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# Full cycle integration
# ---------------------------------------------------------------------------

class TestMonitorCycleIntegration:
    @pytest.mark.asyncio
    async def test_cycle_drains_queue_before_polling(self):
        """Fill queue drain happens before polling checks in each cycle."""
        trade = _make_trade()
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[trade])

        drain_calls = []
        original_drain = monitor._drain_fill_queue

        async def tracked_drain():
            drain_calls.append("drain")
            await original_drain()

        monitor._drain_fill_queue = tracked_drain
        monitor.tool_handlers.get_open_positions = AsyncMock(return_value=[
            {"symbol": trade["short_call_symbol"]},
            {"symbol": trade["short_put_symbol"]},
        ])

        with patch("agentic_ic.services.position_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 17, 10, 30, 0)
            await monitor._run_cycle()

        assert drain_calls == ["drain"]

    @pytest.mark.asyncio
    async def test_cycle_no_trades_returns_early(self):
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[])

        with patch("agentic_ic.services.position_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 17, 10, 30, 0)
            await monitor._run_cycle()

        monitor.tool_handlers.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_ws_fill_prevents_polling_double_close(self):
        """
        Simulate: WS fill arrives for short_call → long_call closed via WS path.
        Polling cycle runs → must NOT close long_call again.
        Put side is still open so polling correctly ignores it.
        """
        trade = _make_trade()
        q = asyncio.Queue()
        monitor = _make_monitor(fill_queue=q, trades=[trade])

        # WS fill event for short call only
        event = _make_stop_fill_event(symbol=trade["short_call_symbol"])
        await q.put(event)

        # Drain queue (WS path closes long call)
        await monitor._drain_fill_queue()
        first_close_count = monitor.tool_handlers.close_position.await_count
        assert first_close_count == 1

        # Polling runs — short_call gone, but short_put still open
        monitor.tool_handlers.get_open_positions = AsyncMock(return_value=[
            {"symbol": trade["short_put_symbol"]},
            {"symbol": trade["long_put_symbol"]},
        ])

        await monitor._check_stop_triggered(trade)

        # close_position count must NOT increase (short_call skipped, put still open)
        assert monitor.tool_handlers.close_position.await_count == first_close_count
