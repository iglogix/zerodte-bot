"""
Position Monitor — runs as a background async loop alongside the entry scheduler.

Responsibilities (checked every `poll_interval_seconds`):
1. Take-profit: close any short option at or below $0.05
2. Trailing stop adjustment: tighten stop-loss as long legs decay
3. Stop-loss trigger detection: detect when a short leg has been stopped out,
   then immediately close the corresponding long leg (market order)
4. EOD forced close: close all open positions before market close (3:55 ET default)
5. Expired position cleanup: mark positions as expired if past 4:15 ET
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, date
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from trading_core.services.order_fill_listener import FillEvent

logger = logging.getLogger(__name__)

_MARKET_CLOSE = time(15, 55)
_EXPIRY_CUTOFF = time(16, 15)


class PositionMonitor:
    """
    Background loop that monitors all open iron condor positions and
    applies take-profit, trailing stop, and stop-loss cleanup logic.
    """

    def __init__(
        self,
        db=None,
        cache=None,
        tool_handlers=None,
        risk_manager=None,
        notifier=None,
        fill_queue=None,
        poll_interval_seconds: int = 30,
        take_profit_threshold: float = 0.05,
        eod_close_time: time = _MARKET_CLOSE,
        mode: str = "dry_run",
        eod_debrief=None,
    ):
        self.db = db
        self.cache = cache
        self.tool_handlers = tool_handlers
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.fill_queue = fill_queue  # asyncio.Queue fed by OrderFillListener
        self.poll_interval_seconds = poll_interval_seconds
        self.take_profit_threshold = take_profit_threshold
        self.eod_close_time = eod_close_time
        self.mode = mode
        self.eod_debrief = eod_debrief
        self._running = False
        self._stop_filled_symbols: set = set()  # symbols confirmed filled via WS
        self._eod_debrief_done: bool = False  # fire once per session

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    async def start(self):
        """Start the monitoring loop. Runs until stop() is called."""
        self._running = True
        logger.info(
            "PositionMonitor started — poll interval: %ds, mode: %s",
            self.poll_interval_seconds,
            self.mode.upper(),
        )
        while self._running:
            try:
                await self._run_cycle()
            except Exception:
                logger.exception("PositionMonitor cycle error")
            await asyncio.sleep(self.poll_interval_seconds)

    async def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        logger.info("PositionMonitor stopped")

    # -------------------------------------------------------------------------
    # Cycle
    # -------------------------------------------------------------------------

    async def _run_cycle(self):
        now = datetime.now()
        current_time = now.time()

        # --- Primary path: drain websocket fill events immediately ---
        await self._drain_fill_queue()

        trades = await self.db.get_open_trades()
        if not trades:
            return

        logger.debug("PositionMonitor: checking %d open trade(s)", len(trades))

        if current_time >= self.eod_close_time:
            await self._close_all_eod(trades)
            return

        for trade in trades:
            await self._check_trade(trade)

        if current_time >= _EXPIRY_CUTOFF:
            await self._mark_expired(trades)

    # -------------------------------------------------------------------------
    # Websocket fill queue drain (primary stop detection)
    # -------------------------------------------------------------------------

    async def _drain_fill_queue(self):
        """Process all pending fill events from OrderFillListener."""
        if not self.fill_queue:
            return
        while not self.fill_queue.empty():
            try:
                event = self.fill_queue.get_nowait()
                await self._handle_fill_event(event)
                self.fill_queue.task_done()
            except Exception:
                logger.exception("Error processing fill event from queue")
                break

    async def _handle_fill_event(self, event: "FillEvent"):
        """
        React to a confirmed order fill from the websocket.

        If a stop order on a short leg has been filled:
          1. Record the symbol so polling fallback skips it
          2. Find the matching trade and close the corresponding long leg
        """
        if not event.is_stop_fill or not event.is_fully_filled:
            logger.debug(
                "[FILL] Ignoring non-stop or partial fill: order_id=%s type=%s status=%s",
                event.order_id, event.order_type, event.status,
            )
            return

        logger.warning(
            "[FILL-STOP] Stop fill detected via websocket: symbol=%s order_id=%s price=%.2f",
            event.symbol, event.order_id, event.fill_price,
        )

        self._stop_filled_symbols.add(event.symbol)

        trades = await self.db.get_open_trades()
        for trade in trades:
            await self._close_long_for_stopped_short(trade, event.symbol)

    async def _close_long_for_stopped_short(self, trade: Dict[str, Any], stopped_symbol: str):
        """Given a stopped short symbol, close the paired long leg."""
        trade_id = trade.get("trade_id", "")
        pairs = [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]
        for short_key, long_key, side in pairs:
            if trade.get(short_key) != stopped_symbol:
                continue
            long_symbol = trade.get(long_key)
            if not long_symbol:
                continue
            logger.warning(
                "[FILL-STOP] Closing long %s leg %s for trade %s (stop filled via WS)",
                side, long_symbol, trade_id,
            )
            await self._close_leg(
                trade_id, long_symbol, f"long_{side}",
                reason="stop_filled_ws_close_long",
            )
            await self._notify_stop_triggered(trade_id, side, trade)

    # -------------------------------------------------------------------------
    # Per-trade checks (polling fallback)
    # -------------------------------------------------------------------------

    async def _check_trade(self, trade: Dict[str, Any]):
        trade_id = trade.get("trade_id", "")

        await self._check_take_profit(trade)
        await self._check_stop_triggered(trade)
        await self._update_trailing_stop(trade)

    async def _check_take_profit(self, trade: Dict[str, Any]):
        """Close any short leg that has decayed to <= $0.05."""
        trade_id = trade.get("trade_id", "")
        legs = [
            ("short_call", trade.get("short_call_symbol")),
            ("short_put", trade.get("short_put_symbol")),
        ]

        for leg_name, symbol in legs:
            if not symbol:
                continue
            quote = await self._get_quote(symbol)
            if quote is None:
                continue

            price = quote.get("price") or quote.get("ask") or 0.0
            if self.risk_manager and self.risk_manager.should_take_profit(price):
                logger.info(
                    "[TAKE-PROFIT] %s %s at $%.2f (threshold $%.2f)",
                    trade_id, leg_name, price, self.take_profit_threshold,
                )
                await self._close_leg(trade_id, symbol, leg_name, reason="take_profit")

    async def _check_stop_triggered(self, trade: Dict[str, Any]):
        """
        Fallback polling: detect if a short leg stop has been triggered.
        Skips symbols already handled by the websocket fill listener.
        """
        trade_id = trade.get("trade_id", "")
        pairs = [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

        for short_key, long_key, side in pairs:
            short_symbol = trade.get(short_key)
            long_symbol = trade.get(long_key)
            if not short_symbol or not long_symbol:
                continue

            # Skip if already handled by websocket fill event
            if short_symbol in self._stop_filled_symbols:
                logger.debug(
                    "[STOP-POLL] Skipping %s — already handled via websocket fill",
                    short_symbol,
                )
                continue

            stopped = await self._is_position_closed(short_symbol)
            if stopped:
                logger.warning(
                    "[STOP-POLL] %s %s short stopped out (detected by polling fallback) "
                    "— closing long leg %s",
                    trade_id, side, long_symbol,
                )
                self._stop_filled_symbols.add(short_symbol)
                await self._close_leg(
                    trade_id, long_symbol, f"long_{side}",
                    reason="stop_triggered_poll_fallback",
                )
                await self._notify_stop_triggered(trade_id, side, trade)

    async def _update_trailing_stop(self, trade: Dict[str, Any]):
        """
        Tighten stop-loss as long options decay.
        New stop = original_stop - (original_long_value - current_long_value)
        """
        if not self.risk_manager:
            return

        trade_id = trade.get("trade_id", "")
        pairs = [
            ("long_call_symbol", "short_call_stop_order_id", "call"),
            ("long_put_symbol", "short_put_stop_order_id", "put"),
        ]

        for long_key, stop_order_key, side in pairs:
            long_symbol = trade.get(long_key)
            stop_order_id = trade.get(stop_order_key)
            if not long_symbol:
                continue

            quote = await self._get_quote(long_symbol)
            if quote is None:
                continue

            current_long_value = quote.get("price") or quote.get("bid") or 0.0
            original_long_value = trade.get(f"original_{side}_long_value", current_long_value)
            original_stop = trade.get("stop_loss_amount", trade.get("total_premium", 0.0))

            new_stop = self.risk_manager.calculate_trailing_stop(
                original_stop_loss=original_stop,
                original_long_value=original_long_value,
                current_long_value=current_long_value,
            )

            if new_stop < original_stop * 0.95:
                logger.info(
                    "[TRAILING-STOP] %s %s side: tightening stop $%.2f → $%.2f",
                    trade_id, side, original_stop, new_stop,
                )
                await self._update_stop_order(trade_id, stop_order_id, new_stop, side)

    # -------------------------------------------------------------------------
    # EOD and expiry
    # -------------------------------------------------------------------------

    async def _close_all_eod(self, trades: List[Dict[str, Any]]):
        """Force-close all open positions before market close."""
        logger.warning(
            "[EOD] Market close approaching (%s) — closing %d open trade(s)",
            self.eod_close_time.strftime("%H:%M"),
            len(trades),
        )
        for trade in trades:
            trade_id = trade.get("trade_id", "")
            logger.info("[EOD] Closing trade %s", trade_id)
            result = await self.tool_handlers.close_position(
                trade_id=trade_id,
                close_all=True,
                order_type="market",
            )
            await self.db.update_trade(trade_id, {
                "status": "closed",
                "close_reason": "eod_forced_close",
                "exit_time": datetime.now(),
            })
            if self.notifier:
                await self.notifier.notify_trade_closed({
                    "trade_id": trade_id,
                    "reason": "eod_forced_close",
                    "result": result,
                })

        # Fire EOD debrief once per session after all positions closed
        if self.eod_debrief and not self._eod_debrief_done:
            self._eod_debrief_done = True
            logger.info("[EOD] Running EOD debrief...")
            try:
                await self.eod_debrief.run()
            except Exception:
                logger.exception("[EOD] Debrief failed")

    async def _mark_expired(self, trades: List[Dict[str, Any]]):
        """Mark any remaining open trades as expired after 4:15 ET."""
        for trade in trades:
            trade_id = trade.get("trade_id", "")
            await self.db.update_trade(trade_id, {
                "status": "expired",
                "close_reason": "option_expiry",
                "exit_time": datetime.now(),
            })
            logger.info("[EXPIRED] Trade %s marked as expired", trade_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    async def _get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return await self.cache.get_cached_quote(symbol, max_age_seconds=60)
        except Exception:
            logger.error("Could not fetch quote for %s", symbol)
            return None

    async def _is_position_closed(self, symbol: str) -> bool:
        """
        Check if a short option position has been closed (stop triggered).
        Returns True if the position no longer exists in open positions.
        """
        try:
            positions = await self.tool_handlers.get_open_positions()
            symbols = {p.get("symbol") for p in positions}
            return symbol not in symbols
        except Exception:
            return False

    async def _close_leg(
        self,
        trade_id: str,
        symbol: str,
        leg_name: str,
        reason: str,
    ):
        """Close a single option leg at market."""
        try:
            result = await self.tool_handlers.close_position(
                symbol=symbol,
                order_type="market",
                quantity=1,
            )
            logger.info(
                "[CLOSE-LEG] %s %s (%s): %s",
                trade_id, leg_name, reason, result,
            )
        except Exception:
            logger.exception("Failed to close leg %s for trade %s", leg_name, trade_id)

    async def _update_stop_order(
        self,
        trade_id: str,
        stop_order_id: Optional[str],
        new_stop: float,
        side: str,
    ):
        """Cancel existing stop and replace with tightened stop."""
        if not stop_order_id:
            return
        try:
            await self.tool_handlers.cancel_order(stop_order_id)
            logger.info("[TRAILING-STOP] Cancelled old stop %s", stop_order_id)
        except Exception:
            logger.error("Could not cancel stop order %s", stop_order_id)

    async def _notify_stop_triggered(
        self,
        trade_id: str,
        side: str,
        trade: Dict[str, Any],
    ):
        if self.notifier:
            try:
                await self.notifier.notify_trade_closed({
                    "trade_id": trade_id,
                    "reason": f"stop_triggered_{side}",
                    "trade": trade,
                })
            except Exception:
                logger.exception("Failed to send stop-triggered notification")
