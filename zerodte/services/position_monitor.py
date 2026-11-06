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

_CONTRACT_MULTIPLIER = 100.0
_SHORT_LEG_KEYS = ("short_put_symbol", "short_call_symbol")
_LONG_LEG_KEYS = ("long_put_symbol", "long_call_symbol")


class TradeOutlivedExpiry(RuntimeError):
    """A trade is being closed on a later date than the option it holds expires.

    On a 0DTE book this is never benign: the contract already settled, so the
    close is booking a fiction. Two ORB trades did exactly this
    (orbdir-call-093125 07-09 -> 07-10, orbdir-call-144640 07-29 -> 07-30) and
    both were recorded as an ordinary `eod_forced_close`; the first was a
    deterministic max loss.
    """


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
        strategy=None,
        notifier=None,
        fill_queue=None,
        poll_interval_seconds: int = 30,
        take_profit_threshold: float = 0.05,
        eod_close_time: time = _MARKET_CLOSE,
        mode: str = "dry_run",
        eod_debrief=None,
        clock=None,
    ):
        self.db = db
        self.cache = cache
        self.tool_handlers = tool_handlers
        self.risk_manager = risk_manager
        self.strategy = strategy
        self.notifier = notifier
        self.fill_queue = fill_queue  # asyncio.Queue fed by OrderFillListener
        self.poll_interval_seconds = poll_interval_seconds
        self.take_profit_threshold = take_profit_threshold
        self.eod_close_time = eod_close_time
        self.mode = mode
        self.eod_debrief = eod_debrief
        from zerodte.backtest.sim_clock import WallClock
        self._clock = clock or WallClock()
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
            await self._clock.sleep(self.poll_interval_seconds)

    async def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        logger.info("PositionMonitor stopped")

    # -------------------------------------------------------------------------
    # Cycle
    # -------------------------------------------------------------------------

    async def _run_cycle(self):
        now = self._clock.now()
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
        pairs = self.strategy.leg_pairs() if self.strategy else [
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

        # Delegate exit decision to strategy
        if self.strategy:
            quotes = await self._get_all_quotes(trade)
            exit_action = self.strategy.check_exit(trade, quotes)

            if exit_action.action == "take_profit":
                logger.info("[TAKE-PROFIT] %s: %s", trade_id, exit_action.reason)
                results = []
                for sym in exit_action.legs_to_close:
                    qty = exit_action.leg_quantities.get(sym, 1)
                    results.append(await self._close_leg(
                        trade_id, sym, "take_profit_leg",
                        reason="take_profit", quantity=qty,
                        exit_quote=quotes.get(sym)))
                # Take-profit closes every leg, so the trade is done — record
                # that. Without it the row stays "open" forever: the monitor
                # re-fires take-profit on every poll, and the entry concurrency
                # guard sees a phantom open trade and blocks the NEXT session.
                if results and all(self._leg_is_flat(r) for r in results):
                    await self._finalize_trade(trade, exit_action.reason, results)
                    logger.info("[TAKE-PROFIT] %s closed out", trade_id)
                else:
                    logger.warning(
                        "[TAKE-PROFIT] %s: not all legs confirmed flat — leaving "
                        "the trade open for the next cycle", trade_id,
                    )
                return

            if exit_action.action == "stop_loss":
                logger.warning("[STOP-LOSS] %s: %s", trade_id, exit_action.reason)
                results = []
                for sym in exit_action.legs_to_close:
                    qty = exit_action.leg_quantities.get(sym, 1)
                    results.append(await self._close_leg(
                        trade_id, sym, "stop_loss_leg",
                        reason="stop_loss", quantity=qty,
                        exit_quote=quotes.get(sym)))
                # Mark trade fully closed only if ALL legs are now closed.
                # Otherwise leave it open so the remaining side keeps being monitored
                # (intraday breaches / TP / trailing logic still apply).
                refreshed = await self.db.get_trade(trade_id) if hasattr(self.db, "get_trade") else None
                if refreshed is None:
                    open_trades = await self.db.get_open_trades()
                    refreshed = next((t for t in open_trades if t.get("trade_id") == trade_id), None)
                if refreshed is None or not self._any_legs_remain(refreshed):
                    await self._finalize_trade(trade, exit_action.reason, results)
                return

            # Trailing stop update
            new_stop = self.strategy.compute_trailing_stop(trade, quotes)
            if new_stop is not None:
                stop_order_id = trade.get("stop_order_id")
                if stop_order_id:
                    logger.info("[TRAILING-STOP] %s: tightening to $%.2f", trade_id, new_stop)
                    await self._update_stop_order(trade_id, stop_order_id, new_stop, "combined")
        else:
            # Legacy fallback: IC-specific per-leg checks
            await self._check_take_profit_legacy(trade)

        await self._check_stop_triggered(trade)

    async def _check_take_profit_legacy(self, trade: Dict[str, Any]):
        """Legacy IC per-leg TP check — used only when no strategy is set."""
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
        pairs = self.strategy.leg_pairs() if self.strategy else [
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
            leg_results = result.get("results") if isinstance(result, dict) else None
            await self._finalize_trade(trade, "eod_forced_close", leg_results)
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
            await self._finalize_trade(trade, "option_expiry", None,
                                       status="expired")
            logger.info("[EXPIRED] Trade %s marked as expired", trade_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    async def _get_all_quotes(self, trade: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Quotes for every leg of a trade, keyed by the trade's leg symbol.

        Resolved from the 0DTE CHAIN by (strike, right), which is the same
        source the strategy prices entries from. The old per-symbol lookup went
        to `get_cached_quote`, an in-memory dict that only ever holds "SPX" and
        the ES futures symbol — an option leg was never in it, so this returned
        {} and check_exit's take-profit branch bailed on `mid is None` every
        time. Take-profit could therefore never fire, in any mode; positions
        only ever left via breach or the EOD close.

        QuoteComparator already avoids per-symbol lookups for this reason: they
        "break on OCC-vs-DXLink symbology".
        """
        from zerodte.backtest.osi_symbol import parse_osi

        symbols = [s for s in (
            trade.get(k) for k in ("short_call_symbol", "long_call_symbol",
                                   "short_put_symbol", "long_put_symbol")
        ) if s]

        quotes: Dict[str, Dict[str, Any]] = {}
        by_strike = await self._chain_by_strike() if symbols else {}
        for sym in symbols:
            opt = None
            try:
                strike, is_call = parse_osi(sym.strip().ljust(21))
                opt = by_strike.get((float(strike), is_call))
            except Exception:
                logger.debug("Could not resolve %s from the chain", sym)
            if opt is None:
                opt = await self._get_quote(sym)  # legacy cached path
            if opt:
                quotes[sym] = opt

        if len(quotes) < len(symbols):
            logger.warning(
                "Only %d/%d leg quotes resolved — take-profit cannot be "
                "evaluated this cycle", len(quotes), len(symbols),
            )

        # Add underlying price if available
        try:
            spx = await self.tool_handlers.get_spx_price()
            if spx and "price" in spx:
                quotes["_underlying"] = spx
        except Exception:
            pass
        return quotes

    async def _chain_by_strike(self) -> Dict[tuple, Dict[str, Any]]:
        """0DTE chain keyed by (strike, is_call)."""
        if self.tool_handlers is None:
            return {}
        try:
            chain = await self.tool_handlers.get_option_chain(expiration="0DTE")
        except Exception:
            logger.exception("Option chain fetch failed — cannot price legs")
            return {}
        out: Dict[tuple, Dict[str, Any]] = {}
        for o in chain or []:
            is_call = str(o.get("option_type", o.get("type", ""))).lower().startswith("c")
            out[(float(o.get("strike", 0) or 0), is_call)] = o
        return out

    async def _get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return await self.cache.get_cached_quote(symbol, max_age_seconds=60)
        except Exception:
            logger.error("Could not fetch quote for %s", symbol)
            return None

    async def _is_position_closed(self, symbol: str) -> bool:
        """Has this short leg been closed (i.e. its stop triggered)?

        Answered from the BROKER's positions. It previously compared the leg
        symbol against `get_open_positions()`, which returns local trade rows
        with no `symbol` key — so the set was {None}, every leg looked closed,
        and the first poll after entry declared a false stop-out and tried to
        dump the hedge leg. Invisible in manual mode (orders are not submitted);
        it fired the moment orders became real.

        Returns False when the broker cannot be asked: never claim a stop-out
        that has not been verified. A missed detection just leaves the position
        under normal TP/breach/EOD management; a false one sells the hedge.
        """
        try:
            positions = await self.tool_handlers.get_broker_positions()
            if positions is None:
                return False
            # Trade records hold the space-padded 21-char OSI; brokers report
            # the compact form. Normalise both sides before comparing.
            held = {str(p.get("symbol", "")).replace(" ", "") for p in positions}
            return symbol.replace(" ", "") not in held
        except Exception:
            logger.exception("Could not determine whether %s is closed", symbol)
            return False

    # -------------------------------------------------------------------------
    # Close bookkeeping — every close MUST funnel through _finalize_trade
    # -------------------------------------------------------------------------

    @staticmethod
    def _norm_symbol(symbol: Any) -> str:
        """OCC symbols reach us in two spellings for the same contract.

        The trade record holds the padded OCC form (`SPXW  260826C07680000`)
        while Alpaca echoes the unpadded one (`SPXW260826C07680000`). Matching
        raw would book nothing for every Alpaca trade.
        """
        return str(symbol).replace(" ", "").upper() if symbol else ""

    @staticmethod
    def _trade_legs(trade: Dict[str, Any]) -> Dict[str, int]:
        """Leg symbol -> sign, where +1 is bought back and -1 is sold out.

        Closing a credit spread pays the short leg and collects the long one, so
        exit debit = sum(short fills) - sum(long fills).
        """
        legs: Dict[str, int] = {}
        for key in _SHORT_LEG_KEYS:
            if trade.get(key):
                legs[PositionMonitor._norm_symbol(trade[key])] = 1
        for key in _LONG_LEG_KEYS:
            if trade.get(key):
                legs[PositionMonitor._norm_symbol(trade[key])] = -1
        return legs

    @staticmethod
    def _close_quantity(trade: Dict[str, Any]) -> int:
        for key in ("quantity", "call_open_qty", "put_open_qty"):
            qty = trade.get(key)
            if isinstance(qty, (int, float)) and qty > 0:
                return int(qty)
        return 1

    @staticmethod
    def _realized_pnl(
        trade: Dict[str, Any],
        leg_results: Optional[List[Any]],
        multiplier: float = _CONTRACT_MULTIPLIER,
    ) -> Optional[float]:
        """(entry credit - exit debit) x multiplier x qty, or None if unknowable.

        Returns None rather than a plausible number whenever the close carries
        no price: manual/dry_run (nothing was submitted), an already-flat leg,
        a missing entry credit, or a leg that reports more than one fill. A NULL
        P&L is recoverable from the execution log; a fabricated one is not
        recoverable at all, and silently becomes "performance".
        """
        credit = trade.get("total_premium")
        if not isinstance(credit, (int, float)):
            return None

        legs = PositionMonitor._trade_legs(trade)
        if not legs:
            return None

        fills: Dict[str, float] = {}
        for result in leg_results or []:
            if not isinstance(result, dict):
                continue
            symbol = PositionMonitor._norm_symbol(
                result.get("leg_symbol") or result.get("symbol")
            )
            price = result.get("fill_price")
            if symbol not in legs or not isinstance(price, (int, float)):
                continue
            if symbol in fills:
                logger.warning(
                    "Refusing to book P&L for %s: leg %s reported multiple "
                    "fills", trade.get("trade_id", ""), symbol,
                )
                return None
            fills[symbol] = float(price)

        if set(fills) != set(legs):
            return None

        exit_debit = sum(fills[sym] * sign for sym, sign in legs.items())
        qty = PositionMonitor._close_quantity(trade)
        # Round to cents: option prices are 2dp, so anything finer is float
        # noise that later reads as precision in a currency column.
        return round((float(credit) - exit_debit) * multiplier * qty, 2)

    @staticmethod
    def _as_date(value: Any) -> Optional[date]:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None

    @staticmethod
    def _check_not_past_expiry(trade: Dict[str, Any], now: datetime) -> None:
        """Raise if this close lands on a later date than the option's expiry."""
        expiry = PositionMonitor._as_date(trade.get("expiration_date"))
        closed_on = PositionMonitor._as_date(now)
        if expiry is None or closed_on is None or closed_on <= expiry:
            return
        raise TradeOutlivedExpiry(
            f"{trade.get('trade_id', '')} expired {expiry.isoformat()} but is "
            f"being closed {closed_on.isoformat()} — it survived past its own "
            f"expiration and any P&L booked here is fiction"
        )

    async def _finalize_trade(
        self,
        trade: Dict[str, Any],
        close_reason: str,
        leg_results: Optional[List[Any]] = None,
        status: str = "closed",
    ) -> None:
        """The single funnel every trade close passes through.

        Kept as one primitive rather than three copies of the same dict literal
        because the guards below were previously absent from ALL of them, and a
        guard that lives in one branch gets forgotten by the next.
        """
        trade_id = trade.get("trade_id", "")
        now = self._clock.now()

        try:
            self._check_not_past_expiry(trade, now)
        except TradeOutlivedExpiry as exc:
            # Loud, but never fatal: killing the monitor loop would strand every
            # other open position. Record it on the row so it cannot be mistaken
            # for a normal close after the fact.
            logger.error("[EXPIRY-INVARIANT] %s", exc)
            close_reason = f"{close_reason} [PAST_EXPIRY]"

        updates: Dict[str, Any] = {
            "status": status,
            "close_reason": close_reason,
            "exit_time": now,
        }
        pnl = self._realized_pnl(trade, leg_results)
        if pnl is None:
            logger.info(
                "[PNL] %s closed with no bookable P&L (no fill prices) — "
                "realized_pnl left NULL", trade_id,
            )
        else:
            updates["realized_pnl"] = pnl
            logger.info("[PNL] %s realized $%.2f", trade_id, pnl)

        await self.db.update_trade(trade_id, updates)

    @staticmethod
    def _leg_is_flat(result: Any) -> bool:
        """Is this leg confirmed no longer held after a close attempt?

        Covers a real fill, a leg that was already gone (`skipped`), and the
        simulated results manual/dry_run return instead of submitting.
        A raised exception yields None, which is NOT flat.
        """
        if not isinstance(result, dict):
            return False
        return bool(
            result.get("filled")
            or result.get("skipped")
            or result.get("status") == "no_position"
            or result.get("manual")
            or result.get("dry_run")
        )

    @staticmethod
    def _any_legs_remain(trade: Dict[str, Any]) -> bool:
        """True if any short/long leg symbol on the trade is still set."""
        return any(trade.get(k) for k in (
            "short_put_symbol", "long_put_symbol",
            "short_call_symbol", "long_call_symbol",
        ))

    async def _close_leg(
        self,
        trade_id: str,
        symbol: str,
        leg_name: str,
        reason: str,
        quantity: int = 1,
        exit_quote: Optional[Dict[str, Any]] = None,
    ):
        """Close a single option leg at market.

        `exit_quote` is the leg's bid/ask at the moment the exit was decided —
        passed through so the execution log can score the fill against the quote
        it actually crossed. Without it an exit fill is an unanchored number.
        """
        try:
            result = await self.tool_handlers.close_position(
                symbol=symbol,
                trade_id=trade_id,
                order_type="market",
                quantity=quantity,
                reason=reason,
                exit_quote=exit_quote,
            )
            logger.info(
                "[CLOSE-LEG] %s %s (%s): %s",
                trade_id, leg_name, reason, result,
            )
            if isinstance(result, dict):
                # Stamp the symbol we ASKED to close. The broker's echo uses its
                # own spelling, so P&L booking must not depend on it.
                result = {**result, "leg_symbol": symbol}
            return result
        except Exception:
            logger.exception("Failed to close leg %s for trade %s", leg_name, trade_id)
            return None

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
