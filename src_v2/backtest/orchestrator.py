"""
BacktestOrchestrator — runs real OrbEngine + PositionMonitor against historical data.

Unlike BacktestEngine which reimplements the strategy logic inline,
this orchestrator uses the actual OrbEngine and PositionMonitor classes,
ensuring backtest behavior matches live trading exactly.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src_v2.backtest.engine import BacktestResult, _FOMC_DATES, _CPI_DATES, _is_nfp_friday
from src_v2.backtest.sim_clock import SimClock
from src_v2.backtest.bt_tool_handlers import BacktestToolHandlers
from src_v2.backtest.mock_db import BacktestDb
from src_v2.backtest.mock_cache import BacktestCache
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.agent.orb_engine import OrbEngine
from src_v2.agent.base_engine import EngineDeps
from src_v2.services.position_monitor import PositionMonitor

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class BacktestOrchestrator:
    """Drives real OrbEngine + PositionMonitor against historical bars + CBBO."""

    def __init__(self, config: Dict[str, Any], strategy_cls, cbbo_store: CbboStore):
        self._config = config
        self._strategy_cls = strategy_cls
        self._cbbo = cbbo_store

        strat = config.get("strategy", {})
        self._skip_nfp = strat.get("skip_nfp_fridays", False)
        self._skip_fomc = strat.get("skip_fomc_days", False)
        self._skip_cpi = strat.get("skip_cpi_days", False)
        self._skip_fomc_next = strat.get("skip_fomc_next_day", False)
        self._skip_cpi_next = strat.get("skip_cpi_next_day", False)
        # weekday skip list — e.g. ["Fri"] to skip Fridays.
        # Accepts short ("Mon"-"Sun") or numeric (0=Mon..6=Sun).
        raw_wd = strat.get("skip_weekdays", []) or []
        _name_to_idx = {n: i for i, n in enumerate(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])}
        self._skip_weekdays = set()
        for w in raw_wd:
            if isinstance(w, int):
                self._skip_weekdays.add(w)
            elif isinstance(w, str) and w[:3].title() in _name_to_idx:
                self._skip_weekdays.add(_name_to_idx[w[:3].title()])
        self._market_open_str = strat.get("market_open", "09:30")
        # Running cbbo-derived prior-session close forward, for gap_box strikes.
        self._prev_close_forward: float | None = None

        opt = config.get("options", {})
        self._multiplier = opt.get("contract_multiplier", 100)
        self._qty = opt.get("quantity", 1)
        self._slippage = opt.get("slippage_per_spread", 0.10)
        # Exit fill realism: fraction of each leg's bid/ask spread crossed when
        # closing. 0.0 = mid (legacy/optimistic); 1.0 = worst case (buy short back
        # at ask, sell long at bid). Breach exits widen near ITM → this bites there.
        self._exit_fill_frac = opt.get("exit_fill_frac", 0.0)
        # Commission per contract per leg (e.g. TT SPX: $1 open only, $0 close)
        self._commission_per_leg = opt.get("commission_per_contract_per_leg", 1.0)
        self._eod_str = strat.get("eod_force_close", "15:50")

    def _calendar_skip(self, sd) -> Optional[str]:
        """Return a skip reason for calendar gates, or None to trade."""
        from datetime import timedelta as _td
        if self._skip_nfp and _is_nfp_friday(sd):
            return "nfp_friday"
        if self._skip_fomc and sd in _FOMC_DATES:
            return "fomc_day"
        if self._skip_cpi and sd in _CPI_DATES:
            return "cpi_day"
        if self._skip_fomc_next and (sd - _td(days=1)) in _FOMC_DATES:
            return "fomc_next_day"
        if self._skip_cpi_next and (sd - _td(days=1)) in _CPI_DATES:
            return "cpi_next_day"
        if sd.weekday() in self._skip_weekdays:
            return f"weekday_skip_{sd.strftime('%a')}"
        return None

    def run(self, days) -> List[BacktestResult]:
        """Run backtest across all days. Returns list of BacktestResult."""
        results = []
        for day in sorted(days, key=lambda d: d.session_date):
            sd = day.session_date
            skip_reason = self._calendar_skip(sd)
            if skip_reason:
                results.append(self._skip(sd, day.symbol, skip_reason))
            else:
                day_results = asyncio.run(self._run_day(day))
                results.extend(day_results)
            # Update prior-close forward for the NEXT day's gap_box strikes —
            # done on every day (incl. skips) so e.g. Monday after a skipped
            # Friday still sees Friday's close. cbbo is cached in memory.
            if self._cbbo.loaded_date != sd:
                self._cbbo.load_day(sd)
            sf = self._cbbo.settlement_forward()
            if sf:
                self._prev_close_forward = sf
        return results

    async def _run_day(self, day) -> List[BacktestResult]:
        sd = day.session_date
        sym = day.symbol
        bars = day.bars

        if not bars:
            return [self._skip(sd, sym, "no_bars")]

        # Load CBBO
        n = self._cbbo.load_day(sd)
        if n == 0:
            return [self._skip(sd, sym, "no_cbbo_data")]

        # Create components
        clock = SimClock(bars[0]["ts_et"])
        db = BacktestDb()
        bt_handlers = BacktestToolHandlers(
            bars, self._cbbo, clock, db, sd,
            slippage=self._slippage,
        )
        cache = BacktestCache(self._cbbo, clock)

        # Clock advance callback: advance to next bar
        clock.set_advance_fn(lambda: bt_handlers.advance_bar())

        # Skip pre-market bars: advance bar pointer to market_open
        # (ORB window data is read by get_orb_levels from historical bars by time,
        # so we don't need the pointer to be within the ORB window).
        mo_h, mo_m = [int(x) for x in self._market_open_str.split(":")]
        for i, b in enumerate(bars):
            bt = b["ts_et"]
            if bt.hour > mo_h or (bt.hour == mo_h and bt.minute >= mo_m):
                bt_handlers.set_bar_index(i)
                clock.advance_to(b["ts_et"])
                break

        # Create strategy
        strategy = self._strategy_cls()
        strategy.configure(self._config)

        # Inject gap-adjusted EM-box anchors (Pulver): today's open + prior
        # close, both cbbo-derived (same scale as the chain). Only OpenIcStrategy
        # gap_box mode reads these; harmless for other strategies.
        if hasattr(strategy, "_gap_open"):
            strategy._gap_prev_close = self._prev_close_forward
            hint = self._prev_close_forward or 5000.0
            strategy._gap_open = self._cbbo.derive_spot("09:31", hint)

        # Create EngineDeps
        deps = EngineDeps(
            tool_handlers=bt_handlers,
            risk_manager=None,
            anomaly_detector=None,
            regime_classifier=None,
            decision_file_logger=None,
            db=None,
            mode="backtest",
            config=self._config,
            clock=clock,
        )

        # Run the engine the strategy requires (OrbEngine, WindowEntryEngine, etc.)
        if hasattr(strategy, "create_engine"):
            engine = strategy.create_engine(deps)
        else:
            engine = OrbEngine(deps, strategy)
        result = await engine.run()

        if result.outcome != "executed":
            return [self._skip(sd, sym, result.reason)]

        # Entry was placed -- run PositionMonitor for exit
        eod_h, eod_m = [int(x) for x in self._eod_str.split(":")]
        eod_time = time(eod_h, eod_m)

        monitor = PositionMonitor(
            db=db,
            cache=cache,
            tool_handlers=bt_handlers,
            strategy=strategy,
            clock=clock,
            fill_queue=None,
            notifier=None,
            eod_debrief=None,
            eod_close_time=eod_time,
            mode="backtest",
        )

        # INVARIANT: an executed entry must leave post-entry session for the
        # monitor to manage. The engine runs to completion BEFORE this loop, and
        # every sim-clock sleep advances time — so an engine that idles to EOD
        # (as the live one does, to outlive run_scheduled) hands over a clock
        # already at/after the close. The monitor's first cycle then goes
        # straight to _close_all_eod and every intraday exit (take-profit,
        # breach) silently disappears, inflating results.
        #
        # Checked on the CLOCK, not on bars remaining: the engine leaves bars in
        # the list, so a has_bars_remaining() check looks fine and catches
        # nothing. Verified by removing the backtest no-op and confirming this
        # raises.
        entry_time_et = clock.now(ET)
        if entry_time_et.time() >= eod_time:
            raise RuntimeError(
                f"{sd}: entry executed but the sim clock is already at "
                f"{entry_time_et.time()} (EOD {eod_time}) — the engine consumed "
                "the post-entry session (see OrbEngine._idle_until_eod, which "
                "must no-op in backtest). Take-profit and breach exits would be "
                "skipped entirely and P&L booked at the final bar."
            )

        # Walk remaining bars through PositionMonitor until EOD
        while bt_handlers.has_bars_remaining():
            bt_handlers.advance_bar()
            await monitor._run_cycle()

        # After main loop: for any leg still without exit_log (e.g. abandoned
        # side after SL closed the trade), record its final CBBO mark at last bar.
        now = clock.now(ET)
        for trade in db.get_all_trades():
            for key in ("short_put_symbol", "long_put_symbol",
                        "short_call_symbol", "long_call_symbol"):
                leg = trade.get(key)
                if leg and leg not in bt_handlers._exit_log:
                    bt_handlers._record_exit_quote(leg, now, reason="expired_worthless")

        # Collect results
        return self._collect_results(db, sd, sym, bt_handlers)

    def _collect_results(
        self,
        db: BacktestDb,
        sd: date,
        sym: str,
        bt_handlers: BacktestToolHandlers,
    ) -> List[BacktestResult]:
        """Convert BacktestDb trades to BacktestResult list."""
        trades = db.get_all_trades()
        if not trades:
            return [self._skip(sd, sym, "no_trades_recorded")]

        results = []
        for trade in trades:
            for side in ("put", "call"):
                short_k = trade.get(f"short_{side}_strike")
                if short_k is None:
                    continue
                long_k = trade.get(f"long_{side}_strike")  # may be None for naked shorts (strangles)
                has_long = long_k is not None

                credit = trade.get(f"{side}_credit", 0)
                width = abs(short_k - long_k) if has_long else 0.0

                # Symbols were built at entry; the trade dict may have them cleared if closed.
                # Reconstruct from strikes + date for exit_log lookup.
                short_sym = bt_handlers._make_symbol(short_k, side == "call")
                long_sym = bt_handlers._make_symbol(long_k, side == "call") if has_long else None

                # Per-leg exit info from exit_log
                s_exit = bt_handlers._exit_log.get(short_sym)
                l_exit = bt_handlers._exit_log.get(long_sym) if has_long else None

                # Close debit: spread = max(0, s_mid - l_mid) capped at width;
                # naked short = max(0, s_mid) uncapped.
                if s_exit and (l_exit or not has_long):
                    f = self._exit_fill_frac
                    s_mid = (s_exit["bid"] + s_exit["ask"]) / 2
                    # buy short back: pay between mid and ask
                    s_fill = s_mid + f * (s_exit["ask"] - s_mid)
                    if has_long:
                        l_mid = (l_exit["bid"] + l_exit["ask"]) / 2
                        # sell long: receive between mid and bid
                        l_fill = l_mid - f * (l_mid - l_exit["bid"])
                        debit = round(min(max(s_fill - l_fill, 0), width), 2)
                    else:
                        debit = round(max(s_fill, 0), 2)
                    reason = s_exit.get("reason", "closed")
                    exit_time = s_exit.get("time")
                    if "take_profit" in reason or "tp" in reason.lower():
                        exit_reason = "take_profit"
                    elif "stop_loss" in reason or "breach" in reason.lower():
                        exit_reason = "short_breached"
                    elif "eod" in reason:
                        # At EOD, if mark is worthless it expired; otherwise forced close
                        exit_reason = "expired_worthless" if debit < 0.05 else "eod_forced_close"
                    else:
                        exit_reason = reason or "closed"
                else:
                    # No exit log = expired worthless (never closed)
                    debit = 0.0
                    exit_reason = "expired_worthless"
                    exit_time = trade.get("exit_time")

                # Commission: 2 legs/side for spreads, 1 leg/side for naked shorts.
                # (TT SPX: $1 open only, $0 close → models opening commission)
                legs_per_side = 2 if has_long else 1
                commission = legs_per_side * self._qty * self._commission_per_leg
                pnl = round((credit - debit) * self._multiplier * self._qty - commission, 2)

                spread_type = f"{side}_credit_spread" if has_long else f"{side}_short"

                orb = bt_handlers._orb_levels or {}
                results.append(BacktestResult(
                    trade_id=f"{trade.get('trade_id', '')}-{side[0].upper()}",
                    session_date=sd,
                    symbol=sym,
                    orb_high=orb.get("high", 0),
                    orb_low=orb.get("low", 0),
                    orb_midline=orb.get("midline", 0),
                    orb_range_pts=orb.get("range_pts", 0),
                    bias=bt_handlers._breakout_bias,
                    entry_time=trade.get("entry_time"),
                    underlying_at_entry=trade.get("entry_price"),
                    spread_type=spread_type,
                    short_strike=short_k,
                    long_strike=long_k if has_long else 0,
                    spread_width=width,
                    credit_received=credit,
                    exit_time=exit_time,
                    debit_paid=debit,
                    exit_reason=exit_reason,
                    realized_pnl_usd=pnl,
                    status="closed",
                    qty=self._qty,
                ))

        return results if results else [self._skip(sd, sym, "no_results")]

    def _compute_debit(
        self,
        bt_handlers: BacktestToolHandlers,
        short_sym: str,
        long_sym: str,
        width: float,
    ) -> float:
        """Compute close debit from exit log entries."""
        s_exit = bt_handlers._exit_log.get(short_sym)
        l_exit = bt_handlers._exit_log.get(long_sym)

        if s_exit and l_exit:
            s_mid = (s_exit["bid"] + s_exit["ask"]) / 2
            l_mid = (l_exit["bid"] + l_exit["ask"]) / 2
            debit = max(s_mid - l_mid, 0)
            return round(min(debit, width), 2)

        # No exit log = expired worthless
        return 0.0

    def _skip(self, sd: date, sym: str, reason: str) -> BacktestResult:
        return BacktestResult(
            trade_id=f"V2-{sd}-SKIP",
            session_date=sd,
            symbol=sym,
            orb_high=0,
            orb_low=0,
            orb_midline=0,
            orb_range_pts=0,
            status="skipped",
            skip_reason=reason,
        )
