"""
src_v2 Backtest Engine — runs LiveStrategy code against historical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class BacktestResult:
    trade_id: str
    session_date: date
    symbol: str

    # ORB
    orb_high: float
    orb_low: float
    orb_midline: float
    orb_range_pts: float

    bias: Optional[str] = None
    breakout_time: Optional[datetime] = None
    breakout_price: Optional[float] = None

    # Spread
    spread_type: Optional[str] = None
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None
    spread_width: float = 0.0

    # Entry
    entry_time: Optional[datetime] = None
    underlying_at_entry: Optional[float] = None
    credit_received: Optional[float] = None
    short_delta_at_entry: Optional[float] = None

    # Exit
    exit_time: Optional[datetime] = None
    underlying_at_exit: Optional[float] = None
    debit_paid: Optional[float] = None
    exit_reason: Optional[str] = None

    realized_pnl_usd: Optional[float] = None

    status: str = "skipped"
    skip_reason: Optional[str] = None

    qty: int = 1


# ---------------------------------------------------------------------------
# Event date sets (FOMC, CPI, NFP first-Friday heuristic)
# ---------------------------------------------------------------------------

_FOMC_DATES: set[date] = {
    # 2021
    date(2021,1,27), date(2021,3,17), date(2021,4,28), date(2021,6,16),
    date(2021,7,28), date(2021,9,22), date(2021,11,3), date(2021,12,15),
    # 2022
    date(2022,1,26), date(2022,3,16), date(2022,5,4), date(2022,6,15),
    date(2022,7,27), date(2022,9,21), date(2022,11,2), date(2022,12,14),
    # 2023
    date(2023,2,1), date(2023,3,22), date(2023,5,3), date(2023,6,14),
    date(2023,7,26), date(2023,9,20), date(2023,11,1), date(2023,12,13),
    # 2024
    date(2024,1,31), date(2024,3,20), date(2024,5,1), date(2024,6,12),
    date(2024,7,31), date(2024,9,18), date(2024,11,7), date(2024,12,18),
    # 2025
    date(2025,1,29), date(2025,3,19), date(2025,5,7), date(2025,6,18),
    date(2025,7,30), date(2025,9,17), date(2025,10,29), date(2025,12,10),
    # 2026
    date(2026,1,28), date(2026,3,18), date(2026,4,29), date(2026,6,17),
    date(2026,7,29), date(2026,9,16), date(2026,10,28), date(2026,12,9),
}

_CPI_DATES: set[date] = {
    # 2021
    date(2021,1,13), date(2021,2,10), date(2021,3,10), date(2021,4,13),
    date(2021,5,12), date(2021,6,10), date(2021,7,13), date(2021,8,11),
    date(2021,9,14), date(2021,10,13), date(2021,11,10), date(2021,12,10),
    # 2022
    date(2022,1,12), date(2022,2,10), date(2022,3,10), date(2022,4,12),
    date(2022,5,11), date(2022,6,10), date(2022,7,13), date(2022,8,10),
    date(2022,9,13), date(2022,10,13), date(2022,11,10), date(2022,12,13),
    # 2023
    date(2023,1,12), date(2023,2,14), date(2023,3,14), date(2023,4,12),
    date(2023,5,10), date(2023,6,13), date(2023,7,12), date(2023,8,10),
    date(2023,9,13), date(2023,10,12), date(2023,11,14), date(2023,12,12),
    # 2024
    date(2024,1,11), date(2024,2,13), date(2024,3,12), date(2024,4,10),
    date(2024,5,15), date(2024,6,12), date(2024,7,11), date(2024,8,14),
    date(2024,9,11), date(2024,10,10), date(2024,11,13), date(2024,12,11),
    # 2025
    date(2025,1,15), date(2025,2,12), date(2025,3,12), date(2025,4,10),
    date(2025,5,13), date(2025,6,11), date(2025,7,15), date(2025,8,12),
    date(2025,9,11), date(2025,10,24), date(2025,12,18),
    # 2026
    date(2026,1,13), date(2026,2,13), date(2026,3,11), date(2026,4,10),
    date(2026,5,12), date(2026,6,10), date(2026,7,14), date(2026,8,12),
    date(2026,9,11), date(2026,10,14), date(2026,11,10), date(2026,12,10),
}


def _is_nfp_friday(d: date) -> bool:
    """First Friday of the month (NFP release)."""
    return d.weekday() == 4 and d.day <= 7


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine synchronously."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Runs a LiveStrategy against historical 1-min bars + CBBO option data."""

    def __init__(self, config: dict, strategy):
        from src_v2.backtest.cbbo_store import CbboStore

        strat_cfg = config.get("strategy", {})
        opt_cfg = config.get("options", {})

        # ORB window
        self._orb_start = self._parse_time(strat_cfg.get("orb_window_start", "08:00"))
        self._orb_end = self._parse_time(strat_cfg.get("orb_window_end", "08:15"))
        self._market_open = self._parse_time(strat_cfg.get("market_open", "09:30"))
        self._eod = self._parse_time(strat_cfg.get("eod_force_close", "15:50"))
        self._midline_tol = strat_cfg.get("midline_tolerance_pts", 1.0)

        # Range filter
        rf = strat_cfg.get("orb_range_filter", {})
        self._range_min = rf.get("min_pts", 2.0)
        self._range_max = rf.get("max_pts", 30.0)

        # Event skips
        self._skip_nfp = strat_cfg.get("skip_nfp_fridays", False)
        self._skip_fomc = strat_cfg.get("skip_fomc_days", False)
        self._skip_cpi = strat_cfg.get("skip_cpi_days", False)

        # Options params
        self._option_root = opt_cfg.get("option_root", "SPXW")
        self._strike_step = opt_cfg.get("strike_step_pts", 5)
        self._tp_pct = opt_cfg.get("tp_pct_of_credit", 0.50)
        self._multiplier = opt_cfg.get("contract_multiplier", 100)
        self._quantity = opt_cfg.get("quantity", 1)

        # CBBO store
        cbbo_dir = config.get("backtest", {}).get("cbbo_cache_dir", "")
        self._cbbo = CbboStore(cbbo_dir, cache_dir=cbbo_dir)

        # Strategy
        self._strategy = strategy
        strategy.configure(config)

        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, days) -> list[BacktestResult]:
        """Run backtest across a list of DayBars. Returns list of BacktestResult."""
        all_results: list[BacktestResult] = []
        for day in days:
            d = day.session_date
            # Event skips
            if self._skip_nfp and _is_nfp_friday(d):
                all_results.append(self._skip_result(d, day.symbol, "nfp_friday"))
                continue
            if self._skip_fomc and d in _FOMC_DATES:
                all_results.append(self._skip_result(d, day.symbol, "fomc_day"))
                continue
            if self._skip_cpi and d in _CPI_DATES:
                all_results.append(self._skip_result(d, day.symbol, "cpi_day"))
                continue

            results = self._process_day(day)
            all_results.extend(results)
        return all_results

    # ------------------------------------------------------------------
    # Day processing
    # ------------------------------------------------------------------

    def _process_day(self, day) -> list[BacktestResult]:
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")

        d = day.session_date
        symbol = day.symbol
        bars = day.bars

        # 1. Build ORB candle
        orb_start_dt = datetime(d.year, d.month, d.day, self._orb_start[0], self._orb_start[1], tzinfo=ET)
        orb_end_dt = datetime(d.year, d.month, d.day, self._orb_end[0], self._orb_end[1], tzinfo=ET)
        orb_bars = [b for b in bars if orb_start_dt <= b["ts_et"] < orb_end_dt]

        if not orb_bars:
            return [self._skip_result(d, symbol, "no_orb_data")]

        orb_high = max(b["high"] for b in orb_bars)
        orb_low = min(b["low"] for b in orb_bars)
        orb_midline = (orb_high + orb_low) / 2
        orb_range = orb_high - orb_low

        # 2. Range filter
        if orb_range < self._range_min:
            return [self._skip_result(d, symbol, "range_too_narrow",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range)]
        if orb_range > self._range_max:
            return [self._skip_result(d, symbol, "range_too_wide",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range)]

        # 3. Find breakout in post-open bars
        market_open_dt = datetime(d.year, d.month, d.day, self._market_open[0], self._market_open[1], tzinfo=ET)
        eod_dt = datetime(d.year, d.month, d.day, self._eod[0], self._eod[1], tzinfo=ET)
        # Include all bars from market open (no EOD filter — exit simulation handles EOD)
        post_open = [b for b in bars if b["ts_et"] >= market_open_dt]

        bias = None
        breakout_bar = None
        breakout_idx = None
        for i, b in enumerate(post_open):
            if b["ts_et"] >= eod_dt:
                break
            if b["close"] > orb_high:
                bias = "long"
                breakout_bar = b
                breakout_idx = i
                break
            elif b["close"] < orb_low:
                bias = "short"
                breakout_bar = b
                breakout_idx = i
                break

        if bias is None:
            return [self._skip_result(d, symbol, "no_breakout",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range)]

        # 4. Find midline retest after breakout
        retest_bar = None
        retest_idx = None
        for i in range(breakout_idx + 1, len(post_open)):
            b = post_open[i]
            if b["ts_et"] >= eod_dt:
                break
            if b["low"] <= orb_midline + self._midline_tol and b["high"] >= orb_midline - self._midline_tol:
                retest_bar = b
                retest_idx = i
                break

        if retest_bar is None:
            return [self._skip_result(d, symbol, "no_retest",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range,
                                      bias=bias, breakout_time=breakout_bar["ts_et"],
                                      breakout_price=breakout_bar["close"])]

        # 5. Fill bar = bar after retest
        fill_idx = retest_idx + 1
        if fill_idx >= len(post_open):
            return [self._skip_result(d, symbol, "no_fill_bar",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range)]

        fill_bar = post_open[fill_idx]
        fill_time = fill_bar["ts_et"]
        fill_minute = fill_time.strftime("%H:%M")

        # 6. Load CBBO for this day
        n_quotes = self._cbbo.load_day(d)
        if n_quotes == 0:
            return [self._skip_result(d, symbol, "no_cbbo_data",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range)]

        # 7. Derive spot from CBBO
        spot = self._cbbo.derive_spot(fill_minute, fill_bar["close"])
        if spot is None:
            spot = fill_bar["close"]

        # 8. Build chain from CBBO
        chain = self._build_chain(fill_minute, spot, d)
        if not chain:
            return [self._skip_result(d, symbol, "no_chain",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range)]

        # 9. Set bias on strategy
        self._strategy._orb_bias = bias

        # 10. Search candidates
        result = _run_async(self._strategy.search_candidates(chain, spot))
        setup = result.get("top_candidate")
        if setup is None:
            return [self._skip_result(d, symbol, "no_candidates",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range,
                                      bias=bias)]

        # 11. Validate
        val = _run_async(self._strategy.validate(setup, chain))
        if not val.get("passed", False):
            return [self._skip_result(d, symbol, "validation_failed",
                                      orb_high=orb_high, orb_low=orb_low,
                                      orb_midline=orb_midline, orb_range_pts=orb_range,
                                      bias=bias)]

        # Assign symbols to legs
        for leg in setup.legs:
            leg.symbol = self._make_symbol(d, leg.strike, leg.option_type)

        # 12. Build per-side trade dicts for check_exit
        # Each side gets its own trade dict so check_exit only evaluates
        # that side's TP/breach — prevents one side blocking the other.
        put_trade = {
            "short_put_symbol": setup.legs[0].symbol,
            "long_put_symbol": setup.legs[1].symbol,
            "short_put_strike": setup.strikes["short_put"],
            "put_credit": setup.extra["put_credit"],
        }
        call_trade = {
            "short_call_symbol": setup.legs[2].symbol,
            "long_call_symbol": setup.legs[3].symbol,
            "short_call_strike": setup.strikes["short_call"],
            "call_credit": setup.extra["call_credit"],
        }

        # 13. Simulate exits for each side independently
        monitoring_bars = post_open[fill_idx + 1:]
        put_result = self._simulate_side(
            "put", put_trade, setup, monitoring_bars, d, symbol,
            fill_time, spot, orb_high, orb_low, orb_midline, orb_range, eod_dt,
        )
        call_result = self._simulate_side(
            "call", call_trade, setup, monitoring_bars, d, symbol,
            fill_time, spot, orb_high, orb_low, orb_midline, orb_range, eod_dt,
        )
        return [put_result, call_result]

    # ------------------------------------------------------------------
    # Exit simulation
    # ------------------------------------------------------------------

    def _simulate_side(
        self, side: str, trade: dict, setup, monitoring_bars: list,
        d: date, symbol: str, fill_time: datetime, spot: float,
        orb_high: float, orb_low: float, orb_midline: float,
        orb_range: float, eod_dt: datetime,
    ) -> BacktestResult:
        if side == "put":
            credit = setup.extra["put_credit"]
            short_strike = setup.strikes["short_put"]
            long_strike = setup.strikes["long_put"]
            width = short_strike - long_strike
            short_sym = trade["short_put_symbol"]
            long_sym = trade["long_put_symbol"]
        else:
            credit = setup.extra["call_credit"]
            short_strike = setup.strikes["short_call"]
            long_strike = setup.strikes["long_call"]
            width = long_strike - short_strike
            short_sym = trade["short_call_symbol"]
            long_sym = trade["long_call_symbol"]

        base = BacktestResult(
            trade_id=f"{d.isoformat()}_{side}",
            session_date=d,
            symbol=symbol,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_midline=orb_midline,
            orb_range_pts=orb_range,
            bias=setup.extra.get("orb_bias"),
            breakout_time=fill_time,
            breakout_price=spot,
            spread_type=side,
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=width,
            entry_time=fill_time,
            underlying_at_entry=spot,
            credit_received=credit,
            qty=self._quantity,
            status="closed",
        )

        # Walk monitoring bars
        hold_put_breach = getattr(self._strategy, '_hold_put_breach', False)

        for bar in monitoring_bars:
            bar_time = bar["ts_et"]

            # EOD: expire worthless
            if bar_time >= eod_dt:
                base.exit_time = bar_time
                base.underlying_at_exit = bar["close"]
                base.debit_paid = 0.0
                base.exit_reason = "expired_worthless"
                base.realized_pnl_usd = round(credit * self._multiplier * self._quantity, 2)
                return base

            bar_minute = bar_time.strftime("%H:%M")

            # Breach detection using bar high/low (matches src_8am)
            # Put breach: bar low touches short put strike
            # Call breach: bar high touches short call strike
            breached = False
            if side == "put" and bar["low"] <= short_strike:
                if hold_put_breach:
                    continue  # put breach held — skip bar entirely
                breached = True
            elif side == "call" and bar["high"] >= short_strike:
                breached = True

            if breached:
                # Compute debit from CBBO at breach bar
                s_q = self._cbbo.lookup(bar_minute, short_strike, side == "call")
                l_q = self._cbbo.lookup(bar_minute, long_strike, side == "call")
                if s_q and l_q:
                    s_mid = (s_q[0] + s_q[1]) / 2
                    l_mid = (l_q[0] + l_q[1]) / 2
                    debit = min(max(s_mid - l_mid, 0), width)
                else:
                    debit = width
                base.exit_time = bar_time
                base.underlying_at_exit = bar["close"]
                base.debit_paid = round(debit, 2)
                base.exit_reason = "short_breached"
                base.realized_pnl_usd = round((credit - debit) * self._multiplier * self._quantity, 2)
                return base

            # TP check via strategy.check_exit (uses CBBO spread marks)
            bar_spot = self._cbbo.derive_spot(bar_minute, bar["close"])
            if bar_spot is None:
                bar_spot = bar["close"]

            quotes = self._build_quotes(bar_minute, trade)
            if not quotes:
                continue  # skip bar if no CBBO quotes available
            quotes["_underlying"] = {"price": bar_spot}

            action = self._strategy.check_exit(trade, quotes)

            if action.action == "take_profit":
                our_syms = {short_sym, long_sym}
                if our_syms.intersection(set(action.legs_to_close)):
                    debit = round(credit * (1 - self._tp_pct), 2)
                    base.exit_time = bar_time
                    base.underlying_at_exit = bar_spot
                    base.debit_paid = debit
                    base.exit_reason = f"tp_{int(self._tp_pct * 100)}pct"
                    base.realized_pnl_usd = round((credit - debit) * self._multiplier * self._quantity, 2)
                    return base

        # No exit found — expire worthless
        base.exit_time = None
        base.debit_paid = 0.0
        base.exit_reason = "expired_worthless"
        base.realized_pnl_usd = round(credit * self._multiplier * self._quantity, 2)
        return base

    # ------------------------------------------------------------------
    # Chain / quotes building
    # ------------------------------------------------------------------

    def _build_chain(self, minute_et: str, spot: float, d: date) -> list[dict]:
        """Build option chain from CBBO data at a given minute."""
        chain = []
        step = self._strike_step
        low_k = int((spot - 200) / step) * step
        high_k = int((spot + 200) / step) * step

        for k in range(low_k, high_k + step, step):
            strike = float(k)
            for is_call, otype in [(False, "put"), (True, "call")]:
                q = self._cbbo.lookup(minute_et, strike, is_call)
                if q is None:
                    continue
                bid, ask = q
                chain.append({
                    "strike": strike,
                    "option_type": otype,
                    "bid": bid,
                    "ask": ask,
                    "symbol": self._make_symbol(d, strike, otype),
                })
        return chain

    def _build_quotes(self, minute_et: str, trade: dict) -> dict:
        """Build quotes dict for check_exit from CBBO at given minute.

        Only includes symbols present in the trade dict (per-side trade dicts
        only have 2 leg symbols, not all 4).
        """
        quotes = {}
        for key in ["short_put_symbol", "long_put_symbol",
                     "short_call_symbol", "long_call_symbol"]:
            sym = trade.get(key)
            if not sym:
                continue
            strike, is_call = self._parse_symbol(sym)
            if strike is None:
                continue
            q = self._cbbo.lookup(minute_et, strike, is_call)
            if q:
                quotes[sym] = {"bid": q[0], "ask": q[1]}
        return quotes

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def _make_symbol(self, d: date, strike: float, option_type: str) -> str:
        """Build OCC-format symbol: SPXW  YYMMDDCSSSSSSSS."""
        root = self._option_root.ljust(6)
        dt_str = d.strftime("%y%m%d")
        cp = "C" if option_type.lower() in ("call", "c") else "P"
        # Strike in 1/1000ths: 5900.0 -> 05900000
        strike_str = f"{int(strike * 1000):08d}"
        return f"{root}{dt_str}{cp}{strike_str}"

    @staticmethod
    def _parse_symbol(sym: str) -> tuple:
        """Extract (strike, is_call) from OCC symbol."""
        try:
            cp_char = sym[12]
            strike_raw = int(sym[13:])
            return float(strike_raw) / 1000.0, cp_char == "C"
        except (IndexError, ValueError):
            return None, None

    @staticmethod
    def _parse_time(t_str: str) -> tuple[int, int]:
        """Parse 'HH:MM' to (hour, minute) tuple."""
        parts = t_str.split(":")
        return int(parts[0]), int(parts[1])

    # ------------------------------------------------------------------
    # Skip result factory
    # ------------------------------------------------------------------

    def _skip_result(
        self, d: date, symbol: str, reason: str, *,
        orb_high: float = 0, orb_low: float = 0,
        orb_midline: float = 0, orb_range_pts: float = 0,
        bias: str | None = None,
        breakout_time: datetime | None = None,
        breakout_price: float | None = None,
    ) -> BacktestResult:
        return BacktestResult(
            trade_id=f"{d.isoformat()}_skip",
            session_date=d,
            symbol=symbol,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_midline=orb_midline,
            orb_range_pts=orb_range_pts,
            bias=bias,
            breakout_time=breakout_time,
            breakout_price=breakout_price,
            status="skipped",
            skip_reason=reason,
        )
