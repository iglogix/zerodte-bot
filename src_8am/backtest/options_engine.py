"""
ORB Options Backtest Engine — simulates SPX 0DTE credit spreads using ES bar data.

No options price data needed. We:
  1. Run the same ORB entry logic (midline retest) as the futures backtest.
  2. At entry time, price the spread via Black-Scholes (ES price ≈ SPX, fixed IV assumption).
  3. Place the short strike at the closest 5-pt increment below/above target delta.
  4. Walk subsequent 1-min bars:
     - If underlying crosses the short strike → compute spread value at that moment (BS) → record exit.
     - TP check: spread mark decays to 50% of credit → exit (assumes smooth theta decay).
     - EOD: spread expires worthless if short strike not breached → max profit.
  5. P&L in dollars: (credit - debit) * 100 per contract.

Assumptions / limitations:
  - IV is fixed per session (configurable, default 15%).
  - No bid-ask spread or slippage on option fills.
  - ES futures price used as proxy for SPX (typically ~1-5 pts difference, negligible for strike placement).
  - Gamma/vega effects are modelled through BS re-pricing at exit time.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    """Standard normal CDF — Abramowitz & Stegun approximation."""
    a = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * a)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (
        1.781477937 + t * (-1.821255978 + t * 1.330274429)
    )))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
    return cdf if x >= 0 else 1.0 - cdf


def _bs_call(s: float, k: float, t: float, iv: float) -> float:
    """Black-Scholes call price (r=0, no dividends)."""
    if t <= 0:
        return max(0.0, s - k)
    sv = iv * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / sv
    d2 = d1 - sv
    return s * _ncdf(d1) - k * _ncdf(d2)


def _bs_put(s: float, k: float, t: float, iv: float) -> float:
    """Black-Scholes put price (r=0)."""
    if t <= 0:
        return max(0.0, k - s)
    sv = iv * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / sv
    d2 = d1 - sv
    return k * _ncdf(-d2) - s * _ncdf(-d1)


def _bs_call_delta(s: float, k: float, t: float, iv: float) -> float:
    if t <= 0:
        return 1.0 if s > k else 0.0
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * math.sqrt(t))
    return _ncdf(d1)


def _bs_put_delta(s: float, k: float, t: float, iv: float) -> float:
    return _bs_call_delta(s, k, t, iv) - 1.0


def _select_strike(
    s: float, target_delta: float, iv: float, t: float,
    spread_type: str, step: float = 5.0
) -> float:
    """
    Find nearest strike (at step increments) where |delta| ≈ target_delta.
    For put_credit_spread: search OTM puts below s.
    For call_credit_spread: search OTM calls above s.
    """
    best_k, best_diff = s, 1.0
    if spread_type == "put_credit_spread":
        k = math.floor(s / step) * step
        while k > s - 200:
            d = abs(_bs_put_delta(s, k, t, iv))
            diff = abs(d - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_k = k
            if d < target_delta * 0.3:
                break
            k -= step
    else:
        k = math.ceil(s / step) * step
        while k < s + 200:
            d = _bs_call_delta(s, k, t, iv)
            diff = abs(d - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_k = k
            if d < target_delta * 0.3:
                break
            k += step
    return best_k


def _spread_credit(
    s: float, short_k: float, long_k: float, t: float, iv: float, spread_type: str
) -> float:
    """Net credit = short option mid - long option mid (no b/a spread in backtest)."""
    if spread_type == "put_credit_spread":
        return _bs_put(s, short_k, t, iv) - _bs_put(s, long_k, t, iv)
    else:
        return _bs_call(s, short_k, t, iv) - _bs_call(s, long_k, t, iv)


def _spread_value(
    s: float, short_k: float, long_k: float, t: float, iv: float, spread_type: str
) -> float:
    """Current mark-to-market value of the spread (cost to close)."""
    return max(0.0, _spread_credit(s, short_k, long_k, t, iv, spread_type))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptionsBacktestResult:
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
    spread_width: float = 5.0
    iv_assumed: float = 0.15

    # Entry
    entry_time: Optional[datetime] = None
    underlying_at_entry: Optional[float] = None
    credit_received: Optional[float] = None     # per-share (e.g. 0.75)
    max_profit_usd: Optional[float] = None       # credit * 100
    max_loss_usd: Optional[float] = None         # (width - credit) * 100
    short_delta_at_entry: Optional[float] = None

    # Exit
    exit_time: Optional[datetime] = None
    underlying_at_exit: Optional[float] = None
    debit_paid: Optional[float] = None           # cost to close
    exit_reason: Optional[str] = None            # expired_worthless|tp_50pct|sl_breach|eod_close|short_breached

    realized_pnl_usd: Optional[float] = None    # (credit - debit) * 100 * qty

    status: str = "skipped"
    skip_reason: Optional[str] = None

    qty: int = 1


# ---------------------------------------------------------------------------
# Options Backtest Engine
# ---------------------------------------------------------------------------

class OrbOptionsBacktestEngine:
    """
    Backtests the ORB credit spread strategy using ES 1-min bar data.

    Strategy recap:
      LONG  bias → sell PUT  credit spread (short strike below ORB low)
      SHORT bias → sell CALL credit spread (short strike above ORB high)

    Exit logic (in priority order):
      1. EOD (15:50 ET): spread expires worthless → max profit.
      2. Intraday short-strike breach: compute spread value at breach bar → record loss/profit.
      3. TP at 50% of credit: detected by checking BS spread mark on each bar.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config

        strat = cfg.get("strategy", {})
        orb_start = strat.get("orb_window_start", "08:00")
        orb_end_s = strat.get("orb_window_end", "08:15")
        market_open_s = strat.get("market_open", "09:30")
        eod_s = strat.get("eod_force_close", "15:50")

        self._orb_start = time(*[int(x) for x in orb_start.split(":")])
        self._orb_end = time(*[int(x) for x in orb_end_s.split(":")])
        self._market_open = time(*[int(x) for x in market_open_s.split(":")])
        self._eod = time(*[int(x) for x in eod_s.split(":")])

        range_cfg = strat.get("orb_range_filter", {})
        self._min_range = range_cfg.get("min_pts", 2.0)
        self._max_range = range_cfg.get("max_pts", 30.0)

        self._midline_tol = strat.get("midline_tolerance_pts", 0.50)
        self._skip_nfp = strat.get("skip_nfp_fridays", False)

        opt = cfg.get("options", {})
        self._target_delta = opt.get("target_short_delta", 0.16)
        self._spread_width = opt.get("spread_width_pts", 5.0)
        self._min_credit = opt.get("min_credit_per_spread", 0.50)
        self._tp_pct = opt.get("tp_pct_of_credit", 0.50)
        self._iv = opt.get("assumed_iv", 0.15)

        # Configurable data sources for more accurate backtesting
        # underlying_source: "es" (use ES bars directly) or "spy" (use SPY*10 as SPX proxy)
        # iv_source: "fixed" (use assumed_iv) or "vix" (use daily VIX9D/VIX)
        self._underlying_source = opt.get("underlying_source", "es")
        self._iv_source = opt.get("iv_source", "fixed")
        self._data_dir = cfg.get("backtest", {}).get("data_dir", "data")
        self._market_data = None  # Lazy-loaded

        risk = cfg.get("risk", {})
        self._max_loss_usd = risk.get("max_loss_per_trade_usd", 500)
        self._qty = cfg.get("bot", {}).get("quantity", 1)

        self._market_close_time = time(16, 0)   # actual market close
        self._session_end_minutes = 6.5 * 60     # trading hours per day

    # -----------------------------------------------------------------------
    # Market data helpers
    # -----------------------------------------------------------------------

    def _get_market_data(self):
        """Lazy-load market data loader for SPY/VIX data."""
        if self._market_data is None:
            from src_8am.backtest.market_data import MarketDataLoader
            self._market_data = MarketDataLoader(data_dir=self._data_dir)
        return self._market_data

    def _get_underlying_price(self, session_date: date, es_price: float) -> float:
        """
        Get the underlying price to use for option pricing.

        If underlying_source is "spy", returns SPY*10 (SPX proxy).
        Otherwise returns the ES price as-is.
        """
        if self._underlying_source == "spy":
            md = self._get_market_data()
            spy_data = md.get_spy_ohlc(session_date)
            if spy_data:
                return spy_data["close"]  # Already scaled by 10
        return es_price

    def _get_session_iv(self, session_date: date) -> float:
        """
        Get the IV to use for option pricing.

        If iv_source is "vix", returns daily VIX9D/VIX (as decimal).
        Otherwise returns the fixed assumed_iv.
        """
        if self._iv_source == "vix":
            md = self._get_market_data()
            return md.get_daily_iv(session_date, fallback=self._iv)
        return self._iv

    def _is_nfp_friday(self, session_date: date) -> bool:
        """
        Check if this is an NFP Friday (first Friday of the month).

        NFP (Non-Farm Payrolls) is released at 8:30 AM ET on the first Friday
        of each month, causing high volatility right after the ORB window.
        Backtesting shows 31.6% loss rate on these days vs 9.2% overall.
        """
        if session_date.weekday() != 4:  # Not Friday
            return False
        # First Friday: day 1-7
        return session_date.day <= 7

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        zip_path: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[OptionsBacktestResult]:
        from src_8am.backtest.data_loader import DatabentoLoader
        loader = DatabentoLoader(zip_path)
        days = list(loader.iter_days(start_date=start_date, end_date=end_date))
        return self.run_on_days(days)

    def run_on_days(self, days) -> List[OptionsBacktestResult]:
        """Run backtest on a pre-loaded list of DayBars (avoids re-parsing zip)."""
        # Group by date and select highest-volume contract per day
        # This handles roll periods where multiple contracts trade simultaneously
        from collections import defaultdict

        by_date: Dict[date, List] = defaultdict(list)
        for day_bars in days:
            by_date[day_bars.session_date].append(day_bars)

        results = []
        for session_date in sorted(by_date.keys()):
            # Skip NFP Fridays (first Friday of month) - high loss rate due to 8:30 AM release
            if self._skip_nfp and self._is_nfp_friday(session_date):
                continue

            candidates = by_date[session_date]

            # Select contract with highest volume (most liquid)
            best_day = max(candidates, key=lambda d: sum(b["volume"] for b in d.bars))

            # Skip AM expiration contracts (3rd Friday, contract expires at 9:30 AM)
            # These have symbol like ESH4 expiring in March - check if bars end early
            if self._is_am_expiration(best_day):
                # Try to use next contract instead
                non_expiring = [d for d in candidates if not self._is_am_expiration(d)]
                if non_expiring:
                    best_day = max(non_expiring, key=lambda d: sum(b["volume"] for b in d.bars))
                else:
                    continue  # Skip this day entirely

            result = self._process_day(best_day)
            if result is not None:
                results.append(result)

        return results

    def _is_am_expiration(self, day_bars) -> bool:
        """
        Check if this is an AM expiration day for this contract.

        ES futures expire at 9:30 AM ET on the 3rd Friday of contract month.
        We detect this by checking if the contract's bars end before market hours.
        """
        if not day_bars.bars:
            return False

        symbol = day_bars.symbol
        session_date = day_bars.session_date

        # ES contract months: H=Mar, M=Jun, U=Sep, Z=Dec
        contract_months = {"H": 3, "M": 6, "U": 9, "Z": 12}

        # Extract contract month from symbol (e.g., ESH4 -> H)
        if len(symbol) >= 3 and symbol[2] in contract_months:
            contract_month = contract_months[symbol[2]]

            # Check if this is the expiration month
            if session_date.month == contract_month:
                # 3rd Friday check: day 15-21 and Friday
                if 15 <= session_date.day <= 21 and session_date.weekday() == 4:
                    # Verify bars end early (before 10:00 AM typical for AM exp)
                    last_bar_time = day_bars.bars[-1]["ts_et"].time()
                    if last_bar_time.hour < 10:
                        return True

        return False

    # -----------------------------------------------------------------------
    # Day processing
    # -----------------------------------------------------------------------

    def _process_day(self, day_bars) -> Optional[OptionsBacktestResult]:
        from src_8am.backtest.data_loader import DayBars
        bars: List[Dict] = day_bars.bars
        session_date: date = day_bars.session_date
        symbol: str = day_bars.symbol

        # --- Step 1: Build ORB candle ---
        orb_bars = [b for b in bars if self._orb_start <= b["ts_et"].time() < self._orb_end]
        if not orb_bars:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=0, orb_low=0, orb_midline=0, orb_range_pts=0,
                status="skipped", skip_reason="no_orb_data",
            )

        orb_high = max(b["high"] for b in orb_bars)
        orb_low = min(b["low"] for b in orb_bars)
        orb_midline = (orb_high + orb_low) / 2
        orb_range = round(orb_high - orb_low, 4)

        if orb_range < self._min_range:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason=f"range_too_narrow_{orb_range}pts",
            )
        if orb_range > self._max_range:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason=f"range_too_wide_{orb_range}pts",
            )

        # --- Step 2: Post-open bars ---
        post_open_bars = [b for b in bars if b["ts_et"].time() >= self._market_open]
        if not post_open_bars:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason="no_post_open_bars",
            )

        # --- Step 3: Detect breakout ---
        breakout_bar = None
        bias = None
        for bar in post_open_bars:
            if bar["ts_et"].time() >= self._eod:
                break
            if bar["close"] > orb_high:
                bias = "long"
                breakout_bar = bar
                break
            if bar["close"] < orb_low:
                bias = "short"
                breakout_bar = bar
                break

        if breakout_bar is None:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason="no_breakout",
            )

        post_breakout_bars = post_open_bars[post_open_bars.index(breakout_bar):]

        # --- Step 4: Midline retest ---
        retest_bar = None
        for bar in post_breakout_bars:
            if bar["ts_et"].time() >= self._eod:
                break
            if bar["low"] <= orb_midline + self._midline_tol and bar["high"] >= orb_midline - self._midline_tol:
                retest_bar = bar
                break

        if retest_bar is None:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                status="skipped", skip_reason="no_retest",
            )

        # --- Step 5: Fill on next bar's open ---
        retest_idx = post_breakout_bars.index(retest_bar)
        post_retest = post_breakout_bars[retest_idx + 1:]
        if not post_retest:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                status="skipped", skip_reason="no_bar_after_retest",
            )

        fill_bar = post_retest[0]
        es_entry_price = fill_bar["open"]  # ES price from bars (used for ORB logic)
        entry_time = fill_bar["ts_et"]

        # --- Step 6: Price the spread via BS ---
        # Get underlying price for option pricing (SPY*10 if configured, else ES)
        option_underlying = self._get_underlying_price(session_date, es_entry_price)
        # Get session IV (VIX if configured, else fixed assumed_iv)
        session_iv = self._get_session_iv(session_date)

        t_entry = self._tte_hours(entry_time) / (252 * 6.5)  # fraction of trading year
        spread_type = "put_credit_spread" if bias == "long" else "call_credit_spread"

        short_k = _select_strike(option_underlying, self._target_delta, session_iv, t_entry, spread_type)
        long_k = (short_k - self._spread_width if spread_type == "put_credit_spread"
                  else short_k + self._spread_width)

        credit = _spread_credit(option_underlying, short_k, long_k, t_entry, session_iv, spread_type)
        credit = round(credit, 2)

        if credit < self._min_credit:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                spread_type=spread_type,
                short_strike=short_k, long_strike=long_k,
                spread_width=self._spread_width, iv_assumed=session_iv,
                underlying_at_entry=option_underlying,
                status="skipped", skip_reason=f"credit_too_low_{credit:.2f}",
            )

        max_profit = round(credit * 100 * self._qty, 2)
        max_loss = round((self._spread_width - credit) * 100 * self._qty, 2)

        if max_loss > self._max_loss_usd:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                spread_type=spread_type,
                short_strike=short_k, long_strike=long_k,
                spread_width=self._spread_width, iv_assumed=session_iv,
                underlying_at_entry=option_underlying,
                credit_received=credit, max_profit_usd=max_profit, max_loss_usd=max_loss,
                status="skipped", skip_reason="risk_limit",
            )

        if spread_type == "put_credit_spread":
            short_delta = abs(_bs_put_delta(option_underlying, short_k, t_entry, session_iv))
        else:
            short_delta = _bs_call_delta(option_underlying, short_k, t_entry, session_iv)

        tp_debit = round(credit * (1 - self._tp_pct), 2)

        # --- Step 7: Simulate exit ---
        exit_time, exit_price, exit_reason, debit = self._simulate_options_exit(
            bars=post_retest[1:],   # bars after fill bar
            fill_bar=fill_bar,
            spread_type=spread_type,
            short_k=short_k,
            long_k=long_k,
            credit=credit,
            tp_debit=tp_debit,
            entry_time=entry_time,
            session_iv=session_iv,
        )

        pnl = round((credit - debit) * 100 * self._qty, 2)

        return OptionsBacktestResult(
            trade_id=f"OPT-{session_date}-{uuid.uuid4().hex[:6].upper()}",
            session_date=session_date, symbol=symbol,
            orb_high=orb_high, orb_low=orb_low,
            orb_midline=orb_midline, orb_range_pts=orb_range,
            bias=bias,
            breakout_time=breakout_bar["ts_et"],
            breakout_price=breakout_bar["close"],
            spread_type=spread_type,
            short_strike=short_k, long_strike=long_k,
            spread_width=self._spread_width, iv_assumed=session_iv,
            entry_time=entry_time,
            underlying_at_entry=option_underlying,
            credit_received=credit,
            max_profit_usd=max_profit,
            max_loss_usd=max_loss,
            short_delta_at_entry=round(short_delta, 4),
            exit_time=exit_time,
            underlying_at_exit=exit_price,
            debit_paid=debit,
            exit_reason=exit_reason,
            realized_pnl_usd=pnl,
            status="closed",
            qty=self._qty,
        )

    # -----------------------------------------------------------------------
    # Options exit simulation
    # -----------------------------------------------------------------------

    def _simulate_options_exit(
        self,
        bars: List[Dict],
        fill_bar: Dict,
        spread_type: str,
        short_k: float,
        long_k: float,
        credit: float,
        tp_debit: float,
        entry_time: datetime,
        session_iv: float,
    ) -> Tuple[datetime, float, str, float]:
        """
        Walk bars after fill, checking in order:
          1. Short strike breached intrabar → reprice spread at breach → exit.
          2. TP: BS spread mark <= tp_debit → exit at tp_debit.
          3. EOD bar → expire worthless.

        Returns (exit_time, underlying_price, exit_reason, debit_paid).
        """
        # Check fill bar itself for immediate breach
        check_bars = [fill_bar] + list(bars)

        for bar in check_bars:
            ts = bar["ts_et"]
            if ts.time() >= self._eod:
                # EOD — expire worthless (conservative: full max profit)
                return ts, bar["close"], "expired_worthless", 0.0

            t_now = self._tte_hours(ts) / (252 * 6.5)
            t_now = max(t_now, 1e-6)

            # Check short strike breach (use bar high/low for worst-case intrabar)
            breached, breach_price = self._check_breach(bar, spread_type, short_k)
            if breached:
                # Price spread at breach: underlying at short strike (worst case)
                debit = _spread_value(breach_price, short_k, long_k, t_now, session_iv, spread_type)
                debit = round(min(debit, self._spread_width), 2)
                return ts, breach_price, "short_breached", debit

            # Check TP: compute BS spread mark at bar close
            mark = _spread_value(bar["close"], short_k, long_k, t_now, session_iv, spread_type)
            if mark <= tp_debit:
                return ts, bar["close"], "tp_50pct", round(tp_debit, 2)

        # No exit found — treat as expired worthless at EOD
        last = bars[-1] if bars else fill_bar
        return last["ts_et"], last["close"], "expired_worthless", 0.0

    def _check_breach(
        self, bar: Dict, spread_type: str, short_k: float
    ) -> Tuple[bool, float]:
        """
        Returns (breached, breach_price) where breach_price is the short strike
        (worst-case assumption: touched exactly).
        """
        if spread_type == "put_credit_spread":
            if bar["low"] <= short_k:
                return True, short_k
        else:
            if bar["high"] >= short_k:
                return True, short_k
        return False, 0.0

    # -----------------------------------------------------------------------
    # Time-to-expiry helpers
    # -----------------------------------------------------------------------

    def _tte_hours(self, ts: datetime) -> float:
        """Trading hours remaining until 16:00 ET on the same day."""
        close_dt = ts.replace(hour=16, minute=0, second=0, microsecond=0)
        if ts.tzinfo is None:
            close_dt = close_dt.replace(tzinfo=ET)
            ts = ts.replace(tzinfo=ET)
        remaining = (close_dt - ts).total_seconds() / 3600.0
        return max(remaining, 0.0)
