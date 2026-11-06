"""
Touch and Turn Scalper — Backtest Engine

Strategy:
  1. Wait for the first 15-min candle after market open (09:30–09:45 ET).
  2. Measure the candle range (high − low).
  3. Confirm "liquidity candle": range >= 25% of 14-day daily ATR.
  4. Entry (trend-following pullback / "touch and turn"):
     - Bearish candle (close < open) → SHORT limit at the candle high.
     - Bullish candle (close > open) → LONG limit at the candle low.
  5. Take-profit via Fibonacci retracement of the 15-min range:
     - Longs: TP at the 38.2% fib level  (= high − 0.382 × range).
     - Shorts: TP at the 61.8% fib level (= high − 0.618 × range).
  6. Stop-loss at half the TP distance (enforces 2:1 R:R):
     - Long SL  = entry − (TP − entry) / 2
     - Short SL = entry + (entry − TP) / 2
  7. Orders valid only during the first 90 minutes (until 11:00 ET).
  8. Any open position force-closed at EOD.
  9. One trade per day maximum; resets daily.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, Iterator, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    trade_id: str
    session_date: date
    symbol: str
    direction: str              # "long" or "short"
    orb_high: float
    orb_low: float
    orb_range: float
    daily_atr: float
    entry_price: float
    tp_price: float
    sl_price: float
    fill_price: float           # actual fill (entry ± slippage)
    exit_price: float
    exit_reason: str            # "tp", "sl", "eod", "expired"
    pnl_pts: float
    pnl_usd: float
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None


@dataclass
class BacktestReport:
    trades: List[TradeResult] = field(default_factory=list)
    skipped_days: int = 0
    total_days: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TouchTurnBacktestEngine:
    """Runs the Touch and Turn Scalper strategy on 1-min ES bar data."""

    def __init__(self, config: Dict[str, Any]):
        strat = config.get("strategy", {})
        risk = config.get("risk", {})

        # Time windows
        self.orb_start = self._parse_time(strat.get("orb_candle_start", "09:30"))
        self.orb_end = self._parse_time(strat.get("orb_candle_end", "09:45"))
        self.window_end = self._parse_time(strat.get("trading_window_end", "11:00"))
        self.eod_close = self._parse_time(strat.get("eod_force_close", "15:50"))

        # ATR filter
        self.atr_period = strat.get("atr_period", 14)
        self.atr_min_pct = strat.get("atr_min_pct", 0.25)

        # Fibonacci TP levels + SL ratio
        self.fib_tp_long = strat.get("fib_tp_long", 0.382)
        self.fib_tp_short = strat.get("fib_tp_short", 0.618)
        self.sl_ratio = strat.get("sl_ratio", 0.5)  # SL = TP_dist * sl_ratio (0.5 = 2:1 R:R)

        # Risk / sizing
        self.position_size = risk.get("position_size", 1)
        self.multiplier = risk.get("contract_multiplier", 50)
        self.max_daily_loss = risk.get("max_daily_loss_usd", 5000)
        self.slippage_pts = risk.get("slippage_per_side_pts", 0.25)

        # Internal: rolling daily OHLC for ATR calculation
        self._daily_history: deque = deque(maxlen=self.atr_period + 1)

    @staticmethod
    def _parse_time(s: str) -> time:
        parts = s.split(":")
        return time(int(parts[0]), int(parts[1]))

    # ------------------------------------------------------------------
    # ATR calculation from daily bars built from 1-min data
    # ------------------------------------------------------------------

    def _build_daily_ohlc(self, bars: List[Dict]) -> Optional[Dict]:
        """Aggregate 1-min bars into a single daily OHLC (RTH only: 09:30-16:00)."""
        rth = [b for b in bars if time(9, 30) <= b["ts_et"].time() < time(16, 0)]
        if not rth:
            return None
        o = rth[0]["open"]
        h = max(b["high"] for b in rth)
        l = min(b["low"] for b in rth)
        c = rth[-1]["close"]
        return {"open": o, "high": h, "low": l, "close": c}

    def _compute_atr(self) -> Optional[float]:
        """Compute ATR from the rolling daily history using true range."""
        history = list(self._daily_history)
        if len(history) < self.atr_period + 1:
            return None

        true_ranges = []
        for i in range(1, len(history)):
            prev_close = history[i - 1]["close"]
            curr = history[i]
            tr = max(
                curr["high"] - curr["low"],
                abs(curr["high"] - prev_close),
                abs(curr["low"] - prev_close),
            )
            true_ranges.append(tr)

        # Use the last atr_period true ranges
        recent = true_ranges[-self.atr_period:]
        return sum(recent) / len(recent)

    # ------------------------------------------------------------------
    # Core: process a single day
    # ------------------------------------------------------------------

    def _process_day(self, session_date: date, symbol: str, bars: List[Dict]) -> Optional[TradeResult]:
        """Run strategy on one day's 1-min bars. Returns a TradeResult or None."""

        # --- Step 1: Build the 15-min opening candle (09:30–09:45) ---
        orb_bars = [
            b for b in bars
            if self.orb_start <= b["ts_et"].time() < self.orb_end
        ]
        if len(orb_bars) < 5:  # need roughly 15 one-minute bars
            logger.debug("%s: not enough ORB bars (%d)", session_date, len(orb_bars))
            return None

        orb_high = max(b["high"] for b in orb_bars)
        orb_low = min(b["low"] for b in orb_bars)
        orb_open = orb_bars[0]["open"]
        orb_close = orb_bars[-1]["close"]
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            return None

        # --- Step 2: ATR confirmation ---
        atr = self._compute_atr()
        if atr is None:
            logger.debug("%s: not enough history for ATR", session_date)
            return None

        if orb_range < self.atr_min_pct * atr:
            logger.debug(
                "%s: ORB range %.2f < %.0f%% of ATR %.2f — skip",
                session_date, orb_range, self.atr_min_pct * 100, atr,
            )
            return None

        # --- Step 3: Determine direction (touch-and-turn pullback) ---
        if orb_close < orb_open:
            # Bearish candle → SHORT at the high (fade the retracement)
            direction = "short"
            entry_price = orb_high
            tp_price = orb_low + self.fib_tp_short * orb_range
            tp_dist = entry_price - tp_price
            sl_price = entry_price + tp_dist * self.sl_ratio
        elif orb_close > orb_open:
            # Bullish candle → LONG at the low (fade the retracement)
            direction = "long"
            entry_price = orb_low
            tp_price = orb_low + self.fib_tp_long * orb_range
            tp_dist = tp_price - entry_price
            sl_price = entry_price - tp_dist * self.sl_ratio
        else:
            # Doji — no trade
            logger.debug("%s: doji candle — skip", session_date)
            return None

        # --- Step 4: Walk 1-min bars after ORB close, look for fill + exit ---
        post_orb_bars = [
            b for b in bars
            if b["ts_et"].time() >= self.orb_end
        ]

        filled = False
        fill_price = 0.0
        actual_sl = sl_price
        entry_time = None

        for bar in post_orb_bars:
            bar_time = bar["ts_et"].time()

            # --- Check if limit order fills ---
            if not filled:
                # Cancel unfilled order after 90-min window
                if bar_time >= self.window_end:
                    return TradeResult(
                        trade_id=str(uuid.uuid4())[:8],
                        session_date=session_date,
                        symbol=symbol,
                        direction=direction,
                        orb_high=orb_high,
                        orb_low=orb_low,
                        orb_range=orb_range,
                        daily_atr=atr,
                        entry_price=entry_price,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        fill_price=0.0,
                        exit_price=0.0,
                        exit_reason="expired",
                        pnl_pts=0.0,
                        pnl_usd=0.0,
                    )

                # Check fill: price must touch the limit level
                if direction == "long" and bar["low"] <= entry_price:
                    fill_price = entry_price + self.slippage_pts  # buy slippage
                    filled = True
                    entry_time = bar["ts_et"]
                elif direction == "short" and bar["high"] >= entry_price:
                    fill_price = entry_price - self.slippage_pts  # sell slippage
                    filled = True
                    entry_time = bar["ts_et"]

                if not filled:
                    continue

                # Recalculate TP/SL from actual fill price for accurate P&L
                if direction == "long":
                    actual_tp_dist = tp_price - fill_price
                    actual_sl = fill_price - actual_tp_dist * self.sl_ratio
                else:
                    actual_tp_dist = fill_price - tp_price
                    actual_sl = fill_price + actual_tp_dist * self.sl_ratio

                continue  # don't check exit on the fill bar

            # --- Position is open: check exit conditions ---

            # EOD force close
            if bar_time >= self.eod_close:
                exit_price = bar["close"]
                if direction == "long":
                    exit_price -= self.slippage_pts
                else:
                    exit_price += self.slippage_pts
                pnl_pts = (exit_price - fill_price) if direction == "long" else (fill_price - exit_price)
                return self._make_result(
                    session_date, symbol, direction, orb_high, orb_low,
                    orb_range, atr, entry_price, tp_price, sl_price,
                    fill_price, exit_price, "eod", pnl_pts, entry_time, bar["ts_et"],
                )

            # Check TP and SL within this bar
            if direction == "long":
                # SL hit (price went below SL)
                if bar["low"] <= actual_sl:
                    exit_price = actual_sl - self.slippage_pts
                    pnl_pts = exit_price - fill_price
                    return self._make_result(
                        session_date, symbol, direction, orb_high, orb_low,
                        orb_range, atr, entry_price, tp_price, sl_price,
                        fill_price, exit_price, "sl", pnl_pts, entry_time, bar["ts_et"],
                    )
                # TP hit (price reached TP level)
                if bar["high"] >= tp_price:
                    exit_price = tp_price - self.slippage_pts
                    pnl_pts = exit_price - fill_price
                    return self._make_result(
                        session_date, symbol, direction, orb_high, orb_low,
                        orb_range, atr, entry_price, tp_price, sl_price,
                        fill_price, exit_price, "tp", pnl_pts, entry_time, bar["ts_et"],
                    )
            else:  # short
                # SL hit (price went above SL)
                if bar["high"] >= actual_sl:
                    exit_price = actual_sl + self.slippage_pts
                    pnl_pts = fill_price - exit_price
                    return self._make_result(
                        session_date, symbol, direction, orb_high, orb_low,
                        orb_range, atr, entry_price, tp_price, sl_price,
                        fill_price, exit_price, "sl", pnl_pts, entry_time, bar["ts_et"],
                    )
                # TP hit (price reached TP level)
                if bar["low"] <= tp_price:
                    exit_price = tp_price + self.slippage_pts
                    pnl_pts = fill_price - exit_price
                    return self._make_result(
                        session_date, symbol, direction, orb_high, orb_low,
                        orb_range, atr, entry_price, tp_price, sl_price,
                        fill_price, exit_price, "tp", pnl_pts, entry_time, bar["ts_et"],
                    )

        # Position open at end of data (shouldn't happen with eod_force_close)
        if filled:
            last = post_orb_bars[-1]
            exit_price = last["close"]
            if direction == "long":
                exit_price -= self.slippage_pts
                pnl_pts = exit_price - fill_price
            else:
                exit_price += self.slippage_pts
                pnl_pts = fill_price - exit_price
            return self._make_result(
                session_date, symbol, direction, orb_high, orb_low,
                orb_range, atr, entry_price, tp_price, sl_price,
                fill_price, exit_price, "eod", pnl_pts, entry_time, last["ts_et"],
            )

        return None

    def _make_result(
        self, session_date, symbol, direction, orb_high, orb_low,
        orb_range, atr, entry_price, tp_price, sl_price,
        fill_price, exit_price, exit_reason, pnl_pts, entry_time, exit_time,
    ) -> TradeResult:
        pnl_usd = pnl_pts * self.multiplier * self.position_size
        return TradeResult(
            trade_id=str(uuid.uuid4())[:8],
            session_date=session_date,
            symbol=symbol,
            direction=direction,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_range=orb_range,
            daily_atr=atr,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            fill_price=fill_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_pts=pnl_pts,
            pnl_usd=pnl_usd,
            entry_time=entry_time,
            exit_time=exit_time,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, day_iter: Iterator) -> BacktestReport:
        """Run backtest over an iterator of DayBars objects.

        For multi-contract data (ES futures), picks highest-volume contract
        per calendar day and only trades/updates ATR once per day.
        """
        report = BacktestReport()

        # Group DayBars by calendar date, pick highest RTH volume
        from collections import defaultdict
        by_date: Dict[Any, Any] = defaultdict(list)
        for day_bars in day_iter:
            by_date[day_bars.session_date].append(day_bars)

        for session_date in sorted(by_date.keys()):
            candidates = by_date[session_date]
            # Pick contract with most RTH bars (proxy for volume/liquidity)
            best = max(candidates, key=lambda d: sum(
                1 for b in d.bars if time(9, 30) <= b["ts_et"].time() < time(16, 0)
            ))

            report.total_days += 1

            # Update daily OHLC history for ATR (RTH only)
            daily = self._build_daily_ohlc(best.bars)
            if daily:
                self._daily_history.append(daily)

            result = self._process_day(
                best.session_date, best.symbol, best.bars,
            )

            if result is None:
                report.skipped_days += 1
            elif result.exit_reason == "expired":
                report.skipped_days += 1
            else:
                report.trades.append(result)

        return report
