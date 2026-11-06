"""
ORB Backtest Engine — replays historical 1-min OHLCV data day by day.

For each trading day:
  1. Load 1-min bars from DatabentoLoader
  2. Resample to 15-min → identify 08:00–08:15 ORB candle
  3. Walk bars from 09:30 → detect first breakout
  4. Apply simplified regime check (rule-based, no LLM)
  5. Walk bars → detect midline touch → simulate entry
  6. Walk bars → detect TP or SL hit → record exit
  7. Persist OrbTrade to DB
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from es_8am.backtest.data_loader import DatabentoLoader, DayBars

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

_ORB_START = time(8, 0)
_ORB_END = time(8, 15)
_MARKET_OPEN = time(9, 30)


class BacktestResult:
    """Single simulated trade result."""

    __slots__ = (
        "trade_id", "session_date", "symbol",
        "orb_high", "orb_low", "orb_midline", "orb_range_pts",
        "bias", "breakout_time", "breakout_price",
        "entry_time", "entry_price", "tp_price", "sl_price",
        "exit_time", "exit_price", "exit_reason", "realized_pnl_pts",
        "status", "skip_reason",
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class OrbBacktestEngine:
    """
    Runs the ORB strategy over a historical date range using Databento data.
    """

    def __init__(self, config: Dict[str, Any], db_service=None):
        self._config = config
        self._db = db_service

        self._strategy = config.get("strategy", {})
        self._exit_cfg = config.get("exit", {})
        self._backtest_cfg = config.get("backtest", {})

        self._tp_pts: float = self._exit_cfg.get("tp_pts", 20.0)
        self._sl_pts: float = self._exit_cfg.get("sl_pts", 5.0)
        self._tolerance: float = self._strategy.get("midline_tolerance_pts", 0.25)
        self._min_range: float = 2.0
        self._max_range: float = 30.0

        eod_str = self._strategy.get("eod_force_close", "15:50")
        h, m = [int(x) for x in eod_str.split(":")]
        self._eod = time(h, m)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(
        self,
        zip_path: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[BacktestResult]:
        """
        Run the backtest over the given date range.
        Returns a list of BacktestResult objects (one per trading day attempted).
        """
        loader = DatabentoLoader(zip_path)
        results: List[BacktestResult] = []

        # Group bars by (date, symbol), pick highest-volume symbol per date
        daily_map: Dict[date, DayBars] = {}
        for day in loader.iter_days(start_date=start_date, end_date=end_date):
            existing = daily_map.get(day.session_date)
            if existing is None:
                daily_map[day.session_date] = day
            else:
                # Keep the symbol with more volume (= active front month)
                if sum(b["volume"] for b in day.bars) > sum(b["volume"] for b in existing.bars):
                    daily_map[day.session_date] = day

        total = len(daily_map)
        logger.info("Backtesting %d trading days", total)

        for i, (session_date, day_bars) in enumerate(sorted(daily_map.items())):
            result = self._process_day(session_date, day_bars)
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info("Progress: %d / %d days processed", i + 1, total)

        logger.info("Backtest complete: %d days processed", len(results))
        return results

    # -------------------------------------------------------------------------
    # Single day processing
    # -------------------------------------------------------------------------

    def _process_day(self, session_date: date, day: DayBars) -> BacktestResult:
        bars = day.bars
        symbol = day.symbol

        # Step 1: Build ORB candle from 08:00–08:15 bars
        orb = self._build_orb_candle(bars)
        if orb is None:
            logger.debug("%s: no ORB candle data — skipping", session_date)
            return BacktestResult(
                trade_id=f"BT-{session_date}-SKIP",
                session_date=session_date,
                symbol=symbol,
                status="skipped",
                skip_reason="no_orb_data",
            )

        orb_high, orb_low = orb
        orb_midline = round((orb_high + orb_low) / 2, 4)
        orb_range = round(orb_high - orb_low, 4)

        # Step 2: Regime check (rule-based)
        skip_reason = self._regime_check(orb_range, bars)
        if skip_reason:
            return BacktestResult(
                trade_id=f"BT-{session_date}-SKIP",
                session_date=session_date,
                symbol=symbol,
                orb_high=orb_high,
                orb_low=orb_low,
                orb_midline=orb_midline,
                orb_range_pts=orb_range,
                status="skipped",
                skip_reason=skip_reason,
            )

        # Step 3: Walk bars from 09:30 to detect breakout
        post_open_bars = [
            b for b in bars
            if b["ts_et"].time() >= _MARKET_OPEN and b["ts_et"].time() < self._eod
        ]
        if not post_open_bars:
            return BacktestResult(
                trade_id=f"BT-{session_date}-SKIP",
                session_date=session_date,
                symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason="no_post_open_bars",
            )

        bias, breakout_bar = self._detect_breakout(post_open_bars, orb_high, orb_low)
        if bias is None or breakout_bar is None:
            return BacktestResult(
                trade_id=f"BT-{session_date}-SKIP",
                session_date=session_date,
                symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason="no_breakout",
            )

        breakout_idx = post_open_bars.index(breakout_bar)

        # Step 4: Walk bars after breakout to detect midline touch
        post_breakout_bars = post_open_bars[breakout_idx + 1:]
        entry_bar = self._detect_midline_touch(post_breakout_bars, orb_midline, self._tolerance)
        if entry_bar is None:
            return BacktestResult(
                trade_id=f"BT-{session_date}-SKIP",
                session_date=session_date,
                symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                status="skipped", skip_reason="no_retest",
            )

        # Step 5: Enter at the NEXT bar's open after the midline touch
        # (realistic: order fills at next bar open, not inside the signal bar)
        entry_idx = post_breakout_bars.index(entry_bar)
        post_entry_bars = post_breakout_bars[entry_idx + 1:]

        if not post_entry_bars:
            return BacktestResult(
                trade_id=f"BT-{session_date}-SKIP",
                session_date=session_date,
                symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                status="skipped", skip_reason="no_bar_after_retest",
            )

        fill_bar = post_entry_bars[0]
        entry_price = fill_bar["open"]  # fill at next bar open
        entry_time_actual = fill_bar["ts_et"]

        if bias == "long":
            tp_price = round(entry_price + self._tp_pts, 4)
            sl_price = round(entry_price - self._sl_pts, 4)
        else:
            tp_price = round(entry_price - self._tp_pts, 4)
            sl_price = round(entry_price + self._sl_pts, 4)

        # Step 6: Check intrabar fill on the fill bar, then walk subsequent bars
        post_fill_bars = post_entry_bars[1:]

        intrabar_exit = self._check_intrabar(fill_bar, bias, tp_price, sl_price)
        if intrabar_exit:
            exit_price, exit_time, exit_reason = intrabar_exit
        else:
            exit_price, exit_time, exit_reason = self._simulate_exit(
                post_fill_bars, bias, tp_price, sl_price, entry_time_actual
            )

        if exit_price is None:
            exit_price = post_entry_bars[-1]["close"] if post_entry_bars else entry_price
            exit_time = post_entry_bars[-1]["ts_et"] if post_entry_bars else entry_bar["ts_et"]
            exit_reason = "eod_close"

        pnl = round(exit_price - entry_price, 4) if bias == "long" else round(entry_price - exit_price, 4)

        trade_id = f"BT-{session_date}-{uuid.uuid4().hex[:6].upper()}"
        result = BacktestResult(
            trade_id=trade_id,
            session_date=session_date,
            symbol=symbol,
            orb_high=orb_high, orb_low=orb_low,
            orb_midline=orb_midline, orb_range_pts=orb_range,
            bias=bias,
            breakout_time=breakout_bar["ts_et"],
            breakout_price=breakout_bar["close"],
            entry_time=entry_time_actual,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            realized_pnl_pts=pnl,
            status="closed",
        )

        if self._db:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                self._db.create_trade(result.to_dict())
            )

        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_orb_candle(self, bars: List[Dict]) -> Optional[Tuple[float, float]]:
        """Return (high, low) of the aggregated 08:00–08:15 candle."""
        orb_bars = [
            b for b in bars
            if _ORB_START <= b["ts_et"].time() < _ORB_END
        ]
        if not orb_bars:
            return None
        high = max(b["high"] for b in orb_bars)
        low = min(b["low"] for b in orb_bars)
        return high, low

    def _regime_check(self, orb_range: float, bars: List[Dict]) -> Optional[str]:
        """Rule-based regime check. Returns skip_reason string or None."""
        if orb_range < self._min_range:
            return f"range_too_narrow_{orb_range:.1f}pts"
        if orb_range > self._max_range:
            return f"range_too_wide_{orb_range:.1f}pts"
        return None

    def _detect_breakout(
        self,
        bars: List[Dict],
        orb_high: float,
        orb_low: float,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Return (bias, first_bar_that_closed_beyond_level)."""
        for bar in bars:
            if bar["close"] > orb_high:
                return "long", bar
            if bar["close"] < orb_low:
                return "short", bar
        return None, None

    def _detect_midline_touch(
        self,
        bars: List[Dict],
        midline: float,
        tolerance: float,
    ) -> Optional[Dict]:
        """Return first bar whose low/high touches the midline within tolerance."""
        for bar in bars:
            if bar["ts_et"].time() >= self._eod:
                break
            # Check if bar range includes midline within tolerance
            if bar["low"] <= midline + tolerance and bar["high"] >= midline - tolerance:
                return bar
        return None

    def _check_intrabar(
        self,
        bar: Dict,
        bias: str,
        tp_price: float,
        sl_price: float,
    ) -> Optional[Tuple[float, datetime, str]]:
        """
        Check if TP or SL is triggered within the entry bar.
        Worst-case conservative assumption: if both levels are within the bar's range,
        SL is assumed to be hit first (pessimistic fill).
        Returns (exit_price, exit_time, reason) or None if neither level is reached.
        """
        h, l = bar["high"], bar["low"]
        ts = bar["ts_et"]

        if bias == "long":
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
        else:
            tp_hit = l <= tp_price
            sl_hit = h >= sl_price

        if sl_hit and tp_hit:
            # Both hit — assume SL (pessimistic)
            return sl_price, ts, "sl_hit"
        if sl_hit:
            return sl_price, ts, "sl_hit"
        if tp_hit:
            return tp_price, ts, "tp_hit"
        return None

    def _simulate_exit(
        self,
        bars: List[Dict],
        bias: str,
        tp_price: float,
        sl_price: float,
        entry_time: datetime,
    ) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
        """
        Walk bars checking if TP or SL is hit.
        Conservative fill: uses bar high/low.
        Returns (exit_price, exit_time, reason).
        """
        for bar in bars:
            if bar["ts_et"].time() >= self._eod:
                return bar["close"], bar["ts_et"], "eod_close"

            if bias == "long":
                sl_hit = bar["low"] <= sl_price
                tp_hit = bar["high"] >= tp_price
            else:
                sl_hit = bar["high"] >= sl_price
                tp_hit = bar["low"] <= tp_price

            if sl_hit and tp_hit:
                return sl_price, bar["ts_et"], "sl_hit"
            if sl_hit:
                return sl_price, bar["ts_et"], "sl_hit"
            if tp_hit:
                return tp_price, bar["ts_et"], "tp_hit"

        return None, None, None
