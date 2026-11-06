"""
1-min bar loader — reads 1-min OHLCV CSV (e.g. from MarketData.app download)
and returns per-day DayBars in ET, compatible with DatabentoLoader interface.

Prices are scaled by config's backtest.price_scale (e.g. 10.0 for SPY * 10 ≈ SPX).
Set underlying_source="bars" so the engine uses bar prices directly.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from zoneinfo import ZoneInfo

from es_8am.backtest.data_loader import DayBars

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _parse_ts(ts_str: str) -> datetime:
    """Parse MarketData.app ISO timestamp to UTC datetime.

    Handles formats:
      - 2024-01-02T14:30:00.000000000Z  (Databento-style nanoseconds)
      - 2024-01-02T14:30:00Z             (standard UTC)
    """
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        base, frac = ts_str.split(".", 1)
        frac = frac[:6]
        ts_str = f"{base}.{frac}"
    return datetime.fromisoformat(ts_str).replace(tzinfo=UTC)


class BarLoader:
    """
    Loads 1-min OHLCV CSV and yields DayBars objects.

    Prices are scaled by `price_scale` (e.g. 10.0 for SPY × 10 ≈ SPX) so that
    all ORB thresholds and strike placement work at SPX/ES scale.

    The CSV must have columns: ts_event, symbol, open, high, low, close, volume
    (same format as Databento CSV, written by download_spy_data.py).

    Usage:
        loader = BarLoader("data/spy_1m.csv", price_scale=10.0, symbol=bot_symbol)
        for day in loader.iter_days(start_date, end_date):
            ...day.bars...  # prices are raw_close * price_scale
    """

    def __init__(self, data_path: str, price_scale: float = 10.0, *, symbol: str):
        self.data_path = Path(data_path)
        self._price_scale = price_scale
        self._symbol = symbol
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")

    def iter_days(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        front_month_only: bool = True,  # ignored for SPY, kept for interface compat
    ) -> Iterator[DayBars]:
        """
        Yield one DayBars per trading day, sorted by timestamp.

        Each bar dict contains: ts_et, open, high, low, close, volume
        """
        from collections import defaultdict

        # buf[date] = list of bar dicts
        buf: Dict[date, List[Dict]] = defaultdict(list)
        prev_date: Optional[date] = None

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    ts_utc = _parse_ts(row["ts_event"])
                except Exception:
                    continue

                ts_et = ts_utc.astimezone(ET)
                bar_date = ts_et.date()

                if start_date and bar_date < start_date:
                    continue
                if end_date and bar_date > end_date:
                    break

                # Yield completed days as we advance
                if prev_date is not None and bar_date > prev_date:
                    for completed_date in sorted(d for d in buf if d < bar_date):
                        bars = buf[completed_date]
                        if bars:
                            yield DayBars(
                                completed_date,
                                self._symbol,
                                sorted(bars, key=lambda b: b["ts_et"]),
                            )
                        del buf[completed_date]

                prev_date = bar_date

                try:
                    s = self._price_scale
                    bar = {
                        "ts_et": ts_et,
                        "open": float(row["open"]) * s,
                        "high": float(row["high"]) * s,
                        "low": float(row["low"]) * s,
                        "close": float(row["close"]) * s,
                        "volume": int(row.get("volume", 0)),
                    }
                    buf[bar_date].append(bar)
                except (ValueError, KeyError):
                    continue

        # Flush remaining
        for remaining_date in sorted(buf.keys()):
            bars = buf[remaining_date]
            if bars:
                yield DayBars(
                    remaining_date,
                    self._symbol,
                    sorted(bars, key=lambda b: b["ts_et"]),
                )
