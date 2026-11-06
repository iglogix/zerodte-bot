"""
Generic equity 1-min CSV loader — reads CSV files in the same format
as download_spy_data.py (MarketData.app) and yields DayBars objects.

Columns expected: ts_event, symbol, open, high, low, close, volume

Unlike BarLoader, no price scaling is applied (equities trade at their own price).
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from zoneinfo import ZoneInfo

from es_8am.backtest.data_loader import DayBars

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp to UTC datetime."""
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        base, frac = ts_str.split(".", 1)
        frac = frac[:6]
        ts_str = f"{base}.{frac}"
    return datetime.fromisoformat(ts_str).replace(tzinfo=UTC)


class EquityLoader:
    """
    Loads equity 1-min OHLCV CSV and yields DayBars objects.

    Usage:
        loader = EquityLoader("data/nvda_1m.csv")
        for day in loader.iter_days(start_date, end_date):
            ...day.bars...
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")

    def iter_days(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Iterator[DayBars]:
        """Yield one DayBars per trading day, sorted by timestamp."""
        buf: Dict[date, List[Dict]] = defaultdict(list)
        prev_date: Optional[date] = None
        symbol = "UNKNOWN"

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    ts_utc = _parse_ts(row["ts_event"])
                except Exception:
                    continue

                ts_et = ts_utc.astimezone(ET)
                bar_date = ts_et.date()
                symbol = row.get("symbol", symbol)

                if start_date and bar_date < start_date:
                    continue
                if end_date and bar_date > end_date:
                    break

                # Yield completed days as we advance
                if prev_date is not None and bar_date > prev_date:
                    for completed in sorted(d for d in buf if d < bar_date):
                        bars = buf[completed]
                        if bars:
                            yield DayBars(
                                completed,
                                symbol,
                                sorted(bars, key=lambda b: b["ts_et"]),
                            )
                        del buf[completed]

                prev_date = bar_date

                try:
                    bar = {
                        "ts_et": ts_et,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row.get("volume", 0)),
                    }
                    buf[bar_date].append(bar)
                except (ValueError, KeyError):
                    continue

        # Flush remaining
        for remaining in sorted(buf.keys()):
            bars = buf[remaining]
            if bars:
                yield DayBars(
                    remaining,
                    symbol,
                    sorted(bars, key=lambda b: b["ts_et"]),
                )
