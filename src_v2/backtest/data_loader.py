"""
SPY data loader — reads SPY 1-min CSV and returns per-day DayBars in ET.

Prices are scaled by price_scale (default 10.0: SPY * 10 ≈ SPX).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp to UTC datetime.

    Handles:
      - 2024-01-02T14:30:00.000000000Z  (nanoseconds)
      - 2024-01-02T14:30:00Z             (standard)
    """
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        base, frac = ts_str.split(".", 1)
        ts_str = f"{base}.{frac[:6]}"
    return datetime.fromisoformat(ts_str).replace(tzinfo=UTC)


class DayBars:
    """Container for one trading day's 1-min bars."""

    __slots__ = ("session_date", "symbol", "bars")

    def __init__(self, session_date: date, symbol: str, bars: List[Dict]):
        self.session_date = session_date
        self.symbol = symbol
        self.bars = bars

    def __repr__(self) -> str:
        return f"DayBars({self.session_date}, {self.symbol}, {len(self.bars)} bars)"


class SpyLoader:
    """Loads SPY 1-min OHLCV CSV and yields DayBars objects.

    Prices scaled by price_scale (10.0 -> SPY*10 ≈ SPX).
    """

    def __init__(self, data_path: str, price_scale: float = 10.0, symbol: str = "SPX"):
        self.data_path = Path(data_path)
        self._price_scale = price_scale
        self._symbol = symbol
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")

    def iter_days(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Iterator[DayBars]:
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

                if prev_date is not None and bar_date > prev_date:
                    for completed in sorted(d for d in buf if d < bar_date):
                        if buf[completed]:
                            yield DayBars(
                                completed, self._symbol,
                                sorted(buf[completed], key=lambda b: b["ts_et"]),
                            )
                        del buf[completed]

                prev_date = bar_date

                try:
                    s = self._price_scale
                    buf[bar_date].append({
                        "ts_et": ts_et,
                        "open": float(row["open"]) * s,
                        "high": float(row["high"]) * s,
                        "low": float(row["low"]) * s,
                        "close": float(row["close"]) * s,
                        "volume": int(row.get("volume", 0)),
                    })
                except (ValueError, KeyError):
                    continue

        for remaining in sorted(buf.keys()):
            if buf[remaining]:
                yield DayBars(
                    remaining, self._symbol,
                    sorted(buf[remaining], key=lambda b: b["ts_et"]),
                )
