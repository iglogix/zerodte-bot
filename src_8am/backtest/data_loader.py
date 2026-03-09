"""
Databento data loader — reads the GLBX zstd-compressed CSV and returns
per-day DataFrames of ES front-month 1-min OHLCV bars in ET.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# Spread contracts contain a dash — filter them out
_SPREAD_MARKER = "-"


def _is_front_month(symbol: str) -> bool:
    """Return True if symbol is a single ES contract (not a spread)."""
    return symbol.startswith("ES") and _SPREAD_MARKER not in symbol


def _parse_ts(ts_str: str) -> datetime:
    """Parse Databento ISO timestamp (nanosecond precision) to UTC datetime."""
    # Format: 2024-01-02T14:30:00.000000000Z
    # Strip nanosecond precision beyond microseconds for Python compat
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        base, frac = ts_str.split(".", 1)
        frac = frac[:6]  # truncate to microseconds
        ts_str = f"{base}.{frac}"
    dt = datetime.fromisoformat(ts_str).replace(tzinfo=UTC)
    return dt


class DayBars:
    """Container for one trading day's 1-min bars."""

    __slots__ = ("session_date", "symbol", "bars")

    def __init__(self, session_date: date, symbol: str, bars: List[Dict]):
        self.session_date = session_date
        self.symbol = symbol
        self.bars = bars  # list of dicts: ts_et, open, high, low, close, volume

    def __repr__(self) -> str:
        return f"DayBars({self.session_date}, {self.symbol}, {len(self.bars)} bars)"


class DatabentoLoader:
    """
    Loads and parses the Databento GLBX OHLCV-1m zip file.

    Usage:
        loader = DatabentoLoader("GLBX-20260227-7PWD5NKFSF.zip")
        for day in loader.iter_days(start_date, end_date):
            ...day.bars...
    """

    def __init__(self, zip_path: str):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Databento zip not found: {self.zip_path}")

    def _open_csv_stream(self):
        """Open the inner .zst CSV and return a text iterator."""
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError("Install zstandard: pip install zstandard")

        zf = zipfile.ZipFile(self.zip_path)
        zst_names = [n for n in zf.namelist() if n.endswith(".csv.zst")]
        if not zst_names:
            raise ValueError(f"No .csv.zst file found in {self.zip_path}")

        zst_data = zf.read(zst_names[0])
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(io.BytesIO(zst_data), read_across_frames=True)
        return io.TextIOWrapper(stream, encoding="utf-8")

    def iter_days(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        front_month_only: bool = True,
    ) -> Iterator[DayBars]:
        """
        Yield one DayBars per (date, symbol) — all bars for that pair collected first.
        The CSV interleaves multiple symbols row-by-row, so we accumulate into a dict
        keyed by (date, symbol) and yield complete days once we advance past them.

        For memory efficiency we only buffer two calendar dates at a time.
        """
        from collections import defaultdict

        # buf[date][symbol] = list of bar dicts
        buf: Dict[date, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        prev_date: Optional[date] = None

        text_stream = self._open_csv_stream()
        reader = csv.DictReader(text_stream)

        for row in reader:
            symbol = row["symbol"]
            if front_month_only and not _is_front_month(symbol):
                continue

            try:
                ts_utc = _parse_ts(row["ts_event"])
            except Exception:
                continue

            ts_et = ts_utc.astimezone(ET)
            bar_date = ts_et.date()

            # Date range filters
            if start_date and bar_date < start_date:
                continue
            if end_date and bar_date > end_date:
                break

            # When we move to a new date, yield everything from dates that are now complete
            if prev_date is not None and bar_date > prev_date:
                for completed_date in sorted(d for d in buf if d < bar_date):
                    for sym, bars in buf[completed_date].items():
                        if bars:
                            yield DayBars(completed_date, sym, sorted(bars, key=lambda b: b["ts_et"]))
                    del buf[completed_date]

            prev_date = bar_date

            try:
                bar = {
                    "ts_et": ts_et,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
                buf[bar_date][symbol].append(bar)
            except (ValueError, KeyError):
                continue

        # Flush remaining
        for remaining_date in sorted(buf.keys()):
            for sym, bars in buf[remaining_date].items():
                if bars:
                    yield DayBars(remaining_date, sym, sorted(bars, key=lambda b: b["ts_et"]))

    def load_days_dict(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        front_month_only: bool = True,
    ) -> Dict[Tuple[date, str], DayBars]:
        """Load all matching days into a dict keyed by (date, symbol)."""
        result: Dict[Tuple[date, str], DayBars] = {}
        for day in self.iter_days(start_date, end_date, front_month_only):
            key = (day.session_date, day.symbol)
            # Keep the symbol with highest total volume for a given date
            if key not in result or sum(b["volume"] for b in day.bars) > sum(
                b["volume"] for b in result[key].bars
            ):
                result[key] = day
        return result

    def get_active_symbol_for_date(self, target_date: date) -> Optional[str]:
        """
        Return the front-month ES symbol with the most volume on target_date.
        Useful to identify which contract was active on a given day.
        """
        best_symbol: Optional[str] = None
        best_volume = 0

        for day in self.iter_days(start_date=target_date, end_date=target_date):
            vol = sum(b["volume"] for b in day.bars)
            if vol > best_volume:
                best_volume = vol
                best_symbol = day.symbol

        return best_symbol
