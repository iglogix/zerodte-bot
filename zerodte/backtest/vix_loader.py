"""Process-level cached VIX daily series loader.

Loads VIX9D / VIX / VIX1D daily CSVs (Date, Close, ...) into {date: close}
and exposes a simple lookup. Cached in-process so workers preload once.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Dict, Optional

_CACHE: Dict[str, Dict[date, float]] = {}


def load_vix_series(csv_path: str | Path) -> Dict[date, float]:
    key = str(csv_path)
    if key in _CACHE:
        return _CACHE[key]

    p = Path(csv_path)
    series: Dict[date, float] = {}
    if not p.exists():
        _CACHE[key] = series
        return series

    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_raw = row.get("Date") or row.get("date")
            c_raw = row.get("Close") or row.get("close")
            if not d_raw or c_raw in (None, ""):
                continue
            try:
                d = date.fromisoformat(d_raw)
                series[d] = float(c_raw)
            except (ValueError, TypeError):
                continue

    _CACHE[key] = series
    return series


def lookup_prior_close(series: Dict[date, float], target: date) -> Optional[float]:
    """Return the most recent VIX close on or before `target` (inclusive)."""
    if not series:
        return None
    cur = target
    for _ in range(10):  # walk back up to 10 calendar days for weekends/holidays
        if cur in series:
            return series[cur]
        cur = date.fromordinal(cur.toordinal() - 1)
    return None
