"""Yields ordered marks (price snapshots) over a Burrito trade's life. EOD marks
come from the DuckStore (DTE>=1); intraday 0DTE marks from a 10-min provider.
A Mark carries everything the lifecycle needs to reprice + decide."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from zerodte.backtest.iron_duck import parity_forward, _to_date


@dataclass
class Mark:
    date: int
    intraday_min: Optional[str]   # "HH:MM" for 0DTE; None for EOD
    chain: dict
    forward: Optional[float]
    T: float                      # years to expiry at this mark
    is_last: bool                 # final mark before settlement


def _years_to(date: int, expiry: int) -> float:
    days = (_to_date(expiry) - _to_date(date)).days
    return max(days / 365.0, 1e-6)


def eod_marks(store, entry_date: int, expiry: int) -> Iterator[Mark]:
    """EOD marks from entry_date up to and including the expiry day. The expiry-day
    mark is flagged is_last (settlement handled by the engine via parity_forward)."""
    dates = [d for d in store.quote_dates() if entry_date <= d <= expiry]
    n = len(dates)
    for i, d in enumerate(dates):
        ch = store.chain(d, expiry)
        if ch is None:
            continue
        F = parity_forward(ch)
        yield Mark(date=d, intraday_min=None, chain=ch, forward=F,
                   T=_years_to(d, expiry), is_last=(i == n - 1))


def intraday_marks(provider, date: int, entry_min: str) -> Iterator[Mark]:
    """0DTE marks from a 10-min intraday provider. `provider` exposes bars(date)->
    ['HH:MM'...], chain_at(date, min)->Chain, forward_at(date, min)->float."""
    bars = [b for b in provider.bars(date) if b >= entry_min]
    n = len(bars)
    for i, b in enumerate(bars):
        ch = provider.chain_at(date, b)
        F = provider.forward_at(date, b)
        hh, mm = int(b[:2]), int(b[3:5])
        frac_day_left = max((16 * 60 - (hh * 60 + mm)) / (6.5 * 60), 0.0)
        T = max(frac_day_left / 252.0, 1e-9)
        yield Mark(date=date, intraday_min=b, chain=ch, forward=F, T=T,
                   is_last=(i == n - 1))
