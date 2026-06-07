"""NYSE (XNYS) trading-day calendar helpers for the DC backtest.

Used to map a session to its front/back expiry DATES by trading-day distance
(robust to the fact that SPXW did not have a daily expiry on every trading day
historically) and to enumerate sessions for the backfill.
"""
from __future__ import annotations

import functools
from datetime import date, timedelta
from typing import List, Optional

import exchange_calendars as xcals


@functools.lru_cache(maxsize=1)
def _xnys():
    return xcals.get_calendar("XNYS")


def sessions_between(start: date, end: date) -> List[date]:
    """Inclusive list of NYSE session dates in [start, end]."""
    if end < start:
        return []
    idx = _xnys().sessions_in_range(start.isoformat(), end.isoformat())
    return [ts.date() for ts in idx]


@functools.lru_cache(maxsize=4096)
def nth_trading_day_after(session: date, n: int) -> Optional[date]:
    """The n-th NYSE session strictly after `session` (n >= 1), or None if unavailable."""
    if n < 1:
        return None
    end_guess = session + timedelta(days=n * 2 + 16)   # generous upper bound for holidays/weekends
    sess = sessions_between(session + timedelta(days=1), end_guess)
    return sess[n - 1] if len(sess) >= n else None


@functools.lru_cache(maxsize=16384)
def trading_days_after(session: date, expiry: date) -> int:
    """Count of NYSE sessions in (session, expiry] (0 if expiry <= session)."""
    if expiry <= session:
        return 0
    return len(sessions_between(session + timedelta(days=1), expiry))
