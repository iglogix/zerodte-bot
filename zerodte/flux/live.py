# zerodte/flux/live.py
"""Intraday RatioPoint buffer — polls the live chain only during the NYSE session."""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

from zerodte.backtest.dc_time_machine.trading_calendar import sessions_between
from zerodte.flux.engine import ratio_point
from zerodte.flux.models import RatioPoint
from zerodte.flux.sources.base import FluxSource

ET = ZoneInfo("America/New_York")


def _market_open(now_utc: datetime) -> bool:
    et = now_utc.astimezone(ET)
    d = et.date()
    if not sessions_between(d, d):          # not a trading day
        return False
    return time(9, 30) <= et.time() <= time(16, 0)


class LivePoller:
    def __init__(self, source: FluxSource, front_dte: int, gap: int, ref_delta: float,
                 poll_seconds: int = 60, now_utc: Optional[Callable[[], datetime]] = None):
        self.source = source
        self.front_dte, self.gap, self.ref_delta = front_dte, gap, ref_delta
        self.poll_seconds = poll_seconds
        self._now = now_utc or (lambda: datetime.now(timezone.utc))
        self.buffer: List[RatioPoint] = []

    def poll_once(self) -> Optional[RatioPoint]:
        now = self._now()
        if not _market_open(now):
            return None
        from zerodte.backtest.dc_time_machine.trading_calendar import nth_trading_day_after
        today = now.astimezone(ET).date()
        front = nth_trading_day_after(today, self.front_dte)
        back = nth_trading_day_after(today, self.front_dte + self.gap)
        if front is None or back is None:
            return None
        rp = ratio_point(self.source.chain(front), self.source.chain(back), self.ref_delta)
        if rp is not None:
            self.buffer.append(rp)
        return rp
