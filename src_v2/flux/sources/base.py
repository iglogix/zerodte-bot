# src_v2/flux/sources/base.py
"""FluxSource interface. chain() is provider-specific; expirations() is derived from the
NYSE calendar (SPX has near-daily expiries; a chain() that returns empty == not an expiry)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

from src_v2.flux.models import ChainSnapshot
from src_v2.backtest.dc_time_machine.trading_calendar import nth_trading_day_after


class FluxSource(ABC):
    @abstractmethod
    def chain(self, expiry: date, on_date: Optional[date] = None) -> ChainSnapshot:
        ...

    def expirations(self, ref: date, min_dte: int, max_dte: int) -> List[date]:
        out: List[date] = []
        for n in range(max(0, min_dte), max_dte + 1):
            d = ref if n == 0 else nth_trading_day_after(ref, n)
            if d is not None:
                out.append(d)
        return out
