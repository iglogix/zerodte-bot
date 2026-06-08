# src_v2/flux/history.py
"""Daily EOD front/back ratio series over the last N NYSE sessions."""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import List, Optional

from src_v2.flux.engine import ratio_point
from src_v2.flux.models import RatioPoint
from src_v2.flux.sources.base import FluxSource
from src_v2.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after, sessions_between,
)


def _resolve(session: date, front_dte: int, gap: int):
    return nth_trading_day_after(session, front_dte), nth_trading_day_after(session, front_dte + gap)


def daily_ratio_series(source: FluxSource, front_dte: int, gap: int, n_days: int,
                       ref_delta: float, end_date: date) -> List[RatioPoint]:
    from datetime import timedelta
    sessions = sessions_between(end_date - timedelta(days=n_days * 2 + 10), end_date)[-n_days:]
    out: List[RatioPoint] = []
    for s in sessions:
        front, back = _resolve(s, front_dte, gap)
        if front is None or back is None:
            continue
        fc = source.chain(front, on_date=s)
        bc = source.chain(back, on_date=s)
        rp = ratio_point(fc, bc, ref_delta)
        if rp is None:
            continue
        # stamp the point at the session's EOD (16:00 ET ≈ 20:00 UTC) for a clean x-axis
        stamped = RatioPoint(datetime.combine(s, time(20, 0, tzinfo=timezone.utc)),
                             rp.front_expiry, rp.back_expiry, ref_delta,
                             rp.front_iv, rp.back_iv, rp.ratio)
        out.append(stamped)
    return out
