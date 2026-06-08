# src_v2/flux/scanner.py
"""Rank expiry-pairs by how much the front/back ratio dropped (or rose) vs the prior session."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

from src_v2.flux.engine import ratio_point
from src_v2.flux.models import ScanRow
from src_v2.flux.sources.base import FluxSource
from src_v2.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after, sessions_between,
)
from src_v2.backtest.dc_time_machine.flux_scanner import ratio_zscore, rank_candidates


def _prev_session(d: date) -> date:
    s = sessions_between(d - timedelta(days=10), d)
    return s[-2] if len(s) >= 2 and s[-1] == d else s[-1]


def _ratio_on(source, session, front, back, ref_delta):
    rp = ratio_point(source.chain(front, on_date=session),
                     source.chain(back, on_date=session), ref_delta)
    return rp.ratio if rp else None


def scan(source: FluxSource, on_date: date, min_dte: int, max_dte: int,
         gaps: List[int], ref_delta: float, by: str = "drop") -> List[ScanRow]:
    prev = _prev_session(on_date)
    raw = []
    for dte in range(min_dte, max_dte + 1):
        front = nth_trading_day_after(on_date, dte)
        if front is None:
            continue
        for gap in gaps:
            back = nth_trading_day_after(on_date, dte + gap)
            if back is None:
                continue
            today = _ratio_on(source, on_date, front, back, ref_delta)
            yest = _ratio_on(source, prev, front, back, ref_delta)
            if today is None or yest is None:
                continue
            raw.append({"front": front, "back": back, "gap": gap,
                        "ratio": today, "ratio_prev": yest})
    ranked = rank_candidates(raw, by=by)
    out: List[ScanRow] = []
    for c in ranked:
        drop = c["ratio_prev"] - c["ratio"]
        z = ratio_zscore(c["ratio"], [c["ratio_prev"], c["ratio"]])
        out.append(ScanRow(front_expiry=c["front"], back_expiry=c["back"], gap_td=c["gap"],
                           ratio=c["ratio"], ratio_prev=c["ratio_prev"], drop=drop,
                           zscore=0.0 if z != z else z,
                           flag=(c["ratio"] > 1.0 and drop > 0)))
    return out
