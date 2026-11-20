# zerodte/flux/scanner.py
"""Rank expiry-pairs by how much the front/back ratio dropped (or rose) vs the prior session."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

from zerodte.backtest.dc_time_machine.flux_scanner import rank_candidates, ratio_zscore
from zerodte.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after,
    sessions_between,
)
from zerodte.flux.engine import ratio_point
from zerodte.flux.models import ScanRow
from zerodte.flux.sources.base import FluxSource


def _prev_session(d: date) -> date:
    s = sessions_between(d - timedelta(days=10), d)
    return s[-2] if len(s) >= 2 and s[-1] == d else s[-1]


def scan(source: FluxSource, on_date: date, min_dte: int, max_dte: int,
         gaps: List[int], ref_delta: float, by: str = "drop") -> List[ScanRow]:
    from concurrent.futures import ThreadPoolExecutor

    prev = _prev_session(on_date)
    pairs = []
    needed = set()
    for dte in range(min_dte, max_dte + 1):
        front = nth_trading_day_after(on_date, dte)
        if front is None:
            continue
        for gap in gaps:
            back = nth_trading_day_after(on_date, dte + gap)
            if back is None:
                continue
            pairs.append((front, back, gap))
            for sess in (on_date, prev):
                needed.add((sess, front)); needed.add((sess, back))  # noqa: E702

    # Pre-fetch every unique (session, expiry) chain once, in parallel & fault-tolerant
    # (shared expiries across pairs → far fewer fetches; the cache makes repeats instant).
    def _fetch(key):
        sess, exp = key
        try:
            return key, source.chain(exp, on_date=sess)
        except Exception:
            return key, None
    chains = {}
    if needed:
        with ThreadPoolExecutor(max_workers=min(3, len(needed))) as ex:
            for key, snap in ex.map(_fetch, list(needed)):
                chains[key] = snap

    def _ratio(sess, front, back):
        fc, bc = chains.get((sess, front)), chains.get((sess, back))
        if fc is None or bc is None:
            return None
        rp = ratio_point(fc, bc, ref_delta)
        return rp.ratio if rp else None

    raw = []
    for front, back, gap in pairs:
        today = _ratio(on_date, front, back)
        yest = _ratio(prev, front, back)
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
