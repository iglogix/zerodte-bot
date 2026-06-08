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
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timedelta

    sessions = sessions_between(end_date - timedelta(days=n_days * 2 + 10), end_date)[-n_days:]
    triples = []
    needed = set()
    for s in sessions:
        front, back = _resolve(s, front_dte, gap)
        if front is None or back is None:
            continue
        triples.append((s, front, back))
        needed.add((s, front)); needed.add((s, back))

    # Fetch every (session, expiry) chain concurrently — marketdata.app EOD reads are
    # ~5-9s each, so serial fetching of ~2·n chains blows the timeout. The cache makes
    # repeat loads instant; this only parallelises the cold first load.
    def _fetch(key):
        s, exp = key
        try:
            return key, source.chain(exp, on_date=s)
        except Exception:
            return key, None
    chains = {}
    if needed:
        # 3 workers: enough to overlap the ~5-9s EOD reads, low enough that marketdata.app
        # doesn't throttle the cold burst (8-way bursts get 429'd → failures).
        with ThreadPoolExecutor(max_workers=min(3, len(needed))) as ex:
            for key, snap in ex.map(_fetch, list(needed)):
                chains[key] = snap

    out: List[RatioPoint] = []
    for s, front, back in triples:
        fc, bc = chains.get((s, front)), chains.get((s, back))
        if fc is None or bc is None:
            continue
        rp = ratio_point(fc, bc, ref_delta)
        if rp is None:
            continue
        # stamp the point at the session's EOD (16:00 ET ≈ 20:00 UTC) for a clean x-axis
        out.append(RatioPoint(datetime.combine(s, time(20, 0, tzinfo=timezone.utc)),
                              rp.front_expiry, rp.back_expiry, ref_delta,
                              rp.front_iv, rp.back_iv, rp.ratio))
    return out
