"""Intraday IV-ratio series from ThetaData (temporary live-feed stand-in for TastyTrade).

ThetaData serves a whole day's 1-minute option panel in one bulk call per expiry, so the
full intraday "Today" line is built without minute-by-minute polling and without needing
the market open or a TT session. IV is computed via Black–Scholes per minute (same
`iv_compute` path as everywhere else). Re-poll to pick up new minutes as the day
progresses (ThetaData is delayed ~15 min).
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime, time
from typing import Callable, Dict, List, Tuple
from zoneinfo import ZoneInfo

from src_v2.flux.engine import ratio_point
from src_v2.flux.iv_compute import RATE, compute_iv_rows, parity_spot
from src_v2.flux.models import ChainSnapshot, RatioPoint
from src_v2.backtest.dc_time_machine.theta_backfill import (
    THETA, _http_json, fetch_expiry_session_theta,
)

ET = ZoneInfo("America/New_York")
QuoteKey = Tuple[str, float, bool]
# The opening-auction bar (09:30) has stale/wide quotes → a garbage IV/ratio outlier that
# blows out the chart's autoscale. Skip bars before this time; mid-day points are kept
# (their spikes are the signal, not noise).
SKIP_BEFORE = "09:35"


def _ts(session: date, minute: str) -> datetime:
    hh, mm = int(minute[:2]), int(minute[3:5])
    return datetime.combine(session, time(hh, mm, tzinfo=ET))


def _by_minute(panel: Dict[QuoteKey, Tuple[float, float]]):
    out: Dict[str, List[Tuple[float, bool, float, float]]] = {}
    for (minute, strike, is_call), (bid, ask) in panel.items():
        out.setdefault(minute, []).append((strike, is_call, bid, ask))
    return out


class ThetaIntradaySource:
    def __init__(self, base: str = THETA, fetch_json: Callable[[str], dict] = _http_json,
                 half_width_pts: int = 300, rate: float = RATE,
                 ttl_seconds: float = 60.0, clock: Callable[[], float] = _time.time,
                 today_fn: Callable[[], date] = date.today, default_spot: float = 7400.0):
        self.base = base
        self.fetch_json = fetch_json
        self.half_width_pts = half_width_pts   # wide enough that a rough hint still covers ATM
        self.rate = rate
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._today = today_fn
        self._cache: Dict[tuple, Tuple[float, List[RatioPoint]]] = {}
        # Self-correcting ATM hint derived from the data — no marketdata/TT dependency.
        self.last_spot: float = default_spot

    def intraday(self, session: date, front_exp: date, back_exp: date,
                 ref_delta: float, atm_hint: float) -> List[RatioPoint]:
        # Past sessions are immutable → cache forever; today refreshes after ttl_seconds
        # (Theta is ~15 min delayed, so a minute of staleness is harmless). This makes
        # switching back to the 1-min view instant.
        key = (session, front_exp, back_exp, round(ref_delta, 3))   # not atm_hint: it drifts
        hit = self._cache.get(key)
        if hit is not None:
            ts, series = hit
            if session < self._today() or (self._clock() - ts) < self.ttl_seconds:
                return series
        series = self._build(session, front_exp, back_exp, ref_delta, atm_hint)
        self._cache[key] = (self._clock(), series)
        return series

    def _build(self, session: date, front_exp: date, back_exp: date,
               ref_delta: float, atm_hint: float) -> List[RatioPoint]:
        f_panel = fetch_expiry_session_theta(session, front_exp, atm_hint,
                                             self.half_width_pts, self.base, self.fetch_json)
        b_panel = fetch_expiry_session_theta(session, back_exp, atm_hint,
                                             self.half_width_pts, self.base, self.fetch_json)
        fm, bm = _by_minute(f_panel), _by_minute(b_panel)
        # Update the self-correcting hint from a mid-session parity spot (so the next fetch
        # centers correctly without any external spot lookup).
        if fm:
            mid = sorted(fm)[len(fm) // 2]
            sp = parity_spot([(s, c, b, a) for (s, c, b, a) in fm[mid]])
            if sp:
                self.last_spot = sp
        Tf = max((front_exp - session).days, 0) / 365.0
        Tb = max((back_exp - session).days, 0) / 365.0
        out: List[RatioPoint] = []
        for minute in sorted(set(fm) & set(bm)):
            if minute < SKIP_BEFORE:          # drop the opening-auction artifact bar
                continue
            f_rows, _ = compute_iv_rows(fm[minute], Tf, None, self.rate)
            b_rows, _ = compute_iv_rows(bm[minute], Tb, None, self.rate)
            ts = _ts(session, minute)
            rp = ratio_point(ChainSnapshot("SPX", front_exp, ts, tuple(f_rows)),
                             ChainSnapshot("SPX", back_exp, ts, tuple(b_rows)), ref_delta)
            if rp is not None:
                out.append(RatioPoint(ts, front_exp, back_exp, ref_delta,
                                      rp.front_iv, rp.back_iv, rp.ratio))
        return out
