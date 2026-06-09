"""Live 1-min intraday from the BROKER (TastyTrade).

The chart's morning is seeded once from `seed` (e.g. ThetaIntradaySource) so a mid-day open
isn't empty; the live tail is TT current-minute snapshots. TT bid/ask is run through the SAME
`compute_iv_rows` path as the seed, so the IV/ratio definition is identical across the seam.

Because this is wired as the cockpit's `intraday_source`, `latest_chain()` also feeds the DC
and transform order builders — so orders get priced off the broker's book (what you execute
against), which was the whole reason to prefer TT.

`chain_fn(expiry) -> [{strike, option_type, bid, ask}]` is injected (the live wiring runs the
TT client's get_option_chain on its event loop) so the source is fully unit-testable offline.
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional

from src_v2.flux.engine import ratio_point
from src_v2.flux.iv_compute import RATE, compute_iv_rows
from src_v2.flux.models import ChainSnapshot, RatioPoint
from src_v2.flux.sources.theta_intraday import ET, SKIP_BEFORE, _ts

ChainFn = Callable[[date], List[dict]]


class TastyTradeIntradaySource:
    def __init__(self, chain_fn: ChainFn, seed=None, rate: float = RATE,
                 clock: Callable[[], float] = _time.time, today_fn: Callable[[], date] = date.today,
                 now_fn: Optional[Callable[[], datetime]] = None,
                 chain_ttl: float = 15.0, default_spot: float = 7400.0):
        self.chain_fn = chain_fn
        self.seed = seed                  # morning backfill source (past minutes); None = pure TT
        self.rate = rate
        self._clock = clock
        self._today = today_fn
        self._now = now_fn or (lambda: datetime.now(ET))   # wall-clock for the live minute
        self.chain_ttl = chain_ttl        # dedupe live fetches within one refresh cycle
        self.last_spot = default_spot
        self._acc: Dict[tuple, Dict[str, RatioPoint]] = {}   # series accumulators (per pair/day)
        self._seeded: set = set()
        self._chain_cache: Dict[date, tuple] = {}            # expiry -> (ts, ChainSnapshot)

    def _live_chain(self, expiry: date) -> ChainSnapshot:
        hit = self._chain_cache.get(expiry)
        if hit is not None and (self._clock() - hit[0]) < self.chain_ttl:
            return hit[1]
        raw = self.chain_fn(expiry) or []
        price_rows = [(float(o["strike"]), o["option_type"] == "call", float(o["bid"]), float(o["ask"]))
                      for o in raw if (o.get("bid") or 0) > 0 and (o.get("ask") or 0) > 0]
        T = max((expiry - self._today()).days, 0) / 365.0
        rows, spot = compute_iv_rows(price_rows, T, None, self.rate)
        if spot:
            self.last_spot = spot
        snap = ChainSnapshot("SPX", expiry, datetime.now(timezone.utc), tuple(rows))
        self._chain_cache[expiry] = (self._clock(), snap)
        return snap

    def latest_chain(self, session: date, expiry: date,
                     atm_hint: Optional[float] = None) -> ChainSnapshot:
        if session < self._today() and self.seed is not None:
            return self.seed.latest_chain(session, expiry, atm_hint)
        return self._live_chain(expiry)

    def intraday(self, session: date, front_exp: date, back_exp: date,
                 ref_delta: float, atm_hint: float) -> List[RatioPoint]:
        # Past sessions aren't live → serve from the seed (or empty if none).
        if session < self._today():
            return self.seed.intraday(session, front_exp, back_exp, ref_delta, atm_hint) if self.seed else []

        key = (session, front_exp, back_exp, round(ref_delta, 3))
        acc = self._acc.setdefault(key, {})
        if key not in self._seeded:               # one-time morning backfill (immutable past minutes)
            self._seeded.add(key)
            if self.seed is not None:
                try:
                    for rp in self.seed.intraday(session, front_exp, back_exp, ref_delta, atm_hint):
                        acc[rp.as_of.astimezone(ET).strftime("%H:%M")] = rp
                except Exception:
                    pass
        try:                                       # append/refresh the current live minute from TT
            rp = ratio_point(self._live_chain(front_exp), self._live_chain(back_exp), ref_delta)
            if rp is not None:
                minute = self._now().strftime("%H:%M")
                if minute >= SKIP_BEFORE:
                    acc[minute] = RatioPoint(_ts(session, minute), front_exp, back_exp, ref_delta,
                                             rp.front_iv, rp.back_iv, rp.ratio)
        except Exception:
            pass
        return [acc[m] for m in sorted(acc)]
