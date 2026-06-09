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

import threading
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
                 chain_ttl: float = 15.0, default_spot: float = 7400.0,
                 retries: int = 2, retry_backoff: float = 0.5,
                 sleep_fn: Callable[[float], None] = _time.sleep, seed_async: bool = True):
        self.chain_fn = chain_fn
        self.seed = seed                  # morning backfill source (past minutes); None = pure TT
        self.rate = rate
        self._clock = clock
        self._today = today_fn
        self._now = now_fn or (lambda: datetime.now(ET))   # wall-clock for the live minute
        self.retries = retries            # guard: re-fetch on a DXLink hiccup (empty/raise)
        self.retry_backoff = retry_backoff
        self._sleep = sleep_fn
        self.seed_async = seed_async      # seed the morning off-thread so first paint isn't blocked
        self.chain_ttl = chain_ttl        # dedupe live fetches within one refresh cycle
        self.last_spot = default_spot
        self._lock = threading.Lock()
        self._acc: Dict[tuple, Dict[str, RatioPoint]] = {}   # series accumulators (per pair/day)
        self._seeded: set = set()
        self._chain_cache: Dict[date, tuple] = {}            # expiry -> (ts, ChainSnapshot)

    def _fetch_chain_once(self, expiry: date) -> Optional[ChainSnapshot]:
        raw = self.chain_fn(expiry) or []
        price_rows = [(float(o["strike"]), o["option_type"] == "call", float(o["bid"]), float(o["ask"]))
                      for o in raw if (o.get("bid") or 0) > 0 and (o.get("ask") or 0) > 0]
        if not price_rows:                # REST chain came back but DXLink enrichment was empty
            return None
        T = max((expiry - self._today()).days, 0) / 365.0
        rows, spot = compute_iv_rows(price_rows, T, None, self.rate)
        if spot:
            self.last_spot = spot
        return ChainSnapshot("SPX", expiry, datetime.now(timezone.utc), tuple(rows)) if rows else None

    def _live_chain(self, expiry: date) -> ChainSnapshot:
        hit = self._chain_cache.get(expiry)
        if hit is not None and (self._clock() - hit[0]) < self.chain_ttl:
            return hit[1]
        # Reconnect guard: a DXLink hiccup shows up as a raise or an empty enrichment. Retry with
        # backoff (a fresh fetch re-subscribes); on total failure serve the last good chain (stale
        # but non-broken) so the live chart never blanks out.
        snap = None
        for attempt in range(self.retries + 1):
            try:
                snap = self._fetch_chain_once(expiry)
            except Exception:
                snap = None
            if snap is not None:
                break
            if attempt < self.retries:
                self._sleep(self.retry_backoff * (attempt + 1))
        if snap is None:
            # TT is down → fall back to a REAL chain from the seed (Theta) before serving stale,
            # so prices stay live. Not cached (it's not TT) → the next call retries TT.
            if self.seed is not None:
                try:
                    tsnap = self.seed.latest_chain(self._today(), expiry)
                    if tsnap is not None and tsnap.rows:
                        return tsnap
                except Exception:
                    pass
            return hit[1] if hit is not None else ChainSnapshot(
                "SPX", expiry, datetime.now(timezone.utc), ())
        self._chain_cache[expiry] = (self._clock(), snap)
        return snap

    def latest_chain(self, session: date, expiry: date,
                     atm_hint: Optional[float] = None) -> ChainSnapshot:
        if session < self._today() and self.seed is not None:
            return self.seed.latest_chain(session, expiry, atm_hint)
        return self._live_chain(expiry)

    def _merge_seed(self, key, pts) -> None:
        """Merge morning seed points WITHOUT clobbering live points already present."""
        with self._lock:
            acc = self._acc.setdefault(key, {})
            for rp in pts:
                acc.setdefault(rp.as_of.astimezone(ET).strftime("%H:%M"), rp)

    def _run_seed(self, key, session, front_exp, back_exp, ref_delta, atm_hint) -> None:
        try:
            pts = self.seed.intraday(session, front_exp, back_exp, ref_delta, atm_hint)
        except Exception:
            pts = []
        self._merge_seed(key, pts)

    def intraday(self, session: date, front_exp: date, back_exp: date,
                 ref_delta: float, atm_hint: float) -> List[RatioPoint]:
        # Past sessions aren't live → serve from the seed (or empty if none).
        if session < self._today():
            return self.seed.intraday(session, front_exp, back_exp, ref_delta, atm_hint) if self.seed else []

        key = (session, front_exp, back_exp, round(ref_delta, 3))
        with self._lock:
            self._acc.setdefault(key, {})
            first = key not in self._seeded
            if first:
                self._seeded.add(key)
        if first and self.seed is not None:        # one-time morning backfill (immutable past minutes)
            if self.seed_async:                    # off-thread → first paint isn't blocked by the seed
                threading.Thread(target=self._run_seed, daemon=True,
                                 args=(key, session, front_exp, back_exp, ref_delta, atm_hint)).start()
            else:
                self._run_seed(key, session, front_exp, back_exp, ref_delta, atm_hint)
        try:                                       # append/refresh the current live minute (TT, Theta fallback)
            rp = ratio_point(self._live_chain(front_exp), self._live_chain(back_exp), ref_delta)
            if rp is not None:
                minute = self._now().strftime("%H:%M")
                if minute >= SKIP_BEFORE:
                    with self._lock:
                        self._acc[key][minute] = RatioPoint(
                            _ts(session, minute), front_exp, back_exp, ref_delta,
                            rp.front_iv, rp.back_iv, rp.ratio)
        except Exception:
            pass
        with self._lock:
            acc = self._acc.get(key, {})
            return [acc[m] for m in sorted(acc)]
