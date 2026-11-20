# zerodte/flux/sources/marketdata.py
"""marketdata.app adapter — returns IV/delta directly (no Black–Scholes)."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from zerodte.backtest.dc_time_machine.iv import bs_delta, implied_vol
from zerodte.flux.cache import ChainCache
from zerodte.flux.models import ChainSnapshot, OptionRow
from zerodte.flux.sources.base import FluxSource

BASE = "https://api.marketdata.app/v1"
COLUMNS = "optionSymbol,side,strike,bid,ask,iv,delta,dte,updated"
RATE = 0.04   # risk-free rate for BS inversion of historical (greek-less) EOD chains


def load_token() -> Optional[str]:
    """MARKET_DATA_TOKEN from env, else ../options_api/.env (existing repo convention)."""
    t = os.getenv("MARKET_DATA_TOKEN")
    if t:
        return t
    for p in (Path("../options_api/.env"), Path(".env")):
        try:
            for line in p.read_text().splitlines():
                if line.startswith("MARKET_DATA_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            continue
    return None


def _default_fetch(token: str) -> Callable[[str], dict]:
    import time as _t
    import urllib.error

    def fetch(url: str) -> dict:
        last: Optional[Exception] = None
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
                with urllib.request.urlopen(req, timeout=25) as r:
                    body = r.read()
                    return json.loads(body) if body else {}
            except urllib.error.HTTPError as e:
                # 400 (e.g. today's EOD / invalid expiry) and other 4xx == no data.
                # 429 (rate limit) and 5xx are transient → retry.
                if e.code == 429 or e.code >= 500:
                    last = e; _t.sleep(1.5 * (attempt + 1)); continue  # noqa: E702
                return {}
            except Exception as e:                      # timeout / URLError → retry
                last = e; _t.sleep(1.0 * (attempt + 1)); continue  # noqa: E702
        if last:
            raise last
        return {}
    return fetch


class MarketDataSource(FluxSource):
    def __init__(self, token: Optional[str] = None,
                 fetch_json: Optional[Callable[[str], dict]] = None,
                 strike_limit: int = 25, underlying: str = "SPX",
                 cache: Optional["ChainCache"] = None):
        self.token = token or load_token()
        self.strike_limit = strike_limit
        self.underlying = underlying
        self.cache = cache
        self._fetch = fetch_json or _default_fetch(self.token or "")

    def chain(self, expiry: date, on_date: Optional[date] = None) -> ChainSnapshot:
        # EOD snapshots are immutable → cache them to bound marketdata.app credit spend.
        # Live snapshots (on_date=None) are never cached.
        if on_date is not None and self.cache is not None:
            cached = self.cache.get(self.underlying, expiry, on_date)
            if cached is not None:
                return cached
            snap = self._fetch_chain(expiry, on_date)
            # Only cache NON-EMPTY snapshots. An empty chain for a recent session means
            # marketdata.app hasn't settled it yet (transient) — caching it would pin a stale
            # "no data" and make the scanner skip that session for good.
            if snap.rows:
                self.cache.put(snap, on_date)
            return snap
        return self._fetch_chain(expiry, on_date)

    def _fetch_chain(self, expiry: date, on_date: Optional[date]) -> ChainSnapshot:
        url = (f"{BASE}/options/chain/{self.underlying}/?expiration={expiry.isoformat()}"
               f"&strikeLimit={self.strike_limit}&columns={COLUMNS}")
        if on_date is not None:
            url += f"&date={on_date.isoformat()}"
        data = self._fetch(url) or {}
        strikes = data.get("strike") or []
        n = len(strikes)
        bid = data.get("bid") or [None] * n
        ask = data.get("ask") or [None] * n
        side = data.get("side") or [""] * n
        iv_raw = data.get("iv") or [None] * n
        dl_raw = data.get("delta") or [None] * n
        upd = data.get("updated") or []

        as_of = datetime.now(timezone.utc)
        if upd and upd[0]:
            as_of = datetime.fromtimestamp(int(upd[0]), tz=timezone.utc)

        prelim = []
        for i in range(n):
            b = float(bid[i]) if bid[i] is not None else 0.0
            a = float(ask[i]) if ask[i] is not None else 0.0
            prelim.append((float(strikes[i]), side[i] == "call", b, a, iv_raw[i], dl_raw[i]))

        # marketdata.app returns iv/delta on the LIVE feed but NOT on historical EOD
        # (`date=`) — there ALL are null. When the whole chain lacks iv, compute it via
        # Black–Scholes from the bid/ask mid, a parity-derived spot, and DTE. (A live
        # chain with the occasional null strike takes the drop-nulls path instead.)
        need_compute = bool(prelim) and all(p[4] is None for p in prelim)
        rows: List[OptionRow] = []
        if need_compute:
            spot = _parity_spot(prelim)
            ref = on_date or datetime.now(timezone.utc).date()
            T = max((expiry - ref).days, 0) / 365.0
            for (k, is_call, b, a, iv, dl) in prelim:
                if iv is None:
                    mid = (b + a) / 2.0
                    if mid <= 0 or spot is None or T <= 0:
                        continue
                    iv = implied_vol(mid, spot, k, T, RATE, is_call)
                    if iv != iv:                       # NaN (price ≤ intrinsic)
                        continue
                    dl = bs_delta(spot, k, T, float(iv), RATE, is_call)
                rows.append(OptionRow(k, is_call, b, a, float(iv), float(dl)))
        else:
            for (k, is_call, b, a, iv, dl) in prelim:
                if iv is None or dl is None:
                    continue
                rows.append(OptionRow(k, is_call, b, a, float(iv), float(dl)))
        return ChainSnapshot(self.underlying, expiry, as_of, tuple(rows))


def _parity_spot(prelim) -> Optional[float]:
    """Spot via put-call parity at the strike with the smallest |call_mid − put_mid|."""
    by_strike: dict = {}
    for (k, is_call, b, a, *_rest) in prelim:
        by_strike.setdefault(k, {})[is_call] = (b + a) / 2.0
    best = None
    for k, cp in by_strike.items():
        if True in cp and False in cp:
            diff = abs(cp[True] - cp[False])
            if best is None or diff < best[0]:
                best = (diff, k + cp[True] - cp[False])
    return best[1] if best else None
