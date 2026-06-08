# src_v2/flux/sources/marketdata.py
"""marketdata.app adapter — returns IV/delta directly (no Black–Scholes)."""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from src_v2.flux.cache import ChainCache
from src_v2.flux.models import ChainSnapshot, OptionRow
from src_v2.flux.sources.base import FluxSource

BASE = "https://api.marketdata.app/v1"
COLUMNS = "optionSymbol,side,strike,bid,ask,iv,delta,dte,updated"


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
    def fetch(url: str) -> dict:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            return json.loads(body) if body else {}
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
        rows: List[OptionRow] = []
        as_of = datetime.now(timezone.utc)
        for i in range(len(strikes)):
            iv = data["iv"][i] if data.get("iv") else None
            dl = data["delta"][i] if data.get("delta") else None
            if iv is None or dl is None:
                continue
            rows.append(OptionRow(
                strike=float(strikes[i]),
                is_call=(data["side"][i] == "call"),
                bid=float(data["bid"][i]) if data["bid"][i] is not None else 0.0,
                ask=float(data["ask"][i]) if data["ask"][i] is not None else 0.0,
                iv=float(iv), delta=float(dl)))
            if data.get("updated") and data["updated"][i]:
                as_of = datetime.fromtimestamp(int(data["updated"][i]), tz=timezone.utc)
        return ChainSnapshot(self.underlying, expiry, as_of, tuple(rows))
