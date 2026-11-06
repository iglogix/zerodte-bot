"""
CBBO Price Store — loads SPX option CBBO quotes from pickle cache for backtesting.

Provides real bid/ask quotes per (minute, strike, is_call) for 0DTE options.
Loads one day at a time to keep memory bounded.

Usage:
    store = CbboStore("data/cbbo_cache", cache_dir="data/cbbo_cache")
    store.load_day(date(2025, 9, 22))
    bid, ask = store.lookup("09:45", 5900.0, False)
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

QuoteMap = Dict[Tuple[str, float, bool], Tuple[float, float]]


class CbboStore:
    """Loads and serves 0DTE SPX option quotes from pickle cache files."""

    def __init__(self, opra_dir: str | Path, cache_dir: str | Path = "",
                 preloaded: Optional[Dict] = None):
        self.opra_dir = Path(opra_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._quotes: QuoteMap = {}
        self._loaded_date: Optional[date] = None
        self._strikes: set[float] = set()
        self._mem_cache: Optional[Dict] = preloaded

    @staticmethod
    def preload_cache(
        cache_dir: str | Path,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> Dict:
        """Load pickle files into memory. Returns {date: QuoteMap}.

        Optional start/end (inclusive) restrict which days are loaded — useful
        for keeping forked-worker memory in check on long backtests.
        """
        cache_path = Path(cache_dir)
        mem: Dict = {}
        for pkl in sorted(cache_path.glob("*.pkl")):
            d = date.fromisoformat(pkl.stem)
            if start and d < start:
                continue
            if end and d > end:
                continue
            with open(pkl, "rb") as f:
                mem[d] = pickle.load(f)
        return mem

    @property
    def loaded_date(self) -> Optional[date]:
        return self._loaded_date

    @property
    def available_strikes(self) -> set[float]:
        return self._strikes

    def load_day(self, trading_date: date) -> int:
        """Load 0DTE quotes for a single trading day. Returns record count."""
        self._quotes = {}
        self._strikes = set()
        self._loaded_date = trading_date

        if self._mem_cache is not None and trading_date in self._mem_cache:
            self._quotes = self._mem_cache[trading_date]
            self._strikes = {k[1] for k in self._quotes}
            return len(self._quotes)

        if self.cache_dir:
            pkl_path = self.cache_dir / f"{trading_date.isoformat()}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    self._quotes = pickle.load(f)
                self._strikes = {k[1] for k in self._quotes}
                return len(self._quotes)

        return 0

    def lookup(
        self,
        minute_et: str,
        strike: float,
        is_call: bool,
        max_stale_minutes: int = 5,
    ) -> Optional[Tuple[float, float]]:
        """Look up (bid, ask) for given minute/strike/type.

        Searches back up to max_stale_minutes if exact minute not found.
        """
        key = (minute_et, strike, is_call)
        if key in self._quotes:
            return self._quotes[key]

        hh, mm = int(minute_et[:2]), int(minute_et[3:5])
        for offset in range(1, max_stale_minutes + 1):
            prev_mm = mm - offset
            prev_hh = hh
            while prev_mm < 0:
                prev_mm += 60
                prev_hh -= 1
            if prev_hh < 0:
                break
            prev_key = (f"{prev_hh:02d}:{prev_mm:02d}", strike, is_call)
            if prev_key in self._quotes:
                return self._quotes[prev_key]

        return None

    def atm_straddle(
        self,
        minute_et: str,
        spot: float,
        max_stale_minutes: int = 5,
    ) -> Optional[float]:
        """ATM straddle price (put_mid + call_mid at nearest strike to spot)."""
        if not self._strikes:
            return None
        atm_k = min(self._strikes, key=lambda k: abs(k - spot))
        put_q = self.lookup(minute_et, atm_k, False, max_stale_minutes)
        call_q = self.lookup(minute_et, atm_k, True, max_stale_minutes)
        if put_q is None or call_q is None:
            return None
        return (put_q[0] + put_q[1]) / 2 + (call_q[0] + call_q[1]) / 2

    def derive_spot(
        self,
        minute_et: str,
        bar_price: float,
        max_stale_minutes: int = 5,
    ) -> Optional[float]:
        """Derive index spot from put-call parity: S = K + C_mid - P_mid."""
        if not self._strikes:
            return None
        atm_k = min(self._strikes, key=lambda k: abs(k - bar_price))
        put_q = self.lookup(minute_et, atm_k, False, max_stale_minutes)
        call_q = self.lookup(minute_et, atm_k, True, max_stale_minutes)
        if put_q is None or call_q is None:
            return None
        c_mid = (call_q[0] + call_q[1]) / 2
        p_mid = (put_q[0] + put_q[1]) / 2
        return round(atm_k + c_mid - p_mid, 2)

    def available_dates(self) -> set[date]:
        return {date.fromisoformat(p.stem)
                for p in Path(self.cache_dir or self.opra_dir).glob("*.pkl")}

    def settlement_forward(self) -> Optional[float]:
        """Cash-settlement forward at the latest loaded minute, using the strike
        that minimises |call_mid - put_mid| (true ATM) for a robust parity value.
        Independent of any external spot reference (avoids the SPYx10 basis)."""
        if not self._quotes:
            return None
        last_min = max(k[0] for k in self._quotes)
        best: Optional[Tuple[float, float]] = None
        for k in {key[1] for key in self._quotes if key[0] == last_min}:
            cq = self._quotes.get((last_min, k, True))
            pq = self._quotes.get((last_min, k, False))
            if cq is None or pq is None:
                continue
            c_mid = (cq[0] + cq[1]) / 2
            p_mid = (pq[0] + pq[1]) / 2
            diff = abs(c_mid - p_mid)
            if best is None or diff < best[0]:
                best = (diff, k + c_mid - p_mid)
        return round(best[1], 2) if best else None


class MergedCbboStore(CbboStore):
    """Lazy per-day store that merges a base cache dir with an overlay dir
    (base wins on key conflicts; the overlay contributes extra strikes/minutes).
    Loads one day at a time to keep memory bounded (caches are ~2.5 GB each)."""

    def __init__(self, base_dir: str, overlay_dir: str = ""):
        super().__init__(base_dir, cache_dir=base_dir)
        self._overlay_dir = Path(overlay_dir) if overlay_dir else None

    def available_dates(self) -> set[date]:
        dates = super().available_dates()
        if self._overlay_dir:
            dates |= {date.fromisoformat(p.stem)
                      for p in self._overlay_dir.glob("*.pkl")}
        return dates

    def load_day(self, trading_date: date) -> int:
        n = super().load_day(trading_date)
        if self._overlay_dir:
            pkl = self._overlay_dir / f"{trading_date.isoformat()}.pkl"
            if pkl.exists():
                with open(pkl, "rb") as f:
                    overlay = pickle.load(f)
                if n == 0:
                    self._quotes = overlay
                else:
                    for k, v in overlay.items():
                        self._quotes.setdefault(k, v)
                self._strikes = {k[1] for k in self._quotes}
                n = len(self._quotes)
        return n
