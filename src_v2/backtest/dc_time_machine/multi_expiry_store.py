# src_v2/backtest/dc_time_machine/multi_expiry_store.py
"""Multi-expiration SPX option quote store for the DC backtest.

One pickle per session: {expiry_date: {(minute,strike,is_call): (bid,ask)}}.
The inner quote-map shape matches the existing CbboStore caches.
"""
from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]
SessionPanel = Dict[date, Dict[QuoteKey, QuoteVal]]


class MultiExpiryStore:
    def __init__(self, cache_dir: Optional[str | Path] = None,
                 preloaded: Optional[Dict[date, SessionPanel]] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: Optional[Dict[date, SessionPanel]] = preloaded
        self._panel: SessionPanel = {}
        self._session: Optional[date] = None

    def _raw(self, session: date) -> SessionPanel:
        if self._mem is not None and session in self._mem:
            return self._mem[session]
        if self.cache_dir:
            p = self.cache_dir / f"{session.isoformat()}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    return pickle.load(f)
        return {}

    def load_session(self, session: date) -> int:
        self._panel = self._raw(session)
        self._session = session
        return sum(len(qm) for qm in self._panel.values())

    def available_expiries(self) -> List[date]:
        return sorted(self._panel.keys())

    def strikes(self, expiry: date) -> set[float]:
        return {k[1] for k in self._panel.get(expiry, {})}

    def lookup(self, expiry: date, minute_et: str, strike: float, is_call: bool,
               max_stale: int = 5) -> Optional[QuoteVal]:
        qm = self._panel.get(expiry)
        if not qm:
            return None
        key = (minute_et, strike, is_call)
        if key in qm:
            return qm[key]
        hh, mm = int(minute_et[:2]), int(minute_et[3:5])
        for off in range(1, max_stale + 1):
            pmm, phh = mm - off, hh
            while pmm < 0:
                pmm += 60
                phh -= 1
            if phh < 0:
                break
            pkey = (f"{phh:02d}:{pmm:02d}", strike, is_call)
            if pkey in qm:
                return qm[pkey]
        return None

    def derive_spot(self, expiry: date, minute_et: str, ref: float,
                    max_stale: int = 5) -> Optional[float]:
        ks = self.strikes(expiry)
        if not ks:
            return None
        atm = min(ks, key=lambda k: abs(k - ref))
        cq = self.lookup(expiry, minute_et, atm, True, max_stale)
        pq = self.lookup(expiry, minute_et, atm, False, max_stale)
        if cq is None or pq is None:
            return None
        c_mid = (cq[0] + cq[1]) / 2
        p_mid = (pq[0] + pq[1]) / 2
        return round(atm + c_mid - p_mid, 2)

    def settlement(self, expiry: date) -> Optional[float]:
        """Cash settlement = parity forward at the expiry's own last loaded minute.

        Reads session == expiry directly (does not disturb the working session).
        """
        panel = self._raw(expiry)
        qm = panel.get(expiry)
        if not qm:
            return None
        last_min = max(k[0] for k in qm)
        best = None
        for strike in {k[1] for k in qm if k[0] == last_min}:
            cq = qm.get((last_min, strike, True))
            pq = qm.get((last_min, strike, False))
            if cq is None or pq is None:
                continue
            c_mid = (cq[0] + cq[1]) / 2
            p_mid = (pq[0] + pq[1]) / 2
            diff = abs(c_mid - p_mid)
            fwd = strike + c_mid - p_mid
            if best is None or diff < best[0]:
                best = (diff, fwd)
        return round(best[1], 2) if best else None
