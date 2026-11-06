# zerodte/backtest/dc_time_machine/spreads.py
"""Empirical half-spread study — the slippage experiment (spec §5.1).

Stratifies observed half-spreads by DTE bucket and time-of-day bucket so the fill
model's `k` is grounded in observed liquidity rather than assumed.
"""
from __future__ import annotations

import statistics
from datetime import date
from typing import Dict, List, Tuple

from zerodte.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore


def _tod_bucket(minute: str) -> str:
    hh = int(minute[:2])
    if hh < 11:
        return "open"
    if hh < 15:
        return "midday"
    return "close"


def half_spread_stats(store: MultiExpiryStore, sessions: List[date]
                      ) -> Dict[Tuple[int, str], Dict[str, float]]:
    """Return {(dte, tod_bucket): {median, p90, n}} of half-spreads ($/share)."""
    buckets: Dict[Tuple[int, str], List[float]] = {}
    for session in sessions:
        store.load_session(session)
        for expiry in store.available_expiries():
            dte = (expiry - session).days
            qm_keys = store.strikes  # noqa: not used; iterate panel directly below
            for (minute, strike, is_call), (bid, ask) in _iter_quotes(store, expiry):
                half = (ask - bid) / 2.0
                if half < 0:
                    continue
                buckets.setdefault((dte, _tod_bucket(minute)), []).append(half)
    out: Dict[Tuple[int, str], Dict[str, float]] = {}
    for key, vals in buckets.items():
        vals_sorted = sorted(vals)
        p90 = vals_sorted[min(len(vals_sorted) - 1, int(0.9 * len(vals_sorted)))]
        out[key] = {"median": statistics.median(vals_sorted), "p90": p90, "n": len(vals_sorted)}
    return out


def _iter_quotes(store: MultiExpiryStore, expiry: date):
    """Yield ((minute, strike, is_call), (bid, ask)) for one expiry of the loaded session."""
    panel = store._panel.get(expiry, {})   # internal access by design (same package)
    for key, val in panel.items():
        yield key, val
