"""ES → SPX basis estimator using existing CBBO cache.

Used offline to estimate SPX_spot from ES futures price for historical dates
where we don't yet have CBBO data. SPX = ES - basis. Basis is dominated by
dividend drift + interest rate carry, so a per-month median is sufficient.
"""
from __future__ import annotations

import pickle
import statistics
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_pickle(cache_dir: Path, day: date) -> Optional[dict]:
    p = cache_dir / f"{day.isoformat()}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def compute_daily_basis(
    cache_dir: Path,
    day: date,
    es_open: float,
    anchor_minute: str = "09:31",
) -> Optional[float]:
    """basis = es_open − spx_spot, where spx_spot is derived from put-call parity.

    Picks the strike closest to es_open with both call and put quotes at anchor_minute.
    Returns None if no usable quotes.
    """
    quotes = _load_pickle(cache_dir, day)
    if not quotes:
        return None

    # All strikes that have BOTH a call and put quote at anchor_minute
    by_strike: Dict[float, Dict[bool, Tuple[float, float]]] = {}
    for (m, strike, is_call), (bid, ask) in quotes.items():
        if m != anchor_minute:
            continue
        by_strike.setdefault(strike, {})[is_call] = (bid, ask)

    paired = [k for k, sides in by_strike.items() if True in sides and False in sides]
    if not paired:
        return None

    atm = min(paired, key=lambda k: abs(k - es_open))
    c_bid, c_ask = by_strike[atm][True]
    p_bid, p_ask = by_strike[atm][False]
    c_mid = (c_bid + c_ask) / 2
    p_mid = (p_bid + p_ask) / 2
    spx = atm + c_mid - p_mid
    return round(es_open - spx, 2)


def build_monthly_basis_table(
    cache_dir: Path,
    es_opens: Dict[date, float],
    anchor_minute: str = "09:31",
) -> Dict[Tuple[int, int], float]:
    """For each (year, month) present in cache_dir, compute median basis across days."""
    by_month: Dict[Tuple[int, int], list] = {}
    for day, es_open in es_opens.items():
        b = compute_daily_basis(cache_dir, day, es_open, anchor_minute)
        if b is None:
            continue
        by_month.setdefault((day.year, day.month), []).append(b)
    return {ym: round(statistics.median(vals), 2) for ym, vals in by_month.items() if vals}


def estimate_spx_from_es(
    day: date,
    es_price: float,
    monthly_basis: Dict[Tuple[int, int], float],
    default_basis: float = 10.0,
) -> float:
    """SPX_spot = ES_price − basis(month). Falls back to default_basis if month missing."""
    basis = monthly_basis.get((day.year, day.month), default_basis)
    return round(es_price - basis, 2)
