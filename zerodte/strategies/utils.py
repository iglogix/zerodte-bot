"""Shared helpers for live trading strategies."""

from __future__ import annotations
from typing import Any, Dict, List, Optional


def mid_price(opt: Optional[Dict[str, Any]]) -> Optional[float]:
    """Compute mid price from bid/ask, or None if invalid."""
    if not opt:
        return None
    bid = opt.get("bid", 0)
    ask = opt.get("ask", 0)
    if bid <= 0 or ask <= 0:
        return None
    return round((bid + ask) / 2, 4)


def derive_spot_from_chain(chain: list, reference_price: float) -> Optional[float]:
    """Derive index spot from put-call parity: S = K + C_mid - P_mid.

    Uses reference_price to find the nearest ATM strike.
    Returns None if chain lacks paired put/call data.
    """
    puts_by_k: Dict[float, Dict] = {}
    calls_by_k: Dict[float, Dict] = {}
    for o in chain:
        k = o.get("strike", 0)
        otype = o.get("option_type", "").lower()
        if otype in ("put", "p") and o.get("bid", 0) > 0:
            puts_by_k[k] = o
        elif otype in ("call", "c") and o.get("bid", 0) > 0:
            calls_by_k[k] = o
    common = sorted(set(puts_by_k) & set(calls_by_k))
    if not common:
        return None
    atm_k = min(common, key=lambda k: abs(k - reference_price))
    c_mid = mid_price(calls_by_k[atm_k])
    p_mid = mid_price(puts_by_k[atm_k])
    if c_mid is None or p_mid is None:
        return None
    return round(atm_k + c_mid - p_mid, 2)
