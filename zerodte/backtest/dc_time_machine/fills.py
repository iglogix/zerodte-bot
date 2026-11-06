"""Parameterized fill model. Every fill is mid ± k·half_spread (k=0 mid, k=1 cross)."""
from __future__ import annotations

from typing import List, Tuple


def fill_price(bid: float, ask: float, side: str, k: float) -> float:
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    if side == "buy":
        return mid + k * half
    if side == "sell":
        return mid - k * half
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def net_combo_credit(legs: List[Tuple[float, float, str]], k: float) -> float:
    """Net credit ($/share) for an atomic combo: +sell mids, −buy mids, less k·Σhalf_spread.

    legs: list of (bid, ask, side). Adverse fill ALWAYS reduces the received credit,
    so a single k·Σhalf term is subtracted regardless of per-leg side.
    """
    net = 0.0
    total_half = 0.0
    for bid, ask, side in legs:
        mid = (bid + ask) / 2.0
        total_half += (ask - bid) / 2.0
        if side == "sell":
            net += mid
        elif side == "buy":
            net -= mid
        else:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return net - k * total_half
