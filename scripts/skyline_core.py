#!/usr/bin/env python3
"""Skyline (4-butterfly SPX 0DTE) — structure definition and payoff.

Parsed from the video interview with Saurabh Khurana (setupsprofits/John).
Example anchor from the video: SPX ~7580 at entry.

  BF1  calls   +1 A+0    -2 A+25   +1 A+50     (25-wide balanced)
  BF2  calls   +1 A+40   -2 A+60   +1 A+75     (20/15 broken wing, 5pt "floor")
  BF3  puts    +1 A-45   -2 A-20   +1 A+5      (25-wide balanced)
  BF4  puts    +1 A-80   -2 A-60   +1 A-35     (25/20 broken wing, 5pt "floor")

A = ATM strike (spot rounded to nearest 5).  16 contracts, 12 distinct strikes.
Video-stated economics for 1 lot: debit 1255, max loss 755, max profit 1245.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

MULT = 100.0

# (offset_from_A, qty, is_call)
LEGS: List[Tuple[int, int, bool]] = [
    # BF1 call fly
    (0,   +1, True), (25,  -2, True), (50,  +1, True),
    # BF2 call BWB
    (40,  +1, True), (60,  -2, True), (75,  +1, True),
    # BF3 put fly
    (-45, +1, False), (-20, -2, False), (5,  +1, False),
    # BF4 put BWB
    (-80, +1, False), (-60, -2, False), (-35, +1, False),
]

N_CONTRACTS = sum(abs(q) for _, q, _ in LEGS)  # 16


def scaled_legs(scale: float = 1.0, step: int = 5) -> List[Tuple[int, int, bool]]:
    """Offsets scaled by `scale`, snapped to the `step`-point strike grid.

    scale=1.0 reproduces the video's literal fixed-point structure.
    """
    if scale == 1.0:
        return list(LEGS)
    out = []
    for off, qty, is_call in LEGS:
        k = int(round(off * scale / step) * step)
        out.append((k, qty, is_call))
    return out


def intrinsic(settle: float, atm: float, off: int, is_call: bool) -> float:
    k = atm + off
    return max(0.0, settle - k) if is_call else max(0.0, k - settle)


def payoff_at(settle: float, atm: float, legs=None) -> float:
    """Terminal value ($, one lot) of the structure at cash settlement."""
    legs = LEGS if legs is None else legs
    return MULT * sum(q * intrinsic(settle, atm, off, c) for off, q, c in legs)


@dataclass
class Fill:
    debit: float          # $ paid to open, one lot (positive = debit)
    leg_prices: dict      # (off, is_call) -> price paid/received per contract ($)
    max_spread_pts: float
    total_spread_pts: float


def price_structure(quotes: dict, atm: float, fill_frac: float,
                    legs=None) -> Fill | None:
    """Price one lot from real bid/ask.

    quotes: {(off, is_call): (bid, ask)} in index points.
    fill_frac: 0.0 = mid, 0.5 = full half-spread (buy ask / sell bid),
               1.0 = cross the spread by a full width.
    Returns None if any leg is unquotable.
    """
    legs = LEGS if legs is None else legs
    debit = 0.0
    lp = {}
    spreads = []
    for off, qty, is_call in legs:
        q = quotes.get((off, is_call))
        if q is None:
            return None
        bid, ask = q
        if ask <= 0 or bid < 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        sp = ask - bid
        spreads.append(sp)
        # buying (qty>0) pays up; selling (qty<0) receives less
        px = mid + (sp * fill_frac if qty > 0 else -sp * fill_frac)
        px = max(px, 0.0)
        debit += qty * px * MULT
        lp[(off, is_call)] = px
    return Fill(debit=debit, leg_prices=lp,
                max_spread_pts=max(spreads), total_spread_pts=sum(spreads))
