"""Compute IV/delta from option bid/ask quotes via Black–Scholes + a parity spot.

Shared by the historical (marketdata.app EOD, greek-less) and live (TastyTrade DXLink
quotes) sources so the IV definition is identical everywhere. `price_rows` is a list of
(strike, is_call, bid, ask) tuples.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from src_v2.flux.models import OptionRow
from src_v2.backtest.dc_time_machine.iv import bs_delta, implied_vol

RATE = 0.04
PriceRow = Tuple[float, bool, float, float]


def parity_spot(price_rows: List[PriceRow]) -> Optional[float]:
    """Spot via put-call parity at the strike with the smallest |call_mid − put_mid|."""
    by_strike: dict = {}
    for k, is_call, b, a in price_rows:
        by_strike.setdefault(k, {})[is_call] = (b + a) / 2.0
    best = None
    for k, cp in by_strike.items():
        if True in cp and False in cp:
            diff = abs(cp[True] - cp[False])
            if best is None or diff < best[0]:
                best = (diff, k + cp[True] - cp[False])
    return best[1] if best else None


def compute_iv_rows(price_rows: List[PriceRow], T: float, spot: Optional[float] = None,
                    rate: float = RATE) -> Tuple[List[OptionRow], Optional[float]]:
    """Return (OptionRows with BS-computed iv/delta, spot). Drops rows whose mid ≤ 0 or
    ≤ intrinsic (IV un-invertible)."""
    if spot is None:
        spot = parity_spot(price_rows)
    out: List[OptionRow] = []
    if spot is None or T <= 0:
        return out, spot
    for k, is_call, b, a in price_rows:
        mid = (b + a) / 2.0
        if mid <= 0:
            continue
        iv = implied_vol(mid, spot, k, T, rate, is_call)
        if iv != iv:                      # NaN → mid ≤ intrinsic
            continue
        out.append(OptionRow(k, is_call, b, a, float(iv), bs_delta(spot, k, T, float(iv), rate, is_call)))
    return out, spot
