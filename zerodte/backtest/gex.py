"""Gamma Exposure (GEX) engine for the 0DTE SPXW butterfly study.

Pure logic — operates on an in-memory open-interest dict {(strike, is_call): oi}
plus a per-day spot, ATM-straddle expected move, and time-to-expiry. Gamma is
Black-Scholes (r=0); the dealer convention follows the FatTail/SqueezeMetrics
definition cited in the strategy guide: dealers are LONG gamma from call OI and
SHORT gamma from put OI, so

    GEX(K) = gamma(K) * (OI_call(K) - OI_put(K)) * multiplier * S^2 * 0.01   [$ / 1pt move]

Total GEX sign is the regime signal (positive = suppression/mean-reversion;
negative = amplification/trend). The magnet strike is the largest dealer-gamma
concentration (a price pin). The gamma flip is the spot level where total GEX
crosses zero.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

Oi = Dict[Tuple[float, bool], int]


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes gamma with r=0. T in years, sigma annualized. 0 on bad input."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    vt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * vt * vt) / vt
    return _norm_pdf(d1) / (S * vt)


def iv_from_em(S: float, em: float, T: float) -> float:
    """Annualized vol implied by the ATM straddle EM (Brenner-Subrahmanyam:
    straddle ~= sqrt(2/pi) * S * sigma * sqrt(T))."""
    if S <= 0 or em <= 0 or T <= 0:
        return 0.0
    return em / (math.sqrt(2.0 / math.pi) * S * math.sqrt(T))


def gex_by_strike(oi: Oi, S: float, sigma: float, T: float, mult: int = 100) -> Dict[float, float]:
    """Per-strike dealer GEX in dollars-per-1pt-move."""
    out: Dict[float, float] = {}
    for K in sorted({k for (k, _) in oi}):
        g = bs_gamma(S, K, T, sigma)
        oc = oi.get((K, True), 0)
        op = oi.get((K, False), 0)
        out[K] = g * (oc - op) * mult * S * S * 0.01
    return out


def total_gex(oi: Oi, S: float, sigma: float, T: float, mult: int = 100) -> float:
    """Net dealer GEX at spot S. Sign is the regime (>0 suppression, <0 amplification)."""
    return sum(gex_by_strike(oi, S, sigma, T, mult).values())


def magnet_strike(oi: Oi, S: float, sigma: float, T: float) -> Optional[float]:
    """Strike with the largest dealer-gamma concentration gamma(K)*(OI_call+OI_put) —
    the pin / gravity strike toward which price is pulled."""
    best: Optional[float] = None
    best_val = -1.0
    for K in sorted({k for (k, _) in oi}):
        tot = oi.get((K, True), 0) + oi.get((K, False), 0)
        val = bs_gamma(S, K, T, sigma) * tot
        if val > best_val:
            best_val, best = val, K
    return best


def gamma_flip(oi: Oi, S: float, sigma: float, T: float, mult: int = 100) -> Optional[float]:
    """Spot level at which total GEX crosses zero, scanning the strike grid. Returns the
    first level where the sign of total_gex(level) flips, else None."""
    strikes = sorted({k for (k, _) in oi})
    if not strikes:
        return None
    prev_level = strikes[0]
    prev = total_gex(oi, prev_level, sigma, T, mult)
    for level in strikes[1:]:
        cur = total_gex(oi, level, sigma, T, mult)
        if (prev < 0 <= cur) or (prev > 0 >= cur):
            return level
        prev, prev_level = cur, level
    return None
