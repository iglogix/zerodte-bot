# src_v2/backtest/dc_time_machine/iv.py
"""Black–Scholes pricing, implied-vol inversion, and the front/back IV ratio (Flux)."""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def _d1(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_price(S: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = _d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if is_call:
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, T, sigma, r)
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def implied_vol(price: float, S: float, K: float, T: float, r: float, is_call: bool) -> float:
    """Invert BS for sigma via brentq. Returns NaN if price ≤ intrinsic or no root."""
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if T <= 0 or price <= intrinsic + 1e-6:
        return float("nan")
    try:
        return brentq(lambda s: bs_price(S, K, T, s, r, is_call) - price, 1e-4, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return float("nan")


def front_back_ratio(front_iv: float, back_iv: float) -> float:
    if back_iv is None or back_iv <= 0 or front_iv != front_iv:
        return float("nan")
    return front_iv / back_iv
