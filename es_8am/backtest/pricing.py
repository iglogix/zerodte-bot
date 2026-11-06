"""
Pure Black-Scholes / Bachelier pricing functions.

All functions are stateless (no config dependencies) and can be imported
independently by any strategy or utility.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

def ncdf(x: float) -> float:
    """Standard normal CDF — Abramowitz & Stegun approximation."""
    a = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * a)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (
        1.781477937 + t * (-1.821255978 + t * 1.330274429)
    )))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
    return cdf if x >= 0 else 1.0 - cdf


def npdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Black-Scholes (log-normal model)
# ---------------------------------------------------------------------------

def bs_call(s: float, k: float, t: float, iv: float) -> float:
    """Black-Scholes call price (r=0, no dividends)."""
    if t <= 0:
        return max(0.0, s - k)
    sv = iv * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / sv
    d2 = d1 - sv
    return s * ncdf(d1) - k * ncdf(d2)


def bs_put(s: float, k: float, t: float, iv: float) -> float:
    """Black-Scholes put price (r=0)."""
    if t <= 0:
        return max(0.0, k - s)
    sv = iv * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / sv
    d2 = d1 - sv
    return k * ncdf(-d2) - s * ncdf(-d1)


def bs_call_delta(s: float, k: float, t: float, iv: float) -> float:
    if t <= 0:
        return 1.0 if s > k else 0.0
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * math.sqrt(t))
    return ncdf(d1)


def bs_put_delta(s: float, k: float, t: float, iv: float) -> float:
    return bs_call_delta(s, k, t, iv) - 1.0


# ---------------------------------------------------------------------------
# Bachelier (normal model — better for 0DTE near expiry)
# ---------------------------------------------------------------------------

def bachelier_call(s: float, k: float, t: float, iv: float) -> float:
    """Bachelier (normal model) call price."""
    if t <= 0:
        return max(0.0, s - k)
    sigma_sqrt_t = iv * s * math.sqrt(t)  # absolute vol = relative iv * spot
    if sigma_sqrt_t < 1e-10:
        return max(0.0, s - k)
    d = (s - k) / sigma_sqrt_t
    return sigma_sqrt_t * (npdf(d) + d * ncdf(d))


def bachelier_put(s: float, k: float, t: float, iv: float) -> float:
    """Bachelier (normal model) put price."""
    if t <= 0:
        return max(0.0, k - s)
    sigma_sqrt_t = iv * s * math.sqrt(t)
    if sigma_sqrt_t < 1e-10:
        return max(0.0, k - s)
    d = (s - k) / sigma_sqrt_t
    return sigma_sqrt_t * (npdf(-d) + (-d) * ncdf(-d))


def bachelier_call_delta(s: float, k: float, t: float, iv: float) -> float:
    if t <= 0:
        return 1.0 if s > k else 0.0
    sigma_sqrt_t = iv * s * math.sqrt(t)
    if sigma_sqrt_t < 1e-10:
        return 1.0 if s > k else 0.0
    d = (s - k) / sigma_sqrt_t
    return ncdf(d)


def bachelier_put_delta(s: float, k: float, t: float, iv: float) -> float:
    return bachelier_call_delta(s, k, t, iv) - 1.0


# ---------------------------------------------------------------------------
# Spread helpers
# ---------------------------------------------------------------------------

def select_strike(
    s: float, target_delta: float, iv: float, t: float,
    spread_type: str, step: float = 5.0, model: str = "bs",
) -> float:
    """Find nearest strike (at step increments) where |delta| ≈ target_delta."""
    if model == "bachelier":
        _put_delta, _call_delta = bachelier_put_delta, bachelier_call_delta
    else:
        _put_delta, _call_delta = bs_put_delta, bs_call_delta

    best_k, best_diff = s, 1.0
    if spread_type == "put_credit_spread":
        k = math.floor(s / step) * step
        while k > s - 200:
            d = abs(_put_delta(s, k, t, iv))
            diff = abs(d - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_k = k
            if d < target_delta * 0.3:
                break
            k -= step
    else:
        k = math.ceil(s / step) * step
        while k < s + 200:
            d = _call_delta(s, k, t, iv)
            diff = abs(d - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_k = k
            if d < target_delta * 0.3:
                break
            k += step
    return best_k


def spread_credit(
    s: float, short_k: float, long_k: float, t: float, iv: float, spread_type: str,
    model: str = "bs",
) -> float:
    """Net credit = short option mid - long option mid."""
    if model == "bachelier":
        _put, _call = bachelier_put, bachelier_call
    else:
        _put, _call = bs_put, bs_call
    if spread_type == "put_credit_spread":
        return _put(s, short_k, t, iv) - _put(s, long_k, t, iv)
    else:
        return _call(s, short_k, t, iv) - _call(s, long_k, t, iv)


def spread_value(
    s: float, short_k: float, long_k: float, t: float, iv: float, spread_type: str,
    model: str = "bs",
) -> float:
    """Current mark-to-market value of the spread (cost to close)."""
    return max(0.0, spread_credit(s, short_k, long_k, t, iv, spread_type, model=model))
