"""
Compare computed %ITM and %Touch with TastyTrade values.

Formulas:
  %ITM  = probability of expiring ITM = N(d2) for calls, N(-d2) for puts
    where d2 = [ln(S/K) + (r - σ²/2)T] / (σ√T)

  %Touch = probability of touching the strike before expiration
    For upper barrier (K > S, OTM call / ITM put):
      P = Φ(a + b) + exp(c) · Φ(a - b)
      where a = ln(S/K)/(σ√T), b = (r-q-σ²/2)√T/σ, c = 2ln(K/S)(r-q-σ²/2)/σ²

    For lower barrier (K < S, ITM call / OTM put):
      P = Φ(-a + b) + exp(-c) · Φ(-a - b)
      (same variables, signs adjusted for downward barrier)

    Simplification: for short DTE with near-zero drift, P(touch) ≈ 2 × P(ITM)
"""
import numpy as np
from scipy.stats import norm

# --- Data from TT screenshot ---
S = 6416.00         # /ESM6 last price
IV_INDEX = 0.341    # IVx 34.1%
EXPECTED_MOVE = 38.73  # ±38.73 points shown in IVx header
r = 0.04            # risk-free rate (approx)
q = 0.0             # futures have no dividend; cost-of-carry is in the futures price

# Back out time-to-expiration from expected move:
# EM = S × IV × √T  →  T = (EM / (S × IV))²
T = (EXPECTED_MOVE / (S * IV_INDEX)) ** 2
print(f"Underlying: /ESM6 = {S}")
print(f"IVx: {IV_INDEX*100:.1f}%  |  Expected Move: ±{EXPECTED_MOVE}")
print(f"Implied T = {T:.6f} years = {T*365:.3f} days = {T*365*24:.1f} hours")
print()

# TT screenshot data: (strike, call_delta, call_itm%, call_touch%, put_itm%, put_delta)
# Extracted from the visible rows
tt_data = [
    # strike, c_delta, c_itm, c_touch, bid,  ask,   p_bid, p_ask, p_touch, p_itm, p_delta
    (6360, 0.86, 85, 30, 59.50, 60.75,  4.20,  4.35, 30, 15, -0.14),
    (6365, 0.85, 83, 34, 55.25, 56.25,  4.95,  5.10, 34, 17, -0.15),
    (6370, 0.82, 81, 38, 51.00, 52.25,  5.70,  5.90, 38, 19, -0.18),
    (6375, 0.80, 78, 44, 47.00, 48.00,  6.70,  6.80, 44, 22, -0.20),
    (6380, 0.78, 75, 49, 43.25, 44.00,  7.70,  7.90, 49, 25, -0.22),
    (6385, 0.75, 73, 55, 39.50, 40.25,  8.90,  9.10, 55, 27, -0.25),
    (6390, 0.72, 69, 61, 36.00, 36.50, 10.30, 10.50, 61, 31, -0.28),
    (6395, 0.69, 66, 68, 32.50, 33.00, 11.80, 12.00, 68, 34, -0.31),
    (6400, 0.65, 62, 75, 29.00, 29.75, 13.50, 13.70, 75, 38, -0.35),
    (6405, 0.62, 59, 83, 26.00, 26.50, 15.30, 15.60, 83, 41, -0.38),
    (6410, 0.58, 55, 90, 23.00, 23.50, 17.40, 17.70, 90, 45, -0.42),
    (6415, 0.54, 51, 98, 20.25, 20.75, 19.70, 19.90, 98, 49, -0.46),
    (6420, 0.50, 47, 94, 17.90, 18.20, 22.00, 22.50, 94, 53, -0.50),
    (6425, 0.46, 43, 85, 15.60, 15.90, 24.75, 25.25, 85, 57, -0.54),
    (6430, 0.42, 39, 77, 13.50, 13.80, 27.50, 28.00, 77, 61, -0.58),
    (6435, 0.38, 35, 70, 11.70, 11.90, 30.50, 31.25, 70, 65, -0.62),
    (6440, 0.35, 31, 62, 10.00, 10.20, 33.75, 34.50, 62, 69, -0.65),
    (6445, 0.31, 28, 55,  8.50,  8.60, 37.25, 38.00, 55, 72, -0.69),
    (6450, 0.27, 24, 48,  7.20,  7.40, 41.00, 41.75, 48, 76, -0.73),
    (6455, 0.24, 21, 42,  6.00,  6.20, 44.75, 45.75, 42, 79, -0.76),
    (6460, 0.21, 18, 36,  5.10,  5.20, 48.75, 49.75, 36, 82, -0.79),
    (6465, 0.18, 16, 31,  4.25,  4.35, 53.00, 54.00, 31, 84, -0.82),
    (6470, 0.16, 13, 27,  3.55,  3.65, 57.25, 58.25, 27, 87, -0.84),
    (6475, 0.14, 11, 23,  3.00,  3.10, 61.75, 62.75, 23, 89, -0.86),
    (6480, 0.12, 10, 19,  2.50,  2.60, 66.25, 67.25, 19, 90, -0.88),
]


def black_scholes_d(S, K, T, sigma, r=0.0, q=0.0):
    """Return (d1, d2) for Black-Scholes."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def pct_itm_call(S, K, T, sigma, r=0.0, q=0.0):
    """P(S_T > K) under risk-neutral measure = N(d2)."""
    _, d2 = black_scholes_d(S, K, T, sigma, r, q)
    return norm.cdf(d2) * 100


def pct_itm_put(S, K, T, sigma, r=0.0, q=0.0):
    """P(S_T < K) = N(-d2)."""
    _, d2 = black_scholes_d(S, K, T, sigma, r, q)
    return norm.cdf(-d2) * 100


def pct_touch(S, K, T, sigma, r=0.0, q=0.0):
    """
    Probability that S_t touches K at any time in [0, T].

    For GBM: ln(S_t) is BM with drift μ = r - q - σ²/2.

    P(touch K) uses the reflection principle for BM with drift:
      For barrier b = ln(K/S):
        If b > 0 (K > S): P = Φ((−b + μT)/(σ√T)) + exp(2μb/σ²) · Φ((−b − μT)/(σ√T))
        If b < 0 (K < S): P = Φ(( b + μT)/(σ√T)) + exp(2μb/σ²) · Φ(( b − μT)/(σ√T))
    """
    mu = r - q - 0.5 * sigma**2
    sqrtT = np.sqrt(T)
    b = np.log(K / S)  # log-distance to barrier

    if b > 0:  # barrier above (OTM call side)
        term1 = norm.cdf((-b + mu * T) / (sigma * sqrtT))
        term2 = np.exp(2 * mu * b / sigma**2) * norm.cdf((-b - mu * T) / (sigma * sqrtT))
    else:  # barrier below (OTM put side)
        b_abs = -b
        term1 = norm.cdf((-b_abs - mu * T) / (sigma * sqrtT))
        term2 = np.exp(-2 * mu * b_abs / sigma**2) * norm.cdf((-b_abs + mu * T) / (sigma * sqrtT))

    return min((term1 + term2) * 100, 100.0)


# --- Method 1: Flat vol (IVx) ---
sigma = IV_INDEX
print("=" * 90)
print("METHOD 1: Flat vol (IVx = {:.1f}%) across all strikes".format(IV_INDEX * 100))
print("=" * 90)

print(f"\n{'Strike':>6}  │ {'TT ITM%':>7} {'Calc':>5} {'Δ':>5}  │ {'TT Touch%':>9} {'Calc':>5} {'Δ':>5}  │  Side")
print("─" * 75)

itm_errors_1 = []
touch_errors_1 = []

for row in tt_data:
    K = row[0]
    c_delta, c_itm_tt, c_touch_tt = row[1], row[2], row[3]
    p_itm_tt, p_delta = row[9], row[10]

    c_itm_calc = pct_itm_call(S, K, T, sigma, r, q)
    p_itm_calc = pct_itm_put(S, K, T, sigma, r, q)
    touch_calc = pct_touch(S, K, T, sigma, r, q)

    if K >= S:
        itm_err = c_itm_calc - c_itm_tt
        touch_err = touch_calc - c_touch_tt
        print(f"{K:>6.0f}  │ {c_itm_tt:>6}% {c_itm_calc:>5.1f} {itm_err:>+5.1f}  │ {c_touch_tt:>8}% {touch_calc:>5.1f} {touch_err:>+5.1f}  │  OTM Call")
    else:
        itm_err = p_itm_calc - p_itm_tt
        touch_err = touch_calc - c_touch_tt
        print(f"{K:>6.0f}  │ {p_itm_tt:>6}% {p_itm_calc:>5.1f} {itm_err:>+5.1f}  │ {c_touch_tt:>8}% {touch_calc:>5.1f} {touch_err:>+5.1f}  │  OTM Put")

    itm_errors_1.append(abs(itm_err))
    touch_errors_1.append(abs(touch_err))

print("─" * 75)
print(f"MAE:  ITM = {np.mean(itm_errors_1):.1f}pp  |  Touch = {np.mean(touch_errors_1):.1f}pp\n")


# --- Method 2: Back out per-strike IV from option mid-prices ---
from scipy.optimize import brentq


def bs_call_price(S, K, T, sigma, r=0.0, q=0.0):
    d1, d2 = black_scholes_d(S, K, T, sigma, r, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, sigma, r=0.0, q=0.0):
    d1, d2 = black_scholes_d(S, K, T, sigma, r, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(S, K, T, price, r=0.0, q=0.0, is_call=True):
    """Solve for IV given an option price using Brent's method."""
    func = bs_call_price if is_call else bs_put_price
    try:
        iv = brentq(lambda s: func(S, K, T, s, r, q) - price, 0.01, 10.0, xtol=1e-8)
        return iv
    except ValueError:
        return None


print("=" * 90)
print("METHOD 2: Per-strike IV (backed out from option mid-prices)")
print("=" * 90)

print(f"\n{'Strike':>6}  {'IV':>6}  │ {'TT ITM%':>7} {'Calc':>5} {'Δ':>5}  │ {'TT Touch%':>9} {'Calc':>5} {'Δ':>5}  │  Side")
print("─" * 80)

itm_errors_2 = []
touch_errors_2 = []

for row in tt_data:
    K = row[0]
    c_delta, c_itm_tt, c_touch_tt = row[1], row[2], row[3]
    c_bid, c_ask = row[4], row[5]
    p_bid, p_ask = row[6], row[7]
    p_itm_tt, p_delta = row[9], row[10]

    # Use OTM side for IV extraction (more reliable)
    if K >= S:
        mid = (c_bid + c_ask) / 2.0
        iv = implied_vol(S, K, T, mid, r, q, is_call=True)
    else:
        mid = (p_bid + p_ask) / 2.0
        iv = implied_vol(S, K, T, mid, r, q, is_call=False)

    if iv is None:
        print(f"{K:>6.0f}  {'N/A':>6}  │  Could not solve for IV")
        continue

    c_itm_calc = pct_itm_call(S, K, T, iv, r, q)
    p_itm_calc = pct_itm_put(S, K, T, iv, r, q)
    touch_calc = pct_touch(S, K, T, iv, r, q)

    if K >= S:
        itm_err = c_itm_calc - c_itm_tt
        touch_err = touch_calc - c_touch_tt
        print(f"{K:>6.0f}  {iv*100:>5.1f}%  │ {c_itm_tt:>6}% {c_itm_calc:>5.1f} {itm_err:>+5.1f}  │ {c_touch_tt:>8}% {touch_calc:>5.1f} {touch_err:>+5.1f}  │  OTM Call")
    else:
        itm_err = p_itm_calc - p_itm_tt
        touch_err = touch_calc - c_touch_tt
        print(f"{K:>6.0f}  {iv*100:>5.1f}%  │ {p_itm_tt:>6}% {p_itm_calc:>5.1f} {itm_err:>+5.1f}  │ {c_touch_tt:>8}% {touch_calc:>5.1f} {touch_err:>+5.1f}  │  OTM Put")

    itm_errors_2.append(abs(itm_err))
    touch_errors_2.append(abs(touch_err))

print("─" * 80)
print(f"MAE:  ITM = {np.mean(itm_errors_2):.1f}pp  |  Touch = {np.mean(touch_errors_2):.1f}pp")
print()
print("CONCLUSION:")
print("  %ITM  = N(d2) using per-strike implied volatility")
print("  %Touch = reflection principle formula ≈ 2 × %ITM (exact for zero drift)")
print("  TT computes both from per-strike IV, not a flat IVx")
