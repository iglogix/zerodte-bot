# src_v2/backtest/gex_regime.py
"""Multi-expiry GEX regime features built on src_v2/backtest/gex.py primitives.

A multi-expiry open-interest map is keyed by (expiry, strike, is_call):
    MultiOi = Dict[Tuple[date, float, bool], int]
Each expiry is evaluated with its own time-to-expiry (kernel widens with T) but a
single, fixed implied vol derived from the 0DTE ATM-straddle EM (flat-vol modeling
choice: the regime SIGN and FLIP location are dominated by the OI distribution, not
the absolute vol level; vol sensitivity is checked in the convergence test).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from src_v2.backtest.gex import bs_gamma, total_gex, gamma_flip, magnet_strike

MultiOi = Dict[Tuple[date, float, bool], int]
Oi = Dict[Tuple[float, bool], int]

TRADING_DAYS_PER_YEAR = 252.0
TRADING_HOURS = 6.5


def expiries(moi: MultiOi) -> List[date]:
    return sorted({e for (e, _, _) in moi})


def slice_expiry(moi: MultiOi, expiry: date) -> Oi:
    return {(k, c): v for (e, k, c), v in moi.items() if e == expiry}


def intraday_trading_fraction(entry_anchor: str) -> float:
    """Remaining fraction of the trading day from entry to 16:00 ET, in trading days.
    Mirrors scripts/analytical_butterfly_eval.py: hours = max(.., 0.25); frac = hours/6.5."""
    mins_to_close = 16 * 60 - (int(entry_anchor[:2]) * 60 + int(entry_anchor[3:5]))
    hours = max(mins_to_close / 60.0, 0.25)
    return hours / TRADING_HOURS


def years_to_expiry(entry_anchor: str, sessions_after_today: int) -> float:
    """sessions_after_today = trading sessions strictly after the entry day up to and
    including the expiry session (0 for 0DTE, 1 for the next-session expiry)."""
    return (intraday_trading_fraction(entry_anchor) + sessions_after_today) / TRADING_DAYS_PER_YEAR


@dataclass
class RegimeFeatures:
    total_gex: float
    sign: int
    flip: Optional[float]
    dist_flip_em: Optional[float]      # (S - flip) / EM  (primary regime variable)
    magnet: Optional[float]
    dist_magnet_em: Optional[float]    # (S - magnet) / EM


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def total_gex_multi(moi: MultiOi, S: float, sigma: float,
                    t_by_expiry: Dict[date, float], mult: int = 100) -> float:
    return sum(total_gex(slice_expiry(moi, e), S, sigma, t_by_expiry[e], mult)
               for e in expiries(moi))


def gamma_flip_multi(moi: MultiOi, S: float, sigma: float,
                     t_by_expiry: Dict[date, float], mult: int = 100) -> Optional[float]:
    strikes = sorted({k for (_, k, _) in moi})
    if not strikes:
        return None
    prev = total_gex_multi(moi, strikes[0], sigma, t_by_expiry, mult)
    for lvl in strikes[1:]:
        cur = total_gex_multi(moi, lvl, sigma, t_by_expiry, mult)
        if (prev < 0 <= cur) or (prev > 0 >= cur):
            return lvl
        prev = cur
    return None


def magnet_strike_multi(moi: MultiOi, S: float, sigma: float,
                        t_by_expiry: Dict[date, float]) -> Optional[float]:
    best: Optional[float] = None
    best_val = -1.0
    for K in sorted({k for (_, k, _) in moi}):
        val = 0.0
        for e in expiries(moi):
            oc = moi.get((e, K, True), 0)
            op = moi.get((e, K, False), 0)
            val += bs_gamma(S, K, t_by_expiry[e], sigma) * (oc + op)
        if val > best_val:
            best_val, best = val, K
    return best


def _features(tg: float, flip: Optional[float], magnet: Optional[float],
              S: float, em: float) -> RegimeFeatures:
    d_flip = (S - flip) / em if (flip is not None and em > 0) else None
    d_mag = (S - magnet) / em if (magnet is not None and em > 0) else None
    return RegimeFeatures(tg, _sign(tg), flip, d_flip, magnet, d_mag)


def standing_regime(moi: MultiOi, S: float, em: float, sigma: float,
                    t_by_expiry: Dict[date, float], mult: int = 100) -> RegimeFeatures:
    tg = total_gex_multi(moi, S, sigma, t_by_expiry, mult)
    flip = gamma_flip_multi(moi, S, sigma, t_by_expiry, mult)
    magnet = magnet_strike_multi(moi, S, sigma, t_by_expiry)
    return _features(tg, flip, magnet, S, em)


def proxy_regime(moi: MultiOi, expiry_1dte: date, S: float, em: float, sigma: float,
                 entry_anchor: str, mult: int = 100) -> RegimeFeatures:
    """1DTE OI distribution evaluated at the 0DTE time-to-expiry kernel."""
    oi = slice_expiry(moi, expiry_1dte)
    t0 = years_to_expiry(entry_anchor, 0)
    tg = total_gex(oi, S, sigma, t0, mult)
    flip = gamma_flip(oi, S, sigma, t0, mult)
    magnet = magnet_strike(oi, S, sigma, t0)
    return _features(tg, flip, magnet, S, em)
