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
