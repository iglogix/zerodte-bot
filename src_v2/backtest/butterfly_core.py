# src_v2/backtest/butterfly_core.py
"""Pure logic for the 0DTE SPX OTM butterfly backtest.

No I/O — operates on the in-memory quote dict
{(minute_et, strike, is_call): (bid, ask)} so it is fully unit-testable.

Reuses JL chain helpers (quote_mid / atm_from_chain / compute_em /
forward_from_chain / settlement_forward). A 1-2-1 long debit butterfly is a
DEBIT structure: profit = exit_value - debit_paid (opposite sign from JL's
credit structures). Direction (call/put side) is set by a 21-period Hull MA
trend; width is set by a VIX regime map. See the design spec for the rule set.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from src_v2.backtest.jl_core import (
    make_anchors, quote_mid, atm_from_chain, compute_em, forward_from_chain,
    settlement_forward, _max_consec_losses, _max_drawdown,
)

Strikes = Tuple[float, float, float]   # (k_low, k_body, k_high)


def round_to_step(x: float, step: float) -> float:
    return round(x / step) * step


@dataclass
class ButterflyConfig:
    entry_anchor: str = "09:30"
    anchors: List[str] = field(default_factory=lambda: make_anchors("09:30", "16:00", 10))
    side: str = "call"            # "call" or "put" (set per-day by the direction filter)
    body_em_mult: float = 1.0     # body placed body_em_mult * EM OTM from the forward
    width: float = 30.0           # symmetric wing width (points)
    step: float = 5.0
    multiplier: int = 100
    # entry gates
    gates_on: bool = True
    max_debit_pct: float = 0.10   # debit <= max_debit_pct * width
    min_r2r: float = 9.0          # (width - debit) / debit >= min_r2r
    # exit model: "hold" | "target" | "trail"
    exit_model: str = "hold"
    target_mult: float = 2.0      # target: exit when profit >= target_mult * debit
    trail_arm_pct: float = 0.50   # trail: arm once profit >= trail_arm_pct * debit
    eod_anchor: str = "15:50"     # no mid-exit trigger fires after this anchor
    # costs (headline = 0)
    slippage_per_leg: float = 0.0
    commission_per_leg: float = 0.0


def select_fly_strikes(forward: float, em: float, cfg: ButterflyConfig) -> Strikes:
    sign = 1.0 if cfg.side == "call" else -1.0
    k_body = round_to_step(forward + sign * cfg.body_em_mult * em, cfg.step)
    return (k_body - cfg.width, k_body, k_body + cfg.width)


def fly_intrinsic(strikes: Strikes, is_call: bool, S: float) -> float:
    k_low, k_body, k_high = strikes
    if is_call:
        return (max(S - k_low, 0.0) - 2 * max(S - k_body, 0.0) + max(S - k_high, 0.0))
    return (max(k_high - S, 0.0) - 2 * max(k_body - S, 0.0) + max(k_low - S, 0.0))


def fly_value(quotes, minute: str, strikes: Strikes, is_call: bool,
              fallback_min: int = 5) -> Optional[float]:
    """Raw combined value of the 1-2-1 fly at `minute` (long low + long high - 2*body)."""
    k_low, k_body, k_high = strikes
    lo = quote_mid(quotes, minute, k_low, is_call, fallback_min)
    bo = quote_mid(quotes, minute, k_body, is_call, fallback_min)
    hi = quote_mid(quotes, minute, k_high, is_call, fallback_min)
    if lo is None or bo is None or hi is None:
        return None
    return lo + hi - 2 * bo


def fly_mark(quotes, minute: str, strikes: Strikes, is_call: bool,
             fallback_min: int = 5) -> Optional[float]:
    """Closeable mark = fly_value clamped to [0, +inf) (a long fly is never worth < 0)."""
    v = fly_value(quotes, minute, strikes, is_call, fallback_min)
    return None if v is None else max(v, 0.0)


def passes_gates(debit: float, cfg: ButterflyConfig) -> bool:
    if not cfg.gates_on:
        return True
    if debit <= 0:
        return False
    r2r = (cfg.width - debit) / debit
    return debit <= cfg.max_debit_pct * cfg.width and r2r >= cfg.min_r2r
