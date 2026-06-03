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


# --- Task 3: Direction filter (Hull MA), coin-flip, regime width ---

def wma(values: List[float], n: int) -> Optional[float]:
    """Linearly-weighted moving average of the last n values (weight = recency)."""
    if len(values) < n or n <= 0:
        return None
    window = values[-n:]
    denom = n * (n + 1) / 2
    return sum((i + 1) * window[i] for i in range(n)) / denom


def _hma_series(values: List[float], n: int) -> List[float]:
    """Hull MA series: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half, sq = max(1, n // 2), max(1, int(round(math.sqrt(n))))
    raw: List[float] = []
    for i in range(len(values)):
        w_half = wma(values[: i + 1], half)
        w_full = wma(values[: i + 1], n)
        if w_half is None or w_full is None:
            raw.append(float("nan"))
        else:
            raw.append(2 * w_half - w_full)
    out: List[float] = []
    for i in range(len(raw)):
        valid = [x for x in raw[: i + 1] if not math.isnan(x)]
        h = wma(valid, sq) if len(valid) >= sq else None
        out.append(h if h is not None else float("nan"))
    return out


def hull_trend(daily_closes: List[float], n: int = 21) -> int:
    """Sign of the Hull-MA slope using only the supplied (pre-entry) closes.

    Returns +1 (up), -1 (down), or 0 (flat / insufficient history).
    Requires at least ~n + round(sqrt(n)) closes (≈26 for n=21) before it can
    return a non-zero trend; until then it returns 0, and hull/anti callers fall
    to their flat-default side.
    """
    hma = _hma_series(daily_closes, n)
    valid = [x for x in hma if not math.isnan(x)]
    if len(valid) < 2:
        return 0
    slope = valid[-1] - valid[-2]
    return 1 if slope > 0 else (-1 if slope < 0 else 0)


def coin_flip_side(d: date) -> str:
    """Deterministic ~50/50 baseline: even ordinal -> call, odd -> put."""
    return "call" if d.toordinal() % 2 == 0 else "put"


def resolve_side(mode: str, d: date, closes_before: List[float], n: int = 21) -> str:
    """Map a direction MODE to a side. closes_before excludes the entry day."""
    if mode == "call":
        return "call"
    if mode == "put":
        return "put"
    if mode == "coin":
        return coin_flip_side(d)
    if mode not in ("hull", "anti"):
        raise ValueError(f"Unknown direction mode: {mode!r}")
    t = hull_trend(closes_before, n)
    if mode == "anti":
        return "put" if t >= 0 else "call"
    return "call" if t >= 0 else "put"   # "hull"


def regime_width(vix: Optional[float]) -> float:
    """Representative wing width per VIX regime (deck bands: <17 / 17-32 / 32+)."""
    if vix is None:
        return 40.0
    if vix < 17:
        return 25.0
    if vix < 32:
        return 40.0
    return 55.0


# --- Task 4: Per-day evaluation with three exit models ---

def _to_min(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def trail_giveback(minute: str) -> float:
    """Time-of-day trailing giveback (fraction of peak profit allowed back).

    Published rule: wide AM (50-75% -> 0.625), tighter midday (30-40% -> 0.35),
    tight PM (30% -> 0.30).
    """
    m = _to_min(minute)
    if m < _to_min("12:00"):
        return 0.625
    if m < _to_min("14:00"):
        return 0.35
    return 0.30


def evaluate_day(quotes, cfg: ButterflyConfig,
                 spot_close: Optional[float] = None) -> Dict:
    """Evaluate one trading day. Returns {'skip': reason} or a trade-result dict."""
    ea = cfg.entry_anchor
    atm = atm_from_chain(quotes, ea)
    if atm is None:
        return {"skip": "no_atm"}
    em = compute_em(quotes, ea, atm)
    if em is None or em <= 0:
        return {"skip": "no_em"}
    fwd = forward_from_chain(quotes, ea, atm)
    if fwd is None:
        return {"skip": "no_forward"}

    strikes = select_fly_strikes(fwd, em, cfg)
    is_call = cfg.side == "call"
    debit = fly_value(quotes, ea, strikes, is_call)
    if debit is None or debit <= 0:
        return {"skip": "no_debit"}
    if not passes_gates(debit, cfg):
        return {"skip": "gate"}

    kind: Optional[str] = None
    exit_anchor: Optional[str] = None
    exit_mark: Optional[float] = None
    peak_profit = 0.0
    eod = _to_min(cfg.eod_anchor)

    if cfg.exit_model in ("target", "trail"):
        for a in cfg.anchors:
            if a == ea or _to_min(a) > eod:
                continue
            m = fly_mark(quotes, a, strikes, is_call)
            if m is None:
                continue
            profit = m - debit
            if cfg.exit_model == "target":
                if profit >= cfg.target_mult * debit:
                    kind, exit_anchor, exit_mark = "target", a, m
                    break
            else:  # trail
                peak_profit = max(peak_profit, profit)
                if (peak_profit >= cfg.trail_arm_pct * debit
                        and profit <= peak_profit * (1 - trail_giveback(a))):
                    kind, exit_anchor, exit_mark = "trail", a, m
                    break

    if kind is None:   # hold model, or no mid-exit trigger fired -> settle at expiry
        S = settlement_forward(quotes, cfg.anchors)
        if S is None:
            S = spot_close
        if S is None:
            return {"skip": "no_settle"}
        kind, exit_anchor, exit_mark = "expiry", cfg.anchors[-1], fly_intrinsic(strikes, is_call, S)

    mult = cfg.multiplier
    gross = (exit_mark - debit) * mult
    n_legs = 4   # 1 + 2 + 1 contracts
    entry_slip = cfg.slippage_per_leg * n_legs
    exit_slip = 0.0 if kind == "expiry" else cfg.slippage_per_leg * n_legs
    n_comm = n_legs if kind == "expiry" else n_legs * 2
    cost = (entry_slip + exit_slip) * mult + cfg.commission_per_leg * n_comm
    net = gross - cost

    return {
        "skip": None, "kind": kind, "side": cfg.side, "exit_anchor": exit_anchor,
        "debit": debit, "exit_mark": exit_mark, "width": cfg.width,
        "gross": gross, "pnl": net, "strikes": strikes, "em": em, "forward": fwd,
        "r2r_theo": (cfg.width - debit) / debit if debit > 0 else 0.0,
    }
