"""Pure-function ORB signal detector for offline replay.

Replicates the deterministic core of zerodte/agent/orb_engine.py without async
or broker coupling, so we can enumerate signal events from years of historical
ES bars in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Literal, Optional, Tuple

Bias = Literal["long", "short"]
RetestMode = Literal["midline", "breakout_level", "opposite_level"]


@dataclass(frozen=True)
class SignalEvent:
    session_date: date
    bias: Bias
    signal_minute_et: str  # "HH:MM"
    es_price_at_signal: float
    orb_high: float
    orb_low: float


def _hhmm_to_min(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:5])


def _bar_min(bar) -> int:
    t = bar["ts_et"]
    return t.hour * 60 + t.minute


def compute_orb_levels(
    bars: List[dict], orb_start: str, orb_end: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (low, midline, high) from bars whose minute is in [orb_start, orb_end)."""
    s, e = _hhmm_to_min(orb_start), _hhmm_to_min(orb_end)
    in_window = [b for b in bars if s <= _bar_min(b) < e]
    if not in_window:
        return (None, None, None)
    hi = max(b["high"] for b in in_window)
    lo = min(b["low"] for b in in_window)
    return (lo, (hi + lo) / 2, hi)


def detect_signals_for_day(
    bars: List[dict],
    session_date: date,
    orb_start: str,
    orb_end: str,
    market_open: str,
    eod: str,
    retest_mode: RetestMode,
    midline_tol_pts: float,
    min_range_pts: float,
    max_range_pts: float,
) -> List[SignalEvent]:
    """Detect at most one signal per day. Returns [] if no breakout or filter rejects."""
    lo, mid, hi = compute_orb_levels(bars, orb_start, orb_end)
    if hi is None:
        return []

    rng = hi - lo
    if rng < min_range_pts or rng > max_range_pts:
        return []

    open_min = _hhmm_to_min(market_open)
    eod_min = _hhmm_to_min(eod)
    post_open = [b for b in bars if open_min <= _bar_min(b) <= eod_min]

    # Phase 1: find first breakout
    bias: Optional[Bias] = None
    breakout_idx: Optional[int] = None
    for i, b in enumerate(post_open):
        if b["high"] > hi:
            bias = "long"
            breakout_idx = i
            break
        if b["low"] < lo:
            bias = "short"
            breakout_idx = i
            break

    if bias is None:
        return []

    # In opposite_level mode the entry inverts the breakout bias
    entry_bias: Bias = bias
    if retest_mode == "opposite_level":
        entry_bias = "short" if bias == "long" else "long"

    # Phase 2: pick the retest target
    if retest_mode == "midline":
        target = mid
    elif retest_mode == "breakout_level":
        target = hi if bias == "long" else lo
    elif retest_mode == "opposite_level":
        target = lo if bias == "long" else hi
    else:
        raise ValueError(f"unknown retest_mode: {retest_mode}")

    # Phase 3: scan for retest after breakout
    for b in post_open[breakout_idx + 1:]:
        # Touch within tolerance counts as retest
        if b["low"] - midline_tol_pts <= target <= b["high"] + midline_tol_pts:
            t = b["ts_et"]
            return [SignalEvent(
                session_date=session_date,
                bias=entry_bias,
                signal_minute_et=f"{t.hour:02d}:{t.minute:02d}",
                es_price_at_signal=float(target),
                orb_high=float(hi),
                orb_low=float(lo),
            )]

    return []
