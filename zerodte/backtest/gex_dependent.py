# zerodte/backtest/gex_dependent.py
"""Vol-normalized realized-move dependent variable (the clean mechanism ICs bet on).

terminal_abs_em   = |close_eod - entry_spot| / EM     (settlement-relevant)
max_excursion_em  = max(|high - entry_spot|, |low - entry_spot|) over [entry, eod] / EM
                    (touch/stop-relevant — the measure weighted most)
"""
from __future__ import annotations

from datetime import time
from typing import Dict, List, Optional


def _to_min(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def realized_move(bars: List[Dict], entry_anchor: str, em: float,
                  eod_after: time = time(15, 55)) -> Optional[Dict]:
    if em is None or em <= 0:
        return None
    anchor_min = _to_min(entry_anchor)
    entry_bar = next((b for b in bars
                      if b["ts_et"].hour * 60 + b["ts_et"].minute == anchor_min), None)
    if entry_bar is None:
        return None
    entry_spot = float(entry_bar["open"])
    window = [b for b in bars if b["ts_et"].hour * 60 + b["ts_et"].minute >= anchor_min]
    if not window:
        return None
    hi = max(float(b["high"]) for b in window)
    lo = min(float(b["low"]) for b in window)
    eod = next((b for b in reversed(window) if b["ts_et"].time() >= eod_after), window[-1])
    close = float(eod["close"])
    return {
        "entry_spot": entry_spot,
        "terminal_abs_em": abs(close - entry_spot) / em,
        "max_excursion_em": max(abs(hi - entry_spot), abs(lo - entry_spot)) / em,
    }
