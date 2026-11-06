# zerodte/backtest/dc_time_machine/flux_scanner.py
"""Reconstructed 'Flux' scan: rank candidate entries by front/back IV-ratio dynamics."""
from __future__ import annotations

import math
import statistics
from typing import Dict, List


def ratio_zscore(today: float, history: List[float]) -> float:
    """z-score of today's ratio vs its recent history. NaN if history too small."""
    if today != today or len(history) < 2:
        return float("nan")
    mu = statistics.mean(history)
    sd = statistics.pstdev(history)
    if sd == 0:
        return 0.0
    return (today - mu) / sd


def rank_candidates(candidates: List[Dict], by: str = "drop") -> List[Dict]:
    """Rank candidates. by='drop' → largest (ratio_prev − ratio) first (reverting down);
    by='rise' → largest (ratio − ratio_prev) first."""
    def delta(c: Dict) -> float:
        d = c.get("ratio_prev", c.get("ratio", 0.0)) - c.get("ratio", 0.0)
        return d if by == "drop" else -d
    return sorted(candidates, key=delta, reverse=True)
