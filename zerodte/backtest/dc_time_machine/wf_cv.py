# zerodte/backtest/dc_time_machine/wf_cv.py
"""Dedicated walk-forward CV for the DC simulator.

Mirrors the project methodology (ratio 5:1, 4 folds, annualized Sharpe) but calls
run_session directly instead of the 0DTE-bound WalkForwardOptimizer. Grid search on
train, evaluate the top combo out-of-sample on val; winner = most val-robust combo.
"""
from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Tuple

from zerodte.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from zerodte.backtest.dc_time_machine.simulator import run_session
from zerodte.backtest.dc_time_machine.strategy import DcParams

TRADING_DAYS = 252


def split_folds(days: List[date], ratio: int = 5, n_folds: int = 4
                ) -> List[Tuple[List[date], List[date]]]:
    """Contiguous, non-overlapping folds; each train ≈ ratio× its val, train before val."""
    days = sorted(days)
    block = len(days) // n_folds
    folds: List[Tuple[List[date], List[date]]] = []
    for i in range(n_folds):
        seg = days[i * block: (i + 1) * block] if i < n_folds - 1 else days[i * block:]
        if len(seg) < ratio + 1:
            continue
        val_len = max(1, len(seg) // (ratio + 1))
        train, val = seg[:-val_len], seg[-val_len:]
        if train and val:
            folds.append((train, val))
    return folds


def annualized_sharpe(pnls: List[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    sd = statistics.pstdev(pnls)
    if sd == 0:
        return 0.0
    return (statistics.mean(pnls) / sd) * math.sqrt(TRADING_DAYS)


def _eval(store: MultiExpiryStore, params: DcParams, days: List[date]) -> Dict[str, float]:
    pnls: List[float] = []
    for d in days:
        res = run_session(store, params, d)
        if res is not None and res.outcome != "no_entry":
            pnls.append(res.pnl)
    return {"sharpe": annualized_sharpe(pnls), "pnl": sum(pnls), "n": len(pnls)}


def _combos(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


@dataclass
class WfFold:
    train_sharpe: float
    val_sharpe: float
    val_pnl: float
    best: Dict[str, Any]


@dataclass
class WfReport:
    folds: List[WfFold] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    mean_val_sharpe: float = 0.0
    n_profitable: int = 0


def run_wf_cv(store: MultiExpiryStore, days: List[date], grid: Dict[str, List[Any]],
              base: Dict[str, Any] | None = None, ratio: int = 5, n_folds: int = 4) -> WfReport:
    base = base or {}
    combos = _combos(grid)
    folds = split_folds(days, ratio, n_folds)
    rep = WfReport()
    winners: Dict[Tuple, int] = {}
    val_sharpes: List[float] = []
    for train, val in folds:
        scored = [(c, _eval(store, DcParams(**{**base, **c}), train)["sharpe"]) for c in combos]
        best_combo = max(scored, key=lambda x: x[1])[0]
        vm = _eval(store, DcParams(**{**base, **best_combo}), val)
        rep.folds.append(WfFold(train_sharpe=max(s for _, s in scored),
                                val_sharpe=vm["sharpe"], val_pnl=vm["pnl"], best=best_combo))
        winners[tuple(sorted(best_combo.items()))] = winners.get(tuple(sorted(best_combo.items())), 0) + 1
        val_sharpes.append(vm["sharpe"])
        if vm["pnl"] > 0:
            rep.n_profitable += 1
    if rep.folds:
        rep.best_params = dict(max(winners.items(), key=lambda x: x[1])[0])
        rep.mean_val_sharpe = statistics.mean(val_sharpes)
    return rep
