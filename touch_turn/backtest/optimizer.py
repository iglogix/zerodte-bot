"""
Touch & Turn Scalper — Grid Search Optimizer with Walk-Forward CV.

Sweeps strategy parameters and ranks combos by composite score
(Sharpe 50% + ProfitFactor 30% + WinRate 20%), with drawdown penalties.

Usage:
  python -m touch_turn.optimizer
  python -m touch_turn.optimizer --config touch_turn/config_nvda.yaml
"""

from __future__ import annotations

import ast
import copy
import csv
import itertools
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

from touch_turn.backtest.engine import TouchTurnBacktestEngine, BacktestReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameter grid
# ---------------------------------------------------------------------------

DEFAULT_GRID: Dict[str, list] = {
    "fib_tp_long":   [0.236, 0.382, 0.500, 0.618],
    "fib_tp_short":  [0.236, 0.382, 0.500, 0.618],
    "atr_min_pct":   [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "sl_ratio":      [0.25, 0.50, 0.75, 1.0],   # SL = TP_dist * sl_ratio (0.5 = 2:1 R:R)
    "window_mins":   [60, 90, 120, 180],          # trading window in minutes after open
}


def _apply_combo(base_config: dict, combo: dict) -> dict:
    """Apply a parameter combination to the base config."""
    cfg = copy.deepcopy(base_config)
    strat = cfg.setdefault("strategy", {})

    strat["fib_tp_long"] = combo["fib_tp_long"]
    strat["fib_tp_short"] = combo["fib_tp_short"]
    strat["atr_min_pct"] = combo["atr_min_pct"]
    strat["sl_ratio"] = combo["sl_ratio"]

    # Convert window_mins to HH:MM
    open_hour, open_min = 9, 30
    total_mins = open_min + combo["window_mins"]
    end_hour = open_hour + total_mins // 60
    end_min = total_mins % 60
    strat["trading_window_end"] = f"{end_hour:02d}:{end_min:02d}"

    return cfg


def _compute_metrics(report: BacktestReport) -> dict:
    """Compute performance metrics from a backtest report."""
    trades = report.trades
    n = len(trades)
    if n < 5:
        return {"trades": n, "total_pnl_usd": 0, "sharpe": 0,
                "profit_factor": 0, "win_rate": 0, "max_dd": 0, "score": -999}

    pnls = [t.pnl_usd for t in trades]
    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    win_rate = len(wins) / n
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    avg = total / n
    std = math.sqrt(sum((p - avg) ** 2 for p in pnls) / (n - 1)) if n > 1 else 1
    sharpe = (avg / std) * math.sqrt(252) if std > 0 else 0

    # Max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    # Composite score: Sharpe 50% + PF 30% + WR 20%
    # Normalize: sharpe already scaled, PF cap at 5, WR is 0-1
    norm_pf = min(pf, 5.0) / 5.0
    score = 0.50 * sharpe + 0.30 * (norm_pf * 4) + 0.20 * (win_rate * 4)

    # Drawdown penalty: reduce score if DD > 50% of total profit
    if total > 0 and max_dd > 0.5 * total:
        score *= 0.8

    return {
        "trades": n,
        "total_pnl_usd": round(total, 2),
        "avg_pnl": round(avg, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "win_rate": round(win_rate, 4),
        "max_dd": round(max_dd, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "score": round(score, 4),
    }


def _run_single(args: tuple) -> Optional[dict]:
    """Worker function: run one combo on provided days. Must be top-level for pickling."""
    base_config, combo, preloaded_days = args
    cfg = _apply_combo(base_config, combo)
    engine = TouchTurnBacktestEngine(cfg)
    report = engine.run(iter(preloaded_days))
    metrics = _compute_metrics(report)
    return {**combo, **metrics}


class TouchTurnOptimizer:
    """Grid search optimizer for the Touch & Turn strategy."""

    def __init__(self, base_config: dict):
        self.base_config = base_config

    def run_grid_search(
        self,
        days: list,
        param_grid: Optional[Dict[str, list]] = None,
        n_workers: int = 0,
        min_trades: int = 10,
    ) -> List[dict]:
        """Run all parameter combinations and return sorted results."""
        grid = param_grid or DEFAULT_GRID

        # Build all combinations
        keys = sorted(grid.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]

        total = len(combos)
        logger.info("Grid search: %d combinations, %d days", total, len(days))
        print(f"  Grid: {total} combos × {len(days)} days")

        workers = n_workers or None  # None = all cores
        ctx = get_context("fork")

        results = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            tasks = [(self.base_config, combo, days) for combo in combos]
            for i, result in enumerate(pool.map(_run_single, tasks, chunksize=4)):
                if result and result.get("trades", 0) >= min_trades:
                    results.append(result)
                if (i + 1) % 50 == 0 or i + 1 == total:
                    print(f"  Progress: {i + 1}/{total}", flush=True)

        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results

    def run_walk_forward_cv(
        self,
        all_days: list,
        param_grid: Optional[Dict[str, list]] = None,
        n_folds: int = 4,
        top_n: int = 5,
        min_trades: int = 10,
        n_workers: int = 0,
        train_val_ratio: float = 5.0,
    ) -> Tuple[Optional[dict], List[dict]]:
        """Walk-forward cross-validation with voting across folds."""
        grid = param_grid or DEFAULT_GRID
        total_days = len(all_days)

        # Sort days by date
        all_days.sort(key=lambda d: d.session_date)

        # Build folds with train:val ratio
        fold_results = []
        combo_votes: Dict[str, int] = {}
        combo_scores: Dict[str, float] = {}

        if train_val_ratio > 0:
            # Ratio-based: each fold has train_ratio:1 split
            val_size = total_days // (n_folds + 1)  # rough val size
            step = val_size
        else:
            step = total_days // n_folds

        for fold_idx in range(n_folds):
            if train_val_ratio > 0:
                val_start = total_days - (n_folds - fold_idx) * step
                val_end = val_start + step
                train_size = int(step * train_val_ratio)
                train_start = max(0, val_start - train_size)
                train_days = all_days[train_start:val_start]
                val_days = all_days[val_start:val_end]
            else:
                fold_start = fold_idx * step
                fold_end = fold_start + step
                val_days = all_days[fold_start:fold_end]
                train_days = all_days[:fold_start] + all_days[fold_end:]

            if not train_days or not val_days:
                continue

            print(f"\n  Fold {fold_idx + 1}/{n_folds}: "
                  f"train={len(train_days)} days "
                  f"({train_days[0].session_date}→{train_days[-1].session_date}), "
                  f"val={len(val_days)} days "
                  f"({val_days[0].session_date}→{val_days[-1].session_date})")

            # Train: grid search
            train_results = self.run_grid_search(
                train_days, grid, n_workers=n_workers, min_trades=min_trades,
            )

            # Top-N on train
            top_train = train_results[:top_n]
            if not top_train:
                fold_results.append({"fold": fold_idx + 1, "top_val_params": []})
                continue

            # Validate top-N on val set
            val_results = []
            for params in top_train:
                combo = {k: params[k] for k in grid.keys()}
                cfg = _apply_combo(self.base_config, combo)
                engine = TouchTurnBacktestEngine(cfg)
                report = engine.run(iter(val_days))
                metrics = _compute_metrics(report)
                val_results.append({**combo, **metrics, "train_score": params["score"]})

            val_results.sort(key=lambda r: r.get("score", 0), reverse=True)

            fold_results.append({
                "fold": fold_idx + 1,
                "top_val_params": val_results,
            })

            # Vote: top validated combos get votes
            for rank, vr in enumerate(val_results[:top_n]):
                key = str({k: vr[k] for k in grid.keys()})
                combo_votes[key] = combo_votes.get(key, 0) + (top_n - rank)
                combo_scores[key] = combo_scores.get(key, 0) + vr.get("score", 0)

        # Pick winner by votes, break ties by cumulative score
        if not combo_votes:
            print("\nNo valid combos found across folds.")
            return None, fold_results

        best_key = max(combo_votes, key=lambda k: (combo_votes[k], combo_scores.get(k, 0)))
        best_params = ast.literal_eval(best_key)

        print(f"\n  Winner (votes={combo_votes[best_key]}): {best_params}")
        return best_params, fold_results

    def eval_on_oos(self, params: dict, oos_days: list) -> dict:
        """Evaluate a parameter set on out-of-sample days."""
        grid_keys = set(DEFAULT_GRID.keys())
        combo = {k: v for k, v in params.items() if k in grid_keys}
        cfg = _apply_combo(self.base_config, combo)
        engine = TouchTurnBacktestEngine(cfg)
        report = engine.run(iter(oos_days))
        return _compute_metrics(report)

    @staticmethod
    def save_csv(rows: List[dict], path: str) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {path}")
