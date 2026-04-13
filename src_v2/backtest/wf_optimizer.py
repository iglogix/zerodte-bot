"""
Walk-forward cross-validation optimizer for src_v2 backtest.

Ratio-based splitting: with ratio R and N folds, each fold's train window
grows by 1 "date_step" (= total_days / (R + N)), val window is one date_step.

For each fold:
  1. Grid search on train window
  2. Rank by train Sharpe, take top-K
  3. Evaluate top-K params on val (OOS) window
  4. Record per-combo val metrics

Final winner: param combo appearing most often in fold top-Ks
(tie-break by mean val Sharpe across folds).
"""

from __future__ import annotations

import copy
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.backtest.report import generate_report

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: Any
    train_end: Any
    val_start: Any
    val_end: Any
    n_train_days: int
    n_val_days: int
    top_params: List[Dict[str, Any]]         # top-K params (by train Sharpe)
    val_metrics: List[Dict[str, Any]]        # val metrics for each top-K params


@dataclass
class WFResult:
    best_params: Dict[str, Any]
    best_appearances: int                     # in how many folds it made top-K
    best_mean_val_sharpe: float
    best_mean_val_pnl: float
    fold_results: List[FoldResult]
    aggregate: Dict[str, Any]                 # concatenated OOS metrics


class WalkForwardOptimizer:
    def __init__(
        self,
        base_config: Dict[str, Any],
        strategy_cls,
        cbbo_store,
        days: List[Any],
        param_grid: Dict[str, List[Any]],
        ratio: int = 5,
        n_folds: int = 4,
        top_k: int = 5,
        min_trades: int = 5,
    ):
        self._base = base_config
        self._strategy_cls = strategy_cls
        self._cbbo = cbbo_store
        self._days = sorted(days, key=lambda d: d.session_date)
        self._grid = param_grid
        self._ratio = ratio
        self._n_folds = n_folds
        self._top_k = top_k
        self._min_trades = min_trades

    def run(self, progress_cb: Optional[Callable[[str], None]] = None) -> WFResult:
        combos = self._enumerate_combos()
        windows = self._make_windows()

        if not windows:
            raise RuntimeError("Not enough data for WF-CV with given ratio/n_folds")

        fold_results: List[FoldResult] = []

        # Track params across folds — key = tuple of sorted items (hashable)
        # Value = list of (fold, val_metrics) for each appearance
        appearances: Dict[tuple, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)

        for fold_idx, (train_days, val_days) in enumerate(windows, 1):
            train_start, train_end = train_days[0].session_date, train_days[-1].session_date
            val_start, val_end = val_days[0].session_date, val_days[-1].session_date
            if progress_cb:
                progress_cb(f"Fold {fold_idx}/{len(windows)} train={train_start}..{train_end} ({len(train_days)}d) val={val_start}..{val_end} ({len(val_days)}d)")

            # 1. Grid search on train
            train_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for combo in combos:
                cfg = self._apply_combo(copy.deepcopy(self._base), combo)
                orchestrator = BacktestOrchestrator(cfg, self._strategy_cls, self._cbbo)
                bt_results = orchestrator.run(train_days)
                report = generate_report(bt_results)
                metrics = self._extract(report)
                if metrics["trades"] >= self._min_trades:
                    train_results.append((combo, metrics))

            if not train_results:
                if progress_cb:
                    progress_cb(f"  Fold {fold_idx}: no valid combos in train (min_trades={self._min_trades})")
                continue

            # 2. Rank by train Sharpe, take top-K
            train_results.sort(key=lambda r: self._sort_key(r[1]), reverse=True)
            top_k_combos = train_results[:self._top_k]
            if progress_cb:
                best = top_k_combos[0]
                progress_cb(f"  Fold {fold_idx} train: {len(train_results)} valid, best Sharpe={best[1]['sharpe_annualised']:.2f}")

            # 3. Evaluate top-K on val
            val_metrics_list = []
            top_params = []
            for combo, train_m in top_k_combos:
                cfg = self._apply_combo(copy.deepcopy(self._base), combo)
                orchestrator = BacktestOrchestrator(cfg, self._strategy_cls, self._cbbo)
                bt_results = orchestrator.run(val_days)
                val_report = generate_report(bt_results)
                val_m = self._extract(val_report)
                val_metrics_list.append(val_m)
                top_params.append(combo)
                appearances[self._combo_key(combo)].append((fold_idx, val_m))

            if progress_cb:
                sharpes = [f"{m['sharpe_annualised']:.2f}" for m in val_metrics_list]
                progress_cb(f"  Fold {fold_idx} val: top-K val Sharpe={sharpes}")

            fold_results.append(FoldResult(
                fold=fold_idx,
                train_start=train_start, train_end=train_end,
                val_start=val_start, val_end=val_end,
                n_train_days=len(train_days), n_val_days=len(val_days),
                top_params=top_params,
                val_metrics=val_metrics_list,
            ))

        # 4. Select winner: most appearances, tie-break by mean val Sharpe
        best_key = None
        best_score = (-1, -float("inf"))  # (appearances, mean_val_sharpe)
        for key, appearance_list in appearances.items():
            n_app = len(appearance_list)
            mean_sharpe = sum(m["sharpe_annualised"] for _, m in appearance_list) / n_app
            score = (n_app, mean_sharpe)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            raise RuntimeError("No params found in any fold top-K")

        best_params = dict(best_key)
        best_appearances = best_score[0]
        best_mean_val_sharpe = best_score[1]
        best_appearance_list = appearances[best_key]
        best_mean_val_pnl = sum(m["total_pnl"] for _, m in best_appearance_list) / len(best_appearance_list)

        # 5. Aggregate OOS: concatenate val metrics across all folds for winning params
        aggregate = self._aggregate_oos(best_appearance_list)

        return WFResult(
            best_params=best_params,
            best_appearances=best_appearances,
            best_mean_val_sharpe=best_mean_val_sharpe,
            best_mean_val_pnl=best_mean_val_pnl,
            fold_results=fold_results,
            aggregate=aggregate,
        )

    # ------------------------------------------------------------------

    def _make_windows(self) -> List[Tuple[List, List]]:
        """Ratio-based anchored splitting on unique dates.

        With ratio R and n_folds N, date_step = total_dates / (R + N).
        Fold k: train = first (R + k - 1) * step dates; val = next step dates.
        """
        unique_dates = sorted({d.session_date for d in self._days})
        total = len(unique_dates)
        step = total // (self._ratio + self._n_folds)
        if step < 1:
            return []

        windows = []
        for k in range(self._n_folds):
            train_end_idx = (self._ratio + k) * step
            val_end_idx = train_end_idx + step
            if val_end_idx > total:
                break

            train_cutoff = unique_dates[train_end_idx]
            val_cutoff = unique_dates[min(val_end_idx, total - 1)]

            train_days = [d for d in self._days if d.session_date < train_cutoff]
            val_days = [d for d in self._days
                        if train_cutoff <= d.session_date <= val_cutoff]
            if train_days and val_days:
                windows.append((train_days, val_days))
        return windows

    def _enumerate_combos(self) -> List[Dict[str, Any]]:
        keys = list(self._grid.keys())
        value_lists = [self._grid[k] for k in keys]
        return [dict(zip(keys, v)) for v in product(*value_lists)]

    @staticmethod
    def _apply_combo(config: Dict[str, Any], combo: Dict[str, Any]) -> Dict[str, Any]:
        for path, value in combo.items():
            parts = path.split(".")
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value
        return config

    @staticmethod
    def _combo_key(combo: Dict[str, Any]) -> tuple:
        return tuple(sorted(combo.items()))

    @staticmethod
    def _extract(report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "trades": report.get("trades_placed", 0),
            "win_rate": report.get("win_rate", 0),
            "total_pnl": report.get("total_pnl_usd", 0),
            "sharpe_annualised": report.get("sharpe_annualised", 0),
            "profit_factor": report.get("profit_factor", 0),
            "max_drawdown": report.get("max_drawdown_usd", 0),
            "avg_pnl_per_trade": report.get("avg_pnl_per_trade_usd", 0),
        }

    @staticmethod
    def _sort_key(m: Dict[str, Any]) -> float:
        v = m.get("sharpe_annualised", 0)
        if isinstance(v, str):
            return float("inf") if v == "inf" else 0.0
        return v or 0.0

    @staticmethod
    def _aggregate_oos(appearances: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate OOS metrics across folds where the winning params appeared."""
        if not appearances:
            return {}
        total_trades = sum(m["trades"] for _, m in appearances)
        total_pnl = sum(m["total_pnl"] for _, m in appearances)
        total_wins = sum(m["win_rate"] * m["trades"] for _, m in appearances)
        avg_wr = total_wins / total_trades if total_trades > 0 else 0

        # Mean across folds (equal weight per fold)
        n = len(appearances)
        mean_sharpe = sum(m["sharpe_annualised"] for _, m in appearances) / n
        max_dd = max(m["max_drawdown"] for _, m in appearances)

        return {
            "n_folds_appeared": n,
            "total_oos_trades": total_trades,
            "total_oos_pnl": round(total_pnl, 2),
            "aggregate_win_rate": round(avg_wr, 4),
            "mean_fold_sharpe": round(mean_sharpe, 2),
            "max_fold_drawdown": round(max_dd, 2),
        }
