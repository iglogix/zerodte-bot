"""
Grid-search optimizer for the ORB options backtest.

Sweeps combinations of:
  - target_short_delta
  - spread_width_pts
  - tp_pct_of_credit
  - assumed_iv
  - min_orb_range_pts
  - midline_tolerance_pts

Pre-loads all DayBars once from the zip, then sweeps in-memory for speed.
Ranks by Sharpe -> profit_factor -> win_rate -> total_pnl.
Prints top-N table and saves full results to CSV.
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src_8am.backtest.options_engine import OrbOptionsBacktestEngine
from src_8am.backtest.options_report import generate_options_report


# ---------------------------------------------------------------------------
# Worker helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------

_worker_days: List = []  # populated in each worker process by _worker_init


def _worker_init(days: List) -> None:
    """Called once per worker process to pre-load DayBars into process memory."""
    global _worker_days
    _worker_days = days


def _run_combo_on_days(args: tuple) -> Optional[Dict[str, Any]]:
    """Run a single combo on an explicit list of DayBars (used for CV folds)."""
    params, base_config, min_trades, days = args
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("options", {})
    cfg.setdefault("strategy", {})
    cfg["strategy"].setdefault("orb_range_filter", {})
    cfg["options"]["target_short_delta"] = params["target_short_delta"]
    cfg["options"]["spread_width_pts"] = params["spread_width_pts"]
    cfg["options"]["tp_pct_of_credit"] = params["tp_pct_of_credit"]
    cfg["options"]["assumed_iv"] = params["assumed_iv"]
    cfg["strategy"]["midline_tolerance_pts"] = params["midline_tolerance"]
    cfg["strategy"]["orb_range_filter"]["min_pts"] = params["min_orb_range_pts"]

    engine = OrbOptionsBacktestEngine(config=cfg)
    results = engine.run_on_days(days)
    report = generate_options_report(results)

    if report["trades_placed"] < min_trades:
        return None

    pf = report.get("profit_factor", 0)
    if pf == "inf":
        pf = 999.0
    row = {
        **params,
        "trades":        report["trades_placed"],
        "win_rate":      report["win_rate"],
        "total_pnl_usd": report["total_pnl_usd"],
        "avg_pnl_usd":   report["avg_pnl_per_trade_usd"],
        "profit_factor": pf,
        "max_dd_usd":    report["max_drawdown_usd"],
        "sharpe":        report["sharpe_annualised"],
        "avg_credit":    report["avg_credit_received"],
        "avg_delta":     report["avg_short_delta"],
    }
    sharpe = report.get("sharpe_annualised", 0.0)
    pf_score = pf if isinstance(pf, float) else 10.0
    wr = report.get("win_rate", 0.0)
    dd = report.get("max_drawdown_usd", 0.0)
    score = 0.50 * sharpe + 0.30 * min(float(pf_score), 10.0) + 0.20 * wr * 10
    if dd > 2000:
        score *= 0.8
    if dd > 5000:
        score *= 0.5
    row["rank_score"] = score
    return row


def _run_combo(args: tuple) -> Optional[Dict[str, Any]]:
    """Run a single parameter combination against the worker-resident DayBars."""
    params, base_config, min_trades = args
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("options", {})
    cfg.setdefault("strategy", {})
    cfg["strategy"].setdefault("orb_range_filter", {})
    cfg["options"]["target_short_delta"] = params["target_short_delta"]
    cfg["options"]["spread_width_pts"] = params["spread_width_pts"]
    cfg["options"]["tp_pct_of_credit"] = params["tp_pct_of_credit"]
    cfg["options"]["assumed_iv"] = params["assumed_iv"]
    cfg["strategy"]["midline_tolerance_pts"] = params["midline_tolerance"]
    cfg["strategy"]["orb_range_filter"]["min_pts"] = params["min_orb_range_pts"]

    engine = OrbOptionsBacktestEngine(config=cfg)
    results = engine.run_on_days(_worker_days)
    report = generate_options_report(results)

    if report["trades_placed"] < min_trades:
        return None

    pf = report.get("profit_factor", 0)
    if pf == "inf":
        pf = 999.0
    row = {
        **params,
        "trades":        report["trades_placed"],
        "win_rate":      report["win_rate"],
        "total_pnl_usd": report["total_pnl_usd"],
        "avg_pnl_usd":   report["avg_pnl_per_trade_usd"],
        "profit_factor": pf,
        "max_dd_usd":    report["max_drawdown_usd"],
        "sharpe":        report["sharpe_annualised"],
        "avg_credit":    report["avg_credit_received"],
        "avg_delta":     report["avg_short_delta"],
    }
    sharpe = report.get("sharpe_annualised", 0.0)
    pf_score = pf if isinstance(pf, float) else 10.0
    wr = report.get("win_rate", 0.0)
    dd = report.get("max_drawdown_usd", 0.0)
    score = 0.50 * sharpe + 0.30 * min(float(pf_score), 10.0) + 0.20 * wr * 10
    if dd > 2000:
        score *= 0.8
    if dd > 5000:
        score *= 0.5
    row["rank_score"] = score
    return row


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

DEFAULT_GRID: Dict[str, List[Any]] = {
    "target_short_delta":  [0.10, 0.12, 0.16, 0.20, 0.25],
    "spread_width_pts":    [5, 10, 15],
    "tp_pct_of_credit":    [0.25, 0.50, 0.75],
    "assumed_iv":          [0.15, 0.20],
    "min_orb_range_pts":   [2.0, 3.0, 5.0],
    "midline_tolerance":   [0.25, 0.50, 1.0],
}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class GridSearchOptimizer:
    """
    Runs OrbOptionsBacktestEngine over every combination in param_grid.

    Pre-loads all DayBars once from the zip, then runs each combo in-memory
    (avoids re-parsing the .zst file on every iteration).

    Usage:
        opt = GridSearchOptimizer(base_config, zip_path)
        rows = opt.run(param_grid=DEFAULT_GRID, top_n=20)
        opt.save_csv(rows, "backtest_optimizer_results.csv")
        opt.print_table(rows[:20])
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        zip_path: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        self._base_config = base_config
        self._zip_path = zip_path
        self._start_date = start_date
        self._end_date = end_date

    def run(
        self,
        param_grid: Dict[str, List[Any]] = None,
        top_n: int = 20,
        min_trades: int = 30,
        n_workers: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Execute the grid search.

        Pre-loads all DayBars once, then sweeps all combos in-memory.
        Returns a list of result dicts sorted by rank score (best first).
        Each dict contains the parameters plus all report metrics.

        Args:
            n_workers: Number of parallel worker processes.  0 (default) uses
                       all available CPU cores.  Set to 1 to disable parallelism.
        """
        if param_grid is None:
            param_grid = DEFAULT_GRID

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))
        total = len(combos)

        # Pre-load all bars once from the zip
        from src_8am.backtest.data_loader import DatabentoLoader
        print("Loading bars from zip (once)...", flush=True)
        t_load = time.time()
        loader = DatabentoLoader(self._zip_path)
        all_days = list(loader.iter_days(
            start_date=self._start_date,
            end_date=self._end_date,
        ))
        print(f"  Loaded {len(all_days)} day-symbol chunks in {time.time()-t_load:.1f}s")

        workers = n_workers if n_workers > 0 else os.cpu_count() or 1
        print(f"Grid search: {total} combinations x {len(all_days)} day chunks  "
              f"[{workers} workers]")
        print()

        task_args = [
            (dict(zip(keys, combo)), self._base_config, min_trades)
            for combo in combos
        ]

        rows: List[Dict[str, Any]] = []
        t0 = time.time()
        done = 0

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_worker_init,
            initargs=(all_days,),
        ) as pool:
            futures = {pool.submit(_run_combo, arg): arg for arg in task_args}
            for fut in as_completed(futures):
                done += 1
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    best = max((r["sharpe"] for r in rows), default=0.0)
                    print(f"  {done}/{total}  elapsed={elapsed:.0f}s  "
                          f"ETA={eta:.0f}s  best_sharpe={best:.2f}")

        rows.sort(key=lambda r: r["rank_score"], reverse=True)
        elapsed = time.time() - t0
        print(f"\nDone -- {len(rows)} valid combos in {elapsed:.1f}s")
        return rows

    # -----------------------------------------------------------------------
    # Config builder
    # -----------------------------------------------------------------------

    def _build_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(self._base_config)
        cfg.setdefault("options", {})
        cfg.setdefault("strategy", {})
        cfg["strategy"].setdefault("orb_range_filter", {})

        cfg["options"]["target_short_delta"] = params["target_short_delta"]
        cfg["options"]["spread_width_pts"] = params["spread_width_pts"]
        cfg["options"]["tp_pct_of_credit"] = params["tp_pct_of_credit"]
        cfg["options"]["assumed_iv"] = params["assumed_iv"]
        cfg["strategy"]["midline_tolerance_pts"] = params["midline_tolerance"]
        cfg["strategy"]["orb_range_filter"]["min_pts"] = params["min_orb_range_pts"]
        return cfg

    def _flatten_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        pf = report.get("profit_factor", 0)
        if pf == "inf":
            pf = 999.0
        return {
            "trades":        report["trades_placed"],
            "win_rate":      report["win_rate"],
            "total_pnl_usd": report["total_pnl_usd"],
            "avg_pnl_usd":   report["avg_pnl_per_trade_usd"],
            "profit_factor": pf,
            "max_dd_usd":    report["max_drawdown_usd"],
            "sharpe":        report["sharpe_annualised"],
            "avg_credit":    report["avg_credit_received"],
            "avg_delta":     report["avg_short_delta"],
        }

    @staticmethod
    def _rank_score(report: Dict[str, Any]) -> float:
        """
        Composite rank score (higher = better).
        Weights: Sharpe 50%, profit_factor 30%, win_rate 20%.
        Penalises max drawdown > $2000.
        """
        sharpe = report.get("sharpe_annualised", 0.0)
        pf = report.get("profit_factor", 0)
        if pf == "inf":
            pf = 10.0
        wr = report.get("win_rate", 0.0)
        dd = report.get("max_drawdown_usd", 0.0)

        score = 0.50 * sharpe + 0.30 * min(float(pf), 10.0) + 0.20 * wr * 10
        if dd > 2000:
            score *= 0.8
        if dd > 5000:
            score *= 0.5
        return score

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def print_table(self, rows: List[Dict[str, Any]], title: str = "Top combinations"):
        if not rows:
            print("No results.")
            return
        sep = "=" * 130
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        hdr = (
            f"{'#':>3}  {'delta':>6}  {'width':>6}  {'tp%':>5}  {'iv':>5}  "
            f"{'min_rng':>7}  {'tol':>5}  "
            f"{'trades':>7}  {'WR%':>6}  {'sharpe':>7}  {'PF':>6}  "
            f"{'total$':>9}  {'avg$':>7}  {'maxDD$':>8}"
        )
        print(hdr)
        print("-" * 130)
        for rank, r in enumerate(rows, 1):
            pf = r["profit_factor"]
            pf_str = f"{pf:.2f}" if isinstance(pf, float) and pf < 100 else "inf"
            print(
                f"{rank:>3}  "
                f"{r['target_short_delta']:>6.2f}  "
                f"{r['spread_width_pts']:>6.0f}  "
                f"{r['tp_pct_of_credit']:>5.2f}  "
                f"{r['assumed_iv']:>5.2f}  "
                f"{r['min_orb_range_pts']:>7.1f}  "
                f"{r['midline_tolerance']:>5.2f}  "
                f"{r['trades']:>7}  "
                f"{r['win_rate']*100:>6.1f}  "
                f"{r['sharpe']:>7.2f}  "
                f"{pf_str:>6}  "
                f"{r['total_pnl_usd']:>9,.0f}  "
                f"{r['avg_pnl_usd']:>7.2f}  "
                f"{r['max_dd_usd']:>8,.0f}"
            )
        print(sep)

    # -----------------------------------------------------------------------
    # Walk-forward cross-validation
    # -----------------------------------------------------------------------

    def run_walk_forward_cv(
        self,
        all_days: List,
        param_grid: Dict[str, List[Any]] = None,
        n_folds: int = 4,
        top_n: int = 5,
        min_trades: int = 20,
        n_workers: int = 0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Walk-forward cross-validation over ``all_days``.

        Splits all_days chronologically into ``n_folds`` equal windows.
        For each fold k:
          - Train on folds 0..k-1  (expanding window)
          - Validate on fold k

        Selects the parameter set that appears most consistently in the
        top-N across all validation folds (majority vote, tie-break by
        mean val sharpe).

        Returns:
            best_params : the winning parameter dict
            fold_results: list of per-fold summary dicts
        """
        if param_grid is None:
            param_grid = DEFAULT_GRID

        keys = list(param_grid.keys())
        combos = [dict(zip(keys, c)) for c in itertools.product(*param_grid.values())]

        sorted_days = sorted(all_days, key=lambda d: d.session_date)
        fold_size = len(sorted_days) // n_folds
        folds = [sorted_days[i * fold_size: (i + 1) * fold_size] for i in range(n_folds)]

        workers = n_workers if n_workers > 0 else os.cpu_count() or 1

        fold_results: List[Dict[str, Any]] = []
        # votes[param_key] = list of val rank_scores across folds where it appeared in top-N
        votes: Dict[str, List[float]] = {}

        for k in range(1, n_folds):
            train_days = [d for fold in folds[:k] for d in fold]
            val_days = folds[k]

            train_start = train_days[0].session_date
            train_end   = train_days[-1].session_date
            val_start   = val_days[0].session_date
            val_end     = val_days[-1].session_date

            print(f"\n{'='*60}")
            print(f"  Fold {k}/{n_folds-1}  "
                  f"train={train_start}..{train_end}  "
                  f"val={val_start}..{val_end}")
            print(f"{'='*60}")

            # --- Train sweep (parallel) ---
            print(f"  Train sweep: {len(combos)} combos x {len(train_days)} days [{workers} workers]")
            train_rows = self._sweep_parallel(combos, train_days, min_trades, workers)
            train_rows.sort(key=lambda r: r["rank_score"], reverse=True)
            print(f"  Train: {len(train_rows)} valid combos, "
                  f"best sharpe={train_rows[0]['sharpe']:.2f}" if train_rows else "  No valid combos")

            # --- Validate top-N ---
            candidates = train_rows[:top_n]
            print(f"  Validating top-{len(candidates)} on val window...")
            val_task_args = [
                (params, self._base_config, min_trades, val_days)
                for params in [{k2: r[k2] for k2 in keys} for r in candidates]
            ]
            val_rows = []
            with ProcessPoolExecutor(
                max_workers=min(workers, len(val_task_args)),
                mp_context=multiprocessing.get_context("fork"),
            ) as pool:
                for result in pool.map(_run_combo_on_days, val_task_args):
                    if result is not None:
                        val_rows.append(result)
            val_rows.sort(key=lambda r: r["rank_score"], reverse=True)

            for r in val_rows[:top_n]:
                pkey = self._param_key(r, keys)
                votes.setdefault(pkey, []).append(r["rank_score"])

            fold_results.append({
                "fold": k,
                "train_start": train_start, "train_end": train_end,
                "val_start": val_start,   "val_end": val_end,
                "train_combos": len(train_rows),
                "val_combos":   len(val_rows),
                "best_train_sharpe": train_rows[0]["sharpe"] if train_rows else 0.0,
                "best_val_sharpe":   val_rows[0]["sharpe"]   if val_rows   else 0.0,
                "top_val_params":    val_rows[:top_n],
            })

        # --- Select winner: most appearances in top-N, tie-break by mean score ---
        if not votes:
            print("\nNo consistent winner found across folds.")
            return {}, fold_results

        best_key = max(votes, key=lambda k2: (len(votes[k2]), sum(votes[k2]) / len(votes[k2])))
        best_params = self._key_to_params(best_key, keys, param_grid)

        appearances = len(votes[best_key])
        mean_score  = sum(votes[best_key]) / appearances
        print(f"\nWinner: {best_params}")
        print(f"  Appeared in top-{top_n} on {appearances}/{n_folds-1} folds, "
              f"mean val score={mean_score:.3f}")

        return best_params, fold_results

    def _sweep_parallel(
        self,
        combos: List[Dict[str, Any]],
        days: List,
        min_trades: int,
        workers: int,
    ) -> List[Dict[str, Any]]:
        """Run all combos against ``days`` in parallel, return valid rows."""
        task_args = [(params, self._base_config, min_trades, days) for params in combos]
        rows: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            for result in pool.map(_run_combo_on_days, task_args):
                if result is not None:
                    rows.append(result)
        return rows

    @staticmethod
    def _param_key(row: Dict[str, Any], keys: List[str]) -> str:
        return "|".join(f"{k}={row[k]}" for k in keys)

    @staticmethod
    def _key_to_params(key: str, keys: List[str], grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        mapping = dict(item.split("=", 1) for item in key.split("|"))
        params: Dict[str, Any] = {}
        for k in keys:
            raw = mapping[k]
            # Cast back to the type used in the grid
            sample = grid[k][0]
            params[k] = type(sample)(raw)
        return params

    def eval_on_oos(
        self,
        best_params: Dict[str, Any],
        oos_days: List,
    ) -> Dict[str, Any]:
        """Evaluate a single parameter set on the out-of-sample window."""
        cfg = self._build_config(best_params)
        engine = OrbOptionsBacktestEngine(config=cfg)
        results = engine.run_on_days(oos_days)
        report = generate_options_report(results)
        flat = self._flatten_report(report)
        flat["rank_score"] = self._rank_score(report)
        return flat

    def print_cv_summary(
        self,
        fold_results: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        oos_report: Optional[Dict[str, Any]] = None,
    ):
        sep = "=" * 70
        print(f"\n{sep}")
        print("  Walk-Forward CV Summary")
        print(sep)
        print(f"  {'Fold':<6} {'Train window':<25} {'Val window':<25} "
              f"{'Train sharpe':>12} {'Val sharpe':>10}")
        print(f"  {'-'*6} {'-'*25} {'-'*25} {'-'*12} {'-'*10}")
        for f in fold_results:
            print(f"  {f['fold']:<6} "
                  f"{str(f['train_start'])+' -> '+str(f['train_end']):<25} "
                  f"{str(f['val_start'])+' -> '+str(f['val_end']):<25} "
                  f"{f['best_train_sharpe']:>12.2f} "
                  f"{f['best_val_sharpe']:>10.2f}")
        print(sep)
        print(f"\n  Best params: {best_params}")
        if oos_report:
            print(f"\n  OOS result:")
            print(f"    trades={oos_report.get('trades')}  "
                  f"sharpe={oos_report.get('sharpe', 0):.2f}  "
                  f"WR={oos_report.get('win_rate', 0)*100:.1f}%  "
                  f"PF={oos_report.get('profit_factor', 0):.2f}  "
                  f"total_pnl=${oos_report.get('total_pnl_usd', 0):,.0f}  "
                  f"maxDD=${oos_report.get('max_dd_usd', 0):,.0f}")
        print(sep)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def save_csv(self, rows: List[Dict[str, Any]], path: str):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows -> {path}")
