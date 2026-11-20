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
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from es_8am.backtest.options_engine import OrbOptionsBacktestEngine
from es_8am.backtest.options_report import generate_options_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------

_worker_days: List = []  # populated in each worker process by _worker_init
_worker_num_years: float = 1.0  # computed from days date range
_worker_cbbo_cache: dict | None = None  # preloaded CBBO data (shared across combos)


def _worker_init(days: List, num_years: float, cbbo_cache: dict | None = None) -> None:
    """Called once per worker process to pre-load DayBars into process memory."""
    global _worker_days, _worker_num_years, _worker_cbbo_cache
    _worker_days = days
    _worker_num_years = num_years
    _worker_cbbo_cache = cbbo_cache


def _get_strategy_instance(config):
    """Instantiate the correct strategy from config."""
    strategy_type = config.get("options", {}).get("strategy_type", "delta")
    if strategy_type == "em":
        from es_8am.backtest.strategy_em import EMStrategy
        return EMStrategy()
    elif strategy_type == "cbbo":
        from es_8am.backtest.strategy_cbbo import CbboDeltaStrategy
        return CbboDeltaStrategy()
    elif strategy_type == "cbbo_em":
        from es_8am.backtest.strategy_cbbo_em import CbboEMStrategy
        return CbboEMStrategy()
    else:
        from es_8am.backtest.strategy_delta import DeltaStrategy
        return DeltaStrategy()


def _build_combo_config(params, base_config):
    """Build config dict from params — delegates to strategy.apply_params()."""
    strat = _get_strategy_instance(base_config)
    return strat.apply_params(base_config, params)


def _score_report(params, report, min_trades, account_size=0, min_arr=0.0,
                   es_multiplier=50, max_bp_pct=0.5, max_contracts_cap=0):
    """Score a backtest report, return result row or None.

    ARR filter: if account_size > 0 and min_arr > 0, reject combos whose
    annualised return (position-sized to account) falls below min_arr.
    max_contracts_cap: hard ceiling on contracts (liquidity/OI constraint).
    """
    if report["trades_placed"] < min_trades:
        return None

    pf = report.get("profit_factor", 0)
    if pf == "inf":
        pf = 999.0

    # --- ARR calculation ---
    num_years = _worker_num_years
    spread_width = params.get("spread_width_pts",
                    max(params.get("spread_width_put", 0), params.get("spread_width_call", 0),
                        params.get("width_with_trend", 0), params.get("width_against_trend", 5)))
    margin_per_contract = spread_width * es_multiplier

    if "quantity" in params:
        # Quantity is swept — PnL already includes qty, validate margin fits BP
        qty = params["quantity"]
        total_margin = margin_per_contract * qty
        if account_size > 0 and total_margin > account_size * max_bp_pct:
            return None  # exceeds buying power
        max_contracts = qty
        annual_pnl_sized = report["total_pnl_usd"] / max(num_years, 0.5)
    else:
        # Quantity not swept — derive max_contracts from account/BP (legacy behavior)
        bp_contracts = max(int(account_size * max_bp_pct / margin_per_contract), 1) if account_size > 0 else 1
        max_contracts = min(bp_contracts, max_contracts_cap) if max_contracts_cap > 0 else bp_contracts
        annual_pnl_1c = report["total_pnl_usd"] / max(num_years, 0.5)
        annual_pnl_sized = annual_pnl_1c * max_contracts

    arr = annual_pnl_sized / account_size if account_size > 0 else 0.0

    if account_size > 0 and min_arr > 0 and arr < min_arr:
        return None

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
        "max_contracts": max_contracts,
        "arr":           round(arr, 4),
        "annual_pnl_sized": round(annual_pnl_sized, 2),
    }
    sharpe = report.get("sharpe_annualised", 0.0)
    pf_score = pf if isinstance(pf, float) else 10.0
    wr = report.get("win_rate", 0.0)
    dd = report.get("max_drawdown_usd", 0.0)
    trades = report["trades_placed"]
    # Cap sharpe to prevent low-trade flukes from dominating
    capped_sharpe = min(sharpe, 20.0)
    score = 0.50 * capped_sharpe + 0.30 * min(float(pf_score), 10.0) + 0.20 * wr * 10
    # Penalize low trade counts — results from <30 trades are unreliable
    if trades < 30:
        score *= trades / 30.0
    # DD penalty: scale by position size, penalize relative to account
    scaled_dd = dd * max_contracts
    if account_size > 0:
        dd_pct = scaled_dd / account_size  # DD as % of account
        if dd_pct > 0.10:    # >10% DD
            score *= 0.9
        if dd_pct > 0.20:    # >20% DD
            score *= 0.7
        if dd_pct > 0.35:    # >35% DD
            score *= 0.5
        if dd_pct > 0.50:    # >50% DD — near wipeout
            score *= 0.3
    else:
        if scaled_dd > 2000:
            score *= 0.8
        if scaled_dd > 5000:
            score *= 0.5
    row["rank_score"] = score
    return row


def _run_combo(args: tuple) -> Optional[Dict[str, Any]]:
    """Run a single parameter combination against the worker-resident DayBars."""
    params, base_config, min_trades, account_size, min_arr, es_multiplier, max_bp_pct, max_contracts_cap = args
    cfg = _build_combo_config(params, base_config)
    engine = OrbOptionsBacktestEngine(config=cfg)
    # Inject preloaded CBBO cache if available (avoids disk I/O per day)
    if _worker_cbbo_cache is not None and engine._cbbo_store is not None:
        engine._cbbo_store._mem_cache = _worker_cbbo_cache
    results = engine.run_on_days(_worker_days)
    report = generate_options_report(results)
    return _score_report(params, report, min_trades, account_size, min_arr,
                         es_multiplier, max_bp_pct, max_contracts_cap)


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# Delta strategy grid (default)
DEFAULT_GRID: Dict[str, List[Any]] = {
    "target_short_delta":  [0.16, 0.20, 0.25],
    "spread_width_pts":    [5, 10, 15, 20, 25],
    "tp_pct_of_credit":    [0.25, 0.50, 0.75],
    "min_orb_range_pts":   [2.0, 3.0, 5.0],
    "midline_tolerance":   [0.25, 0.50, 1.0],
    "hold_put_breach":     [False, True],
    "pricing_model":       ["bs", "bachelier"],
}

# Expected Move strategy grid
EM_GRID: Dict[str, List[Any]] = {
    "em_multiplier":          [0.5, 0.75, 1.0, 1.25, 1.5],
    "spread_width_pts":       [5, 10, 15, 20, 25],
    "tp_pct_of_credit":       [0.25, 0.50, 0.75],
    "min_credit_per_spread":  [0.10, 0.25, 0.50],
    "min_orb_range_pts":      [2.0, 3.0, 5.0],
    "midline_tolerance":      [0.25, 0.50, 1.0],
    "hold_put_breach":        [False, True],
    "pricing_model":          ["bs", "bachelier"],
}

# Quantity values to sweep when --sweep-qty is enabled
QUANTITY_GRID: List[int] = [1, 5, 10, 25, 50]


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _compute_num_years(days: List) -> float:
    """Compute the date span of a day list in years."""
    if not days:
        return 1.0
    dates = [d.session_date for d in days]
    span_days = (max(dates) - min(dates)).days
    return max(span_days / 365.25, 0.5)


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
        account_size: float = 0,
        min_arr: float = 0.0,
        es_multiplier: float = 50,
        max_bp_pct: float = 0.5,
        max_contracts_cap: int = 0,
    ):
        self._base_config = base_config
        self._zip_path = zip_path
        self._cbbo_cache: dict | None = None
        # Preload CBBO cache into memory if configured
        cbbo_cache_dir = base_config.get("options", {}).get("cbbo_cache_dir", "")
        if cbbo_cache_dir and Path(cbbo_cache_dir).exists():
            from es_8am.backtest.cbbo_store import CbboStore
            import time as _t
            t0 = _t.time()
            self._cbbo_cache = CbboStore.preload_cache(cbbo_cache_dir)
            print(f"  CBBO preloaded: {len(self._cbbo_cache)} days in {_t.time()-t0:.1f}s", flush=True)
        self._start_date = start_date
        self._end_date = end_date
        self._account_size = account_size
        self._min_arr = min_arr
        self._es_multiplier = es_multiplier
        self._max_bp_pct = max_bp_pct
        self._max_contracts_cap = max_contracts_cap

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
        from es_8am.backtest.data_loader import DatabentoLoader
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

        num_years = _compute_num_years(all_days)
        task_args = [
            (dict(zip(keys, combo)), self._base_config, min_trades,
             self._account_size, self._min_arr, self._es_multiplier, self._max_bp_pct, self._max_contracts_cap)
            for combo in combos
        ]

        if self._account_size > 0 and self._min_arr > 0:
            cap_str = str(self._max_contracts_cap) if self._max_contracts_cap > 0 else "none"
            print(f"ARR filter: account=${self._account_size:,.0f}  "
                  f"min_arr={self._min_arr*100:.0f}%  "
                  f"ES_mult=${self._es_multiplier:.0f}  "
                  f"max_BP={self._max_bp_pct*100:.0f}%  "
                  f"max_contracts={cap_str}", flush=True)

        rows: List[Dict[str, Any]] = []
        t0 = time.time()
        done = 0

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn"),
            initializer=_worker_init,
            initargs=(all_days, num_years, self._cbbo_cache),
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
        """Build config from params — delegates to strategy.apply_params()."""
        strat = _get_strategy_instance(self._base_config)
        return strat.apply_params(self._base_config, params)

    def _flatten_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        pf = report.get("profit_factor", 0)
        if pf == "inf":
            pf = 999.0
        return {
            "trades":        report.get("trades_placed", 0),
            "win_rate":      report.get("win_rate", 0),
            "total_pnl_usd": report.get("total_pnl_usd", 0),
            "avg_pnl_usd":   report.get("avg_pnl_per_trade_usd", 0),
            "profit_factor": pf,
            "max_dd_usd":    report.get("max_drawdown_usd", 0),
            "sharpe":        report.get("sharpe_annualised", 0),
            "avg_credit":    report.get("avg_credit_received", 0),
            "avg_delta":     report.get("avg_short_delta", 0),
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
        # Build header dynamically based on strategy params
        is_em = "em_multiplier" in (rows[0] if rows else {})
        if is_em:
            strat_hdr = f"{'em_mult':>7}  {'width':>6}  {'tp%':>5}  "
        else:
            strat_hdr = f"{'delta':>6}  {'width':>6}  {'tp%':>5}  {'iv':>5}  "
        hdr = (
            f"{'#':>3}  {strat_hdr}"
            f"{'min_rng':>7}  {'tol':>5}  "
            f"{'trades':>7}  {'WR%':>6}  {'sharpe':>7}  {'PF':>6}  "
            f"{'total$':>9}  {'avg$':>7}  {'maxDD$':>8}"
        )
        print(hdr)
        print("-" * 130)
        for rank, r in enumerate(rows, 1):
            pf = r["profit_factor"]
            pf_str = f"{pf:.2f}" if isinstance(pf, float) and pf < 100 else "inf"
            if is_em:
                strat_cols = f"{r['em_multiplier']:>7.2f}  {r['spread_width_pts']:>6.0f}  {r['tp_pct_of_credit']:>5.2f}  "
            else:
                iv_val = r.get('assumed_iv', 0)
                strat_cols = (
                    f"{r.get('target_short_delta', 0):>6.2f}  "
                    f"{r['spread_width_pts']:>6.0f}  "
                    f"{r['tp_pct_of_credit']:>5.2f}  "
                    f"{iv_val:>5.2f}  "
                )
            print(
                f"{rank:>3}  {strat_cols}"
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
        train_val_ratio: float = 0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Walk-forward cross-validation over ``all_days``.

        If train_val_ratio > 0, uses ratio-based splitting:
          val_size = total / (ratio + 1) per step, expanding train window,
          val always = train / ratio.
        Otherwise falls back to equal-fold splitting.

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

        if train_val_ratio > 0:
            # Ratio-based splitting on UNIQUE DATES (not raw chunks).
            # This ensures val windows contain enough trading days.
            unique_dates = sorted(set(d.session_date for d in sorted_days))
            date_step = len(unique_dates) // (train_val_ratio + n_folds)
            date_step = max(date_step, 1)

            windows: List[Tuple[List, List]] = []
            for k in range(n_folds):
                train_end_date_idx = int(date_step * (train_val_ratio + k))
                val_end_date_idx = int(train_end_date_idx + date_step)
                if val_end_date_idx > len(unique_dates):
                    break
                train_cutoff = unique_dates[train_end_date_idx]
                val_cutoff = unique_dates[min(val_end_date_idx, len(unique_dates) - 1)]

                train_days_w = [d for d in sorted_days if d.session_date < train_cutoff]
                val_days_w = [d for d in sorted_days
                              if train_cutoff <= d.session_date <= val_cutoff]
                if train_days_w and val_days_w:
                    windows.append((train_days_w, val_days_w))
            print(f"  Ratio-based splits: ratio={train_val_ratio:.0f}:1, "
                  f"step={date_step} unique dates, {len(windows)} folds "
                  f"({len(unique_dates)} unique dates total)", flush=True)
        else:
            fold_size = len(sorted_days) // n_folds
            folds_list = [sorted_days[i * fold_size: (i + 1) * fold_size] for i in range(n_folds)]
            windows = []
            for k in range(1, n_folds):
                train_days_w = [d for fold in folds_list[:k] for d in fold]
                val_days_w = folds_list[k]
                windows.append((train_days_w, val_days_w))

        workers = n_workers if n_workers > 0 else os.cpu_count() or 1

        fold_results: List[Dict[str, Any]] = []
        # votes[param_key] = list of val rank_scores across folds where it appeared in top-N
        votes: Dict[str, List[float]] = {}

        for k, (train_days, val_days) in enumerate(windows, 1):

            train_start = train_days[0].session_date
            train_end   = train_days[-1].session_date
            val_start   = val_days[0].session_date
            val_end     = val_days[-1].session_date

            total_folds = len(windows)
            train_ratio = len(train_days) / max(len(val_days), 1)
            print(f"\n{'='*60}", flush=True)
            print(f"  Fold {k}/{total_folds}  "
                  f"train={train_start}..{train_end} ({len(train_days)})  "
                  f"val={val_start}..{val_end} ({len(val_days)})  "
                  f"ratio={train_ratio:.1f}:1", flush=True)
            print(f"{'='*60}", flush=True)
            logger.info(
                "Fold %d/%d: train=%s..%s (%d days), val=%s..%s (%d days), ratio=%.1f:1",
                k, total_folds, train_start, train_end, len(train_days),
                val_start, val_end, len(val_days), train_ratio,
            )

            # --- Train sweep (parallel) ---
            print(f"  Train sweep: {len(combos)} combos x "
                  f"{len(train_days)} days [{workers} workers]", flush=True)
            train_rows = self._sweep_parallel(
                combos, train_days, min_trades, workers,
            )
            train_rows.sort(key=lambda r: r["rank_score"], reverse=True)
            if train_rows:
                msg = (f"  Train: {len(train_rows)} valid combos, "
                       f"best sharpe={train_rows[0]['sharpe']:.2f}")
            else:
                msg = "  No valid combos"
            print(msg, flush=True)
            logger.info("Fold %d train: %s", k, msg.strip())

            # --- Validate top-N (uses initializer to avoid pickling days) ---
            candidates = train_rows[:top_n]
            print(f"  Validating top-{len(candidates)} on val window...",
                  flush=True)
            logger.info("Fold %d: validating %d candidates on %d val days",
                        k, len(candidates), len(val_days))
            val_num_years = _compute_num_years(val_days)
            val_task_args = [
                (params, self._base_config, min_trades,
                 self._account_size, self._min_arr, self._es_multiplier, self._max_bp_pct, self._max_contracts_cap)
                for params in [{k2: r[k2] for k2 in keys} for r in candidates]
            ]
            val_rows = []
            if not val_task_args:
                logging.getLogger(__name__).info(f"Fold {k+1}: 0 valid val combos")
                fold_results.append({"fold": k + 1, "train_start": str(train_days[0].session_date),
                    "train_end": str(train_days[-1].session_date),
                    "val_start": str(val_days[0].session_date),
                    "val_end": str(val_days[-1].session_date),
                    "train_sharpe": train_rows[0]["sharpe"] if train_rows else 0.0, "val_sharpe": 0,
                    "train_ratio": len(train_days) / max(len(val_days), 1),
                    "top_val_params": []})
                continue
            with ProcessPoolExecutor(
                max_workers=min(workers, len(val_task_args)),
                mp_context=multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn"),
                initializer=_worker_init,
                initargs=(val_days, val_num_years, self._cbbo_cache),
            ) as pool:
                futures = {
                    pool.submit(_run_combo, arg): arg
                    for arg in val_task_args
                }
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                    except Exception:
                        logger.exception("Validation worker failed")
                        continue
                    if result is not None:
                        val_rows.append(result)
            val_rows.sort(key=lambda r: r["rank_score"], reverse=True)
            logger.info("Fold %d: %d valid val combos", k, len(val_rows))

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
        """Run all combos against ``days`` in parallel.

        Uses initializer to load days once per worker process (avoids
        pickling the full day list with every task).
        """
        num_years = _compute_num_years(days)
        task_args = [
            (params, self._base_config, min_trades,
             self._account_size, self._min_arr, self._es_multiplier, self._max_bp_pct, self._max_contracts_cap)
            for params in combos
        ]
        rows: List[Dict[str, Any]] = []
        total = len(task_args)
        done = 0
        t0 = time.time()
        logger.info(
            "Starting sweep: %d combos, %d days, %d workers",
            total, len(days), workers,
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn"),
            initializer=_worker_init,
            initargs=(days, num_years, self._cbbo_cache),
        ) as pool:
            futures = {
                pool.submit(_run_combo, arg): arg for arg in task_args
            }
            for fut in as_completed(futures):
                done += 1
                try:
                    result = fut.result()
                except Exception:
                    logger.exception("Worker failed on combo %d/%d", done, total)
                    continue
                if result is not None:
                    rows.append(result)
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    eta = (
                        elapsed / done * (total - done)
                        if done < total else 0
                    )
                    best = max(
                        (r["sharpe"] for r in rows), default=0.0
                    )
                    msg = (
                        f"    {done}/{total}  elapsed={elapsed:.0f}s  "
                        f"ETA={eta:.0f}s  best_sharpe={best:.2f}"
                    )
                    print(msg, flush=True)
                    logger.info(msg.strip())
        logger.info(
            "Sweep done: %d/%d valid in %.1fs",
            len(rows), total, time.time() - t0,
        )
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

        # Add ARR metrics
        num_years = _compute_num_years(oos_days)
        spread_width = best_params.get("spread_width_pts", 5)
        margin_per_contract = spread_width * self._es_multiplier
        bp_contracts = max(int(self._account_size * self._max_bp_pct / margin_per_contract), 1) if self._account_size > 0 else 1
        max_contracts = min(bp_contracts, self._max_contracts_cap) if self._max_contracts_cap > 0 else bp_contracts
        annual_pnl_1c = report["total_pnl_usd"] / max(num_years, 0.5)
        annual_pnl_sized = annual_pnl_1c * max_contracts
        arr = annual_pnl_sized / self._account_size if self._account_size > 0 else 0.0

        flat["max_contracts"] = max_contracts
        flat["arr"] = round(arr, 4)
        flat["annual_pnl_sized"] = round(annual_pnl_sized, 2)
        flat["scaled_max_dd"] = round(flat.get("max_dd_usd", 0) * max_contracts, 2)
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
            print(f"\n  OOS result (per contract):")
            print(f"    trades={oos_report.get('trades')}  "
                  f"sharpe={oos_report.get('sharpe', 0):.2f}  "
                  f"WR={oos_report.get('win_rate', 0)*100:.1f}%  "
                  f"PF={oos_report.get('profit_factor', 0):.2f}  "
                  f"total_pnl=${oos_report.get('total_pnl_usd', 0):,.0f}  "
                  f"maxDD=${oos_report.get('max_dd_usd', 0):,.0f}")
            if oos_report.get("max_contracts"):
                print(f"\n  Position-sized ({oos_report['max_contracts']} contracts):")
                print(f"    annual_pnl=${oos_report.get('annual_pnl_sized', 0):,.0f}  "
                      f"ARR={oos_report.get('arr', 0)*100:.1f}%  "
                      f"scaled_maxDD=${oos_report.get('scaled_max_dd', 0):,.0f}")
        print(sep)

    # -----------------------------------------------------------------------
    # Save optimized config
    # -----------------------------------------------------------------------

    def save_optimized_config(
        self,
        best_params: Dict[str, Any],
        oos_report: Optional[Dict[str, Any]] = None,
        fold_results: Optional[List[Dict[str, Any]]] = None,
        output_path: str = "config/optimized_params.yaml",
    ):
        """Write winning params to a YAML config file for the live bot."""
        now_str = datetime.now().isoformat(timespec="seconds")

        # Map grid params back to config structure
        strategy_cfg = {
            "midline_tolerance_pts": best_params.get("midline_tolerance"),
            "orb_range_filter": {
                "min_pts": best_params.get("min_orb_range_pts"),
            },
        }
        # Quantity: use swept value if available, else derive from OOS position sizing
        if "quantity" in best_params:
            opt_qty = best_params["quantity"]
        elif oos_report and oos_report.get("max_contracts"):
            opt_qty = oos_report["max_contracts"]
        else:
            opt_qty = 1

        options_cfg = {
            "spread_width_pts": best_params.get("spread_width_pts"),
            "tp_pct_of_credit": best_params.get("tp_pct_of_credit"),
            "quantity": opt_qty,
        }
        # Strategy-specific params
        if "target_short_delta" in best_params:
            options_cfg["target_short_delta"] = best_params["target_short_delta"]
        if "em_multiplier" in best_params:
            options_cfg["em_multiplier"] = best_params["em_multiplier"]
            options_cfg["strategy_type"] = "em"
        if "hold_put_breach" in best_params:
            options_cfg["hold_put_breach"] = best_params["hold_put_breach"]
        if "pricing_model" in best_params:
            options_cfg["pricing_model"] = best_params["pricing_model"]

        # Compute risk limit that accommodates the optimized position
        mult = self._es_multiplier
        width = best_params.get("spread_width_pts", 5)
        risk_limit = width * mult * opt_qty
        risk_cfg = {
            "max_loss_per_trade_usd": risk_limit,
            "max_daily_loss_usd": risk_limit,
        }

        dump_data: Dict[str, Any] = {
            "optimized_at": now_str,
            "strategy": strategy_cfg,
            "options": options_cfg,
            "risk": risk_cfg,
        }

        if oos_report:
            pf = oos_report.get("profit_factor", 0)
            dump_data["_oos_metrics"] = {
                "sharpe": round(oos_report.get("sharpe", 0), 2),
                "win_rate": round(oos_report.get("win_rate", 0), 4),
                "profit_factor": round(float(pf), 2),
                "trades": oos_report.get("trades", 0),
                "total_pnl_usd": round(
                    oos_report.get("total_pnl_usd", 0), 2
                ),
                "max_dd_usd": round(
                    oos_report.get("max_dd_usd", 0), 2
                ),
            }
            if oos_report.get("max_contracts"):
                dump_data["_position_sizing"] = {
                    "account_size": self._account_size,
                    "max_contracts": oos_report["max_contracts"],
                    "arr": round(oos_report.get("arr", 0), 4),
                    "annual_pnl_sized": oos_report.get("annual_pnl_sized", 0),
                    "scaled_max_dd": oos_report.get("scaled_max_dd", 0),
                }

        if fold_results:
            dump_data["_cv_folds"] = len(fold_results)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                "# Optimized parameters from walk-forward CV\n"
                f"# Generated: {now_str}\n"
                "# Load as override on top of the strategy config\n\n"
            )
            yaml.dump(
                dump_data, f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"Optimized params saved -> {output_path}")

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
