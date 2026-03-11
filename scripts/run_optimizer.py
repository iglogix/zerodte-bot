"""
Walk-forward CV + OOS optimizer runner.

Data split (5 years: 2021-03-01 -> 2026-02-26):
  CV window  : 2021-03-01 -> 2024-12-31  (~4 years, 4 folds of ~1 year)
  OOS window : 2025-01-01 -> 2026-02-26  (~14 months, never seen during optimisation)

Usage:
  python -m scripts.run_optimizer
  python -m scripts.run_optimizer --n-workers 4
  python -m scripts.run_optimizer --folds 4 --top-n 5
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import yaml

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_8am.backtest.data_loader import DatabentoLoader
from src_8am.backtest.optimizer import DEFAULT_GRID, GridSearchOptimizer


# ---------------------------------------------------------------------------
# Date boundaries
# ---------------------------------------------------------------------------

CV_START  = date(2021, 3,  1)
CV_END    = date(2024, 12, 31)
OOS_START = date(2025, 1,  1)
OOS_END   = date(2026, 2, 26)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward CV + OOS optimizer")
    parser.add_argument("--config",    default="config/config_8am.yaml")
    parser.add_argument("--folds",     type=int, default=4,  help="Number of CV folds")
    parser.add_argument("--top-n",     type=int, default=5,  help="Top-N combos per fold for voting")
    parser.add_argument("--min-trades",type=int, default=20, help="Min trades to count a combo")
    parser.add_argument("--n-workers", type=int, default=0,  help="Worker processes (0=all cores)")
    parser.add_argument("--out-dir",   default="results",    help="Directory for CSV output")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    zip_path = config["backtest"]["data_file"]
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load all bars once
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  ORB Options Optimizer — Walk-Forward CV + OOS")
    print("=" * 70)
    print(f"\nZip  : {zip_path}")
    print(f"CV   : {CV_START} -> {CV_END}  ({args.folds} folds)")
    print(f"OOS  : {OOS_START} -> {OOS_END}  (held out)")
    print()

    loader = DatabentoLoader(zip_path)

    print("Loading CV bars...", flush=True)
    t0 = time.time()
    cv_days = list(loader.iter_days(start_date=CV_START, end_date=CV_END))
    print(f"  {len(cv_days)} day-symbol chunks in {time.time()-t0:.1f}s")

    print("Loading OOS bars...", flush=True)
    t0 = time.time()
    oos_days = list(loader.iter_days(start_date=OOS_START, end_date=OOS_END))
    print(f"  {len(oos_days)} day-symbol chunks in {time.time()-t0:.1f}s\n")

    # -----------------------------------------------------------------------
    # 2. Walk-forward CV
    # -----------------------------------------------------------------------
    opt = GridSearchOptimizer(base_config=config, zip_path=zip_path)

    best_params, fold_results = opt.run_walk_forward_cv(
        all_days   = cv_days,
        param_grid = DEFAULT_GRID,
        n_folds    = args.folds,
        top_n      = args.top_n,
        min_trades = args.min_trades,
        n_workers  = args.n_workers,
    )

    # Save per-fold top-val rows to CSV
    for fr in fold_results:
        fold_rows = fr.get("top_val_params", [])
        if fold_rows:
            path = out_dir / f"fold_{fr['fold']}_val_top{args.top_n}.csv"
            opt.save_csv(fold_rows, str(path))

    # -----------------------------------------------------------------------
    # 3. OOS final test on winning params
    # -----------------------------------------------------------------------
    oos_report: dict | None = None
    if best_params:
        print(f"\nRunning OOS test with best params...")
        oos_report = opt.eval_on_oos(best_params, oos_days)
        oos_path = out_dir / "oos_result.csv"
        opt.save_csv([{**best_params, **oos_report}], str(oos_path))

    # -----------------------------------------------------------------------
    # 4. Summary
    # -----------------------------------------------------------------------
    opt.print_cv_summary(fold_results, best_params, oos_report)

    # Also run a full-CV-window grid search with the best params for reference table
    if best_params:
        print("\nFull CV-window backtest with winning params (reference):")
        full_report = opt.eval_on_oos(best_params, cv_days)
        print(f"  trades={full_report.get('trades')}  "
              f"sharpe={full_report.get('sharpe', 0):.2f}  "
              f"WR={full_report.get('win_rate', 0)*100:.1f}%  "
              f"total_pnl=${full_report.get('total_pnl_usd', 0):,.0f}")

    # -----------------------------------------------------------------------
    # 5. Save optimized params to config file
    # -----------------------------------------------------------------------
    if best_params:
        opt_path = str(out_dir / "optimized_params.yaml")
        opt.save_optimized_config(
            best_params,
            oos_report=oos_report,
            fold_results=fold_results,
            output_path=opt_path,
        )


if __name__ == "__main__":
    main()
