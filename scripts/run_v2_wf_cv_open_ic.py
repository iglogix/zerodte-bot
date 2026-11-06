"""
Walk-forward cross-validation optimizer for the 0DTE open IC strategy.

Sweeps short delta x wing width x entry time x TP/breach handling.

Usage:
  python -m scripts.run_v2_wf_cv_open_ic --config config/config_spx_0dte_open_ic.yaml
  python -m scripts.run_v2_wf_cv_open_ic --config config/config_spx_0dte_open_ic.yaml --ratio 5 --folds 4 --top-k 5
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.wf_optimizer import WalkForwardOptimizer
from zerodte.strategies.open_ic import OpenIcStrategy


def resolve_strategy_class(dotted_path: str | None):
    """Resolve a strategy class from a dotted import path.

    Defaults to OpenIcStrategy when path is None or empty (backcompat).
    """
    if not dotted_path:
        return OpenIcStrategy
    mod_path, _, cls_name = dotted_path.rpartition(".")
    if not mod_path:
        raise ValueError(f"strategy.class must be a dotted path; got {dotted_path!r}")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


# Short delta + wing width are the primary axes; secondary axes capture
# tail-risk/exit handling. Trim further if total combos balloon.
PARAM_GRID = {
    # v7 (2026-05-13): per-side SL + em_mult strikes.
    "options.strike_mode": ["em_mult"],
    "options.em_mult": [0.20, 0.25, 0.30, 0.35],
    "options.wing_width": [20, 25, 30],
    "options.sl_mult_of_credit": [1.5, 2.0, 2.5],
    "options.sl_scope": ["per_side"],
    "options.tp_pct_of_credit": [0.25, 0.50],
    "strategy.skip_fomc_next_day": [True],
    "strategy.skip_weekdays": [["Fri"]],
}
# Combos = 1*4*3*3*1*2*1*1 = 72


def _format_params(params):
    return " ".join(f"{k.split('.')[-1]}={v}" for k, v in params.items())


def _format_metrics(m):
    return (
        f"trades={m['trades']:>3d} "
        f"WR={m['win_rate']*100:>5.1f}% "
        f"PnL=${m['total_pnl']:>+8.0f} "
        f"Sharpe={m['sharpe_annualised']:>6.2f} "
        f"PF={m['profit_factor']}"
    )


def main():
    parser = argparse.ArgumentParser(description="0DTE open IC Walk-Forward CV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--ratio", type=int, default=5)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--min-sharpe", type=float, default=None)
    parser.add_argument(
        "--objective",
        default="sharpe_annualised",
        choices=["sharpe_annualised", "total_pnl", "profit_factor", "win_rate"],
    )
    parser.add_argument("--require-all-folds-profitable", action="store_true")
    default_workers = max(1, (os.cpu_count() or 2) - 2)
    parser.add_argument("--workers", type=int, default=default_workers)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bt = config.get("backtest", {})
    data_file = bt.get("data_file", "data/spy_1m.csv")
    price_scale = bt.get("price_scale", 10.0)
    start = (
        date.fromisoformat(args.start)
        if args.start
        else date.fromisoformat(bt.get("start_date", "2022-05-11"))
    )
    end = (
        date.fromisoformat(args.end)
        if args.end
        else date.fromisoformat(bt.get("end_date", "2026-02-28"))
    )
    symbol = config.get("bot", {}).get("symbol", "SPX")

    cbbo_dir = bt.get("cbbo_cache_dir", "data/cbbo_cache")
    print(f"Preloading CBBO cache from {cbbo_dir} ({start} → {end})...")
    mem = CbboStore.preload_cache(cbbo_dir, start=start, end=end)
    print(f"  {len(mem)} days")
    cbbo = CbboStore(cbbo_dir, preloaded=mem)

    print(f"Loading bars from {data_file}...")
    loader = BarLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} trading days ({start} to {end})")

    n_combos = 1
    for vs in PARAM_GRID.values():
        n_combos *= len(vs)
    print(
        f"Grid: {n_combos} combos, WF-CV: ratio={args.ratio}:1, folds={args.folds}, "
        f"top-k={args.top_k}, workers={args.workers}\n"
    )

    opt = WalkForwardOptimizer(
        base_config=config,
        strategy_cls=resolve_strategy_class(config.get("strategy", {}).get("class")),
        cbbo_store=cbbo,
        days=days,
        param_grid=PARAM_GRID,
        ratio=args.ratio,
        n_folds=args.folds,
        top_k=args.top_k,
        min_trades=args.min_trades,
        objective=args.objective,
        min_sharpe=args.min_sharpe,
        require_all_folds_profitable=args.require_all_folds_profitable,
        n_workers=args.workers,
    )

    start_time = time.time()

    def progress(msg):
        elapsed = time.time() - start_time
        print(f"  [{elapsed:>5.0f}s] {msg}")

    result = opt.run(progress_cb=progress)

    total_time = time.time() - start_time
    print(f"\nWF-CV done in {total_time:.0f}s\n")

    sep = "─" * 150
    print(sep)
    print("  Per-fold OOS results:")
    print(sep)
    for fr in result.fold_results:
        print(
            f"\n  Fold {fr.fold}: train={fr.train_start}..{fr.train_end} ({fr.n_train_days}d)  "
            f"val={fr.val_start}..{fr.val_end} ({fr.n_val_days}d)"
        )
        for i, (p, m) in enumerate(zip(fr.top_params, fr.val_metrics), 1):
            print(f"    {i}. {_format_params(p):<100s}  val: {_format_metrics(m)}")

    print(f"\n{sep}")
    print("  WINNING PARAMS (robust selection)")
    print(sep)
    print(f"    {_format_params(result.best_params)}")
    print(
        f"    Top-{opt._top_k} appearances:  {result.best_appearances}/{len(result.fold_results)} folds"
    )
    print(f"    Profitable folds:    {result.best_n_profitable}/{len(result.fold_results)}")
    print(f"    Worst fold PnL:      ${result.best_min_pnl:+.0f}")
    print(f"    Mean fold val Sharpe: {result.best_mean_val_sharpe:.2f}")
    print(f"    Mean fold val PnL: ${result.best_mean_val_pnl:+.0f}")
    print()
    print("  Aggregate OOS:")
    for k, v in result.aggregate.items():
        print(f"    {k}: {v}")
    print(sep)


if __name__ == "__main__":
    main()
