"""
Walk-forward cross-validation optimizer for src_v2 backtest.

Usage:
  python -m scripts.run_v2_wf_cv --config config/config_orb_em_ic_spx.yaml
  python -m scripts.run_v2_wf_cv --config config/config_orb_em_ic_spx.yaml --ratio 5 --folds 4 --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import SpyLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.wf_optimizer import WalkForwardOptimizer


# Directional grid (strategy: orb_em_directional)
PARAM_GRID = {
    "options.em_with_trend":          [0.5, 1.0],
    "options.width_with_trend":       [15, 20, 25],
    "options.tp_pct_of_credit":       [0.75, 0.85],
    "options.hold_put_breach":        [True],
    "options.min_em_pts":             [12, 15, 18, 20, 25],
    "strategy.midline_tolerance_pts": [1.0],
    "strategy.orb_range_filter.min_pts": [2.0],
}
# Combos = 2*3*2*1*5 = 60


def _strategy_cls(config):
    name = config.get("bot", {}).get("strategy", "orb_em_ic")
    if name == "orb_em_ic":
        from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy
        return OrbEmIronCondorStrategy
    if name == "orb_em_directional":
        from src_v2.strategies.orb_em_directional import OrbEmDirectionalStrategy
        return OrbEmDirectionalStrategy
    raise ValueError(f"Unsupported strategy: {name}")


def _format_params(params):
    return " ".join(f"{k.split('.')[-1]}={v}" for k, v in params.items())


def _format_metrics(m):
    return (
        f"trades={m['trades']:>3d} "
        f"WR={m['win_rate']*100:>5.1f}% "
        f"PnL=${m['total_pnl']:>+7.0f} "
        f"Sharpe={m['sharpe_annualised']:>6.2f} "
        f"PF={m['profit_factor']}"
    )


def main():
    parser = argparse.ArgumentParser(description="src_v2 Walk-Forward CV Optimizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--ratio", type=int, default=5, help="Train:val ratio")
    parser.add_argument("--folds", type=int, default=4, help="Number of CV folds")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K candidates per fold to validate")
    parser.add_argument("--min-trades", type=int, default=5,
                        help="Min trades for a combo to be considered (train)")
    parser.add_argument("--min-sharpe", type=float, default=None,
                        help="Min Sharpe for train combos (filter before top-K)")
    parser.add_argument("--objective", default="sharpe_annualised",
                        choices=["sharpe_annualised", "total_pnl", "profit_factor", "win_rate"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bt = config.get("backtest", {})
    data_file = bt.get("data_file", "data/spy_1m.csv")
    price_scale = bt.get("price_scale", 10.0)
    start = date.fromisoformat(args.start) if args.start else date.fromisoformat(bt.get("start_date", "2025-09-22"))
    end = date.fromisoformat(args.end) if args.end else date.fromisoformat(bt.get("end_date", "2026-03-20"))
    symbol = config.get("bot", {}).get("symbol", "SPX")

    cbbo_dir = bt.get("cbbo_cache_dir", "")
    print(f"Preloading CBBO cache from {cbbo_dir}...")
    mem = CbboStore.preload_cache(cbbo_dir)
    print(f"  {len(mem)} days")
    cbbo = CbboStore(cbbo_dir, preloaded=mem)

    print(f"Loading bars from {data_file}...")
    loader = SpyLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} trading days ({start} to {end})")

    n_combos = 1
    for vs in PARAM_GRID.values():
        n_combos *= len(vs)
    print(f"Grid: {n_combos} combos, WF-CV: ratio={args.ratio}:1, folds={args.folds}, top-k={args.top_k}\n")

    strategy_class = _strategy_cls(config)
    opt = WalkForwardOptimizer(
        base_config=config,
        strategy_cls=strategy_class,
        cbbo_store=cbbo,
        days=days,
        param_grid=PARAM_GRID,
        ratio=args.ratio,
        n_folds=args.folds,
        top_k=args.top_k,
        min_trades=args.min_trades,
        objective=args.objective,
        min_sharpe=args.min_sharpe,
    )

    start_time = time.time()

    def progress(msg):
        elapsed = time.time() - start_time
        print(f"  [{elapsed:>5.0f}s] {msg}")

    result = opt.run(progress_cb=progress)

    total_time = time.time() - start_time
    print(f"\nWF-CV done in {total_time:.0f}s\n")

    # Print per-fold results
    sep = "─" * 150
    print(sep)
    print(f"  Per-fold OOS results:")
    print(sep)
    for fr in result.fold_results:
        print(f"\n  Fold {fr.fold}: train={fr.train_start}..{fr.train_end} ({fr.n_train_days}d)  "
              f"val={fr.val_start}..{fr.val_end} ({fr.n_val_days}d)")
        for i, (p, m) in enumerate(zip(fr.top_params, fr.val_metrics), 1):
            print(f"    {i}. {_format_params(p):<100s}  val: {_format_metrics(m)}")

    # Print winner
    print(f"\n{sep}")
    print(f"  WINNING PARAMS (most top-K appearances, tie-break by mean val Sharpe)")
    print(sep)
    print(f"    {_format_params(result.best_params)}")
    print(f"    Appeared in top-{opt._top_k} on {result.best_appearances}/{len(result.fold_results)} folds")
    print(f"    Mean fold val Sharpe: {result.best_mean_val_sharpe:.2f}")
    print(f"    Mean fold val PnL: ${result.best_mean_val_pnl:+.0f}")
    print()
    print(f"  Aggregate OOS:")
    for k, v in result.aggregate.items():
        print(f"    {k}: {v}")
    print(sep)


if __name__ == "__main__":
    main()
