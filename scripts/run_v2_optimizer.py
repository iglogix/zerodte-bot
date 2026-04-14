"""
Grid search optimizer for src_v2 backtest.

Edit PARAM_GRID below to change search space.

Usage:
  python -m scripts.run_v2_optimizer --config config/config_orb_em_ic_spx.yaml
  python -m scripts.run_v2_optimizer --config config/config_orb_em_ic_spx.yaml --top 20 --objective total_pnl
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import SpyLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.optimizer import GridSearchOptimizer, OptResult


# ---------------------------------------------------------------------------
# Parameter grid — edit values to change search space
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "options.em_with_trend":          [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "options.width_with_trend":       [5, 10, 15, 20, 25],
    "options.tp_pct_of_credit":       [0.50, 0.65, 0.75, 0.85],
    "options.hold_put_breach":        [False, True],
    "options.min_em_pts":             [0, 5, 10, 12, 15, 18, 20, 25, 30],
    "strategy.midline_tolerance_pts": [1.0],
    "strategy.orb_range_filter.min_pts": [2.0],
}
# Combos = 6*5*4*2*9 = 2160 (thorough directional sweep)


def _strategy_cls(config):
    name = config.get("bot", {}).get("strategy", "orb_em_ic")
    if name == "orb_em_ic":
        from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy
        return OrbEmIronCondorStrategy
    if name == "orb_em_directional":
        from src_v2.strategies.orb_em_directional import OrbEmDirectionalStrategy
        return OrbEmDirectionalStrategy
    raise ValueError(f"Unsupported strategy: {name}")


def _results_dir(config_path: str) -> Path:
    stem = Path(config_path).stem
    name = stem.removeprefix("config_").strip("_")
    return Path(f"results/{name or 'default'}")


def _format_params(params):
    return " ".join(f"{k.split('.')[-1]}={v}" for k, v in params.items())


def _format_metrics(m):
    pf = m["profit_factor"]
    pf_str = "inf" if pf == "inf" or pf == float("inf") else f"{pf:.2f}"
    return (
        f"trades={m['trades']:>3d} "
        f"WR={m['win_rate']*100:>5.1f}% "
        f"PnL=${m['total_pnl']:>+7.0f} "
        f"Sharpe={m['sharpe_annualised']:>6.2f} "
        f"PF={pf_str:>5s} "
        f"DD=${m['max_drawdown']:>6.0f}"
    )


def main():
    parser = argparse.ArgumentParser(description="src_v2 Grid Search Optimizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--top", type=int, default=20, help="Print top-N results")
    parser.add_argument("--objective", default="sharpe_annualised",
                        choices=["sharpe_annualised", "total_pnl", "profit_factor", "win_rate"])
    parser.add_argument("--min-trades", type=int, default=30,
                        help="Filter out combos with fewer than N trades (avoids overfit)")
    parser.add_argument("--min-sharpe", type=float, default=None,
                        help="Filter out combos with Sharpe below this floor")
    parser.add_argument("--min-win-rate", type=float, default=None,
                        help="Filter out combos with win rate below this floor (0-1)")
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
    for values in PARAM_GRID.values():
        n_combos *= len(values)
    print(f"Grid: {n_combos} combos, objective={args.objective}\n")

    strategy_class = _strategy_cls(config)
    opt = GridSearchOptimizer(
        base_config=config,
        strategy_cls=strategy_class,
        cbbo_store=cbbo,
        days=days,
        param_grid=PARAM_GRID,
        objective=args.objective,
    )

    # Progress reporting
    start_time = time.time()

    def progress(i, total, result):
        elapsed = time.time() - start_time
        eta = elapsed / i * (total - i) if i else 0
        if i % 10 == 0 or i == total:
            print(f"  [{i:>4d}/{total}] elapsed={elapsed:>5.0f}s eta={eta:>5.0f}s  "
                  f"last: {_format_params(result.params)}  {_format_metrics(result.metrics)}")

    results = opt.run(progress_cb=progress)

    total_time = time.time() - start_time
    print(f"\nGrid search done in {total_time:.0f}s")

    # Apply filters
    filtered = [r for r in results if r.metrics["trades"] >= args.min_trades]
    if args.min_sharpe is not None:
        filtered = [r for r in filtered if (r.metrics["sharpe_annualised"] or 0) >= args.min_sharpe]
    if args.min_win_rate is not None:
        filtered = [r for r in filtered if (r.metrics["win_rate"] or 0) >= args.min_win_rate]
    filter_desc = f"trades>={args.min_trades}"
    if args.min_sharpe is not None:
        filter_desc += f", sharpe>={args.min_sharpe}"
    if args.min_win_rate is not None:
        filter_desc += f", WR>={args.min_win_rate}"
    print(f"  {len(filtered)}/{len(results)} combos with {filter_desc}")

    # Print top-N
    print(f"\n{'─' * 150}")
    print(f"  Top {args.top} combos by {args.objective} (min {args.min_trades} trades):")
    print(f"{'─' * 150}")
    for i, r in enumerate(filtered[:args.top], 1):
        print(f"  {i:>2d}. {_format_params(r.params):<110s}  {_format_metrics(r.metrics)}")
    print(f"{'─' * 150}")

    # Export all results to CSV
    out_dir = _results_dir(args.config)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "optimizer_results.csv"
    _export_csv(results, csv_path)
    print(f"\nAll results → {csv_path}")


def _export_csv(results, path):
    if not results:
        return
    param_keys = list(results[0].params.keys())
    metric_keys = list(results[0].metrics.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(param_keys + metric_keys + ["seconds"])
        for r in results:
            row = [r.params[k] for k in param_keys] + [r.metrics[k] for k in metric_keys] + [round(r.seconds, 2)]
            writer.writerow(row)


if __name__ == "__main__":
    main()
