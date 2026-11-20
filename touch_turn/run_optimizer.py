"""
Touch & Turn Scalper — Optimizer Runner

Usage:
  python -m touch_turn.run_optimizer
  python -m touch_turn.run_optimizer --config touch_turn/config.yaml
  python -m touch_turn.run_optimizer --config touch_turn/config_nvda.yaml
  python -m touch_turn.run_optimizer --n-workers 4 --folds 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_8am.backtest.data_loader import DatabentoLoader
from touch_turn.backtest.equity_loader import EquityLoader
from touch_turn.backtest.optimizer import DEFAULT_GRID, TouchTurnOptimizer


def main():
    parser = argparse.ArgumentParser(description="Touch & Turn optimizer")
    parser.add_argument("--config", default="touch_turn/config.yaml")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    # OOS split
    parser.add_argument("--oos-start", default=None,
                        help="OOS start date (default: auto from config end_date - 25%%)")
    args = parser.parse_args()

    # Derive output dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stem = Path(args.config).stem.removeprefix("config_").strip("_") or "touch_turn"
        out_dir = Path(f"results/{stem}_opt")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "optimizer.log", mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = config["backtest"]["data_file"]
    data_source = config.get("backtest", {}).get("data_source", "databento")
    start = date.fromisoformat(config["backtest"]["start_date"])
    end = date.fromisoformat(config["backtest"]["end_date"])

    # Auto OOS split: last 25% of data
    if args.oos_start:
        oos_start = date.fromisoformat(args.oos_start)
    else:
        total_days_approx = (end - start).days
        oos_offset = int(total_days_approx * 0.75)
        oos_start = start + __import__("datetime").timedelta(days=oos_offset)

    symbol = config.get("bot", {}).get("symbol", "ES")
    print("=" * 60)
    print(f"  Touch & Turn Optimizer — {symbol}")
    print("=" * 60)
    print(f"  Data   : {data_path}")
    print(f"  CV     : {start} → {oos_start}")
    print(f"  OOS    : {oos_start} → {end}")
    print(f"  Folds  : {args.folds}")
    print(f"  Grid   : {sum(1 for _ in __import__('itertools').product(*DEFAULT_GRID.values()))} combos")
    print()

    # Load data
    print(f"Loading {symbol} bars...", flush=True)
    t0 = time.time()
    if data_source == "equity":
        loader = EquityLoader(data_path)
    else:
        loader = DatabentoLoader(data_path)

    all_days = list(loader.iter_days(start_date=start, end_date=end))
    cv_days = [d for d in all_days if d.session_date < oos_start]
    oos_days = [d for d in all_days if d.session_date >= oos_start]
    print(f"  CV: {len(cv_days)} days, OOS: {len(oos_days)} days ({time.time()-t0:.1f}s)\n")

    # Run WF-CV
    opt = TouchTurnOptimizer(config)
    best_params, fold_results = opt.run_walk_forward_cv(
        all_days=cv_days,
        param_grid=DEFAULT_GRID,
        n_folds=args.folds,
        top_n=args.top_n,
        min_trades=args.min_trades,
        n_workers=args.n_workers,
    )

    # Save fold results
    for fr in fold_results:
        rows = fr.get("top_val_params", [])
        if rows:
            opt.save_csv(rows, str(out_dir / f"fold_{fr['fold']}_val.csv"))

    # OOS test
    if best_params and oos_days:
        print("\nRunning OOS test with best params...")
        oos_metrics = opt.eval_on_oos(best_params, oos_days)
        print("\n  OOS Results:")
        print(f"    Trades       : {oos_metrics['trades']}")
        print(f"    Total P&L    : ${oos_metrics['total_pnl_usd']:,.2f}")
        print(f"    Sharpe       : {oos_metrics['sharpe']:.2f}")
        print(f"    Win Rate     : {oos_metrics['win_rate']*100:.1f}%")
        print(f"    Profit Factor: {oos_metrics['profit_factor']:.2f}")
        print(f"    Max DD       : ${oos_metrics['max_dd']:,.2f}")

        opt.save_csv([{**best_params, **oos_metrics}], str(out_dir / "oos_result.csv"))

        # Save best config
        best_config_path = out_dir / "optimized_config.yaml"
        optimized = {**config}
        optimized["strategy"]["fib_tp_long"] = best_params["fib_tp_long"]
        optimized["strategy"]["fib_tp_short"] = best_params["fib_tp_short"]
        optimized["strategy"]["atr_min_pct"] = best_params["atr_min_pct"]
        optimized["strategy"]["sl_ratio"] = best_params["sl_ratio"]
        # window_mins → trading_window_end
        total_mins = 30 + best_params["window_mins"]
        h, m = 9 + total_mins // 60, total_mins % 60
        optimized["strategy"]["trading_window_end"] = f"{h:02d}:{m:02d}"
        with open(best_config_path, "w") as f:
            yaml.dump(optimized, f, default_flow_style=False, sort_keys=False)
        print(f"\n  Optimized config: {best_config_path}")
    elif not best_params:
        print("\nNo profitable parameter combination found.")

    print()


if __name__ == "__main__":
    main()
