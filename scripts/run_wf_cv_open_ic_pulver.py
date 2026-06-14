"""Walk-forward CV for the Pulver-optimized open-IC region.

Centers the grid on the in-sample robust plateau (em_mult~0.25, wider wings,
50% TP, no stop, NO calendar gates) found by sweep_open_ic_pulver. This
validates the *method* (periodic re-optimization picks a good cell each fold)
out-of-sample, per docs/wf-cv.md.

Usage:
  python -m scripts.run_wf_cv_open_ic_pulver --config config/config_open_ic_pulver.yaml \
      --ratio 4 --folds 5 --top-k 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.wf_optimizer import WalkForwardOptimizer
from src_v2.strategies.open_ic import OpenIcStrategy


# Robust plateau from the full-period sweep (in-sample). sl_scope=total per the
# config-comment correction (per_side was a per-minute-sim artifact). No gates
# (gates lowered IS Sharpe 2.05 -> 1.92 and PnL). em_mult is the dominant axis.
import os as _os
_SCOPE = _os.environ.get("WF_SL_SCOPE", "total")  # "total" or "per_side"
PARAM_GRID = {
    "options.strike_mode": ["em_mult"],
    "options.em_mult": [0.20, 0.25, 0.30],
    "options.wing_width": [20, 25, 30],
    "options.sl_scope": [_SCOPE],
    "options.sl_mult_of_credit": [1.5, 2.0],
    "options.tp_pct_of_credit": [0.25, 0.50],
}
# Combos = 1*3*3*1*2*2 = 36 (NO calendar gates -- Pulver config has them off)


def _fmt_p(p):
    return " ".join(f"{k.split('.')[-1]}={v}" for k, v in p.items())


def _fmt_m(m):
    return (f"trades={m['trades']:>3d} WR={m['win_rate']*100:>5.1f}% "
            f"PnL=${m['total_pnl']:>+8.0f} Sharpe={m['sharpe_annualised']:>6.2f} "
            f"PF={m['profit_factor']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_open_ic_pulver.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-trades", type=int, default=15)
    ap.add_argument("--objective", default="sharpe_annualised",
                    choices=["sharpe_annualised", "total_pnl", "profit_factor", "win_rate"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    bt = config.get("backtest", {})
    start = date.fromisoformat(args.start or bt.get("start_date", "2022-05-11"))
    end = date.fromisoformat(args.end or bt.get("end_date", "2026-06-11"))
    symbol = config.get("bot", {}).get("symbol", "SPX")
    cbbo_dir = bt.get("cbbo_cache_dir", "data/cbbo_cache")

    print(f"Preloading CBBO {cbbo_dir} ({start} -> {end})...")
    mem = CbboStore.preload_cache(cbbo_dir, start=start, end=end)
    cbbo = CbboStore(cbbo_dir, preloaded=mem)
    loader = BarLoader(bt.get("data_file"), price_scale=bt.get("price_scale", 10.0),
                       symbol=symbol)
    days = list(loader.iter_days(start_date=start, end_date=end))
    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)
    print(f"  {len(mem)} cbbo days, {len(days)} bar-days. Grid={n_combos} combos, "
          f"ratio={args.ratio}:1 folds={args.folds} top-k={args.top_k}\n")

    opt = WalkForwardOptimizer(
        base_config=config, strategy_cls=OpenIcStrategy, cbbo_store=cbbo, days=days,
        param_grid=PARAM_GRID, ratio=args.ratio, n_folds=args.folds, top_k=args.top_k,
        min_trades=args.min_trades, objective=args.objective, n_workers=args.workers,
    )
    t0 = time.time()
    result = opt.run(progress_cb=lambda m: print(f"  [{time.time()-t0:>5.0f}s] {m}"))
    print(f"\nWF-CV done in {time.time()-t0:.0f}s\n")

    sep = "-" * 140
    print(sep + "\n  Per-fold OOS results:\n" + sep)
    for fr in result.fold_results:
        print(f"\n  Fold {fr.fold}: train={fr.train_start}..{fr.train_end} ({fr.n_train_days}d)  "
              f"val={fr.val_start}..{fr.val_end} ({fr.n_val_days}d)")
        for i, (p, m) in enumerate(zip(fr.top_params, fr.val_metrics), 1):
            print(f"    {i}. {_fmt_p(p):<80s}  val: {_fmt_m(m)}")

    print(f"\n{sep}\n  WINNING PARAMS (robust selection)\n{sep}")
    print(f"    {_fmt_p(result.best_params)}")
    print(f"    Top-{opt._top_k} appearances: {result.best_appearances}/{len(result.fold_results)} folds")
    print(f"    Profitable folds:   {result.best_n_profitable}/{len(result.fold_results)}")
    print(f"    Worst fold PnL:     ${result.best_min_pnl:+.0f}")
    print(f"    Mean fold val Sharpe: {result.best_mean_val_sharpe:.2f}")
    print(f"    Mean fold val PnL:    ${result.best_mean_val_pnl:+.0f}")
    print("\n  Aggregate OOS:")
    for k, v in result.aggregate.items():
        print(f"    {k}: {v}")
    print(sep)


if __name__ == "__main__":
    main()
