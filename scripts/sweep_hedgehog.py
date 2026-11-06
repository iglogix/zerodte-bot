"""Hedge Hog robustness sweeps: fills, deltas, DTE, exit variants, laddering.

Runs the engine across a small named-variant grid and prints a comparison
table. Variants are deliberately few and structural (not an optimizer) —
the point is robustness, not parameter mining.

Usage:
    .venv/bin/python scripts/sweep_hedgehog.py [--cache data/hedgehog_eod/SPY]
        [--start 20200102] [--end 20261231] [--out results/hedgehog_spy]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from zerodte.backtest.hedgehog import (
    HedgeHogEngine,
    HedgeHogStore,
    HhParams,
    summarize,
)

BASE = HhParams()

VARIANTS: dict[str, HhParams] = {
    # fills (slippage realism)
    "base_k0.5": BASE,
    "fill_mid": replace(BASE, fill_k=0.0),
    "fill_nbbo": replace(BASE, fill_k=1.0),
    # hog placement
    "hog_5d": replace(BASE, hog_delta=0.05),
    "hog_10d": replace(BASE, hog_delta=0.10),
    "hog_dte75": replace(BASE, hog_dte_target=75),
    # single-cycle "1-1-1" comparison (hog in the hedge cycle, 45-60 DTE)
    "single_cycle_50": replace(
        BASE,
        hedge_dte_target=50, hedge_dte_min=45, hedge_dte_max=60,
        hog_dte_target=50, hog_dte_min=45, hog_dte_max=60,
    ),
    # exit-variant ablations
    "no_hog_tp": replace(BASE, hog_tp_frac=9.9),
    "no_vert_tp": replace(BASE, vert_tp_frac=9.9),
    "no_fast_breach": replace(BASE, fast_breach_days=0),
    "hold_to_3dte": replace(BASE, vert_close_dte=3),
    # hedge ablation: identical entries/triggers, only the naked put traded
    "hog_only": replace(BASE, trade_hedge=False),
    # quality gate ablation
    "no_pop_gate": replace(BASE, pop_min=0.0),
    "pop_78": replace(BASE, pop_min=0.78),
    # laddered portfolio (one new every 3 days, up to 4 concurrent)
    "ladder_4x3d": replace(BASE, max_concurrent=4, ladder_every_days=3),
}

COLS = [
    ("trades", "{:d}"),
    ("win_rate", "{:.1%}"),
    ("total_pnl_usd", "{:+,.0f}"),
    ("avg_pnl_usd", "{:+.0f}"),
    ("avg_win_usd", "{:+.0f}"),
    ("avg_loss_usd", "{:+.0f}"),
    ("worst_trade_usd", "{:+,.0f}"),
    ("profit_factor", "{:.2f}"),
    ("avg_days_held", "{:.1f}"),
    ("sharpe_annualised", "{:.2f}"),
    ("max_drawdown_usd", "{:,.0f}"),
    ("ann_return_on_avg_margin", "{:.1%}"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/hedgehog_eod/SPY")
    ap.add_argument("--start", type=int, default=20200102)
    ap.add_argument("--end", type=int, default=20261231)
    ap.add_argument("--out", default="results/hedgehog_spy")
    ap.add_argument("--only", default=None, help="comma list of variant names")
    args = ap.parse_args()

    names = list(VARIANTS) if not args.only else args.only.split(",")
    store = HedgeHogStore(args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}
    for name in names:
        engine = HedgeHogEngine(store, VARIANTS[name])
        trades = engine.run(start=args.start, end=args.end)
        stats = summarize(trades, engine.daily_equity)
        all_stats[name] = stats
        print(f"  {name}: {stats.get('trades', 0)} trades", flush=True)

    header = f"{'variant':<18}" + "".join(f"{c:>14}" for c, _ in COLS)
    print("\n" + header)
    print("-" * len(header))
    for name in names:
        s = all_stats[name]
        if s.get("trades", 0) == 0:
            print(f"{name:<18}{'no trades':>14}")
            continue
        row = f"{name:<18}"
        for col, fmt in COLS:
            v = s.get(col)
            try:
                row += f"{fmt.format(v):>14}"
            except (ValueError, TypeError):
                row += f"{str(v):>14}"
        print(row)

    with open(out_dir / "sweep.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nsweep -> {out_dir / 'sweep.json'}")


if __name__ == "__main__":
    main()
