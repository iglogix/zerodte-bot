"""Run the Hedge Hog backtest over the EOD chain cache.

Usage:
    .venv/bin/python scripts/backtest_hedgehog.py [--cache data/hedgehog_eod/SPY]
        [--start 20200102] [--end 20261231] [--fill-k 0.5] [--qty 1]
        [--max-concurrent 1] [--ladder-days 1] [--out results/hedgehog_spy]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

from src_v2.backtest.hedgehog import (
    HedgeHogEngine,
    HedgeHogStore,
    HhParams,
    summarize,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/hedgehog_eod/SPY")
    ap.add_argument("--start", type=int, default=20200102)
    ap.add_argument("--end", type=int, default=20261231)
    ap.add_argument("--fill-k", type=float, default=0.5)
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--ladder-days", type=int, default=1)
    ap.add_argument("--hog-delta", type=float, default=0.07)
    ap.add_argument("--hog-dte", type=int, default=100)
    ap.add_argument("--hedge-dte", type=int, default=30)
    ap.add_argument("--pop-min", type=float, default=0.65)
    ap.add_argument("--out", default="results/hedgehog_spy")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    params = HhParams(
        fill_k=args.fill_k,
        qty=args.qty,
        max_concurrent=args.max_concurrent,
        ladder_every_days=args.ladder_days,
        hog_delta=args.hog_delta,
        hog_dte_target=args.hog_dte,
        hedge_dte_target=args.hedge_dte,
        pop_min=args.pop_min,
    )
    store = HedgeHogStore(args.cache)
    engine = HedgeHogEngine(store, params)
    trades = engine.run(start=args.start, end=args.end)
    stats = summarize(trades, engine.daily_equity)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"trades_{args.tag}.csv"
    cols = [f.name for f in fields(trades[0])] if trades else []
    if trades:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[c for c in cols if c != "daily_marks"])
            w.writeheader()
            for t in trades:
                row = {c: getattr(t, c) for c in cols if c != "daily_marks"}
                w.writerow(row)

    skip_counts: dict[str, int] = {}
    for s in engine.skips:
        skip_counts[s.reason] = skip_counts.get(s.reason, 0) + 1

    print(json.dumps(stats, indent=2, default=str))
    print("skips:", json.dumps(skip_counts, indent=2))
    print(f"trades -> {csv_path}")

    with open(out_dir / f"summary_{args.tag}.json", "w") as f:
        json.dump({"params": vars(args), "stats": stats, "skips": skip_counts},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
