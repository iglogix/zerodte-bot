"""Analytical Jade Lizard evaluator on a CBBO cache.

Headline = mid-price fills, zero cost, exit valued at the combined bid/ask mid at
the exit anchor. Also prints a with-costs sensitivity and a spot-intrinsic
cross-check. Per-trade rows are written to results/jade_lizard/.

Phase 0 (free) smoke test on the true-SPX 5-anchor cache:
    .venv/bin/python -m scripts.analytical_jl_eval \\
        --cache data/cbbo_cache_5anchor --entry 09:35 \\
        --anchors 09:35,11:00,13:00,15:30,15:59 \\
        --start 2025-09-01 --end 2026-02-27

Phase 3 (real) on the backfilled 10-min cache:
    .venv/bin/python -m scripts.analytical_jl_eval \\
        --cache data/cbbo_cache_jl_10min --entry 09:30 \\
        --start 2022-05-11 --end 2026-03-30
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest import jl_core as jl


def load_by_date(cache_dir: Path, bars_path: str, start: date, end: date) -> Dict:
    """Build {date: {quotes, spot_close}} for days that have a cache pickle."""
    loader = BarLoader(bars_path, price_scale=10.0, symbol="SPX")
    by_date: Dict = {}
    for day in loader.iter_days(start_date=start, end_date=end):
        d = day.session_date
        pkl = Path(cache_dir) / f"{d.isoformat()}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            quotes = pickle.load(f)
        b_eod = next((b for b in reversed(day.bars)
                      if b["ts_et"].time() >= time(15, 55)), None)
        by_date[d] = {"quotes": quotes,
                      "spot_close": b_eod["close"] if b_eod else None}
    return by_date


def _print_summary(label: str, res: Dict) -> None:
    if res["n"] == 0:
        print(f"\n[{label}] NO TRADES  skips={res.get('skips')}")
        return
    print(f"\n[{label}]  n={res['n']}  WR={res['wr']:.1f}%  total=${res['total']:+,.0f}  "
          f"mean=${res['mean']:+.1f}  Sharpe={res['sharpe_ann']:.2f}")
    print(f"  avg_credit=${res['avg_credit']:.2f}  maxDD=${res['max_dd']:+,.0f}  "
          f"max_consec_L={res['max_consec_L']}  avg_W=${res['avg_win']:+.0f}  "
          f"avg_L=${res['avg_loss']:+.0f}  worst=${res['worst']:+,.0f}  best=${res['best']:+,.0f}")
    print(f"  exits={res['exits']}  pct_upside_riskfree={res['pct_upside_riskfree']:.1f}%  "
          f"skips={res['skips']}")


def _write_trades(out_dir: Path, res: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trades_headline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "kind", "exit_anchor", "em", "forward",
                    "short_put", "long_put", "short_call", "long_call",
                    "total_credit", "exit_mark", "pnl", "upside_riskfree"])
        for r in res["results"]:
            sp, lp, sc, lc = r["strikes"]
            w.writerow([r["date"], r["kind"], r["exit_anchor"], f"{r['em']:.2f}",
                        f"{r['forward']:.2f}", sp, lp, sc, lc,
                        f"{r['total_credit']:.2f}", f"{r['exit_mark']:.2f}",
                        f"{r['pnl']:.2f}", r["upside_riskfree"]])
    summary = {k: v for k, v in res.items() if k != "results"}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main() -> int:
    p = argparse.ArgumentParser(description="Analytical Jade Lizard evaluator")
    p.add_argument("--cache", default="data/cbbo_cache_jl_10min")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-03-30")
    p.add_argument("--entry", default="09:30")
    p.add_argument("--anchors", default="",
                   help="comma list; default = 10-min grid from --entry to 16:00")
    p.add_argument("--em-mult", type=float, default=0.25)
    p.add_argument("--put-width", type=float, default=30.0)
    p.add_argument("--call-width", type=float, default=10.0)
    p.add_argument("--tp-pct", type=float, default=0.50)
    p.add_argument("--slippage", type=float, default=0.10,
                   help="with-costs slippage per spread (headline always 0)")
    p.add_argument("--commission", type=float, default=1.0,
                   help="with-costs commission per contract per leg")
    p.add_argument("--out", default="results/jade_lizard")
    args = p.parse_args()

    anchors = ([a.strip() for a in args.anchors.split(",")] if args.anchors
               else jl.make_anchors(args.entry, "16:00", 10))
    by_date = load_by_date(Path(args.cache), args.bars,
                           date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"Days with cache: {len(by_date)}  entry={args.entry}  "
          f"anchors={len(anchors)} ({anchors[0]}..{anchors[-1]})")

    base = dict(entry_anchor=args.entry, anchors=anchors,
                em_mult_put=args.em_mult, em_mult_call=args.em_mult,
                put_width=args.put_width, call_width=args.call_width,
                tp_pct=args.tp_pct)

    headline = jl.evaluate(by_date, jl.JlConfig(**base))
    _print_summary("HEADLINE (mid, zero-cost)", headline)

    costs = jl.evaluate(by_date, jl.JlConfig(
        **base, slippage_per_spread=args.slippage,
        commission_per_contract_per_leg=args.commission))
    _print_summary(f"WITH COSTS (slip={args.slippage}/spread, comm=${args.commission}/leg)", costs)

    intrinsic = jl.evaluate(by_date, jl.JlConfig(**base, exit_mode="intrinsic"))
    _print_summary("CROSS-CHECK (expiry at spot-intrinsic)", intrinsic)

    _write_trades(Path(args.out), headline)
    print(f"\nPer-trade rows -> {args.out}/trades_headline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
