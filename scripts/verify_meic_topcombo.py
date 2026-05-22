#!/usr/bin/env python3
"""Verify the MEIC sweep's top combo: per-year P&L breakdown + stop-fill stress.

Runs the top combo (short_em_mult, wing_em_frac, stop_pct from CLI) over the
full period twice — once with the optimistic stop model (fill at the credit
trigger) and once with the honest model (fill at the actual CBBO spread mark at
the breach minute) — and breaks each down by calendar year.
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.engine import BacktestEngine
from src_v2.strategies.meic import MeicStrategy


def _by_year(results):
    """year -> (pnl, n_closed, n_stops). Daily-DD computed separately."""
    agg = defaultdict(lambda: [0.0, 0, 0])
    for r in results:
        if r.status != "closed":
            continue
        y = r.session_date.year
        agg[y][0] += r.realized_pnl_usd or 0.0
        agg[y][1] += 1
        if r.exit_reason == "short_stopped":
            agg[y][2] += 1
    return agg


def _max_daily_dd(results):
    daily = defaultdict(float)
    for r in results:
        if r.status == "closed":
            daily[r.session_date] += r.realized_pnl_usd or 0.0
    eq = peak = mdd = 0.0
    worst = 0.0
    for _, p in sorted(daily.items()):
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
        worst = min(worst, p)
    return mdd, worst


def _run(base, days, cbbo, sem, wef, sp, stop_exec):
    cfg = copy.deepcopy(base)
    cfg["strategy"]["short_em_mult"] = sem
    cfg["strategy"]["wing_em_frac"] = wef
    cfg["strategy"]["stop_pct_of_total_credit"] = sp
    cfg["options"]["stop_execution"] = stop_exec
    engine = BacktestEngine(cfg, MeicStrategy())
    engine._cbbo = cbbo
    return engine.run(days)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sem", type=float, default=1.0)
    p.add_argument("--wef", type=float, default=0.75)
    p.add_argument("--sp", type=float, default=0.70)
    args = p.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    bt = base["backtest"]
    start = date.fromisoformat(bt.get("start_date", "2022-05-11"))
    end = date.fromisoformat(bt.get("end_date", "2026-02-27"))

    print(f"Preloading {bt['cbbo_cache_dir']} ...")
    cbbo = CbboStore(bt["cbbo_cache_dir"],
                     preloaded=CbboStore.preload_cache(bt["cbbo_cache_dir"]))
    days = list(BarLoader(bt.get("data_file", "data/spy_1m.csv"),
                          price_scale=bt.get("price_scale", 10.0),
                          symbol=base.get("bot", {}).get("symbol", "SPX"))
                .iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} days; combo sem={args.sem} wef={args.wef} sp={args.sp}\n")

    for label, stop_exec in [("trigger      (optimistic: debit = credit trigger)", "trigger"),
                             ("short_trigger (short@trigger, sell long leg-by-leg)", "short_trigger"),
                             ("combo        (short@breach-ask, sell long at breach)", "combo"),
                             ("short_only   (short@breach-ask, hold long to EOD)", "short_only")]:
        results = _run(base, days, cbbo, args.sem, args.wef, args.sp, stop_exec)
        agg = _by_year(results)
        mdd, worst = _max_daily_dd(results)
        total = sum(v[0] for v in agg.values())
        print(f"=== {label} ===")
        print(f"  {'year':>5} {'pnl':>13} {'trades':>7} {'stops':>6} {'stop%':>6}")
        for y in sorted(agg):
            pnl, n, st = agg[y]
            print(f"  {y:>5} {pnl:>13,.0f} {n:>7} {st:>6} {100*st/max(n,1):>5.0f}%")
        print(f"  TOTAL {total:>13,.0f}   maxDD(daily)=${mdd:,.0f}  worstDay=${worst:,.0f}\n")


if __name__ == "__main__":
    main()
