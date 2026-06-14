"""Full-period in-sample landscape sweep for the open-IC (Pulver) family.

Maps Sharpe / PnL across em_mult x wing_width x tp x sl, optionally with
calendar gates, to find the direction that maximizes Sharpe relative to
Pulver's as-taught recipe. IN-SAMPLE ONLY — use WF-CV to validate winners.

Usage: python -m scripts.sweep_open_ic_pulver [--gates]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.strategies.open_ic import OpenIcStrategy


def _set(cfg, dotted, v):
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v


def _daily_stats(results):
    by_day = defaultdict(lambda: {"pnl": 0.0, "legs": 0})
    for r in results:
        if r.status == "closed":
            by_day[r.session_date]["pnl"] += r.realized_pnl_usd or 0.0
            by_day[r.session_date]["legs"] += 1
    pnls = [v["pnl"] for v in by_day.values() if v["legs"] == 2]
    n = len(pnls)
    if n < 2:
        return None
    total = sum(pnls)
    mean = total / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1))
    sharpe = (mean / sd * math.sqrt(252)) if sd > 0 else 0.0
    wins = sum(1 for p in pnls if p > 0)
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    pf = gw / gl if gl > 0 else float("inf")
    eq = peak = maxdd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
    return {"n": n, "total": total, "sharpe": sharpe, "wr": wins / n,
            "pf": pf, "maxdd": maxdd}


GRID = {
    "options.em_mult": [0.25, 0.5, 0.75, 1.0, 1.25],
    "options.wing_width": [10, 20, 30, 40, 50],
    "options.tp_pct_of_credit": [0.0, 0.25, 0.5],
    "options.sl_mult_of_credit": [0.0, 2.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true",
                    help="enable calendar gates (skip FOMC/CPI/NFP next-day + Fri)")
    ap.add_argument("--start", default="2022-05-11")
    ap.add_argument("--end", default="2026-06-11")
    args = ap.parse_args()

    with open("config/config_open_ic_pulver.yaml") as f:
        base = yaml.safe_load(f)
    if args.gates:
        _set(base, "strategy.skip_fomc_next_day", True)
        _set(base, "strategy.skip_cpi_next_day", True)
        _set(base, "strategy.skip_nfp_fridays", True)
        _set(base, "strategy.skip_weekdays", ["Fri"])

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    bt = base["backtest"]
    print(f"Preloading CBBO {start}..{end} ...")
    mem = CbboStore.preload_cache(bt["cbbo_cache_dir"], start=start, end=end)
    cbbo = CbboStore(bt["cbbo_cache_dir"], preloaded=mem)
    loader = BarLoader(bt["data_file"], price_scale=bt["price_scale"],
                       symbol=base["bot"]["symbol"])
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(mem)} cbbo days, {len(days)} bar-days. gates={args.gates}\n")

    keys = list(GRID)
    rows = []
    for combo in itertools.product(*GRID.values()):
        cfg = copy.deepcopy(base)
        params = dict(zip(keys, combo))
        for k, v in params.items():
            _set(cfg, k, v)
        # sl_scope total when sl active
        _set(cfg, "options.sl_scope", "total")
        try:
            res = BacktestOrchestrator(cfg, OpenIcStrategy, cbbo).run(days)
        except Exception as e:
            print(f"  ERR {params}: {e}")
            continue
        st = _daily_stats(res)
        if st is None:
            continue
        rows.append((params, st))

    rows.sort(key=lambda r: -r[1]["sharpe"])
    print(f"{'em':>5} {'wing':>5} {'tp':>5} {'sl':>4} | "
          f"{'n':>4} {'Sharpe':>7} {'total$':>9} {'WR%':>5} {'PF':>5} {'maxDD$':>8}")
    print("-" * 70)
    for params, st in rows:
        print(f"{params['options.em_mult']:>5} {params['options.wing_width']:>5} "
              f"{params['options.tp_pct_of_credit']:>5} {params['options.sl_mult_of_credit']:>4} | "
              f"{st['n']:>4} {st['sharpe']:>7.2f} {st['total']:>+9.0f} "
              f"{st['wr']*100:>5.1f} {st['pf']:>5.2f} {st['maxdd']:>8.0f}")


if __name__ == "__main__":
    main()
