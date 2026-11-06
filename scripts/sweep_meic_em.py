#!/usr/bin/env python3
"""Quick full-period parameter sweep for the EM-relative MEIC strategy.

No walk-forward folds — just one full-period backtest per (short_em_mult,
wing_em_frac, stop_pct) combo, sharing a single preloaded CBBO cache. Triage
to see whether any parameter region gets a meaningful edge (PF) with a
tolerable drawdown before committing to a full WF-CV.

Usage:
  python -m scripts.sweep_meic_em --config config/config_meic_spx.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
from datetime import date
from itertools import product
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.engine import BacktestEngine
from zerodte.strategies.meic import MeicStrategy
from zerodte.backtest.report import generate_report

# Sweep grid — vary IC width (short_em_mult) to test whether wider shorts cut
# the stop rate enough to create an edge under honest execution.
SHORT_EM_MULT = [1.0, 1.25, 1.5, 1.75, 2.0]
WING_EM_FRAC = [0.5, 0.75]
STOP_PCT = [0.70]


def main():
    p = argparse.ArgumentParser(description="MEIC EM-relative parameter sweep")
    p.add_argument("--config", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--stop-exec", default="short_trigger",
                   help="stop execution model: trigger|short_trigger|combo|short_only")
    p.add_argument("--gate", type=float, default=None,
                   help="override min_credit_width_ratio (relax so wide shorts trade)")
    args = p.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)

    bt = base.get("backtest", {})
    start = date.fromisoformat(args.start or bt.get("start_date", "2022-05-11"))
    end = date.fromisoformat(args.end or bt.get("end_date", "2026-02-27"))
    cbbo_dir = bt["cbbo_cache_dir"]

    print(f"Preloading CBBO cache from {cbbo_dir} ...")
    mem = CbboStore.preload_cache(cbbo_dir)
    cbbo = CbboStore(cbbo_dir, preloaded=mem)
    print(f"  {len(mem)} days")

    loader = BarLoader(bt.get("data_file", "data/spy_1m.csv"),
                       price_scale=bt.get("price_scale", 10.0),
                       symbol=base.get("bot", {}).get("symbol", "SPX"))
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} trading days ({start}..{end})")

    combos = list(product(SHORT_EM_MULT, WING_EM_FRAC, STOP_PCT))
    print(f"Sweeping {len(combos)} combos\n")

    print(f"stop_execution={args.stop_exec}"
          + (f"  gate={args.gate}" if args.gate is not None else "") + "\n")
    rows = []
    for sem, wef, sp in combos:
        cfg = copy.deepcopy(base)
        cfg["strategy"]["short_em_mult"] = sem
        cfg["strategy"]["wing_em_frac"] = wef
        cfg["strategy"]["stop_pct_of_total_credit"] = sp
        if args.gate is not None:
            cfg["strategy"]["min_credit_width_ratio"] = args.gate
        cfg["options"]["stop_execution"] = args.stop_exec
        engine = BacktestEngine(cfg, MeicStrategy())
        engine._cbbo = cbbo
        rep = generate_report(engine.run(days))
        pnl = rep.get("total_pnl_usd", 0.0)
        dd = rep.get("max_drawdown_daily_usd", 0.0) or 0.0
        ret_dd = (pnl / dd) if dd > 0 else float("inf")
        trades = rep.get("trades_placed", 0)
        stops = rep.get("by_exit_reason", {}).get("short_stopped", {}).get("count", 0)
        stop_pct = 100 * stops / max(trades, 1)
        rows.append({
            "sem": sem, "wef": wef, "sp": sp,
            "trades": trades,
            "wr": rep.get("win_rate", 0.0) * 100,
            "stop_pct": stop_pct,
            "pnl": pnl,
            "pf": rep.get("profit_factor", 0.0),
            "sharpe": rep.get("sharpe_annualised", 0.0),
            "dd": dd,
            "worst_day": rep.get("worst_day_pnl_usd", 0.0),
            "ret_dd": ret_dd,
        })
        print(f"  sem={sem} wef={wef} sp={sp}: trades={trades} stop%={stop_pct:.0f} "
              f"PF={rep.get('profit_factor')} PnL=${pnl:,.0f} DD=${dd:,.0f}")

    # Rank by return/drawdown (risk-adjusted), then PF.
    rows.sort(key=lambda r: (r["ret_dd"], r["pf"]), reverse=True)
    print("\n" + "=" * 116)
    print(f"  {'sem':>4} {'wef':>4} {'sp':>4} | {'trades':>6} {'WR%':>5} {'stop%':>5} "
          f"{'PnL':>11} {'PF':>5} {'Sharpe':>6} {'maxDD':>10} {'worstDay':>9} {'ret/DD':>7}")
    print("=" * 116)
    for r in rows:
        pf = r["pf"] if isinstance(r["pf"], (int, float)) else 0.0
        print(f"  {r['sem']:>4} {r['wef']:>4} {r['sp']:>4} | {r['trades']:>6} "
              f"{r['wr']:>5.1f} {r['stop_pct']:>5.0f} {r['pnl']:>11,.0f} {pf:>5.2f} {r['sharpe']:>6.2f} "
              f"{r['dd']:>10,.0f} {r['worst_day']:>9,.0f} {r['ret_dd']:>7.2f}")
    print("=" * 116)


if __name__ == "__main__":
    main()
