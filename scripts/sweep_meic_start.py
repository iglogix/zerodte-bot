"""
MEIC start-time sweep — test the rumor that starting the entry ladder earlier
(~09:40-09:50) works better, especially lately.

Runs the validated MEIC (hold-to-expiry, uniform tranche sizing) on the
early-morning CBBO cache (data/cbbo_cache, 09:31+, which the MEIC-specific cache
lacks). Two views:

  1. FULL ladder 09:40->15:00 (every 20m): PnL/Sharpe broken down BY ENTRY TIME,
     full-period vs "lately" — directly shows whether early slots outperform.
  2. START-SHIFT: fixed-length ladder whose start time varies; full + recent
     metrics per start.

Memory-safe: the engine lazy-loads one day's CBBO at a time (no preload).

Usage:
  python scripts/sweep_meic_start.py --recent-from 2025-09-01
"""
from __future__ import annotations
import argparse, copy, sys
from collections import defaultdict
from datetime import date
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.engine import BacktestEngine
from zerodte.strategies.meic import MeicStrategy
from zerodte.backtest.report import generate_report

CBBO_EARLY = "data/cbbo_cache"   # has 09:31+ (MEIC's own cache starts 11:00)


def ladder(start_hhmm: str, n: int, step_min: int = 20, last="15:00"):
    h, m = int(start_hhmm[:2]), int(start_hhmm[3:])
    lh, lm = int(last[:2]), int(last[3:])
    out = []
    for _ in range(n):
        if (h, m) > (lh, lm):
            break
        out.append(f"{h:02d}:{m:02d}")
        m += step_min
        while m >= 60:
            m -= 60; h += 1
    return out


def base_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["backtest"]["cbbo_cache_dir"] = CBBO_EARLY
    cfg["strategy"]["exit_mode"] = "hold_to_expiry"
    cfg["strategy"]["sizing_mode"] = "static"
    cfg["strategy"]["tranche_sizing"] = {}        # uniform -> all default_tranche_qty
    cfg["strategy"]["default_tranche_qty"] = 1
    return cfg


def run(cfg, days):
    eng = BacktestEngine(cfg, MeicStrategy())
    return eng.run(days)


def sub_metrics(results, lo=None, hi=None):
    rs = [r for r in results
          if (lo is None or r.session_date >= lo) and (hi is None or r.session_date <= hi)]
    rep = generate_report(rs)
    return rep


def fmt(rep):
    return (f"PnL=${rep.get('total_pnl_usd',0):>10,.0f}  Sh={rep.get('sharpe_annualised',0):>5.2f}  "
            f"PF={rep.get('profit_factor',0):>4.2f}  WR={rep.get('win_rate',0)*100:>4.1f}%  "
            f"DD=${rep.get('max_drawdown_daily_usd',0) or 0:>9,.0f}  n={rep.get('trades_placed',0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_meic_spx.yaml")
    ap.add_argument("--recent-from", default="2025-09-01")
    ap.add_argument("--start", default="2022-05-11")
    ap.add_argument("--end", default="2026-06-11")
    a = ap.parse_args()
    recent = date.fromisoformat(a.recent_from)
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)

    cfg0 = base_cfg(a.config)
    loader = BarLoader(cfg0["backtest"].get("data_file", "data/spy_1m.csv"),
                       price_scale=cfg0["backtest"].get("price_scale", 10.0),
                       symbol=cfg0.get("bot", {}).get("symbol", "SPX"))
    days = list(loader.iter_days(start_date=s, end_date=e))
    print(f"{len(days)} trading days {s}..{e};  'lately' = >= {recent}\n")

    # ---- View 1: full ladder 09:40->15:00, breakdown by entry time --------
    full = copy.deepcopy(cfg0)
    full["entry"]["schedule"] = ladder("09:40", 99, 20, "15:00")
    print(f"=== View 1: full ladder {full['entry']['schedule'][0]}..{full['entry']['schedule'][-1]} "
          f"({len(full['entry']['schedule'])} slots), PnL by entry time ===")
    res = run(full, days)
    closed = [r for r in res if r.status == "closed" and r.realized_pnl_usd is not None]

    def slot_of(r):
        et = r.entry_time
        return et.strftime("%H:%M") if hasattr(et, "strftime") else str(et)[:5]

    agg = defaultdict(lambda: [[], []])  # slot -> [all_pnls, recent_pnls]
    for r in closed:
        agg[slot_of(r)][0].append(r.realized_pnl_usd)
        if r.session_date >= recent:
            agg[slot_of(r)][1].append(r.realized_pnl_usd)
    print(f"  {'slot':>5} | {'n':>4} {'PnL':>9} {'avg':>6} {'wr%':>4} | "
          f"{'n':>4} {'PnL(late)':>9} {'avg':>6} {'wr%':>4}")
    for slot in sorted(agg):
        allp, latep = agg[slot]
        a_n, a_s = len(allp), sum(allp)
        a_avg = a_s / a_n if a_n else 0
        a_wr = 100 * np.mean([x > 0 for x in allp]) if allp else 0
        l_n, l_s = len(latep), sum(latep)
        l_avg = l_s / l_n if l_n else 0
        l_wr = 100 * np.mean([x > 0 for x in latep]) if latep else 0
        print(f"  {slot:>5} | {a_n:>4} {a_s:>9,.0f} {a_avg:>6,.0f} {a_wr:>4.0f} | "
              f"{l_n:>4} {l_s:>9,.0f} {l_avg:>6,.0f} {l_wr:>4.0f}")

    # ---- View 2: start-shift sweep ---------------------------------------
    print("\n=== View 2: start-shift sweep (13-slot ladder, 20m spacing) ===")
    print(f"  {'start':>6} | {'FULL: ':>0}{'':>0}")
    for start in ("09:40", "10:00", "10:30", "11:00", "11:30", "12:00"):
        cfg = copy.deepcopy(cfg0)
        cfg["entry"]["schedule"] = ladder(start, 13, 20, "15:40")
        res = run(cfg, days)
        full_rep = sub_metrics(res)
        late_rep = sub_metrics(res, lo=recent)
        print(f"  {start:>6} | FULL {fmt(full_rep)}")
        print(f"         | LATE {fmt(late_rep)}")


if __name__ == "__main__":
    main()
