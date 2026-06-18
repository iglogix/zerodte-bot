"""Grid sweep + walk-forward CV for the FRVP-ORB strategy.

- sweep:  full-sample grid, ranked by per-trade R Sharpe (min-trade guard).
- wfcv:   anchored walk-forward. For each OOS fold, pick the params that
          maximise Sharpe on the *training* window only, then evaluate that
          single choice on the held-out fold. Reports OOS Sharpe per fold and
          pooled -- this validates the optimisation METHOD, not a single cell.
"""
from __future__ import annotations
import argparse, itertools, json
from dataclasses import replace
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from volume_profile_orb import Params, load, backtest, metrics, TICK, load_trade_vp


def grid_params(base: Params, grid: dict):
    keys = list(grid)
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield replace(base, **dict(zip(keys, combo)))


def run_grid(df, tick, grid, base=Params(), start=None, end=None, min_trades=80,
             vp_cache=None):
    rows = []
    for p in grid_params(base, grid):
        m = backtest(df, p, tick, start, end, vp_cache=vp_cache)
        m.pop("_trades", None)
        if m["n_trades"] >= min_trades:
            rows.append(m)
    rows.sort(key=lambda m: m["sharpe"], reverse=True)
    return rows


DEFAULT_GRID = {
    "or_minutes": [15, 30, 60],
    "entry_tf": [5, 15],
    "rr": [1.0, 1.5, 2.0, 3.0],
    "cutoff": ["10:30", "11:00", "12:00"],
    "body_buffer_ticks": [0.0, 2.0, 5.0],
    "reverse": [False, True],
}


def _grid_for(a):
    grid = json.loads(a.grid) if a.grid else dict(DEFAULT_GRID)
    if getattr(a, "trade_vp", False):
        grid = dict(grid); grid["or_minutes"] = [15]  # cache window is 09:30-09:45
    return grid


def cmd_sweep(a):
    df = load(a.symbol)
    grid = _grid_for(a)
    vp = load_trade_vp(a.symbol) if getattr(a, "trade_vp", False) else None
    rows = run_grid(df, TICK[a.symbol], grid, start=a.start, end=a.end,
                    min_trades=a.min_trades, vp_cache=vp)
    print(f"# {a.symbol}: {len(rows)} cells (min_trades={a.min_trades}). Top 15 by Sharpe:")
    for m in rows[:15]:
        pr = m["params"]
        tag = "FADE" if pr["reverse"] else "foll"
        print(f"  Sh={m['sharpe']:+.2f} exp={m['expectancy_R']:+.3f}R "
              f"wr={m['win_rate']:.2f} n={m['n_trades']} pf={m['profit_factor']:.2f} "
              f"| {tag} or={pr['or_minutes']} tf={pr['entry_tf']} rr={pr['rr']} "
              f"cut={pr['cutoff']} bb={pr['body_buffer_ticks']}")
    if rows:
        print(f"\n# Worst cell Sharpe: {rows[-1]['sharpe']:+.2f}  "
              f"Median: {np.median([m['sharpe'] for m in rows]):+.2f}")


def cmd_wfcv(a):
    day_objs = load(a.symbol)
    tick = TICK[a.symbol]
    grid = _grid_for(a)
    vp = load_trade_vp(a.symbol) if getattr(a, "trade_vp", False) else None
    days = [d.date for d in day_objs]
    n = len(days)
    k = a.folds
    fold_sz = n // (k + 1)  # anchored: fold 0 trains on block 0, tests block 1, ...
    pooled_oos = []
    print(f"# WF-CV {a.symbol}: {n} sessions, {k} OOS folds, "
          f"min_trades(train)={a.min_trades}")
    for i in range(1, k + 1):
        tr0, tr1 = days[0], days[i * fold_sz - 1]
        te0, te1 = days[i * fold_sz], days[min((i + 1) * fold_sz - 1, n - 1)]
        rows = run_grid(day_objs, tick, grid, start=str(tr0), end=str(tr1),
                        min_trades=a.min_trades, vp_cache=vp)
        if not rows:
            print(f"  fold {i}: no train cell met min_trades"); continue
        best = rows[0]["params"]
        from dataclasses import asdict
        p = Params(**best)
        m_oos = backtest(day_objs, p, tick, start=str(te0), end=str(te1), vp_cache=vp)
        m_oos.pop("_trades", None)
        pooled_oos.append((m_oos["sharpe"], m_oos["n_trades"], m_oos["expectancy_R"]))
        tag = "FADE" if best["reverse"] else "foll"
        print(f"  fold {i}: train[{tr0}..{tr1}] -> pick {tag} or={best['or_minutes']} "
              f"tf={best['entry_tf']} rr={best['rr']} cut={best['cutoff']} "
              f"bb={best['body_buffer_ticks']} (trainSh={rows[0]['sharpe']:+.2f}) "
              f"|| OOS[{te0}..{te1}] Sh={m_oos['sharpe']:+.2f} "
              f"exp={m_oos['expectancy_R']:+.3f}R n={m_oos['n_trades']}")
    if pooled_oos:
        shs = [s for s, _, _ in pooled_oos]
        print(f"\n# OOS Sharpe: mean={np.mean(shs):+.2f} median={np.median(shs):+.2f} "
              f"pos_folds={sum(s>0 for s in shs)}/{len(shs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sweep", "wfcv"):
        s = sub.add_parser(name)
        s.add_argument("--symbol", default="SPY", choices=list(TICK))
        s.add_argument("--start"); s.add_argument("--end")
        s.add_argument("--grid"); s.add_argument("--min-trades", type=int, default=80)
        s.add_argument("--trade-vp", action="store_true",
                       help="use trade-level FRVP cache (pins or_minutes=15)")
        if name == "wfcv":
            s.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    {"sweep": cmd_sweep, "wfcv": cmd_wfcv}[a.cmd](a)
