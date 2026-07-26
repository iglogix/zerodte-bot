#!/usr/bin/env python3
"""Skyline walk-forward CV: is ANY variant OOS-viable at a realistic fill?

Grid (params the claimant exposes + two structural fixes):
  entry  : 09:35 / 09:45 / 10:00 / 10:30
  shift  : 0 / +20 / -20   (move the whole structure; +20 puts a PEAK at ATM
                            instead of the deepest valley)
  scale  : fixed point offsets (as taught) / level-scaled (% of spot)
  filter : max overnight drop None / 15 / 25 / 40 pts

Folds: rolling train(252d) -> val(63d) -> test(126d). Selection on TRAIN,
gate on VAL, score on TEST. Always compares frozen (video config) vs re-optimised.
"""
from __future__ import annotations

import json
import math
import pickle
import sqlite3
import statistics as st
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from skyline_core import LEGS, N_CONTRACTS, payoff_at, price_structure, scaled_legs  # noqa: E402
from backtest_skyline import lookup, parity_spot, CACHE, DB, ANCHOR_SPOT  # noqa: E402

ENTRIES = ["09:35", "09:45", "10:00", "10:30"]
SHIFTS = [0, 20, -20]
SCALES = ["fixed", "level"]
DROPS = [None, 15.0, 25.0, 40.0]
COMM = 0.65
FILL = 0.25          # realistic marketable-limit fill
FROZEN = ("09:45", 0, "fixed", 25.0)   # the video's configuration


def build_table():
    con = sqlite3.connect(DB)
    gspc = {d: c for d, c in con.execute("SELECT d, close FROM gspc_daily")}
    con.close()
    gd = sorted(gspc)
    prev = {gd[i]: gspc[gd[i - 1]] for i in range(1, len(gd))}

    table = []
    for f in sorted(CACHE.glob("*.pkl")):
        d = date.fromisoformat(f.stem)
        ds = d.isoformat()
        if ds not in gspc:
            continue
        with open(f, "rb") as fh:
            q = pickle.load(fh)
        strikes = sorted({k[1] for k in q})
        settle = gspc[ds]
        rec = {"date": ds, "drop": None, "pnl": {}}
        for entry in ENTRIES:
            spot = parity_spot(q, entry, strikes)
            if spot is None:
                continue
            if entry == "09:45" and ds in prev:
                rec["drop"] = spot - prev[ds]
            atm = round(spot / 5.0) * 5.0
            for scale in SCALES:
                base = LEGS if scale == "fixed" else scaled_legs(spot / ANCHOR_SPOT)
                for sh in SHIFTS:
                    legs = [(o + sh, qy, c) for o, qy, c in base]
                    quotes, ok = {}, True
                    for o, _, c in legs:
                        v = lookup(q, entry, atm + o, c)
                        if v is None or v[1] <= 0:
                            ok = False
                            break
                        quotes[(o, c)] = v
                    if not ok:
                        continue
                    fl = price_structure(quotes, atm, FILL, legs)
                    if fl is None or not (0 <= fl.debit <= 3000):
                        continue
                    rec["pnl"][(entry, sh, scale)] = (
                        payoff_at(settle, atm, legs) - fl.debit - COMM * N_CONTRACTS)
        if rec["pnl"]:
            table.append(rec)
    return table


def series(table, cfg, lo, hi):
    entry, sh, scale, drop = cfg
    out = []
    for r in table:
        if not (lo <= r["date"] < hi):
            continue
        if drop is not None and r["drop"] is not None and r["drop"] < -drop:
            continue
        v = r["pnl"].get((entry, sh, scale))
        if v is not None:
            out.append(v)
    return out


def sharpe(x):
    if len(x) < 20:
        return None
    sd = st.pstdev(x)
    return (st.mean(x) / sd * math.sqrt(252)) if sd else None


def main():
    cache = ROOT / "backtest_data" / "skyline_wf_table.pkl"
    if cache.exists():
        table = pickle.load(open(cache, "rb"))
    else:
        table = build_table()
        pickle.dump(table, open(cache, "wb"))
    dates = [r["date"] for r in table]
    print(f"table: {len(table)} days {dates[0]}..{dates[-1]}  "
          f"grid={len(ENTRIES)*len(SHIFTS)*len(SCALES)*len(DROPS)} configs")

    configs = [(e, s, sc, d) for e in ENTRIES for s in SHIFTS
               for sc in SCALES for d in DROPS]

    TR, VA, TE = 252, 63, 126
    folds, i = [], 0
    while i + TR + VA + TE <= len(dates):
        folds.append((dates[i], dates[i + TR], dates[i + TR + VA],
                      dates[i + TR + VA + TE - 1]))
        i += TE
    print(f"folds: {len(folds)}\n")

    print(f"{'fold':4} {'train':11} {'test window':24} {'picked config':34} "
          f"{'trainSh':>8} {'valSh':>7} {'TEST $/day':>11} {'TESTSh':>7} "
          f"{'FROZEN$/d':>10} {'FrznSh':>7}")
    reopt_te, frozen_te = [], []
    for n, (a, b, c, e) in enumerate(folds):
        scored = []
        for cfg in configs:
            tr = series(table, cfg, a, b)
            va = series(table, cfg, b, c)
            s_tr, s_va = sharpe(tr), sharpe(va)
            if s_tr is None or s_va is None or len(tr) < 150:
                continue
            scored.append((s_tr, s_va, cfg))
        scored.sort(key=lambda x: -x[0])
        pick = next((s for s in scored if s[1] > 0), None)   # val gate
        if pick is None:
            pick = scored[0]
        s_tr, s_va, cfg = pick
        te = series(table, cfg, c, e + "z")
        fz = series(table, FROZEN, c, e + "z")
        reopt_te += te
        frozen_te += fz
        print(f"{n:<4} {a:11} {c}..{e:11} {str(cfg):34} "
              f"{s_tr:8.2f} {s_va:7.2f} {st.mean(te):11.2f} "
              f"{(sharpe(te) or 0):7.2f} {st.mean(fz):10.2f} {(sharpe(fz) or 0):7.2f}")

    print(f"\nSTITCHED OOS (fill = mid+25% of spread, comm ${COMM}/contract):")
    for nm, x in (("re-optimised", reopt_te), ("frozen (video cfg)", frozen_te)):
        print(f"  {nm:20} n={len(x):5d}  ${st.mean(x):+8.2f}/day  "
              f"total ${sum(x):+10,.0f}  Sharpe={(sharpe(x) or 0):+6.2f}  "
              f"win={100*sum(1 for v in x if v>0)/len(x):.1f}%")

    print("\nFULL-SAMPLE ranking of the whole grid at mid+25% (in-sample, for "
          "reference only):")
    ranked = []
    for cfg in configs:
        s = series(table, cfg, dates[0], dates[-1] + "z")
        if len(s) > 300:
            ranked.append((st.mean(s), sharpe(s) or 0, len(s), cfg))
    ranked.sort(key=lambda x: -x[0])
    for m, sh, n, cfg in ranked[:6]:
        print(f"  {str(cfg):34} n={n:5d} ${m:+8.2f}/day Sharpe={sh:+6.2f}")
    print("  ...")
    for m, sh, n, cfg in ranked[-2:]:
        print(f"  {str(cfg):34} n={n:5d} ${m:+8.2f}/day Sharpe={sh:+6.2f}")
    json.dump({"stitched_oos_reopt_per_day": st.mean(reopt_te),
               "stitched_oos_frozen_per_day": st.mean(frozen_te),
               "best_insample": [ranked[0][3], ranked[0][0], ranked[0][1]]},
              open(ROOT / "backtest_data" / "skyline_wf_summary.json", "w"),
              indent=2, default=str)


if __name__ == "__main__":
    main()
