#!/usr/bin/env python3
"""'Skyline-R' — the structure rebuilt from everything the study found.

R0  as taught (baseline)
R1  geometry fix only: inner two flies, EM-scaled  (8 contracts, not 16)
R2  R1 + avoid-list (events/OPEX, EM tails, gap-down, flat open, wide spread)
R3  R1 + the only |t|>2 condition: overnight gap up > +20pt

Every number is reported across the fill bracket, per year, and fold-by-fold on
OOS test windows only. NOTE: all of it was designed on these same 1,008 days, so
even the fold record is optimistic. Treat as hypothesis, not validated edge.
"""
from __future__ import annotations

import math
import pickle
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from skyline_core import LEGS, payoff_at, price_structure  # noqa: E402
from backtest_skyline import lookup, parity_spot, CACHE, DB  # noqa: E402
from economic_calendar import get_events_on_date  # noqa: E402

EM_ANCHOR = 26.3
COMM = 0.65
BIG = {"CPI", "NFP", "GDP", "PPI"}
INNER = [l for l in LEGS if l[0] in (0, 25, 50, -45, -20, 5)]
FF = (0.0, 0.25, 0.5)
TBL = ROOT / "backtest_data" / "skyline_revised.pkl"

# --holdout swaps the quote cache for the untouched 2026-06-12..07-24 window.
# NOTHING below (variants, thresholds, filters) changes between the two runs.
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--holdout", action="store_true")
_ARGS, _ = _p.parse_known_args()
if _ARGS.holdout:
    CACHE = ROOT / "data" / "cbbo_cache_skyline_holdout"
    TBL = ROOT / "backtest_data" / "skyline_revised_holdout.pkl"


def build():
    con = sqlite3.connect(DB)
    g = {d: c for d, c in con.execute("SELECT d, close FROM gspc_daily")}
    con.close()
    gd = sorted(g)
    prev = {gd[i]: g[gd[i - 1]] for i in range(1, len(gd))}

    out = []
    for f in sorted(CACHE.glob("*.pkl")):
        d = date.fromisoformat(f.stem)
        ds = d.isoformat()
        if ds not in g:
            continue
        q = pickle.load(open(f, "rb"))
        ks = sorted({k[1] for k in q})
        spot = parity_spot(q, "09:45", ks)
        s932 = parity_spot(q, "09:32", ks)
        if spot is None:
            continue
        atm = round(spot / 5.0) * 5.0
        ak = min(ks, key=lambda x: abs(x - spot))
        cq, pq = lookup(q, "09:45", ak, True), lookup(q, "09:45", ak, False)
        if not cq or not pq or cq[1] <= 0 or pq[1] <= 0:
            continue
        em = (cq[0] + cq[1]) / 2 + (pq[0] + pq[1]) / 2
        s = em / EM_ANCHOR
        variants = {
            "R0_taught": LEGS,
            "R1_inner_em": [(int(round(o * s / 5) * 5), qy, c) for o, qy, c in INNER],
        }
        rec = {"date": ds, "em": em, "spot": spot,
               "gap": spot - prev[ds] if ds in prev else None,
               "drive": (spot - s932) if s932 else None,
               "event": bool(set(get_events_on_date(d)) & BIG),
               "opex": d.weekday() == 4 and 15 <= d.day <= 21,
               "pnl": defaultdict(dict), "spread": {}, "debit": {}}
        ok_all = True
        for nm, legs in variants.items():
            quotes, ok = {}, True
            for o, _, c in legs:
                v = lookup(q, "09:45", atm + o, c)
                if v is None or v[1] <= 0:
                    ok = False
                    break
                quotes[(o, c)] = v
            if not ok:
                ok_all = False
                break
            nct = sum(abs(x[1]) for x in legs)
            payoff = payoff_at(g[ds], atm, legs)
            for ff in FF:
                fl = price_structure(quotes, atm, ff, legs)
                if fl is None:
                    ok_all = False
                    break
                rec["pnl"][nm][ff] = payoff - fl.debit - COMM * nct
                if ff == 0.0:
                    rec["spread"][nm] = fl.total_spread_pts
                    rec["debit"][nm] = fl.debit
        if ok_all:
            rec["pnl"] = dict(rec["pnl"])
            out.append(rec)
    pickle.dump(out, open(TBL, "wb"))
    return out


# Regime-neutral avoid-list. A spread cap was tried and DROPPED: it retained
# only 13% of 2022 vs 74-82% of later years, i.e. it was a "skip 2022" filter
# in disguise (the same trap that killed the VIX1D band). Retention with it
# removed is 32/28/28/34/36% across 2022-2026 (min/max 0.79).
AVOID = lambda r: not (
    r["event"] or r["opex"]
    or r["em"] < 19 or r["em"] > 37
    or (r["gap"] is not None and r["gap"] < -20)
    or (r["drive"] is not None and abs(r["drive"]) < 1.8))
GAPUP = lambda r: r["gap"] is not None and r["gap"] > 20

CFG = {
    "R0  as taught":                    ("R0_taught",   lambda r: True),
    "R1  inner x2, EM-scaled":          ("R1_inner_em", lambda r: True),
    "R2  R1 + avoid-list (neutral)":    ("R1_inner_em", AVOID),
    "R3  R1 + gap up > +20pt":          ("R1_inner_em", GAPUP),
}


def stats(p):
    if not p:
        return None
    eq = peak = mdd = 0.0
    for x in p:
        eq += x
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    sd = st.pstdev(p)
    return {"n": len(p), "tot": sum(p), "mean": st.mean(p),
            "sh": st.mean(p) / sd * math.sqrt(252) if sd else 0,
            "dd": mdd, "win": 100 * sum(1 for x in p if x > 0) / len(p)}


def main():
    rows = pickle.load(open(TBL, "rb")) if TBL.exists() else build()
    yrs = len(rows) / 252
    print(f"{len(rows)} days {rows[0]['date']}..{rows[-1]['date']}  ({yrs:.1f} yrs), 1 lot\n")

    print("=" * 100)
    print("FILL BRACKET — $/day (Sharpe)")
    print("=" * 100)
    print(f"  {'config':28}{'days/yr':>8}{'mid':>19}{'mid+25%':>19}{'half-spread':>19}")
    for nm, (v, fn) in CFG.items():
        sel = [r for r in rows if fn(r)]
        cells = []
        for ff in FF:
            s = stats([r["pnl"][v][ff] for r in sel])
            cells.append(f"{s['mean']:+8.2f} ({s['sh']:+5.2f})")
        print(f"  {nm:28}{len(sel)/yrs:8.0f}" + "".join(f"{c:>19}" for c in cells))

    print("\n" + "=" * 100)
    print("AT A REALISTIC FILL (mid+25%) — is it worth trading?")
    print("=" * 100)
    print(f"  {'config':28}{'n':>6}{'$/yr':>10}{'Sharpe':>8}{'maxDD':>9}"
          f"{'win%':>7}{'medDebit':>10}{'spread$':>9}")
    for nm, (v, fn) in CFG.items():
        sel = [r for r in rows if fn(r)]
        s = stats([r["pnl"][v][0.25] for r in sel])
        print(f"  {nm:28}{s['n']:6d}{s['tot']/yrs:10,.0f}{s['sh']:8.2f}"
              f"{s['dd']:9,.0f}{s['win']:7.1f}"
              f"{st.median([r['debit'][v] for r in sel]):10,.0f}"
              f"{100*st.median([r['spread'][v] for r in sel]):9.0f}")

    print("\n" + "=" * 100)
    print("PER YEAR at mid+25%  ($/day)")
    print("=" * 100)
    years = sorted({r["date"][:4] for r in rows})
    print(f"  {'config':28}" + "".join(f"{y:>12}" for y in years))
    for nm, (v, fn) in CFG.items():
        cells = []
        for y in years:
            p = [r["pnl"][v][0.25] for r in rows if fn(r) and r["date"][:4] == y]
            cells.append(f"{st.mean(p):+7.0f}({len(p):>3})" if len(p) >= 10 else "      -")
        print(f"  {nm:28}" + "".join(f"{c:>12}" for c in cells))

    print("\n" + "=" * 100)
    print("FOLD-BY-FOLD, OOS TEST WINDOWS ONLY, mid+25% ($/day)")
    print("=" * 100)
    dates = [r["date"] for r in rows]
    TR, VA, TE = 252, 63, 126
    folds, i = [], 0
    while i + TR + VA + TE <= len(dates):
        folds.append((dates[i + TR + VA], dates[i + TR + VA + TE - 1]))
        i += TE
    print(f"  {'config':28}" + "".join(f"{c[:7]:>13}" for c, e in folds) + f"{'stitched':>12}")
    for nm, (v, fn) in CFG.items():
        cells, allp = [], []
        for c, e in folds:
            p = [r["pnl"][v][0.25] for r in rows if fn(r) and c <= r["date"] <= e]
            cells.append(f"{st.mean(p):+6.0f}({len(p):>2})" if len(p) >= 10 else "     -")
            allp += p
        print(f"  {nm:28}" + "".join(f"{x:>13}" for x in cells)
              + f"{st.mean(allp):+12.1f}")

    print("\n" + "=" * 100)
    print("IS R2 JUST R3 IN DISGUISE?  overlap of the two filters")
    print("=" * 100)
    a = {r["date"] for r in rows if AVOID(r)}
    b = {r["date"] for r in rows if GAPUP(r)}
    both = a & b
    print(f"  avoid-list days={len(a)}  gap-up days={len(b)}  overlap={len(both)}")
    for lbl, sel in (("avoid-list WITHOUT gap-up", a - b), ("gap-up WITHOUT avoid-list", b - a),
                     ("both", both)):
        p = [r["pnl"]["R1_inner_em"][0.25] for r in rows if r["date"] in sel]
        s = stats(p)
        print(f"  {lbl:28} n={s['n']:4d}  ${s['mean']:+8.2f}/day  Sharpe {s['sh']:+5.2f}")


if __name__ == "__main__":
    main()
