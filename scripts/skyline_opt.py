#!/usr/bin/env python3
"""Skyline optimisability study.

Structural variants (the geometry is the suspect, not the entry timing):
  taught     as-taught fixed point offsets
  em0.8/1.0/1.2  offsets scaled by k * EM/26.3 (EM = 09:45 ATM straddle)
  inner      drop the two OUTER broken-wing flies (8 contracts, half the spread)
  inner_em   inner, EM-scaled

Filters (all observable BEFORE entry):
  none / noevent / vix1d band / gap>-20 / spread cap / noevent+spreadcap

Walk-forward: rolling train 252 / val 63 / test 126, select on train, gate on
val, score OOS at mid+25% of spread. Frozen video config always reported.
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

from skyline_core import LEGS, payoff_at, price_structure  # noqa: E402
from backtest_skyline import lookup, parity_spot, CACHE, DB  # noqa: E402
from economic_calendar import get_events_on_date  # noqa: E402
from zerodte.backtest.vix_loader import load_vix_series  # noqa: E402

EM_ANCHOR = 26.3
COMM = 0.65
FILL = 0.25
INNER = [l for l in LEGS if l[0] in (0, 25, 50, -45, -20, 5)]
BIG_EVENTS = {"CPI", "NFP", "GDP", "PPI"}
TABLE = ROOT / "backtest_data" / "skyline_opt_table.pkl"


def variants(em):
    out = {"taught": LEGS}
    if em:
        for k in (0.8, 1.0, 1.2):
            s = k * em / EM_ANCHOR
            out[f"em{k}"] = [(int(round(o * s / 5) * 5), q, c) for o, q, c in LEGS]
        s = em / EM_ANCHOR
        out["inner_em"] = [(int(round(o * s / 5) * 5), q, c) for o, q, c in INNER]
    out["inner"] = INNER
    return out


def build():
    con = sqlite3.connect(DB)
    g = {d: c for d, c in con.execute("SELECT d, close FROM gspc_daily")}
    con.close()
    gd = sorted(g)
    prev = {gd[i]: g[gd[i - 1]] for i in range(1, len(gd))}
    vix1d = load_vix_series(ROOT / "data/vix1d_daily.csv")
    vix = load_vix_series(ROOT / "data/vix_cboe.csv")
    vix9d = load_vix_series(ROOT / "data/vix9d_daily.csv")

    table = []
    for f in sorted(CACHE.glob("*.pkl")):
        d = date.fromisoformat(f.stem)
        ds = d.isoformat()
        if ds not in g:
            continue
        q = pickle.load(open(f, "rb"))
        ks = sorted({k[1] for k in q})
        spot = parity_spot(q, "09:45", ks)
        if spot is None:
            continue
        atm = round(spot / 5.0) * 5.0
        ak = min(ks, key=lambda x: abs(x - spot))
        cq, pq = lookup(q, "09:45", ak, True), lookup(q, "09:45", ak, False)
        em = ((cq[0] + cq[1]) / 2 + (pq[0] + pq[1]) / 2) if cq and pq else None
        ev = set(get_events_on_date(d))
        rec = {"date": ds, "em": em, "vix1d": vix1d.get(d),
               "vix": vix.get(d), "vix9d": vix9d.get(d),
               "gap": spot - prev[ds] if ds in prev else None,
               "event": bool(ev & BIG_EVENTS),
               "opex": d.weekday() == 4 and 15 <= d.day <= 21,
               "pnl": {}, "spread": {}}
        for name, legs in variants(em).items():
            quotes, ok = {}, True
            for o, _, c in legs:
                v = lookup(q, "09:45", atm + o, c)
                if v is None or v[1] <= 0:
                    ok = False
                    break
                quotes[(o, c)] = v
            if not ok:
                continue
            fl = price_structure(quotes, atm, FILL, legs)
            fm = price_structure(quotes, atm, 0.0, legs)
            if fl is None or fm is None or fl.debit < 0:
                continue
            n_ct = sum(abs(x[1]) for x in legs)
            rec["pnl"][name] = payoff_at(g[ds], atm, legs) - fl.debit - COMM * n_ct
            rec["spread"][name] = fl.total_spread_pts
        if rec["pnl"]:
            table.append(rec)
    pickle.dump(table, open(TABLE, "wb"))
    return table


FILTERS = {
    "none":        lambda r: True,
    "noevent":     lambda r: not r["event"] and not r["opex"],
    "em_19_37":    lambda r: r["em"] is not None and 19 <= r["em"] <= 37,
    "vix_13_22":   lambda r: r["vix"] is not None and 13 <= r["vix"] <= 22,
    "gap>-20":     lambda r: r["gap"] is None or r["gap"] > -20,
    "spreadcap":   lambda r: True,          # handled in series()
    "noevent+spr": lambda r: not r["event"] and not r["opex"],
    "noevent+em":  lambda r: (not r["event"] and not r["opex"]
                              and r["em"] is not None and 19 <= r["em"] <= 37),
}

# Any filter keyed on a series with partial history silently becomes a DATE
# filter (VIX1D starts 2023-04-24 and deleted all of 2022 -> a fake edge).
COVERAGE_KEYS = {"em_19_37": "em", "vix_13_22": "vix", "noevent+em": "em"}


def assert_coverage(table):
    from collections import Counter
    tot = Counter(r["date"][:4] for r in table)
    for filt, key in COVERAGE_KEYS.items():
        have = Counter(r["date"][:4] for r in table if r.get(key) is not None)
        worst = min(have[y] / tot[y] for y in tot)
        flag = "OK " if worst > 0.90 else "BAD"
        print(f"  [{flag}] {filt:13} min per-year coverage of '{key}' = "
              f"{100*worst:.0f}%  " + " ".join(f"{y}:{have[y]}/{tot[y]}" for y in sorted(tot)))
        assert worst > 0.90, f"{filt} has partial history -> it is a date filter"


def series(table, cfg, lo, hi):
    var, filt = cfg
    fn = FILTERS[filt]
    out = []
    for r in table:
        if not (lo <= r["date"] < hi):
            continue
        if var not in r["pnl"] or not fn(r):
            continue
        if "spr" in filt or filt == "spreadcap":
            # 0.10pt per contract: 1.6pt for the 16-leg variants, 0.8pt for 8-leg
            cap = 0.8 if var.startswith("inner") else 1.6
            if r["spread"][var] > cap:
                continue
        out.append(r["pnl"][var])
    return out


def sharpe(x):
    if len(x) < 20:
        return None
    sd = st.pstdev(x)
    return (st.mean(x) / sd * math.sqrt(252)) if sd else None


def main():
    table = pickle.load(open(TABLE, "rb")) if TABLE.exists() else build()
    dates = [r["date"] for r in table]
    vs = sorted({v for r in table for v in r["pnl"]})
    cfgs = [(v, f) for v in vs for f in FILTERS]
    print(f"table {len(table)} days {dates[0]}..{dates[-1]}   "
          f"variants={vs}   configs={len(cfgs)}")
    print("\nCOVERAGE GUARD (a partial series = a disguised date filter):")
    assert_coverage(table)

    print("\n" + "=" * 92)
    print("FULL-SAMPLE (in-sample, upper bound) at mid+25% of spread")
    print("=" * 92)
    print(f"  {'variant':10} {'filter':13} {'n':>5} {'$/day':>9} {'Sharpe':>7} "
          f"{'win%':>6} {'medDebit':>9} {'spread$':>8}")
    ranked = []
    for cfg in cfgs:
        s = series(table, cfg, dates[0], dates[-1] + "z")
        if len(s) < 200:
            continue
        ranked.append((st.mean(s), sharpe(s) or 0, len(s), cfg))
    ranked.sort(key=lambda x: -x[0])
    for m, sh, n, (v, f) in ranked[:10]:
        spr = [r["spread"][v] for r in table if v in r["spread"]]
        print(f"  {v:10} {f:13} {n:5d} {m:9.2f} {sh:7.2f} "
              f"{'':>6} {'':>9} {100*st.median(spr):8.0f}")

    TR, VA, TE = 252, 63, 126
    folds, i = [], 0
    while i + TR + VA + TE <= len(dates):
        folds.append((dates[i], dates[i + TR], dates[i + TR + VA],
                      dates[i + TR + VA + TE - 1]))
        i += TE

    print("\n" + "=" * 92)
    print(f"WALK-FORWARD ({len(folds)} folds) — OOS at mid+25%")
    print("=" * 92)
    print(f"  {'fold':4} {'test window':24} {'picked':26} {'trSh':>6} "
          f"{'vaSh':>6} {'TEST $/day':>11} {'teSh':>6}")
    reopt, frozen, best_single = [], [], []
    BEST = ranked[0][3]
    for n, (a, b, c, e) in enumerate(folds):
        scored = []
        for cfg in cfgs:
            tr, va = series(table, cfg, a, b), series(table, cfg, b, c)
            s_tr, s_va = sharpe(tr), sharpe(va)
            if s_tr is None or s_va is None or len(tr) < 120:
                continue
            scored.append((s_tr, s_va, cfg))
        scored.sort(key=lambda x: -x[0])
        pick = next((s for s in scored if s[1] > 0), scored[0])
        te = series(table, pick[2], c, e + "z")
        reopt += te
        frozen += series(table, ("taught", "none"), c, e + "z")
        best_single += series(table, BEST, c, e + "z")
        print(f"  {n:<4} {c}..{e:11} {str(pick[2]):26} {pick[0]:6.2f} "
              f"{pick[1]:6.2f} {st.mean(te):11.2f} {(sharpe(te) or 0):6.2f}")

    print(f"\n  STITCHED OOS:")
    for nm, x in (("re-optimised each fold", reopt),
                  (f"[CONTAMINATED - selected using these same windows] {BEST}",
                   best_single),
                  ("frozen as-taught, no filter", frozen)):
        print(f"    {nm:38} n={len(x):5d} ${st.mean(x):+8.2f}/day "
              f"Sharpe={(sharpe(x) or 0):+6.2f}")

    print("\n" + "=" * 92)
    print("FILTER RETENTION BY YEAR — a filter that drops a whole regime is a "
          "date filter in disguise")
    print("=" * 92)
    from collections import Counter
    tot = Counter(r["date"][:4] for r in table)
    yrs = sorted(tot)
    print(f"  {'filter':13} " + " ".join(f"{y:>10}" for y in yrs))
    for filt, fn in FILTERS.items():
        kept = Counter(r["date"][:4] for r in table
                       if fn(r) and not ("spr" in filt and r["spread"].get("taught", 9) > 1.6))
        print(f"  {filt:13} " + " ".join(
            f"{100*kept[y]/tot[y]:9.0f}%" for y in yrs))

    print("\n" + "=" * 92)
    print("STRUCTURAL DIAGNOSIS — cost vs payoff per variant (full sample, mid+25%)")
    print("=" * 92)
    print(f"  {'variant':10} {'contracts':>10} {'spread$/lot':>12} "
          f"{'$/day mid+25%':>15}")
    for v in vs:
        s = [r["pnl"][v] for r in table if v in r["pnl"]]
        spr = [r["spread"][v] for r in table if v in r["spread"]]
        nct = 8 if v.startswith("inner") else 16
        print(f"  {v:10} {nct:10d} {100*st.median(spr):12.0f} {st.mean(s):15.2f}")
    json.dump({"best_insample": str(ranked[0][3]), "best_insample_perday": ranked[0][0],
               "oos_reopt": st.mean(reopt), "oos_best": st.mean(best_single),
               "oos_frozen": st.mean(frozen)},
              open(ROOT / "backtest_data" / "skyline_opt_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
