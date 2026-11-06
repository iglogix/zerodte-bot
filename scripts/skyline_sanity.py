#!/usr/bin/env python3
"""Skyline DATA SANITY GATE.

1. parity-derived settlement (option tape, 15:59) vs independent ^GSPC close
2. per-day quotability of the exact 12 Skyline contracts at the entry minute
3. quoted-spread distribution (the fill bracket's input)
4. no-arb screen on the entry marks (butterfly must price >= 0 and <= wing width)
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import statistics as st
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from zerodte.backtest.cbbo_store import CbboStore  # noqa: E402
from skyline_core import LEGS, price_structure  # noqa: E402

CACHE = ROOT / "data" / "cbbo_cache_fly_10min"
DB = ROOT / "backtest_data" / "skyline.db"


def parity_spot(quotes, minute, strikes):
    """S ~= K + C_mid - P_mid at the strike minimising |C_mid - P_mid|."""
    best = None
    for k in strikes:
        cq, pq = quotes.get((minute, k, True)), quotes.get((minute, k, False))
        if cq is None or pq is None:
            continue
        if cq[1] <= 0 or pq[1] <= 0:
            continue
        c, p = (cq[0] + cq[1]) / 2, (pq[0] + pq[1]) / 2
        d = abs(c - p)
        if best is None or d < best[0]:
            best = (d, k + c - p)
    return best[1] if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="09:45")
    ap.add_argument("--last", default="15:59")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    gspc = {d: c for d, c in con.execute("SELECT d, close FROM gspc_daily")}

    files = sorted(CACHE.glob("*.pkl"))
    settle_err, miss_rows, spreads_pt, spreads_rel = [], [], [], []
    n_ok = n_days = 0
    noarb_bad = []
    coverage_fail = {}

    for f in files:
        d = date.fromisoformat(f.stem)
        with open(f, "rb") as fh:
            q = pickle.load(fh)
        n_days += 1
        strikes = sorted({k[1] for k in q})

        s_entry = parity_spot(q, args.entry, strikes)
        s_last = parity_spot(q, args.last, strikes)
        if s_entry is None or s_last is None:
            coverage_fail.setdefault("no_parity_spot", []).append(d)
            continue

        gc = gspc.get(d.isoformat())
        if gc:
            settle_err.append(s_last - gc)

        atm = round(s_entry / 5.0) * 5.0
        quotes, missing = {}, []
        for off, _, is_call in LEGS:
            v = q.get((args.entry, atm + off, is_call))
            if v is None or v[1] <= 0:
                missing.append((off, is_call))
            else:
                quotes[(off, is_call)] = v
        if missing:
            miss_rows.append((d, atm, s_entry, missing))
            coverage_fail.setdefault("missing_legs", []).append(d)
            continue

        for (off, is_call), (b, a) in quotes.items():
            spreads_pt.append(a - b)
            mid = (a + b) / 2
            if mid > 0.05:
                spreads_rel.append((a - b) / mid)

        fmid = price_structure(quotes, atm, 0.0)
        if fmid is None:
            coverage_fail.setdefault("unpriceable", []).append(d)
            continue
        # structural bound: the package can never be worth <0 or >max payoff (2500)
        if not (0.0 <= fmid.debit <= 2500.0):
            noarb_bad.append((d, fmid.debit))
        n_ok += 1

    print(f"days in cache: {n_days}   fully quotable at {args.entry}: {n_ok} "
          f"({100*n_ok/n_days:.1f}%)")
    for k, v in coverage_fail.items():
        print(f"  coverage fail [{k}]: {len(v)} days e.g. {v[:5]}")
    if miss_rows:
        print(f"  example missing-leg day: {miss_rows[0][0]} atm={miss_rows[0][1]} "
              f"spot={miss_rows[0][2]:.1f} missing={miss_rows[0][3][:4]}")

    print(f"\nSETTLEMENT CROSS-CHECK  parity(15:59) - ^GSPC close, n={len(settle_err)}")
    se = sorted(settle_err)
    print(f"  mean={st.mean(se):+.2f}  median={st.median(se):+.2f}  "
          f"sd={st.pstdev(se):.2f}  p1={se[len(se)//100]:+.2f} "
          f"p99={se[-len(se)//100]:+.2f}  max|err|={max(abs(x) for x in se):.2f}")
    print(f"  |err|<=1.0pt: {100*sum(1 for x in se if abs(x)<=1)/len(se):.1f}%   "
          f"|err|<=3.0pt: {100*sum(1 for x in se if abs(x)<=3)/len(se):.1f}%")

    sp = sorted(spreads_pt)
    print(f"\nQUOTED SPREADS at {args.entry} (index points, per leg), n={len(sp)}")
    print(f"  median={st.median(sp):.2f}  mean={st.mean(sp):.2f}  "
          f"p75={sp[3*len(sp)//4]:.2f}  p95={sp[19*len(sp)//20]:.2f}  max={sp[-1]:.2f}")
    sr = sorted(spreads_rel)
    print(f"  relative (spread/mid): median={st.median(sr)*100:.1f}%  "
          f"p95={sr[19*len(sr)//20]*100:.1f}%")
    print(f"  => full half-spread cost for 16 legs ~ "
          f"${100*8*st.median(sp):.0f}/lot (16 legs x half-spread)")

    print(f"\nNO-ARB screen on entry package: {len(noarb_bad)} of {n_ok} out of "
          f"[0, 2500]  {noarb_bad[:5]}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
