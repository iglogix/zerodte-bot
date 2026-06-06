#!/usr/bin/env python3
"""Concrete fixed-fractional policy for a ~20% drawdown tolerance, across account
sizes. Finds f for ~20% median maxDD (block bootstrap), then translates to daily
buying-power budget, affordable contracts/tranche, and the minimum account to run
the full 13-tranche structure.

Usage: .venv/bin/python tools/theta/sizing_policy.py
"""
from __future__ import annotations
import csv
import statistics
from collections import defaultdict
from datetime import date
from reconcile_meic import DatabentoCache
from resim_exits import Settlement, sim
from sizing_montecarlo import build_r_series, block_bootstrap, path_stats, RNG

TRADES = "~/Dev/0dte_bot/results/meic_spx/trades.csv"


def one_lot_bp_per_day():
    """Per-day BP to run the full structure at 1 contract/tranche, and per-contract
    IC margins (median) for granularity math."""
    rows = [r for r in csv.DictReader(open(TRADES)) if r["entry_time"]]
    bytr = defaultdict(dict)
    for r in rows:
        bytr[(r["session_date"], r["entry_time"])][r["spread_type"]] = r
    day_bp1 = defaultdict(float); per_contract = []; ntr = defaultdict(int)
    for (ds, et), legs in bytr.items():
        if "put" not in legs or "call" not in legs:
            continue
        p, c = legs["put"], legs["call"]
        W = max(float(p["spread_width"]), float(c["spread_width"]))
        tot = float(p["credit_received"]) + float(c["credit_received"])
        m1 = max(0.0, W - tot) * 100      # 1-contract IC margin
        day_bp1[ds] += m1; per_contract.append(m1); ntr[ds] += 1
    return day_bp1, per_contract, ntr


def main():
    dates, r = build_r_series()
    yrs = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    rng = RNG(); paths = block_bootstrap(r, 3000, 5, rng)

    # find f giving ~20% median maxDD
    print("=" * 78)
    print("Fine f-sweep -> target ~20% median max drawdown")
    print(f"  {'f':>5} {'worstday':>9} {'medCAGR':>8} {'medDD':>7} {'P95DD':>7} {'P99DD':>7}")
    chosen = None
    for f in (0.02, 0.025, 0.03, 0.035, 0.04, 0.05):
        dds = []; cg = []
        for p in paths:
            mdd, eq = path_stats(p, f)
            dds.append(mdd); cg.append(eq ** (1/yrs) - 1 if eq > 0 else -1)
        dds.sort()
        med = dds[len(dds)//2]
        mark = ""
        if chosen is None and med <= -0.20:
            chosen = f; mark = "  <- ~20% DD"
        print(f"  {f:>5.3f} {f*min(r)*100:>8.0f}% {statistics.median(cg)*100:>7.0f}% "
              f"{med*100:>6.0f}% {dds[int(.05*len(dds))]*100:>6.0f}% {dds[int(.01*len(dds))]*100:>6.0f}%{mark}")
    f = chosen or 0.03
    # CAGR at chosen f
    cg = [path_stats(p, f)[1] ** (1/yrs) - 1 for p in paths if path_stats(p, f)[1] > 0]
    cagr = statistics.median(cg)

    day_bp1, per_contract, ntr = one_lot_bp_per_day()
    med_full = statistics.median(day_bp1.values())          # full-structure 1-lot BP, median day
    p90_full = sorted(day_bp1.values())[int(.9*len(day_bp1))]
    med_pc = statistics.median(per_contract)
    min_acct_full = med_full / f

    print("\n" + "=" * 78)
    print(f"CHOSEN POLICY: fixed-fractional, f = {f*100:.1f}% of equity as BP/day "
          f"(~20% median DD, ~{cagr*100:.0f}% CAGR* )")
    print(f"  worst single day ~= {f*min(r)*100:.0f}% of equity (hard bound, defined-risk)")
    print(f"  full 13-tranche structure at 1 lot: ${med_full:,.0f} BP median day "
          f"(${p90_full:,.0f} on busy/high-vol days)")
    print(f"  per-contract IC margin: ~${med_pc:,.0f} median")
    print(f"  => min account for full structure at f={f*100:.0f}%: ${min_acct_full:,.0f}")
    print("=" * 78)
    print(f"{'account':>9} {'BP budget/day':>14} {'can run full?':>14} {'lots/tranche':>13} {'est CAGR*':>10}")
    print("-" * 78)
    for acct in (50_000, 100_000, 250_000, 500_000):
        budget = f * acct
        # lots per tranche if running full structure: budget / med_full (median day)
        lots_full = budget / med_full
        if lots_full >= 1:
            note = "yes"; lots = f"{lots_full:.1f}x full"
            est = cagr   # at target f, returns scale-invariant
        else:
            # can't size down below 1 lot/tranche -> run a SUBSET of tranches at 1 lot
            n_aff = int(budget / med_pc)   # tranches affordable at 1 lot
            note = f"no ({n_aff}/13 tr)"
            lots = "1 lot, subset"
            est = cagr * (n_aff / 13)  # rough: fewer tranches ~ proportionally less $ (same %)
        print(f"${acct:>8,} {budget:>14,.0f} {note:>14} {lots:>13} {est*100:>9.0f}%")
    print("-" * 78)
    print("  *CAGR uses optimistic hold-to-expiry (mid) fills -> treat as a ceiling.")
    print("  Daily rule: BP_budget = f x equity; per tranche, lots = floor(tranche_share / IC_margin).")
    print("  Below the min account, you trade FEWER tranches (lumpier) or accept higher f (more DD).")


if __name__ == "__main__":
    main()
