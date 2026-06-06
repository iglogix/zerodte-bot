#!/usr/bin/env python3
"""Entry-fill sensitivity for hold-to-expiry MEIC. Tests whether 'mid entry' is
realistic given: leg-by-leg working orders, European cash settlement (no exit
fills), and the fact that protective wings are the only illiquid legs.

Fill models for the ENTRY credit (exit is cash settlement at parity intrinsic in
all cases — no exit slippage):
  M1 both-mid        : sell shorts @ mid,  buy longs @ mid     (current baseline)
  M2 shorts-mid/longs-ask : sell shorts @ mid, buy wings @ ask (realistic: shorts
                       are liquid and workable, wings are wide so pay the ask)
  M3 no-improvement  : sell shorts @ bid,  buy longs @ ask     (pessimistic bound)

No extra flat slippage is added (the bid/ask IS the haircut); commission $1/leg.

Usage: .venv/bin/python tools/theta/entry_fill_sensitivity.py
"""
from __future__ import annotations
import csv
from collections import defaultdict
from datetime import date
from reconcile_meic import DatabentoCache, TRADES_CSV
from resim_exits import Settlement

COMM, MULT = 1.0, 100


def side_pnl(src, settle, d, minute, short_k, long_k, is_call, qty, model):
    sq = src.get(d, minute, short_k, is_call)
    lq = src.get(d, minute, long_k, is_call)
    if not sq or not lq or sq[0] <= 0 or sq[1] <= 0 or lq[0] <= 0 or lq[1] <= 0:
        return None
    sbid, sask = sq; lbid, lask = lq
    smid, lmid = (sbid + sask) / 2, (lbid + lask) / 2
    if model == "M1":
        credit = smid - lmid
    elif model == "M2":
        credit = smid - lask              # sell short @ mid, buy wing @ ask
    else:  # M3
        credit = sbid - lask              # sell short @ bid, buy wing @ ask
    width = abs(long_k - short_k)
    if credit <= 0:
        return None
    F = settle.forward(d)
    if F is None:
        return None
    loss = min(max(0.0, (F - short_k) if is_call else (short_k - F)), width)
    return (credit - loss) * MULT * qty - 2 * qty * COMM


def main():
    src = DatabentoCache(); settle = Settlement(src)
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])
    bytr = defaultdict(dict)
    for r in rows:
        bytr[(r["session_date"], r["entry_time"])][r["spread_type"]] = r

    print("=" * 74)
    print("ENTRY-FILL SENSITIVITY — hold-to-expiry MEIC (exit = cash settlement, no fills)")
    print("=" * 74)
    res = {}
    for model in ("M1", "M2", "M3"):
        tot = 0.0; yr = defaultdict(float); n = 0
        for (ds, et), legs in bytr.items():
            if "put" not in legs or "call" not in legs:
                continue
            p, c = legs["put"], legs["call"]
            qty = int(p["qty"]); minute = et[11:16]; d = date.fromisoformat(ds)
            call = side_pnl(src, settle, d, minute, float(c["short_strike"]),
                            float(c["long_strike"]), True, qty, model)
            put = side_pnl(src, settle, d, minute, float(p["short_strike"]),
                           float(p["long_strike"]), False, qty, model)
            if call is None or put is None:
                continue
            tot += call + put; yr[ds[:4]] += call + put; n += 1
        res[model] = (tot, n, yr)

    label = {"M1": "M1 both-mid (baseline)", "M2": "M2 shorts-mid / wings-ask (realistic)",
             "M3": "M3 short-bid / wing-ask (pessimistic)"}
    base = res["M1"][0]
    print(f"  {'fill model':40s} {'total P&L':>13} {'vs M1':>11}")
    for m in ("M1", "M2", "M3"):
        tot, n, _ = res[m]
        print(f"  {label[m]:40s} {tot:>13,.0f} {tot-base:>+11,.0f}")
    print("\n  by year:")
    for m in ("M1", "M2", "M3"):
        _, _, yr = res[m]
        print(f"   {m}: " + "  ".join(f"{y}:${yr[y]/1000:+.0f}k" for y in sorted(yr)))
    print("=" * 74)
    print("  Note: recorded baseline used mid-0.10 flat slip = $1.45M; these drop that")
    print("  flat haircut and price the wings explicitly instead.")


if __name__ == "__main__":
    main()
