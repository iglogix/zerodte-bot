#!/usr/bin/env python3
"""Final execution-realistic sizing policy: M2 fills (shorts @ mid, wings @ ask,
European cash settlement = no exit fills) folded into the fixed-fractional sizing
Monte Carlo, plus a missed-fill (fill-certainty) sensitivity.

Missed fill proxy: if the underlying moved > THRESH pts in the entry minute (parity
forward at m vs m-1), a resting mid limit on the short likely wouldn't fill -> skip
the tranche. Shows whether the miss/selection effect helps or hurts.

Usage: .venv/bin/python tools/theta/sizing_policy_realistic.py
"""
from __future__ import annotations
import csv
import statistics
from collections import defaultdict
from datetime import date
from reconcile_meic import DatabentoCache, TRADES_CSV
from resim_exits import Settlement
from sizing_montecarlo import block_bootstrap, path_stats, RNG

COMM, MULT = 1.0, 100


def parity(src, d, minute, ref):
    src._load(d); q = src._q
    if not q:
        return None
    strikes = sorted({k[1] for k in q if k[0] == minute})
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - ref))
    c = q.get((minute, atm, True)); p = q.get((minute, atm, False))
    if not c or not p:
        return None
    return atm + (c[0]+c[1])/2 - (p[0]+p[1])/2


def m2_side(src, settle, d, minute, short_k, long_k, is_call, qty):
    """M2: sell short @ mid, buy wing @ ask. Returns (pnl, credit, width) or None.
    Margin is computed at the IC level by the caller (one-side max - total credit)."""
    sq = src.get(d, minute, short_k, is_call); lq = src.get(d, minute, long_k, is_call)
    if not sq or not lq or min(sq[0], sq[1], lq[0], lq[1]) <= 0:
        return None
    smid = (sq[0]+sq[1])/2
    credit = smid - lq[1]                       # wing bought at ask
    width = abs(long_k - short_k)
    if credit <= 0:
        return None
    F = settle.forward(d)
    if F is None:
        return None
    loss = min(max(0.0, (F - short_k) if is_call else (short_k - F)), width)
    pnl = (credit - loss) * MULT * qty - 2 * qty * COMM
    return pnl, credit, width


def build(miss_thresh=None):
    src = DatabentoCache(); settle = Settlement(src)
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])
    bytr = defaultdict(dict)
    for r in rows:
        bytr[(r["session_date"], r["entry_time"])][r["spread_type"]] = r
    pnl = defaultdict(float); bp = defaultdict(float); bp1 = defaultdict(float)
    skipped = 0
    for (ds, et), legs in bytr.items():
        if "put" not in legs or "call" not in legs:
            continue
        p, c = legs["put"], legs["call"]
        qty = int(p["qty"]); minute = et[11:16]; d = date.fromisoformat(ds)
        ref = float(p["underlying_at_entry"])
        if miss_thresh is not None:
            mm = int(minute[:2]) * 60 + int(minute[3:5]) - 1
            prev = parity(src, d, f"{mm//60:02d}:{mm%60:02d}", ref)
            cur = parity(src, d, minute, ref)
            if prev is not None and cur is not None and abs(cur - prev) > miss_thresh:
                skipped += 1
                continue
        call = m2_side(src, settle, d, minute, float(c["short_strike"]),
                       float(c["long_strike"]), True, qty)
        put = m2_side(src, settle, d, minute, float(p["short_strike"]),
                      float(p["long_strike"]), False, qty)
        if call is None or put is None:
            continue
        # IC margin = one-side max loss - TOTAL credit (broker charges one side only)
        ic_margin = max(call[2], put[2]) * MULT * qty - (call[1] + put[1]) * MULT * qty
        ic_margin = max(0.0, ic_margin)
        pnl[ds] += call[0] + put[0]; bp[ds] += ic_margin; bp1[ds] += ic_margin / qty
    dates = sorted(d for d in bp if bp[d] > 0)
    r = [pnl[d] / bp[d] for d in dates]
    return dates, r, bp1, sum(pnl.values()), skipped


def main():
    dates, r, bp1, total, _ = build()
    yrs = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    print("=" * 80)
    print(f"EXECUTION-REALISTIC SIZING (M2 fills: shorts@mid, wings@ask, cash settle)")
    print(f"  gross hold-to-expiry total: ${total:,.0f}  ({yrs:.1f}y)   "
          f"r: mean {statistics.mean(r):+.3f} median {statistics.median(r):+.3f} min {min(r):+.3f}")
    print("=" * 80)

    rng = RNG(); paths = block_bootstrap(r, 3000, 5, rng)
    print(f"  {'f':>5} {'worstday':>9} {'medCAGR':>8} {'medDD':>7} {'P95DD':>7} {'P99DD':>7}")
    chosen = None
    for f in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12):
        dds = []; cg = []
        for p in paths:
            mdd, eq = path_stats(p, f)
            dds.append(mdd); cg.append(eq ** (1/yrs) - 1 if eq > 0 else -1)
        dds.sort(); med = dds[len(dds)//2]
        mark = ""
        if chosen is None and med <= -0.20:
            chosen = (f, statistics.median(cg)); mark = "  <- ~20% DD"
        print(f"  {f:>5.3f} {f*min(r)*100:>8.0f}% {statistics.median(cg)*100:>7.0f}% "
              f"{med*100:>6.0f}% {dds[int(.05*len(dds))]*100:>6.0f}% {dds[int(.01*len(dds))]*100:>6.0f}%{mark}")
    f, cagr = chosen or (0.12, statistics.median([path_stats(p, 0.12)[1] ** (1/yrs) - 1
                         for p in paths if path_stats(p, 0.12)[1] > 0]))
    med_full = statistics.median(bp1.values())
    min_acct = med_full / f
    print(f"\n  POLICY @ 20% DD: f={f*100:.1f}%  ~{cagr*100:.0f}% CAGR  "
          f"worst day ~{f*min(r)*100:.0f}%  | full structure 1-lot ${med_full:,.0f}/day  "
          f"min account ${min_acct:,.0f}")
    print(f"  {'account':>9} {'BP/day':>9} {'full?':>16} {'CAGR':>7}")
    for acct in (50_000, 100_000, 250_000, 500_000):
        b = f * acct; lots = b / med_full
        if lots >= 1:
            note = f"yes ({lots:.1f}x)"; est = cagr
        else:
            naff = int(b / (med_full / 13)); note = f"no ({naff}/13 tr)"; est = cagr * naff / 13
        print(f"  ${acct:>8,} {b:>9,.0f} {note:>16} {est*100:>6.0f}%")

    # fill-certainty sensitivity
    print("\n  Missed-fill sensitivity (skip tranches whose entry minute moved > thresh):")
    for th in (2.0, 4.0):
        _, _, _, tot_m, sk = build(miss_thresh=th)
        print(f"    >{th:.0f}pt 1-min move skipped: {sk} tranches  ->  total ${tot_m:,.0f} "
              f"({100*(tot_m-total)/total:+.1f}% vs no-miss)")
    print("=" * 80)


if __name__ == "__main__":
    main()
