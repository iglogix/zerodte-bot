#!/usr/bin/env python3
"""Add a mechanical (no-lookahead) entry filter to the winning hold-to-expiry,
no-scale-out exit, and sweep its tightness.

Filter: skip a tranche (both put+call) when the underlying has already moved
more than k x EM from the day's first-tranche reference by the entry time. This
targets trending/breach days without hindsight (move-so-far is known at entry).

Day EM = ATM straddle (Cmid+Pmid at the strike nearest spot) at the first
tranche minute. Settlement + hold-to-expiry P&L identical to resim_exits.py.

Usage: .venv/bin/python tools/theta/resim_exit_filter.py
"""
from __future__ import annotations
import csv
import statistics
from collections import defaultdict
from datetime import date
from reconcile_meic import DatabentoCache, TRADES_CSV
from resim_exits import Settlement, sim


def day_em(src, d, minute, ref_spot):
    """ATM straddle at `minute` (nearest strike to ref_spot) = expected move."""
    src._load(d)
    q = src._q
    if not q:
        return None
    strikes = sorted({k[1] for k in q if k[0] == minute})
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - ref_spot))
    c = q.get((minute, atm, True)); p = q.get((minute, atm, False))
    if not c or not p:
        return None
    return (c[0] + c[1]) / 2 + (p[0] + p[1]) / 2


def main():
    src = DatabentoCache(); settle = Settlement(src)
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])

    # group by date -> tranches (entry_time) with their rows
    by_day = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_day[r["session_date"]][r["entry_time"]].append(r)

    # precompute per-day reference spot (earliest tranche) + EM, and the
    # cumulative move at each tranche
    day_ref = {}    # date -> (ref_spot, em)
    for ds, tranches in by_day.items():
        first_et = min(tranches)
        ref_row = tranches[first_et][0]
        ref_spot = float(ref_row["underlying_at_entry"])
        minute = first_et[11:16]
        em = day_em(src, date.fromisoformat(ds), minute, ref_spot)
        day_ref[ds] = (ref_spot, em)

    def run(k):
        sess = defaultdict(float); yr = defaultdict(float)
        kept = skipped = 0; pnl = []
        for ds, tranches in by_day.items():
            ref_spot, em = day_ref[ds]
            for et, trrows in tranches.items():
                move = abs(float(trrows[0]["underlying_at_entry"]) - ref_spot)
                if k is not None and em and move > k * em:
                    skipped += 1
                    continue
                kept += 1
                for r in trrows:
                    v = sim(r, src, settle, keep_scaleout=False)
                    if v is None:
                        continue
                    pnl.append(v); sess[ds] += v; yr[ds[:4]] += v
        dates = sorted(sess); eq = peak = mdd = 0
        worst_day = min(sess.values()) if sess else 0
        for d in dates:
            eq += sess[d]; peak = max(peak, eq); mdd = min(mdd, eq - peak)
        return dict(total=sum(pnl), worst_day=worst_day, mdd=mdd,
                    kept=kept, skipped=skipped, n=len(pnl), yr=yr,
                    bad50=sum(1 for v in sess.values() if v < -50000))

    print("=" * 92)
    print("HOLD-to-expiry, NO scale-outs + intraday move-so-far entry filter (skip if move > k*EM)")
    print("=" * 92)
    print(f"{'k*EM':>6} {'total P&L':>13} {'worst day':>12} {'max DD':>12} "
          f"{'tranches kept':>14} {'days<-50k':>10}")
    print("-" * 92)
    for k in (None, 2.0, 1.5, 1.0, 0.75, 0.5):
        r = run(k)
        label = "none" if k is None else f"{k:.2f}"
        tot_tr = r["kept"] + r["skipped"]
        print(f"{label:>6} {r['total']:>13,.0f} {r['worst_day']:>12,.0f} {r['mdd']:>12,.0f} "
              f"{r['kept']:>7}/{tot_tr:<6} {r['bad50']:>10}")
    print("-" * 92)
    print("baselines (no filter): trigger=+$507,886  combo=-$2,892,190  short_only=-$9,422,910")
    # by-year for the most promising filter
    print("\nby-year detail:")
    for k in (None, 1.0, 0.75):
        r = run(k); lab = "none" if k is None else f"{k:.2f}"
        print(f"  k={lab:>4}: " + "  ".join(f"{y}:${r['yr'][y]:,.0f}" for y in sorted(r['yr'])))
    print("=" * 92)


if __name__ == "__main__":
    main()
