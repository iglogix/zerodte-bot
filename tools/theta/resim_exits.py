#!/usr/bin/env python3
"""Explore alternative MEIC exits under HONEST fills, on the same entries.

The trigger-stop +$507k is fantasy; combo = -$2.9M, short_only = -$9.4M. Those
all *stop*. The spreads are defined-risk (long wing caps loss at width), so the
stop is actually ADDING risk. This tests removing the stop entirely:

  hold_keep_scaleout : no stop; keep AM/PM take-profit scale-outs; remaining qty
                       cash-settles at expiry intrinsic (loss capped at width).
  hold_no_scaleout   : no stop, no scale-out; pure hold to settlement.

Settlement forward F per day = ATM put/call parity at the last cached minute
(F = K + Cmid - Pmid at the strike minimising |Cmid - Pmid|), consistent with
the option strikes (avoids the SPY x10 basis). Cash settlement => no closing
bid/ask slippage (best case for hold-to-expiry).

Usage: .venv/bin/python tools/theta/resim_exits.py
"""
from __future__ import annotations
import csv
import math
import statistics
from collections import defaultdict
from datetime import date
from reconcile_meic import DatabentoCache, TRADES_CSV

SLIP = 0.10
COMM = 1.0
MULT = 100
AM_CUTOFF = 12 * 60 + 30
EOD = 15 * 60 + 50
AM_TIERS, AM_FRACS = [0.55, 0.65, 0.75, 0.85], [0.25, 0.25, 0.25, 0.25]
PM_TIERS, PM_FRACS = [0.65, 0.85], [0.50, 0.50]


def hhmm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


class Settlement:
    """Per-day settlement forward via ATM parity at the last cached minute."""
    def __init__(self, src):
        self.src = src
        self.cache = {}

    def forward(self, d):
        if d in self.cache:
            return self.cache[d]
        self.src._load(d)
        q = self.src._q
        if not q:
            self.cache[d] = None
            return None
        last_min = max(int(k[0][:2]) * 60 + int(k[0][3:5]) for k in q)
        m = hhmm(last_min)
        strikes = sorted({k[1] for k in q if k[0] == m})
        best = None
        for k in strikes:
            c = q.get((m, k, True)); p = q.get((m, k, False))
            if not c or not p:
                continue
            cmid = (c[0] + c[1]) / 2; pmid = (p[0] + p[1]) / 2
            diff = abs(cmid - pmid)
            if best is None or diff < best[0]:
                best = (diff, k + cmid - pmid)
        F = best[1] if best else None
        self.cache[d] = F
        return F


def sim(row, src, settle, keep_scaleout):
    d = date.fromisoformat(row["session_date"])
    entry_m = int(row["entry_time"][11:13]) * 60 + int(row["entry_time"][14:16])
    is_call = row["spread_type"] == "call"
    short_k = float(row["short_strike"])
    width = abs(float(row["long_strike"]) - short_k)
    credit = float(row["credit_received"])
    qty = int(row["qty"])
    credit_eff = credit - SLIP
    realized = -2 * qty * COMM
    open_qty = qty

    if keep_scaleout:
        tiers, fracs = (AM_TIERS, AM_FRACS) if entry_m < AM_CUTOFF else (PM_TIERS, PM_FRACS)
        fired = set()
        m = entry_m + 1
        while m < EOD and open_qty > 0:
            sq = src.get(d, hhmm(m), short_k, is_call)
            m += 1
            if not sq or sq[1] <= 0:
                continue
            short_ask = sq[1]
            pnl_pct = (credit - short_ask) / credit
            contracts = 0
            for i, t in enumerate(tiers):
                if i not in fired and pnl_pct >= t:
                    fired.add(i)
                    contracts += int(math.floor(open_qty * fracs[i]))
            if contracts <= 0:
                continue
            contracts = min(contracts, open_qty)
            debit = short_ask + SLIP
            realized += (credit_eff - debit) * MULT * contracts
            open_qty -= contracts

    # remaining cash-settles at expiry intrinsic (capped at width)
    if open_qty > 0:
        F = settle.forward(d)
        if F is None:
            return None
        intrinsic = max(0.0, (F - short_k) if is_call else (short_k - F))
        loss = min(intrinsic, width)
        realized += (credit_eff - loss) * MULT * open_qty
    return round(realized, 2)


def report(name, rows, src, settle, keep_scaleout):
    sess = defaultdict(float); yr = defaultdict(float)
    pnl = []; n = 0
    for r in rows:
        v = sim(r, src, settle, keep_scaleout)
        if v is None:
            continue
        n += 1
        pnl.append(v); sess[r["session_date"]] += v; yr[r["session_date"][:4]] += v
    total = sum(pnl)
    wins = sum(1 for p in pnl if p > 0)
    dates = sorted(sess)
    eq = peak = mdd = 0
    for d in dates:
        eq += sess[d]; peak = max(peak, eq); mdd = min(mdd, eq - peak)
    print(f"\n### {name}  (spreads={n})")
    print(f"   TOTAL P&L : ${total:,.0f}    per-spread ${total/n:,.2f}    "
          f"spread WR {100*wins/n:.1f}%")
    print(f"   max DD    : ${mdd:,.0f}    sessions: {len(dates)}  green {100*sum(1 for v in sess.values() if v>0)/len(dates):.0f}%")
    print(f"   by year   : " + "  ".join(f"{y}:${yr[y]:,.0f}" for y in sorted(yr)))
    return total


def main():
    src = DatabentoCache()
    settle = Settlement(src)
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])
    print("=" * 78)
    print("MEIC ALTERNATIVE EXITS — honest fills, same entries")
    print("  baselines: trigger(optimistic)=+$507,886  combo=-$2,892,190  short_only=-$9,422,910")
    print("=" * 78)
    report("HOLD to expiry, keep take-profit scale-outs", rows, src, settle, True)
    report("HOLD to expiry, NO scale-outs (pure defined-risk hold)", rows, src, settle, False)
    print("=" * 78)


if __name__ == "__main__":
    main()
