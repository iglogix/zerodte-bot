#!/usr/bin/env python3
"""Data-quality scan of the Databento CBBO cache at the minutes MEIC actually
traded. Flags garbage/locked/crossed ticks that could distort entry credits —
the one thing the same-data re-sim cannot catch. Output is a priority watch-list
for the independent Theta cross-check.

A quote (bid, ask) is flagged when:
  - no_market : ask <= 0
  - invalid   : bid < 0
  - crossed   : bid > ask
  - blown_out : (ask-bid) > $1.00 AND (ask-bid) > 1.5*mid   (e.g. 0.05/45.00)

Usage: .venv/bin/python tools/theta/scan_cache_quality.py
"""
from __future__ import annotations
import csv
from collections import defaultdict
from datetime import date
from reconcile_meic import DatabentoCache, TRADES_CSV


def classify(q):
    if q is None:
        return None  # missing handled separately
    bid, ask = q
    if ask <= 0:
        return "no_market"
    if bid < 0:
        return "invalid"
    if bid > ask:
        return "crossed"
    mid = (bid + ask) / 2
    if (ask - bid) > 1.0 and (ask - bid) > 1.5 * mid:
        return "blown_out"
    return None


def main():
    src = DatabentoCache()
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])

    n_legs = 0
    flagged_trades = []
    by_type = defaultdict(int)
    credit_flagged = credit_total = 0.0
    for r in rows:
        d = date.fromisoformat(r["session_date"])
        minute = r["entry_time"][11:16]
        is_call = r["spread_type"] == "call"
        credit = float(r["credit_received"])
        credit_total += credit
        hits = []
        for leg, k in (("short", float(r["short_strike"])), ("long", float(r["long_strike"]))):
            q = src.get(d, minute, k, is_call)
            n_legs += 1
            c = classify(q)
            if c:
                by_type[c] += 1
                hits.append((leg, k, q, c))
        if hits:
            credit_flagged += credit
            flagged_trades.append((r["trade_id"], r["spread_type"], credit,
                                   float(r["realized_pnl_usd"]), hits))

    print("=" * 78)
    print(f"CBBO CACHE DATA-QUALITY SCAN @ MEIC entry minutes  (cbbo_cache_meic)")
    print("=" * 78)
    print(f"  spreads scanned       : {len(rows)}   entry legs checked: {n_legs}")
    print(f"  anomalies by type     : {dict(by_type)}")
    print(f"  trades with >=1 bad leg: {len(flagged_trades)} "
          f"({100*len(flagged_trades)/len(rows):.2f}%)")
    print(f"  credit on flagged     : ${credit_flagged:,.1f} of ${credit_total:,.1f} "
          f"({100*credit_flagged/credit_total:.2f}%)")
    print("\n  Flagged trades (Theta cross-check priority list):")
    print(f"  {'trade_id':28s} {'type':4s} {'credit':>7s} {'pnl':>9s}  bad legs")
    for tid, st, cr, pnl, hits in sorted(flagged_trades, key=lambda x: -abs(x[3]))[:40]:
        legs = "; ".join(f"{leg} K{k:.0f} {q} [{c}]" for leg, k, q, c in hits)
        print(f"  {tid:28s} {st:4s} {cr:7.2f} {pnl:9.2f}  {legs}")
    print("=" * 78)


if __name__ == "__main__":
    main()
