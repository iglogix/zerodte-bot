#!/usr/bin/env python3
"""Layer C — independent full re-simulation of the MEIC scale-out/stop engine.

Reconciles the part Layer A/B can't: the intermediate take-profit scale-out
fills (priced at short_ask + slippage) that the per-trade CSV discards. For
every spread we re-walk the minute grid from entry to EOD, reproduce the
strategy's stop + AM/PM scale-out tier logic against quotes, and compare the
resulting P&L and exit path to the recorded results.

Faithful to zerodte/strategies/meic.py (check_exit) and
zerodte/backtest/engine.py (_simulate_meic_side, stop_execution="trigger"):
  stop_price   = short_mid@entry + 0.70*total_credit         (total = put+call)
  stop hit     when short_ask >= stop_price
  stop debit   = min(0.70*total_credit + 0.25, width)        ("trigger" model)
  scale tier i fires when (credit - short_ask)/credit >= tier_i  (once each)
  scale qty    = floor(open_qty * fraction_i)  (per newly-fired tier this minute)
  scale debit  = short_ask + 0.10
  expiry       remaining qty kept at credit_eff (credit - 0.10)
  realized     starts at -2*qty*$1 commission

Same quote adapters as reconcile_meic.py, so --source theta swaps in later.

Usage:
  .venv/bin/python tools/theta/resim_meic.py --source databento
  .venv/bin/python tools/theta/resim_meic.py --source databento --limit 2000
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

from reconcile_meic import DatabentoCache, ThetaTerminal, TRADES_CSV

# Cost / strategy params (config/config_meic_spx.yaml)
SLIP = 0.10
STOP_SLIP = 0.25
COMM = 1.0
MULT = 100
STOP_PCT = 0.70
AM_CUTOFF = 12 * 60 + 30           # 12:30 ET in minutes
EOD = 15 * 60 + 50                 # 15:50 ET (engine breaks at >= eod)
AM_TIERS = [0.55, 0.65, 0.75, 0.85]
AM_FRACS = [0.25, 0.25, 0.25, 0.25]
PM_TIERS = [0.65, 0.85]
PM_FRACS = [0.50, 0.50]


def hhmm_to_min(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:5])


def min_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def resim_spread(row, total_credit, src):
    """Return (pnl, reason, debit, ok). ok=False if entry quote missing."""
    d = date.fromisoformat(row["session_date"])
    entry_m = hhmm_to_min(row["entry_time"][11:16])
    is_call = row["spread_type"] == "call"
    short_k = float(row["short_strike"])
    width = abs(float(row["long_strike"]) - short_k)
    credit = float(row["credit_received"])
    qty = int(row["qty"])
    credit_eff = credit - SLIP

    sq0 = src.get(d, min_to_hhmm(entry_m), short_k, is_call)
    if sq0 is None:
        return (None, None, None, False)
    short_mid_entry = (sq0[0] + sq0[1]) / 2
    stop_amount = round(STOP_PCT * total_credit, 4)
    stop_price = round(short_mid_entry + stop_amount, 4)

    tiers, fracs = (AM_TIERS, AM_FRACS) if entry_m < AM_CUTOFF else (PM_TIERS, PM_FRACS)
    fired: set = set()
    open_qty = qty
    realized = -2 * qty * COMM

    # Position is managed from the minute AFTER the fill; the engine does not
    # check exits on the entry bar itself (matters when the entry-minute quote
    # is a stale/garbage tick, e.g. 2025-04-09 13:20 short put 0.05/45.00).
    m = entry_m + 1
    while m < EOD:
        sq = src.get(d, min_to_hhmm(m), short_k, is_call)
        m += 1
        if sq is None:
            continue
        short_ask = sq[1]
        if short_ask <= 0:
            continue

        # --- stop (takes precedence) ---
        if short_ask >= stop_price:
            debit = min(max(stop_amount, 0.0) + STOP_SLIP, width)
            realized += (credit_eff - debit) * MULT * open_qty
            return (round(realized, 2), "short_stopped", round(debit, 2), True)

        # --- scale-out tiers ---
        pnl_pct = (credit - short_ask) / credit
        contracts = 0
        for i, tier in enumerate(tiers):
            if i in fired:
                continue
            if pnl_pct >= tier:
                fired.add(i)
                contracts += int(math.floor(open_qty * fracs[i]))
        if contracts <= 0:
            continue
        contracts = min(contracts, open_qty)
        debit = short_ask + SLIP
        realized += (credit_eff - debit) * MULT * contracts
        open_qty -= contracts
        if open_qty <= 0:
            return (round(realized, 2), "scaled_out", round(debit, 2), True)

    # --- EOD expiry of remaining ---
    realized += credit_eff * MULT * open_qty
    reason = "expired_worthless" if open_qty == qty else "partial_then_expiry"
    return (round(realized, 2), reason, 0.0, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["databento", "theta"], default="databento")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    src = DatabentoCache() if args.source == "databento" else ThetaTerminal()

    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    if args.limit:
        rows = rows[:args.limit]
    # total_credit per tranche (put+call share a stop based on combined credit)
    tot = defaultdict(float)
    for r in rows:
        tot[(r["session_date"], r["entry_time"])] += float(r["credit_received"])
    # process in date order so the day cache loads each pickle once
    rows.sort(key=lambda r: (r["session_date"], r["entry_time"]))

    n = ok = 0
    pnl_diffs, abs_diffs = [], []
    reason_match = 0
    confusion = defaultdict(int)           # (recorded, resim) -> count
    by_reason_absdiff = defaultdict(list)
    sum_rec = sum_res = 0.0
    worst = []
    for r in rows:
        tc = tot[(r["session_date"], r["entry_time"])]
        pnl, reason, debit, good = resim_spread(r, tc, src)
        n += 1
        if not good:
            continue
        ok += 1
        rec_pnl = float(r["realized_pnl_usd"])
        rec_reason = r["exit_reason"]
        diff = pnl - rec_pnl
        pnl_diffs.append(diff)
        abs_diffs.append(abs(diff))
        sum_rec += rec_pnl
        sum_res += pnl
        if reason == rec_reason:
            reason_match += 1
        confusion[(rec_reason, reason)] += 1
        by_reason_absdiff[rec_reason].append(abs(diff))
        worst.append((abs(diff), r["trade_id"], rec_reason, reason, rec_pnl, pnl))

    print("=" * 80)
    print(f"MEIC RE-SIMULATION (Layer C)  |  source={src.__class__.__name__}  |  spreads={n}")
    print("=" * 80)
    print(f"  simulated (entry quote found) : {ok}/{n}")
    print(f"\n  Headline P&L (sum of realized_pnl_usd):")
    print(f"     recorded : ${sum_rec:>14,.2f}")
    print(f"     re-sim   : ${sum_res:>14,.2f}")
    print(f"     diff     : ${sum_res - sum_rec:>14,.2f}  ({100*(sum_res-sum_rec)/abs(sum_rec):+.2f}%)")
    print(f"\n  Per-spread P&L agreement:")
    print(f"     mean diff (resim-rec)  : ${statistics.mean(pnl_diffs):+.2f}")
    print(f"     mean |diff|            : ${statistics.mean(abs_diffs):.2f}")
    print(f"     median |diff|          : ${statistics.median(abs_diffs):.2f}")
    print(f"     p95 |diff|             : ${sorted(abs_diffs)[int(0.95*len(abs_diffs))-1]:.2f}")
    print(f"     max |diff|             : ${max(abs_diffs):.2f}")
    for thr in (1, 10, 50):
        c = sum(1 for x in abs_diffs if x <= thr)
        print(f"     within ${thr:<3d}            : {c}/{len(abs_diffs)} ({100*c/len(abs_diffs):.1f}%)")
    print(f"\n  Exit-path match: {reason_match}/{ok} ({100*reason_match/ok:.1f}%)")
    print("  by recorded reason  |  mean |P&L diff|  |  exit-path match:")
    for reason in ("expired_worthless", "short_stopped", "partial_then_expiry", "scaled_out"):
        vals = by_reason_absdiff.get(reason, [])
        if not vals:
            continue
        same = confusion.get((reason, reason), 0)
        tot_r = sum(c for (rr, _), c in confusion.items() if rr == reason)
        print(f"     {reason:22s} n={len(vals):5d}  mean|d|=${statistics.mean(vals):7.2f}"
              f"  path={100*same/tot_r:5.1f}%")
    print("\n  Exit-path confusion (recorded -> resim), mismatches only:")
    for (rec, res), c in sorted(confusion.items(), key=lambda x: -x[1]):
        if rec != res:
            print(f"     {rec:22s} -> {res:22s} : {c}")
    print("\n  Worst 10 P&L mismatches:")
    for ad, tid, rr, sr, rp, sp in sorted(worst, reverse=True)[:10]:
        print(f"     {tid:26s} rec[{rr:18s}]=${rp:9.2f}  resim[{sr:18s}]=${sp:9.2f}  d=${sp-rp:+.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
