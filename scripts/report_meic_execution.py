#!/usr/bin/env python3
"""Report MEIC paper-trading execution: real stop fills vs backtest assumptions.

Reads logs/execution_comparison/<date>.jsonl and prints, per stop fill, the
slippage of the actual fill vs the assumed credit-relative trigger — and which
backtest stop model that reality implies.

Usage:
  python -m scripts.report_meic_execution                 # today
  python -m scripts.report_meic_execution --date 2026-05-22
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.services.execution_log import compare


def main():
    p = argparse.ArgumentParser(description="MEIC execution comparison report")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--dir", default="logs/execution_comparison")
    args = p.parse_args()

    path = Path(args.dir) / f"{args.date}.jsonl"
    if not path.exists():
        print(f"No execution log at {path}")
        return

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n_assumed = sum(1 for r in records if r.get("type") == "assumed")
    n_fills = sum(1 for r in records if r.get("type") == "fill")
    rows = compare(records)

    print(f"=== MEIC execution comparison {args.date} ===")
    print(f"  entries logged: {n_assumed}   fills logged: {n_fills}   stop fills: {len(rows)}\n")
    if not rows:
        print("  (no stop fills yet)")
        return

    print(f"  {'role':>11} {'symbol':>22} {'trigger':>8} {'actual':>8} {'slip':>7}  implied model")
    for r in rows:
        if r["match"] == "unmatched":
            print(f"  {'?':>11} {r['symbol']:>22} {'-':>8} {r['actual_fill']:>8.2f} {'-':>7}  UNMATCHED")
            continue
        print(f"  {r['role']:>11} {r['symbol']:>22} {r['stop_trigger']:>8.2f} "
              f"{r['actual_fill']:>8.2f} {r['slippage']:>7.2f}  {r['implied_model']}")

    matched = [r for r in rows if r["match"] == "matched" and r.get("slippage") is not None]
    if matched:
        slips = [r["slippage"] for r in matched]
        print(f"\n  mean stop slippage vs trigger: ${sum(slips)/len(slips):.2f}  "
              f"(min ${min(slips):.2f}, max ${max(slips):.2f})")
        models = Counter(r["implied_model"] for r in matched)
        print("  implied-model distribution:")
        for m, c in models.most_common():
            print(f"    {c:>3}  {m}")
        print("\n  Reminder: TT cert fills may be synthetic — slippage ~$0 across the "
              "board likely means the cert engine, not real execution.")


if __name__ == "__main__":
    main()
