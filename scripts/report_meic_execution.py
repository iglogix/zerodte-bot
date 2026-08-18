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

from src_v2.services.execution_log import compare, compare_quotes


def main():
    p = argparse.ArgumentParser(description="MEIC execution comparison report")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--dir", default="logs/execution_comparison/meic")
    args = p.parse_args()

    path = Path(args.dir) / f"{args.date}.jsonl"
    if not path.exists():
        print(f"No execution log at {path}")
        return

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n_assumed = sum(1 for r in records if r.get("type") == "assumed")
    n_fills = sum(1 for r in records if r.get("type") == "fill")
    n_quotes = sum(1 for r in records if r.get("type") == "quote")

    print(f"=== MEIC execution comparison {args.date} ===")
    print(f"  entries: {n_assumed}   fills: {n_fills}   quote polls: {n_quotes}\n")

    # --- Manual-mode: quote-based stop-cross comparison ---
    qrows = compare_quotes(records)
    if qrows:
        print("--- Quote-based (manual mode): where the short's real ask crossed the trigger ---")
        print(f"  {'role':>11} {'trigger':>8} {'maxAsk':>8} {'crossed':>8} {'overshoot':>9}")
        crossed = [r for r in qrows if r["crossed"]]
        for r in qrows:
            ov = "-" if r["overshoot_at_cross"] is None else f"{r['overshoot_at_cross']:.2f}"
            trig = "-" if r["trigger"] is None else f"{r['trigger']:.2f}"
            mx = "-" if r["max_ask"] is None else f"{r['max_ask']:.2f}"
            print(f"  {r['role']:>11} {trig:>8} {mx:>8} {str(r['crossed']):>8} {ov:>9}")
        print(f"  short legs tracked: {len(qrows)}   crossed trigger: {len(crossed)}")
        ov = [r["overshoot_at_cross"] for r in crossed if r["overshoot_at_cross"] is not None]
        if ov:
            print(f"  mean overshoot past trigger at first cross: ${sum(ov)/len(ov):.2f} "
                  f"(max ${max(ov):.2f}) — this is the real stop-fill slippage the "
                  f"backtest's 'trigger' model ignored.")
        print()

    # --- Live/sandbox: fill-based comparison ---
    rows = compare(records)
    print(f"--- Fill-based (live/sandbox): stop fills vs trigger ({len(rows)}) ---")
    if not rows:
        print("  (no stop fills logged)")
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
