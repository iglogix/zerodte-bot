#!/usr/bin/env python3
"""Targeted Databento CBBO-1m backfill driver.

Reads a signal manifest (JSONL produced by enumerate_signals.py), fetches CBBO
quotes for each (date, signal_minute, atm_strike) using the Databento API, and
writes per-day pickles into data/cbbo_cache/ in the format the existing
CbboStore reads.

Modes:
  --dry-run         : call metadata.get_cost (FREE) per day, print total, exit
  (default)         : fetch, persist pickles, append spend to cost ledger
  --max-cost-usd N  : abort mid-run if accumulated live cost exceeds N

Cost control: each request fetches only ~42 OSI symbols across a ~12-minute
window, so the full 5-year backfill should land in the $1-5 range. Always run
--dry-run first against a new manifest to confirm the bill.

Usage:
    # In .env: DATABENTO_API_KEY=db-...
    # 1. Estimate cost (free):
    python scripts/databento_backfill.py \\
        --manifest data/signal_manifest_orb_breakout.jsonl --dry-run
    # 2. Run for real, with safety cap:
    python scripts/databento_backfill.py \\
        --manifest data/signal_manifest_orb_breakout.jsonl \\
        --max-cost-usd 10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import databento as db

from zerodte.backtest.basis_estimator import (
    build_monthly_basis_table,
    estimate_spx_from_es,
)
from zerodte.backtest.databento_backfill import (
    fetch_day_quotes,
    build_request_window,
)
from zerodte.backtest.databento_cost import (
    estimate_request_cost,
    CostLedger,
)
from zerodte.backtest.osi_symbol import build_strike_window


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_manifest(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _es_opens_from_manifest(entries) -> Dict[date, float]:
    """First-seen es_price_at_signal per day — used as anchor for basis estimation.
    Good enough since basis is a monthly median, not per-day."""
    out: Dict[date, float] = {}
    for e in entries:
        d = _parse_date(e["date"])
        out.setdefault(d, e["es_price_at_signal"])
    return out


def _build_request_for_entry(entry, monthly_basis, default_basis, half_width_pts, step_pts,
                             before_min, after_min):
    """Returns (symbols_list, start_iso, end_iso, atm_strike) for one manifest entry."""
    day = _parse_date(entry["date"])
    atm = estimate_spx_from_es(day, entry["es_price_at_signal"], monthly_basis, default_basis)
    atm_snapped = round(atm / step_pts) * step_pts
    symbols = build_strike_window(
        expiry=day, atm_strike=float(atm_snapped),
        half_width_pts=half_width_pts, step_pts=step_pts,
    )
    start_iso, end_iso = build_request_window(
        day, entry["signal_minute_et"], before_min, after_min
    )
    return symbols, start_iso, end_iso, float(atm_snapped)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="data/cbbo_cache")
    p.add_argument("--cbbo-cache", default="data/cbbo_cache",
                   help="Existing CBBO pickles used to seed the basis table")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--before-min", type=int, default=2)
    p.add_argument("--after-min", type=int, default=10)
    p.add_argument("--half-width-pts", type=int, default=50)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--default-basis", type=float, default=10.0)
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if output pickle already exists")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N requests (0 = no limit)")
    p.add_argument("--sleep-ms", type=int, default=0,
                   help="Sleep N ms between requests to stay under rate limits")
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost via metadata.get_cost (FREE) and exit")
    p.add_argument("--max-cost-usd", type=float, default=0.0,
                   help="Abort if accumulated live cost exceeds this (0 = no cap)")
    args = p.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = list(_load_manifest(Path(args.manifest)))
    print(f"Loaded {len(entries)} signal events from manifest")

    es_opens = _es_opens_from_manifest(entries)
    monthly_basis = build_monthly_basis_table(Path(args.cbbo_cache), es_opens)
    print(f"Built basis table for {len(monthly_basis)} months")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    if ledger.total_usd("live") > 0:
        print(f"Existing ledger: ${ledger.total_usd('live'):.4f} live spend, "
              f"{ledger.n_records('live')} records")

    by_day: Dict[date, list] = defaultdict(list)
    for e in entries:
        by_day[_parse_date(e["date"])].append(e)

    mode = "dry_run" if args.dry_run else "live"
    n_calls = 0
    n_skipped = 0
    n_quotes = 0
    run_cost = 0.0

    for day in sorted(by_day.keys()):
        out_path = out_dir / f"{day.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue

        merged: Dict = {}
        day_cost = 0.0
        day_n_symbols = 0
        for entry in by_day[day]:
            symbols, start_iso, end_iso, _atm = _build_request_for_entry(
                entry, monthly_basis, args.default_basis,
                args.half_width_pts, args.step_pts,
                args.before_min, args.after_min,
            )

            try:
                est = estimate_request_cost(
                    client=client, symbols=symbols, start=start_iso, end=end_iso,
                )
            except Exception as exc:
                print(f"  [{day}] cost-estimate FAILED: {exc}", file=sys.stderr)
                continue

            day_cost += est
            day_n_symbols += len(symbols)

            if args.dry_run:
                run_cost += est
                continue

            # Live: enforce cost cap BEFORE making the call
            if args.max_cost_usd > 0 and (run_cost + est) > args.max_cost_usd:
                print(f"\nSTOP: next call would push run cost past --max-cost-usd "
                      f"${args.max_cost_usd:.2f} (current ${run_cost:.4f} + est ${est:.4f})")
                print(f"Run total: ${run_cost:.4f}, ledger total: ${ledger.total_usd('live'):.4f}")
                return 1

            try:
                quotes = fetch_day_quotes(
                    client=client,
                    session_date=day,
                    anchor_minute_et=entry["signal_minute_et"],
                    atm_strike=_atm,
                    half_width_pts=args.half_width_pts,
                    step_pts=args.step_pts,
                    before_min=args.before_min,
                    after_min=args.after_min,
                )
            except Exception as exc:
                print(f"  [{day}] fetch FAILED: {exc}", file=sys.stderr)
                continue

            merged.update(quotes)
            n_quotes += len(quotes)
            run_cost += est
            n_calls += 1
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
            if args.limit and n_calls >= args.limit:
                break

        # Record this day's cost (one ledger entry per day, sums all per-entry costs)
        ledger.record(day, mode, day_cost,
                     n_symbols=day_n_symbols, n_quotes=len(merged))

        if not args.dry_run and merged:
            with open(out_path, "wb") as f:
                pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [{day}] {len(merged):>6} quotes  ${day_cost:.4f}  → {out_path}")
        elif args.dry_run:
            print(f"  [{day}] est ${day_cost:.4f}  ({day_n_symbols} symbols across "
                  f"{len(by_day[day])} signal(s))")

        if args.limit and n_calls >= args.limit:
            print(f"\nReached --limit {args.limit}, stopping")
            break

    if args.dry_run:
        print(f"\nDRY RUN total estimate: ${run_cost:.4f} for {len(by_day)} days "
              f"(saved to {args.ledger} as 'dry_run' records — does NOT count against live total)")
    else:
        print(f"\nDone. {n_calls} API calls, {n_skipped} days skipped (cached), "
              f"{n_quotes} total quotes")
        print(f"Run cost: ${run_cost:.4f} | Ledger live total: "
              f"${ledger.total_usd('live'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
