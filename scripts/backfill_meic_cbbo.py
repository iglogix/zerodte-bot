#!/usr/bin/env python3
"""Targeted 0DTE CBBO-1m backfill for the MEIC strategy.

For each trading day in [--start, --end], fetches the MEIC monitoring window
(default 11:00-15:50 ET) of cbbo-1m quotes for SPXW SAME-DAY-expiry strikes
spanning the day's spot range +/- a buffer, and writes a per-day pickle in the
CbboStore format ({(minute, strike, is_call): (bid, ask)}). The ATM is taken
DIRECTLY from the (SPY-derived) SPX spot, so this works for months with no
existing CBBO cache (unlike the ES-basis databento_backfill).

Modes:
  --dry-run         : metadata.get_cost (FREE) per day, print running total, exit
  (default)         : fetch, persist pickles, append spend to the cost ledger
  --max-cost-usd N  : abort the live run once accumulated cost exceeds N

Existing per-day pickles are skipped unless --force (the sparse legacy 2022-2024
files must be overwritten with --force to be usable).

Usage:
    # 1. FREE cost estimate for a sample:
    python scripts/backfill_meic_cbbo.py --start 2023-06-01 --end 2023-06-09 --dry-run
    # 2. Live fetch with a hard cost cap:
    python scripts/backfill_meic_cbbo.py --start 2022-05-11 --end 2025-09-19 \\
        --force --max-cost-usd 80
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import databento as db

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.databento_cost import estimate_request_cost, CostLedger
from src_v2.backtest.meic_backfill import (
    spot_range_from_bars, meic_day_symbols, meic_window_utc, quotes_from_df,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Targeted 0DTE MEIC CBBO backfill")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--out", default="data/cbbo_cache")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--buffer-pts", type=int, default=300,
                   help="Strike span beyond the day's spot range (covers OTM shorts + wings)")
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--window-start", default="11:00")
    p.add_argument("--window-end", default="15:50")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if a per-day pickle already exists")
    p.add_argument("--limit", type=int, default=0, help="Stop after N requests (0 = no limit)")
    p.add_argument("--sleep-ms", type=int, default=0, help="Sleep between live requests")
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost via metadata.get_cost (FREE) and exit")
    p.add_argument("--max-cost-usd", type=float, default=0.0,
                   help="Abort live run once accumulated cost exceeds this (0 = no cap)")
    args = p.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    loader = BarLoader(args.bars, price_scale=args.price_scale, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"{len(days)} trading days {start}..{end}")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    if ledger.total_usd("live") > 0:
        print(f"Existing ledger: ${ledger.total_usd('live'):.4f} live spend")

    mode = "dry_run" if args.dry_run else "live"
    n_calls = n_skipped = n_quotes = 0
    run_cost = 0.0

    for day in days:
        d = day.session_date
        out_path = out_dir / f"{d.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue

        rng = spot_range_from_bars(day.bars, args.window_start, args.window_end)
        if rng is None:
            n_skipped += 1
            continue
        lo, hi = rng
        symbols = meic_day_symbols(d, lo, hi, args.buffer_pts, args.step_pts)
        start_iso, end_iso = meic_window_utc(d, args.window_start, args.window_end)

        if args.dry_run:
            try:
                cost = estimate_request_cost(client, symbols, start_iso, end_iso)
            except Exception as e:
                print(f"{d}: cost-estimate ERROR {type(e).__name__}: {str(e)[:120]}")
                continue
            run_cost += cost
            ledger.record(d, "dry_run", cost, len(symbols), 0)
            print(f"{d}: {len(symbols):>4} symbols  est ${cost:.4f}  cum ${run_cost:.2f}")
        else:
            if args.max_cost_usd and run_cost >= args.max_cost_usd:
                print(f"Cost cap ${args.max_cost_usd:.2f} reached (cum ${run_cost:.2f}); stopping.")
                break
            try:
                resp = client.timeseries.get_range(
                    dataset="OPRA.PILLAR", schema="cbbo-1m",
                    symbols=symbols, stype_in="raw_symbol",
                    start=start_iso, end=end_iso,
                )
                df = resp.to_df()
            except Exception as e:
                print(f"{d}: fetch ERROR {type(e).__name__}: {str(e)[:120]}")
                continue
            quotes = quotes_from_df(df)
            with open(out_path, "wb") as f:
                pickle.dump(quotes, f)
            try:
                cost = estimate_request_cost(client, symbols, start_iso, end_iso)
            except Exception:
                cost = 0.0
            run_cost += cost
            n_quotes += len(quotes)
            ledger.record(d, "live", cost, len(symbols), len(quotes))
            print(f"{d}: {len(symbols):>4} symbols -> {len(quotes):>6} quotes  ${cost:.4f}  cum ${run_cost:.2f}")
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)

        n_calls += 1
        if args.limit and n_calls >= args.limit:
            print(f"Limit {args.limit} reached.")
            break

    print(f"\n{mode}: {n_calls} days processed, {n_skipped} skipped, "
          f"{n_quotes} quotes, cumulative cost ${run_cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
