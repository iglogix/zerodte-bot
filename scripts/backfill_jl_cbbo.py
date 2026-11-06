#!/usr/bin/env python3
"""Targeted 0DTE SPXW CBBO-1m backfill for the Jade Lizard backtest.

For each trading day in [--start, --end], fetches the FULL session window
(09:30-16:00 ET) of cbbo-1m quotes for SPXW same-day strikes in a FIXED band
[center-100, center+100] where center = the 09:30 bar's spot (SPY-derived).
Fixed (non-rolling) centering keeps the entry-set legs covered all day. Writes a
per-day pickle {(minute, strike, is_call): (bid, ask)} to data/cbbo_cache_jl_10min/.

Modes:
  --dry-run         : metadata.get_cost (FREE) per day, print running total, exit
  (default)         : fetch, persist pickles, append spend to the cost ledger
  --max-cost-usd N  : abort the live run once accumulated cost exceeds N (default 15)

Usage:
  # 1. FREE cost estimate for the full range:
  .venv/bin/python scripts/backfill_jl_cbbo.py --start 2022-05-11 --end 2026-03-30 --dry-run
  # 2. Live fetch with a hard cap:
  .venv/bin/python scripts/backfill_jl_cbbo.py --start 2022-05-11 --end 2026-03-30 --max-cost-usd 15
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import databento as db

from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.databento_cost import estimate_request_cost, CostLedger
from zerodte.backtest.meic_backfill import (
    meic_day_symbols, meic_window_utc, quotes_from_df,
)


def jl_center_and_symbols(
    session_date: date, bars: List[dict], half_width_pts: int = 100, step_pts: int = 5,
) -> Tuple[Optional[float], List[str]]:
    """center = 09:30 bar spot; symbols span [center-half, center+half], calls+puts."""
    b0930 = next((b for b in bars if b["ts_et"].time().hour == 9
                  and b["ts_et"].time().minute == 30), None)
    if b0930 is None:
        return None, []
    center = float(b0930["open"])
    symbols = meic_day_symbols(session_date, center, center,
                               buffer_pts=half_width_pts, step_pts=step_pts)
    return center, symbols


def main() -> int:
    p = argparse.ArgumentParser(description="Targeted 0DTE SPXW Jade Lizard backfill")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--out", default="data/cbbo_cache_jl_10min")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--half-width-pts", type=int, default=100)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--window-start", default="09:30")
    p.add_argument("--window-end", default="16:00")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep-ms", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=15.0)
    args = p.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    loader = BarLoader(args.bars, price_scale=args.price_scale, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"{len(days)} trading days {start}..{end}")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    mode = "dry_run" if args.dry_run else "live"
    n_calls = n_skipped = n_quotes = 0
    run_cost = 0.0

    for day in days:
        d = day.session_date
        out_path = out_dir / f"{d.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue

        center, symbols = jl_center_and_symbols(
            d, day.bars, args.half_width_pts, args.step_pts)
        if center is None or not symbols:
            n_skipped += 1
            continue
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
