#!/usr/bin/env python3
"""Multi-expiry open-interest backfill for the GEX regime study.

For each trading day: pick study expiries (next dailies + monthlies/EOM <=120 DTE) from
the bar calendar, pull the OPRA `statistics` OPEN_INTEREST stat for SPXW strikes around
the 09:30 spot across those expiries, and write a per-day pickle
{(expiry, strike, is_call): open_interest} to data/oi_cache_multi/.

OI is published pre-market and reflects the prior session's close — exactly the standing
positioning carried into the session. Statistics is a tiny schema; run --dry-run first.

Usage:
  .venv/bin/python scripts/backfill_oi_multi.py --start 2022-05-11 --end 2026-06-13 --dry-run
  .venv/bin/python scripts/backfill_oi_multi.py --start 2022-05-11 --end 2026-06-13 --max-cost-usd 60
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import dotenv_values
import databento as db
from databento import StatType

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.databento_cost import CostLedger
from src_v2.backtest.meic_backfill import (multi_expiry_symbols, parse_osi_multi,
                                           meic_window_utc)
from src_v2.backtest.study_expiries import study_expiries

OI_STAT = int(StatType.OPEN_INTEREST)

MAX_SYMBOLS_PER_REQ = 1900  # under Databento's ~2000 symbols/request limit


def chunked(seq, size):
    """Yield successive `size`-length chunks of `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def oi_multi_from_df(df) -> Dict[Tuple[date, float, bool], int]:
    out: Dict[Tuple[date, float, bool], int] = {}
    sub = df[df["stat_type"] == OI_STAT]
    if "symbol" not in sub.columns:
        return out
    for sym, q in zip(sub["symbol"], sub["quantity"]):
        pk = parse_osi_multi(str(sym))
        if pk is None or q is None:
            continue
        try:
            out[pk] = int(q)
        except (ValueError, TypeError):
            continue
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-expiry SPXW OI backfill (GEX regime study)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--out", default="data/oi_cache_multi")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--buffer-down-pts", type=int, default=400)  # wider downside: flip sits below spot
    p.add_argument("--buffer-up-pts", type=int, default=200)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--n-dailies", type=int, default=5)
    p.add_argument("--max-dte-days", type=int, default=120)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=60.0)
    args = p.parse_args()

    env = dotenv_values(str(Path(__file__).resolve().parent.parent / ".env"))
    api_key = env.get("DATABENTO_API_KEY") or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    loader = BarLoader(args.bars, price_scale=args.price_scale, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    calendar = [d.session_date for d in days]
    print(f"{len(days)} trading days {start}..{end}")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    run_cost = 0.0
    n_calls = n_skipped = 0

    for day in days:
        d = day.session_date
        out_path = out_dir / f"{d.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue
        b0930 = next((b for b in day.bars if b["ts_et"].time().hour == 9
                      and b["ts_et"].time().minute == 30), None)
        if b0930 is None:
            n_skipped += 1
            continue
        center = float(b0930["open"])
        exps = study_expiries(d, calendar, n_dailies=args.n_dailies,
                              max_dte_days=args.max_dte_days)
        symbols = multi_expiry_symbols(exps, center, center,
                                       buffer_down_pts=args.buffer_down_pts,
                                       buffer_up_pts=args.buffer_up_pts,
                                       step_pts=args.step_pts)
        if not symbols:
            n_skipped += 1
            continue
        start_iso, end_iso = meic_window_utc(d, "04:00", "12:00")
        chunks = list(chunked(symbols, MAX_SYMBOLS_PER_REQ))
        cost = 0.0
        cost_err = False
        for ch in chunks:
            try:
                cost += client.metadata.get_cost(
                    dataset="OPRA.PILLAR", schema="statistics", symbols=ch,
                    stype_in="raw_symbol", start=start_iso, end=end_iso)
            except Exception as e:
                print(f"{d}: cost ERROR {type(e).__name__}: {str(e)[:80]}")
                cost_err = True
                break
        if cost_err:
            cost = 0.0

        if args.dry_run:
            run_cost += cost
            ledger.record(d, "dry_run", cost, len(symbols), 0)
            print(f"{d}: {len(exps):>2} exps {len(symbols):>5} syms {len(chunks)}ch  est ${cost:.5f}  cum ${run_cost:.3f}")
            continue

        if args.max_cost_usd and run_cost >= args.max_cost_usd:
            print(f"STOP: cum cost ${run_cost:.2f} >= cap ${args.max_cost_usd}")
            break
        oi = {}
        fetch_err = False
        for ch in chunks:
            try:
                df = client.timeseries.get_range(
                    dataset="OPRA.PILLAR", schema="statistics", symbols=ch,
                    stype_in="raw_symbol", start=start_iso, end=end_iso).to_df()
            except Exception as e:
                print(f"{d}: fetch ERROR {type(e).__name__}: {str(e)[:80]}")
                fetch_err = True
                break
            if len(df):
                oi.update(oi_multi_from_df(df))
        if fetch_err:
            continue
        with open(out_path, "wb") as f:
            pickle.dump(oi, f)
        run_cost += cost
        n_calls += 1
        ledger.record(d, "fetch", cost, len(symbols), len(oi))
        print(f"{d}: {len(exps):>2} exps {len(symbols):>5} syms {len(chunks)}ch {len(oi):>5} OI  ${cost:.5f}  cum ${run_cost:.3f}")

    print(f"done: {n_calls} fetched, {n_skipped} skipped, cum ${run_cost:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
