#!/usr/bin/env python3
"""Open-interest backfill for the 0DTE SPXW GEX engine.

For each trading day, pulls the OPRA `statistics` schema (OPEN_INTEREST stat) for
the same-day SPXW strikes around the 09:30 spot and writes a per-day pickle
{(strike, is_call): open_interest} to data/oi_cache/. OI is published in the
morning and reflects the prior session's close — the dealer positioning carried
into the 0DTE session, exactly what GEX needs. Statistics is a tiny schema
(~$0.0002/day), so the full ~940-day range costs well under $1.

Usage:
  .venv/bin/python scripts/backfill_oi.py --start 2022-05-11 --end 2026-03-30 --dry-run
  .venv/bin/python scripts/backfill_oi.py --start 2022-05-11 --end 2026-03-30 --max-cost-usd 3
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import dotenv_values

import databento as db
from databento import StatType

from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.databento_cost import CostLedger
from zerodte.backtest.meic_backfill import meic_day_symbols, meic_window_utc

OI_STAT = int(StatType.OPEN_INTEREST)


def parse_osi(symbol: str) -> Optional[Tuple[float, bool]]:
    """OSI 'SPXW  240105P04700000' -> (strike=4700.0, is_call=False)."""
    s = symbol.strip()
    try:
        is_call = s[-9] == "C"
        strike = int(s[-8:]) / 1000.0
        return strike, is_call
    except (ValueError, IndexError):
        return None


def oi_from_df(df) -> Dict[Tuple[float, bool], int]:
    """Reduce a statistics df to {(strike, is_call): open_interest} (latest per symbol)."""
    out: Dict[Tuple[float, bool], int] = {}
    sub = df[df["stat_type"] == OI_STAT]
    if "symbol" not in sub.columns:
        return out
    # take the last reported OI per symbol (df is time-ordered by ts_recv index)
    for sym, q in zip(sub["symbol"], sub["quantity"]):
        pk = parse_osi(str(sym))
        if pk is None or q is None:
            continue
        try:
            out[pk] = int(q)
        except (ValueError, TypeError):
            continue
    return out


def day_center_symbols(bars: List[dict], session_date: date,
                       half_width_pts: int, step_pts: int) -> Tuple[Optional[float], List[str]]:
    b0930 = next((b for b in bars if b["ts_et"].time().hour == 9
                  and b["ts_et"].time().minute == 30), None)
    if b0930 is None:
        return None, []
    center = float(b0930["open"])
    return center, meic_day_symbols(session_date, center, center,
                                    buffer_pts=half_width_pts, step_pts=step_pts)


def main() -> int:
    p = argparse.ArgumentParser(description="0DTE SPXW open-interest backfill (GEX)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--out", default="data/oi_cache")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--half-width-pts", type=int, default=150)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=3.0)
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
    print(f"{len(days)} trading days {start}..{end}")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    n_calls = n_skipped = n_strikes = 0
    run_cost = 0.0

    for day in days:
        d = day.session_date
        out_path = out_dir / f"{d.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue
        center, symbols = day_center_symbols(day.bars, d, args.half_width_pts, args.step_pts)
        if center is None or not symbols:
            n_skipped += 1
            continue
        # OI is published pre-market; pull a morning window to capture it.
        start_iso, end_iso = meic_window_utc(d, "04:00", "12:00")

        try:
            cost = client.metadata.get_cost(
                dataset="OPRA.PILLAR", schema="statistics", symbols=symbols,
                stype_in="raw_symbol", start=start_iso, end=end_iso)
        except Exception as e:
            print(f"{d}: cost ERROR {type(e).__name__}: {str(e)[:100]}")
            cost = 0.0

        if args.dry_run:
            run_cost += cost
            ledger.record(d, "dry_run", cost, len(symbols), 0)
            print(f"{d}: {len(symbols):>4} syms  est ${cost:.5f}  cum ${run_cost:.3f}")
        else:
            if args.max_cost_usd and run_cost >= args.max_cost_usd:
                print(f"Cost cap ${args.max_cost_usd:.2f} reached (cum ${run_cost:.3f}); stopping.")
                break
            try:
                df = client.timeseries.get_range(
                    dataset="OPRA.PILLAR", schema="statistics", symbols=symbols,
                    stype_in="raw_symbol", start=start_iso, end=end_iso).to_df()
            except Exception as e:
                print(f"{d}: fetch ERROR {type(e).__name__}: {str(e)[:100]}")
                continue
            oi = oi_from_df(df) if len(df) else {}
            with open(out_path, "wb") as f:
                pickle.dump(oi, f)
            run_cost += cost
            n_strikes += len(oi)
            ledger.record(d, "live", cost, len(symbols), len(oi))
            print(f"{d}: {len(symbols):>4} syms -> {len(oi):>4} OI strikes  ${cost:.5f}  cum ${run_cost:.3f}")

        n_calls += 1
        if args.limit and n_calls >= args.limit:
            print(f"Limit {args.limit} reached.")
            break

    print(f"\n{'dry_run' if args.dry_run else 'live'}: {n_calls} days, {n_skipped} skipped, "
          f"{n_strikes} OI strikes, cumulative cost ${run_cost:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
