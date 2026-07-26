#!/usr/bin/env python3
"""Fetch a TRUE untouched holdout for Skyline-R: 2026-06-12..2026-07-24.

Nothing in this window was used to design the structure, the filters or the
thresholds — the working cache ends 2026-06-11. Centred on the ^GSPC open (a
price index, so no adjustment basis question); final strike selection still
comes from put-call parity inside the fetched chain.

Narrow window 09:30-09:50 only: entry is 09:45, the 09:32 parity spot supplies
the opening-drive filter, and settlement comes from the ^GSPC close.

  .venv/bin/python scripts/skyline_holdout_backfill.py --dry-run
  .venv/bin/python scripts/skyline_holdout_backfill.py --max-cost-usd 5
"""
from __future__ import annotations

import argparse
import os
import pickle
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()
import databento as db  # noqa: E402

from src_v2.backtest.databento_cost import estimate_request_cost, CostLedger  # noqa: E402
from src_v2.backtest.meic_backfill import (  # noqa: E402
    meic_day_symbols, meic_window_utc, quotes_from_df,
)

OUT = ROOT / "data" / "cbbo_cache_skyline_holdout"
DB = ROOT / "backtest_data" / "skyline.db"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-06-12")
    p.add_argument("--end", default="2026-07-24")
    p.add_argument("--half-width-pts", type=int, default=120)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--window-start", default="09:30")
    p.add_argument("--window-end", default="09:50")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=5.0)
    a = p.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        print("ERROR: DATABENTO_API_KEY not set", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(DB)
    days = [(date.fromisoformat(d), o) for d, o in con.execute(
        "SELECT d, open FROM gspc_daily WHERE d>=? AND d<=? ORDER BY d",
        (a.start, a.end))]
    con.close()
    print(f"{len(days)} sessions {a.start}..{a.end}  band +/-{a.half_width_pts}pt "
          f"@{a.step_pts}pt  window {a.window_start}-{a.window_end}")

    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    ledger = CostLedger(ROOT / "data" / "databento_cost_ledger.jsonl")
    run_cost = 0.0
    n_done = n_skip = 0

    for d, gopen in days:
        out_path = OUT / f"{d.isoformat()}.pkl"
        if not a.dry_run and out_path.exists():
            n_skip += 1
            continue
        symbols = meic_day_symbols(d, gopen, gopen,
                                   buffer_pts=a.half_width_pts, step_pts=a.step_pts)
        s_iso, e_iso = meic_window_utc(d, a.window_start, a.window_end)
        if a.dry_run:
            try:
                c = estimate_request_cost(client, symbols, s_iso, e_iso)
            except Exception as exc:
                print(f"{d}: estimate ERROR {type(exc).__name__}: {str(exc)[:100]}")
                continue
            run_cost += c
            print(f"{d}: {len(symbols):>4} symbols  est ${c:.4f}  cum ${run_cost:.3f}")
            continue
        if run_cost >= a.max_cost_usd:
            print(f"cost cap ${a.max_cost_usd} reached (cum ${run_cost:.2f}); stopping")
            break
        c = estimate_request_cost(client, symbols, s_iso, e_iso)
        data = client.timeseries.get_range(
            dataset="OPRA.PILLAR", schema="cbbo-1m", symbols=symbols,
            stype_in="raw_symbol", start=s_iso, end=e_iso)
        quotes = quotes_from_df(data.to_df())
        with open(out_path, "wb") as fh:
            pickle.dump(quotes, fh)
        run_cost += c
        ledger.record(d, "live", c, len(symbols), len(quotes))
        n_done += 1
        print(f"{d}: {len(symbols):>4} symbols -> {len(quotes):>6} quotes  "
              f"${c:.4f}  cum ${run_cost:.3f}")

    print(f"\n{'DRY RUN' if a.dry_run else 'FETCHED'}: total est/actual "
          f"${run_cost:.3f}   fetched={n_done} skipped={n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
