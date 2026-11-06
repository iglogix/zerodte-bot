"""Backfill the MORNING CBBO (09:30-11:00 ET) for SPX 0DTE days whose cache
pickles were only populated 11:00+ by the Power Hour backfill.

ORB IC enters at 09:45/10:15/11:15... so it needs full-session quotes. The
afternoon (11:00-15:49) already exists for these days; this script fetches only
the missing morning window and MERGES it into the existing pickle (union of
keys), so Power Hour's afternoon coverage is preserved.

Usage:
    .venv/bin/python scripts/backfill_orb_ic_morning.py [--dry-run] [--start 2026-03-23] [--end 2026-06-11]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.databento_backfill import fetch_day_quotes
from zerodte.backtest.databento_cost import CostLedger, estimate_request_cost
from zerodte.backtest.osi_symbol import build_strike_window

ET = ZoneInfo("America/New_York")
CACHE = Path("data/cbbo_cache")
TRUNC_THRESHOLD = "10:30"   # min cached minute >= this => morning is missing
HALF_WIDTH = 250            # +/- 250 pts around open ATM (covers SPY/SPX basis + gap)
STEP = 5


def _load_env_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("DATABENTO_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("DATABENTO_API_KEY not found in env or .env")
    return key


def _open_atm(spy: pd.DataFrame, d: date) -> float | None:
    """SPX-proxy spot at 09:30 ET (SPY x10), rounded to nearest 5."""
    day = spy[spy["et_date"] == d]
    if day.empty:
        return None
    # 09:30 ET bar (or first RTH bar >= 09:30)
    rth = day[day["et_min"] >= "09:30"]
    if rth.empty:
        return None
    spot = float(rth.iloc[0]["close"]) * 10.0
    return round(spot / 5.0) * 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="cost-estimate only, no fetch")
    ap.add_argument("--start", default="2026-03-21")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--spy", default="data/spy_1m.csv")
    args = ap.parse_args()

    import databento as db
    client = db.Historical(_load_env_key())
    ledger = CostLedger(Path("results/databento_spend.jsonl"))

    # which cached days are truncated (missing morning)?
    targets: list[date] = []
    for p in sorted(CACHE.glob("*.pkl")):
        d = p.stem
        if d < args.start or d > args.end:
            continue
        obj = pickle.load(open(p, "rb"))
        tmin = min(k[0] for k in obj.keys())
        if tmin >= TRUNC_THRESHOLD:
            targets.append(date.fromisoformat(d))
        del obj
    print(f"{len(targets)} truncated days to backfill ({args.start}..{args.end})", flush=True)
    if not targets:
        return 0

    # SPY bars for open ATM
    spy = pd.read_csv(args.spy)
    ts = pd.to_datetime(spy["ts_event"], utc=True).dt.tz_convert(ET)
    spy["et_date"] = ts.dt.date
    spy["et_min"] = ts.dt.strftime("%H:%M")

    total_cost = 0.0
    done = 0
    for d in targets:
        atm = _open_atm(spy, d)
        if atm is None:
            print(f"  {d}: no SPY open bar, SKIP", flush=True)
            continue
        syms = build_strike_window(expiry=d, atm_strike=atm, half_width_pts=HALF_WIDTH, step_pts=STEP)
        # morning window 09:30-11:00 via anchor 10:15 +/- 45m
        if args.dry_run:
            start_iso = f"{d.isoformat()}T13:30:00"
            end_iso = f"{d.isoformat()}T15:00:00"
            c = estimate_request_cost(client, syms, start_iso, end_iso)
            total_cost += c
            ledger.record(d, "dry_run", c, len(syms), 0)
            print(f"  {d}: atm~{atm:.0f} n_sym={len(syms)} est=${c:.4f}", flush=True)
            continue

        q = fetch_day_quotes(
            client, d, anchor_minute_et="10:15", atm_strike=atm,
            half_width_pts=HALF_WIDTH, step_pts=STEP, before_min=45, after_min=45,
        )
        # keep only morning minutes (< 11:00) to avoid touching existing afternoon
        morning = {k: v for k, v in q.items() if k[0] < "11:00"}
        path = CACHE / f"{d.isoformat()}.pkl"
        existing = pickle.load(open(path, "rb"))
        before_keys = len(existing)
        existing.update(morning)
        pickle.dump(existing, open(path, "wb"))
        new_times = sorted({k[0] for k in morning})
        ledger.record(d, "live", 0.0, len(syms), len(morning))
        done += 1
        print(
            f"  {d}: +{len(morning)} morning quotes ({len(new_times)} min "
            f"{new_times[0] if new_times else '-'}..{new_times[-1] if new_times else '-'}) "
            f"keys {before_keys}->{len(existing)}",
            flush=True,
        )

    if args.dry_run:
        print(f"\nDRY RUN total estimated cost: ${total_cost:.2f} for {len(targets)} days")
    else:
        print(f"\nBackfilled {done}/{len(targets)} days. Spend ledger: results/databento_spend.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
