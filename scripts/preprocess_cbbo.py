#!/usr/bin/env python3
"""
Pre-extract CBBO zst files into fast-loading pickle files for backtesting.

Reads each day's zst, filters to 0DTE SPXW, and saves a dict:
  {(minute_str, strike, is_call): (bid, ask)}

Usage:
    python scripts/preprocess_cbbo.py /path/to/OPRA-dir --out data/cbbo_cache
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from es_8am.backtest.cbbo_store import CbboStore


def main():
    parser = argparse.ArgumentParser(description="Pre-extract CBBO zst → pickle")
    parser.add_argument("opra_dir", help="Directory with opra-pillar-*.cbbo-1m.csv.zst files")
    parser.add_argument("--out", default="data/cbbo_cache", help="Output directory for pickle files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = CbboStore(args.opra_dir)
    days = store.available_days()
    print(f"Found {len(days)} trading days in {args.opra_dir}")

    total_quotes = 0
    for i, d in enumerate(days, 1):
        out_path = out_dir / f"{d.isoformat()}.pkl"
        if out_path.exists():
            print(f"  [{i}/{len(days)}] {d} — already cached, skipping")
            continue

        n = store.load_day(d)
        if n == 0:
            print(f"  [{i}/{len(days)}] {d} — no 0DTE quotes, skipping")
            continue

        with open(out_path, "wb") as f:
            pickle.dump(store._quotes, f, protocol=pickle.HIGHEST_PROTOCOL)

        total_quotes += n
        print(f"  [{i}/{len(days)}] {d} — {n:,} quotes → {out_path.name}")

    print(f"\nDone. {total_quotes:,} total quotes across {len(days)} days → {out_dir}")


if __name__ == "__main__":
    main()
