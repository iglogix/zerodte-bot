#!/usr/bin/env python3
"""
Fast CBBO preprocessor — uses shell pipeline to pre-filter before Python parsing.

Each zst has ~7M rows but only ~150K are 0DTE. Shell grep filters 98% before
Python sees it, making it ~50x faster than pure Python.

Usage:
    python scripts/preprocess_cbbo_fast.py /path/to/OPRA-dir --out data/cbbo_cache
"""
from __future__ import annotations

import argparse
import csv
import io
import pickle
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")

SYM_RE = re.compile(r"SPXW\s+(\d{6})([CP])(\d{8})")


def extract_day(zst_path: Path, trading_date: date) -> Dict[Tuple[str, float, bool], Tuple[float, float]]:
    """Extract 0DTE quotes from a single zst file using shell pre-filter."""
    exp_str = trading_date.strftime("%y%m%d")  # "250922"

    # Shell pipeline: decompress → grep only 0DTE rows (fast C code)
    cmd = f"zstd -d --stdout '{zst_path}' | grep 'SPXW  {exp_str}'"
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )

    quotes: Dict[Tuple[str, float, bool], Tuple[float, float]] = {}

    # Get header from file separately (grep skips it)
    # We know the column positions: ts_recv=0, bid_px_00=9, ask_px_00=10, symbol=15
    for line in proc.stdout:
        parts = line.rstrip().split(",")
        if len(parts) < 16:
            continue

        ts_str = parts[0]
        bid_str = parts[9]
        ask_str = parts[10]
        sym_raw = parts[15]

        bid = float(bid_str) if bid_str else 0.0
        ask = float(ask_str) if ask_str else 0.0
        if bid <= 0 and ask <= 0:
            continue

        m = SYM_RE.match(sym_raw.strip())
        if not m:
            continue
        date_str, cp, strike_raw = m.groups()
        if date_str != exp_str:
            continue

        strike = int(strike_raw) / 1000.0
        is_call = cp == "C"

        # Parse timestamp: "2025-09-22T13:31:00.000000000Z" → ET minute
        try:
            dt_utc = datetime(
                int(ts_str[:4]), int(ts_str[5:7]), int(ts_str[8:10]),
                int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
                tzinfo=UTC,
            )
            dt_et = dt_utc.astimezone(ET)
        except (ValueError, IndexError):
            continue

        minute_key = f"{dt_et.hour:02d}:{dt_et.minute:02d}"
        quotes[(minute_key, strike, is_call)] = (bid, ask)

    proc.wait()
    return quotes


def main():
    parser = argparse.ArgumentParser(description="Fast CBBO pre-extraction")
    parser.add_argument("opra_dir", help="Directory with opra-pillar-*.cbbo-1m.csv.zst files")
    parser.add_argument("--out", default="data/cbbo_cache", help="Output directory")
    args = parser.parse_args()

    opra_dir = Path(args.opra_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zst_files = sorted(opra_dir.glob("opra-pillar-*.cbbo-1m.csv.zst"))
    print(f"Found {len(zst_files)} zst files in {opra_dir}")

    for i, zst_path in enumerate(zst_files, 1):
        # Extract date: "opra-pillar-20250922.cbbo-1m.csv.zst" → 2025-09-22
        ds = zst_path.name.split("-")[2].split(".")[0]
        try:
            trading_date = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        except ValueError:
            continue

        out_path = out_dir / f"{trading_date.isoformat()}.pkl"
        if out_path.exists():
            print(f"  [{i}/{len(zst_files)}] {trading_date} — cached")
            continue

        quotes = extract_day(zst_path, trading_date)
        if not quotes:
            print(f"  [{i}/{len(zst_files)}] {trading_date} — no 0DTE quotes")
            continue

        with open(out_path, "wb") as f:
            pickle.dump(quotes, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  [{i}/{len(zst_files)}] {trading_date} — {len(quotes):,} quotes")

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
