#!/usr/bin/env python3
"""
Convert Databento SPXW CBBO-1m files to JSONL format for ANN training.

Extracts 0DTE options (expiry == trading date), derives spot via put-call
parity, and outputs one JSONL file per day in data/iv_collection/.

Usage:
    python scripts/convert_spx_cbbo.py /path/to/OPRA-dir
    python scripts/convert_spx_cbbo.py /path/to/OPRA-dir --all-dte  # include non-0DTE
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")

# Symbol format: "SPXW  YYMMDDCSSSSSSSS" or "SPXW  YYMMDDPSSSSSSSS"
# Example: "SPXW  250922P06445000" -> exp=2025-09-22, put, strike=6445.000
SYM_RE = re.compile(
    r"SPXW\s+(\d{6})([CP])(\d{8})"
)


def parse_symbol(raw: str) -> dict | None:
    """Parse SPXW option symbol into components."""
    m = SYM_RE.match(raw.strip())
    if not m:
        return None
    date_str, cp, strike_raw = m.groups()
    yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
    year = 2000 + yy
    strike = int(strike_raw) / 1000.0  # "06445000" -> 6445.0
    return {
        "expiry": date(year, mm, dd),
        "is_call": cp == "C",
        "strike": strike,
    }


def derive_spot_from_parity(records: list[dict]) -> dict[str, float]:
    """Derive implied spot per timestamp via put-call parity on ATM options.

    F = K + C_mid - P_mid (for same strike & expiry).
    Returns {timestamp_minute: median_implied_spot}.
    """
    # Group by (timestamp_minute, strike)
    pairs: dict[tuple[str, float], dict[str, float]] = {}
    for r in records:
        key = (r["timestamp"], r["strike"])
        if key not in pairs:
            pairs[key] = {}
        otype = "call" if r["is_call"] else "put"
        pairs[key][otype] = r["mid"]

    # Compute implied spot per timestamp
    group_spots: dict[str, list[float]] = defaultdict(list)
    for (ts, strike), data in pairs.items():
        if "call" in data and "put" in data:
            implied = strike + data["call"] - data["put"]
            if implied > 0:
                group_spots[ts].append(implied)

    return {ts: median(spots) for ts, spots in group_spots.items() if spots}


def process_day_file(zst_path: Path, zero_dte_only: bool = True) -> list[dict]:
    """Process one daily CBBO zstd file, return JSONL-ready records."""
    # Extract trading date from filename: opra-pillar-YYYYMMDD.cbbo-1m.csv.zst
    fname = zst_path.stem.replace(".cbbo-1m.csv", "")
    date_match = re.search(r"(\d{8})", fname)
    if not date_match:
        print(f"  SKIP {zst_path.name}: can't parse date")
        return []
    trading_date = date(
        int(date_match.group()[:4]),
        int(date_match.group()[4:6]),
        int(date_match.group()[6:8]),
    )

    # Decompress and read
    proc = subprocess.run(
        ["zstd", "-d", "-c", str(zst_path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        print(f"  SKIP {zst_path.name}: zstd error")
        return []

    lines = proc.stdout.splitlines()
    if not lines:
        return []

    reader = csv.DictReader(lines)
    raw_records = []

    for row in reader:
        sym_info = parse_symbol(row.get("symbol", ""))
        if sym_info is None:
            continue

        # Filter to 0DTE if requested
        if zero_dte_only and sym_info["expiry"] != trading_date:
            continue

        bid = float(row.get("bid_px_00") or 0)
        ask = float(row.get("ask_px_00") or 0)
        bid_sz = int(float(row.get("bid_sz_00") or 0))
        ask_sz = int(float(row.get("ask_sz_00") or 0))

        # Skip empty quotes
        if bid <= 0 or ask <= 0:
            continue
        # Skip crossed/locked markets
        if bid >= ask:
            continue

        mid = (bid + ask) / 2.0

        # Parse timestamp: convert UTC -> ET (trainer assumes ET)
        ts_raw = row.get("ts_recv", "")
        try:
            ts_utc = datetime.strptime(
                ts_raw[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=UTC)
            ts_et = ts_utc.astimezone(ET)
            ts_minute = ts_et.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue

        raw_records.append({
            "timestamp": ts_minute,
            "strike": sym_info["strike"],
            "is_call": sym_info["is_call"],
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "bid_size": bid_sz,
            "ask_size": ask_sz,
            "expiry": sym_info["expiry"].isoformat(),
        })

    if not raw_records:
        return []

    # Derive spot via put-call parity
    spot_map = derive_spot_from_parity(raw_records)
    if not spot_map:
        print(f"  SKIP {zst_path.name}: could not derive spot")
        return []

    # Attach spot to each record
    output = []
    n_no_spot = 0
    for r in raw_records:
        ts = r["timestamp"]
        spot = spot_map.get(ts)
        if spot is None:
            n_no_spot += 1
            continue

        # Spread filter: skip very wide markets (>50% of mid)
        spread_pct = (r["ask"] - r["bid"]) / r["mid"] if r["mid"] > 0 else 999
        if spread_pct > 0.50:
            continue

        # Moneyness filter: stay within ±10%
        moneyness = r["strike"] / spot
        if moneyness < 0.90 or moneyness > 1.10:
            continue

        dte = 0 if zero_dte_only else (
            (datetime.strptime(r["expiry"], "%Y-%m-%d").date() - trading_date).days
        )

        output.append({
            "symbol": "SPX",
            "timestamp": r["timestamp"],
            "strike": r["strike"],
            "spot": round(spot, 2),
            "bid": r["bid"],
            "ask": r["ask"],
            "mid": round(r["mid"], 4),
            "option_type": "call" if r["is_call"] else "put",
            "is_call": 1 if r["is_call"] else 0,
            "bid_size": r["bid_size"],
            "ask_size": r["ask_size"],
            "dte": dte,
        })

    return output


def main():
    parser = argparse.ArgumentParser(description="Convert SPXW CBBO to JSONL")
    parser.add_argument("data_dir", help="Path to OPRA download directory")
    parser.add_argument("--out-dir", default="data/iv_collection",
                        help="Output JSONL directory")
    parser.add_argument("--all-dte", action="store_true",
                        help="Include all DTEs, not just 0DTE")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zst_files = sorted(data_dir.glob("*.cbbo-1m.csv.zst"))
    if not zst_files:
        print(f"No .cbbo-1m.csv.zst files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(zst_files)} daily files")
    print(f"Mode: {'all DTE' if args.all_dte else '0DTE only'}")
    print(f"Output: {out_dir}/\n")

    total_records = 0
    total_days = 0

    for i, zst in enumerate(zst_files, 1):
        records = process_day_file(zst, zero_dte_only=not args.all_dte)
        if not records:
            print(f"  [{i}/{len(zst_files)}] {zst.name}: 0 records")
            continue

        # Extract date for filename
        date_match = re.search(r"(\d{8})", zst.name)
        day_str = date_match.group() if date_match else f"unknown_{i}"
        out_file = out_dir / f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}_spx.jsonl"

        with open(out_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        total_records += len(records)
        total_days += 1

        # Sample spot for this day
        spots = [r["spot"] for r in records]
        avg_spot = sum(spots) / len(spots) if spots else 0
        n_calls = sum(1 for r in records if r["is_call"] == 1)
        n_puts = len(records) - n_calls

        print(f"  [{i}/{len(zst_files)}] {zst.name}: {len(records):,} records "
              f"(C={n_calls}, P={n_puts})  spot~{avg_spot:.0f}  -> {out_file.name}")

    print(f"\nDone: {total_records:,} records across {total_days} days")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
