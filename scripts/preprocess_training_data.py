#!/usr/bin/env python3
"""
Pre-process JSONL option data into numpy arrays for fast ANN training.

Streams one file at a time to avoid OOM on large datasets.
Subsequent training loads .npz in seconds instead of parsing JSONL for minutes.

Usage:
    python scripts/preprocess_training_data.py --symbols SPX --otm-only --v4
    python scripts/preprocess_training_data.py --symbols SPX --otm-only --v4 --em-filter 1.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_price_ann import (
    build_features, load_vix_daily, _compute_tte_hours,
    N_FEATURES, N_FEATURES_V4, N_FEATURES_V5, N_FEATURES_VOL, SYMBOL_ORDER,
)


FUTURES = {"ES", "NQ", "CL", "GC", "RTY", "ZN"}


def load_single_file(path: Path, symbols: set[str] | None,
                     otm_only: bool, em_filter: float,
                     vix_map: dict[str, float]) -> list[dict]:
    """Load and filter one JSONL file. Returns filtered records."""
    records = []
    n_skip = 0

    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_skip += 1
                continue

            # Normalize symbol
            sym = r.get("symbol", "")
            if not sym:
                fname = path.name.lower()
                if "_es" in fname or "es_" in fname:
                    sym = "ES"
                elif "_spx" in fname or "spx_" in fname:
                    sym = "SPX"
                else:
                    sym = "ES"
                r["symbol"] = sym

            if symbols and sym not in symbols:
                continue
            if sym not in SYMBOL_ORDER:
                n_skip += 1
                continue

            # Normalize fields
            spot = r.get("spot", 0) or r.get("underlying", 0)
            if "option_type" not in r and "type" in r:
                r["option_type"] = r["type"]

            bid = r.get("bid", 0)
            ask = r.get("ask", 0)
            mid = r.get("mid", 0)
            if not mid and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0

            if bid <= 0 or ask <= 0 or mid <= 0 or spot <= 0:
                n_skip += 1
                continue

            # Spread filter
            spread = ask - bid
            is_future = sym in FUTURES
            max_spread_pct = 0.15 if is_future else 0.50
            if mid > 0 and spread / mid > max_spread_pct:
                n_skip += 1
                continue

            strike = r.get("strike", 0)
            moneyness = strike / spot if spot > 0 else 0
            if moneyness < 0.90 or moneyness > 1.10:
                continue

            is_call = 1.0 if r.get("option_type") == "call" or r.get("is_call") == 1 else 0.0

            # OTM filter
            if otm_only:
                if is_call == 1.0 and moneyness <= 1.0:
                    continue
                if is_call == 0.0 and moneyness >= 1.0:
                    continue

            # TTE
            ts_str = r.get("timestamp", "")
            tte_hours = _compute_tte_hours(ts_str)
            if tte_hours is None or tte_hours <= 0:
                continue

            obs_date = ts_str[:10] if len(ts_str) >= 10 else ""

            # EM filter
            if em_filter > 0:
                vix = vix_map.get(obs_date, 0)
                if vix <= 0:
                    continue
                em = spot * (vix / 100.0) / math.sqrt(252) * math.sqrt(max(tte_hours, 0.01) / 6.5)
                if abs(strike - spot) > em_filter * em:
                    continue

            records.append({
                "symbol": sym,
                "timestamp": ts_str,
                "obs_date": obs_date,
                "strike": strike,
                "spot": spot,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "moneyness": moneyness,
                "is_call": is_call,
                "tte_hours": tte_hours,
                "market_iv": r.get("market_iv", 0),
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="Pre-process JSONL → numpy for ANN training")
    parser.add_argument("--data-dir", default="data/iv_collection", help="JSONL data directory")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help=f"Symbols to include. Choices: {SYMBOL_ORDER}")
    parser.add_argument("--otm-only", action="store_true", help="OTM options only")
    parser.add_argument("--v4", action="store_true", help="v4 features (VIX+vol+ToD)")
    parser.add_argument("--v5", action="store_true", help="v5 features")
    parser.add_argument("--vix-csv", default="data/vix_daily.csv", help="VIX daily CSV")
    parser.add_argument("--em-filter", type=float, default=0,
                        help="Keep only strikes within N × expected move (e.g. 1.5)")
    parser.add_argument("--out", default=None, help="Output .npz path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sym_set = set(s.upper() for s in args.symbols) if args.symbols else None

    v5 = args.v5
    if v5:
        args.v4 = True

    vix_map = load_vix_daily(args.vix_csv) if (args.v4 or args.em_filter > 0) else {}

    # Mode string for output filename
    mode_parts = []
    if args.otm_only:
        mode_parts.append("OTM")
    if v5:
        mode_parts.append("v5")
    elif args.v4:
        mode_parts.append("v4")
    if args.em_filter > 0:
        mode_parts.append(f"EM{args.em_filter}")
    if sym_set:
        mode_parts.append("+".join(sorted(sym_set)))
    mode = "_".join(mode_parts) if mode_parts else "all"

    # Stream file-by-file: load, filter, build features, collect numpy chunks
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files in {data_dir}")
        sys.exit(1)

    print(f"Streaming {len(files)} files from {data_dir} ({mode})...")

    X_chunks = []
    y_both_chunks = []
    y_bid_chunks = []
    y_ask_chunks = []
    meta_chunks = []  # list of dicts with arrays
    total_records = 0
    all_dates = set()
    sym_counts = defaultdict(int)

    for i, fpath in enumerate(files, 1):
        records = load_single_file(
            fpath, sym_set, args.otm_only, args.em_filter, vix_map
        )
        if not records:
            continue

        # Build features for this batch
        X, y_b = build_features(records, v4=args.v4, v5=v5, vix_map=vix_map, target="both")
        _, y_bid = build_features(records, v4=args.v4, v5=False, vix_map=vix_map, target="bid")
        _, y_ask = build_features(records, v4=args.v4, v5=False, vix_map=vix_map, target="ask")

        X_chunks.append(X)
        y_both_chunks.append(y_b)
        y_bid_chunks.append(y_bid)
        y_ask_chunks.append(y_ask)

        # Metadata arrays for this batch
        meta_chunks.append({
            "symbols": np.array([r["symbol"] for r in records], dtype="U8"),
            "spots": np.array([r["spot"] for r in records], dtype=np.float32),
            "strikes": np.array([r["strike"] for r in records], dtype=np.float32),
            "is_call": np.array([r["is_call"] for r in records], dtype=np.float32),
            "moneyness": np.array([r["moneyness"] for r in records], dtype=np.float32),
            "tte": np.array([r["tte_hours"] for r in records], dtype=np.float32),
            "dates": np.array([r["obs_date"] for r in records], dtype="U10"),
            "bids": np.array([r["bid"] for r in records], dtype=np.float32),
            "asks": np.array([r["ask"] for r in records], dtype=np.float32),
            "market_iv": np.array([r.get("market_iv", 0.0) for r in records], dtype=np.float32),
            "timestamps": np.array([r.get("timestamp", "") for r in records], dtype="U25"),
        })

        for r in records:
            sym_counts[r["symbol"]] += 1
            all_dates.add(r["obs_date"])
        total_records += len(records)

        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {total_records:,} records so far", flush=True)

        # Free records immediately
        del records

    if total_records < 100:
        print(f"ERROR: Only {total_records} records total")
        sys.exit(1)

    # Concatenate all chunks
    print(f"\nConcatenating {len(X_chunks)} chunks...")
    X_all = np.concatenate(X_chunks)
    y_both_all = np.concatenate(y_both_chunks)
    y_bid_all = np.concatenate(y_bid_chunks)
    y_ask_all = np.concatenate(y_ask_chunks)

    # Concatenate metadata
    def concat_meta(key):
        return np.concatenate([m[key] for m in meta_chunks])

    dates_sorted = sorted(all_dates)
    print(f"\nTotal: {total_records:,} records across {len(all_dates)} dates")
    for sym in SYMBOL_ORDER:
        if sym in sym_counts:
            print(f"  {sym}: {sym_counts[sym]:,}")
    print(f"  Dates: {dates_sorted[0]} -> {dates_sorted[-1]}")
    print(f"  X: {X_all.shape}  y_both: {y_both_all.shape}")

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(f"data/training_{mode}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X_all,
        y_both=y_both_all,
        y_bid=y_bid_all,
        y_ask=y_ask_all,
        v5=np.array([v5]),
        meta_symbols=concat_meta("symbols"),
        meta_spots=concat_meta("spots"),
        meta_strikes=concat_meta("strikes"),
        meta_is_call=concat_meta("is_call"),
        meta_moneyness=concat_meta("moneyness"),
        meta_tte=concat_meta("tte"),
        meta_dates=concat_meta("dates"),
        meta_bids=concat_meta("bids"),
        meta_asks=concat_meta("asks"),
        meta_market_iv=concat_meta("market_iv"),
        meta_timestamps=concat_meta("timestamps"),
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
    print(f"  {total_records:,} samples, {X_all.shape[1]} features")
    ver = "--v5" if v5 else ("--v4" if args.v4 else "")
    print(f"\nTo train:")
    print(f"  python scripts/train_price_ann.py --npz {out_path} --target bid --otm-only {ver}")
    print(f"  python scripts/train_price_ann.py --npz {out_path} --target ask --otm-only {ver}")


if __name__ == "__main__":
    main()
