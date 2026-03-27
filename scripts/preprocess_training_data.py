#!/usr/bin/env python3
"""
Pre-process JSONL option data into numpy arrays for fast ANN training.

Reads JSONL files, applies filters, computes features, and saves as .npz.
Subsequent training loads .npz in seconds instead of parsing JSONL for minutes.

Usage:
    # Full preprocessing (all symbols, all options)
    python scripts/preprocess_training_data.py

    # SPX OTM only with v4 features
    python scripts/preprocess_training_data.py --symbols SPX --otm-only --v4

    # With max records cap
    python scripts/preprocess_training_data.py --symbols SPX --otm-only --v4 --max-records 1000000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import from training script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_price_ann import (
    load_jsonl_files, build_features, load_vix_daily,
    N_FEATURES, N_FEATURES_V4, N_FEATURES_V5, N_FEATURES_VOL, SYMBOL_ORDER,
    _compute_tte_hours,
)
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Pre-process JSONL → numpy for ANN training")
    parser.add_argument("--data-dir", default="data/iv_collection", help="JSONL data directory")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help=f"Symbols to include. Choices: {SYMBOL_ORDER}")
    parser.add_argument("--otm-only", action="store_true", help="OTM options only")
    parser.add_argument("--v4", action="store_true", help="v4 features (VIX+vol+ToD)")
    parser.add_argument("--v5", action="store_true", help="v5 features (v4 + atm_iv, skew, vol_5min + BS-residual targets)")
    parser.add_argument("--vix-csv", default="data/vix_daily.csv", help="VIX daily CSV")
    parser.add_argument("--max-records", type=int, default=0, help="Max records (0=all)")
    parser.add_argument("--em-filter", type=float, default=0,
                        help="Keep only strikes within N × expected move of spot (e.g. 1.5). Requires VIX data.")
    parser.add_argument("--out", default=None, help="Output .npz path (auto-generated if omitted)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    symbols = [s.upper() for s in args.symbols] if args.symbols else None

    v5 = args.v5
    if v5:
        args.v4 = True  # v5 implies v4
    # Load VIX (needed for v4/v5 features and/or EM filter)
    vix_map = load_vix_daily(args.vix_csv) if (args.v4 or args.em_filter > 0) else {}

    # Load and filter records
    mode_parts = []
    if args.otm_only:
        mode_parts.append("OTM")
    if v5:
        mode_parts.append("v5")
    elif args.v4:
        mode_parts.append("v4")
    if args.em_filter > 0:
        mode_parts.append(f"EM{args.em_filter}")
    if symbols:
        mode_parts.append("+".join(symbols))
    mode = "_".join(mode_parts) if mode_parts else "all"

    print(f"Loading data from {data_dir} ({mode})...")
    records = load_jsonl_files(data_dir, symbols=symbols, otm_only=args.otm_only,
                                max_records=args.max_records)
    if len(records) < 100:
        print(f"ERROR: Only {len(records)} records")
        sys.exit(1)

    # EM filter: keep only strikes within N × expected move of spot
    if args.em_filter > 0:
        import math
        em_mult = args.em_filter
        before = len(records)
        filtered = []
        n_no_vix = 0
        for r in records:
            vix = vix_map.get(r["obs_date"], 0)
            if vix <= 0:
                n_no_vix += 1
                continue
            spot = r["spot"]
            tte_hours = r["tte_hours"]
            # EM = spot × VIX/100 / √252 × √(tte_hours/6.5)
            em = spot * (vix / 100.0) / math.sqrt(252) * math.sqrt(max(tte_hours, 0.01) / 6.5)
            distance = abs(r["strike"] - spot)
            if distance <= em_mult * em:
                filtered.append(r)
        records = filtered
        print(f"  EM filter ({em_mult}×): {before:,} → {len(records):,} "
              f"(removed {before - len(records):,}, {n_no_vix} had no VIX)")
        if len(records) < 100:
            print(f"ERROR: Only {len(records)} records after EM filter")
            sys.exit(1)

    # Stats
    dates = sorted(set(r["obs_date"] for r in records))
    sym_counts = defaultdict(int)
    for r in records:
        sym_counts[r["symbol"]] += 1
    print(f"\nTotal: {len(records):,} records across {len(dates)} dates")
    for sym in SYMBOL_ORDER:
        if sym in sym_counts:
            print(f"  {sym}: {sym_counts[sym]:,}")
    print(f"  Dates: {dates[0]} -> {dates[-1]}")

    # Build feature arrays
    # X uses v5 features (19 dims) if --v5, targets are always price/spot (not BS-residual)
    print("\nBuilding feature matrices...")
    X_both, _ = build_features(records, v4=args.v4, v5=v5, vix_map=vix_map, target="both")
    # Targets: always price/spot normalization (BS-residual computed at train time if needed)
    _, y_both = build_features(records, v4=args.v4, v5=False, vix_map=vix_map, target="both")
    _, y_bid = build_features(records, v4=args.v4, v5=False, vix_map=vix_map, target="bid")
    _, y_ask = build_features(records, v4=args.v4, v5=False, vix_map=vix_map, target="ask")

    print(f"  X: {X_both.shape}  y_both: {y_both.shape}  y_bid: {y_bid.shape}  y_ask: {y_ask.shape}")

    # Save metadata for each record (symbol, obs_date, spot, strike, etc.)
    meta_symbols = np.array([r["symbol"] for r in records], dtype="U8")
    meta_spots = np.array([r["spot"] for r in records], dtype=np.float32)
    meta_strikes = np.array([r["strike"] for r in records], dtype=np.float32)
    meta_is_call = np.array([r["is_call"] for r in records], dtype=np.float32)
    meta_moneyness = np.array([r["moneyness"] for r in records], dtype=np.float32)
    meta_tte = np.array([r["tte_hours"] for r in records], dtype=np.float32)
    meta_dates = np.array([r["obs_date"] for r in records], dtype="U10")
    meta_bids = np.array([r["bid"] for r in records], dtype=np.float32)
    meta_asks = np.array([r["ask"] for r in records], dtype=np.float32)
    meta_market_iv = np.array([r.get("market_iv", 0.0) for r in records], dtype=np.float32)
    meta_timestamps = np.array([r.get("timestamp", r.get("obs_date", "")) for r in records], dtype="U25")

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(f"data/training_{mode}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X_both,
        y_both=y_both,
        y_bid=y_bid,
        y_ask=y_ask,
        v5=np.array([v5]),
        meta_symbols=meta_symbols,
        meta_spots=meta_spots,
        meta_strikes=meta_strikes,
        meta_is_call=meta_is_call,
        meta_moneyness=meta_moneyness,
        meta_tte=meta_tte,
        meta_dates=meta_dates,
        meta_bids=meta_bids,
        meta_asks=meta_asks,
        meta_market_iv=meta_market_iv,
        meta_timestamps=meta_timestamps,
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
    print(f"  {len(records):,} samples, {X_both.shape[1]} features")
    ver = "--v5" if v5 else ("--v4" if args.v4 else "")
    print(f"\nTo train:")
    print(f"  python scripts/train_price_ann.py --npz {out_path} --target bid --otm-only {ver}")
    print(f"  python scripts/train_price_ann.py --npz {out_path} --target ask --otm-only {ver}")


if __name__ == "__main__":
    main()
