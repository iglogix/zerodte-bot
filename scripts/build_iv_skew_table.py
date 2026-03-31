#!/usr/bin/env python3
"""
Build an empirical IV skew table from ES options BBO data.

Measures how implied vol varies by moneyness relative to ATM IV.
Output: a lookup table mapping moneyness → IV correction factor (skew_ratio)

In the backtest: effective_IV = ATM_IV * skew_ratio(moneyness)
This replaces flat-IV BS pricing with skew-aware pricing.

Usage:
    python scripts/build_iv_skew_table.py --bbo-dir /path/to/GLBX-data --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse data pipeline from train_option_ann
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_option_ann import (
    load_underlying_prices,
    load_bbo_day,
    enrich_with_iv,
)


def build_skew_table(all_records: list[dict],
                     n_bins: int = 40) -> dict:
    """Build moneyness → skew_ratio lookup from enriched BBO records.

    skew_ratio = strike_IV / ATM_IV
    Bucketed by moneyness with median + percentiles.
    """
    # Compute skew ratios
    skew_data = []
    for r in all_records:
        atm_iv = r["atm_iv"]
        if atm_iv < 0.01:
            continue
        skew_ratio = r["iv"] / atm_iv
        skew_data.append({
            "moneyness": r["moneyness"],
            "skew_ratio": skew_ratio,
            "opt_type": r["opt_type"],
            "dte_days": r["dte_days"],
            "iv": r["iv"],
            "atm_iv": atm_iv,
        })

    moneynesses = np.array([d["moneyness"] for d in skew_data])
    skew_ratios = np.array([d["skew_ratio"] for d in skew_data])

    # Create bins
    bin_edges = np.linspace(0.90, 1.10, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    table = []
    for i in range(n_bins):
        mask = (moneynesses >= bin_edges[i]) & (moneynesses < bin_edges[i + 1])
        if mask.sum() < 10:
            continue
        ratios = skew_ratios[mask]
        table.append({
            "moneyness_lo": round(float(bin_edges[i]), 4),
            "moneyness_hi": round(float(bin_edges[i + 1]), 4),
            "moneyness_mid": round(float(bin_centers[i]), 4),
            "skew_ratio_median": round(float(np.median(ratios)), 4),
            "skew_ratio_mean": round(float(np.mean(ratios)), 4),
            "skew_ratio_p25": round(float(np.percentile(ratios, 25)), 4),
            "skew_ratio_p75": round(float(np.percentile(ratios, 75)), 4),
            "count": int(mask.sum()),
        })

    # Also build separate tables for calls and puts
    call_mask = np.array([d["opt_type"] == "C" for d in skew_data])
    put_mask = ~call_mask

    for opt_label, opt_mask in [("calls", call_mask), ("puts", put_mask)]:
        m_filtered = moneynesses[opt_mask]
        s_filtered = skew_ratios[opt_mask]
        sub_table = []
        for i in range(n_bins):
            mask = (m_filtered >= bin_edges[i]) & (m_filtered < bin_edges[i + 1])
            if mask.sum() < 10:
                continue
            ratios = s_filtered[mask]
            sub_table.append({
                "moneyness_mid": round(float(bin_centers[i]), 4),
                "skew_ratio_median": round(float(np.median(ratios)), 4),
                "count": int(mask.sum()),
            })

    # Summary by DTE buckets
    dte_buckets = [(22, 45, "short"), (45, 90, "medium"), (90, 200, "long"), (200, 2000, "very_long")]
    dte_tables = {}
    for dte_lo, dte_hi, label in dte_buckets:
        dte_mask = np.array([dte_lo <= d["dte_days"] < dte_hi for d in skew_data])
        m_filtered = moneynesses[dte_mask]
        s_filtered = skew_ratios[dte_mask]
        sub = []
        for i in range(n_bins):
            mask = (m_filtered >= bin_edges[i]) & (m_filtered < bin_edges[i + 1])
            if mask.sum() < 10:
                continue
            ratios = s_filtered[mask]
            sub.append({
                "moneyness_mid": round(float(bin_centers[i]), 4),
                "skew_ratio": round(float(np.median(ratios)), 4),
                "count": int(mask.sum()),
            })
        dte_tables[label] = sub

    return {
        "all": table,
        "by_dte": dte_tables,
        "n_records": len(skew_data),
    }


def main():
    parser = argparse.ArgumentParser(description="Build IV skew table from BBO data")
    parser.add_argument("--bbo-dir", required=True, help="Path to GLBX BBO directory")
    parser.add_argument("--ohlcv", default="glbx-mdp3-20210227-20260226.ohlcv-1m.csv")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default="models/iv_skew_table.json")
    args = parser.parse_args()

    bbo_dir = Path(args.bbo_dir)
    bbo_files = sorted(bbo_dir.glob("glbx-mdp3-*.bbo-1m.csv.zst"))
    bbo_files = [f for f in bbo_files if f.stat().st_size > 5_000_000]
    if not args.all:
        bbo_files = bbo_files[:args.days]
    print(f"Found {len(bbo_files)} trading day BBO files")

    und_prices = load_underlying_prices(args.ohlcv)

    all_records: list[dict] = []
    for i, bbo_file in enumerate(bbo_files):
        day_str = bbo_file.name.split(".")[0].replace("glbx-mdp3-", "")
        day_date = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        day_und = und_prices.get(day_date, {})
        if not day_und:
            print(f"  [{i+1}/{len(bbo_files)}] {day_date}: no underlying, skip")
            continue

        print(f"  [{i+1}/{len(bbo_files)}] {day_date}: ", end="", flush=True)
        records = load_bbo_day(str(bbo_file), day_und)
        records = enrich_with_iv(records)
        print(f"{len(records):,} records")
        all_records.extend(records)

    print(f"\nTotal: {len(all_records):,} records")

    result = build_skew_table(all_records)

    # Print summary
    print(f"\n{'Moneyness':>10s} {'Skew Ratio':>11s} {'Count':>8s}")
    print("-" * 32)
    for row in result["all"]:
        m = row["moneyness_mid"]
        sr = row["skew_ratio_median"]
        n = row["count"]
        bar = "█" * int(sr * 20)
        print(f"  {m:>8.4f}   {sr:>8.4f}    {n:>6,d}  {bar}")

    # Key insight: what's the skew ratio at delta≈0.16 moneyness?
    # For ES at 6000 with delta=0.16, put strike ≈ 5900 → moneyness ≈ 1.017
    # Call strike ≈ 6100 → moneyness ≈ 0.983
    print("\n--- Key moneyness levels for 0DTE at delta~0.16 ---")
    for target_m, label in [(0.983, "OTM call (delta~0.16)"),
                             (1.017, "OTM put  (delta~0.16)"),
                             (1.000, "ATM (reference)")]:
        best = min(result["all"], key=lambda r: abs(r["moneyness_mid"] - target_m))
        print(f"  {label}: moneyness={best['moneyness_mid']:.4f}, "
              f"skew_ratio={best['skew_ratio_median']:.4f} "
              f"(p25={best['skew_ratio_p25']:.4f}, p75={best['skew_ratio_p75']:.4f})")

    # Print DTE breakdown
    print("\n--- Skew by DTE bucket (at moneyness≈0.98 for OTM calls) ---")
    for dte_label, rows in result["by_dte"].items():
        target = [r for r in rows if abs(r["moneyness_mid"] - 0.98) < 0.01]
        if target:
            r = target[0]
            print(f"  {dte_label:>10s}: skew_ratio={r['skew_ratio']:.4f} (n={r['count']:,})")

    # Save
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSkew table saved to {args.out}")


if __name__ == "__main__":
    main()
