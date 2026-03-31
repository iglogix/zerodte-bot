#!/usr/bin/env python3
"""Validate ANN IV model accuracy against market IV from collected 0DTE option data."""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_near_atm(records: list[dict]) -> list[dict]:
    """Filter to near-ATM options with valid ANN and market IV."""
    filtered = []
    for r in records:
        m = r.get("moneyness", 0)
        miv = r.get("market_iv")
        aiv = r.get("ann_iv")
        if miv and aiv and miv > 0 and aiv > 0 and 0.97 <= m <= 1.03:
            filtered.append(r)
    return filtered


def moneyness_bucket(m: float) -> str:
    if m < 0.99:
        return "0.97-0.99 (OTM put / ITM call)"
    elif m <= 1.01:
        return "0.99-1.01 (ATM)"
    else:
        return "1.01-1.03 (ITM put / OTM call)"


def compute_stats(records: list[dict], label: str = "") -> dict:
    if not records:
        return {"label": label, "count": 0}

    errs = np.array([r["ann_iv_err"] for r in records])
    mivs = np.array([r["market_iv"] for r in records])
    aivs = np.array([r["ann_iv"] for r in records])
    pct_errs = (errs / mivs) * 100  # as percentage of market IV

    return {
        "label": label,
        "count": len(records),
        "mean_err": float(np.mean(errs)),
        "median_err": float(np.median(errs)),
        "std_err": float(np.std(errs)),
        "mean_abs_err": float(np.mean(np.abs(errs))),
        "median_abs_err": float(np.median(np.abs(errs))),
        "mean_pct_err": float(np.mean(pct_errs)),
        "median_pct_err": float(np.median(pct_errs)),
        "std_pct_err": float(np.std(pct_errs)),
        "mean_abs_pct_err": float(np.mean(np.abs(pct_errs))),
        "mean_market_iv": float(np.mean(mivs)),
        "mean_ann_iv": float(np.mean(aivs)),
        "over_predict_pct": float(np.mean(errs > 0) * 100),
        "under_predict_pct": float(np.mean(errs < 0) * 100),
    }


def print_stats(s: dict):
    if s["count"] == 0:
        print(f"  {s['label']}: No records")
        return
    direction = "OVER" if s["mean_err"] > 0 else "UNDER"
    print(f"  {s['label']} (n={s['count']:,})")
    print(f"    Market IV (mean):     {s['mean_market_iv']:.4f}")
    print(f"    ANN IV (mean):        {s['mean_ann_iv']:.4f}")
    print(f"    Error (ann - market):")
    print(f"      Mean:    {s['mean_err']:+.4f}  ({s['mean_pct_err']:+.1f}% of market IV) → {direction}-predicts")
    print(f"      Median:  {s['median_err']:+.4f}  ({s['median_pct_err']:+.1f}%)")
    print(f"      Std:     {s['std_err']:.4f}   ({s['std_pct_err']:.1f}%)")
    print(f"    Absolute Error:")
    print(f"      Mean:    {s['mean_abs_err']:.4f}   ({s['mean_abs_pct_err']:.1f}% of market IV)")
    print(f"      Median:  {s['median_abs_err']:.4f}")
    print(f"    Over-predict: {s['over_predict_pct']:.1f}%  |  Under-predict: {s['under_predict_pct']:.1f}%")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/iv_collection/2026-03-17.jsonl"
    path = Path(data_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path}...")
    all_records = load_records(str(path))
    print(f"Total records: {len(all_records):,}")

    # Count records with ANN predictions
    has_ann = [r for r in all_records if r.get("ann_iv") is not None and r.get("ann_iv") > 0]
    print(f"Records with ANN predictions: {len(has_ann):,} ({len(has_ann)/len(all_records)*100:.1f}%)")

    # Filter near-ATM
    near_atm = filter_near_atm(all_records)
    print(f"Near-ATM (0.97-1.03) with valid IVs: {len(near_atm):,}")
    print()

    # ── Overall ──
    print("=" * 80)
    print("OVERALL SUMMARY (Near-ATM)")
    print("=" * 80)
    overall = compute_stats(near_atm, "All near-ATM")
    print_stats(overall)
    print()

    # ── By Symbol ──
    print("=" * 80)
    print("BREAKDOWN BY SYMBOL")
    print("=" * 80)
    by_symbol = defaultdict(list)
    for r in near_atm:
        by_symbol[r["symbol"]].append(r)
    for sym in sorted(by_symbol.keys()):
        print_stats(compute_stats(by_symbol[sym], sym))
        print()

    # ── By Option Type ──
    print("=" * 80)
    print("BREAKDOWN BY OPTION TYPE")
    print("=" * 80)
    by_type = defaultdict(list)
    for r in near_atm:
        by_type[r["option_type"]].append(r)
    for ot in sorted(by_type.keys()):
        print_stats(compute_stats(by_type[ot], ot.upper()))
        print()

    # ── By Hour ──
    print("=" * 80)
    print("BREAKDOWN BY HOUR (ET)")
    print("=" * 80)
    by_hour = defaultdict(list)
    for r in near_atm:
        ts = r.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            by_hour[dt.hour].append(r)
        except (ValueError, TypeError):
            pass
    for hour in sorted(by_hour.keys()):
        print_stats(compute_stats(by_hour[hour], f"{hour:02d}:00"))
        print()

    # ── By Moneyness Bucket ──
    print("=" * 80)
    print("BREAKDOWN BY MONEYNESS BUCKET")
    print("=" * 80)
    by_bucket = defaultdict(list)
    for r in near_atm:
        by_bucket[moneyness_bucket(r["moneyness"])].append(r)
    for bucket in sorted(by_bucket.keys()):
        print_stats(compute_stats(by_bucket[bucket], bucket))
        print()

    # ── Systematic Bias Summary ──
    print("=" * 80)
    print("SYSTEMATIC BIAS SUMMARY")
    print("=" * 80)
    if overall["count"] > 0:
        direction = "OVER" if overall["mean_err"] > 0 else "UNDER"
        print(f"  ANN systematically {direction}-predicts IV by {abs(overall['mean_pct_err']):.1f}% on average")
        print(f"  {overall['over_predict_pct']:.1f}% of predictions are too high, {overall['under_predict_pct']:.1f}% too low")
        print()

        # Check if bias varies meaningfully
        print("  Bias by time of day:")
        for hour in sorted(by_hour.keys()):
            s = compute_stats(by_hour[hour], "")
            if s["count"] > 0:
                d = "+" if s["mean_pct_err"] > 0 else ""
                print(f"    {hour:02d}:00  {d}{s['mean_pct_err']:.1f}%  (n={s['count']:,})")

        print()
        print("  Bias by symbol:")
        for sym in sorted(by_symbol.keys()):
            s = compute_stats(by_symbol[sym], "")
            if s["count"] > 0:
                d = "+" if s["mean_pct_err"] > 0 else ""
                print(f"    {sym:<6} {d}{s['mean_pct_err']:.1f}%  (n={s['count']:,})")
    print()


if __name__ == "__main__":
    main()
