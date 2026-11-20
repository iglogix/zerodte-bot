#!/usr/bin/env python3
"""
Analyze ES futures options IV comparison data from CandleRecorder
and test the ANN v2 model (trained on equity options) on ES data.

Cross-asset generalization test: equity-trained model -> ES futures options.
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ─── Model Architecture ─────────────────────────────────────────────────────

class PricingANN(nn.Module):
    """Feedforward net matching the saved v2 checkpoint."""

    def __init__(self, input_dim: int = 5, hidden_layers: list = None):
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 64, 64]
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def moneyness_bucket(m: float) -> str:
    if m > 1.03:
        return "Deep ITM Call / Deep OTM Put"
    elif m > 1.01:
        return "ITM Call / OTM Put"
    elif m >= 0.99:
        return "ATM (0.99-1.01)"
    elif m >= 0.97:
        return "OTM Call / ITM Put"
    else:
        return "Deep OTM Call / Deep ITM Put"


def time_bucket(ts_str: str) -> str:
    dt = datetime.fromisoformat(ts_str)
    h = dt.hour
    if h < 10:
        return "09:30-10:00"
    elif h < 11:
        return "10:00-11:00"
    elif h < 12:
        return "11:00-12:00"
    elif h < 13:
        return "12:00-13:00"
    elif h < 14:
        return "13:00-14:00"
    elif h < 15:
        return "14:00-15:00"
    else:
        return "15:00-16:00"


def print_header(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=5)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "iv_comparison" / "2026-03-18.jsonl"
    model_path = repo_root / "models" / "0dte_ann_v2.pt"

    # ── Load data ──
    print(f"Loading data from {data_path}...")
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} records")
    print(f"Underlying range: {df['underlying'].min():.2f} - {df['underlying'].max():.2f}")
    print(f"Strike range: {df['strike'].min():.0f} - {df['strike'].max():.0f}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Option types: {df['type'].value_counts().to_dict()}")

    # Add derived columns
    df["moneyness_bucket"] = df["moneyness"].apply(moneyness_bucket)
    df["time_bucket"] = df["timestamp"].apply(time_bucket)
    df["log_moneyness"] = np.log(df["moneyness"])

    # ════════════════════════════════════════════════════════════════════════
    # PART 1: Old ANN Table IV Predictions vs Market
    # ════════════════════════════════════════════════════════════════════════
    print_header("PART 1: Old ANN Table IV Predictions vs Market IV")

    # Overall stats
    err = df["ann_iv_err"]
    price_err = df["ann_price_err"]
    print(f"\n  IV Error (ann_iv - market_iv):")
    print(f"    Mean:   {err.mean():.5f}")
    print(f"    Median: {err.median():.5f}")
    print(f"    Std:    {err.std():.5f}")
    print(f"    MAE:    {err.abs().mean():.5f}")
    print(f"    Min:    {err.min():.5f}   Max: {err.max():.5f}")

    print(f"\n  Price Error (ann_price - mid):")
    print(f"    Mean:   ${price_err.mean():.2f}")
    print(f"    Median: ${price_err.median():.2f}")
    print(f"    MAE:    ${price_err.abs().mean():.2f}")

    # By option type
    print(f"\n  By Option Type:")
    rows = []
    for otype in ["call", "put"]:
        sub = df[df["type"] == otype]
        rows.append([
            otype,
            f"{sub['ann_iv_err'].mean():.5f}",
            f"{sub['ann_iv_err'].median():.5f}",
            f"{sub['ann_iv_err'].abs().mean():.5f}",
            f"${sub['ann_price_err'].abs().mean():.2f}",
            len(sub),
        ])
    print_table(["Type", "Mean IV Err", "Median IV Err", "MAE IV", "MAE Price", "Count"], rows)

    # By moneyness bucket
    print(f"\n  By Moneyness Bucket:")
    rows = []
    for bucket in sorted(df["moneyness_bucket"].unique()):
        sub = df[df["moneyness_bucket"] == bucket]
        rows.append([
            bucket[:30],
            f"{sub['ann_iv_err'].mean():.5f}",
            f"{sub['ann_iv_err'].abs().mean():.5f}",
            f"${sub['ann_price_err'].abs().mean():.2f}",
            f"{sub['market_iv'].mean():.5f}",
            len(sub),
        ])
    print_table(["Moneyness", "Mean IV Err", "MAE IV", "MAE Price", "Avg Mkt IV", "Count"],
                rows, col_widths=[32, 14, 12, 12, 14, 8])

    # By time of day
    print(f"\n  By Time of Day:")
    rows = []
    for bucket in sorted(df["time_bucket"].unique()):
        sub = df[df["time_bucket"] == bucket]
        rows.append([
            bucket,
            f"{sub['ann_iv_err'].mean():.5f}",
            f"{sub['ann_iv_err'].abs().mean():.5f}",
            f"${sub['ann_price_err'].abs().mean():.2f}",
            f"{sub['tte_hours'].mean():.1f}h",
            len(sub),
        ])
    print_table(["Time Bucket", "Mean IV Err", "MAE IV", "MAE Price", "Avg TTE", "Count"],
                rows, col_widths=[16, 14, 12, 12, 10, 8])

    # ════════════════════════════════════════════════════════════════════════
    # PART 2: ANN v2 Model Cross-Asset Test
    # ════════════════════════════════════════════════════════════════════════
    print_header("PART 2: ANN v2 Model — Cross-Asset Generalization Test")

    # Load model
    print(f"\n  Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    features = checkpoint["features"]
    hidden_layers = checkpoint["hidden_layers"]
    feat_mean = torch.tensor(checkpoint["feat_mean"], dtype=torch.float32)
    feat_std = torch.tensor(checkpoint["feat_std"], dtype=torch.float32)

    print(f"  Architecture: {checkpoint.get('architecture', 'unknown')}")
    print(f"  Version: {checkpoint.get('version', 'unknown')}")
    print(f"  Features: {features}")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Training dates: {checkpoint.get('training_dates', 'unknown')}")
    print(f"  Test date: {checkpoint.get('test_date', 'unknown')}")
    print(f"  Feature means: {[f'{v:.6f}' for v in checkpoint['feat_mean']]}")
    print(f"  Feature stds:  {[f'{v:.6f}' for v in checkpoint['feat_std']]}")

    model = PricingANN(input_dim=len(features), hidden_layers=hidden_layers)
    # Handle dropout layers that may not be in state dict
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    # Prepare features: [moneyness, tte_years, tte_hours, is_call, log_moneyness]
    X = np.column_stack([
        df["moneyness"].values,
        df["tte_years"].values,
        df["tte_hours"].values,
        (df["type"] == "call").astype(float).values,
        df["log_moneyness"].values,
    ])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Normalize using training stats
    X_norm = (X_tensor - feat_mean) / (feat_std + 1e-8)

    # Run inference
    with torch.no_grad():
        pred_norm_mid = model(X_norm).squeeze().numpy()

    # Convert: model predicts normalized mid (base=100 underlying)
    # pred_price = pred_norm_mid * underlying / 100
    underlying = df["underlying"].values
    pred_price = pred_norm_mid * underlying / 100.0
    actual_mid = df["mid"].values

    # Filter out zero/near-zero mid prices to avoid division issues
    valid = actual_mid > 0.05
    pred_valid = pred_price[valid]
    actual_valid = actual_mid[valid]
    df_valid = df[valid].copy()

    errors = pred_valid - actual_valid
    abs_errors = np.abs(errors)
    pct_errors = abs_errors / actual_valid * 100

    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    ss_res = ((errors) ** 2).sum()
    ss_tot = ((actual_valid - actual_valid.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    mape = pct_errors.mean()
    median_ape = np.median(pct_errors)

    print(f"\n  Overall Results ({valid.sum():,} valid records, excluded {(~valid).sum()} near-zero):")
    print(f"    MAE:        ${mae:.2f}")
    print(f"    RMSE:       ${rmse:.2f}")
    print(f"    R²:         {r2:.4f}")
    print(f"    MAPE:       {mape:.1f}%")
    print(f"    Median APE: {median_ape:.1f}%")
    print(f"    Mean Error:  ${errors.mean():.2f} (bias)")
    print(f"    Pred range: ${pred_valid.min():.2f} to ${pred_valid.max():.2f}")
    print(f"    Actual range: ${actual_valid.min():.2f} to ${actual_valid.max():.2f}")

    # Assign predictions back for breakdown
    df_valid = df_valid.copy()
    df_valid["pred_price"] = pred_valid
    df_valid["price_error"] = errors
    df_valid["abs_price_error"] = abs_errors
    df_valid["pct_error"] = pct_errors

    # ── By option type ──
    print(f"\n  By Option Type:")
    rows = []
    for otype in ["call", "put"]:
        sub = df_valid[df_valid["type"] == otype]
        e = sub["price_error"]
        ae = sub["abs_price_error"]
        pe = sub["pct_error"]
        rows.append([
            otype,
            f"${ae.mean():.2f}",
            f"${np.sqrt((e**2).mean()):.2f}",
            f"{pe.mean():.1f}%",
            f"${e.mean():.2f}",
            len(sub),
        ])
    print_table(["Type", "MAE", "RMSE", "MAPE", "Bias", "Count"], rows)

    # ── By moneyness bucket ──
    print(f"\n  By Moneyness Bucket:")
    rows = []
    for bucket in sorted(df_valid["moneyness_bucket"].unique()):
        sub = df_valid[df_valid["moneyness_bucket"] == bucket]
        e = sub["price_error"]
        ae = sub["abs_price_error"]
        pe = sub["pct_error"]
        rows.append([
            bucket[:30],
            f"${ae.mean():.2f}",
            f"${np.sqrt((e**2).mean()):.2f}",
            f"{pe.mean():.1f}%",
            f"${e.mean():.2f}",
            f"${sub['mid'].mean():.2f}",
            len(sub),
        ])
    print_table(["Moneyness", "MAE", "RMSE", "MAPE", "Bias", "Avg Mid", "Count"],
                rows, col_widths=[32, 10, 10, 10, 10, 10, 8])

    # ── By time of day ──
    print(f"\n  By Time of Day:")
    rows = []
    for bucket in sorted(df_valid["time_bucket"].unique()):
        sub = df_valid[df_valid["time_bucket"] == bucket]
        e = sub["price_error"]
        ae = sub["abs_price_error"]
        pe = sub["pct_error"]
        rows.append([
            bucket,
            f"${ae.mean():.2f}",
            f"${np.sqrt((e**2).mean()):.2f}",
            f"{pe.mean():.1f}%",
            f"${e.mean():.2f}",
            f"{sub['tte_hours'].mean():.1f}h",
            len(sub),
        ])
    print_table(["Time Bucket", "MAE", "RMSE", "MAPE", "Bias", "Avg TTE", "Count"],
                rows, col_widths=[16, 10, 10, 10, 10, 10, 8])

    # ── Sample predictions for near-ATM options ──
    print_header("SAMPLE: Near-ATM Predictions vs Actuals")
    atm = df_valid[df_valid["moneyness_bucket"] == "ATM (0.99-1.01)"].copy()
    # Pick a sample across different times
    if len(atm) > 0:
        sample = atm.groupby("time_bucket").apply(
            lambda g: g.sample(min(3, len(g)), random_state=42)
        ).reset_index(drop=True).head(20)
        rows = []
        for _, r in sample.iterrows():
            ts = datetime.fromisoformat(r["timestamp"]).strftime("%H:%M")
            rows.append([
                ts,
                r["type"],
                f"{r['strike']:.0f}",
                f"{r['moneyness']:.4f}",
                f"{r['tte_hours']:.1f}h",
                f"${r['mid']:.2f}",
                f"${r['pred_price']:.2f}",
                f"${r['price_error']:.2f}",
                f"{r['pct_error']:.1f}%",
            ])
        print_table(
            ["Time", "Type", "Strike", "Moneyness", "TTE", "Actual", "Predicted", "Error", "Pct"],
            rows, col_widths=[8, 6, 8, 12, 8, 10, 12, 10, 8]
        )

    # ── Compare old ANN table vs new ANN v2 for same records ──
    print_header("COMPARISON: Old ANN Table vs New ANN v2")

    # Old ANN price error (already in data)
    old_mae = df_valid["ann_price_err"].abs().mean()
    old_rmse = np.sqrt((df_valid["ann_price_err"] ** 2).mean())
    old_bias = df_valid["ann_price_err"].mean()

    print(f"\n  {'Metric':<20} {'Old ANN Table':>15} {'New ANN v2':>15}")
    print(f"  {'-'*50}")
    print(f"  {'MAE':<20} ${old_mae:>13.2f} ${mae:>13.2f}")
    print(f"  {'RMSE':<20} ${old_rmse:>13.2f} ${rmse:>13.2f}")
    print(f"  {'Bias':<20} ${old_bias:>13.2f} ${errors.mean():>13.2f}")
    print(f"  {'MAPE':<20} {'N/A':>15} {mape:>14.1f}%")
    print(f"  {'R²':<20} {'N/A':>15} {r2:>14.4f}")

    # ── Verdict ──
    print_header("VERDICT: Cross-Asset Generalization")
    print(f"""
  Model trained on: SPY/QQQ/IWM/GLD/TLT (equity options, normalized to base=100)
  Tested on:        ES futures options (underlying ~{underlying.mean():.0f})

  Key observations:
  - ES underlying is ~67x SPY, so normalization (base=100) is critical
  - The model sees moneyness/tte_hours/log_moneyness which are scale-invariant
  - R² = {r2:.4f} — {'good' if r2 > 0.9 else 'moderate' if r2 > 0.7 else 'weak' if r2 > 0.5 else 'poor'} fit
  - MAE = ${mae:.2f} on ES (typical mid ${actual_valid.mean():.2f})
  - MAPE = {mape:.1f}% — {'excellent' if mape < 5 else 'good' if mape < 10 else 'moderate' if mape < 20 else 'needs work'}
  - Bias = ${errors.mean():.2f} — {'model overprices' if errors.mean() > 0 else 'model underprices'}

  {'PASS: Model generalizes well to ES futures options!' if r2 > 0.85 and mape < 15 else 'MIXED: Model shows some transfer but needs fine-tuning on ES data.' if r2 > 0.7 else 'FAIL: Model does not generalize well to ES futures options.'}
""")


if __name__ == "__main__":
    main()
