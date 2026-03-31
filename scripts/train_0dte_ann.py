#!/usr/bin/env python3
"""
Train a universal 0DTE option pricing ANN (v2).

Predicts norm_mid (and optionally market_iv) from moneyness, tte, option_type.
Normalizing to base=100 underlying makes SPY/QQQ/IWM data comparable.

v2 changes:
- Loads ALL .jsonl files from data/iv_collection/
- Date-stratified train/test split (train day 1, test day 2)
- Also evaluates with random 80/20 split for comparison
- Added tte_hours feature for non-linear TTE compression
- 300 epochs for larger dataset
- Saves to models/0dte_ann_v2.pt
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data/iv_collection"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "0dte_ann_v2.pt"

FEATURES = ["moneyness", "tte_years", "tte_hours", "is_call", "log_moneyness"]
HIDDEN_LAYERS = [128, 128, 64, 64]
EPOCHS = 300
BATCH_SIZE = 512
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Data Loading ────────────────────────────────────────────────────────
def load_all_data(data_dir: Path):
    """Load all .jsonl files from the data directory."""
    all_records = []
    files = sorted(data_dir.glob("*.jsonl"))
    print(f"Found {len(files)} data files in {data_dir}")

    for path in files:
        count_before = len(all_records)
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                # Filters
                if not r["bid"] or r["bid"] <= 0 or not r["ask"] or r["ask"] <= 0 or not r["mid"] or r["mid"] <= 0:
                    continue
                if not r["market_iv"] or r["market_iv"] <= 0:
                    continue
                if not (0.95 <= r["moneyness"] <= 1.05):
                    continue
                r["is_call"] = 1.0 if r["option_type"] == "call" else 0.0
                r["log_moneyness"] = np.log(r["moneyness"])
                # Extract date from timestamp for grouping
                r["date"] = r["timestamp"][:10]
                all_records.append(r)
        loaded = len(all_records) - count_before
        print(f"  {path.name}: {loaded} records after filtering")

    print(f"Total: {len(all_records)} records")
    return all_records


# ── Model ───────────────────────────────────────────────────────────────
class PricingANN(nn.Module):
    def __init__(self, n_inputs, hidden_layers, n_outputs=1):
        super().__init__()
        layers = []
        prev = n_inputs
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Training Loop ───────────────────────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test, target_name, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = PricingANN(X_tr.shape[1], HIDDEN_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()

    print(f"\n{'='*60}")
    print(f"Training target: {target_name}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_pred = model(X_te)
            test_loss = criterion(test_pred, y_te).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train_mse={train_loss:.6f}  test_mse={test_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.1e}")

    return model, train_losses, test_losses, device


# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, records_test, target_name, device, label=""):
    model.eval()
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_te).cpu().numpy().flatten()
    actuals = y_test

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-10))) * 100

    header = f"RESULTS for target: {target_name}"
    if label:
        header += f" [{label}]"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  R²   = {r2:.6f}")
    print(f"  MAPE = {mape:.2f}%")

    # Breakdown by symbol
    symbols = sorted(set(r["symbol"] for r in records_test))
    print(f"\n  Per-symbol breakdown:")
    print(f"  {'Symbol':<8} {'Count':>6} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE%':>8}")
    for sym in symbols:
        mask = np.array([r["symbol"] == sym for r in records_test])
        if mask.sum() == 0:
            continue
        s_mae = mean_absolute_error(actuals[mask], preds[mask])
        s_rmse = np.sqrt(mean_squared_error(actuals[mask], preds[mask]))
        s_r2 = r2_score(actuals[mask], preds[mask]) if mask.sum() > 1 else 0
        s_mape = np.mean(np.abs((actuals[mask] - preds[mask]) / (actuals[mask] + 1e-10))) * 100
        print(f"  {sym:<8} {mask.sum():>6} {s_mae:>10.6f} {s_rmse:>10.6f} {s_r2:>8.4f} {s_mape:>8.2f}")

    # Breakdown by moneyness bucket
    moneyness = np.array([r["moneyness"] for r in records_test])
    print(f"\n  Per-moneyness breakdown:")
    print(f"  {'Bucket':<12} {'Count':>6} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    buckets = [
        ("Deep ITM", 0.95, 0.97),
        ("ITM", 0.97, 0.99),
        ("ATM", 0.99, 1.01),
        ("OTM", 1.01, 1.03),
        ("Deep OTM", 1.03, 1.05),
    ]
    for bucket_label, lo, hi in buckets:
        mask = (moneyness >= lo) & (moneyness < hi)
        if mask.sum() == 0:
            continue
        b_mae = mean_absolute_error(actuals[mask], preds[mask])
        b_rmse = np.sqrt(mean_squared_error(actuals[mask], preds[mask]))
        b_r2 = r2_score(actuals[mask], preds[mask]) if mask.sum() > 1 else 0
        print(f"  {bucket_label:<12} {mask.sum():>6} {b_mae:>10.6f} {b_rmse:>10.6f} {b_r2:>8.4f}")

    # Sample predictions near ATM
    print(f"\n  Sample predictions (near-ATM, 0.995 <= moneyness <= 1.005):")
    print(f"  {'Symbol':<6} {'Type':<5} {'Money':>7} {'TTE_h':>6} {'Actual':>8} {'Pred':>8} {'Err':>8} {'Err%':>7}")
    atm_indices = [i for i, r in enumerate(records_test)
                   if 0.995 <= r["moneyness"] <= 1.005]
    step = max(1, len(atm_indices) // 20)
    for idx in atm_indices[::step][:20]:
        r = records_test[idx]
        err = preds[idx] - actuals[idx]
        err_pct = err / (actuals[idx] + 1e-10) * 100
        print(f"  {r['symbol']:<6} {r['option_type']:<5} {r['moneyness']:>7.4f} {r['tte_hours']:>6.2f} "
              f"{actuals[idx]:>8.4f} {preds[idx]:>8.4f} {err:>8.4f} {err_pct:>6.1f}%")

    return preds


# ── Main ────────────────────────────────────────────────────────────────
def main():
    records = load_all_data(DATA_DIR)

    # Show date distribution
    dates = sorted(set(r["date"] for r in records))
    print(f"\nDates in dataset: {dates}")
    for d in dates:
        count = sum(1 for r in records if r["date"] == d)
        print(f"  {d}: {count} records")

    # Feature / target extraction
    feature_names = FEATURES
    X = np.array([[r[f] for f in feature_names] for r in records])
    y_mid = np.array([r["norm_mid"] for r in records])
    y_iv = np.array([r["market_iv"] for r in records])
    symbols = np.array([r["symbol"] for r in records])
    record_dates = np.array([r["date"] for r in records])

    print(f"\nFeatures: {feature_names}")
    print(f"X shape: {X.shape}")
    print(f"norm_mid stats: mean={y_mid.mean():.4f}, std={y_mid.std():.4f}, min={y_mid.min():.4f}, max={y_mid.max():.4f}")
    print(f"market_iv stats: mean={y_iv.mean():.4f}, std={y_iv.std():.4f}, min={y_iv.min():.4f}, max={y_iv.max():.4f}")

    # Unique symbols
    unique_symbols = sorted(set(symbols))
    print(f"Symbols: {unique_symbols}")
    for s in unique_symbols:
        print(f"  {s}: {(symbols == s).sum()} records")

    # ════════════════════════════════════════════════════════════════════
    # EVALUATION 1: Date-stratified split (train on earlier days, test on latest)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print(f"# EVALUATION 1: DATE-STRATIFIED SPLIT")
    print(f"# Train on: {dates[:-1]}, Test on: {dates[-1]}")
    print(f"{'#'*70}")

    train_date = dates[-1]  # latest date is test
    train_mask = record_dates != train_date
    test_mask = record_dates == train_date

    X_train_ds, X_test_ds = X[train_mask], X[test_mask]
    records_train_ds = [records[i] for i in range(len(records)) if train_mask[i]]
    records_test_ds = [records[i] for i in range(len(records)) if test_mask[i]]

    # Normalize features (fit on train only)
    feat_mean_ds = X_train_ds.mean(axis=0)
    feat_std_ds = X_train_ds.std(axis=0) + 1e-8
    X_train_ds_norm = (X_train_ds - feat_mean_ds) / feat_std_ds
    X_test_ds_norm = (X_test_ds - feat_mean_ds) / feat_std_ds

    print(f"\nTrain: {len(X_train_ds)} records, Test: {len(X_test_ds)} records")

    # Target 1: norm_mid (date-stratified)
    y_mid_train_ds = y_mid[train_mask]
    y_mid_test_ds = y_mid[test_mask]
    model_mid_ds, tl_mid_ds, vl_mid_ds, device = train_model(
        X_train_ds_norm, y_mid_train_ds, X_test_ds_norm, y_mid_test_ds,
        "norm_mid", epochs=EPOCHS
    )
    evaluate(model_mid_ds, X_test_ds_norm, y_mid_test_ds, records_test_ds,
             "norm_mid", device, label=f"Cross-day: trained on {dates[:-1]}, tested on {train_date}")

    # Target 2: market_iv (date-stratified)
    y_iv_train_ds = y_iv[train_mask]
    y_iv_test_ds = y_iv[test_mask]
    model_iv_ds, tl_iv_ds, vl_iv_ds, device = train_model(
        X_train_ds_norm, y_iv_train_ds, X_test_ds_norm, y_iv_test_ds,
        "market_iv", epochs=EPOCHS
    )
    evaluate(model_iv_ds, X_test_ds_norm, y_iv_test_ds, records_test_ds,
             "market_iv", device, label=f"Cross-day: trained on {dates[:-1]}, tested on {train_date}")

    # ════════════════════════════════════════════════════════════════════
    # EVALUATION 2: Random 80/20 split (for comparison)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print(f"# EVALUATION 2: RANDOM 80/20 SPLIT (symbol-stratified)")
    print(f"{'#'*70}")

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=symbols)

    X_train_r, X_test_r = X[train_idx], X[test_idx]
    records_train_r = [records[i] for i in train_idx]
    records_test_r = [records[i] for i in test_idx]

    # Normalize features (fit on train only)
    feat_mean_r = X_train_r.mean(axis=0)
    feat_std_r = X_train_r.std(axis=0) + 1e-8
    X_train_r_norm = (X_train_r - feat_mean_r) / feat_std_r
    X_test_r_norm = (X_test_r - feat_mean_r) / feat_std_r

    # Target 1: norm_mid (random split)
    y_mid_train_r, y_mid_test_r = y_mid[train_idx], y_mid[test_idx]
    model_mid_r, tl_mid_r, vl_mid_r, device = train_model(
        X_train_r_norm, y_mid_train_r, X_test_r_norm, y_mid_test_r,
        "norm_mid", epochs=EPOCHS
    )
    evaluate(model_mid_r, X_test_r_norm, y_mid_test_r, records_test_r,
             "norm_mid", device, label="Random 80/20 split")

    # Target 2: market_iv (random split)
    y_iv_train_r, y_iv_test_r = y_iv[train_idx], y_iv[test_idx]
    model_iv_r, tl_iv_r, vl_iv_r, device = train_model(
        X_train_r_norm, y_iv_train_r, X_test_r_norm, y_iv_test_r,
        "market_iv", epochs=EPOCHS
    )
    evaluate(model_iv_r, X_test_r_norm, y_iv_test_r, records_test_r,
             "market_iv", device, label="Random 80/20 split")

    # ════════════════════════════════════════════════════════════════════
    # Save the date-stratified model (the real generalization test)
    # ════════════════════════════════════════════════════════════════════
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "model_state": model_mid_ds.state_dict(),
        "feat_mean": feat_mean_ds.tolist(),
        "feat_std": feat_std_ds.tolist(),
        "features": feature_names,
        "hidden_layers": HIDDEN_LAYERS,
        "architecture": "PricingANN",
        "training_dates": dates[:-1],
        "test_date": train_date,
        "version": "v2",
    }
    torch.save(save_dict, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # ── Loss curves summary ─────────────────────────────────────────────
    milestones = [1, 50, 100, 150, 200, 250, 300]

    print(f"\n{'='*60}")
    print("Loss curve summary — DATE-STRATIFIED (norm_mid):")
    for m in milestones:
        if m <= len(tl_mid_ds):
            print(f"  Epoch {m:>3d}: train={tl_mid_ds[m-1]:.6f}  test={vl_mid_ds[m-1]:.6f}")

    print(f"\nLoss curve summary — DATE-STRATIFIED (market_iv):")
    for m in milestones:
        if m <= len(tl_iv_ds):
            print(f"  Epoch {m:>3d}: train={tl_iv_ds[m-1]:.6f}  test={vl_iv_ds[m-1]:.6f}")

    print(f"\nLoss curve summary — RANDOM 80/20 (norm_mid):")
    for m in milestones:
        if m <= len(tl_mid_r):
            print(f"  Epoch {m:>3d}: train={tl_mid_r[m-1]:.6f}  test={vl_mid_r[m-1]:.6f}")

    print(f"\nLoss curve summary — RANDOM 80/20 (market_iv):")
    for m in milestones:
        if m <= len(tl_iv_r):
            print(f"  Epoch {m:>3d}: train={tl_iv_r[m-1]:.6f}  test={vl_iv_r[m-1]:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
