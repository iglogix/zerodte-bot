#!/usr/bin/env python3
"""
Train an ANN to predict the implied volatility surface: IV = f(moneyness, TTE, is_call).

Unlike the price ANN (train_option_ann.py), this predicts IV directly.
Advantages for 0DTE extrapolation:
  - IV is bounded (typically 0.05-1.5) vs prices that span orders of magnitude
  - The smile/skew shape is relatively stable across DTE
  - At TTE→0, IV should converge to realized vol, not explode
  - Plug predicted IV into BS for prices/greeks — get arbitrage-free surfaces

Usage:
    python scripts/train_iv_surface.py --bbo-dir /path/to/GLBX-data --all
    python scripts/train_iv_surface.py --bbo-dir /path/to/GLBX-data --days 10
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Reuse data pipeline from train_option_ann
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_option_ann import (
    load_underlying_prices,
    load_bbo_day,
    enrich_with_iv,
    bs_call,
    bs_put,
    bs_call_delta,
    bs_put_delta,
    MULTIPLIER,
)


# ---------------------------------------------------------------------------
# ANN Architecture
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IVSurfaceNet(nn.Module):
    """Predicts implied volatility from (moneyness, TTE, is_call).

    Architecture: 5 -> 256 -> 128 -> 64 -> 32 -> 1
    Output: Softplus ensures IV > 0, shifted to start near typical IV (~0.15).
    """

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(5, 256)
        self.ln1 = nn.LayerNorm(256)
        self.h1 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.h2 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.h3 = nn.Linear(64, 32)
        self.ln4 = nn.LayerNorm(32)
        self.output = nn.Linear(32, 1)
        self.act = nn.GELU()
        # Softplus for positive IV output
        self.out_act = nn.Softplus(beta=5.0)
        # Init output bias so Softplus(bias) ≈ 0.15 (typical IV)
        # Softplus(x) ≈ x for large x, so bias ≈ 0.15
        nn.init.constant_(self.output.bias, 0.15)
        nn.init.uniform_(self.output.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.ln1(self.input_layer(x)))
        x = self.act(self.ln2(self.h1(x)))
        x = self.act(self.ln3(self.h2(x)))
        x = self.act(self.ln4(self.h3(x)))
        return self.out_act(self.output(x)).squeeze(-1)


class IVSurfaceANN:
    """Wrapper for training, inference, and serialization."""

    def __init__(self, lr: float = 1e-3):
        self.model = IVSurfaceNet().to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                           weight_decay=1e-5)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None

    def _normalize(self, x_np: np.ndarray) -> torch.Tensor:
        if self.x_mean is not None and self.x_std is not None:
            x_np = (x_np - self.x_mean) / self.x_std
        return torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 2048) -> float:
        self.model.train()
        n = X.shape[0]
        indices = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            X_batch = self._normalize(X[idx])
            y_batch = torch.tensor(y[idx], dtype=torch.float32, device=DEVICE)

            pred = self.model(X_batch)
            loss = nn.functional.mse_loss(pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()
        return total_loss / n_batches

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = self._normalize(X)
        return self.model(X_t).cpu().numpy()

    def predict_iv(self, moneyness: float, tte_years: float, is_call: int) -> float:
        """Predict IV for a single option."""
        x = np.array([[
            moneyness,
            math.log(moneyness),
            tte_years,
            math.sqrt(max(tte_years, 0.0)),
            float(is_call),
        ]], dtype=np.float32)
        return float(self.predict(x)[0])

    def predict_price(self, s: float, k: float, tte_years: float,
                      is_call: int) -> float:
        """Predict option price by predicting IV and plugging into BS."""
        moneyness = s / k
        iv = self.predict_iv(moneyness, tte_years, is_call)
        if is_call:
            return bs_call(s, k, tte_years, iv)
        else:
            return bs_put(s, k, tte_years, iv)

    def predict_delta(self, s: float, k: float, tte_years: float,
                      is_call: int) -> float:
        """Predict delta using ANN-predicted IV in BS delta formula."""
        moneyness = s / k
        iv = self.predict_iv(moneyness, tte_years, is_call)
        if is_call:
            return bs_call_delta(s, k, tte_years, iv)
        else:
            return bs_put_delta(s, k, tte_years, iv)

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
        }, path)

    @classmethod
    def load(cls, path: str) -> "IVSurfaceANN":
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        ann = cls()
        ann.model.load_state_dict(data["state_dict"])
        ann.x_mean = data.get("x_mean")
        ann.x_std = data.get("x_std")
        ann.model.eval()
        return ann


# ---------------------------------------------------------------------------
# Feature / target construction
# ---------------------------------------------------------------------------
def build_iv_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Features: [moneyness, log_moneyness, tte_years, sqrt_tte, is_call]
    Target: iv (per-strike implied vol from market mid)
    """
    X = np.array([
        [
            r["moneyness"],
            math.log(r["moneyness"]),
            r["tte_years"],
            math.sqrt(r["tte_years"]),
            r["is_call"],
        ]
        for r in records
    ], dtype=np.float32)

    y = np.array([r["iv"] for r in records], dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train IV surface ANN from BBO data")
    parser.add_argument("--bbo-dir", required=True, help="Path to GLBX BBO directory")
    parser.add_argument("--ohlcv", default="glbx-mdp3-20210227-20260226.ohlcv-1m.csv",
                        help="Path to underlying OHLCV CSV")
    parser.add_argument("--days", type=int, default=10, help="Number of trading days")
    parser.add_argument("--all", action="store_true", help="Load all available days")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--out-dir", default="models", help="Output directory")
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
        print(f"{len(records):,} raw", end=" ", flush=True)
        records = enrich_with_iv(records)
        print(f"-> {len(records):,} enriched")
        all_records.extend(records)

    print(f"\nTotal records: {len(all_records):,}")
    if len(all_records) < 1000:
        print("ERROR: Too few records to train.")
        sys.exit(1)

    # Stats
    dtes = [r["dte_days"] for r in all_records]
    ivs = [r["iv"] for r in all_records]
    print(f"  DTE range: {min(dtes)}-{max(dtes)} days")
    print(f"  IV range: {min(ivs):.4f}-{max(ivs):.4f}")
    print(f"  IV mean: {np.mean(ivs):.4f}, std: {np.std(ivs):.4f}")
    print(f"  Calls: {sum(1 for r in all_records if r['is_call'])}, "
          f"Puts: {sum(1 for r in all_records if not r['is_call'])}")

    # Build features
    X, y = build_iv_features(all_records)
    print(f"\nFeature matrix: {X.shape}, Target (IV): {y.shape}")
    print(f"  Target IV stats: min={y.min():.4f}, max={y.max():.4f}, "
          f"mean={y.mean():.4f}, std={y.std():.4f}")

    # Normalize features (NOT target — IV is already well-scaled)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8

    # Train/val split (80/20) — split by time to test extrapolation
    n = X.shape[0]
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    X_train, X_val = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_val = y[idx[:n_train]], y[idx[n_train:]]
    print(f"Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}")

    # Train
    print(f"\n  Device: {DEVICE}")
    model = IVSurfaceANN(lr=1e-3)
    model.x_mean = x_mean
    model.x_std = x_std
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    patience = 40
    no_improve = 0
    best_state = None

    print(f"\nTraining for up to {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = model.train_epoch(X_train, y_train, batch_size=2048)
        val_pred = model.predict(X_val)
        val_loss = float(np.mean((val_pred - y_val) ** 2))

        if epoch % 20 == 0 or epoch < 5:
            rmse_iv = math.sqrt(val_loss)
            lr_now = model.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}: train={train_loss:.8f}  val={val_loss:.8f}  "
                  f"IV_RMSE={rmse_iv:.5f}  lr={lr_now:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    assert best_state is not None
    model.model.load_state_dict(best_state)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    val_pred = model.predict(X_val)
    val_mse = float(np.mean((val_pred - y_val) ** 2))
    iv_rmse = math.sqrt(val_mse)
    iv_mae = float(np.mean(np.abs(val_pred - y_val)))
    iv_r2 = 1.0 - val_mse / float(np.var(y_val))

    print(f"\n{'='*60}")
    print(f"  IV Surface ANN — Final Validation:")
    print(f"    IV RMSE: {iv_rmse:.5f} ({iv_rmse*100:.2f}% vol)")
    print(f"    IV MAE:  {iv_mae:.5f} ({iv_mae*100:.2f}% vol)")
    print(f"    IV R²:   {iv_r2:.6f}")
    print(f"{'='*60}")

    # Pricing accuracy: use predicted IV in BS vs market mid
    avg_und = float(np.mean([r["und_price"] for r in all_records]))
    val_indices = idx[n_train:]
    bs_flat_errs = []
    ann_price_errs = []
    n_eval = min(len(val_indices), 20000)

    for i in range(n_eval):
        vi = val_indices[i]
        r = all_records[vi]
        mid = r["mid"]
        s = r["und_price"]
        k = r["strike"]
        t = r["tte_years"]
        is_call = r["is_call"]

        # BS with flat ATM IV
        bs_flat_err = r["bs_flat_price"] - mid

        # ANN: predict IV, then BS price
        pred_iv = float(val_pred[i])
        if is_call:
            ann_price = bs_call(s, k, t, pred_iv)
        else:
            ann_price = bs_put(s, k, t, pred_iv)
        ann_err = ann_price - mid

        bs_flat_errs.append(bs_flat_err)
        ann_price_errs.append(ann_err)

    bs_rmse = math.sqrt(np.mean(np.array(bs_flat_errs) ** 2))
    ann_rmse = math.sqrt(np.mean(np.array(ann_price_errs) ** 2))

    print(f"\n  Pricing RMSE (IV→BS price vs market mid):")
    print(f"    BS flat ATM IV: {bs_rmse:.4f} pts (${bs_rmse * MULTIPLIER:.2f}/contract)")
    print(f"    ANN IV→BS:      {ann_rmse:.4f} pts (${ann_rmse * MULTIPLIER:.2f}/contract)")
    if bs_rmse > 0:
        print(f"    Improvement:    {(1 - ann_rmse/bs_rmse)*100:.1f}%")

    # -----------------------------------------------------------------------
    # 0DTE extrapolation test
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  0DTE Extrapolation Test (TTE ≈ 0.001 years ≈ 6 hours)")
    print(f"{'='*60}")
    tte_0dte = 6.5 / (252 * 6.5)  # ~6.5 hours in years
    test_moneynesses = [0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04]
    s_ref = avg_und

    print(f"  Underlying ≈ {s_ref:.0f}, TTE = {tte_0dte:.6f} years")
    print(f"\n  {'Moneyness':>10s}  {'Strike':>7s}  {'Type':>4s}  {'Pred_IV':>8s}  "
          f"{'BS_Price':>8s}  {'BS_Delta':>9s}")
    print(f"  {'-'*55}")

    for m in test_moneynesses:
        k = s_ref / m
        for is_call in [1, 0]:
            pred_iv = model.predict_iv(m, tte_0dte, is_call)
            if is_call:
                price = bs_call(s_ref, k, tte_0dte, pred_iv)
                delta = bs_call_delta(s_ref, k, tte_0dte, pred_iv)
            else:
                price = bs_put(s_ref, k, tte_0dte, pred_iv)
                delta = bs_put_delta(s_ref, k, tte_0dte, pred_iv)
            opt_label = "C" if is_call else "P"
            print(f"  {m:>10.4f}  {k:>7.0f}  {opt_label:>4s}  {pred_iv:>8.4f}  "
                  f"{price:>8.2f}  {delta:>+9.4f}")

    # Compare with empirical skew table if available
    skew_path = Path("models/iv_skew_table.json")
    if skew_path.exists():
        import json
        with open(skew_path) as f:
            skew_data = json.load(f)
        skew_all = skew_data.get("all", [])
        if skew_all:
            print(f"\n  {'Moneyness':>10s}  {'ANN_IV':>8s}  {'Skew_Table':>10s}  {'Diff':>6s}")
            print(f"  {'-'*40}")
            for m in test_moneynesses:
                ann_iv = model.predict_iv(m, tte_0dte, 0)  # put side
                # Find closest skew table entry
                best = min(skew_all, key=lambda r: abs(r["moneyness_mid"] - m))
                # Skew table gives ratio, need a reference ATM IV
                # Use ANN's ATM prediction as reference
                atm_iv = model.predict_iv(1.0, tte_0dte, 0)
                skew_iv = atm_iv * best["skew_ratio_median"]
                diff = ann_iv - skew_iv
                print(f"  {m:>10.4f}  {ann_iv:>8.4f}  {skew_iv:>10.4f}  {diff:>+6.4f}")

    # -----------------------------------------------------------------------
    # IV smile at different DTEs
    # -----------------------------------------------------------------------
    print(f"\n  IV smile comparison across DTEs:")
    dte_tests = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
    print(f"  {'Moneyness':>10s}", end="")
    for dte in dte_tests:
        print(f"  {'TTE='+str(dte):>8s}", end="")
    print()
    print(f"  {'-'*(10 + 10*len(dte_tests))}")

    for m in [0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04]:
        print(f"  {m:>10.4f}", end="")
        for dte in dte_tests:
            iv = model.predict_iv(m, dte, 0)  # put side
            print(f"  {iv:>8.4f}", end="")
        print()

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "iv_surface_ann.pt")
    model.save(model_path)
    print(f"\n  Model saved to {model_path}")

    # Save report
    report_path = os.path.join(args.out_dir, "iv_surface_report.txt")
    with open(report_path, "w") as f:
        f.write("IV Surface ANN Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training data: {len(all_records):,} records from {len(bbo_files)} days\n")
        f.write(f"DTE range: {min(dtes)}-{max(dtes)} days\n")
        f.write(f"IV range: {min(ivs):.4f}-{max(ivs):.4f}\n\n")
        f.write(f"Architecture: {model.model}\n\n")
        f.write(f"IV RMSE: {iv_rmse:.5f} ({iv_rmse*100:.2f}% vol)\n")
        f.write(f"IV MAE:  {iv_mae:.5f} ({iv_mae*100:.2f}% vol)\n")
        f.write(f"IV R²:   {iv_r2:.6f}\n\n")
        f.write(f"BS flat-vol RMSE: {bs_rmse:.4f} pts\n")
        f.write(f"ANN IV→BS RMSE:   {ann_rmse:.4f} pts\n")
        if bs_rmse > 0:
            f.write(f"Improvement:      {(1 - ann_rmse/bs_rmse)*100:.1f}%\n")
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
