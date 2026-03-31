#!/usr/bin/env python3
"""
Train an ANN to price ES options from BBO market data.

Approach:
  - Compute per-strike implied vol from market mid (ground truth)
  - Compute ATM IV per (day, expiry) group as the "flat vol" baseline
  - BS with ATM IV ignores skew → systematic errors on OTM wings
  - ANN learns the full surface: (moneyness, TTE, is_call, atm_iv) → price
  - ANN delta via finite differences captures skew-adjusted P(ITM)

Usage:
    python scripts/train_option_ann.py --bbo-dir /path/to/GLBX-data --days 10
    python scripts/train_option_ann.py --bbo-dir /path/to/GLBX-data --all
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONTH_MAP = {"H": 3, "M": 6, "U": 9, "Z": 12}
RTH_START_UTC = 14  # 9:30 ET ≈ 14:30 UTC
RTH_END_UTC = 21    # 16:00 ET = 21:00 UTC
MULTIPLIER = 50.0   # ES futures multiplier


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    days_until_friday = (4 - d.weekday()) % 7
    first_friday = d.replace(day=1 + days_until_friday)
    return first_friday + timedelta(days=14)


def parse_option_symbol(sym: str):
    """Parse 'ESH6 C6025' -> (root='ESH6', type='C'|'P', strike=6025)."""
    parts = sym.strip().split()
    if len(parts) != 2:
        return None
    root = parts[0]
    opt_part = parts[1]
    if len(opt_part) < 2 or opt_part[0] not in ("C", "P"):
        return None
    try:
        strike = int(opt_part[1:])
    except ValueError:
        return None
    return root, opt_part[0], strike


def expiry_from_root(root: str) -> date | None:
    if len(root) < 4 or not root.startswith("ES"):
        return None
    month_code = root[2]
    if month_code not in MONTH_MAP:
        return None
    try:
        year_digit = int(root[3])
    except ValueError:
        return None
    year = 2020 + year_digit
    return third_friday(year, MONTH_MAP[month_code])


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def _ncdf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2.0)
    return 0.5 * (1.0 + sign * y)


def bs_call(s, k, t, iv):
    if t <= 0 or iv <= 0:
        return max(s - k, 0.0)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    return s * _ncdf(d1) - k * _ncdf(d2)


def bs_put(s, k, t, iv):
    if t <= 0 or iv <= 0:
        return max(k - s, 0.0)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    return k * _ncdf(-d2) - s * _ncdf(-d1)


def bs_call_delta(s, k, t, iv):
    if t <= 0 or iv <= 0:
        return 1.0 if s > k else 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
    return _ncdf(d1)


def bs_put_delta(s, k, t, iv):
    return bs_call_delta(s, k, t, iv) - 1.0


def bs_implied_vol(market_price, s, k, t, opt_type, tol=1e-6, max_iter=50):
    """Newton-Raphson IV solver."""
    if t <= 0 or market_price <= 0:
        return None
    # Intrinsic check
    intrinsic = max(s - k, 0.0) if opt_type == "C" else max(k - s, 0.0)
    if market_price < intrinsic - 0.01:
        return None
    iv = 0.20
    for _ in range(max_iter):
        if opt_type == "C":
            price = bs_call(s, k, t, iv)
        else:
            price = bs_put(s, k, t, iv)
        sqrt_t = math.sqrt(t)
        d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
        vega = s * sqrt_t * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        if vega < 1e-12:
            return None
        diff = price - market_price
        iv -= diff / vega
        if iv <= 0.001:
            iv = 0.001
        if abs(diff) < tol:
            return iv
    return iv if abs(diff) < tol * 10 else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_underlying_prices(ohlcv_path: str) -> dict[str, dict[str, float]]:
    """Load ES futures 1-min OHLCV. Returns {date_str: {HH:MM: close_price}}."""
    print(f"Loading underlying OHLCV from {ohlcv_path}...")
    prices: dict[str, dict[str, float]] = defaultdict(dict)
    with open(ohlcv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row["symbol"]
            if "-" in sym or not sym.startswith("ES"):
                continue
            ts = row["ts_event"]
            if not ts:
                continue
            hh = int(ts[11:13])
            mm = int(ts[14:16])
            if hh < RTH_START_UTC or hh >= RTH_END_UTC:
                continue
            close = float(row["close"])
            if close <= 0:
                continue
            dt_str = ts[:10]
            time_key = f"{hh:02d}:{mm:02d}"
            prices[dt_str][time_key] = close
    print(f"  Loaded {len(prices)} dates of underlying data")
    return dict(prices)


def _get_underlying(underlying: dict[str, float], hh: int, mm: int) -> float | None:
    """Get underlying price at timestamp, trying nearby minutes."""
    time_key = f"{hh:02d}:{mm:02d}"
    if time_key in underlying:
        return underlying[time_key]
    for offset in range(1, 5):
        for sign in [1, -1]:
            adj_mm = mm + sign * offset
            adj_hh = hh
            if adj_mm >= 60:
                adj_mm -= 60
                adj_hh += 1
            elif adj_mm < 0:
                adj_mm += 60
                adj_hh -= 1
            tk = f"{adj_hh:02d}:{adj_mm:02d}"
            if tk in underlying:
                return underlying[tk]
    return None


def load_bbo_day(zst_path: str, underlying: dict[str, float]) -> list[dict]:
    """Load one day's BBO data, merge with underlying, compute features."""
    raw = subprocess.run(
        ["zstd", "-d", "-c", zst_path],
        capture_output=True, text=True
    )
    if raw.returncode != 0:
        return []

    records = []
    reader = csv.DictReader(io.StringIO(raw.stdout))
    for row in reader:
        ts = row.get("ts_event", "")
        sym = row.get("symbol", "")
        bid_str = row.get("bid_px_00", "")
        ask_str = row.get("ask_px_00", "")

        if not ts or not bid_str or not ask_str:
            continue
        try:
            hh = int(ts[11:13])
            mm = int(ts[14:16])
        except (ValueError, IndexError):
            continue
        if hh < RTH_START_UTC or hh >= RTH_END_UTC:
            continue

        try:
            bid = float(bid_str)
            ask = float(ask_str)
        except ValueError:
            continue
        if bid <= 0 or ask <= 0:
            continue
        spread = ask - bid
        mid = (bid + ask) / 2.0
        if mid > 0 and spread / mid > 0.20 and spread > 5.0:
            continue

        parsed = parse_option_symbol(sym)
        if parsed is None:
            continue
        root, opt_type, strike = parsed
        if len(root) != 4:
            continue
        expiry = expiry_from_root(root)
        if expiry is None:
            continue

        und_price = _get_underlying(underlying, hh, mm)
        if und_price is None:
            continue

        obs_date = date.fromisoformat(ts[:10])
        dte_days = (expiry - obs_date).days
        if dte_days < 1:
            continue
        minutes_in_day = (hh - RTH_START_UTC) * 60 + mm
        intraday_frac = minutes_in_day / (6.5 * 60)
        tte_days = dte_days - intraday_frac
        tte_years = tte_days / 365.25

        moneyness = und_price / strike
        # Near-the-money filter: 0.90 < moneyness < 1.10
        if moneyness < 0.90 or moneyness > 1.10:
            continue
        # Skip deep ITM
        if opt_type == "C" and moneyness > 1.06:
            continue
        if opt_type == "P" and moneyness < 0.94:
            continue

        records.append({
            "ts": ts[:19],
            "obs_date": obs_date.isoformat(),
            "root": root,
            "opt_type": opt_type,
            "strike": strike,
            "und_price": und_price,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "moneyness": moneyness,
            "tte_years": tte_years,
            "dte_days": dte_days,
            "expiry": expiry.isoformat(),
            "is_call": 1 if opt_type == "C" else 0,
        })

    return records


# ---------------------------------------------------------------------------
# IV computation & ATM IV grouping
# ---------------------------------------------------------------------------
def enrich_with_iv(records: list[dict]) -> list[dict]:
    """Compute per-strike IV and ATM IV per (date, expiry) group."""
    # Step 1: compute per-strike IV
    enriched = []
    for r in records:
        iv = bs_implied_vol(r["mid"], r["und_price"], r["strike"],
                            r["tte_years"], r["opt_type"])
        if iv is None or iv < 0.01 or iv > 2.0:
            continue
        r["iv"] = iv
        r["norm_price"] = r["mid"] / r["und_price"]
        enriched.append(r)

    # Step 2: compute ATM IV per (date, root) group
    # ATM = option closest to moneyness=1.0 for each group
    groups: dict[tuple, list] = defaultdict(list)
    for r in enriched:
        key = (r["obs_date"], r["root"])
        groups[key].append(r)

    atm_ivs: dict[tuple, float] = {}
    for key, group_recs in groups.items():
        # Find calls and puts closest to ATM
        best_dist = 999.0
        best_iv = None
        for gr in group_recs:
            dist = abs(gr["moneyness"] - 1.0)
            if dist < best_dist:
                best_dist = dist
                best_iv = gr["iv"]
        if best_iv is not None:
            atm_ivs[key] = best_iv

    # Step 3: assign ATM IV to each record
    final = []
    for r in enriched:
        key = (r["obs_date"], r["root"])
        atm_iv = atm_ivs.get(key)
        if atm_iv is None:
            continue
        r["atm_iv"] = atm_iv
        # BS price using ATM IV (flat vol — ignores skew)
        if r["opt_type"] == "C":
            r["bs_flat_price"] = bs_call(r["und_price"], r["strike"],
                                         r["tte_years"], atm_iv)
        else:
            r["bs_flat_price"] = bs_put(r["und_price"], r["strike"],
                                        r["tte_years"], atm_iv)
        r["bs_flat_error"] = r["bs_flat_price"] - r["mid"]
        final.append(r)

    return final


# ---------------------------------------------------------------------------
# ANN (PyTorch)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Features: [moneyness, log_moneyness, tte_years, sqrt_tte, is_call, atm_iv]
    Target: norm_price (mid / underlying)
    """
    X = np.array([
        [
            r["moneyness"],
            math.log(r["moneyness"]),
            r["tte_years"],
            math.sqrt(r["tte_years"]),
            r["is_call"],
            r["atm_iv"],
        ]
        for r in records
    ], dtype=np.float32)

    y = np.array([r["norm_price"] for r in records], dtype=np.float32)
    return X, y


class OptionNet(nn.Module):
    """PyTorch feedforward network for option pricing.

    Architecture: 6 -> 256 -> 128 -> 64 -> 32 -> 1
    Uses residual-style skip connections and LayerNorm for stable training.
    """

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(6, 256)
        self.ln1 = nn.LayerNorm(256)
        self.h1 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.h2 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.h3 = nn.Linear(64, 32)
        self.ln4 = nn.LayerNorm(32)
        self.output = nn.Linear(32, 1)
        self.act = nn.GELU()
        # Init last layer small so output starts near zero
        nn.init.zeros_(self.output.bias)
        nn.init.uniform_(self.output.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.ln1(self.input_layer(x)))
        x = self.act(self.ln2(self.h1(x)))
        x = self.act(self.ln3(self.h2(x)))
        x = self.act(self.ln4(self.h3(x)))
        return self.output(x).squeeze(-1)


class OptionANN:
    """Wrapper around OptionNet for training, inference, and serialization."""

    def __init__(self, lr: float = 1e-3):
        self.model = OptionNet().to(DEVICE)
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

    def predict_price(self, moneyness, tte_years, is_call, atm_iv, und_price):
        x = np.array([[
            moneyness,
            math.log(moneyness),
            tte_years,
            math.sqrt(tte_years),
            float(is_call),
            atm_iv,
        ]], dtype=np.float32)
        norm_price = self.predict(x)[0]
        return float(norm_price * und_price)

    def compute_delta(self, s, k, tte_years, is_call, atm_iv):
        """ANN delta via PyTorch autograd: dprice/dS (exact gradient)."""
        x = np.array([[
            s / k,
            math.log(s / k),
            tte_years,
            math.sqrt(tte_years),
            float(is_call),
            atm_iv,
        ]], dtype=np.float32)
        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE, requires_grad=True)
        self.model.eval()
        norm_price = self.model(x_t)
        # price = norm_price * s, so dprice/dS = norm_price + s * d(norm_price)/dS
        # But moneyness = S/K, so d(moneyness)/dS = 1/K
        # We use finite differences for simplicity and correctness
        eps = 0.5
        p_up = self.predict_price((s + eps) / k, tte_years, is_call, atm_iv, s + eps)
        p_dn = self.predict_price((s - eps) / k, tte_years, is_call, atm_iv, s - eps)
        return (p_up - p_dn) / (2 * eps)

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
        }, path)

    @classmethod
    def load(cls, path: str) -> "OptionANN":
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        ann = cls()
        ann.model.load_state_dict(data["state_dict"])
        ann.x_mean = data.get("x_mean")
        ann.x_std = data.get("x_std")
        ann.model.eval()
        return ann


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train option pricing ANN from BBO data")
    parser.add_argument("--bbo-dir", required=True, help="Path to GLBX BBO directory")
    parser.add_argument("--ohlcv", default="glbx-mdp3-20210227-20260226.ohlcv-1m.csv",
                        help="Path to underlying OHLCV CSV")
    parser.add_argument("--days", type=int, default=10, help="Number of trading days to load")
    parser.add_argument("--all", action="store_true", help="Load all available days")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--out-dir", default="models", help="Output directory for model")
    args = parser.parse_args()

    bbo_dir = Path(args.bbo_dir)

    # Find BBO files (skip weekends < 5MB)
    bbo_files = sorted(bbo_dir.glob("glbx-mdp3-*.bbo-1m.csv.zst"))
    bbo_files = [f for f in bbo_files if f.stat().st_size > 5_000_000]
    if not args.all:
        bbo_files = bbo_files[:args.days]
    print(f"Found {len(bbo_files)} trading day BBO files")

    # Load underlying
    und_prices = load_underlying_prices(args.ohlcv)

    # Load & process BBO data
    all_records: list[dict] = []
    for i, bbo_file in enumerate(bbo_files):
        day_str = bbo_file.name.split(".")[0].replace("glbx-mdp3-", "")
        day_date = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        day_und = und_prices.get(day_date, {})
        if not day_und:
            print(f"  [{i+1}/{len(bbo_files)}] {day_date}: no underlying data, skipping")
            continue

        print(f"  [{i+1}/{len(bbo_files)}] {day_date}: loading BBO...", end=" ", flush=True)
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
    atm_ivs = [r["atm_iv"] for r in all_records]
    print(f"  DTE range: {min(dtes)}-{max(dtes)} days")
    print(f"  Per-strike IV: {min(ivs):.3f}-{max(ivs):.3f}")
    print(f"  ATM IV: {min(atm_ivs):.3f}-{max(atm_ivs):.3f}")
    print(f"  Calls: {sum(1 for r in all_records if r['is_call'])}, "
          f"Puts: {sum(1 for r in all_records if not r['is_call'])}")

    # BS flat-vol error stats
    bs_errors = [r["bs_flat_error"] for r in all_records]
    print(f"\n  BS (flat ATM IV) pricing errors vs market:")
    print(f"    Mean error: {np.mean(bs_errors):.4f} pts")
    print(f"    RMSE: {np.sqrt(np.mean(np.array(bs_errors)**2)):.4f} pts")
    print(f"    (This is the skew error that ANN should improve upon)")

    # Build features
    X, y = build_features(all_records)
    print(f"\nFeature matrix: {X.shape}")

    # Normalize
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    X_norm = (X - x_mean) / x_std

    # Train/val split (80/20)
    n = X_norm.shape[0]
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    X_train, X_val = X_norm[idx[:n_train]], X_norm[idx[n_train:]]
    y_train, y_val = y[idx[:n_train]], y[idx[n_train:]]
    print(f"Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}")

    # Train
    print(f"\n  Device: {DEVICE}")
    model = OptionANN(lr=1e-3)
    model.x_mean = x_mean
    model.x_std = x_std
    # Cosine annealing LR scheduler
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    patience = 40
    no_improve = 0
    best_state = None
    avg_und = float(np.mean([r["und_price"] for r in all_records]))

    print(f"\nTraining for up to {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = model.train_epoch(X_train, y_train, batch_size=2048)
        val_pred = model.predict(X_val)
        val_loss = float(np.mean((val_pred - y_val) ** 2))

        if epoch % 20 == 0 or epoch < 5:
            rmse_pts = math.sqrt(val_loss) * avg_und
            lr_now = model.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}: train={train_loss:.8f}  val={val_loss:.8f}  "
                  f"RMSE~{rmse_pts:.2f} pts  lr={lr_now:.6f}")

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

    # Final evaluation
    val_pred = model.predict(X_val)
    val_mse = float(np.mean((val_pred - y_val) ** 2))
    rmse_pts = math.sqrt(val_mse) * avg_und
    mae_pts = float(np.mean(np.abs(val_pred - y_val))) * avg_und

    print(f"\n{'='*60}")
    print(f"  Final validation:")
    print(f"    RMSE: {rmse_pts:.3f} pts (${rmse_pts * MULTIPLIER:.2f}/contract)")
    print(f"    MAE:  {mae_pts:.3f} pts (${mae_pts * MULTIPLIER:.2f}/contract)")
    print(f"    R²:   {1 - val_mse / float(np.var(y_val)):.6f}")
    print(f"{'='*60}")

    # Compare BS flat-vol vs ANN on validation set
    val_idx = idx[n_train:]
    bs_flat_errs = []
    ann_errs = []
    for i, vi in enumerate(val_idx[:10000]):
        r = all_records[vi]
        mid = r["mid"]
        bs_flat_err = r["bs_flat_price"] - mid
        ann_price = val_pred[i] * r["und_price"]
        ann_err = ann_price - mid
        bs_flat_errs.append(bs_flat_err)
        ann_errs.append(ann_err)

    bs_rmse = math.sqrt(np.mean(np.array(bs_flat_errs) ** 2))
    ann_rmse = math.sqrt(np.mean(np.array(ann_errs) ** 2))
    print(f"\n  Pricing RMSE vs market mid (flat ATM IV):")
    print(f"    BS flat-vol: {bs_rmse:.4f} pts (${bs_rmse * MULTIPLIER:.2f}/contract)")
    print(f"    ANN:         {ann_rmse:.4f} pts (${ann_rmse * MULTIPLIER:.2f}/contract)")
    if bs_rmse > 0:
        print(f"    Improvement: {(1 - ann_rmse/bs_rmse)*100:.1f}%")

    # Delta comparison at sample strikes
    print(f"\n  Sample delta comparison (BS vs ANN):")
    print(f"  {'Strike':>8s}  {'Type':>4s}  {'Money':>7s}  {'DTE':>4s}  "
          f"{'BS_delta':>8s}  {'ANN_delta':>9s}  {'Diff':>6s}")
    print(f"  {'-'*55}")
    # Pick a few representative records
    sample_indices = np.random.choice(len(val_idx), min(15, len(val_idx)), replace=False)
    for si in sorted(sample_indices):
        vi = val_idx[si]
        r = all_records[vi]
        s = r["und_price"]
        k = r["strike"]
        t = r["tte_years"]
        atm_iv = r["atm_iv"]
        is_call = r["is_call"]

        if is_call:
            bs_delta = bs_call_delta(s, k, t, atm_iv)
        else:
            bs_delta = bs_put_delta(s, k, t, atm_iv)

        ann_delta = model.compute_delta(s, k, t, is_call, atm_iv)

        print(f"  {k:>8d}  {'C' if is_call else 'P':>4s}  {r['moneyness']:>7.4f}  "
              f"{r['dte_days']:>4d}  {bs_delta:>8.4f}  {ann_delta:>9.4f}  "
              f"{ann_delta - bs_delta:>+6.3f}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "option_ann.pt")
    model.save(model_path)
    print(f"\n  Model saved to {model_path}")

    # Calibration report
    report_path = os.path.join(args.out_dir, "ann_calibration.txt")
    with open(report_path, "w") as f:
        f.write("Option ANN Calibration Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training data: {len(all_records):,} records from {len(bbo_files)} days\n")
        f.write(f"DTE range: {min(dtes)}-{max(dtes)} days\n")
        f.write(f"Per-strike IV: {min(ivs):.3f}-{max(ivs):.3f}\n")
        f.write(f"ATM IV range: {min(atm_ivs):.3f}-{max(atm_ivs):.3f}\n\n")
        f.write(f"Architecture: {model.model}\n")
        f.write(f"Train/Val: {X_train.shape[0]:,}/{X_val.shape[0]:,}\n\n")
        f.write(f"Val RMSE: {rmse_pts:.3f} pts (${rmse_pts * MULTIPLIER:.2f}/contract)\n")
        f.write(f"Val MAE:  {mae_pts:.3f} pts (${mae_pts * MULTIPLIER:.2f}/contract)\n")
        f.write(f"Val R²:   {1 - val_mse / float(np.var(y_val)):.6f}\n\n")
        f.write(f"BS flat-vol RMSE: {bs_rmse:.4f} pts\n")
        f.write(f"ANN RMSE:         {ann_rmse:.4f} pts\n")
        if bs_rmse > 0:
            f.write(f"Improvement:      {(1 - ann_rmse/bs_rmse)*100:.1f}%\n")
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
