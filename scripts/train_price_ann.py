#!/usr/bin/env python3
"""
Train a multi-symbol ANN to predict 0DTE option prices directly.

Predicts normalized option price (mid / spot) from:
  - moneyness (strike / spot)
  - log_moneyness
  - tte_hours (hours remaining to 16:15 ET)
  - sqrt_tte
  - is_call (1/0)
  - realized_vol (optional, rolling 20-period return stdev as vol proxy)
  - symbol one-hot (ES, NQ, CL, GC, SPY, QQQ, IWM)

Usage:
    python scripts/train_price_ann.py                        # train on all data
    python scripts/train_price_ann.py --epochs 500           # more epochs
    python scripts/train_price_ann.py --symbols ES NQ        # specific symbols only
    python scripts/train_price_ann.py --otm-only             # OTM options only
    python scripts/train_price_ann.py --use-vol              # add vol feature
    python scripts/train_price_ann.py --otm-only --use-vol   # both
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/iv_collection")
MODEL_DIR = Path("models")

# Symbol encoding order (fixed for model compatibility)
SYMBOL_ORDER = ["ES", "NQ", "CL", "GC", "SPY", "QQQ", "IWM", "SPX"]
N_SYMBOLS = len(SYMBOL_ORDER)
# Base features: moneyness, log_moneyness, tte_hours, sqrt_tte, is_call
N_BASE_FEATURES = 5
# Extended features (v4): +vix_norm, +realized_vol, +time_of_day
N_EXT_FEATURES = 3
N_FEATURES = N_BASE_FEATURES + N_SYMBOLS  # 5 + 8 = 13 (v3 compat)
N_FEATURES_V4 = N_BASE_FEATURES + N_EXT_FEATURES + N_SYMBOLS  # 5 + 3 + 8 = 16
# Legacy vol-only feature count
N_FEATURES_VOL = N_BASE_FEATURES + 1 + N_SYMBOLS  # 6 + 8 = 14
# v5 extended features: v4 + atm_iv, iv_skew_proxy, realized_vol_5min
N_V5_EXT = 3
N_FEATURES_V5 = N_FEATURES_V4 + N_V5_EXT  # 16 + 3 = 19
PRICE_FLOOR = 0.05  # minimum price for log targets (avoids log(0))


def _bs_price(spot: float, strike: float, tte_years: float, iv: float,
              is_call: bool) -> float:
    """Black-Scholes option price. Returns 0 if inputs are degenerate."""
    if tte_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    from scipy.stats import norm
    sqrt_t = math.sqrt(tte_years)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * tte_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    if is_call:
        return float(spot * norm.cdf(d1) - strike * norm.cdf(d2))
    return float(strike * norm.cdf(-d2) - spot * norm.cdf(-d1))


def load_vix_daily(csv_path: str = "data/vix_daily.csv") -> dict[str, float]:
    """Load VIX daily close into {date_str: vix_close} dict.

    Handles yfinance multi-index CSV format where:
      Row 1: Price,Close,High,Low,Open,Volume
      Row 2: Ticker,^VIX,^VIX,...
      Row 3: Date,,,,,
      Row 4+: 2025-09-02,17.17,...
    First column contains dates, second column contains Close.
    """
    import csv as csv_mod
    vix_map: dict[str, float] = {}
    try:
        with open(csv_path) as f:
            reader = csv_mod.reader(f)
            header = next(reader, [])
            # Find Close column index
            close_idx = 1  # default: second column
            for i, h in enumerate(header):
                if h.strip().lower() == "close":
                    close_idx = i
                    break
            for row in reader:
                if not row or len(row) <= close_idx:
                    continue
                date_str = row[0].strip()[:10]
                # Skip junk rows (Ticker, Date, empty)
                if not date_str or not date_str[0].isdigit():
                    continue
                try:
                    vix_map[date_str] = float(row[close_idx])
                except (ValueError, TypeError):
                    pass
        print(f"  VIX data: {len(vix_map)} days loaded from {csv_path}")
    except FileNotFoundError:
        print(f"  WARNING: VIX file {csv_path} not found — VIX feature will be 0")
    return vix_map


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _derive_implied_forwards(raw_records: list[dict]) -> dict[tuple[str, str], float]:
    """Derive implied forward price per (symbol, timestamp, strike) via put-call parity.

    For futures options: F = K + C_mid - P_mid.
    Returns median forward per (symbol, timestamp_sec) group, which is more
    accurate than the recorded spot (which may be from a different contract).
    """
    from collections import defaultdict

    # Group by (symbol, timestamp_truncated_to_second)
    pairs: dict[tuple[str, str, float], dict] = {}  # (sym, ts, strike) -> {call_mid, put_mid}
    for r in raw_records:
        key = (r["symbol"], r["timestamp"][:19], r["strike"])
        otype = r.get("option_type", "")
        if otype in ("call", "put"):
            if key not in pairs:
                pairs[key] = {}
            pairs[key][otype] = r["mid"]

    # Compute implied forward per (symbol, timestamp) group
    group_forwards: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (sym, ts, strike), data in pairs.items():
        if "call" in data and "put" in data:
            f_implied = strike + data["call"] - data["put"]
            group_forwards[(sym, ts)].append(f_implied)

    # Median forward per group (robust to outliers)
    forward_map: dict[tuple[str, str], float] = {}
    for key, forwards in group_forwards.items():
        forward_map[key] = float(np.median(forwards))

    return forward_map


def load_jsonl_files(data_dir: Path, symbols: list[str] | None = None,
                     otm_only: bool = False,
                     max_records: int = 0) -> list[dict]:
    """Load all JSONL files from the collection directory.

    For futures options, derives implied forward from put-call parity
    to correct for spot/underlying contract mismatches.

    Args:
        otm_only: If True, keep only OTM options (calls with moneyness > 1, puts < 1).
        max_records: If > 0, randomly subsample to this many records (before filtering).
    """
    FUTURES = {"ES", "NQ", "CL", "GC", "RTY", "ZN"}

    # First pass: load raw records for implied forward calculation
    raw_records = []
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        print(f"ERROR: No JSONL files found in {data_dir}")
        sys.exit(1)

    # Shuffle files for random sampling when max_records is set
    if max_records > 0:
        import random
        random.shuffle(files)

    print(f"Found {len(files)} JSONL files")
    n_otm_filtered = 0
    for f in files:
        # Early stop: if we already have enough raw records, stop loading
        if max_records > 0 and len(raw_records) >= max_records * 2:
            print(f"  ... reached {len(raw_records):,} raw records, skipping remaining files")
            break

        n_loaded = 0
        n_skipped = 0
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue

                # Normalize field names across formats:
                # - Collector uses "underlying", converter uses "spot"
                # - Collector uses "type", converter uses "option_type"
                # - Collector may omit "symbol" (infer from filename)
                sym = r.get("symbol", "")
                if not sym:
                    # Infer symbol from filename: *_es.jsonl -> ES, *_spx.jsonl -> SPX
                    fname_lower = f.name.lower()
                    if "_es" in fname_lower or "es_" in fname_lower:
                        sym = "ES"
                    elif "_spx" in fname_lower or "spx_" in fname_lower:
                        sym = "SPX"
                    elif "spy" in fname_lower:
                        sym = "SPY"
                    else:
                        sym = "ES"  # default for legacy collector files
                    r["symbol"] = sym
                if symbols and sym not in symbols:
                    continue
                if sym not in SYMBOL_ORDER:
                    n_skipped += 1
                    continue

                # Normalize spot/underlying
                spot = r.get("spot", 0) or r.get("underlying", 0)
                # Normalize option_type/type
                if "option_type" not in r and "type" in r:
                    r["option_type"] = r["type"]

                # Filter: valid bid/ask/mid
                bid = r.get("bid", 0)
                ask = r.get("ask", 0)
                mid = r.get("mid", 0)
                if not mid and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0

                if bid <= 0 or ask <= 0 or mid <= 0 or spot <= 0:
                    n_skipped += 1
                    continue

                # Spread filter: skip wide markets
                # Futures options have much wider spreads — use tighter
                # threshold to avoid training on noisy mid prices.
                spread = ask - bid
                is_future = sym in FUTURES
                max_spread_pct = 0.15 if is_future else 0.50
                if mid > 0 and spread / mid > max_spread_pct:
                    n_skipped += 1
                    continue

                raw_records.append({
                    **r,
                    "bid": bid, "ask": ask, "mid": mid, "spot": spot,
                    "_file": f.name,
                })
                n_loaded += 1

        print(f"  {f.name}: {n_loaded:,} loaded, {n_skipped:,} skipped", flush=True)

    # Subsample if too many raw records (memory control)
    if max_records > 0 and len(raw_records) > max_records * 2:
        import random
        random.shuffle(raw_records)
        raw_records = raw_records[:max_records * 2]
        print(f"  Subsampled to {len(raw_records):,} raw records (target: {max_records:,})")

    # Derive implied forwards for futures (corrects contract mismatch)
    forward_map = _derive_implied_forwards(raw_records)
    n_forward_corrected = 0

    records = []
    for r in raw_records:
        sym = r["symbol"]
        spot = r["spot"]
        ts_str = r.get("timestamp", "")

        # For futures: use implied forward if available (more accurate than spot)
        if sym in FUTURES:
            ts_key = (sym, ts_str[:19])
            if ts_key in forward_map:
                corrected_spot = forward_map[ts_key]
                # Sanity check: forward should be within 5% of recorded spot
                if abs(corrected_spot - spot) / spot < 0.05:
                    if abs(corrected_spot - spot) > 0.5:
                        n_forward_corrected += 1
                    spot = corrected_spot

        strike = r.get("strike", 0)
        moneyness = strike / spot if spot > 0 else 0
        if moneyness < 0.90 or moneyness > 1.10:
            continue

        is_call = 1.0 if r.get("option_type") == "call" or r.get("is_call") == 1 else 0.0

        # OTM filter
        if otm_only:
            if is_call == 1.0 and moneyness <= 1.0:
                n_otm_filtered += 1
                continue
            if is_call == 0.0 and moneyness >= 1.0:
                n_otm_filtered += 1
                continue

        # Compute TTE (hours to 16:15 ET)
        tte_hours = _compute_tte_hours(ts_str)
        if tte_hours is None or tte_hours <= 0:
            continue

        obs_date = ts_str[:10] if len(ts_str) >= 10 else ""

        records.append({
            "symbol": sym,
            "obs_date": obs_date,
            "strike": strike,
            "spot": spot,
            "bid": r["bid"],
            "ask": r["ask"],
            "mid": r["mid"],
            "moneyness": moneyness,
            "is_call": is_call,
            "tte_hours": tte_hours,
            "norm_mid": r["mid"] / spot,
            "market_iv": r.get("market_iv", 0),
            "delta": r.get("delta", 0),
        })

    if n_forward_corrected > 0:
        print(f"  Forward correction: {n_forward_corrected:,} records adjusted via put-call parity")
    if otm_only:
        print(f"  OTM filter: removed {n_otm_filtered:,} ITM options")
    return records


def _compute_tte_hours(ts_str: str) -> float | None:
    """Compute hours remaining to 16:15 ET from timestamp string."""
    try:
        # Handle both ISO format variations
        ts_str = ts_str.replace("T", " ")
        if "." in ts_str:
            ts_str = ts_str.split(".")[0]
        if "+" in ts_str or ts_str.endswith("Z"):
            ts_str = ts_str.split("+")[0].rstrip("Z")

        # Try parsing with timezone info
        if len(ts_str) == 19:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        elif len(ts_str) >= 10:
            dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        else:
            return None

        # Assume ET if no timezone
        dt_et = dt.replace(tzinfo=ET)
        close = dt_et.replace(hour=16, minute=15, second=0, microsecond=0)

        tte_sec = (close - dt_et).total_seconds()
        return tte_sec / 3600.0
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _compute_realized_vol(records: list[dict]) -> dict[tuple[str, str], float]:
    """Compute realized vol per (symbol, date) from spot prices.

    Uses rolling log-returns of spot across observations, annualized.
    Returns a dict mapping (symbol, date) -> realized_vol (annualized).
    """
    from collections import defaultdict
    # Group spots by (symbol, date) in time order
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        key = (r["symbol"], r["obs_date"])
        grouped[key].append(r["spot"])

    vol_map: dict[tuple[str, str], float] = {}
    for key, spots in grouped.items():
        if len(spots) < 20:
            # Not enough data points — use a simple spread proxy
            if len(spots) >= 2:
                arr = np.array(spots)
                vol_map[key] = float(np.std(np.diff(np.log(arr))))
            else:
                vol_map[key] = 0.0
        else:
            arr = np.array(spots)
            log_returns = np.diff(np.log(arr))
            vol_map[key] = float(np.std(log_returns))

    return vol_map


def build_features(records: list[dict], use_vol: bool = False,
                    v4: bool = False,
                    vix_map: dict[str, float] | None = None,
                    target: str = "both") -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and target vector.

    v3 features (use_vol=False):
      [0] moneyness, [1] log_moneyness, [2] tte_hours, [3] sqrt_tte,
      [4] is_call, [5..12] symbol one-hot

    v4 features (v4=True):
      [0] moneyness, [1] log_moneyness, [2] tte_hours, [3] sqrt_tte,
      [4] is_call, [5] vix_norm, [6] realized_vol, [7] time_of_day,
      [8..15] symbol one-hot

    Target: depends on `target` param:
      "both" -> (norm_bid, norm_ask) shape (n, 2)
      "bid"  -> (norm_bid,) shape (n, 1)
      "ask"  -> (norm_ask,) shape (n, 1)
    """
    SYMBOL_ALIASES = {"ES": "SPX", "SPY": "SPX"}
    sym_to_idx = {s: i for i, s in enumerate(SYMBOL_ORDER)}

    if v4:
        n_feat = N_FEATURES_V4
        base_feat = N_BASE_FEATURES + N_EXT_FEATURES  # 5 + 3 = 8
    elif use_vol:
        n_feat = N_FEATURES_VOL
        base_feat = N_BASE_FEATURES + 1  # 6
    else:
        n_feat = N_FEATURES
        base_feat = N_BASE_FEATURES  # 5

    vol_map = _compute_realized_vol(records) if (use_vol or v4) else {}
    if vix_map is None:
        vix_map = {}

    n_out = 1 if target in ("bid", "ask") else 2
    X = np.zeros((len(records), n_feat), dtype=np.float32)
    y = np.zeros((len(records), n_out), dtype=np.float32)

    for i, r in enumerate(records):
        m = r["moneyness"]
        tte = r["tte_hours"]
        spot = r["spot"]
        X[i, 0] = m
        X[i, 1] = math.log(m) if m > 0 else 0
        X[i, 2] = tte
        X[i, 3] = math.sqrt(tte) if tte > 0 else 0
        X[i, 4] = r["is_call"]

        if v4:
            # VIX normalized (divide by 100 to keep ~0.1-0.5 range)
            vix_val = vix_map.get(r["obs_date"], 0.0)
            X[i, 5] = vix_val / 100.0 if vix_val else 0.0
            # Realized vol
            vol_key = (r["symbol"], r["obs_date"])
            X[i, 6] = vol_map.get(vol_key, 0.0)
            # Time of day (fraction: 0=market open 9:30, 1=close 16:00)
            ts_str = r.get("timestamp", r.get("obs_date", ""))
            X[i, 7] = _time_of_day_frac(ts_str)
        elif use_vol:
            vol_key = (r["symbol"], r["obs_date"])
            X[i, 5] = vol_map.get(vol_key, 0.0)

        # One-hot symbol (with aliases: ES->SPX, SPY->SPX)
        mapped_sym = SYMBOL_ALIASES.get(r["symbol"], r["symbol"])
        si = sym_to_idx.get(mapped_sym)
        if si is not None:
            X[i, base_feat + si] = 1.0

        # Target
        if target == "bid":
            y[i, 0] = r["bid"] / spot if spot > 0 else 0
        elif target == "ask":
            y[i, 0] = r["ask"] / spot if spot > 0 else 0
        else:
            y[i, 0] = r["bid"] / spot if spot > 0 else 0
            y[i, 1] = r["ask"] / spot if spot > 0 else 0

    return X, y


def _time_of_day_frac(ts_str: str) -> float:
    """Convert timestamp to fraction of trading day (0=9:30, 1=16:00)."""
    try:
        ts_str = ts_str.replace("T", " ")
        if "." in ts_str:
            ts_str = ts_str.split(".")[0]
        if "+" in ts_str or ts_str.endswith("Z"):
            ts_str = ts_str.split("+")[0].rstrip("Z")
        parts = ts_str.split(" ")
        if len(parts) < 2:
            return 0.5
        time_parts = parts[1].split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1]) if len(time_parts) > 1 else 0
        minutes_since_open = (hour - 9) * 60 + (minute - 30)
        # Trading day is 390 minutes (9:30 - 16:00)
        return max(0.0, min(1.0, minutes_since_open / 390.0))
    except (ValueError, IndexError):
        return 0.5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PriceNet(nn.Module):
    """Multi-symbol option price predictor.

    Architecture: features → 256 → 128 → 64 → 32 → n_outputs
    Uses LayerNorm + GELU for stable training.
    """

    def __init__(self, n_features: int = N_FEATURES, n_outputs: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape: (batch, n_outputs)


class PriceNetV5(nn.Module):
    """Option price predictor with residual (skip) connections.

    Architecture: features -> proj(256) -> ResBlock(256,128) -> ResBlock(128,64)
                  -> ResBlock(64,32) -> output
    Skip connections help smooth interpolation across the vol surface.
    """

    def __init__(self, n_features: int = N_FEATURES_V5, n_outputs: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.block1 = self._make_block(256, 128)
        self.skip1 = nn.Linear(256, 128)
        self.block2 = self._make_block(128, 64)
        self.skip2 = nn.Linear(128, 64)
        self.block3 = self._make_block(64, 32)
        self.skip3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, n_outputs)

    @staticmethod
    def _make_block(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.block1(h) + self.skip1(h)
        h = self.block2(h) + self.skip2(h)
        h = self.block3(h) + self.skip3(h)
        return self.output(h)


class PriceANN:
    """Wrapper for training, inference, serialization."""

    def __init__(self, lr: float = 1e-3, n_features: int = N_FEATURES,
                 n_outputs: int = 2, target: str = "both", v5: bool = False):
        self.n_outputs = n_outputs
        self.target = target  # "both", "bid", or "ask"
        self.v5 = v5
        if v5:
            self.model = PriceNetV5(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
        else:
            self.model = PriceNet(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None

    def _to_tensor(self, x_np: np.ndarray) -> torch.Tensor:
        if self.x_mean is not None and self.x_std is not None:
            x_np = (x_np - self.x_mean) / (self.x_std + 1e-8)
        return torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 2048) -> float:
        """y shape: (n, 2) — columns are [norm_bid, norm_ask]."""
        self.model.train()
        n = X.shape[0]
        indices = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            X_batch = self._to_tensor(X[idx])
            y_batch = torch.tensor(y[idx], dtype=torch.float32, device=DEVICE)

            pred = self.model(X_batch)  # (batch, 2)
            loss = nn.functional.mse_loss(pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, 2) array of [norm_bid, norm_ask]."""
        self.model.eval()
        X_t = self._to_tensor(X)
        return self.model(X_t).cpu().numpy()

    # Symbol aliases for inference (ES/SPY -> SPX one-hot slot)
    SYMBOL_ALIASES = {"ES": "SPX", "SPY": "SPX"}

    def predict_bid_ask(self, symbol: str, spot: float, strike: float,
                        tte_hours: float, is_call: bool) -> tuple[float, float]:
        """Predict bid and ask for a single option. Returns (bid, ask) in points."""
        m = strike / spot
        mapped = self.SYMBOL_ALIASES.get(symbol, symbol)
        sym_idx = SYMBOL_ORDER.index(mapped) if mapped in SYMBOL_ORDER else 0
        one_hot = [0.0] * N_SYMBOLS
        one_hot[sym_idx] = 1.0

        x = np.array([[
            m,
            math.log(m),
            tte_hours,
            math.sqrt(max(tte_hours, 0)),
            1.0 if is_call else 0.0,
            *one_hot,
        ]], dtype=np.float32)

        pred = self.predict(x)[0]  # [norm_bid, norm_ask]
        return float(pred[0] * spot), float(pred[1] * spot)

    def predict_price(self, symbol: str, spot: float, strike: float,
                      tte_hours: float, is_call: bool) -> float:
        """Predict mid price (average of bid and ask). Backward compatible."""
        bid, ask = self.predict_bid_ask(symbol, spot, strike, tte_hours, is_call)
        return (bid + ask) / 2

    def save(self, path: str, use_vol: bool = False, otm_only: bool = False,
             v4: bool = False, v5: bool = False):
        if self.v5:
            n_feat = self.model.input_proj[0].in_features
            n_out = self.model.output.out_features
        else:
            n_feat = self.model.net[0].in_features
            n_out = self.model.net[-1].out_features
        version = "v5" if v5 else ("v4" if v4 else "v3")
        outputs = {
            "both": ["norm_bid", "norm_ask"],
            "bid": ["norm_bid"],
            "ask": ["norm_ask"],
        }.get(self.target, ["norm_bid", "norm_ask"])
        torch.save({
            "state_dict": self.model.state_dict(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "symbols": SYMBOL_ORDER,
            "n_features": n_feat,
            "n_outputs": n_out,
            "architecture": "PriceNetV5" if v5 else "PriceNet",
            "use_vol": use_vol,
            "otm_only": otm_only,
            "v4": v4,
            "v5": v5,
            "target": self.target,
            "version": version,
            "outputs": outputs,
        }, path)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PriceANN":
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        n_feat = data.get("n_features", N_FEATURES)
        n_out = data.get("n_outputs", 2)
        target = data.get("target", "both")
        v5 = data.get("v5", False)
        ann = cls(n_features=n_feat, n_outputs=n_out, target=target, v5=v5)
        ann.model.load_state_dict(data["state_dict"])
        ann.x_mean = data.get("x_mean")
        ann.x_std = data.get("x_std")
        ann.model.eval()
        return ann


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(ann: PriceANN, X: np.ndarray, y: np.ndarray,
             records: list[dict], label: str = "Test"):
    """Compute and print evaluation metrics.

    y shape: (n, 2) for dual-output or (n, 1) for single-output.
    """
    pred = ann.predict(X)
    n_out = y.shape[1]

    if n_out == 2:
        # Dual output: evaluate on mid, bid, ask
        y_mid = (y[:, 0] + y[:, 1]) / 2
        pred_mid = (pred[:, 0] + pred[:, 1]) / 2
        errors_mid = pred_mid - y_mid
        errors_bid = pred[:, 0] - y[:, 0]
        errors_ask = pred[:, 1] - y[:, 1]
        rmse_bid = math.sqrt(float(np.mean(errors_bid ** 2)))
        rmse_ask = math.sqrt(float(np.mean(errors_ask ** 2)))
    else:
        # Single output
        y_mid = y[:, 0]
        pred_mid = pred[:, 0]
        errors_mid = pred_mid - y_mid
        rmse_bid = rmse_ask = 0.0

    mse_mid = float(np.mean(errors_mid ** 2))
    rmse_mid = math.sqrt(mse_mid)
    mae_mid = float(np.mean(np.abs(errors_mid)))
    r2_mid = 1.0 - mse_mid / max(float(np.var(y_mid)), 1e-12)

    target_label = ann.target if hasattr(ann, 'target') else "mid"

    print(f"\n{'='*70}")
    print(f"  {label} Results ({len(y):,} samples)")
    print(f"{'='*70}")
    if n_out == 2:
        print(f"  Mid  RMSE (norm): {rmse_mid:.6f}   MAE: {mae_mid:.6f}   R²: {r2_mid:.6f}")
        print(f"  Bid  RMSE (norm): {rmse_bid:.6f}")
        print(f"  Ask  RMSE (norm): {rmse_ask:.6f}")
    else:
        print(f"  {target_label}  RMSE (norm): {rmse_mid:.6f}   MAE: {mae_mid:.6f}   R²: {r2_mid:.6f}")

    # Per-symbol breakdown
    sym_results = defaultdict(lambda: {"pred": [], "true": [], "spots": [],
                                        "pred_bid": [], "true_bid": [],
                                        "pred_ask": [], "true_ask": []})
    for i, r in enumerate(records):
        sym_results[r["symbol"]]["pred"].append(pred_mid[i])
        sym_results[r["symbol"]]["true"].append(y_mid[i])
        sym_results[r["symbol"]]["spots"].append(r["spot"])
        sym_results[r["symbol"]]["pred_bid"].append(pred[i, 0])
        sym_results[r["symbol"]]["true_bid"].append(y[i, 0])
        if n_out == 2:
            sym_results[r["symbol"]]["pred_ask"].append(pred[i, 1])
            sym_results[r["symbol"]]["true_ask"].append(y[i, 1])

    if n_out == 2:
        print(f"\n  {'Symbol':>6}  {'N':>8}  {'Mid RMSE':>10}  {'Bid RMSE':>10}  "
              f"{'Ask RMSE':>10}  {'RMSE(pts)':>10}  {'R²(mid)':>8}")
    else:
        print(f"\n  {'Symbol':>6}  {'N':>8}  {'RMSE(norm)':>10}  {'RMSE(pts)':>10}  {'R²':>8}")
    print(f"  {'-'*68}")

    for sym in SYMBOL_ORDER:
        if sym not in sym_results:
            continue
        sr = sym_results[sym]
        p = np.array(sr["pred"])
        t = np.array(sr["true"])
        s = np.array(sr["spots"])

        sym_rmse = math.sqrt(float(np.mean((p - t) ** 2)))
        sym_r2 = 1.0 - float(np.mean((p - t) ** 2)) / max(float(np.var(t)), 1e-12)
        avg_spot = float(np.mean(s))
        rmse_pts = sym_rmse * avg_spot

        if n_out == 2:
            pb = np.array(sr["pred_bid"])
            tb = np.array(sr["true_bid"])
            pa = np.array(sr["pred_ask"])
            ta = np.array(sr["true_ask"])
            sym_rmse_bid = math.sqrt(float(np.mean((pb - tb) ** 2)))
            sym_rmse_ask = math.sqrt(float(np.mean((pa - ta) ** 2)))
            print(f"  {sym:>6}  {len(p):>8,}  {sym_rmse:>10.6f}  {sym_rmse_bid:>10.6f}  "
                  f"{sym_rmse_ask:>10.6f}  {rmse_pts:>10.3f}  {sym_r2:>8.5f}")
        else:
            print(f"  {sym:>6}  {len(p):>8,}  {sym_rmse:>10.6f}  {rmse_pts:>10.3f}  {sym_r2:>8.5f}")

    # Error distribution (on mid)
    pct_within_1 = float(np.mean(np.abs(errors_mid) < 0.001)) * 100
    pct_within_5 = float(np.mean(np.abs(errors_mid) < 0.005)) * 100
    print(f"\n  Error distribution (mid):")
    print(f"    Within 0.1% of spot: {pct_within_1:.1f}%")
    print(f"    Within 0.5% of spot: {pct_within_5:.1f}%")

    return {"rmse": rmse_mid, "mae": mae_mid, "r2": r2_mid, "pred": pred}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train multi-symbol option price ANN")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help=f"Symbols to include (default: all). Choices: {SYMBOL_ORDER}")
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="JSONL data directory")
    parser.add_argument("--npz", default=None, help="Pre-processed .npz file (skips JSONL loading)")
    parser.add_argument("--out-dir", default=str(MODEL_DIR), help="Output directory")
    parser.add_argument("--otm-only", action="store_true", help="Train on OTM options only")
    parser.add_argument("--use-vol", action="store_true", help="Add realized vol as input feature")
    parser.add_argument("--v4", action="store_true",
                        help="Use v4 features: +VIX, +realized_vol, +time_of_day")
    parser.add_argument("--target", choices=["both", "bid", "ask"], default="both",
                        help="Train target: 'both' (dual output), 'bid', or 'ask' (single output)")
    parser.add_argument("--vix-csv", default="data/vix_daily.csv", help="Path to VIX daily CSV")
    parser.add_argument("--temporal-split", action="store_true",
                        help="Split by date (train=oldest 70%%, val=next 15%%, test=newest 15%%) instead of random")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Max training records (0=unlimited). Use to control memory.")
    parser.add_argument("--test-dir", default=None,
                        help="Separate JSONL directory for external test set (e.g. ES collected data)")
    parser.add_argument("--test-symbols", nargs="+", default=None,
                        help="Symbols to include from test-dir (default: all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    symbols = [s.upper() for s in args.symbols] if args.symbols else None

    use_vol = args.use_vol
    otm_only = args.otm_only
    v4 = args.v4
    target = args.target

    n_out = 1 if target in ("bid", "ask") else 2
    vix_map: dict[str, float] = {}
    if v4:
        vix_map = load_vix_daily(args.vix_csv)

    # -----------------------------------------------------------------------
    # Fast path: load from pre-processed .npz
    # -----------------------------------------------------------------------
    if args.npz:
        print(f"Loading pre-processed data from {args.npz}...")
        npz = np.load(args.npz, allow_pickle=True)
        X = npz["X"]
        y_key = f"y_{target}" if target in ("bid", "ask") else "y_both"
        y = npz[y_key]
        meta_symbols = npz["meta_symbols"]
        meta_spots = npz["meta_spots"]
        meta_dates = npz["meta_dates"]
        meta_bids = npz["meta_bids"]
        meta_asks = npz["meta_asks"]
        meta_moneyness = npz["meta_moneyness"]
        meta_is_call = npz["meta_is_call"]
        meta_tte = npz["meta_tte"]

        # Reconstruct minimal records for evaluate()
        meta_strikes = npz["meta_strikes"]
        records = [
            {"symbol": str(meta_symbols[i]), "spot": float(meta_spots[i]),
             "obs_date": str(meta_dates[i]), "bid": float(meta_bids[i]),
             "ask": float(meta_asks[i]), "moneyness": float(meta_moneyness[i]),
             "is_call": float(meta_is_call[i]), "tte_hours": float(meta_tte[i]),
             "strike": float(meta_strikes[i]),
             "mid": (float(meta_bids[i]) + float(meta_asks[i])) / 2}
            for i in range(len(X))
        ]

        n_feat = X.shape[1]
        dates = sorted(set(str(d) for d in meta_dates))
        sym_counts = defaultdict(int)
        for s in meta_symbols:
            sym_counts[str(s)] += 1
        print(f"  {len(X):,} samples, {n_feat} features, {len(dates)} dates")
        for sym in SYMBOL_ORDER:
            if sym in sym_counts:
                print(f"    {sym}: {sym_counts[sym]:,}")
        print(f"  Target: {target} -> y shape {y.shape}")

    # -----------------------------------------------------------------------
    # Slow path: load from JSONL files
    # -----------------------------------------------------------------------
    else:
        mode_str = []
        if otm_only:
            mode_str.append("OTM-only")
        if v4:
            mode_str.append("v4 features (VIX+vol+ToD)")
        elif use_vol:
            mode_str.append("with vol feature")
        if target != "both":
            mode_str.append(f"target={target}")
        mode_label = f" ({', '.join(mode_str)})" if mode_str else ""
        print(f"Loading data from {data_dir}...{mode_label}")
        records = load_jsonl_files(data_dir, symbols=symbols, otm_only=otm_only,
                                   max_records=args.max_records)
        if len(records) < 1000:
            print(f"ERROR: Only {len(records)} records — need at least 1000")
            sys.exit(1)

        dates = sorted(set(r["obs_date"] for r in records))
        sym_counts = defaultdict(int)
        for r in records:
            sym_counts[r["symbol"]] += 1

        print(f"\nTotal: {len(records):,} records across {len(dates)} dates")
        for sym in SYMBOL_ORDER:
            if sym in sym_counts:
                print(f"  {sym}: {sym_counts[sym]:,}")
        print(f"  Dates: {dates[0]} → {dates[-1]}")

        if v4:
            n_feat = N_FEATURES_V4
        elif use_vol:
            n_feat = N_FEATURES_VOL
        else:
            n_feat = N_FEATURES

        X, y = build_features(records, use_vol=use_vol, v4=v4, vix_map=vix_map,
                               target=target)

    print(f"\nFeature matrix: {X.shape}, Target: {y.shape}")
    if n_out == 2:
        print(f"  Bid range:  [{y[:, 0].min():.6f}, {y[:, 0].max():.6f}]")
        print(f"  Ask range:  [{y[:, 1].min():.6f}, {y[:, 1].max():.6f}]")
        print(f"  Mid mean:   {((y[:, 0] + y[:, 1]) / 2).mean():.6f}")
    else:
        print(f"  {target} range: [{y[:, 0].min():.6f}, {y[:, 0].max():.6f}]")
        print(f"  {target} mean:  {y[:, 0].mean():.6f}")

    # Split: temporal or random
    n = len(records)
    if args.temporal_split:
        # Sort indices by date, then split chronologically
        all_dates = [r["obs_date"] for r in records]
        sorted_idx = np.array(sorted(range(n), key=lambda i: all_dates[i]))
        n_test = int(0.15 * n)
        n_val = int(0.15 * n)
        n_train = n - n_test - n_val
        trn_idx = sorted_idx[:n_train]
        val_idx = sorted_idx[n_train:n_train + n_val]
        tst_idx = sorted_idx[n_train + n_val:]
        trn_dates = sorted(set(all_dates[i] for i in trn_idx))
        val_dates = sorted(set(all_dates[i] for i in val_idx))
        tst_dates = sorted(set(all_dates[i] for i in tst_idx))
        print(f"\nTemporal split (70/15/15):")
        print(f"  train: {trn_dates[0]}..{trn_dates[-1]} ({len(trn_dates)} days)")
        print(f"  val:   {val_dates[0]}..{val_dates[-1]} ({len(val_dates)} days)")
        print(f"  test:  {tst_dates[0]}..{tst_dates[-1]} ({len(tst_dates)} days)")
    else:
        idx_all = np.random.permutation(n)
        n_test = int(0.15 * n)
        n_val = int(0.15 * n)
        n_train = n - n_test - n_val
        trn_idx = idx_all[:n_train]
        val_idx = idx_all[n_train:n_train + n_val]
        tst_idx = idx_all[n_train + n_val:]

    X_trn, y_trn = X[trn_idx], y[trn_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[tst_idx], y[tst_idx]
    test_records = [records[i] for i in tst_idx]
    train_records = [records[i] for i in trn_idx]

    print(f"\nSplit (random 70/15/15):")
    print(f"  train={len(y_trn):,}, val={len(y_val):,}, test={len(y_test):,}")

    # Normalize features (fit on training data only)
    x_mean = X_trn.mean(axis=0)
    x_std = X_trn.std(axis=0)
    # Don't normalize one-hot columns (keep them 0/1)
    if v4:
        oh_start = N_BASE_FEATURES + N_EXT_FEATURES
    elif use_vol:
        oh_start = N_BASE_FEATURES + 1
    else:
        oh_start = N_BASE_FEATURES
    for i in range(oh_start, n_feat):
        x_mean[i] = 0.0
        x_std[i] = 1.0

    print(f"\n  Device: {DEVICE}")

    # Train
    ann = PriceANN(lr=args.lr, n_features=n_feat, n_outputs=n_out, target=target)
    ann.x_mean = x_mean
    ann.x_std = x_std
    ann.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ann.optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n")
    for epoch in range(args.epochs):
        train_loss = ann.train_epoch(X_trn, y_trn, batch_size=args.batch_size)

        # Validation loss
        val_pred = ann.predict(X_val)
        if n_out == 2:
            val_mid = (val_pred[:, 0] + val_pred[:, 1]) / 2
            y_val_mid = (y_val[:, 0] + y_val[:, 1]) / 2
            val_loss = float(np.mean((val_mid - y_val_mid) ** 2))
        else:
            val_loss = float(np.mean((val_pred[:, 0] - y_val[:, 0]) ** 2))

        if epoch % 20 == 0 or epoch < 5:
            lr_now = ann.optimizer.param_groups[0]["lr"]
            rmse = math.sqrt(val_loss)
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.8f}  "
                  f"val_rmse={rmse:.6f}  lr={lr_now:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in ann.model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    assert best_state is not None
    ann.model.load_state_dict(best_state)

    # Evaluate on held-out test set
    evaluate(ann, X_test, y_test, test_records, label="Test")

    # Also evaluate on training data for comparison
    evaluate(ann, X_trn, y_trn, train_records[:len(y_trn)], label="Train")

    # External test set (e.g. ES collected data to test SPX-trained model)
    if args.test_dir:
        test_dir = Path(args.test_dir)
        test_syms = [s.upper() for s in args.test_symbols] if args.test_symbols else None
        print(f"\n{'='*70}")
        print(f"  Loading external test set from {test_dir}")
        print(f"{'='*70}")
        ext_records = load_jsonl_files(test_dir, symbols=test_syms, otm_only=otm_only)
        if ext_records:
            ext_dates = sorted(set(r["obs_date"] for r in ext_records))
            ext_sym_counts = defaultdict(int)
            for r in ext_records:
                ext_sym_counts[r["symbol"]] += 1
            print(f"  {len(ext_records):,} records across {len(ext_dates)} dates")
            for sym, cnt in sorted(ext_sym_counts.items()):
                print(f"    {sym}: {cnt:,}")

            X_ext, y_ext = build_features(ext_records, use_vol=use_vol, v4=v4,
                                           vix_map=vix_map, target=target)
            evaluate(ann, X_ext, y_ext, ext_records, label="External (OOS)")
        else:
            print("  No records loaded from external test dir")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_v4" if v4 else "_v3"
    if otm_only:
        suffix += "_otm"
    if target != "both":
        suffix += f"_{target}"
    model_path = os.path.join(args.out_dir, f"price_ann{suffix}.pt")
    ann.save(model_path, use_vol=use_vol, otm_only=otm_only, v4=v4)

    # OTM credit-spread zone analysis (bid/ask)
    print(f"\n{'='*70}")
    print(f"  OTM Credit Spread Zone Analysis (test set)")
    print(f"{'='*70}")
    test_pred_all = ann.predict(X_test)  # (n, 2)
    for sym in SYMBOL_ORDER:
        otm_indices = []
        for j, r in enumerate(test_records):
            if r["symbol"] != sym:
                continue
            if r["is_call"] == 1.0 and r["moneyness"] > 1.0:
                otm_indices.append(j)
            elif r["is_call"] == 0.0 and r["moneyness"] < 1.0:
                otm_indices.append(j)
        if not otm_indices:
            continue
        otm_idx = np.array(otm_indices)
        otm_spots = np.array([test_records[j]["spot"] for j in otm_idx])
        otm_mids = np.array([test_records[j]["mid"] for j in otm_idx])
        if n_out == 2:
            otm_pred_mid = (test_pred_all[otm_idx, 0] + test_pred_all[otm_idx, 1]) / 2
            otm_true_mid = (y_test[otm_idx, 0] + y_test[otm_idx, 1]) / 2
        else:
            otm_pred_mid = test_pred_all[otm_idx, 0]
            otm_true_mid = y_test[otm_idx, 0]
        err_pts = (otm_pred_mid - otm_true_mid) * otm_spots
        rmse_pts = float(np.sqrt(np.mean(err_pts ** 2)))
        avg_mid = float(np.mean(otm_mids))
        pct_err = rmse_pts / avg_mid * 100 if avg_mid > 0 else 0
        if n_out == 2:
            bid_err = (test_pred_all[otm_idx, 0] - y_test[otm_idx, 0]) * otm_spots
            ask_err = (test_pred_all[otm_idx, 1] - y_test[otm_idx, 1]) * otm_spots
            bid_rmse = float(np.sqrt(np.mean(bid_err ** 2)))
            ask_rmse = float(np.sqrt(np.mean(ask_err ** 2)))
            print(f"  {sym:>6}: {len(otm_idx):>6,} OTM, avg mid={avg_mid:.2f}, "
                  f"mid RMSE={rmse_pts:.3f} ({pct_err:.1f}%), "
                  f"bid={bid_rmse:.3f}, ask={ask_rmse:.3f} pts")
        else:
            print(f"  {sym:>6}: {len(otm_idx):>6,} OTM, avg mid={avg_mid:.2f}, "
                  f"{target} RMSE={rmse_pts:.3f} ({pct_err:.1f}%)")

    # Sample predictions
    print(f"\n  Sample predictions (test set):")
    if n_out == 2:
        print(f"  {'Symbol':>6}  {'Strike':>8}  {'Type':>4}  {'Spot':>9}  "
              f"{'MktBid':>7}  {'ANNBid':>7}  {'MktAsk':>7}  {'ANNAsk':>7}  "
              f"{'TTE(h)':>6}")
    else:
        print(f"  {'Symbol':>6}  {'Strike':>8}  {'Type':>4}  {'Spot':>9}  "
              f"{'Market':>7}  {'ANN':>7}  {'Err':>7}  {'TTE(h)':>6}")
    print(f"  {'-'*72}")

    sample_idx = np.random.choice(len(test_records), min(20, len(test_records)), replace=False)
    for si in sorted(sample_idx):
        r = test_records[si]
        spot = r["spot"]
        if n_out == 2:
            pred_bid = test_pred_all[si, 0] * spot
            pred_ask = test_pred_all[si, 1] * spot
            print(f"  {r['symbol']:>6}  {r['strike']:>8.1f}  "
                  f"{'C' if r['is_call'] else 'P':>4}  {spot:>9.2f}  "
                  f"{r['bid']:>7.2f}  {pred_bid:>7.2f}  "
                  f"{r['ask']:>7.2f}  {pred_ask:>7.2f}  "
                  f"{r['tte_hours']:>6.2f}")
        else:
            pred_val = test_pred_all[si, 0] * spot
            mkt_val = r["bid"] if target == "bid" else r["ask"]
            err = pred_val - mkt_val
            print(f"  {r['symbol']:>6}  {r['strike']:>8.1f}  "
                  f"{'C' if r['is_call'] else 'P':>4}  {spot:>9.2f}  "
                  f"{mkt_val:>7.2f}  {pred_val:>7.2f}  {err:>+7.2f}  "
                  f"{r['tte_hours']:>6.2f}")


if __name__ == "__main__":
    main()
