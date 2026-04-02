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
    """Black-Scholes option price (scalar). Returns 0 if inputs are degenerate."""
    if tte_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    from scipy.stats import norm
    sqrt_t = math.sqrt(tte_years)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * tte_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    if is_call:
        return float(spot * norm.cdf(d1) - strike * norm.cdf(d2))
    return float(strike * norm.cdf(-d2) - spot * norm.cdf(-d1))


def _bs_price_vec(spots: np.ndarray, strikes: np.ndarray, tte_years: np.ndarray,
                  ivs: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes option price. ~100x faster than per-sample loop."""
    from scipy.stats import norm as sp_norm
    prices = np.zeros(len(spots), dtype=np.float64)
    valid = (tte_years > 0) & (ivs > 0) & (spots > 0) & (strikes > 0)
    if not valid.any():
        return prices
    s = spots[valid]
    k = strikes[valid]
    t = tte_years[valid]
    v = ivs[valid]
    ic = is_call[valid]
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + 0.5 * v * v * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    call_price = s * sp_norm.cdf(d1) - k * sp_norm.cdf(d2)
    put_price = k * sp_norm.cdf(-d2) - s * sp_norm.cdf(-d1)
    prices[valid] = np.where(ic > 0.5, call_price, put_price)
    return prices


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
            "timestamp": ts_str,
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


def _compute_atm_iv(records: list[dict], vix_map: dict[str, float]) -> dict[str, float]:
    """Get ATM IV per date. Uses VIX as proxy (annualized, divide by 100).

    Returns {date_str: atm_iv_annualized}.
    """
    atm_iv_map: dict[str, float] = {}
    for date_str, vix_val in vix_map.items():
        atm_iv_map[date_str] = vix_val / 100.0
    return atm_iv_map


def _compute_realized_vol_5min(records: list[dict]) -> dict[tuple[str, str], float]:
    """Compute 5-minute realized vol per (symbol, date).

    Samples spot every ~5 min (dedup by 5-min bucket), computes stdev of log returns.
    """
    from collections import defaultdict
    bucketed: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in records:
        ts_str = r.get("timestamp", r.get("obs_date", ""))
        try:
            parts = ts_str.replace("T", " ").split(" ")
            if len(parts) < 2:
                continue
            time_parts = parts[1].split(":")
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            bucket = hour * 12 + minute // 5
        except (ValueError, IndexError):
            continue
        key = (r["symbol"], r["obs_date"])
        bucketed[key][bucket] = r["spot"]

    vol_map: dict[tuple[str, str], float] = {}
    for key, buckets in bucketed.items():
        spots = [buckets[b] for b in sorted(buckets.keys())]
        if len(spots) >= 5:
            arr = np.array(spots)
            log_ret = np.diff(np.log(arr))
            vol_map[key] = float(np.std(log_ret))
        else:
            vol_map[key] = 0.0
    return vol_map


def build_features(records: list[dict], use_vol: bool = False,
                    v4: bool = False, v5: bool = False,
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

    if v5:
        n_feat = N_FEATURES_V5
        base_feat = N_BASE_FEATURES + N_EXT_FEATURES + N_V5_EXT  # 5 + 3 + 3 = 11
    elif v4:
        n_feat = N_FEATURES_V4
        base_feat = N_BASE_FEATURES + N_EXT_FEATURES  # 5 + 3 = 8
    elif use_vol:
        n_feat = N_FEATURES_VOL
        base_feat = N_BASE_FEATURES + 1  # 6
    else:
        n_feat = N_FEATURES
        base_feat = N_BASE_FEATURES  # 5

    vol_map = _compute_realized_vol(records) if (use_vol or v4 or v5) else {}
    atm_iv_map = _compute_atm_iv(records, vix_map) if v5 else {}
    vol_5min_map = _compute_realized_vol_5min(records) if v5 else {}
    if vix_map is None:
        vix_map = {}

    n_out = 1 if target in ("bid", "ask") else 2
    n = len(records)
    X = np.zeros((n, n_feat), dtype=np.float32)
    y = np.zeros((n, n_out), dtype=np.float32)

    # Vectorized base features (all versions)
    print("  Building base features (vectorized)...", flush=True)
    moneyness_arr = np.array([r["moneyness"] for r in records], dtype=np.float32)
    tte_arr = np.array([r["tte_hours"] for r in records], dtype=np.float32)
    is_call_arr = np.array([r["is_call"] for r in records], dtype=np.float32)
    safe_m = np.maximum(moneyness_arr, 1e-8)
    X[:, 0] = moneyness_arr
    X[:, 1] = np.log(safe_m)
    X[:, 2] = tte_arr
    X[:, 3] = np.sqrt(np.maximum(tte_arr, 0))
    X[:, 4] = is_call_arr

    if v5 or v4:
        print("  Building extended features...", flush=True)
        vix_arr = np.array([vix_map.get(r["obs_date"], 0.0) / 100.0 for r in records], dtype=np.float32)
        vol_arr = np.array([vol_map.get((r["symbol"], r["obs_date"]), 0.0) for r in records], dtype=np.float32)
        tod_arr = np.array([_time_of_day_frac(r.get("timestamp", r.get("obs_date", ""))) for r in records], dtype=np.float32)
        X[:, 5] = vix_arr
        X[:, 6] = vol_arr
        X[:, 7] = tod_arr

    if v5:
        atm_iv_arr = np.array([atm_iv_map.get(r["obs_date"], 0.15) for r in records], dtype=np.float32)
        X[:, 8] = atm_iv_arr
        X[:, 9] = np.log(safe_m) * atm_iv_arr  # iv_skew_proxy
        vol5_arr = np.array([vol_5min_map.get((r["symbol"], r["obs_date"]), 0.0) for r in records], dtype=np.float32)
        X[:, 10] = vol5_arr

    # One-hot symbols
    for i, r in enumerate(records):
        mapped_sym = SYMBOL_ALIASES.get(r["symbol"], r["symbol"])
        si = sym_to_idx.get(mapped_sym)
        if si is not None:
            X[i, base_feat + si] = 1.0

    # Per-record features for use_vol only (v4/v5 handled above)
    if not v5 and not v4 and use_vol:
        for i, r in enumerate(records):
            vol_key = (r["symbol"], r["obs_date"])
            X[i, 5] = vol_map.get(vol_key, 0.0)

    # Targets for v3/v4 (v5 targets computed vectorized below)
    if not v5:
        spots_arr = np.array([r["spot"] for r in records], dtype=np.float32)
        bids_arr = np.array([r["bid"] for r in records], dtype=np.float32)
        asks_arr = np.array([r["ask"] for r in records], dtype=np.float32)
        safe_spots = np.maximum(spots_arr, 1e-8)
        if target == "bid":
            y[:, 0] = bids_arr / safe_spots
        elif target == "ask":
            y[:, 0] = asks_arr / safe_spots
        else:
            y[:, 0] = bids_arr / safe_spots
            y[:, 1] = asks_arr / safe_spots

    # v5 targets: vectorized BS-residual log(market_price / BS_price)
    if v5:
        print("  Computing BS-residual targets (vectorized)...", flush=True)
        spots_arr = np.array([r["spot"] for r in records], dtype=np.float64)
        strikes_arr = np.array([r["strike"] for r in records], dtype=np.float64)
        tte_arr = np.array([r["tte_hours"] for r in records], dtype=np.float64)
        tte_years_arr = tte_arr / (252.0 * 6.5)
        ivs_arr = np.array([atm_iv_map.get(r["obs_date"], 0.15) for r in records], dtype=np.float64)
        is_call_arr = np.array([r["is_call"] for r in records], dtype=np.float64)

        bs_prices = _bs_price_vec(spots_arr, strikes_arr, tte_years_arr, ivs_arr, is_call_arr)
        bs_norm = np.maximum(bs_prices / spots_arr, PRICE_FLOOR / spots_arr)

        if target == "bid":
            raw = np.maximum(np.array([r["bid"] for r in records], dtype=np.float64) / spots_arr,
                             PRICE_FLOOR / spots_arr)
            y[:, 0] = np.log(raw / bs_norm).astype(np.float32)
        elif target == "ask":
            raw = np.maximum(np.array([r["ask"] for r in records], dtype=np.float64) / spots_arr,
                             PRICE_FLOOR / spots_arr)
            y[:, 0] = np.log(raw / bs_norm).astype(np.float32)
        else:
            raw_bid = np.maximum(np.array([r["bid"] for r in records], dtype=np.float64) / spots_arr,
                                 PRICE_FLOOR / spots_arr)
            raw_ask = np.maximum(np.array([r["ask"] for r in records], dtype=np.float64) / spots_arr,
                                 PRICE_FLOOR / spots_arr)
            y[:, 0] = np.log(raw_bid / bs_norm).astype(np.float32)
            y[:, 1] = np.log(raw_ask / bs_norm).astype(np.float32)

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


def _v5_pred_to_price(pred_log_ratio: float, spot: float, strike: float,
                      tte_hours: float, is_call: bool, atm_iv: float) -> float:
    """Convert v5 log-residual prediction back to option price in points."""
    tte_years = tte_hours / (252.0 * 6.5)
    bs = _bs_price(spot, strike, tte_years, atm_iv, is_call)
    bs = max(bs, PRICE_FLOOR)
    return bs * math.exp(pred_log_ratio)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PriceNet(nn.Module):
    """Multi-symbol option price predictor.

    Architecture: features → 256 → 128 → 64 → 32 → n_outputs
    Uses LayerNorm + GELU for stable training.
    Softplus ensures non-negative output (option prices can't be negative).
    """

    def __init__(self, n_features: int = N_FEATURES, n_outputs: int = 2,
                 nonneg: bool = True):
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
        self.softplus = nn.Softplus(beta=5.0) if nonneg else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)  # shape: (batch, n_outputs)
        if self.softplus is not None:
            out = self.softplus(out)
        return out


class PriceNetV5(nn.Module):
    """Option price predictor with residual (skip) connections.

    Architecture: features -> proj(256) -> ResBlock(256,128) -> ResBlock(128,64)
                  -> ResBlock(64,32) -> output
    Skip connections help smooth interpolation across the vol surface.
    """

    def __init__(self, n_features: int = N_FEATURES_V5, n_outputs: int = 2,
                 nonneg: bool = True):
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
        self.nonneg = nonneg
        self.softplus = nn.Softplus(beta=5.0) if nonneg else None

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
        out = self.output(h)
        if self.softplus is not None:
            out = self.softplus(out)
        return out


class PriceANN:
    """Wrapper for training, inference, serialization."""

    def __init__(self, lr: float = 1e-3, n_features: int = N_FEATURES,
                 n_outputs: int = 2, target: str = "both", v5: bool = False,
                 loss_fn: str = "mse"):
        self.n_outputs = n_outputs
        self.target = target  # "both", "bid", or "ask"
        self.v5 = v5
        self.loss_fn = loss_fn
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

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on configured loss function."""
        if self.loss_fn == "msre":
            # Relative MSE: ((pred - target) / target)^2
            # Equalizes importance across price ranges (OTM vs ITM)
            rel_err = (pred - target) / torch.clamp(target, min=1e-6)
            return (rel_err ** 2).mean()
        elif self.loss_fn == "msle":
            # For v5 BS-residual targets, values are log-ratios (can be negative)
            # Use plain MSE since targets are already in log space
            return nn.functional.mse_loss(pred, target)
        elif self.loss_fn == "weighted":
            # Weighted MSE: emphasize cheap options where target is small
            w = 1.0 / torch.clamp(target.abs(), min=0.0001)
            w = w / w.mean()  # normalize so mean weight = 1
            return (w * (pred - target) ** 2).mean()
        else:
            return nn.functional.mse_loss(pred, target)

    def _to_tensor(self, x_np: np.ndarray) -> torch.Tensor:
        if self.x_mean is not None and self.x_std is not None:
            x_np = (x_np - self.x_mean) / (self.x_std + 1e-8)
        return torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    def preload_gpu(self, X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize and move all data to GPU once. Returns (X_gpu, y_gpu)."""
        if self.x_mean is not None and self.x_std is not None:
            X_norm = (X - self.x_mean) / (self.x_std + 1e-8)
        else:
            X_norm = X
        X_gpu = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
        y_gpu = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        return X_gpu, y_gpu

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 2048,
                    X_gpu: torch.Tensor | None = None,
                    y_gpu: torch.Tensor | None = None) -> float:
        """Train one epoch. If X_gpu/y_gpu provided, uses pre-loaded GPU tensors."""
        self.model.train()

        if X_gpu is not None and y_gpu is not None:
            n = X_gpu.shape[0]
            indices = torch.randperm(n, device=DEVICE)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                pred = self.model(X_gpu[idx])
                loss = self._compute_loss(pred, y_gpu[idx])
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        else:
            # Fallback: CPU numpy path
            n = X.shape[0]
            indices = np.random.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                X_batch = self._to_tensor(X[idx])
                y_batch = torch.tensor(y[idx], dtype=torch.float32, device=DEVICE)
                pred = self.model(X_batch)
                loss = self._compute_loss(pred, y_batch)
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
        nonneg = hasattr(self.model, 'softplus') and self.model.softplus is not None
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
            "nonneg": nonneg,
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
             records: list[dict], label: str = "Test",
             v5: bool = False, vix_map: dict[str, float] | None = None):
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

    # Denormalized OTM accuracy (option price MAPE — the metric that matters)
    if records:
        spots = np.array([r["spot"] for r in records])
        mids = np.array([r["mid"] for r in records])
        pred_prices = pred_mid * spots  # denormalize: norm_price * spot = price
        abs_errors_arr = np.abs(pred_prices - mids)
        mape = float(np.mean(abs_errors_arr / np.maximum(mids, PRICE_FLOOR))) * 100
        rmse_dollars = float(np.sqrt(np.mean(abs_errors_arr ** 2)))
        print(f"\n  Denormalized (price):")
        print(f"    MAPE: {mape:.1f}%")
        print(f"    RMSE: ${rmse_dollars:.2f}")

        # Delta-zone breakdown: approximate BS delta per sample, bucket by |delta|
        from scipy.stats import norm as sp_norm
        tte_arr = np.array([r["tte_hours"] for r in records], dtype=np.float64)
        strikes_arr = np.array([r["strike"] for r in records], dtype=np.float64)
        is_call_arr = np.array([r["is_call"] for r in records], dtype=np.float64)
        tte_years = tte_arr / (252.0 * 6.5)
        # Use VIX as IV proxy; fall back to 0.20 if unavailable
        if vix_map:
            iv_arr = np.array([vix_map.get(r.get("obs_date", ""), 20.0) / 100.0
                               for r in records], dtype=np.float64)
        else:
            iv_arr = np.full(len(records), 0.20, dtype=np.float64)

        valid = (tte_years > 0) & (iv_arr > 0) & (spots > 0) & (strikes_arr > 0)
        abs_delta = np.zeros(len(records), dtype=np.float64)
        if valid.any():
            sqrt_t = np.sqrt(tte_years[valid])
            d1 = (np.log(spots[valid] / strikes_arr[valid]) +
                  0.5 * iv_arr[valid]**2 * tte_years[valid]) / (iv_arr[valid] * sqrt_t)
            call_delta = sp_norm.cdf(d1)
            put_delta = call_delta - 1.0
            abs_delta[valid] = np.where(is_call_arr[valid] > 0.5,
                                        call_delta, np.abs(put_delta))

        # Delta buckets relevant to credit spread trading
        buckets = [
            ("0.03-0.08 (deep OTM)", 0.03, 0.08),
            ("0.08-0.12 (far OTM)",  0.08, 0.12),
            ("0.12-0.18 (target)",   0.12, 0.18),
            ("0.18-0.25 (near ATM)", 0.18, 0.25),
            ("0.25-0.40 (ATM-ish)",  0.25, 0.40),
        ]
        print(f"\n  Delta-zone breakdown:")
        print(f"    {'Zone':<22} {'N':>8}  {'RMSE($)':>8}  {'MAE($)':>8}  "
              f"{'MAPE':>7}  {'AvgMid':>7}  {'AvgErr':>7}")
        print(f"    {'-'*75}")
        for zone_name, lo, hi in buckets:
            mask = (abs_delta >= lo) & (abs_delta < hi)
            n_zone = int(mask.sum())
            if n_zone < 10:
                print(f"    {zone_name:<22} {n_zone:>8}  {'--':>8}")
                continue
            zone_pred = pred_prices[mask]
            zone_mkt = mids[mask]
            zone_err = zone_pred - zone_mkt
            zone_abs_err = np.abs(zone_err)
            zone_rmse = float(np.sqrt(np.mean(zone_err ** 2)))
            zone_mae = float(np.mean(zone_abs_err))
            zone_mape = float(np.mean(zone_abs_err / np.maximum(zone_mkt, PRICE_FLOOR))) * 100
            zone_avg_mid = float(np.mean(zone_mkt))
            zone_bias = float(np.mean(zone_err))
            print(f"    {zone_name:<22} {n_zone:>8,}  {zone_rmse:>8.2f}  {zone_mae:>8.2f}  "
                  f"{zone_mape:>6.1f}%  {zone_avg_mid:>7.2f}  {zone_bias:>+7.2f}")

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
    parser.add_argument("--v5", action="store_true",
                        help="Use v5 features: v4 + atm_iv, iv_skew_proxy, realized_vol_5min; BS-residual targets")
    parser.add_argument("--target", choices=["both", "bid", "ask"], default="both",
                        help="Train target: 'both' (dual output), 'bid', or 'ask' (single output)")
    parser.add_argument("--vix-csv", default="data/vix_daily.csv", help="Path to VIX daily CSV")
    parser.add_argument("--temporal-split", action="store_true",
                        help="Sort days chronologically before splitting (oldest->train, newest->test). "
                             "Default: random day shuffle (valid for 0DTE since each day is independent)")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Max training records (0=unlimited). Use to control memory.")
    parser.add_argument("--test-dir", default=None,
                        help="Separate JSONL directory for external test set (e.g. ES collected data)")
    parser.add_argument("--loss", choices=["mse", "msle", "weighted", "msre"], default="mse",
                        help="Loss function: mse (default), msle (log-space), weighted (inverse-price), msre (relative)")
    parser.add_argument("--test-symbols", nargs="+", default=None,
                        help="Symbols to include from test-dir (default: all)")
    parser.add_argument("--oversample-otm", type=int, default=0,
                        help="Oversample OTM options N times in training set (0=disabled, 3 recommended)")
    parser.add_argument("--finetune", default=None,
                        help="Path to pretrained model for fine-tuning (freezes all but last 2 layers)")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning (default: 1e-4, lower than full training)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    symbols = [s.upper() for s in args.symbols] if args.symbols else None

    use_vol = args.use_vol
    otm_only = args.otm_only
    v4 = args.v4
    v5 = args.v5
    target = args.target

    n_out = 1 if target in ("bid", "ask") else 2
    vix_map: dict[str, float] = {}
    if v4 or v5:
        vix_map = load_vix_daily(args.vix_csv)

    # -----------------------------------------------------------------------
    # Fast path: load from pre-processed .npz
    # -----------------------------------------------------------------------
    if args.npz:
        print(f"Loading pre-processed data from {args.npz}...")
        npz = np.load(args.npz, allow_pickle=True)
        X = npz["X"]
        if target in ("bid", "ask"):
            y = npz[f"y_{target}"]
        else:
            # Stack y_bid + y_ask as (n, 2) — avoids broken BS-residual y_both
            y = np.hstack([npz["y_bid"], npz["y_ask"]])
        meta_symbols = npz["meta_symbols"]
        meta_spots = npz["meta_spots"]
        meta_dates = npz["meta_dates"]
        meta_bids = npz["meta_bids"]
        meta_asks = npz["meta_asks"]
        meta_moneyness = npz["meta_moneyness"]
        meta_is_call = npz["meta_is_call"]
        meta_tte = npz["meta_tte"]

        # Store metadata arrays — defer record creation to only test/eval indices
        meta_strikes = npz["meta_strikes"]
        meta_timestamps = npz["meta_timestamps"] if "meta_timestamps" in npz else meta_dates
        meta_market_iv = npz["meta_market_iv"] if "meta_market_iv" in npz else np.zeros(len(X))

        def _make_records(indices):
            """Create record dicts only for given indices (lazy, avoids 2M+ dicts)."""
            return [
                {"symbol": str(meta_symbols[i]), "spot": float(meta_spots[i]),
                 "obs_date": str(meta_dates[i]), "bid": float(meta_bids[i]),
                 "ask": float(meta_asks[i]), "moneyness": float(meta_moneyness[i]),
                 "is_call": float(meta_is_call[i]), "tte_hours": float(meta_tte[i]),
                 "strike": float(meta_strikes[i]),
                 "mid": (float(meta_bids[i]) + float(meta_asks[i])) / 2,
                 "timestamp": str(meta_timestamps[i]),
                 "market_iv": float(meta_market_iv[i])}
                for i in indices
            ]
        # Placeholder — real records built lazily after split
        records = None

        npz_is_v5 = bool(npz["v5"][0]) if "v5" in npz else False
        if v5 and npz_is_v5:
            print(f"  NPZ has v5 features ({X.shape[1]} dims), targets are price/spot")
        elif v5 and not npz_is_v5:
            print("  WARNING: NPZ is v4 but --v5 requested. Use v5 NPZ for best results.")
            print("  Using v4 features with v5 architecture (no v5-specific features)")

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
        if v5:
            mode_str.append("v5 features (BS-residual targets)")
        elif v4:
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
        print(f"  Dates: {dates[0]} -> {dates[-1]}")

        if v5:
            n_feat = N_FEATURES_V5
        elif v4:
            n_feat = N_FEATURES_V4
        elif use_vol:
            n_feat = N_FEATURES_VOL
        else:
            n_feat = N_FEATURES

        X, y = build_features(records, use_vol=use_vol, v4=v4, v5=v5,
                               vix_map=vix_map, target=target)

    print(f"\nFeature matrix: {X.shape}, Target: {y.shape}")
    if n_out == 2:
        print(f"  Bid range:  [{y[:, 0].min():.6f}, {y[:, 0].max():.6f}]")
        print(f"  Ask range:  [{y[:, 1].min():.6f}, {y[:, 1].max():.6f}]")
        print(f"  Mid mean:   {((y[:, 0] + y[:, 1]) / 2).mean():.6f}")
    else:
        print(f"  {target} range: [{y[:, 0].min():.6f}, {y[:, 0].max():.6f}]")
        print(f"  {target} mean:  {y[:, 0].mean():.6f}")

    # -----------------------------------------------------------------------
    # Day-level split: each 0DTE day is an independent data chunk.
    # Split by whole days to prevent intraday leakage across sets.
    # Default: random day shuffle (days are independent — no time-series).
    # --temporal-split: sort days chronologically (for comparison only).
    # -----------------------------------------------------------------------
    n = len(X)
    if records is not None:
        all_dates_arr = np.array([r["obs_date"] for r in records], dtype="U10")
    else:
        all_dates_arr = meta_dates  # from NPZ

    # Group row indices by date
    unique_dates = np.unique(all_dates_arr)
    date_to_indices: dict[str, np.ndarray] = {}
    for d in unique_dates:
        date_to_indices[str(d)] = np.where(all_dates_arr == d)[0]

    n_days = len(unique_dates)
    n_test_days = max(1, int(0.15 * n_days))
    n_val_days = max(1, int(0.15 * n_days))
    n_train_days = n_days - n_test_days - n_val_days

    if args.temporal_split:
        # Sort days chronologically (oldest → train, middle → val, newest → test)
        day_order = np.array(sorted(unique_dates))
        split_label = "Temporal day-level"
    else:
        # Random day shuffle — valid because 0DTE days are independent
        day_order = np.array(sorted(unique_dates))
        np.random.shuffle(day_order)
        split_label = "Random day-level"

    trn_days = day_order[:n_train_days]
    val_days = day_order[n_train_days:n_train_days + n_val_days]
    tst_days = day_order[n_train_days + n_val_days:]

    # Expand day-level splits back to row indices
    trn_idx = np.concatenate([date_to_indices[str(d)] for d in trn_days])
    val_idx = np.concatenate([date_to_indices[str(d)] for d in val_days])
    tst_idx = np.concatenate([date_to_indices[str(d)] for d in tst_days])

    # Shuffle row order within each split (intra-day order doesn't matter for training)
    np.random.shuffle(trn_idx)
    np.random.shuffle(val_idx)

    X_trn, y_trn = X[trn_idx], y[trn_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[tst_idx], y[tst_idx]

    # Build records lazily — only for test/train eval (not all 2M+)
    if records is not None:
        test_records = [records[i] for i in tst_idx]
        train_records = [records[i] for i in trn_idx]
    else:
        # NPZ path: build only test records now, train records deferred
        print("  Building test records for evaluation...", flush=True)
        test_records = _make_records(tst_idx)
        train_records = None  # built later if needed

    # OTM oversampling (uses numpy arrays, not records)
    if args.oversample_otm > 0:
        trn_is_call = meta_is_call[trn_idx] if records is None else np.array([r["is_call"] for r in train_records])
        trn_moneyness = meta_moneyness[trn_idx] if records is None else np.array([r["moneyness"] for r in train_records])
        otm_mask = ((trn_is_call > 0.5) & (trn_moneyness > 1.0)) | \
                   ((trn_is_call < 0.5) & (trn_moneyness < 1.0))
        n_otm = int(otm_mask.sum())
        if n_otm > 0:
            otm_X = X_trn[otm_mask]
            otm_y = y_trn[otm_mask]
            extras = [otm_X] * (args.oversample_otm - 1)
            extras_y = [otm_y] * (args.oversample_otm - 1)
            X_trn = np.concatenate([X_trn] + extras)
            y_trn = np.concatenate([y_trn] + extras_y)
            print(f"  OTM oversampling: {n_otm:,} OTM samples x{args.oversample_otm} "
                  f"-> {len(X_trn):,} total training samples")

    trn_days_sorted = sorted(trn_days)
    val_days_sorted = sorted(val_days)
    tst_days_sorted = sorted(tst_days)
    print(f"\n{split_label} split (70/15/15):")
    print(f"  train: {len(trn_days)} days, {len(y_trn):,} samples "
          f"({trn_days_sorted[0]}..{trn_days_sorted[-1]})")
    print(f"  val:   {len(val_days)} days, {len(y_val):,} samples "
          f"({val_days_sorted[0]}..{val_days_sorted[-1]})")
    print(f"  test:  {len(tst_days)} days, {len(y_test):,} samples "
          f"({tst_days_sorted[0]}..{tst_days_sorted[-1]})")

    # Normalize features (fit on training data only)
    x_mean = X_trn.mean(axis=0)
    x_std = X_trn.std(axis=0)
    # Don't normalize one-hot columns (keep them 0/1)
    if v5:
        oh_start = N_BASE_FEATURES + N_EXT_FEATURES + N_V5_EXT  # 5+3+3=11
    elif v4:
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
    ann = PriceANN(lr=args.lr, n_features=n_feat, n_outputs=n_out, target=target,
                   v5=v5, loss_fn=args.loss)
    ann.x_mean = x_mean
    ann.x_std = x_std
    ann.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ann.optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Fine-tuning: load pretrained weights, freeze early layers
    if args.finetune:
        print(f"\n  Fine-tuning from {args.finetune}")
        pretrained = torch.load(args.finetune, map_location=DEVICE, weights_only=False)
        state = pretrained["state_dict"]
        model_state = ann.model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        ann.model.load_state_dict(model_state)
        print(f"  Loaded {loaded}/{len(state)} layers from pretrained model")

        # Load normalization stats if available
        if pretrained.get("x_mean") is not None:
            ann.x_mean = pretrained["x_mean"]
            ann.x_std = pretrained["x_std"]
            print("  Loaded feature normalization from pretrained")

        # Freeze all but last 2 layers
        if ann.v5:
            # PriceNetV5: freeze input_proj, block1, skip1, block2, skip2
            for name, param in ann.model.named_parameters():
                if not any(s in name for s in ["block3", "skip3", "output"]):
                    param.requires_grad = False
        else:
            # PriceNet: freeze all but last Linear + LayerNorm + output
            layers = list(ann.model.net)
            for layer in layers[:-4]:  # freeze all but last block + output
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in ann.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in ann.model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} parameters")

        # Use lower LR for fine-tuning
        ann.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, ann.model.parameters()),
            lr=args.finetune_lr, weight_decay=1e-5
        )
        ann.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            ann.optimizer, T_max=args.epochs, eta_min=1e-6
        )

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    # Pre-load training data to GPU for fast epochs
    if str(DEVICE) != "cpu":
        print(f"  Pre-loading {len(X_trn):,} training samples to GPU...", flush=True)
        X_trn_gpu, y_trn_gpu = ann.preload_gpu(X_trn, y_trn)
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    else:
        X_trn_gpu, y_trn_gpu = None, None

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n")
    for epoch in range(args.epochs):
        train_loss = ann.train_epoch(X_trn, y_trn, batch_size=args.batch_size,
                                     X_gpu=X_trn_gpu, y_gpu=y_trn_gpu)

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
    evaluate(ann, X_test, y_test, test_records, label="Test", v5=v5, vix_map=vix_map)

    # Also evaluate on training data for comparison (subsample to avoid building 6M records)
    if train_records is None:
        # NPZ path: build records for a subsample of training data
        trn_sample_n = min(50000, len(trn_idx))
        trn_sample_idx = trn_idx[:trn_sample_n]
        train_records_sample = _make_records(trn_sample_idx)
        evaluate(ann, X_trn[:trn_sample_n], y_trn[:trn_sample_n],
                 train_records_sample, label="Train", v5=v5, vix_map=vix_map)
    else:
        evaluate(ann, X_trn, y_trn, train_records[:len(y_trn)], label="Train", v5=v5, vix_map=vix_map)

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
                                           v5=v5, vix_map=vix_map, target=target)
            evaluate(ann, X_ext, y_ext, ext_records, label="External (OOS)", v5=v5, vix_map=vix_map)
        else:
            print("  No records loaded from external test dir")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_v5" if v5 else ("_v4" if v4 else "_v3")
    if otm_only:
        suffix += "_otm"
    if target != "both":
        suffix += f"_{target}"
    model_path = os.path.join(args.out_dir, f"price_ann{suffix}.pt")
    ann.save(model_path, use_vol=use_vol, otm_only=otm_only, v4=v4, v5=v5)

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
