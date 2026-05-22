"""
Build and walk-forward-evaluate a regime selector that chooses between
the two open-IC variants (A=sl2x WF-CV winner, B=vanilla TT) per day.

Pipeline:
  1. Run both strategies, capture per-day PnL.
  2. For each day, compute features observable AT ENTRY (no look-ahead):
       open_range_pts  -- range of 09:30-09:34 1-min bars
       em_pts          -- ATM straddle estimate at 09:35
       open_ratio      -- open_range_pts / em_pts
       vix9d_prev      -- VIX9D close on prior trading day
       overnight_gap   -- |today_open - yesterday_close|
       weekday         -- 0=Mon..4=Fri
  3. Target: 1 if A_pnl > B_pnl, else 0.
  4. Walk-forward eval: train on first 60% of overlap days, test on last 40%.
     Try two model families:
       (a) Univariate cutoff rules on each feature
       (b) Logistic regression with all features (via scipy.optimize)
  5. Compare realized PnL: A baseline, B baseline, selector OOS, oracle OOS.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import yaml
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.backtest.vix_loader import load_vix_series, lookup_prior_close
from src_v2.strategies.open_ic import OpenIcStrategy


VARIANTS = {
    "A_sl2x": {
        "options.target_short_delta": 0.25, "options.wing_width": 25,
        "options.tp_pct_of_credit": 0.25, "options.tp_mode": "total",
        "options.sl_mult_of_credit": 2.0,
        "options.hold_put_breach": True, "options.hold_call_breach": True,
        "entry.opening_range_filter.enabled": True,
        "entry.opening_range_filter.max_open_range_em_ratio": 0.5,
        "strategy.skip_fomc_next_day": True,
        "strategy.skip_weekdays": ["Fri"],
    },
    "B_vanilla_tt": {
        "options.target_short_delta": 0.25, "options.wing_width": 20,
        "options.tp_pct_of_credit": 0.75, "options.tp_mode": "total",
        "options.sl_mult_of_credit": 0.0,
        "options.hold_put_breach": True, "options.hold_call_breach": True,
        "entry.opening_range_filter.enabled": False,
        "strategy.skip_fomc_next_day": False,
        "strategy.skip_weekdays": [],
    },
}


def _set_dotted(cfg, k, v):
    parts = k.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v


def _per_day_pnl(cfg, days, cbbo):
    orch = BacktestOrchestrator(cfg, OpenIcStrategy, cbbo)
    res = orch.run(days)
    by_day = defaultdict(lambda: {"pnl": 0.0, "legs": 0})
    for r in res:
        if r.status == "closed":
            by_day[r.session_date]["pnl"] += r.realized_pnl_usd or 0.0
            by_day[r.session_date]["legs"] += 1
    return {d: v["pnl"] for d, v in by_day.items() if v["legs"] == 2}


def _features_for_day(day_obj, vix_series, prev_close_by_date) -> Dict[str, float] | None:
    """Compute features observable at 09:35 entry. Returns None if insufficient data."""
    bars = day_obj.bars
    if not bars:
        return None

    # Opening 5-min range (09:30-09:34 bars inclusive)
    open_bars = [b for b in bars if 930 <= b["ts_et"].hour * 100 + b["ts_et"].minute < 935]
    if not open_bars:
        return None
    open_range = max(b["high"] for b in open_bars) - min(b["low"] for b in open_bars)

    # ATM straddle EM at 09:35 — approximate via 0.85 × VIX9D × sqrt(1/252) × spot
    today_open = open_bars[0]["open"]
    vix9d = lookup_prior_close(vix_series, day_obj.session_date - timedelta(days=1))
    if vix9d is None:
        return None
    em_pts = (vix9d / 100.0) * today_open / math.sqrt(252)

    # Overnight gap |today_open - yesterday_close|
    prev_close = prev_close_by_date.get(day_obj.session_date)
    overnight_gap = abs(today_open - prev_close) if prev_close else 0.0

    return {
        "open_range": open_range,
        "em_pts": em_pts,
        "open_ratio": open_range / em_pts if em_pts > 0 else 0,
        "vix9d_prev": vix9d,
        "overnight_gap": overnight_gap,
        "weekday": day_obj.session_date.weekday(),
    }


def _prev_close_index(days) -> Dict[date, float]:
    out = {}
    prev = None
    for d in sorted(days, key=lambda x: x.session_date):
        if d.bars:
            if prev is not None:
                out[d.session_date] = prev
            prev = d.bars[-1]["close"]
    return out


# ---------------------------------------------------------------- selector models

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0):
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    def loss(w):
        z = Xb @ w
        # binary cross-entropy with L2 on non-bias terms
        p = _sigmoid(z)
        eps = 1e-9
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return nll / n + l2 * np.sum(w[1:] ** 2)
    res = minimize(loss, np.zeros(k + 1), method="L-BFGS-B")
    return res.x


def _logistic_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return _sigmoid(Xb @ w)  # P(A wins)


def _univariate_cutoff_rules(X_train: np.ndarray, y_train: np.ndarray,
                             pnl_train: np.ndarray, names: list[str]):
    """For each feature, find threshold that maximizes train PnL by A-if-above rule."""
    best_rules = []
    for i, name in enumerate(names):
        col = X_train[:, i]
        candidates = np.unique(col)
        best = None
        for direction in (-1, +1):
            for thr in candidates:
                mask = (col > thr) if direction == +1 else (col < thr)
                # use A on mask (predict y=1), B otherwise
                pnl = pnl_train[mask].sum() if hasattr(pnl_train, "sum") else 0
                if best is None or pnl > best[0]:
                    best = (pnl, thr, direction)
        best_rules.append((name, *best))
    return best_rules


def main():
    with open("config/config_spx_0dte_open_ic.yaml") as f:
        base = yaml.safe_load(f)
    start, end = date(2022, 4, 18), date(2026, 2, 28)
    print(f"Preloading CBBO {start} → {end}…")
    mem = CbboStore.preload_cache("data/cbbo_cache", start=start, end=end)
    cbbo = CbboStore("data/cbbo_cache", preloaded=mem)
    loader = BarLoader(base["backtest"]["data_file"], price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    vix_series = load_vix_series("data/vix9d_daily.csv")
    prev_close = _prev_close_index(days)

    results = {}
    for name, overrides in VARIANTS.items():
        cfg = copy.deepcopy(base)
        for k, v in overrides.items():
            _set_dotted(cfg, k, v)
        print(f"Running {name}…")
        results[name] = _per_day_pnl(cfg, days, cbbo)

    a, b = results["A_sl2x"], results["B_vanilla_tt"]
    overlap = sorted(d for d in a if d in b)
    print(f"\n  Overlap days: {len(overlap)}")
    print(f"  A total on overlap: ${sum(a[d] for d in overlap):+,.0f}")
    print(f"  B total on overlap: ${sum(b[d] for d in overlap):+,.0f}")
    oracle = sum(max(a[d], b[d]) for d in overlap)
    print(f"  Oracle (best-of-day): ${oracle:+,.0f}")

    # Feature engineering on overlap days
    days_by_date = {d.session_date: d for d in days}
    feat_rows = []
    y = []
    a_pnl, b_pnl, dates = [], [], []
    for d in overlap:
        f = _features_for_day(days_by_date[d], vix_series, prev_close)
        if f is None:
            continue
        feat_rows.append(f)
        y.append(1 if a[d] > b[d] else 0)
        a_pnl.append(a[d]); b_pnl.append(b[d]); dates.append(d)

    feat_names = ["open_range", "em_pts", "open_ratio", "vix9d_prev", "overnight_gap", "weekday"]
    X = np.array([[r[k] for k in feat_names] for r in feat_rows], dtype=float)
    y = np.array(y, dtype=float)
    a_pnl = np.array(a_pnl); b_pnl = np.array(b_pnl)
    margin = a_pnl - b_pnl

    n = len(y)
    split = int(n * 0.6)
    print(f"\n  Feature rows: {n}  Train: 1..{split}  Test: {split+1}..{n}")
    print(f"  Train period: {dates[0]} → {dates[split-1]}")
    print(f"  Test  period: {dates[split]} → {dates[-1]}")

    # Baseline OOS sums
    a_oos = a_pnl[split:].sum()
    b_oos = b_pnl[split:].sum()
    oracle_oos = np.maximum(a_pnl[split:], b_pnl[split:]).sum()

    print(f"\n  Baselines (OOS = test set only):")
    print(f"    A alone:           ${a_oos:+,.0f}")
    print(f"    B alone:           ${b_oos:+,.0f}")
    print(f"    Oracle best-of:    ${oracle_oos:+,.0f}")

    # ----- Univariate cutoff rules
    print(f"\n  Univariate cutoff rules (train, evaluated OOS):")
    rules = _univariate_cutoff_rules(X[:split], y[:split], margin[:split], feat_names)
    for name, train_pnl, thr, direction in rules:
        i = feat_names.index(name)
        if direction == +1:
            choose_a = X[split:, i] > thr
        else:
            choose_a = X[split:, i] < thr
        oos_pnl = np.where(choose_a, a_pnl[split:], b_pnl[split:]).sum()
        cmp = ">" if direction == +1 else "<"
        print(f"    if {name} {cmp} {thr:>8.3f}: A-side  train=${train_pnl:>+8.0f}  OOS=${oos_pnl:>+8.0f}")

    # ----- Logistic regression — standardize features
    mu = X[:split].mean(axis=0)
    sd = X[:split].std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    w = _logistic_fit(Xs[:split], y[:split], l2=0.5)
    p_a_test = _logistic_predict(w, Xs[split:])
    choose_a_logit = p_a_test >= 0.5
    logit_oos = np.where(choose_a_logit, a_pnl[split:], b_pnl[split:]).sum()
    print(f"\n  Logistic regression (all features, L2=0.5):")
    print(f"    Weights:   bias={w[0]:+.2f}  " +
          "  ".join(f"{n}={v:+.3f}" for n, v in zip(feat_names, w[1:])))
    print(f"    OOS PnL:   ${logit_oos:+,.0f}  (correctly chose A on "
          f"{(choose_a_logit == y[split:]).mean()*100:.1f}% of test days)")
    print(f"    OOS A-pick fraction: {choose_a_logit.mean()*100:.1f}%")

    # ----- Tiered weekday + logit hybrid
    # Always use B on Fridays (weekday==4), else use logistic prediction
    fri_mask = X[split:, feat_names.index("weekday")] == 4
    hybrid_a = np.where(fri_mask, False, choose_a_logit)
    hybrid_oos = np.where(hybrid_a, a_pnl[split:], b_pnl[split:]).sum()
    print(f"\n  Hybrid (Fri=B always, else logit):  ${hybrid_oos:+,.0f}")


if __name__ == "__main__":
    main()
