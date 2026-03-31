"""Compare BS pricing accuracy across IV sources vs real market prices.

Uses collected option data (data/iv_comparison/*.jsonl) which has real
bid/ask/mid from TastyTrade, and computes BS prices with different IV inputs:
  1. flat_iv (fixed 15%)
  2. VIX (30-day)
  3. VIX9D (9-day)
  4. VIX1D (1-day) — best for 0DTE
  5. market_iv (implied from market mid — circular, should be ~0)

Shows how each IV source affects pricing accuracy for the backtest.
"""

import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ET = ZoneInfo("America/New_York")


def bs_price(s, k, t, iv, opt_type):
    if t <= 0 or iv is None or iv <= 0:
        return None

    def _ncdf(x):
        a = abs(x)
        t2 = 1.0 / (1.0 + 0.2316419 * a)
        poly = t2 * (0.319381530 + t2 * (-0.356563782 + t2 * (
            1.781477937 + t2 * (-1.821255978 + t2 * 1.330274429))))
        cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
        return cdf if x >= 0 else 1.0 - cdf

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    if opt_type == "call":
        return s * _ncdf(d1) - k * _ncdf(d2)
    return k * _ncdf(-d2) - s * _ncdf(-d1)


def load_vix_data():
    """Load all VIX variants and return date->iv dicts."""
    result = {}
    for name, filename in [("VIX", "vix_daily.csv"), ("VIX9D", "vix9d_daily.csv"), ("VIX1D", "vix1d_daily.csv")]:
        path = Path("data") / filename
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            lookup = {}
            for idx, row in df.iterrows():
                d = idx.date() if hasattr(idx, 'date') else idx
                lookup[d] = float(row["Close"]) / 100.0
            result[name] = lookup
        else:
            result[name] = {}
    return result


def main():
    data_dir = Path("data/iv_comparison")
    files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        print("No data in data/iv_comparison/")
        return

    vix_data = load_vix_data()

    # Per-source error accumulators
    sources = ["flat_15pct", "VIX", "VIX9D", "VIX1D", "market_iv"]
    errors = {s: [] for s in sources}
    errors_by_date = defaultdict(lambda: {s: [] for s in sources})

    total_records = 0
    for f in files:
        session_date_str = f.stem.split("_")[0]  # handle 2026-03-16_rest.jsonl
        try:
            session_date = date.fromisoformat(session_date_str)
        except ValueError:
            continue

        vix_iv = vix_data["VIX"].get(session_date)
        vix9d_iv = vix_data["VIX9D"].get(session_date)
        vix1d_iv = vix_data["VIX1D"].get(session_date)

        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                mid = rec.get("mid", 0)
                if mid < 0.10:  # skip very cheap options
                    continue

                spot = rec.get("underlying", 0)
                strike = rec.get("strike", 0)
                opt_type = rec.get("type", "")
                tte_years = rec.get("tte_years", 0)
                market_iv_val = rec.get("market_iv")

                if spot <= 0 or strike <= 0 or tte_years <= 0:
                    continue

                # Filter to near-ATM (±4%)
                moneyness = spot / strike
                if moneyness < 0.96 or moneyness > 1.04:
                    continue

                total_records += 1

                # Compute BS price with each IV source
                iv_inputs = {
                    "flat_15pct": 0.15,
                    "VIX": vix_iv,
                    "VIX9D": vix9d_iv,
                    "VIX1D": vix1d_iv,
                    "market_iv": market_iv_val,
                }

                for src, iv in iv_inputs.items():
                    if iv is None or iv <= 0:
                        continue
                    p = bs_price(spot, strike, tte_years, iv, opt_type)
                    if p is not None:
                        err = p - mid
                        errors[src].append({"abs_err": abs(err), "pct_err": abs(err) / mid * 100, "err": err})
                        errors_by_date[session_date][src].append(abs(err))

    # Print summary
    print(f"\n{'='*70}")
    print(f"  BS Pricing Accuracy vs Real Market Mid")
    print(f"  {total_records} option quotes across {len(errors_by_date)} sessions")
    print(f"{'='*70}\n")

    print(f"{'IV Source':<14} {'Count':>6} {'MAE':>8} {'MAPE':>8} {'Med Err':>8} {'Max Err':>8}")
    print("-" * 58)

    for src in sources:
        errs = errors[src]
        if not errs:
            print(f"{src:<14} {'N/A':>6}")
            continue
        n = len(errs)
        mae = sum(e["abs_err"] for e in errs) / n
        mape = sum(e["pct_err"] for e in errs) / n
        sorted_errs = sorted(e["abs_err"] for e in errs)
        median = sorted_errs[n // 2]
        max_err = sorted_errs[-1]
        print(f"{src:<14} {n:>6} {mae:>8.2f} {mape:>7.1f}% {median:>8.2f} {max_err:>8.2f}")

    # Per-date breakdown
    print(f"\n{'Date':<12} ", end="")
    for src in sources:
        print(f"{'MAE '+src:>14}", end="")
    print()
    print("-" * (12 + 14 * len(sources)))

    for d in sorted(errors_by_date.keys()):
        print(f"{d.isoformat():<12} ", end="")
        for src in sources:
            errs = errors_by_date[d][src]
            if errs:
                mae = sum(errs) / len(errs)
                print(f"{mae:>14.2f}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()


if __name__ == "__main__":
    main()
