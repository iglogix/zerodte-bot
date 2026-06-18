"""ORB directional — post-retest favorable-excursion diagnostic.

Gate question (debit-spread idea): after the breakout-retest entry, does the
underlying CONTINUE in the trend direction far enough and FAST enough to make a
long-premium (debit) structure pay, or is the move small/slow (so 0DTE theta
eats it and the OTM credit spread is the right structure)?

Method: run the surviving credit config to get the real entry set (entry_time +
bias per day), then overlay the SPY*10 1-min path from entry -> EOD and measure
signed excursion in the trend direction (favorable = with the breakout).

No option pricing here — this is the cheap underlying-behavior gate. If it says
"big & fast", THEN it's worth building/backtesting an actual ATM debit variant.

Usage:
    .venv/bin/python -m scripts.orb_dir_excursion_diag \
        --config config/config_orb_dir_breakout_spx.yaml \
        --start 2022-04-18 --end 2026-06-11
"""
from __future__ import annotations

import argparse
import logging
import statistics as st
import sys
from datetime import date, time
from pathlib import Path

import yaml

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.strategies.orb_em_directional import OrbEmDirectionalStrategy

EOD = time(15, 50)


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_orb_dir_breakout_spx.yaml")
    ap.add_argument("--start", default="2022-04-18")
    ap.add_argument("--end", default="2026-06-11")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("backtest", {})["cbbo_cache_dir"] = "data/cbbo_cache"
    data_file = cfg["backtest"].get("data_file", "data/spy_1m.csv")
    price_scale = cfg["backtest"].get("price_scale", 10.0)
    symbol = cfg.get("bot", {}).get("symbol", "SPX")

    print(f"Preloading CBBO {args.start}..{args.end}...", flush=True)
    mem = CbboStore.preload_cache(
        "data/cbbo_cache",
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
    cbbo = CbboStore("data/cbbo_cache", preloaded=mem)

    loader = BarLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
    ))
    bars_by_day = {d.session_date: d.bars for d in days}
    print(f"  {len(days)} trading days", flush=True)

    orch = BacktestOrchestrator(cfg, OrbEmDirectionalStrategy, cbbo)
    results = orch.run(days)
    trades = [r for r in results if r.status == "closed" and r.entry_time and r.bias]
    print(f"  {len(trades)} executed directional entries\n", flush=True)

    rows = []
    for r in trades:
        bars = bars_by_day.get(r.session_date)
        if not bars:
            continue
        et = r.entry_time
        # entry-time-of-day in ET
        et_tod = et.time() if hasattr(et, "time") else et
        # path from the first bar at/after entry through EOD
        path = [b for b in bars if et_tod <= b["ts_et"].time() <= EOD]
        if len(path) < 2:
            continue
        anchor = path[0]["close"]
        sign = 1.0 if r.bias == "long" else -1.0

        mfe = 0.0          # max favorable excursion (pts)
        t_mfe = 0          # minutes from entry to MFE
        mae = 0.0          # max adverse excursion (pts, positive number)
        fav_at = {15: None, 30: None, 60: None}
        for i, b in enumerate(path):
            # intrabar favorable extreme: use high for long, low for short
            fav_extreme = sign * ((b["high"] if sign > 0 else b["low"]) - anchor)
            adv_extreme = sign * ((b["low"] if sign > 0 else b["high"]) - anchor)
            if fav_extreme > mfe:
                mfe = fav_extreme
                t_mfe = i
            if -adv_extreme > mae:
                mae = -adv_extreme
            for m in fav_at:
                if i == m and fav_at[m] is None:
                    fav_at[m] = sign * (b["close"] - anchor)
        eod_fav = sign * (path[-1]["close"] - anchor)
        # fraction of MFE captured within the first 30 min
        f30 = (fav_at[30] / mfe) if (mfe > 0 and fav_at[30] is not None) else 0.0
        rows.append({
            "date": r.session_date, "bias": r.bias, "mfe": mfe, "t_mfe": t_mfe,
            "mae": mae, "eod_fav": eod_fav,
            "f15": fav_at[15], "f30": fav_at[30], "f60": fav_at[60], "f30frac": f30,
        })

    n = len(rows)
    if not n:
        print("No rows.")
        return 1

    mfe = [r["mfe"] for r in rows]
    mae = [r["mae"] for r in rows]
    t_mfe = [r["t_mfe"] for r in rows]
    eod = [r["eod_fav"] for r in rows]

    print(f"=== POST-RETEST FAVORABLE EXCURSION (n={n} entries) ===\n")
    print("Favorable excursion = move WITH the breakout direction, in SPX points.\n")

    print("MFE (max favorable excursion, pts):")
    print(f"  median {_pct(mfe,.5):5.1f}   p25 {_pct(mfe,.25):5.1f}   p75 {_pct(mfe,.75):5.1f}"
          f"   p90 {_pct(mfe,.9):5.1f}   max {max(mfe):5.1f}")
    print(f"  mean {st.mean(mfe):5.1f}\n")

    print("MAE (max adverse excursion, pts — how far it goes against first):")
    print(f"  median {_pct(mae,.5):5.1f}   p75 {_pct(mae,.75):5.1f}   p90 {_pct(mae,.9):5.1f}\n")

    print("Net move at EOD (signed, with-trend +):")
    print(f"  median {_pct(eod,.5):5.1f}   mean {st.mean(eod):5.1f}"
          f"   %positive {100*sum(1 for x in eod if x>0)/n:4.0f}%\n")

    print("Fraction of entries reaching favorable threshold (any time before EOD):")
    for thr in (5, 10, 15, 20, 25):
        frac = sum(1 for x in mfe if x >= thr) / n
        print(f"  MFE >= {thr:2d} pt : {100*frac:5.1f}%")
    print()

    print("Speed — minutes from entry to MFE:")
    print(f"  median {_pct(t_mfe,.5):5.0f}m   p25 {_pct(t_mfe,.25):5.0f}m   p75 {_pct(t_mfe,.75):5.0f}m")
    early = sum(1 for r in rows if r["t_mfe"] <= 60) / n
    print(f"  MFE reached within 60 min: {100*early:4.0f}%")
    f30fracs = [r["f30frac"] for r in rows if r["mfe"] > 0]
    if f30fracs:
        print(f"  median share of MFE captured by +30 min: {100*_pct(f30fracs,.5):4.0f}%\n")

    # Debit-spread read: an ATM bull/bear debit spread of width W needs roughly a
    # full-W favorable move to approach max value; partial credit below that.
    print("Debit-spread implication (ATM long-premium, theta headwind on 0DTE):")
    for W in (10, 15, 20):
        reach = sum(1 for x in mfe if x >= W) / n
        half = sum(1 for x in mfe if x >= W / 2) / n
        print(f"  width {W:2d}: {100*reach:4.0f}% reach full-width move, "
              f"{100*half:4.0f}% reach half-width")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
