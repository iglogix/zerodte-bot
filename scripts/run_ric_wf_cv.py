#!/usr/bin/env python3
"""
Walk-forward cross-validation for the RIC production candidate.

Fold scheme mirrors src_v2/backtest/wf_optimizer.py: ratio-based splitting —
with ratio R and N folds, step = total_sessions // (R + N); fold k trains on
sessions[k*step : (R+k)*step] and validates OOS on the next step.

Grid: 3 scale-out tier schemes (real engine runs, single shared data pass)
x 6 VIX floors (offline day filter — exact, since the floor only skips days).
Selection per fold on the train window by annualised daily Sharpe (non-trade
days count as $0), min 5 train trades; the chosen combo is then scored on the
held-out validation window.

All runs: qty 4, natural pricing (pay ask / hit bid), slippage 0.

Usage:
    .venv/bin/python scripts/run_ric_wf_cv.py [--ratio 5] [--folds 4]
"""
from __future__ import annotations

import argparse
import bisect
import copy
import csv
import math
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_v2.backtest.cbbo_store import MergedCbboStore
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.reverse_ic import RicEngine


class OncePerDayStore(MergedCbboStore):
    def load_day(self, trading_date: date) -> int:
        if self.loaded_date == trading_date:
            return len(self._quotes)
        return super().load_day(trading_date)


TIER_SCHEMES = {
    "base ½@20/¼@30": [{"profit": 0.20, "close_frac": 0.50},
                       {"profit": 0.30, "close_frac": 0.25}],
    "half-runner ½@20": [{"profit": 0.20, "close_frac": 0.50}],
    "tiers ½@25/¼@40": [{"profit": 0.25, "close_frac": 0.50},
                        {"profit": 0.40, "close_frac": 0.25}],
}
VIX_FLOORS = [None, 14, 15, 16, 17, 18]


def load_vix(path="data/vix_cboe.csv"):
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out[date.fromisoformat(r["date"])] = float(r["close"])
        except (ValueError, KeyError):
            continue
    return out


def metrics(day_pnls: list[float], n_trades: int) -> dict:
    total = sum(day_pnls)
    n = len(day_pnls)
    mean = total / n if n else 0.0
    var = sum((p - mean) ** 2 for p in day_pnls) / max(n - 1, 1)
    std = math.sqrt(var)
    sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0
    wins = sum(1 for p in day_pnls if p > 0)
    losses = [p for p in day_pnls if p < 0]
    gross_w = sum(p for p in day_pnls if p > 0)
    gross_l = abs(sum(losses))
    return {
        "trades": n_trades,
        "total": total,
        "sharpe": sharpe,
        "pf": gross_w / gross_l if gross_l else float("inf"),
        "wr": wins / n_trades * 100 if n_trades else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="RIC walk-forward CV")
    ap.add_argument("--config", default="config/config_ric_spx.yaml")
    ap.add_argument("--ratio", type=int, default=5)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--min-trades", type=int, default=5)
    args = ap.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    base["strategy"]["pricing"] = "natural"
    base["strategy"]["qty"] = 4
    base["strategy"]["vix_floor"] = None
    base["options"]["slippage_per_spread"] = 0.0

    bt = base["backtest"]
    store = OncePerDayStore(bt["cbbo_cache_dir"], bt.get("cbbo_overlay_dir", ""))
    vix = load_vix()
    vdates = sorted(vix)

    def prior_vix(d):
        i = bisect.bisect_left(vdates, d)
        return vix[vdates[i - 1]] if i > 0 else None

    engines = {}
    for name, tiers in TIER_SCHEMES.items():
        cfg = copy.deepcopy(base)
        cfg["strategy"]["scale_out"] = {"enabled": True, "tiers": tiers}
        eng = RicEngine(cfg)
        eng.set_cbbo_store(store)
        engines[name] = eng

    loader = BarLoader(bt["data_file"], price_scale=bt.get("price_scale", 10.0),
                       symbol="SPX")
    days = [(day.session_date, day.bars)
            for day in loader.iter_days(
                start_date=date.fromisoformat(bt["start_date"]),
                end_date=date.fromisoformat(bt["end_date"]))]
    sessions = [d for d, _ in days]
    print(f"{len(days)} sessions, {len(engines)} engine variants "
          f"x {len(VIX_FLOORS)} floors = {len(engines) * len(VIX_FLOORS)} combos")

    # Single interleaved pass: pnl_by[(scheme)][session_date] = (pnl, prior_vix)
    prev = {name: (None, None, None) for name in engines}
    pnl_by = {name: {} for name in engines}
    for i, (d, bars) in enumerate(days):
        for name, eng in engines.items():
            pd, ps, pc = prev[name]
            res, settle = eng._process_day(d, bars, pd, ps, pc)
            if res.status == "closed":
                pnl_by[name][d] = res.realized_pnl_usd
            prev[name] = (d, settle, float(bars[-1]["close"]) if bars else None)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(days)} days")

    def window_metrics(scheme: str, floor, window: list[date]) -> dict:
        pnls, n_trades = [], 0
        for d in window:
            p = pnl_by[scheme].get(d)
            if p is not None and floor is not None:
                v = prior_vix(d)
                if v is None or v < floor:
                    p = None
            if p is None:
                pnls.append(0.0)
            else:
                pnls.append(p)
                n_trades += 1
        return metrics(pnls, n_trades)

    combos = [(s, f) for s in TIER_SCHEMES for f in VIX_FLOORS]

    # ---- Ratio-based folds (mirrors wf_optimizer._make_windows)
    step = len(sessions) // (args.ratio + args.folds)
    if step == 0:
        raise RuntimeError("not enough sessions for ratio/folds")

    print(f"\nWF-CV ratio={args.ratio}:1 folds={args.folds} "
          f"(step={step} sessions ≈ {step // 21} months)\n")

    oos_total = 0.0
    oos_trades = 0
    picks = []
    bench_total = 0.0          # fixed production candidate, no selection
    bench = ("base ½@20/¼@30", 15)

    for k in range(args.folds):
        train = sessions[k * step: (args.ratio + k) * step]
        val = sessions[(args.ratio + k) * step: (args.ratio + k + 1) * step]

        scored = []
        for s, f in combos:
            m = window_metrics(s, f, train)
            if m["trades"] < args.min_trades:
                continue
            scored.append(((s, f), m))
        scored.sort(key=lambda x: x[1]["sharpe"], reverse=True)
        (s_best, f_best), tm = scored[0]
        vm = window_metrics(s_best, f_best, val)
        bm = window_metrics(*bench, val)
        picks.append((s_best, f_best))
        oos_total += vm["total"]
        oos_trades += vm["trades"]
        bench_total += bm["total"]

        print(f"Fold {k + 1}: train {train[0]}..{train[-1]} ({len(train)}d)  "
              f"val {val[0]}..{val[-1]} ({len(val)}d)")
        print(f"  picked: {s_best} + VIX>={f_best}   "
              f"(train Sharpe {tm['sharpe']:.2f}, total ${tm['total']:+,.0f}, "
              f"n={tm['trades']})")
        print(f"  OOS:    n={vm['trades']:>3}  WR {vm['wr']:.1f}%  "
              f"PF {vm['pf']:.2f}  total ${vm['total']:+,.0f}  "
              f"Sharpe {vm['sharpe']:.2f}")
        print(f"  bench:  {bench[0]} + VIX>={bench[1]}: "
              f"n={bm['trades']:>3}  total ${bm['total']:+,.0f}  "
              f"Sharpe {bm['sharpe']:.2f}\n")

    print("=" * 70)
    print(f"OOS aggregate (selected per fold): {oos_trades} trades, "
          f"total ${oos_total:+,.0f}")
    print(f"OOS aggregate (fixed candidate {bench[0]} + VIX>={bench[1]}): "
          f"total ${bench_total:+,.0f}")
    from collections import Counter
    print("Selection stability:",
          ", ".join(f"{s} + VIX>={f}: {c}/{args.folds}"
                    for (s, f), c in Counter(picks).most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
