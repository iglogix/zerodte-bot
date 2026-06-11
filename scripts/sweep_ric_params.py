#!/usr/bin/env python3
"""
RIC structural-parameter sweep, cut by VIX regime.

Runs one-at-a-time variations around the course spec (long deltas, wing
offsets, scale-out tiers) in a SINGLE pass over the CBBO cache: all engine
variants share one store and each day's pickle is loaded once, so 16 variants
cost roughly one normal backtest run.

All variants: scale-out qty 4, natural pricing, slip 0, NO vix floor — the
report cuts every variant by VIX band offline so the param response per
regime (incl. <15) is visible.

Usage:
    .venv/bin/python scripts/sweep_ric_params.py [--out results/ric_spx/param_sweep.csv]
"""
from __future__ import annotations

import argparse
import bisect
import copy
import csv
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_v2.backtest.cbbo_store import MergedCbboStore
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.reverse_ic import RicEngine


class OncePerDayStore(MergedCbboStore):
    """Skip the pickle re-read when the requested day is already loaded."""

    def load_day(self, trading_date: date) -> int:
        if self.loaded_date == trading_date:
            return len(self._quotes)
        return super().load_day(trading_date)


def tiers(*pairs):
    return [{"profit": p, "close_frac": f} for p, f in pairs]


VARIANTS = [
    ("base 50/35 20/35 t20/30", {}),
    ("lc_delta 0.40", {"long_call_delta": 0.40}),
    ("lc_delta 0.45", {"long_call_delta": 0.45}),
    ("lc_delta 0.55", {"long_call_delta": 0.55}),
    ("lp_delta 0.25", {"long_put_delta": 0.25}),
    ("lp_delta 0.30", {"long_put_delta": 0.30}),
    ("lp_delta 0.40", {"long_put_delta": 0.40}),
    ("call_wing 15", {"call_wing_offset_pts": 15}),
    ("call_wing 25", {"call_wing_offset_pts": 25}),
    ("call_wing 30", {"call_wing_offset_pts": 30}),
    ("put_wing 25", {"put_wing_offset_pts": 25}),
    ("put_wing 30", {"put_wing_offset_pts": 30}),
    ("put_wing 45", {"put_wing_offset_pts": 45}),
    ("tiers 15/25", {"scale_out": {"enabled": True, "tiers": tiers((0.15, 0.50), (0.25, 0.25))}}),
    ("tiers 25/40", {"scale_out": {"enabled": True, "tiers": tiers((0.25, 0.50), (0.40, 0.25))}}),
    ("tiers half@20 only", {"scale_out": {"enabled": True, "tiers": tiers((0.20, 0.50))}}),
]

VIX_BANDS = [(0, 15, "<15"), (15, 18, "15-18"), (18, 22, "18-22"), (22, 99, ">=22")]


def load_vix(path="data/vix_cboe.csv"):
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out[date.fromisoformat(r["date"])] = float(r["close"])
        except (ValueError, KeyError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RIC structural param sweep by VIX band")
    ap.add_argument("--config", default="config/config_ric_spx.yaml")
    ap.add_argument("--out", default="results/ric_spx/param_sweep.csv")
    args = ap.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    base["strategy"]["pricing"] = "natural"
    base["strategy"]["qty"] = 4
    base["strategy"].setdefault("scale_out", {})["enabled"] = True
    base["strategy"]["vix_floor"] = None
    base["options"]["slippage_per_spread"] = 0.0

    bt = base["backtest"]
    store = OncePerDayStore(bt["cbbo_cache_dir"], bt.get("cbbo_overlay_dir", ""))
    vix = load_vix()
    vdates = sorted(vix)

    def prior_vix(d):
        i = bisect.bisect_left(vdates, d)
        return vix[vdates[i - 1]] if i > 0 else None

    engines = []
    for name, overrides in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["strategy"].update(copy.deepcopy(overrides))
        eng = RicEngine(cfg)
        eng.set_cbbo_store(store)
        engines.append((name, eng))

    loader = BarLoader(bt["data_file"], price_scale=bt.get("price_scale", 10.0),
                       symbol="SPX")
    days = [(day.session_date, day.bars)
            for day in loader.iter_days(
                start_date=date.fromisoformat(bt["start_date"]),
                end_date=date.fromisoformat(bt["end_date"]))]
    print(f"{len(days)} sessions x {len(engines)} variants (single data pass)")

    # Interleave variants per day so each pickle loads once. Per-engine prior
    # state mirrors RicEngine.run().
    prev = {name: (None, None, None) for name, _ in engines}
    results = {name: [] for name, _ in engines}
    for i, (d, bars) in enumerate(days):
        for name, eng in engines:
            pd, ps, pc = prev[name]
            res, settle = eng._process_day(d, bars, pd, ps, pc)
            results[name].append(res)
            prev[name] = (d, settle, float(bars[-1]["close"]) if bars else None)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(days)} days")

    # ---- Report: per variant x VIX band
    def stats(pnls):
        if not pnls:
            return None
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p <= 0))
        return {
            "n": len(pnls),
            "wr": len([p for p in pnls if p > 0]) / len(pnls) * 100,
            "pf": (gross_w / gross_l) if gross_l else float("inf"),
            "total": sum(pnls),
            "avg": sum(pnls) / len(pnls),
        }

    rows = []
    hdr = f"{'variant':<24} | {'band':>6} | {'n':>4} {'WR%':>6} {'PF':>6} {'total':>11} {'avg':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, _ in engines:
        closed = [(r, prior_vix(r.session_date)) for r in results[name]
                  if r.status == "closed"]
        for lo, hi, label in [(0, 99, "all")] + VIX_BANDS:
            pnls = [r.realized_pnl_usd for r, v in closed
                    if v is not None and lo <= v < hi]
            s = stats(pnls)
            if not s:
                continue
            rows.append({"variant": name, "vix_band": label, **s})
            print(f"{name:<24} | {label:>6} | {s['n']:>4} {s['wr']:>6.1f} "
                  f"{s['pf']:>6.2f} {s['total']:>+11,.0f} {s['avg']:>+7,.0f}")
        print("-" * len(hdr))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSweep table written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
