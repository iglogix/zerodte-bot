"""Pre-simulate ORB EM iron-condor (orb_em_ic) parameter combos for the
three-tier walk-forward METHOD validation (docs/wf-cv.md).

Day-independence shortcut: simulate each combo ONCE over the full span; the WF
runner slices folds from these logs (non-compounding / static sizing).

Writes one CSV per combo under results/orb_em_ic_wf/ in the format
run_wf_method_strategy.py consumes: columns (status, session_date,
realized_pnl_usd). For 0DTE, session_date == trade date == exit date.

Usage:
    .venv/bin/python -m scripts.gen_orb_ic_combo_logs \
        --start 2022-04-18 --end 2026-06-11 --workers 6
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import logging
import os
import sys
import time

logging.disable(logging.WARNING)  # silence per-trade [STOP-LOSS]/[EOD] spam
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.orchestrator import BacktestOrchestrator
from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy

OUT = Path("results/orb_em_ic_wf")

# Module-level handles for worker processes (populated by _init_worker)
_DAYS = None
_CBBO = None
_BASE_CFG = None


def _tag(c: dict) -> str:
    return (
        f"ew{c['em_with_trend']:g}_ea{c['em_against_trend']:g}"
        f"_ww{c['width_with_trend']:g}_wa{c['width_against_trend']:g}"
        f"_tp{c['tp_pct_of_credit']:g}_mt{c['midline_tolerance_pts']:g}"
    )


def _run_combo(combo: dict) -> tuple[str, int, float]:
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["options"]["em_with_trend"] = combo["em_with_trend"]
    cfg["options"]["em_against_trend"] = combo["em_against_trend"]
    cfg["options"]["width_with_trend"] = combo["width_with_trend"]
    cfg["options"]["width_against_trend"] = combo["width_against_trend"]
    cfg["options"]["tp_pct_of_credit"] = combo["tp_pct_of_credit"]
    cfg["strategy"]["midline_tolerance_pts"] = combo["midline_tolerance_pts"]

    orch = BacktestOrchestrator(cfg, OrbEmIronCondorStrategy, _CBBO)
    results = orch.run(_DAYS)

    path = OUT / f"combo_{_tag(combo)}.csv"
    n = 0
    net = 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["status", "session_date", "realized_pnl_usd"])
        for r in results:
            if r.status != "closed" or r.realized_pnl_usd is None:
                continue
            w.writerow(["closed", str(r.session_date), round(r.realized_pnl_usd, 2)])
            n += 1
            net += r.realized_pnl_usd
    return _tag(combo), n, net


def _init_worker(days, cbbo, base_cfg):
    global _DAYS, _CBBO, _BASE_CFG
    _DAYS, _CBBO, _BASE_CFG = days, cbbo, base_cfg


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_orb_em_ic_spx.yaml")
    ap.add_argument("--start", default="2022-04-18")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--em-with", default="0.5,1.0")
    ap.add_argument("--em-against", default="1.5,2.0,2.5")
    ap.add_argument("--w-with", default="10,15,20")
    ap.add_argument("--w-against", default="10,15")
    ap.add_argument("--tp", default="0.5,0.75")
    ap.add_argument("--mtol", default="0.25,1.0")
    ap.add_argument("--limit-days", type=int, default=0, help="probe: cap number of days")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg.setdefault("backtest", {})["cbbo_cache_dir"] = "data/cbbo_cache"
    data_file = base_cfg["backtest"].get("data_file", "data/spy_1m.csv")
    price_scale = base_cfg["backtest"].get("price_scale", 10.0)
    symbol = base_cfg.get("bot", {}).get("symbol", "SPX")

    print(f"Preloading CBBO cache for {args.start}..{args.end} into RAM...", flush=True)
    mem = CbboStore.preload_cache(
        "data/cbbo_cache",
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
    cbbo = CbboStore("data/cbbo_cache", preloaded=mem)
    print(f"  {len(mem)} days in cache", flush=True)

    loader = BarLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
    ))
    if args.limit_days:
        days = days[: args.limit_days]
    print(f"  {len(days)} trading days {args.start}..{args.end}", flush=True)

    def _floats(s):
        return [float(x) for x in s.split(",")]

    grid = [
        {
            "em_with_trend": ew, "em_against_trend": ea,
            "width_with_trend": ww, "width_against_trend": wa,
            "tp_pct_of_credit": tp, "midline_tolerance_pts": mt,
        }
        for ew, ea, ww, wa, tp, mt in itertools.product(
            _floats(args.em_with), _floats(args.em_against),
            _floats(args.w_with), _floats(args.w_against),
            _floats(args.tp), _floats(args.mtol),
        )
        if ea >= ew  # against-trend leg must be >= with-trend distance
    ]
    # as-taught baseline (pre-optimization config params) — the WF method's
    # static reference: "trade these constants forever" vs periodic re-opt.
    baseline = {
        "em_with_trend": 0.5, "em_against_trend": 0.75,
        "width_with_trend": 5, "width_against_trend": 10,
        "tp_pct_of_credit": 0.5, "midline_tolerance_pts": 0.25,
    }
    if baseline not in grid:
        grid.append(baseline)
    print(f"baseline combo tag: {_tag(baseline)}", flush=True)
    # skip already-generated combos
    todo = [c for c in grid if not (OUT / f"combo_{_tag(c)}.csv").exists()]
    print(f"{len(grid)} combos ({len(todo)} to run), workers={args.workers}", flush=True)

    t0 = time.time()
    if args.workers <= 1:
        _init_worker(days, cbbo, base_cfg)
        for i, c in enumerate(todo, 1):
            tag, n, net = _run_combo(c)
            el = (time.time() - t0) / 60
            print(f"[{i}/{len(todo)}] {tag} n={n} net=${net:,.0f} ({el:.1f}m)", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")  # share preloaded CBBO via copy-on-write
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(days, cbbo, base_cfg)) as pool:
            for i, (tag, n, net) in enumerate(pool.imap_unordered(_run_combo, todo), 1):
                el = (time.time() - t0) / 60
                print(f"[{i}/{len(todo)}] {tag} n={n} net=${net:,.0f} ({el:.1f}m)", flush=True)

    print(f"\nDone. {len(grid)} combos in {(time.time()-t0)/60:.1f}m -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
