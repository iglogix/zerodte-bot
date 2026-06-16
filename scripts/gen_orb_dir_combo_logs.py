"""Pre-simulate ORB EM DIRECTIONAL (orb_em_directional) combos for the three-tier
walk-forward METHOD validation (docs/wf-cv.md). Single with-trend credit spread
(PCS on long breakout / CCS on short) after ORB retest.

This re-runs the old src_8am "directional worst-case (0.20Δ, 15pt)" optimizer
winner — which was BS-priced fiction — on REAL CBBO quotes over the full span.
Strike placement is em_mult (model-free); em_mult≈0.5 lands near 0.20Δ (verified).

Simulate-once-full-span; WF runner slices folds from the logs. Columns:
(status, session_date, realized_pnl_usd). 0DTE => session_date == exit date.

Usage:
    .venv/bin/python -m scripts.gen_orb_dir_combo_logs \
        --start 2022-04-18 --end 2026-06-11 --out results/orb_em_dir_wf --workers 1
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import logging
import sys
import time
from datetime import date
from pathlib import Path

import yaml

logging.disable(logging.WARNING)  # silence per-trade [STOP-LOSS]/[EOD] spam
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.strategies.orb_em_directional import OrbEmDirectionalStrategy

OUT = Path("results/orb_em_dir_wf")
_DAYS = None
_CBBO = None
_BASE_CFG = None


def _tag(c: dict) -> str:
    return (
        f"ew{c['em_with_trend']:g}_w{c['width_with_trend']:g}_tp{c['tp_pct_of_credit']:g}"
        f"_me{c['min_em_pts']:g}_hpb{int(c['hold_put_breach'])}_{c['retest_mode']}"
    )


def _run_combo(combo: dict) -> tuple[str, int, float]:
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["options"]["em_with_trend"] = combo["em_with_trend"]
    cfg["options"]["width_with_trend"] = combo["width_with_trend"]
    cfg["options"]["tp_pct_of_credit"] = combo["tp_pct_of_credit"]
    cfg["options"]["min_em_pts"] = combo["min_em_pts"]
    cfg["options"]["hold_put_breach"] = combo["hold_put_breach"]
    cfg["strategy"]["retest_mode"] = combo["retest_mode"]
    cfg["strategy"]["midline_tolerance_pts"] = combo["midline_tolerance_pts"]

    orch = BacktestOrchestrator(cfg, OrbEmDirectionalStrategy, _CBBO)
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
    ap.add_argument("--config", default="config/config_orb_em_directional_spx.yaml")
    ap.add_argument("--start", default="2022-04-18")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--em", default="0.4,0.5,0.6,0.75")
    ap.add_argument("--width", default="10,15,20")
    ap.add_argument("--tp", default="0.5,0.75")
    ap.add_argument("--min-em", default="0,20")
    ap.add_argument("--hold-breach", default="0,1")
    ap.add_argument("--retest", default="midline,breakout_level")
    ap.add_argument("--mtol", type=float, default=1.0)
    ap.add_argument("--limit-days", type=int, default=0)
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
            "em_with_trend": ew, "width_with_trend": w, "tp_pct_of_credit": tp,
            "min_em_pts": me, "hold_put_breach": bool(int(hpb)), "retest_mode": rt,
            "midline_tolerance_pts": args.mtol,
        }
        for ew, w, tp, me, hpb, rt in itertools.product(
            _floats(args.em), _floats(args.width), _floats(args.tp),
            _floats(args.min_em), args.hold_breach.split(","), args.retest.split(","),
        )
    ]
    todo = [c for c in grid if not (OUT / f"combo_{_tag(c)}.csv").exists()]
    print(f"{len(grid)} combos ({len(todo)} to run), workers={args.workers}", flush=True)

    t0 = time.time()
    if args.workers <= 1:
        _init_worker(days, cbbo, base_cfg)
        for i, c in enumerate(todo, 1):
            tag, n, net = _run_combo(c)
            print(f"[{i}/{len(todo)}] {tag} n={n} net=${net:,.0f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(days, cbbo, base_cfg)) as pool:
            for i, (tag, n, net) in enumerate(pool.imap_unordered(_run_combo, todo), 1):
                print(f"[{i}/{len(todo)}] {tag} n={n} net=${net:,.0f} ({(time.time()-t0)/60:.1f}m)", flush=True)

    print(f"\nDone. {len(grid)} combos in {(time.time()-t0)/60:.1f}m -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
