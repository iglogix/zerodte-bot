"""ORB directional — fade / mean-reversion comparison.

The excursion diagnostic (docs/orb_dir_debit_excursion_diag_2026-06-18.md) showed
the post-retest move round-trips (net EOD ~0, MAE~MFE). This asks the follow-up:
is there a FADE edge — does betting AGAINST the breakout beat betting with it?

Compares, full-span, same structure params (em0.5 / w20 / min_em20 / tp0.75 /
mtol3.0, both breach sides held), only the DIRECTIONAL choice differs:

  with_trend     breakout_level retest, trade WITH the breakout   (the survivor)
  fade_breakout  breakout_level retest, INVERT the spread side     (fade, same entries)
  opposite_level failed-breakout reversal (price returns to opposite ORB level)

Real CBBO pricing via BacktestOrchestrator. In-sample full-span first cut — if a
fade shows a real edge it then earns a WF-CV pass; if not, it's dead here.

Usage:
    .venv/bin/python -m scripts.orb_dir_fade_compare --start 2022-04-18 --end 2026-06-11
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.orchestrator import BacktestOrchestrator
from zerodte.strategies.orb_em_directional import OrbEmDirectionalStrategy

VARIANTS = {
    "with_trend (survivor)": {"retest_mode": "breakout_level", "fade_breakout": False},
    "fade_breakout":         {"retest_mode": "breakout_level", "fade_breakout": True},
    "opposite_level":        {"retest_mode": "opposite_level", "fade_breakout": False},
}


def _stats(pnls):
    n = len(pnls)
    if n == 0:
        return None
    total = sum(pnls)
    mean = total / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnls) / n) if n > 1 else 0.0
    wins = sum(1 for x in pnls if x > 0)
    return {"n": n, "total": total, "mean": mean, "sd": sd, "wr": wins / n,
            "sharpe_trade": (mean / sd) if sd else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="config/config_orb_dir_breakout_spx.yaml")
    ap.add_argument("--start", default="2022-04-18")
    ap.add_argument("--end", default="2026-06-11")
    args = ap.parse_args()

    with open(args.base) as f:
        base = yaml.safe_load(f)
    base.setdefault("backtest", {})["cbbo_cache_dir"] = "data/cbbo_cache"
    # Lock structure params to the survivor; symmetric breach-hold so the only
    # difference between variants is the directional choice.
    base["strategy"]["midline_tolerance_pts"] = 3.0
    base["options"].update({
        "em_with_trend": 0.5, "width_with_trend": 20, "tp_pct_of_credit": 0.75,
        "min_em_pts": 20, "hold_put_breach": True, "hold_call_breach": True,
    })
    data_file = base["backtest"].get("data_file", "data/spy_1m.csv")
    price_scale = base["backtest"].get("price_scale", 10.0)
    symbol = base.get("bot", {}).get("symbol", "SPX")

    print(f"Preloading CBBO {args.start}..{args.end}...", flush=True)
    mem = CbboStore.preload_cache(
        "data/cbbo_cache", start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
    cbbo = CbboStore("data/cbbo_cache", preloaded=mem)
    loader = BarLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(
        start_date=date.fromisoformat(args.start), end_date=date.fromisoformat(args.end),
    ))
    print(f"  {len(days)} trading days\n", flush=True)

    n_years = max((date.fromisoformat(args.end) - date.fromisoformat(args.start)).days / 365.25, 1e-9)

    for name, ov in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg["strategy"]["retest_mode"] = ov["retest_mode"]
        cfg["strategy"]["fade_breakout"] = ov["fade_breakout"]
        orch = BacktestOrchestrator(cfg, OrbEmDirectionalStrategy, cbbo)
        results = orch.run(days)
        closed = [r for r in results if r.status == "closed" and r.realized_pnl_usd is not None]
        pnls = [r.realized_pnl_usd for r in closed]
        s = _stats(pnls)
        print(f"=== {name} ===")
        if not s:
            print("  no trades\n")
            continue
        ann_sharpe = s["sharpe_trade"] * math.sqrt(s["n"] / n_years)
        print(f"  trades {s['n']:4d}   total ${s['total']:>10,.0f}   "
              f"$/trade {s['mean']:>7.1f}   win% {100*s['wr']:4.0f}%   "
              f"Sharpe/trade {s['sharpe_trade']:5.2f}   ann ~{ann_sharpe:5.2f}")
        # by year
        by_yr = defaultdict(list)
        for r in closed:
            by_yr[r.session_date.year].append(r.realized_pnl_usd)
        parts = []
        for y in sorted(by_yr):
            ys = _stats(by_yr[y])
            parts.append(f"{y}:${ys['total']:>7,.0f}({ys['n']})")
        print("  by year  " + "  ".join(parts) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
