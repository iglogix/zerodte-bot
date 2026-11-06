"""
Compare three exit-management variants for the open IC strategy across the
full CBBO range (2022-04-18 → 2026-02-28). Same entry rules, three exits:

  A) Hybrid       : sl_mult=1.0, asymmetric exit (stop bad side, run good side to EOD)
  B) Strict TT    : sl_mult=1.0, close BOTH sides when one stops
  C) Vanilla TT   : no SL, no breach stop, just TP=25% of total credit
"""

from __future__ import annotations

import copy
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.orchestrator import BacktestOrchestrator
from zerodte.strategies.open_ic import OpenIcStrategy


VARIANTS = {
    "A_hybrid": {
        "options.sl_mult_of_credit": 1.0,
        "options.tp_mode": "total",
        "options.tp_pct_of_credit": 0.25,
        "options.hold_put_breach": False,
        "options.hold_call_breach": False,
        "options.close_remaining_on_first_stop": False,
    },
    "B_strict_TT": {
        "options.sl_mult_of_credit": 1.0,
        "options.tp_mode": "total",
        "options.tp_pct_of_credit": 0.25,
        "options.hold_put_breach": False,
        "options.hold_call_breach": False,
        "options.close_remaining_on_first_stop": True,
    },
    "C_vanilla_TT": {
        "options.sl_mult_of_credit": 0.0,           # no credit-mult SL
        "options.tp_mode": "total",
        "options.tp_pct_of_credit": 0.25,
        "options.hold_put_breach": True,            # let losers run
        "options.hold_call_breach": True,
        "options.close_remaining_on_first_stop": False,
    },
}


def _set_dotted(cfg, dotted, val):
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def _by_day_stats(results):
    by_day = defaultdict(lambda: {"pnl": 0.0, "legs": []})
    skips = Counter()
    for r in results:
        if r.status == "skipped":
            skips[r.skip_reason] += 1
            continue
        by_day[str(r.session_date)]["pnl"] += r.realized_pnl_usd or 0.0
        by_day[str(r.session_date)]["legs"].append(r.exit_reason)
    complete = [v for v in by_day.values() if len(v["legs"]) == 2]
    pnls = [v["pnl"] for v in complete]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total = sum(pnls)
    return {
        "days": len(complete),
        "wr_pct": (100 * len(wins) / len(complete)) if complete else 0.0,
        "total_pnl": total,
        "mean_day": (total / len(complete)) if complete else 0.0,
        "mean_win": (sum(wins) / len(wins)) if wins else 0.0,
        "mean_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "worst_day": min(pnls) if pnls else 0.0,
        "best_day": max(pnls) if pnls else 0.0,
        "skips": skips,
    }


def main():
    with open("config/config_spx_0dte_open_ic.yaml") as f:
        base_cfg = yaml.safe_load(f)

    bt = base_cfg.get("backtest", {})
    start = date(2022, 4, 18)
    end = date(2026, 2, 28)

    print(f"Preloading CBBO {start} → {end}…")
    mem = CbboStore.preload_cache(bt.get("cbbo_cache_dir", "data/cbbo_cache"),
                                  start=start, end=end)
    cbbo = CbboStore(bt.get("cbbo_cache_dir"), preloaded=mem)
    print(f"  {len(mem)} days")

    loader = BarLoader(bt.get("data_file"), price_scale=bt.get("price_scale", 10.0),
                       symbol=base_cfg["bot"]["symbol"])
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} trading days\n")

    results_by_variant = {}
    for name, overrides in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        for k, v in overrides.items():
            _set_dotted(cfg, k, v)
        print(f"=== {name} ===")
        for k, v in overrides.items():
            print(f"    {k:>50} = {v}")
        orch = BacktestOrchestrator(cfg, OpenIcStrategy, cbbo)
        res = orch.run(days)
        stats = _by_day_stats(res)
        results_by_variant[name] = stats
        print(f"    days={stats['days']:>3d}  WR={stats['wr_pct']:>5.1f}%  "
              f"total=${stats['total_pnl']:>+8.0f}  mean_day=${stats['mean_day']:>+6.0f}  "
              f"worst=${stats['worst_day']:>+6.0f}  best=${stats['best_day']:>+5.0f}  "
              f"avg_win=${stats['mean_win']:>+5.0f}  avg_loss=${stats['mean_loss']:>+5.0f}")
        print()

    print("─" * 80)
    print(f"{'Variant':<14} {'Days':>5} {'WR%':>7} {'Total':>10} {'Mean/day':>10} "
          f"{'Worst':>8} {'AvgWin':>8} {'AvgLoss':>8}")
    print("─" * 80)
    for name, s in results_by_variant.items():
        print(f"{name:<14} {s['days']:>5d} {s['wr_pct']:>6.1f}% "
              f"{s['total_pnl']:>+10.0f} {s['mean_day']:>+10.0f} "
              f"{s['worst_day']:>+8.0f} {s['mean_win']:>+8.0f} {s['mean_loss']:>+8.0f}")


if __name__ == "__main__":
    main()
