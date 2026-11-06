"""
Run both the SL-based WF-CV winner and the vanilla-TT no-SL variant on the
same days. Compare per-day PnL to see whether the "good days" overlap.
Then report three combination policies:

  P1) Parallel portfolio       -- both trades placed every day, PnLs summed (2× capital).
  P2) Best-of-day (look-ahead) -- theoretical upper bound: pick whichever was better that day.
  P3) Either-fired             -- if only one config fires (after its gates), take that one;
                                  if both fire, average (proxy for "I don't know which is best").
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.orchestrator import BacktestOrchestrator
from zerodte.strategies.open_ic import OpenIcStrategy


VARIANTS = {
    "A_sl2x": {     # WF-CV winner — current config
        "options.target_short_delta": 0.25,
        "options.wing_width": 25,
        "options.tp_pct_of_credit": 0.25,
        "options.tp_mode": "total",
        "options.sl_mult_of_credit": 2.0,
        "options.hold_put_breach": True,
        "options.hold_call_breach": True,
        "entry.opening_range_filter.enabled": True,
        "entry.opening_range_filter.max_open_range_em_ratio": 0.5,
        "strategy.skip_fomc_next_day": True,
        "strategy.skip_weekdays": ["Fri"],
    },
    "B_vanilla_tt": {     # canonical TT — no SL
        "options.target_short_delta": 0.25,
        "options.wing_width": 20,
        "options.tp_pct_of_credit": 0.75,   # canonical TT "25% TP"
        "options.tp_mode": "total",
        "options.sl_mult_of_credit": 0.0,
        "options.hold_put_breach": True,
        "options.hold_call_breach": True,
        # vanilla TT keeps fewer gates — only calendar skips, no opening-range etc.
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


def main():
    with open("config/config_spx_0dte_open_ic.yaml") as f:
        base = yaml.safe_load(f)
    start, end = date(2022, 4, 18), date(2026, 2, 28)

    print(f"Preloading CBBO {start} → {end}…")
    mem = CbboStore.preload_cache("data/cbbo_cache", start=start, end=end)
    cbbo = CbboStore("data/cbbo_cache", preloaded=mem)
    loader = BarLoader(base["backtest"]["data_file"], price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} bar-days\n")

    results = {}
    for name, overrides in VARIANTS.items():
        cfg = copy.deepcopy(base)
        for k, v in overrides.items():
            _set_dotted(cfg, k, v)
        print(f"Running {name}…")
        results[name] = _per_day_pnl(cfg, days, cbbo)

    a = results["A_sl2x"]
    b = results["B_vanilla_tt"]

    all_days = sorted(set(a) | set(b))
    only_a = [d for d in all_days if d in a and d not in b]
    only_b = [d for d in all_days if d in b and d not in a]
    both = [d for d in all_days if d in a and d in b]

    sep = "─" * 90
    print()
    print(sep)
    print(f"  Coverage (days with full IC trade)")
    print(sep)
    print(f"  A_sl2x only:    {len(only_a):>4d} days  total: ${sum(a[d] for d in only_a):+,.0f}")
    print(f"  B_vanilla only: {len(only_b):>4d} days  total: ${sum(b[d] for d in only_b):+,.0f}")
    print(f"  Both fired:     {len(both):>4d} days  A=${sum(a[d] for d in both):+,.0f}  "
          f"B=${sum(b[d] for d in both):+,.0f}")
    print()

    # Quadrant analysis on "both fired" days
    aw_bw = [d for d in both if a[d] > 0 and b[d] > 0]
    aw_bl = [d for d in both if a[d] > 0 and b[d] < 0]
    al_bw = [d for d in both if a[d] < 0 and b[d] > 0]
    al_bl = [d for d in both if a[d] < 0 and b[d] < 0]
    print(sep)
    print(f"  On days BOTH fired ({len(both)}):")
    print(sep)
    print(f"  Both win:    {len(aw_bw):>4d}  A=${sum(a[d] for d in aw_bw):+,.0f}  B=${sum(b[d] for d in aw_bw):+,.0f}")
    print(f"  A win, B loss:{len(aw_bl):>4d}  A=${sum(a[d] for d in aw_bl):+,.0f}  B=${sum(b[d] for d in aw_bl):+,.0f}")
    print(f"  A loss, B win:{len(al_bw):>4d}  A=${sum(a[d] for d in al_bw):+,.0f}  B=${sum(b[d] for d in al_bw):+,.0f}")
    print(f"  Both loss:   {len(al_bl):>4d}  A=${sum(a[d] for d in al_bl):+,.0f}  B=${sum(b[d] for d in al_bl):+,.0f}")
    print()

    # Combination policies
    p1_parallel = sum(a.values()) + sum(b.values())
    p2_bestof = sum(a[d] for d in only_a) + sum(b[d] for d in only_b) + sum(max(a[d], b[d]) for d in both)
    p3_either = sum(a[d] for d in only_a) + sum(b[d] for d in only_b) + sum((a[d] + b[d]) / 2 for d in both)

    print(sep)
    print(f"  Combination policies")
    print(sep)
    print(f"  Standalone A_sl2x:                ${sum(a.values()):+,.0f}  ({len(a)} days)")
    print(f"  Standalone B_vanilla_tt:          ${sum(b.values()):+,.0f}  ({len(b)} days)")
    print(f"  P1 Parallel (sum of both):        ${p1_parallel:+,.0f}  (uses 2x capital)")
    print(f"  P2 Best-of-day (look-ahead, max): ${p2_bestof:+,.0f}  (theoretical upper bound)")
    print(f"  P3 Either-fired (avg on overlap): ${p3_either:+,.0f}  (achievable without look-ahead)")
    print()

    # Correlation between the two
    if both:
        import statistics
        a_vals = [a[d] for d in both]
        b_vals = [b[d] for d in both]
        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        cov = sum((a_vals[i] - mean_a) * (b_vals[i] - mean_b) for i in range(len(both))) / len(both)
        std_a = statistics.pstdev(a_vals)
        std_b = statistics.pstdev(b_vals)
        corr = cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0
        print(f"  Per-day PnL correlation A↔B (on overlap): {corr:+.3f}")


if __name__ == "__main__":
    main()
