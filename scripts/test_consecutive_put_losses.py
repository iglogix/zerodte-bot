"""
Test the tastytrade claim: "there were never two consecutive losing put credit
spreads at 25 delta."

Run a put-only-equivalent backtest with ALL filters off, every day:
  - strike_mode=delta, target_short_delta swept
  - tp_mode=per_side, sl_scope=per_side  -> put side trades independently
  - No NFP/FOMC/CPI/Fri/opening-range/stability gates
  - Just enter at 09:35 and let put side play out

For each target delta, find the maximum run of consecutive losing put spreads.
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import SpyLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.strategies.open_ic import OpenIcStrategy


def _set_dotted(cfg, k, v):
    parts = k.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v


def _run(base, days, cbbo, em_mult: float, sl_mult: float = 2.0, tp_pct: float = 0.50):
    """em_mult strike picker (model-free) — every CBBO-valid day produces a trade,
    no BS-inversion selection bias."""
    cfg = copy.deepcopy(base)
    _set_dotted(cfg, "options.strike_mode", "em_mult")
    _set_dotted(cfg, "options.em_mult", em_mult)
    _set_dotted(cfg, "options.wing_width", 25)
    # Per-side exits (put runs independently of call)
    _set_dotted(cfg, "options.tp_mode", "per_side")
    _set_dotted(cfg, "options.sl_scope", "per_side")
    _set_dotted(cfg, "options.tp_pct_of_credit", tp_pct)
    _set_dotted(cfg, "options.sl_mult_of_credit", sl_mult)
    _set_dotted(cfg, "options.hold_put_breach", True)
    _set_dotted(cfg, "options.hold_call_breach", True)
    # ALL FILTERS OFF
    _set_dotted(cfg, "entry.entry_mode", "fixed_time")
    _set_dotted(cfg, "entry.entry_time", "09:35")
    _set_dotted(cfg, "entry.stability_check", False)
    _set_dotted(cfg, "entry.opening_range_filter.enabled", False)
    _set_dotted(cfg, "regime.enabled", False)
    _set_dotted(cfg, "strategy.skip_nfp_fridays", False)
    _set_dotted(cfg, "strategy.skip_fomc_days", False)
    _set_dotted(cfg, "strategy.skip_cpi_days", False)
    _set_dotted(cfg, "strategy.skip_fomc_next_day", False)
    _set_dotted(cfg, "strategy.skip_cpi_next_day", False)
    _set_dotted(cfg, "strategy.skip_weekdays", [])
    _set_dotted(cfg, "options.min_credit_per_spread", 0.01)
    _set_dotted(cfg, "options.max_credit_imbalance", 99.0)

    orch = BacktestOrchestrator(cfg, OpenIcStrategy, cbbo)
    return orch.run(days)


def _analyze_put_streaks(results):
    """Extract chronologically-ordered put-side PnL and find max consecutive-loss streak."""
    by_day = defaultdict(dict)
    for r in results:
        if r.status != "closed":
            continue
        if r.spread_type == "put_credit_spread":
            by_day[r.session_date]["put_pnl"] = r.realized_pnl_usd or 0.0
        elif r.spread_type == "call_credit_spread":
            by_day[r.session_date]["call_pnl"] = r.realized_pnl_usd or 0.0
    days_sorted = sorted(by_day.keys())
    put_series = [(d, by_day[d].get("put_pnl", 0.0)) for d in days_sorted
                  if "put_pnl" in by_day[d]]
    losers = [(d, p) for (d, p) in put_series if p < 0]
    # streaks
    current = 0
    max_streak = 0
    max_run = []
    cur_run = []
    for (d, p) in put_series:
        if p < 0:
            current += 1
            cur_run.append((d, p))
            if current > max_streak:
                max_streak = current
                max_run = list(cur_run)
        else:
            current = 0
            cur_run = []
    return {
        "n_trades": len(put_series),
        "n_winners": sum(1 for (_, p) in put_series if p > 0),
        "n_losers": len(losers),
        "wr_pct": 100 * sum(1 for (_, p) in put_series if p > 0) / max(1, len(put_series)),
        "total_pnl": sum(p for (_, p) in put_series),
        "max_consecutive_losses": max_streak,
        "max_loss_run": max_run[:10],
        "all_losers": losers,
    }


def main():
    with open("config/config_spx_0dte_open_ic.yaml") as f:
        base = yaml.safe_load(f)
    start, end = date(2022, 4, 18), date(2026, 2, 28)
    print(f"Preloading CBBO {start} → {end}…")
    mem = CbboStore.preload_cache("data/cbbo_cache", start=start, end=end)
    cbbo = CbboStore("data/cbbo_cache", preloaded=mem)
    loader = SpyLoader(base["backtest"]["data_file"], price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} bar-days\n")

    # em_mult values approx. corresponding to deltas (rough heuristic):
    #   em_mult=0.20 ≈ 0.30 delta, 0.30 ≈ 0.22 delta, 0.40 ≈ 0.16 delta, 0.50 ≈ 0.12 delta
    em_mults = [0.10, 0.20, 0.30, 0.50, 0.80, 1.0]
    print("WITH TT-style SL (2x credit) — original setup")
    print(f"{'em_mult':>8}  {'n':>5}  {'WR%':>6}  {'total':>10}  {'max_L':>6}  {'cred':>6}")
    print("─" * 65)
    for em in em_mults:
        results = _run(base, days, cbbo, em, sl_mult=2.0)
        stats = _analyze_put_streaks(results)
        credits = [r.credit_received for r in results
                   if r.status == "closed" and r.spread_type == "put_credit_spread"
                   and r.credit_received]
        ac = sum(credits) / len(credits) if credits else 0
        print(f"{em:>8.2f}  {stats['n_trades']:>5d}  "
              f"{stats['wr_pct']:>5.1f}%  ${stats['total_pnl']:>+8,.0f}  "
              f"{stats['max_consecutive_losses']:>6d}  ${ac:>4.2f}")

    print("\nNO SL — hold to expiry / TP only")
    print(f"{'em_mult':>8}  {'n':>5}  {'WR%':>6}  {'total':>10}  {'max_L':>6}  {'cred':>6}")
    print("─" * 65)
    for em in em_mults:
        results = _run(base, days, cbbo, em, sl_mult=0.0)
        stats = _analyze_put_streaks(results)
        credits = [r.credit_received for r in results
                   if r.status == "closed" and r.spread_type == "put_credit_spread"
                   and r.credit_received]
        ac = sum(credits) / len(credits) if credits else 0
        print(f"{em:>8.2f}  {stats['n_trades']:>5d}  "
              f"{stats['wr_pct']:>5.1f}%  ${stats['total_pnl']:>+8,.0f}  "
              f"{stats['max_consecutive_losses']:>6d}  ${ac:>4.2f}")

    # Detail on em_mult=0.30 (closest to 25-delta) — no SL version
    print("\n" + "=" * 70)
    print("Detail: em_mult=0.30, NO SL — longest losing streak")
    print("=" * 70)
    results = _run(base, days, cbbo, 0.30, sl_mult=0.0)
    stats = _analyze_put_streaks(results)
    print(f"Max consecutive losses: {stats['max_consecutive_losses']}")
    if stats["max_loss_run"]:
        print("Dates and PnL in longest streak:")
        for d, p in stats["max_loss_run"]:
            print(f"  {d}: ${p:+.0f}")


if __name__ == "__main__":
    main()
