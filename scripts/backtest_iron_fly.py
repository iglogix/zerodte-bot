#!/usr/bin/env python3
"""
Backtest: 0DTE Iron Fly on SPX.

Entry at 10:00 AM ET with ATM short strikes, EM-based wings.
Uses real CBBO quotes for pricing, SPY×10 bars for underlying.

Usage:
    python scripts/backtest_iron_fly.py
    python scripts/backtest_iron_fly.py --config config/config_iron_fly_spx.yaml
    python scripts/backtest_iron_fly.py --optimize
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_8am.backtest.cbbo_store import CbboStore
from src_8am.backtest.iron_fly_engine import IronFlyEngine, IronFlyResult
from src_8am.backtest.market_data import MarketDataLoader
from src_8am.backtest.bar_loader import BarLoader
from src_8am.backtest.strategy_iron_fly import IronFlyStrategy

# Lazy import for WF-CV (avoid circular / heavy import on simple runs)
# from src_8am.backtest.iron_fly_optimizer import IronFlyOptimizer


def load_data(config: dict):
    """Load CBBO cache, SPY bars, and VIX IV data."""
    bt = config.get("backtest", {})

    # CBBO quotes
    cbbo_dir = bt.get("cbbo_cache_dir", "data/cbbo_cache")
    print(f"Loading CBBO cache from {cbbo_dir}...")
    mem = CbboStore.preload_cache(cbbo_dir)
    cbbo_dates = set(mem.keys())
    print(f"  {len(cbbo_dates)} days loaded")

    # SPY bars (scaled ×10 → SPX)
    data_file = bt.get("data_file", "data/spy_1m.csv")
    price_scale = bt.get("price_scale", 10.0)
    start = date.fromisoformat(bt.get("start_date", "2025-09-22"))
    end = date.fromisoformat(bt.get("end_date", "2026-02-26"))

    print(f"Loading SPY bars from {data_file} (scale={price_scale})...")
    loader = BarLoader(data_file, price_scale=price_scale, symbol="SPX")
    days_data = []
    for day in loader.iter_days(start_date=start, end_date=end):
        if day.session_date in cbbo_dates:
            days_data.append((day.session_date, day.bars))
    print(f"  {len(days_data)} days with both bars and CBBO")

    # VIX IV
    data_dir = bt.get("data_dir", "data")
    md = MarketDataLoader(data_dir=data_dir)
    assumed_iv = config.get("options", {}).get("assumed_iv", 0.15)
    iv_by_date = {}
    for d, _ in days_data:
        iv_by_date[d] = md.get_daily_iv(d, fallback=assumed_iv)

    return mem, days_data, iv_by_date


def run_single(config: dict, mem: dict, days_data, iv_by_date) -> list[IronFlyResult]:
    """Run a single backtest with given config."""
    strategy = IronFlyStrategy()
    strategy.configure(config)

    engine = IronFlyEngine(strategy, config)
    cbbo = CbboStore("", preloaded=mem)
    engine.set_cbbo_store(cbbo)

    return engine.run(days_data, iv_by_date)


def print_report(results: list[IronFlyResult], title: str = "Iron Fly Backtest"):
    """Print summary report."""
    closed = [r for r in results if r.status == "closed"]
    skipped = [r for r in results if r.status == "skipped"]

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Total days:       {len(results)}")
    print(f"  Trades placed:    {len(closed)}")
    print(f"  Skipped days:     {len(skipped)}")

    if not closed:
        print("  No trades to report.")
        print(sep)
        _print_skips(skipped)
        return

    pnls = [r.realized_pnl_usd for r in closed]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    avg_pnl = total_pnl / len(pnls)
    win_rate = len(winners) / len(pnls) * 100

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    peak = equity = max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    # Sharpe
    import math
    n = len(pnls)
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / max(n - 1, 1)
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0

    print(f"  Win rate:         {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Total P&L:        ${total_pnl:+,.0f}")
    print(f"  Avg P&L/trade:    ${avg_pnl:+,.0f}")
    print(f"  Avg winner:       ${sum(winners)/len(winners):+,.0f}" if winners else "")
    print(f"  Avg loser:        ${sum(losers)/len(losers):+,.0f}" if losers else "")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Max drawdown:     ${max_dd:,.0f}")
    print(f"  Sharpe (ann.):    {sharpe:.2f}")
    print()

    # By exit reason
    print("  By exit reason:")
    from collections import Counter
    exit_counts = Counter(r.exit_reason for r in closed)
    for reason, count in exit_counts.most_common():
        reason_pnl = sum(r.realized_pnl_usd for r in closed if r.exit_reason == reason)
        print(f"    {reason:<20} {count:>4} trades  ${reason_pnl:>+8,.0f}")

    print()
    _print_skips(skipped)

    # Per-trade detail
    print(f"\n  Per-trade detail:")
    print(f"  {'Date':<12} {'Spot':>7} {'Cntr':>6} {'LPut':>6} {'LCall':>6} "
          f"{'EM':>5} {'Credit':>7} {'Exit':<18} {'P&L':>8}")
    for r in closed:
        print(f"  {r.session_date}  {r.underlying_at_entry:>7.0f} {r.center_strike:>6.0f} "
              f"{r.long_put:>6.0f} {r.long_call:>6.0f} {r.em_pts:>5.0f} "
              f"${r.credit_received:>6.2f} {r.exit_reason:<18} ${r.realized_pnl_usd:>+7.0f}")
    print(sep)


def _print_skips(skipped):
    if not skipped:
        return
    from collections import Counter
    skip_counts = Counter(r.skip_reason for r in skipped)
    print("  Skip reasons:")
    for reason, count in skip_counts.most_common():
        print(f"    {reason:<35} {count:>4}")


def run_optimize(config: dict, mem: dict, days_data, iv_by_date):
    """Grid optimization over key parameters."""
    grid = {
        "tp_fixed_dollars": [50, 75, 100],
        "slippage_per_leg": [0.00, 0.05, 0.10],
        "orb_filter_tolerance_pct": [0.0, 0.05, 0.10],
        "macro_filter_enabled": [True, False],
    }

    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    print(f"Grid: {len(combos)} combos x {len(days_data)} days\n")

    results_table = []
    for combo in combos:
        p = dict(zip(keys, combo))

        cfg = copy.deepcopy(config)
        cfg["options"]["tp_fixed_dollars"] = p["tp_fixed_dollars"]
        cfg["options"]["slippage_per_leg"] = p["slippage_per_leg"]
        cfg["strategy"]["orb_filter_tolerance_pct"] = p["orb_filter_tolerance_pct"]
        cfg["strategy"]["macro_filter"]["enabled"] = p["macro_filter_enabled"]

        results = run_single(cfg, mem, days_data, iv_by_date)
        closed = [r for r in results if r.status == "closed"]
        if not closed:
            continue

        pnls = [r.realized_pnl_usd for r in closed]
        total = sum(pnls)
        wins = len([x for x in pnls if x > 0])
        wr = wins / len(pnls)

        import numpy as np
        pnl_arr = np.array(pnls)
        sharpe = float(np.mean(pnl_arr) / max(np.std(pnl_arr), 0.01) * np.sqrt(252))

        peak = equity = dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            dd = max(dd, peak - equity)

        results_table.append({
            **p, "n": len(closed), "wr": wr, "pnl": total,
            "avg_pnl": total / len(closed), "sharpe": sharpe, "dd": dd,
        })

    if not results_table:
        print("No valid combos found")
        return

    hdr = (f'{"TP$":>4} {"Slip":>5} {"OrbTol":>6} {"Macro":>5} | '
           f'{"N":>3} {"WR":>5} | '
           f'{"PnL":>8} {"AvgPnL":>7} | '
           f'{"Shrp":>6} {"DD":>7}')

    results_table.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"Top 20 by PnL:\n{hdr}")
    for r in results_table[:20]:
        macro_str = "Y" if r["macro_filter_enabled"] else "N"
        print(f'{r["tp_fixed_dollars"]:>4} {r["slippage_per_leg"]:>5.2f} '
              f'{r["orb_filter_tolerance_pct"]:>6.2f} {macro_str:>5} | '
              f'{r["n"]:>3} {r["wr"]:>4.0%} | '
              f'{r["pnl"]:>8.0f} {r["avg_pnl"]:>7.1f} | '
              f'{r["sharpe"]:>6.1f} {r["dd"]:>7.0f}')

    results_table.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\nTop 20 by Sharpe:\n{hdr}")
    for r in results_table[:20]:
        macro_str = "Y" if r["macro_filter_enabled"] else "N"
        print(f'{r["tp_fixed_dollars"]:>4} {r["slippage_per_leg"]:>5.2f} '
              f'{r["orb_filter_tolerance_pct"]:>6.2f} {macro_str:>5} | '
              f'{r["n"]:>3} {r["wr"]:>4.0%} | '
              f'{r["pnl"]:>8.0f} {r["avg_pnl"]:>7.1f} | '
              f'{r["sharpe"]:>6.1f} {r["dd"]:>7.0f}')


def run_wf_cv(config: dict, mem: dict, days_data, iv_by_date):
    """Walk-forward cross-validation (same logic as ORB optimizer)."""
    from src_8am.backtest.iron_fly_optimizer import IronFlyOptimizer

    account_size = 100_000
    es_multiplier = config.get("options", {}).get("contract_multiplier", 100)

    opt = IronFlyOptimizer(
        base_config=config,
        cbbo_mem=mem,
        days_data=days_data,
        iv_by_date=iv_by_date,
        account_size=account_size,
        es_multiplier=es_multiplier,
    )

    best_params, fold_results = opt.run_walk_forward_cv(
        n_folds=4,
        top_n=5,
        min_trades=15,
        train_val_ratio=5.0,
    )

    if not best_params:
        print("No winner found.")
        return

    # Save optimized params
    output_path = "results/iron_fly_spx/optimized_params.yaml"
    opt.save_optimized_params(best_params, fold_results, output_path)

    # Run full backtest with winning params
    print("\n--- Full backtest with optimized params ---")
    strategy = IronFlyStrategy()
    final_config = strategy.apply_params(config, best_params)
    results = run_single(final_config, mem, days_data, iv_by_date)
    print_report(results, title="Iron Fly (WF-CV optimized)")


def main():
    parser = argparse.ArgumentParser(description="0DTE Iron Fly backtest")
    parser.add_argument("--config", default="config/config_iron_fly_spx.yaml")
    parser.add_argument("--optimize", action="store_true", help="Run grid optimization")
    parser.add_argument("--wf-cv", action="store_true", help="Walk-forward CV optimization")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mem, days_data, iv_by_date = load_data(config)

    if args.wf_cv:
        run_wf_cv(config, mem, days_data, iv_by_date)
    elif args.optimize:
        run_optimize(config, mem, days_data, iv_by_date)
    else:
        results = run_single(config, mem, days_data, iv_by_date)
        print_report(results)


if __name__ == "__main__":
    main()
