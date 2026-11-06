"""
Analyze SPX 0DTE strategy for:
1. Iron Condor (IC) benefit
2. Breach recovery — how often does price return OTM after touching short strike

Uses SPX optimized params: delta=0.25, width=5, tp=0.75, qty=50
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_8am.backtest.bar_loader import BarLoader
from src_8am.backtest.options_engine import OrbOptionsBacktestEngine


# SPX optimized params (from WF-CV 2026-03-14)
SPX_PARAMS = {
    "target_short_delta": 0.25,
    "spread_width_pts": 5,
    "tp_pct_of_credit": 0.75,
    "sl_mult_of_credit": 0,
    "slippage_per_spread": 0.10,
    "quantity": 50,
}

CV_START = date(2022, 5, 11)
CV_END = date(2026, 2, 28)


def load_and_run():
    config_path = "config/config_8am_spx_options.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply SPX optimized params
    for k, v in SPX_PARAMS.items():
        config["options"][k] = v

    # Scale risk limit
    width = SPX_PARAMS["spread_width_pts"]
    mult = config["options"].get("contract_multiplier", 100)
    qty = SPX_PARAMS["quantity"]
    needed = width * mult * qty
    config.setdefault("risk", {})["max_loss_per_trade_usd"] = needed
    config["risk"]["max_daily_loss_usd"] = needed

    data_file = config["backtest"]["data_file"]
    price_scale = config.get("backtest", {}).get("price_scale", 10.0)
    print(f"Loading bars from {data_file} (scale={price_scale})...")
    loader = BarLoader(data_file, price_scale=price_scale, symbol="SPX")
    days = list(loader.iter_days(start_date=CV_START, end_date=CV_END))
    print(f"  {len(days)} day-symbol chunks")

    print(f"Running backtest with SPX params (delta=0.25, width=5, qty=50)...")
    engine = OrbOptionsBacktestEngine(config=config)
    results = engine.run_on_days(days)

    trades = [r for r in results if r.status == "closed"]
    print(f"  {len(trades)} filled trades\n")
    return trades, config


def analyze_ic(trades, config):
    """Estimate IC benefit."""
    print("=" * 70)
    print("  IRON CONDOR ANALYSIS — SPX (delta=0.25, width=5, qty=50)")
    print("=" * 70)

    mult = config["options"].get("contract_multiplier", 100)
    qty = SPX_PARAMS["quantity"]
    tp_pct = config["options"].get("tp_pct_of_credit", 0.75)

    by_date = defaultdict(list)
    for t in trades:
        by_date[t.session_date].append(t)

    single_pnl = 0.0
    ic_pnl = 0.0
    single_wins = 0
    ic_wins = 0
    total = 0
    by_reason = defaultdict(lambda: {"count": 0, "single": 0.0, "ic": 0.0})

    for dt, day_trades in sorted(by_date.items()):
        t = day_trades[0]
        total += 1

        trade_pnl = t.realized_pnl_usd or 0.0
        single_pnl += trade_pnl
        if trade_pnl > 0:
            single_wins += 1

        credit = t.credit_received or 0.0
        reason = t.exit_reason or "unknown"

        if reason in ("expired_worthless", "tp_75pct", "tp_50pct", "tp_25pct"):
            opposite_pnl = credit * tp_pct * mult * qty
        elif reason in ("short_breached", "eod_close"):
            opposite_pnl = credit * mult * qty
        else:
            opposite_pnl = 0

        ic_trade_pnl = trade_pnl + opposite_pnl
        ic_pnl += ic_trade_pnl
        if ic_trade_pnl > 0:
            ic_wins += 1

        by_reason[reason]["count"] += 1
        by_reason[reason]["single"] += trade_pnl
        by_reason[reason]["ic"] += ic_trade_pnl

    print(f"\nTrades: {total}")
    print(f"\nSingle-side credit spread:")
    print(f"  Total PnL:  ${single_pnl:>12,.0f}")
    print(f"  Win rate:   {single_wins/total*100:.1f}%")
    print(f"  Per trade:  ${single_pnl/total:>8,.0f}")

    print(f"\nIron Condor estimate (conservative — opposite side at same TP):")
    print(f"  Total PnL:  ${ic_pnl:>12,.0f}")
    print(f"  Win rate:   {ic_wins/total*100:.1f}%")
    print(f"  Per trade:  ${ic_pnl/total:>8,.0f}")
    if single_pnl > 0:
        print(f"  IC / Single ratio: {ic_pnl/single_pnl:.2f}x")

    print(f"\n  PnL breakdown by exit reason:")
    for reason, data in sorted(by_reason.items(), key=lambda x: -x[1]["count"]):
        print(f"    {reason:20s}  n={data['count']:>4d}  "
              f"single=${data['single']:>10,.0f}  ic=${data['ic']:>10,.0f}")


def analyze_breach_recovery(trades, config):
    """Analyze breach recovery patterns."""
    print("\n")
    print("=" * 70)
    print("  BREACH RECOVERY ANALYSIS — SPX (delta=0.25, width=5, qty=50)")
    print("=" * 70)

    breached = [t for t in trades if t.exit_reason == "short_breached"]
    non_breached = [t for t in trades if t.exit_reason != "short_breached"]

    print(f"\nTotal trades: {len(trades)}")
    print(f"Breached:     {len(breached)} ({len(breached)/len(trades)*100:.1f}%)")
    print(f"Not breached: {len(non_breached)} ({len(non_breached)/len(trades)*100:.1f}%)")

    if not breached:
        print("No breached trades to analyze.")
        return

    breach_pnl = sum(t.realized_pnl_usd or 0 for t in breached)
    print(f"\nBreach trade total PnL: ${breach_pnl:,.0f}")
    print(f"Avg breach loss: ${breach_pnl/len(breached):,.0f}")

    # Partial recoveries
    recovery_possible = sum(1 for t in breached if (t.debit_paid or 0) < (t.credit_received or 0))
    print(f"\nBreach exits where debit < credit (partial recovery): "
          f"{recovery_possible} ({recovery_possible/len(breached)*100:.1f}%)")

    # P&L distribution
    breach_pnls = sorted([t.realized_pnl_usd or 0 for t in breached])
    print(f"\nBreach P&L distribution:")
    print(f"  Worst:    ${breach_pnls[0]:>10,.0f}")
    print(f"  25th pct: ${breach_pnls[len(breach_pnls)//4]:>10,.0f}")
    print(f"  Median:   ${breach_pnls[len(breach_pnls)//2]:>10,.0f}")
    print(f"  75th pct: ${breach_pnls[3*len(breach_pnls)//4]:>10,.0f}")
    print(f"  Best:     ${breach_pnls[-1]:>10,.0f}")

    # By spread type
    print(f"\nBreach by direction:")
    for st in ("put_credit_spread", "call_credit_spread"):
        b = [t for t in breached if t.spread_type == st]
        if b:
            pnl = sum(t.realized_pnl_usd or 0 for t in b)
            label = "PUT (long bias)" if "put" in st else "CALL (short bias)"
            print(f"  {label:20s}: {len(b)} trades, PnL=${pnl:>10,.0f}, avg=${pnl/len(b):>8,.0f}")

    # By day of week
    print(f"\nBreach by day of week:")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    dow_counts = defaultdict(int)
    dow_total = defaultdict(int)
    for t in trades:
        dow = t.session_date.weekday()
        dow_total[dow] += 1
        if t.exit_reason == "short_breached":
            dow_counts[dow] += 1
    for dow in range(5):
        total = dow_total[dow]
        breaches = dow_counts[dow]
        rate = breaches / total * 100 if total > 0 else 0
        print(f"  {dow_names[dow]}: {breaches:>3d}/{total:>3d} = {rate:.1f}%")


def main():
    trades, config = load_and_run()
    analyze_ic(trades, config)
    analyze_breach_recovery(trades, config)


if __name__ == "__main__":
    main()
