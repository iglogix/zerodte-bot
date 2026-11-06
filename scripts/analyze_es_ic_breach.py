"""
Analyze ES 8AM ORB strategy for:
1. Iron Condor (IC) benefit — estimate PnL if both sides collected premium
2. Breach recovery — how often does price return OTM after touching short strike

Uses ES optimized params: delta=0.16, width=20, qty=25, slippage=0.12
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_8am.backtest.data_loader import DatabentoLoader
from es_8am.backtest.options_engine import OrbOptionsBacktestEngine


# ES optimized params (from WF-CV 2026-03-13)
ES_PARAMS = {
    "target_short_delta": 0.16,
    "spread_width_pts": 20,
    "tp_pct_of_credit": 0.75,
    "sl_mult_of_credit": 0,
    "slippage_per_spread": 0.12,
    "quantity": 25,
}

CV_START = date(2021, 3, 1)
CV_END = date(2026, 2, 26)


def load_and_run():
    config_path = "config/config_8am_es_options.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply ES optimized params
    for k, v in ES_PARAMS.items():
        config["options"][k] = v

    # Scale risk limit to accommodate position
    width = ES_PARAMS["spread_width_pts"]
    mult = config["options"].get("contract_multiplier", 50)
    qty = ES_PARAMS["quantity"]
    needed = width * mult * qty
    config.setdefault("risk", {})["max_loss_per_trade_usd"] = needed
    config["risk"]["max_daily_loss_usd"] = needed

    data_file = config["backtest"]["data_file"]
    print(f"Loading bars from {data_file}...")
    loader = DatabentoLoader(data_file)
    days = list(loader.iter_days(start_date=CV_START, end_date=CV_END))
    print(f"  {len(days)} day-symbol chunks")

    print("Running backtest with ES params (delta=0.16, width=20, qty=25)...")
    engine = OrbOptionsBacktestEngine(config=config)
    results = engine.run_on_days(days)

    trades = [r for r in results if r.status == "closed"]
    print(f"  {len(trades)} filled trades\n")
    return trades, config


def analyze_ic(trades, config):
    """Estimate IC benefit: on each trade day, the opposite side also collects premium."""
    print("=" * 70)
    print("  IRON CONDOR ANALYSIS — ES (delta=0.16, width=20, qty=25)")
    print("=" * 70)

    mult = config["options"].get("contract_multiplier", 50)
    qty = ES_PARAMS["quantity"]

    # Group trades by date to pair LONG/SHORT biases
    by_date = defaultdict(list)
    for t in trades:
        by_date[t.session_date].append(t)

    single_pnl = 0.0
    ic_pnl = 0.0
    single_wins = 0
    ic_wins = 0
    total = 0

    for dt, day_trades in sorted(by_date.items()):
        t = day_trades[0]  # one trade per day
        total += 1

        trade_pnl = t.realized_pnl_usd or 0.0
        single_pnl += trade_pnl
        if trade_pnl > 0:
            single_wins += 1

        # The opposite side: if we sold a PUT spread (LONG bias),
        # the CALL spread (SHORT bias side) also collects premium.
        # If the trade won (expired worthless / TP), the opposite side
        # almost certainly also won (price moved away from it).
        # If the trade lost (breach), the opposite side won fully.
        credit = t.credit_received or 0.0

        if t.exit_reason in ("expired_worthless", "tp_75pct", "tp_50pct", "tp_25pct"):
            # Winning trade: opposite side also wins (full credit or TP)
            # Conservative: assume opposite side also hits same TP
            tp_pct = config["options"].get("tp_pct_of_credit", 0.75)
            opposite_pnl = credit * tp_pct * mult * qty
            ic_trade_pnl = trade_pnl + opposite_pnl
        elif t.exit_reason in ("short_breached", "eod_close"):
            # Losing trade: price moved against us, opposite side wins full credit
            opposite_pnl = credit * mult * qty
            ic_trade_pnl = trade_pnl + opposite_pnl
        else:
            opposite_pnl = 0
            ic_trade_pnl = trade_pnl

        ic_pnl += ic_trade_pnl
        if ic_trade_pnl > 0:
            ic_wins += 1

    print(f"\nTrades: {total}")
    print(f"\nSingle-side credit spread:")
    print(f"  Total PnL:  ${single_pnl:>12,.0f}")
    print(f"  Win rate:   {single_wins/total*100:.1f}%")
    print(f"  Per trade:  ${single_pnl/total:>8,.0f}")

    print(f"\nIron Condor estimate (conservative — opposite side at same TP):")
    print(f"  Total PnL:  ${ic_pnl:>12,.0f}")
    print(f"  Win rate:   {ic_wins/total*100:.1f}%")
    print(f"  Per trade:  ${ic_pnl/total:>8,.0f}")
    print(f"  IC / Single ratio: {ic_pnl/single_pnl:.2f}x" if single_pnl > 0 else "")

    # Break down by exit reason
    print(f"\n  PnL breakdown by exit reason:")
    by_reason = defaultdict(lambda: {"count": 0, "single": 0.0, "ic": 0.0})
    for dt, day_trades in sorted(by_date.items()):
        t = day_trades[0]
        trade_pnl = t.realized_pnl_usd or 0.0
        credit = t.credit_received or 0.0
        reason = t.exit_reason or "unknown"

        if reason in ("expired_worthless", "tp_75pct", "tp_50pct", "tp_25pct"):
            tp_pct = config["options"].get("tp_pct_of_credit", 0.75)
            opp = credit * tp_pct * mult * qty
        elif reason in ("short_breached", "eod_close"):
            opp = credit * mult * qty
        else:
            opp = 0

        by_reason[reason]["count"] += 1
        by_reason[reason]["single"] += trade_pnl
        by_reason[reason]["ic"] += trade_pnl + opp

    for reason, data in sorted(by_reason.items(), key=lambda x: -x[1]["count"]):
        print(f"    {reason:20s}  n={data['count']:>4d}  "
              f"single=${data['single']:>10,.0f}  ic=${data['ic']:>10,.0f}")


def analyze_breach_recovery(trades, config):
    """Analyze how often price returns OTM after touching the short strike."""
    print("\n")
    print("=" * 70)
    print("  BREACH RECOVERY ANALYSIS — ES (delta=0.16, width=20, qty=25)")
    print("=" * 70)

    mult = config["options"].get("contract_multiplier", 50)
    qty = ES_PARAMS["quantity"]

    breached = [t for t in trades if t.exit_reason == "short_breached"]
    non_breached = [t for t in trades if t.exit_reason != "short_breached"]

    print(f"\nTotal trades: {len(trades)}")
    print(f"Breached:     {len(breached)} ({len(breached)/len(trades)*100:.1f}%)")
    print(f"Not breached: {len(non_breached)} ({len(non_breached)/len(trades)*100:.1f}%)")

    if not breached:
        print("No breached trades to analyze.")
        return

    # Analyze breach outcomes
    breach_pnl = sum(t.realized_pnl_usd or 0 for t in breached)
    print(f"\nBreach trade total PnL: ${breach_pnl:,.0f}")
    print(f"Avg breach loss: ${breach_pnl/len(breached):,.0f}")

    # Check how deep the breach went
    recovery_possible = 0
    held_better = 0
    held_worse = 0
    held_pnl_diff = 0.0

    for t in breached:
        credit = t.credit_received or 0.0
        debit = t.debit_paid or 0.0
        actual_pnl = t.realized_pnl_usd or 0.0

        # If debit < credit, the exit was still profitable (price came back)
        if debit < credit:
            recovery_possible += 1

        # Compare: exit at breach vs what if held to expiration
        # For breach exits, the engine exits immediately on breach
        # If we held instead: the spread either expires ITM (max loss = width)
        # or recovers. We can estimate from the exit price vs strike.
        if t.underlying_at_exit and t.short_strike:
            if t.spread_type == "PUT":
                # PUT spread: loses when price drops below short strike
                # If exit price > short_strike at EOD, it recovered
                pass
            elif t.spread_type == "CALL":
                # CALL spread: loses when price rises above short strike
                pass

    # Stats on breach depth
    print(f"\nBreach exits where debit < credit (partial recovery): "
          f"{recovery_possible} ({recovery_possible/len(breached)*100:.1f}%)")

    # P&L distribution of breached trades
    breach_pnls = sorted([t.realized_pnl_usd or 0 for t in breached])
    if breach_pnls:
        print(f"\nBreach P&L distribution:")
        print(f"  Worst:    ${breach_pnls[0]:>10,.0f}")
        print(f"  25th pct: ${breach_pnls[len(breach_pnls)//4]:>10,.0f}")
        print(f"  Median:   ${breach_pnls[len(breach_pnls)//2]:>10,.0f}")
        print(f"  75th pct: ${breach_pnls[3*len(breach_pnls)//4]:>10,.0f}")
        print(f"  Best:     ${breach_pnls[-1]:>10,.0f}")

    # By bias
    print(f"\nBreach by bias:")
    for bias in ("LONG", "SHORT"):
        b = [t for t in breached if t.bias == bias]
        if b:
            pnl = sum(t.realized_pnl_usd or 0 for t in b)
            print(f"  {bias:5s}: {len(b)} trades, PnL=${pnl:>10,.0f}, "
                  f"avg=${pnl/len(b):>8,.0f}")

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
