"""
Options backtest report — P&L summary for ORB credit spread simulation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from es_8am.backtest.options_engine import OptionsBacktestResult


def generate_options_report(results: List[OptionsBacktestResult]) -> Dict[str, Any]:
    closed = [r for r in results if r.status == "closed"]
    skipped = [r for r in results if r.status == "skipped"]

    if not closed:
        return {
            "total_days": len(results),
            "trades_placed": 0,
            "skipped_days": len(skipped),
            "win_rate": 0.0,
            "total_pnl_usd": 0.0,
            "skip_reasons": _count_reasons(skipped),
        }

    wins = [r for r in closed if (r.realized_pnl_usd or 0) > 0]
    losses = [r for r in closed if (r.realized_pnl_usd or 0) <= 0]

    total_pnl = sum(r.realized_pnl_usd or 0 for r in closed)
    win_rate = len(wins) / len(closed) if closed else 0.0

    avg_win = sum(r.realized_pnl_usd or 0 for r in wins) / len(wins) if wins else 0.0
    avg_loss = sum(r.realized_pnl_usd or 0 for r in losses) / len(losses) if losses else 0.0

    gross_profit = sum(r.realized_pnl_usd or 0 for r in wins)
    gross_loss = abs(sum(r.realized_pnl_usd or 0 for r in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown (running equity curve)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in closed:
        equity += r.realized_pnl_usd or 0
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualised, daily returns, risk-free=0)
    daily_pnl: Dict[str, float] = defaultdict(float)
    for r in closed:
        daily_pnl[str(r.session_date)] += r.realized_pnl_usd or 0
    daily_returns = list(daily_pnl.values())
    sharpe = 0.0
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0

    # By exit reason
    by_reason: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for r in closed:
        key = r.exit_reason or "unknown"
        by_reason[key]["count"] += 1
        by_reason[key]["pnl"] += r.realized_pnl_usd or 0

    # By bias
    by_bias: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
    for r in closed:
        b = r.bias or "unknown"
        by_bias[b]["count"] += 1
        if (r.realized_pnl_usd or 0) > 0:
            by_bias[b]["wins"] += 1
        by_bias[b]["pnl"] += r.realized_pnl_usd or 0

    # Credit stats
    credits = [r.credit_received for r in closed if r.credit_received]
    avg_credit = sum(credits) / len(credits) if credits else 0.0

    # Delta stats
    deltas = [r.short_delta_at_entry for r in closed if r.short_delta_at_entry]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

    return {
        "total_days": len(results),
        "trades_placed": len(closed),
        "skipped_days": len(skipped),
        "win_rate": round(win_rate, 4),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl_usd": round(total_pnl, 2),
        "avg_pnl_per_trade_usd": round(total_pnl / len(closed), 2),
        "avg_winner_usd": round(avg_win, 2),
        "avg_loser_usd": round(avg_loss, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "max_drawdown_usd": round(max_dd, 2),
        "sharpe_annualised": round(sharpe, 2),
        "avg_credit_received": round(avg_credit, 3),
        "avg_short_delta": round(avg_delta, 4),
        "by_exit_reason": {k: {"count": v["count"], "pnl_usd": round(v["pnl"], 2)}
                           for k, v in sorted(by_reason.items())},
        "by_bias": {k: {
            "count": v["count"],
            "win_rate": round(v["wins"] / v["count"], 3) if v["count"] else 0,
            "pnl_usd": round(v["pnl"], 2),
        } for k, v in sorted(by_bias.items())},
        "skip_reasons": _count_reasons(skipped),
    }


def print_options_report(report: Dict[str, Any], title: str = "ORB Options Backtest"):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Total days:        {report['total_days']}")
    print(f"  Trades placed:     {report['trades_placed']}")
    print(f"  Skipped days:      {report['skipped_days']}")
    print(f"  Win rate:          {report['win_rate']*100:.1f}%  "
          f"({report.get('wins',0)}W / {report.get('losses',0)}L)")
    print(f"  Total P&L:         ${report['total_pnl_usd']:,.2f}")
    print(f"  Avg P&L/trade:     ${report['avg_pnl_per_trade_usd']:,.2f}")
    print(f"  Avg winner:        ${report['avg_winner_usd']:,.2f}")
    print(f"  Avg loser:         ${report['avg_loser_usd']:,.2f}")
    print(f"  Profit factor:     {report['profit_factor']}")
    print(f"  Max drawdown:      ${report['max_drawdown_usd']:,.2f}")
    print(f"  Sharpe (ann.):     {report['sharpe_annualised']:.2f}")
    print(f"  Avg credit:        ${report['avg_credit_received']:.3f} ({report['avg_credit_received']*100:.1f}¢ / share)")
    print(f"  Avg short delta:   {report['avg_short_delta']:.4f}")

    if report.get("by_exit_reason"):
        print()
        print("  By exit reason:")
        for reason, v in report["by_exit_reason"].items():
            print(f"    {reason:<28} {v['count']:>4} trades  ${v['pnl_usd']:>9,.2f}")

    if report.get("by_bias"):
        print()
        print("  By bias:")
        for bias, v in report["by_bias"].items():
            print(f"    {bias:<8} {v['count']:>4} trades  WR={v['win_rate']*100:.0f}%  ${v['pnl_usd']:>9,.2f}")

    if report.get("skip_reasons"):
        print()
        print("  Skip reasons:")
        for reason, count in sorted(report["skip_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<40} {count:>4}")

    print(sep)


def _count_reasons(skipped: List[OptionsBacktestResult]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in skipped:
        counts[r.skip_reason or "unknown"] += 1
    return dict(counts)
