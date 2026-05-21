"""
Backtest report — P&L summary for src_v2 strategy backtests.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from src_v2.backtest.engine import BacktestResult


def generate_report(results: List[BacktestResult]) -> Dict[str, Any]:
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

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in closed:
        equity += r.realized_pnl_usd or 0
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    daily_pnl: Dict[str, float] = defaultdict(float)
    for r in closed:
        daily_pnl[str(r.session_date)] += r.realized_pnl_usd or 0
    daily_returns = list(daily_pnl.values())

    # Order-robust drawdown on the per-day net-P&L equity curve, plus the worst
    # single-day net loss — these surface correlated same-day risk (e.g. many
    # short legs stopping on one big directional day).
    d_equity = 0.0
    d_peak = 0.0
    max_dd_daily = 0.0
    for _, day_pnl in sorted(daily_pnl.items()):
        d_equity += day_pnl
        d_peak = max(d_peak, d_equity)
        max_dd_daily = max(max_dd_daily, d_peak - d_equity)
    worst_day_pnl = min(daily_returns) if daily_returns else 0.0
    best_day_pnl = max(daily_returns) if daily_returns else 0.0
    sharpe = 0.0
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std > 0:
            sharpe = mean / std * math.sqrt(252)
        elif mean > 0:
            # All daily returns are identical and positive — treat as very high Sharpe
            sharpe = 999.0

    by_reason: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for r in closed:
        by_reason[r.exit_reason or "unknown"]["count"] += 1
        by_reason[r.exit_reason or "unknown"]["pnl"] += r.realized_pnl_usd or 0

    by_bias: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
    for r in closed:
        b = r.bias or "unknown"
        by_bias[b]["count"] += 1
        if (r.realized_pnl_usd or 0) > 0:
            by_bias[b]["wins"] += 1
        by_bias[b]["pnl"] += r.realized_pnl_usd or 0

    credits = [r.credit_received for r in closed if r.credit_received]
    avg_credit = sum(credits) / len(credits) if credits else 0.0

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
        "max_drawdown_daily_usd": round(max_dd_daily, 2),
        "worst_day_pnl_usd": round(worst_day_pnl, 2),
        "best_day_pnl_usd": round(best_day_pnl, 2),
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


def print_report(report: Dict[str, Any], title: str = "src_v2 Backtest") -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Total days:        {report['total_days']}")
    print(f"  Trades placed:     {report['trades_placed']}")
    print(f"  Skipped days:      {report['skipped_days']}")
    if report["trades_placed"] == 0:
        print(sep)
        return
    print(f"  Win rate:          {report['win_rate']*100:.1f}%  "
          f"({report.get('wins',0)}W / {report.get('losses',0)}L)")
    print(f"  Total P&L:         ${report['total_pnl_usd']:,.2f}")
    print(f"  Avg P&L/trade:     ${report['avg_pnl_per_trade_usd']:,.2f}")
    print(f"  Avg winner:        ${report['avg_winner_usd']:,.2f}")
    print(f"  Avg loser:         ${report['avg_loser_usd']:,.2f}")
    print(f"  Profit factor:     {report['profit_factor']}")
    print(f"  Max drawdown:      ${report['max_drawdown_usd']:,.2f}")
    if "max_drawdown_daily_usd" in report:
        print(f"  Max DD (daily eq): ${report['max_drawdown_daily_usd']:,.2f}")
        print(f"  Worst day P&L:     ${report['worst_day_pnl_usd']:,.2f}")
        print(f"  Best day P&L:      ${report['best_day_pnl_usd']:,.2f}")
    print(f"  Sharpe (ann.):     {report['sharpe_annualised']:.2f}")
    print(f"  Avg credit:        ${report['avg_credit_received']:.3f}")
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


def print_trades(results: List[BacktestResult]) -> None:
    """Print per-trade table to console."""
    closed = [r for r in results if r.status == "closed"]
    if not closed:
        return
    closed.sort(key=lambda r: (r.session_date, r.spread_type or ""))

    print(f"\n{'─' * 100}")
    print(f"  {'Date':<12} {'Bias':<6} {'Side':<5} {'Short':>7} {'Long':>7} "
          f"{'Credit':>8} {'Debit':>8} {'PnL':>9} {'Exit'}")
    print(f"{'─' * 100}")
    for r in closed:
        side = "put" if r.spread_type and "put" in r.spread_type else "call"
        print(f"  {r.session_date!s:<12} {(r.bias or '?'):<6} {side:<5} "
              f"{r.short_strike or 0:>7.0f} {r.long_strike or 0:>7.0f} "
              f"${r.credit_received or 0:>6.2f} ${r.debit_paid or 0:>6.2f} "
              f"${r.realized_pnl_usd or 0:>+8.0f} {r.exit_reason or '?'}")
    print(f"{'─' * 100}")


def export_trades_csv(results: List[BacktestResult], path: str) -> None:
    """Export all BacktestResult fields to CSV."""
    import csv
    from pathlib import Path
    from dataclasses import fields

    closed = [r for r in results if r.status == "closed"]
    if not closed:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    field_names = [f.name for f in fields(BacktestResult)]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for r in closed:
            row = {name: getattr(r, name) for name in field_names}
            writer.writerows([row])
    print(f"  Trades CSV → {path}")


def _count_reasons(skipped: List[BacktestResult]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in skipped:
        counts[r.skip_reason or "unknown"] += 1
    return dict(counts)
