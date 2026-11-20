"""
ORB Backtest Report — P&L summary, win rate, drawdown, profit factor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_report(results: List[Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a summary report from a list of BacktestResult objects.
    Returns a dict with all key metrics.
    """
    closed = [r for r in results if r.status == "closed" and r.realized_pnl_pts is not None]
    skipped = [r for r in results if r.status == "skipped"]
    total_days = len(results)

    if not closed:
        return {
            "total_days": total_days,
            "trades": 0,
            "skipped": len(skipped),
            "win_rate": 0.0,
            "total_pnl_pts": 0.0,
            "avg_pnl_pts": 0.0,
            "max_drawdown_pts": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "by_exit_reason": {},
            "by_bias": {},
        }

    pnls = [r.realized_pnl_pts for r in closed]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_pnl = round(sum(pnls), 2)
    avg_pnl = round(total_pnl / len(pnls), 4)
    win_rate = round(len(winners) / len(pnls) * 100, 1)

    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Max drawdown (running cumulative)
    max_dd = _max_drawdown(pnls)

    # Sharpe (annualised, assuming 252 trading days, daily returns)
    sharpe = _sharpe(pnls)

    # By exit reason
    exit_counts: Dict[str, Dict[str, Any]] = {}
    for r in closed:
        reason = r.exit_reason or "unknown"
        if reason not in exit_counts:
            exit_counts[reason] = {"count": 0, "total_pnl": 0.0}
        exit_counts[reason]["count"] += 1
        exit_counts[reason]["total_pnl"] = round(
            exit_counts[reason]["total_pnl"] + r.realized_pnl_pts, 2
        )

    # By bias
    bias_counts: Dict[str, Dict[str, Any]] = {}
    for r in closed:
        b = r.bias or "unknown"
        if b not in bias_counts:
            bias_counts[b] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        bias_counts[b]["count"] += 1
        if r.realized_pnl_pts > 0:
            bias_counts[b]["wins"] += 1
        bias_counts[b]["total_pnl"] = round(bias_counts[b]["total_pnl"] + r.realized_pnl_pts, 2)

    skip_reasons: Dict[str, int] = {}
    for r in skipped:
        reason = r.skip_reason or "unknown"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    report = {
        "total_days": total_days,
        "trades": len(closed),
        "winners": len(winners),
        "losers": len(losers),
        "skipped": len(skipped),
        "win_rate_pct": win_rate,
        "total_pnl_pts": total_pnl,
        "avg_pnl_pts": avg_pnl,
        "avg_winner_pts": round(sum(winners) / len(winners), 2) if winners else 0.0,
        "avg_loser_pts": round(sum(losers) / len(losers), 2) if losers else 0.0,
        "max_winner_pts": round(max(winners), 2) if winners else 0.0,
        "max_loser_pts": round(min(losers), 2) if losers else 0.0,
        "gross_profit_pts": round(gross_profit, 2),
        "gross_loss_pts": round(gross_loss, 2),
        "profit_factor": profit_factor,
        "max_drawdown_pts": max_dd,
        "sharpe": sharpe,
        "by_exit_reason": exit_counts,
        "by_bias": bias_counts,
        "skip_reasons": skip_reasons,
    }
    return report


def print_report(report: Dict[str, Any], title: str = "ORB Backtest Report"):
    """Print a formatted report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Total days:       {report['total_days']}")
    print(f"  Trades placed:    {report['trades']}")
    print(f"  Skipped days:     {report['skipped']}")
    print(f"  Win rate:         {report.get('win_rate_pct', 0):.1f}%  "
          f"({report.get('winners', 0)}W / {report.get('losers', 0)}L)")
    print(f"  Total P&L:        {report['total_pnl_pts']:+.2f} pts")
    print(f"  Avg P&L/trade:    {report['avg_pnl_pts']:+.4f} pts")
    print(f"  Avg winner:       {report.get('avg_winner_pts', 0):+.2f} pts")
    print(f"  Avg loser:        {report.get('avg_loser_pts', 0):+.2f} pts")
    print(f"  Profit factor:    {report['profit_factor']:.2f}")
    print(f"  Max drawdown:     {report['max_drawdown_pts']:.2f} pts")
    print(f"  Sharpe (ann.):    {report['sharpe']:.2f}")
    print()
    print("  By exit reason:")
    for reason, stats in sorted(report.get("by_exit_reason", {}).items()):
        print(f"    {reason:<20} {stats['count']:>4} trades  {stats['total_pnl']:+.2f} pts")
    print()
    print("  By bias:")
    for bias, stats in sorted(report.get("by_bias", {}).items()):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
        print(f"    {bias:<10} {stats['count']:>4} trades  WR={wr:.0f}%  {stats['total_pnl']:+.2f} pts")
    print()
    print("  Skip reasons:")
    for reason, count in sorted(report.get("skip_reasons", {}).items()):
        print(f"    {reason:<35} {count:>4}")
    print(sep)


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _max_drawdown(pnls: List[float]) -> float:
    """Compute max drawdown in points from a list of per-trade P&Ls."""
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def _sharpe(pnls: List[float], periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (no risk-free rate)."""
    if len(pnls) < 2:
        return 0.0
    import math
    n = len(pnls)
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return round((mean / std) * math.sqrt(periods_per_year), 2)
