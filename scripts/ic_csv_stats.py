"""Quick stats from an open-IC trades.csv: daily P&L, annualized Sharpe, WR, DD.

Usage: python -m scripts.ic_csv_stats results/<dir>/trades.csv [label]
"""
import sys
import csv
import math
from collections import defaultdict


def summarize(path: str, label: str = "") -> dict:
    daily = defaultdict(float)
    with open(path) as f:
        for row in csv.DictReader(f):
            pnl = row.get("realized_pnl_usd", "")
            if pnl in ("", None):
                continue
            daily[row["session_date"]] += float(pnl)
    days = sorted(daily)
    pnls = [daily[d] for d in days]
    n = len(pnls)
    if n == 0:
        print(f"{label}: no trades")
        return {}
    total = sum(pnls)
    mean = total / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0.0
    sharpe_ann = (mean / sd * math.sqrt(252)) if sd > 0 else 0.0
    wins = sum(1 for p in pnls if p > 0)
    losses = n - wins
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    pf = gw / gl if gl > 0 else float("inf")
    eq = peak = maxdd = 0.0
    streak = max_streak = 0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
        streak = streak + 1 if p < 0 else 0
        max_streak = max(max_streak, streak)
    print(f"=== {label or path} ===")
    print(f"  period         : {days[0]} -> {days[-1]}  ({n} trading days)")
    print(f"  total P&L      : ${total:,.0f}   (mean ${mean:,.1f}/day, sd ${sd:,.1f})")
    print(f"  Sharpe (ann)   : {sharpe_ann:.2f}")
    print(f"  win rate       : {wins/n*100:.1f}%   ({wins}W / {losses}L)")
    print(f"  profit factor  : {pf:.2f}")
    print(f"  avg win/loss   : +${(gw/wins if wins else 0):,.0f} / -${(gl/losses if losses else 0):,.0f}")
    print(f"  max drawdown   : ${maxdd:,.0f}")
    print(f"  max loss streak: {max_streak} days")
    return {"sharpe_ann": sharpe_ann, "total": total, "win_rate": wins / n,
            "pf": pf, "max_dd": maxdd, "days": n}


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
