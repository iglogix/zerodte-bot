"""Hedge Hog trade-level analysis: crash-window stress, distributions, streaks.

Reads a trades CSV produced by scripts/backtest_hedgehog.py.

Usage:
    .venv/bin/python scripts/analyze_hedgehog.py results/hedgehog_spy/trades_base.csv
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict

# fast-crash / vol-event windows in the data span (entry OR exit inside)
STRESS_WINDOWS = {
    "covid_crash":    (20200214, 20200410),
    "2022_h1_bear":   (20220103, 20220630),
    "2022_h2_bear":   (20220815, 20221031),
    "aug24_volmageddon": (20240722, 20240816),
    "apr25_tariffs":  (20250328, 20250430),
}


def main(path: str) -> None:
    with open(path) as f:
        trades = [
            {
                **row,
                "entry_date": int(row["entry_date"]),
                "exit_date": int(row["exit_date"]),
                "pnl_usd": float(row["pnl_usd"]),
                "days_held": int(row["days_held"]),
                "margin_usd": float(row["margin_usd"]),
                "net_credit": float(row["net_credit"]),
                "hog_delta": float(row["hog_delta"]),
                "atm_iv_entry": float(row.get("atm_iv_entry") or "nan"),
            }
            for row in csv.DictReader(f)
        ]
    pnls = [t["pnl_usd"] for t in trades]
    print(f"{len(trades)} trades, total {sum(pnls):+,.0f}")

    print("\n-- P&L distribution --")
    qs = statistics.quantiles(pnls, n=20)
    print(f"  p5 {qs[0]:+8.0f}  p25 {qs[4]:+8.0f}  median {qs[9]:+8.0f}  "
          f"p75 {qs[14]:+8.0f}  p95 {qs[18]:+8.0f}")
    print(f"  min {min(pnls):+8.0f}  max {max(pnls):+8.0f}")
    rom = [t["pnl_usd"] / t["margin_usd"] for t in trades if t["margin_usd"] > 0]
    print(f"  return-on-margin per trade: median {statistics.median(rom):+.2%}, "
          f"mean {statistics.fmean(rom):+.2%}")

    print("\n-- worst 10 trades --")
    for t in sorted(trades, key=lambda t: t["pnl_usd"])[:10]:
        print(f"  {t['entry_date']} -> {t['exit_date']}  {t['pnl_usd']:+8.0f}  "
              f"held {t['days_held']:>2}d  exit={t['exit_reason']}  "
              f"hog {float(t['k_hog']):.0f}@{t['hog_delta']:.2f}d")

    print("\n-- stress windows --")
    for name, (a, b) in STRESS_WINDOWS.items():
        sel = [t for t in trades if a <= t["exit_date"] <= b or a <= t["entry_date"] <= b]
        if not sel:
            print(f"  {name:<18} no trades")
            continue
        w = sum(1 for t in sel if t["pnl_usd"] > 0)
        print(f"  {name:<18} {len(sel):>3} trades  WR {w/len(sel):>4.0%}  "
              f"P&L {sum(t['pnl_usd'] for t in sel):+9,.0f}  "
              f"worst {min(t['pnl_usd'] for t in sel):+8,.0f}")

    print("\n-- losing streaks --")
    streak = best = 0
    for t in sorted(trades, key=lambda t: t["exit_date"]):
        streak = streak + 1 if t["pnl_usd"] <= 0 else 0
        best = max(best, streak)
    print(f"  max consecutive losers: {best}")

    print("\n-- by exit reason --")
    g = defaultdict(list)
    for t in trades:
        g[t["exit_reason"]].append(t["pnl_usd"])
    for reason, xs in sorted(g.items()):
        w = sum(1 for x in xs if x > 0)
        print(f"  {reason:<18} {len(xs):>4}  WR {w/len(xs):>4.0%}  "
              f"avg {statistics.fmean(xs):+7.0f}  total {sum(xs):+10,.0f}")

    print("\n-- by entry ATM IV (30-DTE, chain-derived) --")
    buckets = [(0.0, 0.12), (0.12, 0.16), (0.16, 0.22), (0.22, 0.30), (0.30, 9.9)]
    for lo, hi in buckets:
        xs = [t["pnl_usd"] for t in trades
              if t["atm_iv_entry"] == t["atm_iv_entry"] and lo <= t["atm_iv_entry"] < hi]
        if not xs:
            continue
        w = sum(1 for x in xs if x > 0)
        print(f"  IV {lo:.2f}-{hi:.2f}  {len(xs):>4} trades  WR {w/len(xs):>4.0%}  "
              f"avg {statistics.fmean(xs):+7.0f}  total {sum(xs):+10,.0f}")

    print("\n-- by entry year --")
    gy = defaultdict(list)
    for t in trades:
        gy[t["entry_date"] // 10000].append(t["pnl_usd"])
    for y, xs in sorted(gy.items()):
        w = sum(1 for x in xs if x > 0)
        print(f"  {y}  {len(xs):>3} trades  WR {w/len(xs):>4.0%}  total {sum(xs):+10,.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
