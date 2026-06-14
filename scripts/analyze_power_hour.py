#!/usr/bin/env python3
"""
Cross-variant analysis for the Power Hour backtests.

Reads results/power_hour_spx/trades_*.csv and reports:
  - variant comparison (P&L, WR, PF, Sharpe, max DD)
  - premium-regime buckets (day's T1 strangle credit as the richness proxy —
    the strategy doc says the edge broke when premium collapsed, §12)
  - monthly P&L for the headline variant
  - portfolio sizing simulation (T1=8/T2=10/T3=10 contracts, $200K start, like
    the lecture 1.2 reference portfolio) + rolling 20-day worst windows (§14)

Usage:
    .venv/bin/python scripts/analyze_power_hour.py
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

RESULTS = Path("results/power_hour_spx")

VARIANTS = {
    "A trigger_plus (base)": "trades_A_triggerplus.csv",
    "A entry_plus": "trades_A_entryplus.csv",
    "A stop=quote": "trades_A_stopquote.csv",
    "A exit=expire": "trades_A_expire.csv",
    "A T2@15:14": "trades_A_t2_1514.csv",
    "C regime overlay": "trades_C_overlay.csv",
    "C + expire": "trades_C_expire.csv",
    "C + stop=quote": "trades_C_stopquote.csv",
    "B 20d/trail (4-lot)": "trades_B.csv",
}


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "closed":
                continue
            r["pnl"] = float(r["realized_pnl_usd"])
            r["date"] = date.fromisoformat(r["session_date"])
            rows.append(r)
    return rows


def day_pnls(rows: list[dict]) -> list[tuple[date, float]]:
    d = defaultdict(float)
    for r in rows:
        d[r["date"]] += r["pnl"]
    return sorted(d.items())


def stats(rows: list[dict]) -> dict:
    pnls = [r["pnl"] for r in rows]
    days = day_pnls(rows)
    dp = [p for _, p in days]
    n = len(pnls)
    if not n:
        return {}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_w, gross_l = sum(wins), abs(sum(losses))
    peak = eq = dd = 0.0
    for p in dp:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    mean = sum(dp) / len(dp)
    var = sum((p - mean) ** 2 for p in dp) / max(len(dp) - 1, 1)
    sharpe = mean / math.sqrt(var) * math.sqrt(252) if var > 0 else 0.0
    return {
        "trades": n, "days": len(dp),
        "wr": len(wins) / n * 100,
        "total": sum(pnls), "avg_day": mean,
        "pf": gross_w / gross_l if gross_l else float("inf"),
        "dd": dd, "sharpe": sharpe,
        "max_loss": min(pnls), "max_win": max(pnls),
    }


def premium_buckets(rows: list[dict]) -> None:
    # day's premium richness = T1 strangle credit (ATM-ish straddle at 14:55)
    t1_credit = {r["date"]: float(r["strangle_credit"]) for r in rows
                 if r["tranche"] == "T1" and r["strangle_credit"]}
    buckets = [(0, 6, "<$6"), (6, 8, "$6-8"), (8, 10, "$8-10"),
               (10, 14, "$10-14"), (14, 999, ">$14")]
    print("\n  Premium regime (day's T1 strangle credit ~ last-hour straddle):")
    print(f"    {'bucket':<8} {'days':>5} {'trades':>7} {'WR%':>6} {'total':>12} {'avg/day':>9}")
    for lo, hi, label in buckets:
        days = {d for d, c in t1_credit.items() if lo <= c < hi}
        sub = [r for r in rows if r["date"] in days]
        if not sub:
            continue
        s = stats(sub)
        print(f"    {label:<8} {len(days):>5} {s['trades']:>7} {s['wr']:>6.1f} "
              f"{s['total']:>+12,.0f} {s['total']/len(days):>+9,.0f}")


def monthly(rows: list[dict]) -> None:
    by_m = defaultdict(float)
    for r in rows:
        by_m[r["date"].strftime("%Y-%m")] += r["pnl"]
    print("\n  Monthly P&L (1-lot):")
    months = sorted(by_m)
    for i in range(0, len(months), 6):
        line = "    " + "  ".join(f"{m} {by_m[m]:>+8,.0f}" for m in months[i:i + 6])
        print(line)


def portfolio_sim(rows: list[dict], sizes=(8, 10, 10), start_equity=200_000.0) -> None:
    qty = {"T1": sizes[0], "T2": sizes[1], "T3": sizes[2]}
    days = defaultdict(float)
    for r in rows:
        days[r["date"]] += r["pnl"] * qty.get(r["tranche"], 1)
    series = sorted(days.items())
    eq = start_equity
    peak = eq
    max_dd_pct = 0.0
    curve = []
    for d, p in series:
        eq += p
        peak = max(peak, eq)
        max_dd_pct = max(max_dd_pct, (peak - eq) / peak * 100)
        curve.append((d, eq))
    yrs = (series[-1][0] - series[0][0]).days / 365.25
    cagr = ((eq / start_equity) ** (1 / yrs) - 1) * 100 if eq > 0 else -100.0
    print(f"\n  Portfolio sim (fixed T1={sizes[0]}/T2={sizes[1]}/T3={sizes[2]} contracts, "
          f"${start_equity:,.0f} start, no compounding of size):")
    print(f"    End equity: ${eq:>12,.0f}   Net: ${eq - start_equity:>+12,.0f}")
    print(f"    CAGR: {cagr:>6.1f}%   Max DD: {max_dd_pct:.1f}%")
    dp = [p for _, p in series]
    worst20 = min(sum(dp[i:i + 20]) for i in range(max(1, len(dp) - 19)))
    worst5 = min(sum(dp[i:i + 5]) for i in range(max(1, len(dp) - 4)))
    worst_day = min(dp)
    print(f"    Worst day: ${worst_day:>+12,.0f}   Worst 5d: ${worst5:>+12,.0f}   "
          f"Worst 20d: ${worst20:>+12,.0f}")


def main() -> None:
    print("=" * 88)
    print(f"  {'variant':<24} {'trades':>6} {'WR%':>6} {'total':>11} "
          f"{'avg/day':>8} {'PF':>6} {'Sharpe':>7} {'maxDD':>9} {'maxL':>7}")
    print("-" * 88)
    loaded = {}
    for name, fname in VARIANTS.items():
        p = RESULTS / fname
        if not p.exists():
            print(f"  {name:<24} (missing: {fname})")
            continue
        rows = load(p)
        loaded[name] = rows
        s = stats(rows)
        pf = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"
        print(f"  {name:<24} {s['trades']:>6} {s['wr']:>6.1f} {s['total']:>+11,.0f} "
              f"{s['avg_day']:>+8,.0f} {pf:>6} {s['sharpe']:>7.2f} "
              f"{s['dd']:>9,.0f} {s['max_loss']:>+7,.0f}")
    print("=" * 88)

    base = loaded.get("A trigger_plus (base)")
    if base:
        premium_buckets(base)
        monthly(base)
        portfolio_sim(base)

    overlay = loaded.get("C regime overlay")
    if overlay:
        print("\n  --- Version C (regime overlay) ---")
        premium_buckets(overlay)
        portfolio_sim(overlay)


if __name__ == "__main__":
    main()
