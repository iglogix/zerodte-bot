"""Overlay Pulver's broken-wing-butterfly DEFENSE on the as-taught IC trades.

Per the video screenshot, the defense is mechanically specifiable relative to
the IC short strikes (Kc = short call, Kp = short put):
  Call BWB:  +1 (Kc-5)  -2 (Kc+5)  +1 (Kc+25)
  Put  BWB:  +1 (Kp+5)  -2 (Kp-5)  +1 (Kp-25)
entered ~15 min after the IC (09:50 ET), held to 15:59 settlement.

Long-fly mark M = price(low) - 2*price(mid) + price(high). Defense P&L =
(M_exit - M_entry)*100 - slippage*100 - 3*commission, per side. Skips a side
if any leg is unquoted that day. Reports IC-only vs IC+defense daily-P&L stats.

Usage: python -m scripts.overlay_defense_bwb [results/open_ic_pulver/trades.csv]
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_v2.backtest.cbbo_store import CbboStore

ENTRY_MIN = "09:50"   # IC at 09:35 + ~15 min (Pulver adds defense 5-15 min later)
EXIT_MIN = "15:59"
SLIP = 0.10
COMM = 1.0
MULT = 100


def mid(store, minute, strike, is_call):
    q = store.lookup(minute, float(strike), is_call)
    if q is None:
        return None
    return (q[0] + q[1]) / 2


def bwb_mark(store, minute, low, mid_k, high, is_call):
    a = mid(store, minute, low, is_call)
    b = mid(store, minute, mid_k, is_call)
    c = mid(store, minute, high, is_call)
    if a is None or b is None or c is None:
        return None
    return a - 2 * b + c


def defense_pnl(store, minute_in, minute_out, low, mid_k, high, is_call):
    m_in = bwb_mark(store, minute_in, low, mid_k, high, is_call)
    m_out = bwb_mark(store, minute_out, low, mid_k, high, is_call)
    if m_in is None or m_out is None:
        return None  # unquoted -> no defense this side
    return (m_out - m_in) * MULT - SLIP * MULT - 3 * COMM


def stats(pnls, label):
    n = len(pnls)
    total = sum(pnls)
    mean = total / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0
    sharpe = (mean / sd * math.sqrt(252)) if sd > 0 else 0
    wins = sum(1 for p in pnls if p > 0)
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    pf = gw / gl if gl > 0 else float("inf")
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    print(f"  {label:<22} n={n:>4} Sharpe {sharpe:>5.2f}  PnL ${total:>+8.0f}  "
          f"WR {wins/n*100:4.1f}%  PF {pf:4.2f}  avgW +${(gw/wins if wins else 0):>4.0f} "
          f"avgL -${(gl/(n-wins) if n-wins else 0):>4.0f}  maxDD ${dd:>6.0f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/open_ic_pulver/trades.csv"
    rows = list(csv.DictReader(open(path)))
    # group by day -> short call / short put strikes + IC pnl
    by_day = {}
    for r in rows:
        if r.get("realized_pnl_usd") in ("", None):
            continue
        d = date.fromisoformat(r["session_date"])
        rec = by_day.setdefault(d, {"ic": 0.0, "Kc": None, "Kp": None})
        rec["ic"] += float(r["realized_pnl_usd"])
        if r["spread_type"] == "call_credit_spread":
            rec["Kc"] = float(r["short_strike"])
        elif r["spread_type"] == "put_credit_spread":
            rec["Kp"] = float(r["short_strike"])

    days = sorted(d for d, v in by_day.items() if v["Kc"] and v["Kp"])
    start, end = days[0], days[-1]
    print(f"Preloading CBBO {start}..{end} ...")
    mem = CbboStore.preload_cache("data/cbbo_cache", start=start, end=end)
    store = CbboStore("data/cbbo_cache", preloaded=mem)

    ic_only, ic_def = [], []
    def_call_n = def_put_n = both_missing = 0
    for d in days:
        rec = by_day[d]
        store.load_day(d)
        Kc, Kp = rec["Kc"], rec["Kp"]
        dc = defense_pnl(store, ENTRY_MIN, EXIT_MIN, Kc - 5, Kc + 5, Kc + 25, True)
        dp = defense_pnl(store, ENTRY_MIN, EXIT_MIN, Kp - 25, Kp - 5, Kp + 5, False)
        def_total = 0.0
        if dc is not None:
            def_total += dc
            def_call_n += 1
        if dp is not None:
            def_total += dp
            def_put_n += 1
        if dc is None and dp is None:
            both_missing += 1
        ic_only.append(rec["ic"])
        ic_def.append(rec["ic"] + def_total)

    print(f"  days={len(days)}  call-BWB quoted {def_call_n}, put-BWB quoted "
          f"{def_put_n}, both-missing {both_missing}\n")
    stats(ic_only, "IC only (as-taught)")
    stats(ic_def, "IC + BWB defense")
    dtot = sum(ic_def) - sum(ic_only)
    print(f"\n  defense net P&L contribution: ${dtot:+,.0f} over {len(days)} days "
          f"(${dtot/len(days):+.1f}/day)")


if __name__ == "__main__":
    main()
