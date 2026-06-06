#!/usr/bin/env python3
"""Fixed-fractional sizing analysis for hold-to-expiry MEIC.

Daily return-on-capital r_day = P&L_day / BP_day (sizing-invariant: both scale
with qty). Under fixed-fractional sizing you deploy BP = f*equity each day, so
equity_{t+1} = equity_t * (1 + f * r_day). Because the ICs are defined-risk,
r_day >= -1, so the worst possible day is exactly -f (ruin impossible).

Block bootstrap (preserves bad-day clustering, e.g. 2025-04-08/09) estimates the
distribution of max drawdown and CAGR vs f, so f can be picked for a DD tolerance.
Scale-invariant in %, so it applies to ANY account size (subject to a contract-
granularity floor noted at the end).

Usage: .venv/bin/python tools/theta/sizing_montecarlo.py
"""
from __future__ import annotations
import csv
import statistics
from collections import defaultdict
from datetime import date
from reconcile_meic import DatabentoCache
from resim_exits import Settlement, sim

# deterministic LCG (Math.random is unavailable; keep it reproducible)
class RNG:
    def __init__(self, seed=12345):
        self.s = seed
    def rand(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.s >> 11) / float(1 << 53)
    def randint(self, n):
        return int(self.rand() * n) % n


TRADES_CSV = "~/Dev/0dte_bot/results/meic_spx/trades.csv"


def build_r_series():
    """Daily return-on-capital r = hold-to-expiry P&L / IC buying-power."""
    src = DatabentoCache(); settle = Settlement(src)
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])
    bytr = defaultdict(dict)
    for r in rows:
        bytr[(r["session_date"], r["entry_time"])][r["spread_type"]] = r
    pnl = defaultdict(float); bp = defaultdict(float)
    for (ds, et), legs in bytr.items():
        if "put" not in legs or "call" not in legs:
            continue
        p, c = legs["put"], legs["call"]
        qty = int(p["qty"])
        W = max(float(p["spread_width"]), float(c["spread_width"]))
        tot_credit = float(p["credit_received"]) + float(c["credit_received"])
        bp[ds] += max(0.0, W - tot_credit) * 100 * qty           # IC max loss (defined risk)
        for r in (p, c):
            v = sim(r, src, settle, keep_scaleout=False)
            if v is not None:
                pnl[ds] += v
    dates = sorted(d for d in bp if bp[d] > 0)
    return dates, [pnl[d] / bp[d] for d in dates]


def block_bootstrap(r, n_paths, block, rng):
    n = len(r)
    paths = []
    for _ in range(n_paths):
        seq = []
        while len(seq) < n:
            start = rng.randint(n)
            seq.extend(r[start:start + block])
        paths.append(seq[:n])
    return paths


def path_stats(r, f):
    eq = 1.0; peak = 1.0; mdd = 0.0
    for x in r:
        eq *= (1 + f * x)
        if eq <= 0:
            return -1.0, 1.0  # ruin
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    return mdd, eq


def main():
    dates, r = build_r_series()
    yrs = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    r_sorted = sorted(r)
    print("=" * 84)
    print("FIXED-FRACTIONAL SIZING — hold-to-expiry MEIC daily return-on-capital r=P&L/BP")
    print("=" * 84)
    print(f"  days={len(r)}  span={yrs:.1f}y   r: mean {statistics.mean(r):+.3f}  "
          f"median {statistics.median(r):+.3f}  min {min(r):+.3f}  "
          f"p1 {r_sorted[len(r)//100]:+.3f}  max {max(r):+.3f}")
    print(f"  worst day = {min(r)*100:+.0f}% of deployed BP  ->  at fraction f, worst day ~= f*{min(r):.2f}")
    print()
    rng = RNG()
    paths = block_bootstrap(r, 3000, 5, rng)  # 5-day blocks preserve clustering
    print(f"  Block-bootstrap (3000 paths, 5-day blocks), per fraction f of equity deployed as BP:")
    print(f"  {'f':>5} {'worst-day':>10} {'median CAGR':>12} {'med maxDD':>10} {'P95 maxDD':>10} {'P99 maxDD':>10} {'P(DD>50%)':>10}")
    for f in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        dds = []; cagrs = []
        for p in paths:
            mdd, eq = path_stats(p, f)
            dds.append(mdd); cagrs.append(eq ** (1 / yrs) - 1 if eq > 0 else -1)
        dds.sort()
        worst_day = f * min(r)
        pdd50 = sum(1 for d in dds if d < -0.50) / len(dds)
        print(f"  {f:>5.2f} {worst_day*100:>9.0f}% {statistics.median(cagrs)*100:>11.0f}% "
              f"{dds[len(dds)//2]*100:>9.0f}% {dds[int(.05*len(dds))]*100:>9.0f}% "
              f"{dds[int(.01*len(dds))]*100:>9.0f}% {pdd50*100:>9.0f}%")
    print("=" * 84)
    print("  worst-day is a hard guarantee (defined-risk); maxDD distribution is the")
    print("  sequence-risk picture from clustering multiple bad days.")


if __name__ == "__main__":
    main()
