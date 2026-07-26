#!/usr/bin/env python3
"""Skyline verdict analysis: per-year, claimant capital framing, fill breakeven,
settlement-vs-structure diagnostic, and controls."""
from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from skyline_core import payoff_at  # noqa: E402

FF = ["0.0", "0.25", "0.5"]
NAMES = {"0.0": "mid", "0.25": "mid+25%", "0.5": "half-spread"}


def load(name):
    with open(ROOT / "backtest_data" / name) as fh:
        rows = [r for r in csv.DictReader(fh)]
    for r in rows:
        for k in ("spot", "atm", "settle", "settle_off_atm", "payoff",
                  "sum_spread", "max_spread", "drop"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
        for f in FF:
            r[f"pnl_{f}"] = float(r[f"pnl_{f}"])
            r[f"debit_{f}"] = float(r[f"debit_{f}"])
        r["gated"] = r["gated"] == "True"
    return rows


def ann_sharpe(x):
    if len(x) < 2:
        return 0.0
    sd = st.pstdev(x)
    return (st.mean(x) / sd * math.sqrt(252)) if sd else 0.0


def maxdd(x):
    eq = peak = mdd = 0.0
    for p in x:
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd


def per_year(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r["date"][:4]].append(r[key])
    print(f"{'year':6} {'n':>5} {'total$':>10} {'$/day':>8} {'Sharpe':>7} "
          f"{'win%':>6} {'maxDD$':>9}")
    for y in sorted(by):
        v = by[y]
        print(f"{y:6} {len(v):5d} {sum(v):10,.0f} {st.mean(v):8.2f} "
              f"{ann_sharpe(v):7.2f} {100*sum(1 for p in v if p>0)/len(v):6.1f} "
              f"{maxdd(v):9,.0f}")


def capital_sim(rows, key, start_cap=5000.0, lots=2):
    """Claimant's framing: fixed contract count, additive P&L on a small account."""
    eq = start_cap
    peak = eq
    mdd_pct = 0.0
    mdd_abs = 0.0
    curve = []
    ruin_date = None
    for r in rows:
        eq += lots * r[key]
        if eq <= 0 and ruin_date is None:
            ruin_date = r["date"]
        peak = max(peak, eq)
        mdd_abs = max(mdd_abs, peak - eq)
        mdd_pct = max(mdd_pct, (peak - eq) / peak)
        curve.append((r["date"], eq))
    yrs = len(rows) / 252.0
    cagr = ((eq / start_cap) ** (1 / yrs) - 1) * 100 if eq > 0 else float("nan")
    return {"end": eq, "cagr": cagr, "mdd_pct": mdd_pct * 100,
            "mdd_abs": mdd_abs, "mar": cagr / (mdd_pct * 100) if mdd_pct else 0,
            "ruin": ruin_date, "curve": curve, "yrs": yrs}


def main():
    full = [r for r in load("sky_full_nofilter.csv") if not r["gated"]]
    repro = [r for r in load("sky_repro_nofilter.csv") if not r["gated"]]
    full_f = [r for r in load("sky_full_drop25.csv") if not r["gated"]]

    print("=" * 78)
    print("PER-YEAR, mid fills (the claimant's fill assumption), 1 lot, no filter")
    print("=" * 78)
    per_year(full, "pnl_0.0")
    print("\nPER-YEAR, mid+25% of spread (realistic marketable-limit fill), 1 lot")
    per_year(full, "pnl_0.25")

    print("\n" + "=" * 78)
    print("CLAIMANT'S CAPITAL FRAMING  ($5,000 start, 2 contracts, additive)")
    print("=" * 78)
    print(f"{'window':26} {'fill':12} {'end$':>10} {'CAGR%':>8} {'maxDD%':>8} "
          f"{'MAR':>6} {'ruined':>11}")
    for label, rows in (("claimed ~2yr (24-06>26-06)", repro),
                        ("full cycle (22-05>26-06)", full)):
        for f in FF:
            c = capital_sim(rows, f"pnl_{f}")
            print(f"{label:26} {NAMES[f]:12} {c['end']:10,.0f} {c['cagr']:8.1f} "
                  f"{c['mdd_pct']:8.1f} {c['mar']:6.2f} {str(c['ruin'] or '-'):>11}")
    print("\n  video claim:  end $78,956 | CAGR 297% | MAR 5.5 | win 53%")

    print("\n" + "=" * 78)
    print("FILL BREAKEVEN — where does the edge vanish?")
    print("=" * 78)
    for label, rows in (("claimed 2yr", repro), ("full cycle", full)):
        ss = st.median([r["sum_spread"] for r in rows])
        m0 = st.mean([r["pnl_0.0"] for r in rows])
        m5 = st.mean([r["pnl_0.5"] for r in rows])
        be = m0 / (m0 - m5) * 0.5 if m0 != m5 else float("nan")
        print(f"  {label:12} mid=${m0:+7.2f}/day  half-spread=${m5:+7.2f}/day  "
              f"median total quoted spread={ss:.2f}pt (=${100*ss:.0f}/lot)")
        print(f"  {'':12} breakeven at {be*100:.0f}% of the half-spread "
              f"(i.e. you must fill {be*100:.0f}% of the way from mid to touch, or better)")

    print("\n" + "=" * 78)
    print("WHY: where settlement lands vs the structure (1 lot, full sample)")
    print("=" * 78)
    buckets = [(-10_000, -80, "below range (loss)"), (-80, -70, "-80..-70"),
               (-70, -50, "-70..-50 (peak -60)"), (-50, -30, "-50..-30"),
               (-30, -10, "-30..-10 (peak -20)"), (-10, 10, "-10..+10 ATM VALLEY"),
               (10, 35, "+10..+35 (peak +25)"), (35, 50, "+35..+50"),
               (50, 70, "+50..+70 (peak +60)"), (70, 75, "+70..+75"),
               (75, 10_000, "above range (loss)")]
    print(f"{'settle - ATM':24} {'days':>5} {'%':>6} {'avg payoff$':>12} "
          f"{'avg pnl@mid$':>13}")
    for lo, hi, nm in buckets:
        sel = [r for r in full if lo <= r["settle_off_atm"] < hi]
        if not sel:
            continue
        print(f"{nm:24} {len(sel):5d} {100*len(sel)/len(full):6.1f} "
              f"{st.mean([r['payoff'] for r in sel]):12,.0f} "
              f"{st.mean([r['pnl_0.0'] for r in sel]):13,.0f}")
    absmove = [abs(r["settle_off_atm"]) for r in full]
    print(f"\n  |settle-ATM|: median={st.median(absmove):.1f}pt  "
          f"mean={st.mean(absmove):.1f}pt  "
          f"P(|move|<10)={100*sum(1 for m in absmove if m<10)/len(absmove):.1f}%  "
          f"P(|move|>80)={100*sum(1 for m in absmove if m>80)/len(absmove):.1f}%")
    print(f"  structure needs the close to land in a ~20pt-wide band around one of "
          f"4 peaks and avoid ATM+/-10 and |move|>80")

    print("\n" + "=" * 78)
    print("CONTROL: does the 25pt overnight-drop filter add anything? (full cycle)")
    print("=" * 78)
    for f in FF:
        a = [r[f"pnl_{f}"] for r in full]
        b = [r[f"pnl_{f}"] for r in full_f]
        # Welch t on difference of means vs unconditional
        ma, mb = st.mean(a), st.mean(b)
        sa, sb = st.pstdev(a) / math.sqrt(len(a)), st.pstdev(b) / math.sqrt(len(b))
        t = (mb - ma) / math.sqrt(sa ** 2 + sb ** 2)
        print(f"  {NAMES[f]:12} unconditional ${ma:+7.2f}/day (n={len(a)})   "
              f"filtered ${mb:+7.2f}/day (n={len(b)})   diff t={t:+.2f}")
    print("  (|t| < 2 => the filter is not distinguishable from noise)")


if __name__ == "__main__":
    main()
