#!/usr/bin/env python3
"""Skyline conditioning study: is there a market-context / event filter that
rescues it, and which regimes are worst?

Enriches the per-day results with 09:45 implied move (ATM straddle), VIX family,
overnight gap, opening drive, day-of-week and event flags, then reports
univariate conditional P&L at both the claimant's fill (mid) and a realistic one.
"""
from __future__ import annotations

import csv
import math
import pickle
import statistics as st
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_skyline import lookup, parity_spot, CACHE  # noqa: E402
from zerodte.backtest.vix_loader import load_vix_series  # noqa: E402
from economic_calendar import get_events_on_date  # noqa: E402

FEAT = ROOT / "backtest_data" / "skyline_features.pkl"


def atm_straddle(q, minute, strikes, spot):
    k = min(strikes, key=lambda x: abs(x - spot))
    c = lookup(q, minute, k, True)
    p = lookup(q, minute, k, False)
    if not c or not p or c[1] <= 0 or p[1] <= 0:
        return None
    return (c[0] + c[1]) / 2 + (p[0] + p[1]) / 2


def build():
    rows = {r["date"]: r for r in
            csv.DictReader(open(ROOT / "backtest_data" / "sky_full_nofilter.csv"))}
    vix = load_vix_series(ROOT / "data/vix_daily.csv")
    vix1d = load_vix_series(ROOT / "data/vix1d_daily.csv")
    vix9d = load_vix_series(ROOT / "data/vix9d_daily.csv")

    out = []
    prev_settle = None
    for f in sorted(CACHE.glob("*.pkl")):
        d = date.fromisoformat(f.stem)
        r = rows.get(d.isoformat())
        if r is None:
            prev_settle = None
            continue
        q = pickle.load(open(f, "rb"))
        ks = sorted({k[1] for k in q})
        spot = float(r["spot"])
        em = atm_straddle(q, "09:45", ks, spot)
        s931 = parity_spot(q, "09:32", ks)
        ev = get_events_on_date(d)
        rec = {
            "date": d, "spot": spot, "atm": float(r["atm"]),
            "settle": float(r["settle"]), "off": float(r["settle_off_atm"]),
            "em": em,
            "gap": float(r["drop"]) if r["drop"] not in ("", "None") else None,
            "drive": (spot - s931) if s931 else None,
            "sum_spread": float(r["sum_spread"]),
            "vix": vix.get(d), "vix1d": vix1d.get(d), "vix9d": vix9d.get(d),
            "dow": d.weekday(), "events": ev,
            "opex": d.weekday() == 4 and 15 <= d.day <= 21,
            "prev_move": None if prev_settle is None else abs(spot - prev_settle),
            "mid": float(r["pnl_0.0"]), "q25": float(r["pnl_0.25"]),
            "half": float(r["pnl_0.5"]),
        }
        out.append(rec)
        prev_settle = float(r["settle"])
    pickle.dump(out, open(FEAT, "wb"))
    return out


def bucket_report(rows, key, label, nb=5, fn=None):
    vals = [(fn(r) if fn else r[key]) for r in rows]
    pairs = [(v, r) for v, r in zip(vals, rows) if v is not None]
    if len(pairs) < 100:
        print(f"  {label}: insufficient data ({len(pairs)})")
        return
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    print(f"\n  {label}  (quintiles, n={n})")
    print(f"    {'range':>20} {'n':>5} {'$/day mid':>11} {'$/day +25%':>11} "
          f"{'win%':>6} {'spread$':>8}")
    for i in range(nb):
        chunk = pairs[i * n // nb:(i + 1) * n // nb]
        v = [c[1] for c in chunk]
        lo, hi = chunk[0][0], chunk[-1][0]
        print(f"    {f'{lo:8.2f}..{hi:8.2f}':>20} {len(v):5d} "
              f"{st.mean([x['mid'] for x in v]):11.2f} "
              f"{st.mean([x['q25'] for x in v]):11.2f} "
              f"{100*sum(1 for x in v if x['q25']>0)/len(v):6.1f} "
              f"{100*st.mean([x['sum_spread'] for x in v]):8.0f}")


def main():
    rows = pickle.load(open(FEAT, "rb")) if FEAT.exists() else build()
    print(f"feature table: {len(rows)} days {rows[0]['date']}..{rows[-1]['date']}")

    cost25 = st.mean([0.25 * r["sum_spread"] * 100 for r in rows])
    cost50 = st.mean([0.50 * r["sum_spread"] * 100 for r in rows])
    print(f"\nTHE BAR: mean spread cost is ${cost25:.0f}/lot at mid+25% and "
          f"${cost50:.0f}/lot at half-spread.")
    print(f"  A filter must lift the MID edge above those to be worth anything.")
    print(f"  Unconditional mid edge = ${st.mean([r['mid'] for r in rows]):.2f}/day.")

    print("\n" + "=" * 78)
    print("MARKET CONTEXT — univariate quintiles")
    print("=" * 78)
    bucket_report(rows, "em", "09:45 ATM straddle (implied move to close, pts)")
    bucket_report(rows, None, "structure half-range / implied move (80 / EM)",
                  fn=lambda r: 80.0 / r["em"] if r["em"] else None)
    bucket_report(rows, "vix1d", "VIX1D (1-day implied vol)")
    bucket_report(rows, "vix", "VIX")
    bucket_report(rows, None, "|overnight gap| (pts)",
                  fn=lambda r: abs(r["gap"]) if r["gap"] is not None else None)
    bucket_report(rows, None, "signed overnight gap (pts)", fn=lambda r: r["gap"])
    bucket_report(rows, None, "|opening drive 09:32->09:45| (pts)",
                  fn=lambda r: abs(r["drive"]) if r["drive"] is not None else None)
    bucket_report(rows, None, "prior-day |move| (pts)", fn=lambda r: r["prev_move"])
    bucket_report(rows, "sum_spread", "quoted spread across the 16 legs (pts)")

    print("\n" + "=" * 78)
    print("EVENTS AND CALENDAR")
    print("=" * 78)
    groups = defaultdict(list)
    for r in rows:
        for e in (r["events"] or ["(none)"]):
            groups[e].append(r)
        groups["OPEX" if r["opex"] else "non-OPEX"].append(r)
        groups[["Mon", "Tue", "Wed", "Thu", "Fri"][r["dow"]]].append(r)
    print(f"  {'group':14} {'n':>5} {'$/day mid':>11} {'$/day +25%':>11} "
          f"{'win%':>6} {'|settle-ATM|':>13}")
    order = ["(none)", "FOMC", "CPI", "NFP", "PPI", "GDP", "OPEX", "non-OPEX",
             "Mon", "Tue", "Wed", "Thu", "Fri"]
    for g in order:
        v = groups.get(g)
        if not v or len(v) < 15:
            continue
        print(f"  {g:14} {len(v):5d} {st.mean([x['mid'] for x in v]):11.2f} "
              f"{st.mean([x['q25'] for x in v]):11.2f} "
              f"{100*sum(1 for x in v if x['q25']>0)/len(v):6.1f} "
              f"{st.mean([abs(x['off']) for x in v]):13.1f}")

    print("\n" + "=" * 78)
    print("MECHANISM — realised move in units of the day's implied move")
    print("=" * 78)
    print(f"  {'|settle-ATM| / EM':>20} {'n':>5} {'%':>6} {'$/day mid':>11}")
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.2), (1.2, 9)):
        v = [r for r in rows if r["em"] and lo <= abs(r["off"]) / r["em"] < hi]
        if v:
            print(f"  {f'{lo:.1f} - {hi:.1f}':>20} {len(v):5d} "
                  f"{100*len(v)/len(rows):6.1f} {st.mean([x['mid'] for x in v]):11.2f}")
    ems = [r["em"] for r in rows if r["em"]]
    print(f"\n  EM (ATM straddle at 09:45): median {st.median(ems):.1f}pt  "
          f"p10 {sorted(ems)[len(ems)//10]:.1f}  p90 {sorted(ems)[9*len(ems)//10]:.1f}")
    print(f"  Structure geometry is FIXED (peaks 20/25/60, bounds 80) while EM "
          f"ranges {sorted(ems)[len(ems)//10]:.0f}-{sorted(ems)[9*len(ems)//10]:.0f}pt "
          f"-> the peaks sit at a different quantile of the move distribution every day.")


if __name__ == "__main__":
    main()
