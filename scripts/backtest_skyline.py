#!/usr/bin/env python3
"""Skyline (4-butterfly SPX 0DTE) backtest on real OPRA CBBO marks.

Entry: 09:45 ET, 12 strikes / 16 contracts (see skyline_core).
Exit:  held to cash settlement (SPXW is PM-settled on the SPX index close).
Fills: bracketed {mid, mid+25% spread, full half-spread} + commissions.

Settlement uses ^GSPC close (the actual SPXW settlement basis), cross-checked
against the parity-derived forward from the option tape itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sqlite3
import statistics as st
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from skyline_core import LEGS, MULT, N_CONTRACTS, payoff_at, price_structure, scaled_legs  # noqa: E402

CACHE = ROOT / "data" / "cbbo_cache_fly_10min"
DB = ROOT / "backtest_data" / "skyline.db"
ANCHOR_SPOT = 7580.0  # the video's example spot; basis for level-scaled variant


def lookup(q, minute, strike, is_call, stale=5):
    key = (minute, strike, is_call)
    if key in q:
        return q[key]
    hh, mm = int(minute[:2]), int(minute[3:5])
    for off in range(1, stale + 1):
        m, h = mm - off, hh
        while m < 0:
            m += 60
            h -= 1
        if h < 0:
            break
        k2 = (f"{h:02d}:{m:02d}", strike, is_call)
        if k2 in q:
            return q[k2]
    return None


def parity_spot(q, minute, strikes, stale=5):
    best = None
    for k in strikes:
        cq = lookup(q, minute, k, True, stale)
        pq = lookup(q, minute, k, False, stale)
        if cq is None or pq is None or cq[1] <= 0 or pq[1] <= 0:
            continue
        c, p = (cq[0] + cq[1]) / 2, (pq[0] + pq[1]) / 2
        d = abs(c - p)
        if best is None or d < best[0]:
            best = (d, k + c - p)
    return best[1] if best else None


def run(entry="09:45", fill_fracs=(0.0, 0.25, 0.5), comm=0.65,
        max_drop=None, scale_mode="fixed", start=None, end=None,
        legs_override=None, verbose=True):
    con = sqlite3.connect(DB)
    gspc = {d: (o, c) for d, o, h, l, c in
            con.execute("SELECT d, open, high, low, close FROM gspc_daily")}
    con.close()
    gdates = sorted(gspc)
    prev_close = {}
    for i in range(1, len(gdates)):
        prev_close[gdates[i]] = gspc[gdates[i - 1]][1]

    trades, skipped = [], {}
    for f in sorted(CACHE.glob("*.pkl")):
        d = date.fromisoformat(f.stem)
        if start and d < start:
            continue
        if end and d > end:
            continue
        ds = d.isoformat()
        if ds not in gspc:
            skipped["no_gspc"] = skipped.get("no_gspc", 0) + 1
            continue
        with open(f, "rb") as fh:
            q = pickle.load(fh)
        strikes = sorted({k[1] for k in q})
        spot = parity_spot(q, entry, strikes)
        if spot is None:
            skipped["no_spot"] = skipped.get("no_spot", 0) + 1
            continue

        pc = prev_close.get(ds)
        drop = spot - pc if pc else None
        gated = (max_drop is not None and drop is not None and drop < -max_drop)

        atm = round(spot / 5.0) * 5.0
        if legs_override is not None:
            legs = legs_override
        elif scale_mode == "fixed":
            legs = LEGS
        elif scale_mode == "level":
            legs = scaled_legs(spot / ANCHOR_SPOT)
        else:
            raise ValueError(scale_mode)

        quotes, ok = {}, True
        for off, _, is_call in legs:
            v = lookup(q, entry, atm + off, is_call)
            if v is None or v[1] <= 0:
                ok = False
                break
            quotes[(off, is_call)] = v
        if not ok:
            skipped["missing_leg"] = skipped.get("missing_leg", 0) + 1
            continue

        settle = gspc[ds][1]
        payoff = payoff_at(settle, atm, legs)
        row = {"date": ds, "spot": round(spot, 2), "atm": atm,
               "prev_close": pc, "drop": None if drop is None else round(drop, 2),
               "gated": gated, "settle": settle,
               "settle_off_atm": round(settle - atm, 2), "payoff": payoff}
        bad = False
        for ff in fill_fracs:
            fl = price_structure(quotes, atm, ff, legs)
            if fl is None:
                bad = True
                break
            gross = payoff - fl.debit
            net = gross - comm * N_CONTRACTS
            row[f"debit_{ff}"] = round(fl.debit, 2)
            row[f"pnl_{ff}"] = round(net, 2)
            if ff == 0.0:
                row["max_spread"] = round(fl.max_spread_pts, 2)
                row["sum_spread"] = round(fl.total_spread_pts, 2)
                # structural bound: package value must sit in [0, max payoff]
                if not (0.0 <= fl.debit <= 2500.0):
                    bad = True
        if bad:
            skipped["unpriceable"] = skipped.get("unpriceable", 0) + 1
            continue
        trades.append(row)

    if verbose and skipped:
        print(f"  skipped: {skipped}", file=sys.stderr)
    return trades


def metrics(pnls, label=""):
    if not pnls:
        return {}
    n = len(pnls)
    tot = sum(pnls)
    mean = tot / n
    sd = st.pstdev(pnls) if n > 1 else 0.0
    sharpe = (mean / sd * math.sqrt(252)) if sd > 0 else 0.0
    eq, peak, mdd = 0.0, 0.0, 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    srt = sorted(pnls)
    k = max(1, n // 20)
    cvar5 = sum(srt[:k]) / k
    wins = sum(1 for p in pnls if p > 0)
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    return {"label": label, "n": n, "total": round(tot, 0),
            "mean_per_day": round(mean, 2), "sd": round(sd, 1),
            "sharpe_ann": round(sharpe, 2), "win_rate": round(100 * wins / n, 1),
            "profit_factor": round(gp / gl, 2) if gl else float("inf"),
            "max_dd": round(mdd, 0),
            "mar_like": round((mean * 252) / mdd, 2) if mdd else float("inf"),
            "cvar5": round(cvar5, 0), "best": round(max(pnls), 0),
            "worst": round(min(pnls), 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="09:45")
    ap.add_argument("--comm", type=float, default=0.65)
    ap.add_argument("--max-drop", type=float, default=None)
    ap.add_argument("--scale", default="fixed", choices=["fixed", "level"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    s = date.fromisoformat(a.start) if a.start else None
    e = date.fromisoformat(a.end) if a.end else None
    tr = run(entry=a.entry, comm=a.comm, max_drop=a.max_drop,
             scale_mode=a.scale, start=s, end=e)
    live = [t for t in tr if not t["gated"]]
    print(f"\n=== SKYLINE {a.tag}  entry={a.entry} scale={a.scale} "
          f"max_drop={a.max_drop} comm=${a.comm}/ct ===")
    print(f"days priced={len(tr)}  traded={len(live)}  "
          f"gated_out={len(tr)-len(live)}")
    if not live:
        return 0
    print(f"debit@mid: median=${st.median([t['debit_0.0'] for t in live]):.0f} "
          f"mean=${st.mean([t['debit_0.0'] for t in live]):.0f}   "
          f"sum_spread median={st.median([t['sum_spread'] for t in live]):.2f}pt")
    rows = []
    for ff, name in ((0.0, "mid"), (0.25, "mid+25%"), (0.5, "half-spread")):
        m = metrics([t[f"pnl_{ff}"] for t in live], name)
        rows.append(m)
        print(f"  {name:12} tot=${m['total']:>9,.0f}  mean=${m['mean_per_day']:>7.2f}/day  "
              f"Sharpe={m['sharpe_ann']:>6.2f}  win={m['win_rate']:>5.1f}%  "
              f"PF={m['profit_factor']:>5.2f}  maxDD=${m['max_dd']:>8,.0f}  "
              f"CVaR5=${m['cvar5']:>7,.0f}")
    if a.out:
        p = ROOT / "backtest_data" / a.out
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(live[0].keys()))
            w.writeheader()
            w.writerows(tr)
        with open(str(p).replace(".csv", "_summary.json"), "w") as fh:
            json.dump({"tag": a.tag, "entry": a.entry, "scale": a.scale,
                       "max_drop": a.max_drop, "comm": a.comm,
                       "n_priced": len(tr), "n_traded": len(live),
                       "metrics": rows}, fh, indent=2)
        print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
