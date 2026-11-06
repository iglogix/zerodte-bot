#!/usr/bin/env python3
"""VIX9D-regime-dependent MEIC: adjust SIZING and IC STRIKES by prior-day VIX9D
regime, on the winning hold-to-expiry (no stop, no scale-out) exit.

Regime (mirrors butterfly bands, prior-day VIX9D close, no lookahead):
    low <17 | mid 17-32 | high 32+

Strikes are re-selected from the chain (so em_mult can vary by regime):
    short = spot +/- em_mult[reg]*EM (snapped to step), wing = wing_frac*EM
    credit = spread mid from cbbo; credit/width gate 0.08; dislocation guard.
Sizing: base tranche qty * size_mult[reg] (rounded, 0 => skip the regime).
Exit: hold to cash settlement at ATM-parity forward intrinsic (capped at width).

Usage: .venv/bin/python tools/theta/resim_regime.py
"""
from __future__ import annotations
import csv
import math
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
sys.path.insert(0, "~/Dev/0dte_bot")
from zerodte.backtest.vix_loader import load_vix_series, lookup_prior_close
from reconcile_meic import DatabentoCache, TRADES_CSV
from resim_exits import Settlement

SLIP, COMM, MULT, STEP, WING_FRAC, GATE = 0.10, 1.0, 100, 5, 0.5, 0.08
VIX = load_vix_series("~/Dev/0dte_bot/data/vix9d_daily.csv")


def regime(d):
    v = lookup_prior_close(VIX, d - timedelta(days=1))
    if v is None:
        return "mid"           # default to mid if missing
    return "low" if v < 17 else ("mid" if v < 32 else "high")


def em_at(src, d, minute, spot):
    src._load(d)
    q = src._q
    if not q:
        return None
    strikes = sorted({k[1] for k in q if k[0] == minute})
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - spot))
    c = q.get((minute, atm, True)); p = q.get((minute, atm, False))
    if not c or not p:
        return None
    return (c[0] + c[1]) / 2 + (p[0] + p[1]) / 2


def mid(q):
    return None if (not q or q[0] <= 0 or q[1] <= 0) else (q[0] + q[1]) / 2


def dislocated(q):
    if not q or q[0] <= 0 or q[1] <= 0:
        return False
    s = q[1] - q[0]
    return s > 1.0 and s > 1.5 * ((q[0] + q[1]) / 2)


def side_pnl(src, settle, d, minute, short_k, long_k, is_call, qty):
    """Hold-to-expiry P&L for one credit spread; None if unpriceable/rejected."""
    sq = src.get(d, minute, short_k, is_call)
    lq = src.get(d, minute, long_k, is_call)
    sm, lm = mid(sq), mid(lq)
    if sm is None or lm is None:
        return None
    if dislocated(sq) or dislocated(lq):
        return None
    width = abs(long_k - short_k)
    credit = sm - lm
    if credit <= 0 or credit / width < GATE:
        return None
    F = settle.forward(d)
    if F is None:
        return None
    intrinsic = max(0.0, (F - short_k) if is_call else (short_k - F))
    loss = min(intrinsic, width)
    return (credit - SLIP - loss) * MULT * qty - 2 * qty * COMM


def run(scn, tranches, src, settle, day_ref):
    em_mult, size_mult = scn["em_mult"], scn["size_mult"]
    filter_k = scn.get("filter_k")
    sess = defaultdict(float); yr = defaultdict(float); byreg = defaultdict(float)
    pnl = []; entered = 0
    for (ds, et), info in tranches.items():
        d = date.fromisoformat(ds)
        reg = regime(d)
        sm = size_mult[reg]
        if sm <= 0:
            continue
        spot, em, minute, base_q = info
        if em is None:
            continue
        # intraday move-so-far filter (no lookahead)
        if filter_k is not None:
            ref_spot, ref_em = day_ref[ds]
            if ref_em and abs(spot - ref_spot) > filter_k * ref_em:
                continue
        qty = max(1, round(base_q * sm))
        m = em_mult[reg]
        wing = max(STEP, round(WING_FRAC * em / STEP) * STEP)
        sc = math.ceil((spot + m * em) / STEP) * STEP
        sp = math.floor((spot - m * em) / STEP) * STEP
        call = side_pnl(src, settle, d, minute, sc, sc + wing, True, qty)
        put = side_pnl(src, settle, d, minute, sp, sp - wing, False, qty)
        if call is None or put is None:        # all-or-nothing per tranche (engine behaviour)
            continue
        entered += 1
        for v in (call, put):
            pnl.append(v); sess[ds] += v; yr[ds[:4]] += v; byreg[reg] += v
    dates = sorted(sess); eq = peak = mdd = 0
    for d in dates:
        eq += sess[d]; peak = max(peak, eq); mdd = min(mdd, eq - peak)
    worst = min(sess.values()) if sess else 0
    return dict(total=sum(pnl), n=len(pnl), entered=entered, worst=worst, mdd=mdd,
                yr=yr, byreg=byreg, bad50=sum(1 for v in sess.values() if v < -50000),
                sessions=len(dates))


def main():
    src = DatabentoCache(); settle = Settlement(src)
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])
    # one entry per tranche: (date, entry_time) -> (spot, EM, minute, base_qty)
    tranches = {}
    for r in rows:
        key = (r["session_date"], r["entry_time"])
        if key in tranches:
            continue
        minute = r["entry_time"][11:16]
        spot = float(r["underlying_at_entry"])
        d = date.fromisoformat(r["session_date"])
        tranches[key] = (spot, em_at(src, d, minute, spot), minute, int(r["qty"]))

    # per-day reference (earliest tranche spot + its EM) for the move filter
    day_ref = {}
    for (ds, et), (spot, em, minute, q) in sorted(tranches.items()):
        if ds not in day_ref:
            day_ref[ds] = (spot, em)

    F = lambda lo, md, hi: {"low": lo, "mid": md, "high": hi}
    scenarios = {
        "baseline (m=1, size=1)":          dict(em_mult=F(1,1,1),       size_mult=F(1,1,1)),
        "size: up-vol [.5/1/1.5]":         dict(em_mult=F(1,1,1),       size_mult=F(0.5,1,1.5)),
        "size: favor-mid [.5/1.5/.5]":     dict(em_mult=F(1,1,1),       size_mult=F(0.5,1.5,0.5)),
        "size: de-risk-hi [.5/1.25/.4]":   dict(em_mult=F(1,1,1),       size_mult=F(0.5,1.25,0.4)),
        "favor-mid + filter k0.75":        dict(em_mult=F(1,1,1),       size_mult=F(0.5,1.5,0.5), filter_k=0.75),
        "favor-mid + strk hi1.25 + filt k0.75": dict(em_mult=F(1,1,1.25), size_mult=F(0.5,1.5,0.5), filter_k=0.75),
        "de-risk-hi + filter k0.5":        dict(em_mult=F(1,1,1),       size_mult=F(0.5,1.25,0.4), filter_k=0.5),
    }
    print("=" * 100)
    print("VIX9D-REGIME MEIC (hold-to-expiry, no stop/scale). baselines: flat hold=+$1.45M")
    print("=" * 100)
    print(f"{'scenario':40s} {'total P&L':>12} {'worst day':>11} {'max DD':>11} {'tranches':>9} {'d<-50k':>7}")
    print("-" * 100)
    detail = {}
    for name, scn in scenarios.items():
        r = run(scn, tranches, src, settle, day_ref); detail[name] = r
        print(f"{name:40s} {r['total']:>12,.0f} {r['worst']:>11,.0f} {r['mdd']:>11,.0f} "
              f"{r['entered']:>9} {r['bad50']:>7}")
    print("-" * 100)
    print("\nby-year (P&L):")
    for name in scenarios:
        r = detail[name]
        print(f"  {name:40s} " + "  ".join(f"{y}:{r['yr'][y]/1000:+.0f}k" for y in sorted(r['yr'])))
    print("=" * 100)


if __name__ == "__main__":
    main()
