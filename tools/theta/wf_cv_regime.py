#!/usr/bin/env python3
"""Walk-forward CV of the VIX9D-regime hold-to-expiry MEIC.

Mirrors the project methodology: make_wf_folds(ratio=5, folds=4) expanding
walk-forward, objective = annualized Sharpe (scale-free; the framework default).
Per fold: optimize the regime config (size profile x move-filter k) on TRAIN,
then validate that exact config OOS on the next step. Reports param stability
across folds and OOS persistence vs the flat baseline — the overfitting gate.

em_mult fixed at 1.0 (the sweep showed strikes barely matter; sizing is the lever).
Tranche economics are priced once; size/filter are cheap post-multipliers, so the
grid x folds search is fast.

Usage: .venv/bin/python tools/theta/wf_cv_regime.py
"""
from __future__ import annotations
import csv
import math
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
sys.path.insert(0, "~/Dev/0dte_bot")
from zerodte.backtest.butterfly_core import make_wf_folds
from zerodte.backtest.vix_loader import load_vix_series, lookup_prior_close
from reconcile_meic import DatabentoCache, TRADES_CSV
from resim_exits import Settlement

SLIP, COMM, MULT, STEP, WING_FRAC, GATE = 0.10, 1.0, 100, 5, 0.5, 0.08
VIX = load_vix_series("~/Dev/0dte_bot/data/vix9d_daily.csv")


def regime(d):
    v = lookup_prior_close(VIX, d - timedelta(days=1))
    if v is None:
        return "mid"
    return "low" if v < 17 else ("mid" if v < 32 else "high")


def mid(q):
    return None if (not q or q[0] <= 0 or q[1] <= 0) else (q[0] + q[1]) / 2


def dislocated(q):
    if not q or q[0] <= 0 or q[1] <= 0:
        return False
    s = q[1] - q[0]
    return s > 1.0 and s > 1.5 * ((q[0] + q[1]) / 2)


def per_contract(src, settle, d, minute, short_k, long_k, is_call):
    """P&L per 1 contract of this hold-to-expiry credit spread (None if rejected)."""
    sq = src.get(d, minute, short_k, is_call); lq = src.get(d, minute, long_k, is_call)
    sm, lm = mid(sq), mid(lq)
    if sm is None or lm is None or dislocated(sq) or dislocated(lq):
        return None
    width = abs(long_k - short_k); credit = sm - lm
    if credit <= 0 or credit / width < GATE:
        return None
    F = settle.forward(d)
    if F is None:
        return None
    loss = min(max(0.0, (F - short_k) if is_call else (short_k - F)), width)
    return (credit - SLIP - loss) * MULT - 2 * COMM   # per-contract $ (qty=1)


def sharpe(daily):
    if len(daily) < 5:
        return 0.0
    sd = statistics.pstdev(daily)
    return 0.0 if sd == 0 else (statistics.mean(daily) / sd) * math.sqrt(252)


def metrics(tranches, dates, size_mult, filter_k):
    """Daily P&L over `dates`; returns dict of metrics for this config."""
    dset = {d.isoformat() for d in dates}
    day = defaultdict(float)
    for t in tranches:
        if t["ds"] not in dset or t["pc"] is None:
            continue
        sm = size_mult[t["reg"]]
        if sm <= 0:
            continue
        if filter_k is not None and t["ref_em"] and t["move"] > filter_k * t["ref_em"]:
            continue
        qty = max(1, round(t["base_q"] * sm))
        day[t["ds"]] += qty * t["pc"]
    series = [day.get(d.isoformat(), 0.0) for d in dates]   # 0 on no-trade days
    total = sum(series)
    traded = [v for v in series if v != 0]
    return dict(total=total, sharpe=sharpe(series), n_days=len(dates),
                worst=min(series) if series else 0,
                ndtrade=len(traded))


GRID = {
    "flat[1/1/1]":        {"low": 1, "mid": 1, "high": 1},
    "up-vol[.5/1/1.5]":   {"low": .5, "mid": 1, "high": 1.5},
    "favor-mid[.5/1.5/.5]": {"low": .5, "mid": 1.5, "high": .5},
    "de-risk-hi[.5/1.25/.4]": {"low": .5, "mid": 1.25, "high": .4},
    "skip-low[0/1.5/.5]": {"low": 0, "mid": 1.5, "high": .5},
}
KS = [None, 1.0, 0.75, 0.5]


def main():
    src = DatabentoCache(); settle = Settlement(src)
    rows = [r for r in csv.DictReader(open(TRADES_CSV)) if r["entry_time"]]
    rows.sort(key=lambda r: r["session_date"])

    # one record per tranche, priced once
    seen = {}; day_ref = {}
    for r in rows:
        ds, et = r["session_date"], r["entry_time"]
        if (ds, et) in seen:
            continue
        minute = et[11:16]; spot = float(r["underlying_at_entry"])
        d = date.fromisoformat(ds)
        src._load(d); q = src._q
        em = None
        if q:
            strikes = sorted({k[1] for k in q if k[0] == minute})
            if strikes:
                atm = min(strikes, key=lambda k: abs(k - spot))
                c = q.get((minute, atm, True)); p = q.get((minute, atm, False))
                if c and p:
                    em = (c[0]+c[1])/2 + (p[0]+p[1])/2
        seen[(ds, et)] = dict(ds=ds, reg=regime(d), base_q=int(r["qty"]),
                              spot=spot, em=em, minute=minute, d=d)
        if ds not in day_ref:
            day_ref[ds] = (spot, em)
    tranches = list(seen.values())
    # price each tranche (em_mult=1) + move metric
    for t in tranches:
        t["pc"] = None; t["move"] = 0.0; t["ref_em"] = day_ref[t["ds"]][1]
        if t["em"] is None:
            continue
        wing = max(STEP, round(WING_FRAC * t["em"] / STEP) * STEP)
        sc = math.ceil((t["spot"] + t["em"]) / STEP) * STEP
        sp = math.floor((t["spot"] - t["em"]) / STEP) * STEP
        call = per_contract(src, settle, t["d"], t["minute"], sc, sc+wing, True)
        put = per_contract(src, settle, t["d"], t["minute"], sp, sp-wing, False)
        t["pc"] = None if (call is None or put is None) else (call + put)
        t["move"] = abs(t["spot"] - day_ref[t["ds"]][0])

    all_dates = sorted({date.fromisoformat(t["ds"]) for t in tranches})
    folds = make_wf_folds(all_dates, ratio=5, folds=4)

    print("=" * 96)
    print("WALK-FORWARD CV — VIX9D-regime hold-to-expiry MEIC (ratio 5:1, 4 folds, obj=annualized Sharpe)")
    print("=" * 96)
    combos = [(f"{name} k={k}", sm, k) for name, sm in GRID.items() for k in KS]

    wf_oos_total = 0.0; wf_oos_sharpes = []; winners = []
    base_oos_total = 0.0; base_oos_sharpes = []
    filt_oos_total = 0.0; filt_oos_sharpes = []   # flat sizing + fixed move filter k=0.75
    for tr, val in folds:
        # optimize on train by Sharpe (min trade-days guard)
        scored = []
        for label, sm, k in combos:
            mt = metrics(tranches, tr, sm, k)
            if mt["ndtrade"] >= 20:
                scored.append((mt["sharpe"], label, sm, k, mt))
        scored.sort(reverse=True)
        _, best_label, best_sm, best_k, tr_m = scored[0]
        v = metrics(tranches, val, best_sm, best_k)
        b = metrics(tranches, val, GRID["flat[1/1/1]"], None)
        fk = metrics(tranches, val, GRID["flat[1/1/1]"], 0.75)
        wf_oos_total += v["total"]; wf_oos_sharpes.append(v["sharpe"]); winners.append(best_label)
        base_oos_total += b["total"]; base_oos_sharpes.append(b["sharpe"])
        filt_oos_total += fk["total"]; filt_oos_sharpes.append(fk["sharpe"])
        print(f"\nFold {tr[0]}..{tr[-1]} (train {len(tr)}d) -> val {val[0]}..{val[-1]} ({len(val)}d)")
        print(f"   train-best: {best_label:28s}  train Sharpe {tr_m['sharpe']:5.2f}  trainPnL ${tr_m['total']:>+10,.0f}")
        print(f"   OOS  selected: Sharpe {v['sharpe']:5.2f}  PnL ${v['total']:>+10,.0f}  worstday ${v['worst']:>+9,.0f}")
        print(f"   OOS  flat base: Sharpe {b['sharpe']:5.2f}  PnL ${b['total']:>+10,.0f}")

    nf = len(folds)
    print("\n" + "=" * 96)
    print("WF-CV VERDICT")
    print(f"   train-winner per fold : {winners}")
    print(f"   stable winner?        : {'YES — ' + winners[0] if len(set(winners))==1 else 'NO (varies): ' + str(set(winners))}")
    print(f"   OOS selected : mean Sharpe {statistics.mean(wf_oos_sharpes):5.2f}  total PnL ${wf_oos_total:>+12,.0f}  "
          f"profitable folds {sum(1 for s in wf_oos_sharpes if s>0)}/{nf}")
    print(f"   OOS flat base: mean Sharpe {statistics.mean(base_oos_sharpes):5.2f}  total PnL ${base_oos_total:>+12,.0f}  "
          f"profitable folds {sum(1 for s in base_oos_sharpes if s>0)}/{nf}")
    print(f"   OOS flat+k.75: mean Sharpe {statistics.mean(filt_oos_sharpes):5.2f}  total PnL ${filt_oos_total:>+12,.0f}  "
          f"profitable folds {sum(1 for s in filt_oos_sharpes if s>0)}/{nf}")
    print(f"   regime selection beats flat OOS? {'YES' if statistics.mean(wf_oos_sharpes)>statistics.mean(base_oos_sharpes) else 'NO'}")
    print(f"   move-filter-only beats flat OOS? {'YES' if statistics.mean(filt_oos_sharpes)>statistics.mean(base_oos_sharpes) else 'NO'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
