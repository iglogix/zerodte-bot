"""
VP-MEIC — multi-entry iron condor where EVERY tranche is anchored to the
DEVELOPING volume-area (FRVP from 09:30 up to that tranche's entry), vs the
no-VP control (EM-anchored: shorts at spot +/- k*ATM-straddle).

Same ladder / qty / hold-to-expiry for both, so the only difference is the
strike rule -> isolates whether VP strike placement helps the multi-entry
structure (the thing that actually harvests premium).

VP uses 1-min bars (smear) — trade-level barely moves the levels and full-RTH
trades x 893 days is wasteful. Priced + cash-settled on real CBBO (data/cbbo_cache).
Daily P&L = sum of the day's tranches; Sharpe on the daily series.

Usage:
  python scripts/backtest_vp_meic.py --mode vp
  python scripts/backtest_vp_meic.py --mode em
  python scripts/backtest_vp_meic.py --mode vp --sweep
  python scripts/backtest_vp_meic.py --mode vp --wfcv
"""
from __future__ import annotations
import argparse, itertools, math, sys
from dataclasses import dataclass, replace, asdict
from datetime import date
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zerodte.backtest.cbbo_store import CbboStore
from volume_profile_orb import load as load_days, volume_profile

CBBO_DIR = "data/cbbo_cache_meic"   # wide strikes (117+), dense, 11:00+; the validated MEIC cache
MULT = 100
TICK = 0.01


def make_ladder(start="11:00", step=20, last="15:00", n=99):
    h, m = int(start[:2]), int(start[3:]); lh, lm = int(last[:2]), int(last[3:])
    out = []
    for _ in range(n):
        if (h, m) > (lh, lm):
            break
        out.append(f"{h:02d}:{m:02d}")
        m += step
        while m >= 60:
            m -= 60; h += 1
    return out


@dataclass
class MParams:
    mode: str = "vp"             # vp | em | blend
    start: str = "11:00"
    window_min: int = 0          # 0 = cumulative (09:30->entry); else rolling last-W min
    va_pct: float = 0.70
    buffer_pts: float = 5.0
    em_mult: float = 1.0         # em control / blend scale
    alpha: float = 1.0           # blend: weight on VP distance (1=pure VP, 0=pure EM)
    width: int = 30
    min_credit: float = 0.30
    slip_per_leg: float = 0.05


def _mid(q):
    return None if q is None else (q[0] + q[1]) / 2.0


def _emod(hhmm):
    return (int(hhmm[:2]) - 9) * 60 + int(hhmm[3:]) - 30


def _dev_vp(day, emod, window_min, va_pct, memo):
    key = (emod, window_min, va_pct)
    if key in memo:
        return memo[key]
    lo = emod - window_min if window_min > 0 else 0
    mask = (day.mod >= max(0, lo)) & (day.mod < emod)
    if mask.sum() < 5:
        memo[key] = None; return None
    vp = volume_profile(day.l[mask], day.h[mask], day.v[mask], TICK, 1.0, va_pct)
    memo[key] = vp
    return vp


def simulate_day(cb, day, p: MParams, schedule):
    final = cb.settlement_forward()
    if final is None:
        return None
    memo = {}
    tranche_pnls = []
    for hhmm in schedule:
        emod = _emod(hhmm)
        c = day.c[day.mod == emod]
        if len(c) == 0:
            continue
        spy = float(c[0])
        spx = cb.derive_spot(hhmm, spy * 10.0)
        if spx is None:
            continue
        if p.mode == "em":
            straddle = cb.atm_straddle(hhmm, spx)
            if straddle is None:
                continue
            move = p.em_mult * straddle
            sc = math.ceil((spx + move) / 5) * 5
            sp = math.floor((spx - move) / 5) * 5
        elif p.mode == "blend":
            # VP asymmetry + EM (time-aware) scale: dist = a*VP + (1-a)*em*straddle
            straddle = cb.atm_straddle(hhmm, spx)
            vp = _dev_vp(day, emod, p.window_min, p.va_pct, memo)
            if straddle is None or vp is None:
                continue
            _, vah, val = vp
            ratio = spx / spy
            em_d = p.em_mult * straddle
            call_d = p.alpha * (vah * ratio - spx) + (1 - p.alpha) * em_d + p.buffer_pts
            put_d = p.alpha * (spx - val * ratio) + (1 - p.alpha) * em_d + p.buffer_pts
            call_d = max(call_d, 5.0); put_d = max(put_d, 5.0)
            sc = math.ceil((spx + call_d) / 5) * 5
            sp = math.floor((spx - put_d) / 5) * 5
        else:
            vp = _dev_vp(day, emod, p.window_min, p.va_pct, memo)
            if vp is None:
                continue
            _, vah, val = vp
            ratio = spx / spy
            sc = math.ceil((vah * ratio + p.buffer_pts) / 5) * 5
            sp = math.floor((val * ratio - p.buffer_pts) / 5) * 5
        lc, lp = sc + p.width, sp - p.width
        q = [cb.lookup(hhmm, sc, True), cb.lookup(hhmm, lc, True),
             cb.lookup(hhmm, sp, False), cb.lookup(hhmm, lp, False)]
        qm = [_mid(x) for x in q]
        if any(x is None for x in qm):
            continue
        credit = (qm[0] - qm[1]) + (qm[2] - qm[3]) - 4 * p.slip_per_leg
        if credit < p.min_credit:
            continue
        if (p.width - credit) <= 0:
            continue
        cl = min(max(final - sc, 0.0), p.width)
        pl = min(max(sp - final, 0.0), p.width)
        tranche_pnls.append((credit - cl - pl) * MULT)
    if not tranche_pnls:
        return None
    return {"date": day.date, "pnl": sum(tranche_pnls), "n_tranche": len(tranche_pnls),
            "wins": sum(x > 0 for x in tranche_pnls)}


def run_many(days, dates, plist, schedules):
    cb = CbboStore("", cache_dir=CBBO_DIR)
    out = [[] for _ in plist]
    for d in dates:
        day = days.get(d)
        if day is None:
            continue
        if cb.load_day(date.fromisoformat(d)) == 0:
            continue
        for i, (p, sched) in enumerate(zip(plist, schedules)):
            r = simulate_day(cb, day, p, sched)
            if r is not None:
                out[i].append(r)
    return [metrics(t, p) for t, p in zip(out, plist)]


def backtest(days, dates, p):
    sched = make_ladder(p.start)
    return run_many(days, dates, [p], [sched])[0]


def metrics(days_res, p):
    if not days_res:
        return {"n_days": 0, "sharpe": 0.0, "params": asdict(p)}
    pnl = np.array([d["pnl"] for d in days_res])
    ntr = sum(d["n_tranche"] for d in days_res)
    nwin = sum(d["wins"] for d in days_res)
    sd = pnl.std(ddof=1)
    sharpe = float(pnl.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    eq = np.cumsum(pnl); dd = float((eq - np.maximum.accumulate(eq)).min())
    win = pnl[pnl > 0]; loss = pnl[pnl <= 0]
    pf = float(win.sum() / abs(loss.sum())) if loss.sum() != 0 else float("inf")
    return {
        "n_days": len(days_res), "n_tranche": ntr,
        "tranche_wr": round(nwin / ntr, 3) if ntr else 0,
        "day_wr": round(float((pnl > 0).mean()), 3),
        "sharpe": round(sharpe, 3),
        "total_usd": round(float(pnl.sum()), 0),
        "avg_day_usd": round(float(pnl.mean()), 0),
        "profit_factor": round(pf, 3),
        "max_dd_usd": round(dd, 0),
        "params": asdict(p),
    }


def _load():
    days = {d.date: d for d in load_days("SPY")}
    cd = {q.stem for q in Path(CBBO_DIR).glob("*.pkl")}
    dates = sorted(set(days) & cd)
    print(f"{len(dates)} days", flush=True)
    return days, dates


# cbbo_cache_meic is 11:00+, so start is pinned at 11:00 (the deployed ladder).
GRID_VP = {"start": ["11:00"], "window_min": [0, 90], "va_pct": [0.70, 0.80, 0.90],
           "buffer_pts": [0.0, 5.0, 10.0], "width": [20, 30]}
GRID_EM = {"start": ["11:00"], "em_mult": [0.8, 1.0, 1.25, 1.5], "width": [20, 30]}
# Time-aware blend: VP skew (rolling-90, va0.8) + EM scale; sweep alpha.
GRID_BLEND = {"start": ["11:00"], "window_min": [90], "va_pct": [0.80],
              "alpha": [0.0, 0.25, 0.5, 0.75, 1.0], "em_mult": [0.8, 1.0],
              "buffer_pts": [0.0], "width": [30]}


def run_grid(days, dates, base, grid, min_days=120):
    plist = [replace(base, **dict(zip(grid, c))) for c in itertools.product(*grid.values())]
    scheds = [make_ladder(p.start) for p in plist]
    rows = [m for m in run_many(days, dates, plist, scheds) if m["n_days"] >= min_days]
    rows.sort(key=lambda m: m["sharpe"], reverse=True)
    return rows


def _fmt(m):
    pr = m["params"]
    if pr["mode"] == "vp":
        anc = f"win={pr['window_min']} va={pr['va_pct']} buf={pr['buffer_pts']}"
    elif pr["mode"] == "blend":
        anc = f"alpha={pr['alpha']} emX={pr['em_mult']} win={pr['window_min']} va={pr['va_pct']}"
    else:
        anc = f"emX={pr['em_mult']}"
    return (f"Sh={m['sharpe']:+.2f} ${m['total_usd']:+,.0f} avgday=${m['avg_day_usd']:+.0f} "
            f"twr={m['tranche_wr']:.2f} pf={m['profit_factor']:.2f} dd=${m['max_dd_usd']:,.0f} "
            f"nd={m['n_days']} | {pr['mode']} start={pr['start']} {anc} w={pr['width']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="vp", choices=["vp", "em", "blend"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--wfcv", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    days, dates = _load()
    base = MParams(mode=a.mode)
    grid = {"em": GRID_EM, "blend": GRID_BLEND}.get(a.mode, GRID_VP)

    if a.sweep:
        rows = run_grid(days, dates, base, grid)
        print(f"# {a.mode}-MEIC: {len(rows)} cells. Top 12:")
        for m in rows[:12]:
            print("  " + _fmt(m))
        if rows:
            print(f"# median {np.median([m['sharpe'] for m in rows]):+.2f} worst {rows[-1]['sharpe']:+.2f}")
    elif a.wfcv:
        n = len(dates); k = a.folds; fs = n // (k + 1); oos = []
        print(f"# WF-CV {a.mode}-MEIC: {n} days")
        for i in range(1, k + 1):
            tr, te = dates[:i * fs], dates[i * fs:(i + 1) * fs]
            rows = run_grid(days, tr, base, grid, min_days=60)
            if not rows:
                print(f"  fold {i}: none"); continue
            b = MParams(**rows[0]["params"]); mo = backtest(days, te, b)
            oos.append(mo["sharpe"])
            print(f"  fold {i}: {_fmt(rows[0]).split('|')[1].strip()} (trSh={rows[0]['sharpe']:+.2f}) "
                  f"|| OOS Sh={mo['sharpe']:+.2f} ${mo['total_usd']:+,.0f} avgday=${mo['avg_day_usd']:+.0f} nd={mo['n_days']}")
        if oos:
            print(f"# OOS Sharpe mean={np.mean(oos):+.2f} median={np.median(oos):+.2f} pos={sum(s>0 for s in oos)}/{len(oos)}")
    else:
        m = backtest(days, dates, base)
        print(f"# Baseline {a.mode}-MEIC (start 11:00, ladder->15:00, hold-to-expiry, qty1):")
        print("  " + _fmt(m))


if __name__ == "__main__":
    main()
