"""
Value-Area Iron Condor on SPX 0DTE — the premium-SELLING translation of the
volume-profile opening-range strategy.

Thesis (from the directional study): breakouts fail ~67% of the time and price
respects the opening-range value area. So instead of buying the breakout, SELL a
0DTE iron condor with short strikes anchored at the value-area edges:
  short call just above VAH, short put just below VAL, fixed-width wings.

Levels come from the FAITHFUL trade-level FRVP (data/or_trade_vp/SPY.pkl, true
volume-at-price over 09:30-09:45). SPY value-area levels are mapped into SPX
strike space using the put-call-parity spot ratio at the entry minute (sidesteps
the static SPYx10 basis). Pricing + cash settlement use real SPX 0DTE CBBO quotes
(data/cbbo_cache). Held to expiry (no stop-fill fiction); optional take-profit.

Usage:
    python scripts/backtest_vp_ic.py                       # baseline
    python scripts/backtest_vp_ic.py --sweep
    python scripts/backtest_vp_ic.py --wfcv
"""
from __future__ import annotations
import argparse, itertools, math, sys
from dataclasses import dataclass, replace, asdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src_v2.backtest.cbbo_store import CbboStore
from volume_profile_orb import load as load_days, load_trade_vp, vp_from_hist

CBBO_DIR = "data/cbbo_cache"
MULT = 100


@dataclass
class ICParams:
    entry_min: str = "09:45"     # entry time ET (must be in cbbo_cache minutes)
    strike_mode: str = "vp"      # vp = VAH/VAL anchors; em = no-VP control (spot ± k*straddle)
    va_pct: float = 0.70         # value-area fraction -> VAH/VAL strike anchors (vp mode)
    buffer_pts: float = 0.0      # extra SPX pts beyond VAH/VAL for the shorts (vp mode)
    em_mult: float = 1.0         # em mode: short strikes at spot ± em_mult * ATM straddle
    width: int = 20              # wing width (SPX pts)
    min_credit: float = 0.30     # skip if net credit (pts) below this
    slip_per_leg: float = 0.05   # entry slippage per leg (pts)
    tp_pct: float = 0.0          # 0 = hold to expiry; else buy back at tp_pct*credit
    qty: int = 1


def _mid(q):
    return None if q is None else (q[0] + q[1]) / 2.0


def simulate_day(cb: CbboStore, day, vph, p: ICParams):
    em = p.entry_min
    emod = (int(em[:2]) - 9) * 60 + int(em[3:]) - 30
    # SPY spot at entry
    c = day.c[day.mod == emod]
    if len(c) == 0:
        return None
    spy = float(c[0])
    vp = vp_from_hist(vph["p"], vph["v"], p.va_pct)
    if vp is None:
        return None
    poc, vah, val = vp
    spx = cb.derive_spot(em, spy * 10.0)
    if spx is None:
        return None
    ratio = spx / spy
    if p.strike_mode == "em":
        # No-VP control: shorts at spot ± em_mult * ATM straddle (the 0DTE EM).
        straddle = cb.atm_straddle(em, spx)
        if straddle is None:
            return None
        move = p.em_mult * straddle
        sc = math.ceil((spx + move) / 5.0) * 5
        sp = math.floor((spx - move) / 5.0) * 5
    else:
        # VP: short strikes just outside the value area (rounded to 5-pt grid)
        vah_spx, val_spx = vah * ratio, val * ratio
        sc = math.ceil((vah_spx + p.buffer_pts) / 5.0) * 5
        sp = math.floor((val_spx - p.buffer_pts) / 5.0) * 5
    lc, lp = sc + p.width, sp - p.width
    # price legs at entry
    scq, lcq = cb.lookup(em, sc, True), cb.lookup(em, lc, True)
    spq, lpq = cb.lookup(em, sp, False), cb.lookup(em, lp, False)
    mids = [_mid(scq), _mid(lcq), _mid(spq), _mid(lpq)]
    if any(m is None for m in mids):
        return None
    sc_m, lc_m, sp_m, lp_m = mids
    credit = (sc_m - lc_m) + (sp_m - lp_m) - 4 * p.slip_per_leg
    if credit < p.min_credit:
        return None
    risk = p.width - credit
    if risk <= 0:
        return None

    final = cb.settlement_forward()
    if final is None:
        return None
    call_loss = min(max(final - sc, 0.0), p.width)
    put_loss = min(max(sp - final, 0.0), p.width)
    pnl_pts = credit - call_loss - put_loss
    outcome = "expiry"

    # Optional take-profit: buy back when IC mark <= (1-tp)*credit, scanning minutes.
    if p.tp_pct > 0:
        tgt = credit * (1 - p.tp_pct)  # remaining debit to close
        for mm in range(emod + 1, 386):
            mstr = f"{9 + (30 + mm) // 60:02d}:{(30 + mm) % 60:02d}"
            q = [cb.lookup(mstr, sc, True), cb.lookup(mstr, lc, True),
                 cb.lookup(mstr, sp, False), cb.lookup(mstr, lp, False)]
            qm = [_mid(x) for x in q]
            if any(x is None for x in qm):
                continue
            mark = (qm[0] - qm[1]) + (qm[2] - qm[3])  # cost to close (debit)
            if mark <= tgt:
                pnl_pts = credit - mark - 2 * p.slip_per_leg  # exit slip (2 short legs)
                outcome = "tp"
                break

    return {"date": day.date, "sc": sc, "sp": sp, "credit": round(credit, 2),
            "risk": round(risk, 2), "pnl_pts": pnl_pts,
            "pnl_usd": pnl_pts * MULT * p.qty, "r": pnl_pts / risk,
            "outcome": outcome, "final": final}


def run_many(days_by_date, vp_cache, dates, params_list):
    """Evaluate many param cells while loading each day's CBBO pkl ONCE
    (lazy per-day via cache_dir -> memory-safe; no full preload)."""
    cb = CbboStore("", cache_dir=CBBO_DIR)
    trades = [[] for _ in params_list]
    for d in dates:
        day = days_by_date.get(d)
        vph = vp_cache.get(d)
        if day is None or vph is None:
            continue
        if cb.load_day(date.fromisoformat(d)) == 0:
            continue
        for i, p in enumerate(params_list):
            r = simulate_day(cb, day, vph, p)
            if r is not None:
                trades[i].append(r)
    return [metrics(t, len(dates), p) for t, p in zip(trades, params_list)]


def backtest(days_by_date, vp_cache, p: ICParams, dates):
    return run_many(days_by_date, vp_cache, dates, [p])[0]


def metrics(trades, n_sessions, p: ICParams):
    if not trades:
        return {"n": 0, "sharpe": 0.0, "params": asdict(p), "_t": []}
    r = np.array([t["r"] for t in trades])
    usd = np.array([t["pnl_usd"] for t in trades])
    wins = usd[usd > 0]; losses = usd[usd <= 0]
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    eq = np.cumsum(usd); dd = float((eq - np.maximum.accumulate(eq)).min())
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    return {
        "n": len(trades), "n_sessions": n_sessions,
        "win_rate": round(float((usd > 0).mean()), 3),
        "sharpe": round(sharpe, 3),
        "expectancy_R": round(float(r.mean()), 4),
        "total_usd": round(float(usd.sum()), 0),
        "avg_usd": round(float(usd.mean()), 1),
        "profit_factor": round(pf, 3),
        "avg_win": round(float(wins.mean()), 0) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 0) if len(losses) else 0.0,
        "max_dd_usd": round(dd, 0),
        "tp_rate": round(float(np.mean([t["outcome"] == "tp" for t in trades])), 2),
        "params": asdict(p), "_t": trades,
    }


# ----------------------------------------------------------------------------
def _load_all():
    days = {d.date: d for d in load_days("SPY")}
    vp = load_trade_vp("SPY")
    cdates = {p.stem for p in Path(CBBO_DIR).glob("*.pkl")}  # lazy: just date list
    dates = sorted(set(days) & set(vp) & cdates)
    print(f"{len(dates)} days with SPY-VP + CBBO", flush=True)
    return days, vp, dates


GRID_VP = {
    "entry_min": ["09:45", "10:00"],
    "va_pct": [0.60, 0.70, 0.80, 0.90],
    "buffer_pts": [0.0, 5.0, 10.0],
    "width": [20, 30],
    "tp_pct": [0.0, 0.5],
}
GRID_EM = {
    "entry_min": ["09:45", "10:00"],
    "em_mult": [0.6, 0.8, 1.0, 1.2, 1.5],
    "width": [20, 30],
    "tp_pct": [0.0, 0.5],
}
GRID = GRID_VP  # default; overridden by --strike-mode


def run_grid(days, vp, dates, base=ICParams(), min_trades=80, grid=None):
    grid = grid or GRID
    params = [replace(base, **dict(zip(grid, combo)))
              for combo in itertools.product(*grid.values())]
    rows = [m for m in run_many(days, vp, dates, params) if m["n"] >= min_trades]
    for m in rows:
        m.pop("_t", None)
    rows.sort(key=lambda m: m["sharpe"], reverse=True)
    return rows


def _fmt(m):
    pr = m["params"]
    anc = (f"va={pr['va_pct']} buf={pr['buffer_pts']}" if pr.get("strike_mode", "vp") == "vp"
           else f"emX={pr['em_mult']}")
    return (f"Sh={m['sharpe']:+.2f} avg=${m['avg_usd']:+.0f} wr={m['win_rate']:.2f} "
            f"pf={m['profit_factor']:.2f} n={m['n']} dd=${m['max_dd_usd']:.0f} "
            f"| {pr.get('strike_mode','vp')} entry={pr['entry_min']} {anc} "
            f"w={pr['width']} tp={pr['tp_pct']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--wfcv", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-trades", type=int, default=80)
    ap.add_argument("--strike-mode", default="vp", choices=["vp", "em"])
    a = ap.parse_args()
    days, vp, dates = _load_all()
    base = ICParams(strike_mode=a.strike_mode)
    grid = GRID_EM if a.strike_mode == "em" else GRID_VP

    if a.sweep:
        rows = run_grid(days, vp, dates, base=base, min_trades=a.min_trades, grid=grid)
        print(f"# {len(rows)} cells. Top 15 by Sharpe:")
        for m in rows[:15]:
            print("  " + _fmt(m))
        if rows:
            print(f"\n# worst {rows[-1]['sharpe']:+.2f}  median "
                  f"{np.median([m['sharpe'] for m in rows]):+.2f}")
    elif a.wfcv:
        n = len(dates); k = a.folds; fs = n // (k + 1)
        oos = []
        print(f"# WF-CV value-area IC: {n} days, {k} folds")
        for i in range(1, k + 1):
            tr = dates[:i * fs]; te = dates[i * fs:(i + 1) * fs]
            rows = run_grid(days, vp, tr, base=base, min_trades=max(30, a.min_trades // 2), grid=grid)
            if not rows:
                print(f"  fold {i}: no train cell"); continue
            best = ICParams(**rows[0]["params"])
            mo = backtest(days, vp, best, te); mo.pop("_t", None)
            oos.append(mo["sharpe"])
            pr = rows[0]["params"]
            print(f"  fold {i}: pick va={pr['va_pct']} buf={pr['buffer_pts']} w={pr['width']} "
                  f"tp={pr['tp_pct']} entry={pr['entry_min']} (trSh={rows[0]['sharpe']:+.2f}) "
                  f"|| OOS[{te[0]}..{te[-1]}] Sh={mo['sharpe']:+.2f} avg=${mo['avg_usd']:+.0f} "
                  f"wr={mo['win_rate']:.2f} n={mo['n']}")
        if oos:
            print(f"\n# OOS Sharpe mean={np.mean(oos):+.2f} median={np.median(oos):+.2f} "
                  f"pos={sum(s>0 for s in oos)}/{len(oos)}")
    else:
        m = backtest(days, vp, base, dates)
        m.pop("_t", None)
        print(f"# Baseline {a.strike_mode} IC (entry 09:45, w=20, hold-to-expiry):")
        print("  " + _fmt(m))
        for kk in ("total_usd", "avg_win", "avg_loss", "max_dd_usd", "expectancy_R"):
            print(f"    {kk}: {m[kk]}")


if __name__ == "__main__":
    main()
