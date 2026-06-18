"""
Directional credit spread on the volume-profile breakout (SPX 0DTE, real CBBO).

On a breakout (first entry_tf candle closing outside the faithful VAH/VAL before
the cutoff), SELL a single defined-risk credit spread:

  mode=fade        : sell the spread on the BREAKOUT side (bet the breakout fails
                     and price stays back inside value). Up-break -> sell CALL
                     spread with short just above VAH; down-break -> sell PUT
                     spread with short just below VAL. (Directional study says
                     breakouts fail ~67%, so this is the natural edge.)
  mode=with_trend  : sell the spread on the FAR side, protected by the move (a la
                     orb_em_directional). Up-break -> sell PUT spread with short at
                     the opposite value edge / POC; down-break -> sell CALL spread.

SPY value levels mapped to SPX via parity-spot ratio at entry; priced + cash-settled
on data/cbbo_cache. Held to expiry (no stop-fill fiction). R-multiple + $ Sharpe.

Usage:
  python scripts/backtest_vp_dir_spread.py --mode fade
  python scripts/backtest_vp_dir_spread.py --mode fade --sweep
  python scripts/backtest_vp_dir_spread.py --mode fade --wfcv
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
class DSParams:
    mode: str = "fade"           # fade | with_trend
    entry_tf: int = 5            # breakout candle minutes
    or_minutes: int = 15
    cutoff: str = "11:00"
    va_pct: float = 0.70
    anchor: str = "edge"         # with_trend short anchor: edge (VAL/VAH) | poc
    buffer_pts: float = 0.0      # SPX pts beyond the short-strike level
    width: int = 20
    min_credit: float = 0.30
    slip_per_leg: float = 0.05
    body_buffer_ticks: float = 0.0


def _mid(q):
    return None if q is None else (q[0] + q[1]) / 2.0


def _detect_breakout(day, vah, val, p: DSParams):
    """Returns (dir, entry_mod, spy_close) in SPY space, or None."""
    mod = day.mod
    cmod = (int(p.cutoff[:2]) - 9) * 60 + int(p.cutoff[3:]) - 30
    scan = (mod >= p.or_minutes) & (mod < cmod)
    if not scan.any():
        return None
    s_mod, s_close = mod[scan], day.c[scan]
    bucket = s_mod // p.entry_tf
    last = np.zeros(len(bucket), bool); last[-1] = True
    last[:-1] = bucket[1:] != bucket[:-1]
    cc, cm = s_close[last], s_mod[last]
    bb = p.body_buffer_ticks * 0.01
    longs, shorts = cc > vah + bb, cc < val - bb
    trig = longs | shorts
    if not trig.any():
        return None
    j = int(np.argmax(trig))
    return (1 if longs[j] else -1), int(cm[j]), float(cc[j])


def simulate_day(cb, day, vph, p: DSParams):
    vp = vp_from_hist(vph["p"], vph["v"], p.va_pct)
    if vp is None:
        return None
    poc, vah, val = vp
    bk = _detect_breakout(day, vah, val, p)
    if bk is None:
        return None
    bdir, emod, spy = bk
    em = f"{9 + (30 + emod) // 60:02d}:{(30 + emod) % 60:02d}"
    spx = cb.derive_spot(em, spy * 10.0)
    if spx is None:
        return None
    ratio = spx / spy
    poc_x, vah_x, val_x = poc * ratio, vah * ratio, val * ratio

    # Choose spread side + short strike anchor.
    if p.mode == "fade":
        is_call = (bdir == 1)                     # up-break -> sell call spread
        lvl = vah_x if bdir == 1 else val_x
        if is_call:
            sk = math.ceil((lvl + p.buffer_pts) / 5) * 5
            lk = sk + p.width
        else:
            sk = math.floor((lvl - p.buffer_pts) / 5) * 5
            lk = sk - p.width
    else:  # with_trend: sell the FAR side, protected by the move
        is_call = (bdir == -1)                    # up-break -> sell PUT spread
        if p.anchor == "poc":
            lvl = poc_x
        else:
            lvl = val_x if bdir == 1 else vah_x
        if is_call:
            sk = math.ceil((lvl + p.buffer_pts) / 5) * 5
            lk = sk + p.width
        else:
            sk = math.floor((lvl - p.buffer_pts) / 5) * 5
            lk = sk - p.width

    sq, lq = cb.lookup(em, sk, is_call), cb.lookup(em, lk, is_call)
    sm, lm = _mid(sq), _mid(lq)
    if sm is None or lm is None:
        return None
    credit = (sm - lm) - 2 * p.slip_per_leg
    if credit < p.min_credit:
        return None
    risk = p.width - credit
    if risk <= 0:
        return None
    final = cb.settlement_forward()
    if final is None:
        return None
    if is_call:
        loss = min(max(final - sk, 0.0), p.width)
    else:
        loss = min(max(sk - final, 0.0), p.width)
    pnl = credit - loss
    return {"date": day.date, "dir": bdir, "side": "C" if is_call else "P",
            "sk": sk, "credit": round(credit, 2), "risk": round(risk, 2),
            "pnl_pts": pnl, "pnl_usd": pnl * MULT, "r": pnl / risk,
            "win": pnl > 0}


def run_many(days, vp, dates, plist):
    cb = CbboStore("", cache_dir=CBBO_DIR)
    tr = [[] for _ in plist]
    for d in dates:
        day, vph = days.get(d), vp.get(d)
        if day is None or vph is None:
            continue
        if cb.load_day(date.fromisoformat(d)) == 0:
            continue
        for i, p in enumerate(plist):
            r = simulate_day(cb, day, vph, p)
            if r is not None:
                tr[i].append(r)
    return [metrics(t, p) for t, p in zip(tr, plist)]


def backtest(days, vp, p, dates):
    return run_many(days, vp, dates, [p])[0]


def metrics(trades, p):
    if not trades:
        return {"n": 0, "sharpe": 0.0, "params": asdict(p), "_t": []}
    r = np.array([t["r"] for t in trades])
    usd = np.array([t["pnl_usd"] for t in trades])
    w, l = usd[usd > 0], usd[usd <= 0]
    sd = r.std(ddof=1)
    sh = float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    eq = np.cumsum(usd); dd = float((eq - np.maximum.accumulate(eq)).min())
    pf = float(w.sum() / abs(l.sum())) if l.sum() != 0 else float("inf")
    return {"n": len(trades), "win_rate": round(float((usd > 0).mean()), 3),
            "sharpe": round(sh, 3), "expectancy_R": round(float(r.mean()), 4),
            "total_usd": round(float(usd.sum()), 0), "avg_usd": round(float(usd.mean()), 1),
            "profit_factor": round(pf, 3), "max_dd_usd": round(dd, 0),
            "params": asdict(p), "_t": trades}


def _load():
    days = {d.date: d for d in load_days("SPY")}
    vp = load_trade_vp("SPY")
    cd = {q.stem for q in Path(CBBO_DIR).glob("*.pkl")}
    dates = sorted(set(days) & set(vp) & cd)
    print(f"{len(dates)} days", flush=True)
    return days, vp, dates


GRID = {"va_pct": [0.60, 0.70, 0.80], "buffer_pts": [0.0, 5.0, 10.0, 15.0],
        "width": [20, 30], "cutoff": ["10:30", "11:00"], "anchor": ["edge", "poc"]}


def run_grid(days, vp, dates, base, min_trades=60):
    plist = [replace(base, **dict(zip(GRID, c))) for c in itertools.product(*GRID.values())]
    rows = [m for m in run_many(days, vp, dates, plist) if m["n"] >= min_trades]
    for m in rows:
        m.pop("_t", None)
    rows.sort(key=lambda m: m["sharpe"], reverse=True)
    return rows


def _fmt(m):
    pr = m["params"]
    return (f"Sh={m['sharpe']:+.2f} avg=${m['avg_usd']:+.0f} wr={m['win_rate']:.2f} "
            f"pf={m['profit_factor']:.2f} n={m['n']} dd=${m['max_dd_usd']:.0f} "
            f"| {pr['mode']} va={pr['va_pct']} buf={pr['buffer_pts']} w={pr['width']} "
            f"cut={pr['cutoff']} anc={pr['anchor']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fade", choices=["fade", "with_trend"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--wfcv", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-trades", type=int, default=60)
    a = ap.parse_args()
    days, vp, dates = _load()
    base = DSParams(mode=a.mode)

    if a.sweep:
        rows = run_grid(days, vp, dates, base, a.min_trades)
        print(f"# {a.mode}: {len(rows)} cells. Top 12:")
        for m in rows[:12]:
            print("  " + _fmt(m))
        if rows:
            print(f"# median {np.median([m['sharpe'] for m in rows]):+.2f} worst {rows[-1]['sharpe']:+.2f}")
    elif a.wfcv:
        n = len(dates); k = a.folds; fs = n // (k + 1); oos = []
        print(f"# WF-CV {a.mode}: {n} days")
        for i in range(1, k + 1):
            tr, te = dates[:i * fs], dates[i * fs:(i + 1) * fs]
            rows = run_grid(days, vp, tr, base, max(25, a.min_trades // 2))
            if not rows:
                print(f"  fold {i}: none"); continue
            b = DSParams(**rows[0]["params"]); mo = backtest(days, vp, b, te); mo.pop("_t", None)
            oos.append(mo["sharpe"]); pr = rows[0]["params"]
            print(f"  fold {i}: va={pr['va_pct']} buf={pr['buffer_pts']} w={pr['width']} "
                  f"cut={pr['cutoff']} anc={pr['anchor']} (trSh={rows[0]['sharpe']:+.2f}) || "
                  f"OOS Sh={mo['sharpe']:+.2f} avg=${mo['avg_usd']:+.0f} wr={mo['win_rate']:.2f} n={mo['n']}")
        if oos:
            print(f"# OOS Sharpe mean={np.mean(oos):+.2f} median={np.median(oos):+.2f} pos={sum(s>0 for s in oos)}/{len(oos)}")
    else:
        m = backtest(days, vp, base, dates); m.pop("_t", None)
        print(f"# Baseline {a.mode} (va0.7 buf0 w20 cut11:00): " + _fmt(m))


if __name__ == "__main__":
    main()
