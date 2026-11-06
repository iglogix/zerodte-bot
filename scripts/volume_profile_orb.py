"""
Volume-Profile Opening-Range Breakout (FRVP-ORB) backtester.

Faithful implementation of the strategy in the YouTube volume-profile strategy notes (not published):

  1. Build a Fixed-Range Volume Profile over the 09:30-09:45 ET opening block.
     -> Point of Control (POC), Value Area High (VAH), Value Area Low (VAL).
  2. After 09:45, wait for the first `entry_tf`-minute candle whose BODY (close)
     closes fully outside [VAL, VAH]. Close above VAH => long; below VAL => short.
  3. Enter at that candle close. Stop = POC -/+ `poc_buffer` (2 ticks in the video).
     Target = entry +/- rr * risk, where risk = |entry - stop|.
  4. Hard entry cutoff at 11:00 ET. Manage to stop/target/EOD on 1-min bars.

`reverse=True` fades the breakout instead: POC becomes the TARGET (mean reversion
to value) and the stop is rr-scaled beyond entry on the breakout side.

Trades the UNDERLYING directionally (matches the video's index/futures framing),
so PnL is clean of options-pricing confounds. Returns are R-multiples
(risk-normalised, fixed-fractional sizing); Sharpe is on the per-trade R series,
annualised by sqrt(252). Vectorised per-day for sweep speed.

Usage:
    python scripts/volume_profile_orb.py --symbol SPY
    python scripts/volume_profile_orb.py --symbol SPY --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

CSV = {
    "SPY": "data/spy_1m.csv",
    "IWM": "data/iwm_1m.csv",
    "NVDA": "data/nvda_1m.csv",
}
TICK = {"SPY": 0.01, "IWM": 0.01, "NVDA": 0.01}


@dataclass
class Params:
    or_minutes: int = 15          # opening-range length
    va_pct: float = 0.70          # value-area volume fraction
    entry_tf: int = 5             # breakout candle timeframe (min)
    cutoff: str = "11:00"         # hard entry cutoff (ET)
    rr: float = 2.0               # reward:risk
    poc_buffer_ticks: float = 2.0 # stop/target offset beyond POC, in ticks
    body_buffer_ticks: float = 0.0  # required close beyond level, in ticks
    slippage_ticks: float = 1.0   # per-side slippage, in ticks
    eod: str = "15:55"            # forced exit time (ET)
    row_size_ticks: float = 1.0   # VP price-bin width, in ticks
    reverse: bool = False         # fade the breakout instead of following it
    min_range_frac: float = 0.0   # require OR range >= this * 20d median OR range
    max_range_frac: float = 0.0   # 0 = no cap; else require OR range <= this * ref


def _mod(t_str: str) -> int:
    h, m = map(int, t_str.split(":"))
    return (h - 9) * 60 + m - 30  # minutes from 09:30


# ----------------------------------------------------------------------------
# Data loading -> per-day numpy arrays (built once)
# ----------------------------------------------------------------------------
class Day:
    __slots__ = ("date", "mod", "o", "h", "l", "c", "v")

    def __init__(self, date, mod, o, h, l, c, v):
        self.date, self.mod = date, mod
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v


def load(symbol: str):
    df = pd.read_csv(CSV[symbol])
    ts = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("America/New_York")
    mod = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    keep = (mod >= 0) & (mod < 390)  # 09:30..15:59 RTH
    df = df.loc[keep].copy()
    df["mod"] = mod[keep].values
    df["date"] = ts[keep].dt.date.values
    days = []
    for d, g in df.groupby("date", sort=True):
        days.append(Day(str(d), g["mod"].to_numpy(),
                        g["open"].to_numpy(), g["high"].to_numpy(),
                        g["low"].to_numpy(), g["close"].to_numpy(),
                        g["volume"].to_numpy(float)))
    return days


# ----------------------------------------------------------------------------
# Volume profile
# ----------------------------------------------------------------------------
def volume_profile(lo_a, hi_a, vol_a, tick, row_size_ticks, va_pct):
    row = tick * row_size_ticks
    lo = lo_a.min(); hi = hi_a.max()
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    n_bins = int(round((hi - lo) / row)) + 1
    if n_bins < 2:
        return None
    centers = lo + (np.arange(n_bins) + 0.5) * row
    vol = np.zeros(n_bins)
    for b_low, b_high, b_vol in zip(lo_a, hi_a, vol_a):
        i0 = max(0, min(n_bins - 1, int((b_low - lo) / row)))
        i1 = max(0, min(n_bins - 1, int((b_high - lo) / row)))
        vol[i0:i1 + 1] += b_vol / (i1 - i0 + 1)
    total = vol.sum()
    if total <= 0:
        return None
    poc_i = int(np.argmax(vol))
    lo_i = hi_i = poc_i
    acc = vol[poc_i]; target = va_pct * total
    while acc < target and (lo_i > 0 or hi_i < n_bins - 1):
        below = vol[lo_i - 1] if lo_i > 0 else -1.0
        above = vol[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if above >= below:
            hi_i += 1; acc += vol[hi_i]
        else:
            lo_i -= 1; acc += vol[lo_i]
    return centers[poc_i], centers[hi_i], centers[lo_i]  # poc, vah, val


def vp_from_hist(prices_cents, vols, va_pct):
    """POC/VAH/VAL from a true penny volume-at-price histogram (trade-level)."""
    if prices_cents is None or len(prices_cents) == 0:
        return None
    lo = int(prices_cents.min()); hi = int(prices_cents.max())
    n = hi - lo + 1
    if n < 2:
        return None
    dense = np.zeros(n)
    dense[prices_cents - lo] += vols
    total = dense.sum()
    if total <= 0:
        return None
    poc_i = int(np.argmax(dense))
    lo_i = hi_i = poc_i; acc = dense[poc_i]; target = va_pct * total
    while acc < target and (lo_i > 0 or hi_i < n - 1):
        below = dense[lo_i - 1] if lo_i > 0 else -1.0
        above = dense[hi_i + 1] if hi_i < n - 1 else -1.0
        if above >= below:
            hi_i += 1; acc += dense[hi_i]
        else:
            lo_i -= 1; acc += dense[lo_i]
    return (lo + poc_i) / 100.0, (lo + hi_i) / 100.0, (lo + lo_i) / 100.0


def load_trade_vp(symbol: str):
    path = f"data/or_trade_vp/{symbol}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Per-day simulation (vectorised)
# ----------------------------------------------------------------------------
def simulate_day(day: Day, p: Params, tick: float,
                 range_ref: Optional[float] = None,
                 vp_hist: Optional[dict] = None) -> Optional[dict]:
    mod = day.mod
    or_mask = (mod >= 0) & (mod < p.or_minutes)
    if or_mask.sum() < max(3, p.or_minutes // 2):
        return None
    if vp_hist is not None:  # trade-level FRVP (09:30-09:45 window; or_minutes pinned 15)
        vp = vp_from_hist(vp_hist["p"], vp_hist["v"], p.va_pct)
    else:
        vp = volume_profile(day.l[or_mask], day.h[or_mask], day.v[or_mask],
                            tick, p.row_size_ticks, p.va_pct)
    if vp is None:
        return None
    poc, vah, val = vp

    if range_ref and (p.min_range_frac > 0 or p.max_range_frac > 0):
        rng = day.h[or_mask].max() - day.l[or_mask].min()
        if p.min_range_frac > 0 and rng < p.min_range_frac * range_ref:
            return None
        if p.max_range_frac > 0 and rng > p.max_range_frac * range_ref:
            return None

    cutoff_m = _mod(p.cutoff); eod_m = _mod(p.eod)
    body_buf = p.body_buffer_ticks * tick
    slip = p.slippage_ticks * tick
    poc_buf = p.poc_buffer_ticks * tick

    # Breakout scan: entry_tf candle closes after the OR, before cutoff.
    scan = (mod >= p.or_minutes) & (mod < cutoff_m)
    if not scan.any():
        return None
    s_mod = mod[scan]; s_close = day.c[scan]
    bucket = (s_mod // p.entry_tf)
    # last 1-min bar of each bucket = candle close
    last_in_bucket = np.zeros(len(bucket), bool)
    last_in_bucket[-1] = True
    last_in_bucket[:-1] = bucket[1:] != bucket[:-1]
    cand_close = s_close[last_in_bucket]
    cand_mod = s_mod[last_in_bucket]
    long_sig = cand_close > vah + body_buf
    short_sig = cand_close < val - body_buf
    trig = long_sig | short_sig
    if not trig.any():
        return None
    j = int(np.argmax(trig))
    signal_dir = 1 if long_sig[j] else -1
    entry_mod = int(cand_mod[j])
    entry_px = float(cand_close[j])

    direction = signal_dir
    if p.reverse:
        direction = -signal_dir
        target = poc - direction * poc_buf
        reward = abs(target - entry_px)
        if reward <= 0:
            return None
        risk = reward / p.rr
        stop = entry_px - direction * risk
    else:
        if direction == 1:
            stop = poc - poc_buf; risk = entry_px - stop
        else:
            stop = poc + poc_buf; risk = stop - entry_px
        if risk <= 0:
            return None
        target = entry_px + direction * p.rr * risk

    fill = entry_px + direction * slip

    # Forward management on 1-min bars after the entry bar, until eod.
    fwd = (mod > entry_mod) & (mod <= eod_m)
    if not fwd.any():
        return None
    fh = day.h[fwd]; fl = day.l[fwd]; fc = day.c[fwd]
    if direction == 1:
        hit_stop = fl <= stop; hit_tgt = fh >= target
    else:
        hit_stop = fh >= stop; hit_tgt = fl <= target
    si = int(np.argmax(hit_stop)) if hit_stop.any() else 10**9
    ti = int(np.argmax(hit_tgt)) if hit_tgt.any() else 10**9
    if si == 10**9 and ti == 10**9:
        exit_px, outcome = float(fc[-1]), "eod"
    elif si <= ti:                       # pessimistic: tie -> stop
        exit_px, outcome = stop, "stop"
    else:
        exit_px, outcome = target, "target"

    exit_fill = exit_px
    if outcome in ("stop", "eod"):
        exit_fill = exit_px - direction * slip

    pnl_pts = direction * (exit_fill - fill)
    risk = float(risk)
    return {
        "date": day.date, "direction": int(direction), "signal_dir": int(signal_dir),
        "entry": float(fill), "stop": float(stop), "target": float(target),
        "exit": float(exit_fill), "risk_pts": risk, "outcome": outcome,
        "pnl_pts": float(pnl_pts), "r_mult": float(pnl_pts / risk),
        "ret_pct": float(pnl_pts / fill),
        "poc": float(poc), "vah": float(vah), "val": float(val),
    }


# ----------------------------------------------------------------------------
# Driver + metrics
# ----------------------------------------------------------------------------
def backtest(days, p: Params, tick: float,
             start: Optional[str] = None, end: Optional[str] = None,
             vp_cache: Optional[dict] = None) -> dict:
    sub = [d for d in days if (not start or d.date >= start) and (not end or d.date <= end)]
    trades = []
    hist = []
    for d in sub:
        ref = float(np.median(hist[-20:])) if len(hist) >= 5 else None
        if vp_cache is not None:
            vph = vp_cache.get(d.date)
            if vph is None:
                continue  # no trade histogram for this day -> skip (no smear fallback)
        else:
            vph = None
        res = simulate_day(d, p, tick, range_ref=ref, vp_hist=vph)
        if res is not None:
            trades.append(res)
        om = (d.mod >= 0) & (d.mod < p.or_minutes)
        if om.sum() >= 3:
            hist.append(float(d.h[om].max() - d.l[om].min()))
    return metrics(trades, len(sub), p)


def metrics(trades: list, n_sessions: int, p: Params) -> dict:
    if not trades:
        return {"n_trades": 0, "n_sessions": n_sessions, "sharpe": 0.0,
                "expectancy_R": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "params": asdict(p), "_trades": []}
    r = np.array([t["r_mult"] for t in trades])
    wins = r[r > 0]; losses = r[r <= 0]
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    return {
        "n_trades": len(trades), "n_sessions": n_sessions,
        "trade_rate": round(len(trades) / n_sessions, 3) if n_sessions else 0,
        "win_rate": round(float((r > 0).mean()), 3),
        "expectancy_R": round(float(r.mean()), 4),
        "sharpe": round(sharpe, 3),
        "total_R": round(float(r.sum()), 2),
        "profit_factor": round(pf, 3),
        "avg_win_R": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_R": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "max_dd_R": round(_max_dd(r), 2),
        "outcomes": {k: int(sum(1 for t in trades if t["outcome"] == k))
                     for k in ("target", "stop", "eod")},
        "params": asdict(p), "_trades": trades,
    }


def _max_dd(r: np.ndarray) -> float:
    eq = np.cumsum(r); peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY", choices=list(CSV))
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--trade-vp", action="store_true",
                    help="use trade-level FRVP cache (data/or_trade_vp/<SYM>.pkl)")
    ap.add_argument("--json")
    a = ap.parse_args()
    days = load(a.symbol)
    p = Params(reverse=a.reverse)
    vp_cache = load_trade_vp(a.symbol) if a.trade_vp else None
    m = backtest(days, p, TICK[a.symbol], a.start, a.end, vp_cache=vp_cache)
    trades = m.pop("_trades", [])
    print(json.dumps(m, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"metrics": m, "trades": trades}, f, indent=2)


if __name__ == "__main__":
    main()
