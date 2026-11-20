"""SPX 0DTE expected-move fade — study driver.

Spec:     docs/spx_0dte_em_fade_spec.md
Engine:   zerodte/backtest/em_fade_core.py  (+ zerodte/backtest/noarb.py)
Data:     data/cbbo_cache_fly_10min/  (real CBBO, 2022-05-11 -> 2026-06-11)

Modes
    sanity    blocking gate: data-layer reproduction, parity-spot vs SPYx10,
              ladder-edge, skip accounting, band-shift sensitivity
    measure   full-span depth x exit-regime x trail sweep, claim-by-claim table
    controls  with-trend mirror / unconditional 10:00 / random minute / buy-hold
    wf        walk-forward, frozen vs re-optimised, three-tier ladder

Usage
    .venv/bin/python -m scripts.backtest_em_fade sanity
    .venv/bin/python -m scripts.backtest_em_fade measure  --jobs 12
    .venv/bin/python -m scripts.backtest_em_fade controls --jobs 12
    .venv/bin/python -m scripts.backtest_em_fade wf       --jobs 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import date
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.em_fade_core import (
    DayContext,
    EmFadeConfig,
    simulate_day,
)
from zerodte.backtest.noarb import quote_ok, vertical_mark
from zerodte.backtest.vix_loader import load_vix_series, lookup_prior_close

CACHE_DIR = Path("data/cbbo_cache_fly_10min")
OUT_DIR = Path("backtest_data")
SPY_1M = Path("data/spy_1m.csv")
VIX_CSV = Path("data/vix_cboe.csv")


# ----------------------------------------------------------------------
# day loading
# ----------------------------------------------------------------------

def all_dates(start: Optional[date] = None, end: Optional[date] = None) -> List[date]:
    ds = sorted(date.fromisoformat(p.stem) for p in CACHE_DIR.glob("*.pkl"))
    if start:
        ds = [d for d in ds if d >= start]
    if end:
        ds = [d for d in ds if d <= end]
    return ds


def load_quotes(d: date) -> Optional[dict]:
    p = CACHE_DIR / f"{d.isoformat()}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# worker: run a list of configs over a chunk of days
# ----------------------------------------------------------------------

_SPY: Optional[dict] = None


def spy_cache() -> dict:
    """{bars: {date: {minute: (hi10, lo10, cl10)}}, prior: {date: (hi, lo)}}"""
    global _SPY
    if _SPY is None:
        pth = OUT_DIR / "spy_intraday_cache.pkl"
        with open(pth, "rb") as f:
            _SPY = pickle.load(f)
    return _SPY


def _run_chunk(args) -> List[dict]:
    days, configs = args
    need_spy = any(c.touch_basis == "intrabar" or c.sr_mode != "off" for c in configs)
    spy = spy_cache() if need_spy else {"bars": {}, "prior": {}}
    rows: List[dict] = []
    for d in days:
        quotes = load_quotes(d)
        if not quotes:
            for c in configs:
                rows.append({"cfg": c.key(), "date": d.isoformat(), "skip": "no_cbbo_data"})
            continue
        ctx = DayContext.build(d, quotes, bars=spy["bars"].get(d),
                               prior_hl=spy["prior"].get(d))
        for c in configs:
            out = simulate_day(ctx, c)
            if out.trade is None:
                rows.append({"cfg": c.key(), "date": d.isoformat(),
                             "skip": out.skip_reason, "touched": out.touched,
                             "em": out.em_pts, "open": out.open_spot})
            else:
                t = asdict(out.trade)
                t["session_date"] = d.isoformat()
                t["cfg"] = c.key()
                t["skip"] = None
                rows.append(t)
    return rows


def run(dates: Sequence[date], configs: Sequence[EmFadeConfig], jobs: int = 1) -> List[dict]:
    if jobs <= 1:
        return _run_chunk((list(dates), list(configs)))
    chunks = [list(dates[i::jobs]) for i in range(jobs)]
    with Pool(jobs) as pool:
        parts = pool.map(_run_chunk, [(c, list(configs)) for c in chunks if c])
    return [r for p in parts for r in p]


# ----------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------

def stats(trades: List[dict]) -> Optional[dict]:
    n = len(trades)
    if n == 0:
        return None
    pnls = [t["pnl_usd"] for t in trades]
    total = sum(pnls)
    mean = total / n
    sd = statistics.pstdev(pnls) if n > 1 else 0.0
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    risk = [t["max_loss_usd"] for t in trades]
    # t-stat on the per-trade mean: Sharpe/trade alone flatters low-frequency
    # strategies (see project_skyline_backtest).
    tstat = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    return {
        "n": n,
        "total": round(total, 0),
        "mean": round(mean, 2),
        "sd": round(sd, 2),
        "win_rate": round(len(wins) / n, 4),
        "avg_win": round(statistics.mean(wins), 2) if wins else 0.0,
        "avg_loss": round(statistics.mean(losses), 2) if losses else 0.0,
        "sharpe_trade": round(mean / sd, 4) if sd else 0.0,
        "t_stat": round(tstat, 2),
        "mean_risk": round(statistics.mean(risk), 0),
        "ror": round(mean / statistics.mean(risk), 4) if risk else 0.0,
        "hold_med": statistics.median([t["hold_minutes"] for t in trades]),
        "worst": round(min(pnls), 0),
        "best": round(max(pnls), 0),
    }


def breakeven_win_rate(trades: List[dict]) -> Optional[float]:
    """w* such that w*avg_win + (1-w)*avg_loss = 0, from the realised averages."""
    wins = [t["pnl_usd"] for t in trades if t["pnl_usd"] > 0]
    losses = [t["pnl_usd"] for t in trades if t["pnl_usd"] <= 0]
    if not wins or not losses:
        return None
    aw, al = statistics.mean(wins), statistics.mean(losses)
    return round(-al / (aw - al), 4)


def equity_stats(trades: List[dict]) -> dict:
    """Additive $-per-1-lot equity curve: max drawdown and annualised figures."""
    by_day: Dict[str, float] = defaultdict(float)
    for t in trades:
        by_day[t["session_date"]] += t["pnl_usd"]
    days = sorted(by_day)
    if not days:
        return {}
    eq, peak, mdd = 0.0, 0.0, 0.0
    for d in days:
        eq += by_day[d]
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    span_days = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days or 1
    years = span_days / 365.25
    return {
        "total": round(eq, 0),
        "per_year": round(eq / years, 0),
        "max_dd": round(mdd, 0),
        "trade_days": len(days),
    }


# ----------------------------------------------------------------------
# SANITY GATE
# ----------------------------------------------------------------------

def load_spy_by_day(dates: set) -> Dict[date, Dict[str, float]]:
    """{date: {minute: spy_close*10}} for the requested days."""
    import csv
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    out: Dict[date, Dict[str, float]] = defaultdict(dict)
    with open(SPY_1M, newline="") as f:
        for row in csv.DictReader(f):
            ts = row.get("ts_event", "")
            if not ts:
                continue
            base = ts.rstrip("Z").split(".")[0]
            try:
                dt = datetime.fromisoformat(base).replace(tzinfo=timezone.utc).astimezone(ET)
            except ValueError:
                continue
            if dt.date() not in dates:
                continue
            out[dt.date()][dt.strftime("%H:%M")] = float(row["close"]) * 10.0
    return out


def cmd_sanity(args) -> int:
    dates = all_dates()
    print(f"cache: {CACHE_DIR}  days={len(dates)}  {dates[0]} -> {dates[-1]}")

    # --- F3: reproduce the repo's established primitives -------------
    sample = dates[::len(dates) // 40 or 1][:40]
    spot_diffs, straddle_diffs = [], []
    minutes_per_day, strikes_per_day = [], []
    for d in sample:
        quotes = load_quotes(d)
        ctx = DayContext.build(d, quotes)
        store = CbboStore(CACHE_DIR, cache_dir=CACHE_DIR)
        store.load_day(d)
        minutes_per_day.append(len(ctx.minutes))
        strikes_per_day.append(len(ctx.strikes))
        for m in ("09:35", "10:30", "12:00", "14:00"):
            mine = ctx.spot_at(m)
            if mine is None:
                continue
            theirs = store.derive_spot(m, mine)
            if theirs is not None:
                spot_diffs.append(abs(mine - theirs))
            s_mine = ctx.atm_straddle(m)
            s_theirs = store.atm_straddle(m, mine)
            if s_mine is not None and s_theirs is not None:
                straddle_diffs.append(abs(s_mine - s_theirs))
    print("\n[F3] reproduction of established CbboStore primitives "
          f"({len(sample)} days x 4 minutes)")
    print(f"  parity spot   : n={len(spot_diffs)} max|diff|={max(spot_diffs):.4f} "
          f"median={statistics.median(spot_diffs):.4f}")
    print(f"  ATM straddle  : n={len(straddle_diffs)} max|diff|={max(straddle_diffs):.4f}")
    print(f"  coverage      : minutes median={statistics.median(minutes_per_day):.0f} "
          f"min={min(minutes_per_day)} | strikes median={statistics.median(strikes_per_day):.0f} "
          f"min={min(strikes_per_day)}")

    # --- parity spot vs SPYx10 ---------------------------------------
    spy_days = set(sample)
    spy = load_spy_by_day(spy_days)
    basis = []
    for d in sample:
        if d not in spy:
            continue
        quotes = load_quotes(d)
        ctx = DayContext.build(d, quotes)
        for m in ("09:35", "10:30", "12:00", "14:00"):
            a, b = ctx.spot_at(m), spy[d].get(m)
            if a and b:
                basis.append(a - b)
    if basis:
        basis.sort()
        print(f"\n[gate] parity spot - SPYx10, n={len(basis)} obs on "
              f"{len(spy_days & set(spy))} days")
        print(f"  median {statistics.median(basis):+.2f} pt | p5 {basis[len(basis)//20]:+.2f} "
              f"| p95 {basis[-len(basis)//20]:+.2f} | mean {statistics.mean(basis):+.2f}")
        print("  (parity spot is authoritative — it shares the chain's basis; "
              "the SPYx10 gap is the known basis offset, not an error)")
    else:
        print("\n[gate] SPY bars unavailable for the sample — parity spot used unchecked")

    # --- headline config over the full span: skips, ladder edge -------
    base = EmFadeConfig()
    shifts = [replace(base, band_shift_pct=s) for s in (-0.05, 0.0, 0.05)]
    rows = run(dates, shifts, jobs=args.jobs)
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["cfg"]].append(r)

    for cfg in shifts:
        rs = by_cfg[cfg.key()]
        trades = [r for r in rs if r.get("skip") is None]
        skips = Counter(r["skip"] for r in rs if r.get("skip"))
        touched = sum(1 for r in rs if r.get("touched") or r.get("skip") is None)
        interior = sum(1 for t in trades if t["interior_strike"])
        glitch = sum(t["glitch_minutes"] for t in trades)
        print(f"\n[gate] band_shift {cfg.band_shift_pct:+.2f}%  "
              f"days={len(rs)} touch_days={touched} trades={len(trades)}")
        print(f"  skips: {dict(skips.most_common())}")
        if trades:
            print(f"  ladder-interior: {interior}/{len(trades)} "
                  f"({interior / len(trades):.1%})   [gate >= 99%]")
            print(f"  glitch minutes skipped: {glitch} "
                  f"({glitch / max(sum(t['hold_minutes'] for t in trades), 1):.3%} of held minutes)")
            s = stats(trades)
            print(f"  pnl: n={s['n']} total=${s['total']:,.0f} mean=${s['mean']} "
                  f"win={s['win_rate']:.1%} t={s['t_stat']}")

    # --- both-hit-same-minute ambiguity ------------------------------
    trades0 = [r for r in by_cfg[base.key()] if r.get("skip") is None]
    both = sum(1 for t in trades0 if t["both_hit_same_minute"])
    print(f"\n[gate] TP and SL conditions true in the same minute: {both}/{len(trades0)} "
          f"(resolved adversely — SL wins)")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "em_fade_sanity.pkl", "wb") as f:
        pickle.dump(rows, f)
    print(f"\nwrote {OUT_DIR / 'em_fade_sanity.pkl'}")
    return 0


# ----------------------------------------------------------------------
# MEASURE
# ----------------------------------------------------------------------

DEPTHS = (0.5, 0.6, 0.7, 0.85, 1.0)
REGIMES = ("optimistic", "middle", "pessimistic")
TOUCH = "close"          # set from --touch; "intrabar" uses SPY high/low + per-minute basis


def suffix() -> str:
    return "" if TOUCH == "close" else f"_{TOUCH}"


def base_cfg(**kw) -> EmFadeConfig:
    kw.setdefault("touch_basis", TOUCH)
    return EmFadeConfig(**kw)


def cmd_measure(args) -> int:
    dates = all_dates()
    configs = []
    for f in DEPTHS:
        for regime in REGIMES:
            for trail in (False, True):
                configs.append(base_cfg(entry_depth_f=f, exit_regime=regime,
                                        trail_enabled=trail))
    # assumption-inversion arms, all at the middle regime / f=0.7 reference point
    ref = base_cfg(entry_depth_f=0.7, exit_regime="middle")
    sens = [
        ("entry delay +1 min", replace(ref, entry_delay_min=1)),
        ("entry delay +2 min", replace(ref, entry_delay_min=2)),
        ("EM = 0.8x straddle", replace(ref, em_scale=0.8)),
        ("commission $2.60 (spec A10)", replace(ref, commission_per_spread=2.60)),
        ("width 5", replace(ref, width_pts=5.0)),
        ("width 20", replace(ref, width_pts=20.0)),
        ("delta 0.10", replace(ref, target_delta=0.10, delta_min=0.05, delta_max=0.15)),
        ("delta 0.25", replace(ref, target_delta=0.25, delta_min=0.20, delta_max=0.32)),
        ("no stop (TP + EOD only)", replace(ref, sl_mult_of_credit=99.0)),
        ("TP 25% of credit", replace(ref, tp_pct_of_credit=0.25)),
        ("TP 75% of credit", replace(ref, tp_pct_of_credit=0.75)),
    ]
    configs += [c for _, c in sens]

    print(f"measure: {len(dates)} days x {len(configs)} configs")
    rows = run(dates, configs, jobs=args.jobs)
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"em_fade_measure{suffix()}.pkl", "wb") as f:
        pickle.dump({"rows": rows,
                     "sweep": [(c.entry_depth_f, c.exit_regime, c.trail_enabled, c.key())
                               for c in configs[:len(DEPTHS) * len(REGIMES) * 2]],
                     "sens": [(lab, c.key()) for lab, c in sens],
                     "ref_key": ref.key()}, f)

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["cfg"]].append(r)

    print("\n=== depth x exit-regime x trail (the fill bracket) ===")
    print(f"{'depth f':>8} {'regime':>12} {'trail':>6} {'n':>5} {'win%':>7} {'BEwin%':>7} "
          f"{'$/trade':>9} {'total':>10} {'t':>6} {'RoR':>7} {'hold':>5}")
    for cfg in configs[:len(DEPTHS) * len(REGIMES) * 2]:
        trades = [r for r in by_cfg[cfg.key()] if r.get("skip") is None]
        s = stats(trades)
        if not s:
            print(f"{cfg.entry_depth_f:>8} {cfg.exit_regime:>12} "
                  f"{str(cfg.trail_enabled):>6}  (no trades)")
            continue
        be = breakeven_win_rate(trades)
        print(f"{cfg.entry_depth_f:>8} {cfg.exit_regime:>12} {str(cfg.trail_enabled):>6} "
              f"{s['n']:>5} {s['win_rate']:>7.1%} {(f'{be:.1%}' if be else 'n/a'):>7} "
              f"{s['mean']:>9.2f} {s['total']:>10,.0f} "
              f"{s['t_stat']:>6.2f} {s['ror']:>7.3f} {s['hold_med']:>5.0f}")

    print("\n=== assumption-inversion arms (all f=0.7, middle regime, trail off) ===")
    print(f"{'arm':>30} {'n':>5} {'win%':>7} {'$/trade':>9} {'total':>10} {'t':>6}")
    for label, cfg in [("REFERENCE", ref)] + sens:
        trades = [r for r in by_cfg[cfg.key()] if r.get("skip") is None]
        s = stats(trades)
        if not s:
            print(f"{label:>30}  (no trades)")
            continue
        print(f"{label:>30} {s['n']:>5} {s['win_rate']:>7.1%} {s['mean']:>9.2f} "
              f"{s['total']:>10,.0f} {s['t_stat']:>6.2f}")
    return 0


# ----------------------------------------------------------------------
# ANALYZE — disaggregation of the reference config (structural move 4)
# ----------------------------------------------------------------------

def cmd_analyze(args) -> int:
    with open(OUT_DIR / f"em_fade_measure{suffix()}.pkl", "rb") as f:
        blob = pickle.load(f)
    rows, ref_key = blob["rows"], blob["ref_key"]
    rs = [r for r in rows if r["cfg"] == ref_key]
    trades = [r for r in rs if r.get("skip") is None]
    print(f"reference config: f=0.7, middle regime, trail off, 20d/10w/TP50/SL100")
    print(f"days={len(rs)} trades={len(trades)}")

    s = stats(trades)
    eq = equity_stats(trades)
    be = breakeven_win_rate(trades)
    print(f"\nHEADLINE  n={s['n']}  win={s['win_rate']:.1%}  BE-win={be:.1%}  "
          f"${s['mean']}/trade  t={s['t_stat']}  total=${s['total']:,.0f}")
    print(f"          avg win ${s['avg_win']}  avg loss ${s['avg_loss']}  "
          f"reward:risk 1:{abs(s['avg_loss']) / max(s['avg_win'], 1e-9):.2f}")
    print(f"          mean risk/trade ${s['mean_risk']:,.0f}  RoR {s['ror']:.2%}  "
          f"maxDD ${eq['max_dd']:,.0f}  ${eq['per_year']:,.0f}/yr per 1 lot")
    # C3 is stated as %/yr at 1-2% of account risked per trade -> convert on that basis
    for risk_pct in (0.01, 0.02):
        acct = s['mean_risk'] / risk_pct
        print(f"          at {risk_pct:.0%} of account risked per trade "
              f"(acct ${acct:,.0f}): {eq['per_year'] / acct:+.1%}/yr, "
              f"maxDD {eq['max_dd'] / acct:.1%}   [claim C3: +35%/yr]")

    # --- move 4: extremes vs structural bounds, and ceiling clustering
    pnl_pts = [(t["credit"] - t["exit_debit"], t) for t in trades]
    viol = [t for v, t in pnl_pts if v > t["credit"] + 1e-6 or v < -(t["width"] - t["credit"]) - 1e-6]
    on_ceiling = sum(1 for v, t in pnl_pts if abs(v - t["credit"]) < 1e-9)
    print(f"\n[move 4] no-arb violations: {len(viol)}   pnl exactly == credit "
          f"(structural ceiling): {on_ceiling}")
    print(f"          worst ${s['worst']:,.0f}  best ${s['best']:,.0f}  "
          f"(max possible win = credit x100; max loss = -(width-credit)x100)")

    def table(title, groups):
        print(f"\n--- {title}")
        print(f"{'group':>18} {'n':>5} {'win%':>7} {'$/trade':>9} {'total':>10} {'t':>6}")
        for g, ts in groups:
            st = stats(ts)
            if not st:
                print(f"{str(g):>18}  (none)")
                continue
            print(f"{str(g):>18} {st['n']:>5} {st['win_rate']:>7.1%} {st['mean']:>9.2f} "
                  f"{st['total']:>10,.0f} {st['t_stat']:>6.2f}")

    by_year = defaultdict(list)
    for t in trades:
        by_year[t["session_date"][:4]].append(t)
    table("per year (C3)", sorted(by_year.items()))

    by_side = defaultdict(list)
    for t in trades:
        by_side[t["side"]].append(t)
    table("put vs call (C5)", sorted(by_side.items()))

    def bucket(m):
        h = int(m[:2]) * 60 + int(m[3:5])
        if h < 10 * 60:
            return "1 09:35-10:00"
        if h < 11 * 60:
            return "2 10:00-11:00"
        if h < 12 * 60:
            return "3 11:00-12:00"
        return "4 12:00-13:00"
    by_touch = defaultdict(list)
    for t in trades:
        by_touch[bucket(t["touch_minute"])].append(t)
    table("touch time bucket (C7)", sorted(by_touch.items()))

    vix = load_vix_series(VIX_CSV)
    by_vix = defaultdict(list)
    for t in trades:
        v = lookup_prior_close(vix, date.fromisoformat(t["session_date"]))
        if v is None:
            by_vix["unknown"].append(t)
        elif v < 15:
            by_vix["1 VIX<15"].append(t)
        elif v < 20:
            by_vix["2 VIX 15-20"].append(t)
        elif v < 28:
            by_vix["3 VIX 20-28"].append(t)
        else:
            by_vix["4 VIX>=28"].append(t)
    table("VIX regime (C6)", sorted(by_vix.items()))

    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t["exit_reason"]].append(t)
    table("exit mix", sorted(by_exit.items()))

    holds = sorted(t["hold_minutes"] for t in trades)
    n = len(holds)
    print(f"\n--- hold time (C4): median {holds[n // 2]} min  "
          f"p25 {holds[n // 4]}  p75 {holds[3 * n // 4]}  p90 {holds[int(0.9 * n)]}  "
          f"max {holds[-1]}   share <=30min {sum(1 for h in holds if h <= 30) / n:.1%}")

    deltas = sorted(t["short_delta"] for t in trades)
    print(f"--- realised short delta: median {deltas[n // 2]:.3f} "
          f"p10 {deltas[n // 10]:.3f} p90 {deltas[int(0.9 * n)]:.3f}")
    creds = sorted(t["credit"] for t in trades)
    print(f"--- entry credit (pts): median {creds[n // 2]:.2f} "
          f"p10 {creds[n // 10]:.2f} p90 {creds[int(0.9 * n)]:.2f}  "
          f"=> $2.00 commission is {2.00 / (creds[n // 2] * 100):.2%} of median credit")
    return 0


# ----------------------------------------------------------------------
# CONTROLS
# ----------------------------------------------------------------------

def cmd_controls(args) -> int:
    dates = all_dates()
    base = base_cfg(entry_depth_f=0.7, exit_regime="middle")
    configs = [
        base,
        replace(base, direction="with_trend"),
        replace(base, entry_mode="fixed", fixed_side="put"),
        replace(base, entry_mode="fixed", fixed_side="call"),
        replace(base, entry_mode="random_minute", random_seed=7),
        replace(base, entry_mode="random_minute", random_seed=13),
        replace(base, entry_mode="random_minute", random_seed=29),
        replace(base, entry_mode="random_uncond", random_seed=7),
        replace(base, entry_mode="random_uncond", random_seed=13),
        replace(base, entry_mode="random_uncond", random_seed=29),
    ]
    labels = ["fade (the claim)", "with-trend mirror", "uncond 10:00 put",
              "uncond 10:00 call", "rand min>=touch s7", "rand min>=touch s13",
              "rand min>=touch s29", "rand day+side s7", "rand day+side s13",
              "rand day+side s29"]
    print(f"controls: {len(dates)} days x {len(configs)} arms")
    rows = run(dates, configs, jobs=args.jobs)
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"em_fade_controls{suffix()}.pkl", "wb") as f:
        pickle.dump(rows, f)

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["cfg"]].append(r)
    print(f"\n{'arm':>20} {'n':>5} {'win%':>7} {'$/trade':>9} {'total':>10} "
          f"{'t':>6} {'BE win%':>8}")
    for label, cfg in zip(labels, configs):
        trades = [r for r in by_cfg[cfg.key()] if r.get("skip") is None]
        s = stats(trades)
        if not s:
            print(f"{label:>20}  (no trades)")
            continue
        be = breakeven_win_rate(trades)
        print(f"{label:>20} {s['n']:>5} {s['win_rate']:>7.1%} {s['mean']:>9.2f} "
              f"{s['total']:>10,.0f} {s['t_stat']:>6.2f} "
              f"{(f'{be:.1%}' if be else 'n/a'):>8}")

    # buy-and-hold SPX over the same span, parity spot at first/last day
    q0, q1 = load_quotes(dates[0]), load_quotes(dates[-1])
    c0, c1 = DayContext.build(dates[0], q0), DayContext.build(dates[-1], q1)
    s0, s1 = c0.spot_at("09:35"), c1.spot_at("15:50")
    years = (dates[-1] - dates[0]).days / 365.25
    print(f"\nbuy-and-hold SPX (parity spot) {dates[0]} {s0:.0f} -> {dates[-1]} {s1:.0f}: "
          f"{(s1 / s0 - 1) * 100:.1f}% total, {((s1 / s0) ** (1 / years) - 1) * 100:.1f}%/yr "
          f"(price only, no dividends)")
    return 0


# ----------------------------------------------------------------------
# WALK-FORWARD
# ----------------------------------------------------------------------

def wf_grid() -> List[EmFadeConfig]:
    return [
        base_cfg(entry_depth_f=0.7, exit_regime="middle",
                 target_delta=d, delta_min=max(0.05, d - 0.07), delta_max=d + 0.07,
                 width_pts=w, trail_enabled=tr, em_multiplier=em)
        for d in (0.10, 0.15, 0.20, 0.25)
        for w in (5.0, 10.0, 20.0)
        for tr in (False, True)
        for em in (0.8, 1.0)
    ]


def _wf_eval(trades: List[dict]) -> float:
    """Objective: per-trade mean $ with a minimum-sample gate (D12)."""
    if len(trades) < 12:
        return float("-inf")
    return statistics.mean([t["pnl_usd"] for t in trades])


def cmd_wf(args) -> int:
    dates = all_dates()
    # the frozen arm must be simulated too — its delta band differs from every
    # grid cell, so it has its own key and is NOT a member of WF_GRID
    frozen_cfg = base_cfg(entry_depth_f=0.7, exit_regime="middle")
    grid = wf_grid()
    sim_set = list(grid) + [frozen_cfg]
    print(f"wf: {len(dates)} days x {len(sim_set)} configs "
          f"(pre-simulating trade logs once, then folding)")
    rows = run(dates, sim_set, jobs=args.jobs)
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"em_fade_wf_logs{suffix()}.pkl", "wb") as f:
        pickle.dump(rows, f)

    by_cfg: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("skip") is None:
            by_cfg[r["cfg"]].append(r)

    frozen = frozen_cfg.key()
    assert frozen in by_cfg, "frozen arm was not simulated"
    print(f"\nfrozen (spec default) = {frozen}")

    for train_n, test_n in ((100, 50), (130, 65), (160, 80)):
        picks, oos_reopt, oos_frozen = [], [], []
        i = 0
        while i + train_n + test_n <= len(dates):
            tr_days = set(d.isoformat() for d in dates[i:i + train_n])
            te_days = set(d.isoformat() for d in dates[i + train_n:i + train_n + test_n])
            best, best_score = None, float("-inf")
            for cfg in grid:
                sc = _wf_eval([t for t in by_cfg[cfg.key()] if t["session_date"] in tr_days])
                if sc > best_score:
                    best, best_score = cfg.key(), sc
            picks.append(best)
            oos_reopt += [t for t in by_cfg[best] if t["session_date"] in te_days]
            oos_frozen += [t for t in by_cfg[frozen] if t["session_date"] in te_days]
            i += test_n
        sr, sf = stats(oos_reopt), stats(oos_frozen)
        uniq = len(set(picks))
        print(f"\n--- ladder train/test {train_n}/{test_n}: {len(picks)} folds, "
              f"{uniq} distinct winners (stability {1 - uniq / max(len(picks),1):.0%})")
        for label, s in (("re-optimised", sr), ("frozen", sf)):
            if s:
                print(f"  OOS {label:>13}: n={s['n']:>4} win={s['win_rate']:.1%} "
                      f"${s['mean']:>7.2f}/tr total=${s['total']:>9,.0f} t={s['t_stat']:>5.2f}")
            else:
                print(f"  OOS {label:>13}: no trades")
        print(f"  winners: {Counter(picks).most_common(4)}")
    return 0



# ----------------------------------------------------------------------
# CLOSE — the four gaps an independent review named as still open
# ----------------------------------------------------------------------

def _block_bootstrap(day_pnl: Dict[str, float], block: int = 10,
                     draws: int = 20000, seed: int = 11) -> Tuple[float, float, float]:
    """Day-block bootstrap CI on the mean per trading day.

    Naive t-stats assume trade independence; 0DTE outcomes cluster by regime,
    so a moving-block bootstrap over calendar days is the honest interval.
    """
    import random as _r
    days = sorted(day_pnl)
    xs = [day_pnl[d] for d in days]
    n = len(xs)
    if n < block * 2:
        return (float("nan"),) * 3
    rng = _r.Random(seed)
    nblocks = (n + block - 1) // block
    means = []
    for _ in range(draws):
        acc = []
        for _b in range(nblocks):
            i = rng.randrange(0, n - block + 1)
            acc.extend(xs[i:i + block])
        means.append(sum(acc[:n]) / n)
    means.sort()
    return (means[int(0.025 * draws)], sum(means) / draws, means[int(0.975 * draws)])


def _paired(a: List[dict], b: List[dict]) -> Optional[dict]:
    """Day-matched paired difference a - b, with its own t-stat."""
    ad = {t["session_date"]: t["pnl_usd"] for t in a}
    bd = {t["session_date"]: t["pnl_usd"] for t in b}
    common = sorted(set(ad) & set(bd))
    if len(common) < 20:
        return None
    diffs = [ad[d] - bd[d] for d in common]
    m = statistics.mean(diffs)
    sd = statistics.pstdev(diffs)
    return {"n": len(common), "mean": round(m, 2), "sd": round(sd, 2),
            "t": round(m / (sd / math.sqrt(len(common))), 2) if sd else 0.0}


def _dir_of(t: dict) -> str:
    return "up" if t["band_level"] > t["open_spot"] else "down"


def cmd_close(args) -> int:
    dates = all_dates()
    ref = EmFadeConfig(entry_depth_f=0.7, exit_regime="middle")
    opt = EmFadeConfig(entry_depth_f=0.5, exit_regime="optimistic", trail_enabled=True)
    arms = [
        ("reference (close touch, no S/R)", ref),
        ("optimistic corner", opt),
        ("mirror (with-trend)", replace(ref, direction="with_trend")),
        ("S/R proxy: prior-day extreme", replace(ref, sr_mode="prior_day")),
        ("S/R proxy: prior-day +10pt buffer", replace(ref, sr_mode="prior_day", sr_buffer_pts=10.0)),
        ("S/R proxy, optimistic fills", replace(opt, sr_mode="prior_day")),
        ("intrabar touch", replace(ref, touch_basis="intrabar")),
        ("intrabar touch, mirror", replace(ref, touch_basis="intrabar", direction="with_trend")),
        ("intrabar touch, optimistic fills", replace(opt, touch_basis="intrabar")),
        ("intrabar + S/R proxy", replace(ref, touch_basis="intrabar", sr_mode="prior_day")),
    ]
    print(f"close: {len(dates)} days x {len(arms)} arms")
    rows = run(dates, [c for _, c in arms], jobs=args.jobs)
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "em_fade_close.pkl", "wb") as f:
        pickle.dump({"rows": rows, "arms": [(l, c.key()) for l, c in arms]}, f)

    by = defaultdict(list)
    for r in rows:
        by[r["cfg"]].append(r)

    print(f"\n=== gap 1+2: ex-ante S/R strike proxy, and intrabar touch detection ===")
    print(f"{'arm':>36} {'n':>5} {'win%':>7} {'$/trade':>9} {'total':>10} {'t':>6} {'skips':>26}")
    for label, cfg in arms:
        rs = by[cfg.key()]
        trades = [r for r in rs if r.get("skip") is None]
        sk = Counter(r["skip"] for r in rs if r.get("skip"))
        top = ", ".join(f"{k}:{v}" for k, v in sk.most_common(2))
        st = stats(trades)
        if not st:
            print(f"{label:>36}  (no trades)")
            continue
        print(f"{label:>36} {st['n']:>5} {st['win_rate']:>7.1%} {st['mean']:>9.2f} "
              f"{st['total']:>10,.0f} {st['t_stat']:>6.2f} {top:>26}")

    print(f"\n=== gap 3: day-block bootstrap 95% CI on $/trading-day (block=10, 20k draws) ===")
    for label, cfg in arms:
        trades = [r for r in by[cfg.key()] if r.get("skip") is None]
        if len(trades) < 40:
            continue
        dp = defaultdict(float)
        for t in trades:
            dp[t["session_date"]] += t["pnl_usd"]
        lo, mu, hi = _block_bootstrap(dp)
        flag = "  <-- CI CROSSES ZERO" if hi > 0 else ""
        print(f"{label:>36}  mean ${mu:>7.2f}  95% CI [${lo:>8.2f}, ${hi:>7.2f}]{flag}")

    print(f"\n=== gap 4: paired + skew-controlled tests ===")
    which = args.touch
    rc = ref if which == "close" else replace(ref, touch_basis="intrabar")
    mc = replace(rc, direction="with_trend")
    print(f"  [touch basis: {which}]")
    fade = [r for r in by[rc.key()] if r.get("skip") is None]
    mirror = [r for r in by[mc.key()] if r.get("skip") is None]
    pr = _paired(fade, mirror)
    print(f"  day-matched paired difference (fade - mirror): n={pr['n']} "
          f"mean=${pr['mean']} sd=${pr['sd']} t={pr['t']}")
    print("  (theta and commission cancel in the pairing; what remains is side choice)")

    print("\n  skew control — SAME instrument, opposite trigger direction:")
    print(f"  {'instrument':>12} {'trigger':>28} {'n':>5} {'win%':>7} {'$/trade':>9} {'t':>6}")
    groups = [
        ("put spread", "fade: sold after a DOWN touch", [t for t in fade if t["side"] == "put"]),
        ("put spread", "mirror: sold after an UP touch", [t for t in mirror if t["side"] == "put"]),
        ("call spread", "fade: sold after an UP touch", [t for t in fade if t["side"] == "call"]),
        ("call spread", "mirror: sold after a DOWN touch", [t for t in mirror if t["side"] == "call"]),
    ]
    for inst, trig, ts in groups:
        st = stats(ts)
        print(f"  {inst:>12} {trig:>28} {st['n']:>5} {st['win_rate']:>7.1%} "
              f"{st['mean']:>9.2f} {st['t_stat']:>6.2f}")
    return 0


# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sanity", "measure", "analyze", "controls", "wf", "close"])
    ap.add_argument("--touch", choices=["close", "intrabar"], default="close")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    global TOUCH
    TOUCH = args.touch
    return {"sanity": cmd_sanity, "measure": cmd_measure, "analyze": cmd_analyze,
            "controls": cmd_controls, "wf": cmd_wf, "close": cmd_close}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
