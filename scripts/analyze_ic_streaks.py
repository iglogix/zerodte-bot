"""
Drill into losing streaks of the IC strategy.

For each losing day in a streak, report:
  - SPX open/close/realized range
  - EM at entry vs realized move (in σ)
  - Which side breached (put / call / neither but TP'd at loss)
  - Day of week
  - Proximity to calendar events (NFP/FOMC/CPI ±1)
  - Trend posture (close vs N-day MA)

Configurable em_mult / wing / management — defaults to the current best.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import Counter, defaultdict
from datetime import date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.engine import _FOMC_DATES, _CPI_DATES, _is_nfp_friday
from scripts.analytical_ic_eval import (
    _compute_em, _ic_credit, _ic_mark, _load_data, _max_consec_losses,
)

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _proximity(d: date) -> str:
    if _is_nfp_friday(d):
        return "NFP"
    if d in _FOMC_DATES:
        return "FOMC"
    if d in _CPI_DATES:
        return "CPI"
    for off in (-1, 1):
        adj = d + timedelta(days=off)
        if _is_nfp_friday(adj):
            return f"NFP{'+1' if off == -1 else '-1'}"
        if adj in _FOMC_DATES:
            return f"FOMC{'+1' if off == -1 else '-1'}"
        if adj in _CPI_DATES:
            return f"CPI{'+1' if off == -1 else '-1'}"
    return "—"


def _evaluate_with_details(em_mult, wing_width, tp_pct, sl_mult, anchors, by_date,
                          spx_closes: Dict[date, float]):
    """Same as _evaluate but returns per-day diagnostic rows."""
    rows = []
    step = 5.0
    dates_sorted = sorted(by_date.keys())
    for d in dates_sorted:
        info = by_date[d]
        spot_open = info["spot_open"]
        spot_close = info["spot_close"]
        em = info["em_0935"]
        quotes = info["quotes"]
        sp_k = math.floor((spot_open - em_mult * em) / step) * step
        lp_k = sp_k - wing_width
        sc_k = math.ceil((spot_open + em_mult * em) / step) * step
        lc_k = sc_k + wing_width
        cred = _ic_credit(quotes, "09:35", sp_k, lp_k, sc_k, lc_k)
        if cred is None:
            continue
        total_credit = cred["total_credit"]

        exit_event = None
        for anchor in anchors:
            if anchor == "09:35":
                continue
            mark = _ic_mark(quotes, anchor, sp_k, lp_k, sc_k, lc_k)
            if mark is None:
                continue
            if mark <= tp_pct * total_credit:
                exit_event = ("tp", anchor, mark); break
            if sl_mult > 0 and mark >= sl_mult * total_credit:
                exit_event = ("sl", anchor, mark); break

        if exit_event is None:
            put_loss = max(sp_k - spot_close, 0) - max(lp_k - spot_close, 0)
            call_loss = max(spot_close - sc_k, 0) - max(spot_close - lc_k, 0)
            debit = put_loss + call_loss
            pnl = (total_credit - debit) * 100
            exit_kind = "expiry"
        else:
            kind, anchor, mark = exit_event
            pnl = (total_credit - mark) * 100
            exit_kind = kind

        move = spot_close - spot_open
        move_sigma = move / em if em else 0

        # Breach analysis (which side, if any)
        put_breached = spot_close <= sp_k
        call_breached = spot_close >= sc_k
        # Also intraday (only have open/close — use day_low/day_high from bars if needed)
        rows.append({
            "date": d,
            "spot_open": spot_open,
            "spot_close": spot_close,
            "em": em,
            "move": move,
            "move_sigma": move_sigma,
            "sp_k": sp_k,
            "sc_k": sc_k,
            "credit": total_credit,
            "pnl": pnl,
            "exit": exit_kind,
            "weekday": WEEKDAY_NAMES[d.weekday()],
            "proximity": _proximity(d),
            "put_breached": put_breached,
            "call_breached": call_breached,
        })
    return rows


def _find_streaks(rows, min_length=3):
    """Find loss streaks of given min length. Returns list of [rows]."""
    streaks = []
    cur = []
    for r in rows:
        if r["pnl"] < 0:
            cur.append(r)
        else:
            if len(cur) >= min_length:
                streaks.append(cur)
            cur = []
    if len(cur) >= min_length:
        streaks.append(cur)
    return streaks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/cbbo_cache_5anchor")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--em-mult", type=float, default=0.20)
    p.add_argument("--wing", type=float, default=30)
    p.add_argument("--tp-pct", type=float, default=0.50)
    p.add_argument("--sl-mult", type=float, default=2.0)
    p.add_argument("--anchors", default="09:35,11:00,13:00,15:30,15:59")
    p.add_argument("--min-streak", type=int, default=3)
    args = p.parse_args()

    anchors = [a.strip() for a in args.anchors.split(",")]
    by_date, _ = _load_data(
        Path(args.cache), args.bars,
        date(2022, 4, 18), date(2026, 2, 28), anchors,
    )

    # Cache spx_closes for trend analysis
    loader = BarLoader(args.bars, price_scale=10.0, symbol="SPX")
    spx_closes = {}
    for d in loader.iter_days():
        if d.bars:
            spx_closes[d.session_date] = d.bars[-1]["close"]

    rows = _evaluate_with_details(
        args.em_mult, args.wing, args.tp_pct, args.sl_mult,
        anchors, by_date, spx_closes,
    )
    print(f"Days analyzed: {len(rows)}")
    pnls = [r["pnl"] for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    print(f"WR: {100*len(wins)/len(rows):.1f}%  Total: ${sum(pnls):+,.0f}  "
          f"Max consec losses: {_max_consec_losses(pnls)}")
    print()

    streaks = _find_streaks(rows, min_length=args.min_streak)
    streaks.sort(key=lambda s: -len(s))
    print(f"Loss streaks of ≥{args.min_streak} days: {len(streaks)}")
    print()

    # Aggregate loss-day characteristics
    loss_days = [r for r in rows if r["pnl"] < 0]
    print(f"Loss-day characteristics (n={len(loss_days)}):")
    by_weekday = Counter(r["weekday"] for r in loss_days)
    by_proximity = Counter(r["proximity"] for r in loss_days)
    side = Counter("put" if r["put_breached"] else ("call" if r["call_breached"] else "neither")
                   for r in loss_days)
    print(f"  Weekday distribution: {dict(by_weekday)}")
    print(f"  Calendar proximity:   {dict(by_proximity)}")
    print(f"  Breach side:          {dict(side)}")
    moves = [r["move_sigma"] for r in loss_days]
    print(f"  Move (σ-of-EM): mean={sum(moves)/len(moves):+.2f}, "
          f"abs_mean={sum(abs(m) for m in moves)/len(moves):.2f}, "
          f"max_down={min(moves):+.2f}, max_up={max(moves):+.2f}")
    print()

    print(f"All streaks (length ≥ {args.min_streak}, sorted by length):")
    for s in streaks[:15]:
        first, last = s[0]["date"], s[-1]["date"]
        total_loss = sum(r["pnl"] for r in s)
        side_counter = Counter("put" if r["put_breached"]
                               else ("call" if r["call_breached"] else "n/a")
                               for r in s)
        prox = Counter(r["proximity"] for r in s if r["proximity"] != "—")
        print(f"\n  {first} → {last}  ({len(s)} days, total {total_loss:+,.0f})")
        print(f"    Breach mix: {dict(side_counter)}, calendar: {dict(prox) or 'none'}")
        for r in s:
            tag = ""
            if r["put_breached"]: tag = "put"
            elif r["call_breached"]: tag = "call"
            else: tag = "no breach"
            print(f"    {r['date']} {r['weekday']:>3}  open={r['spot_open']:.1f} "
                  f"close={r['spot_close']:.1f} move={r['move']:+.1f} "
                  f"({r['move_sigma']:+.2f}σ) EM={r['em']:.1f}  "
                  f"pnl=${r['pnl']:+,.0f}  {tag}  exit={r['exit']}  "
                  f"prox={r['proximity']}")


if __name__ == "__main__":
    main()
