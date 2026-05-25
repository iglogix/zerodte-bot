"""
Analytical iron-condor evaluator on the multi-anchor CBBO cache.

For each day, derives:
  - EM at 09:35 (ATM straddle mid)
  - Strike prices at 09:35 for the 4 IC legs (computes credit collected)
  - Spot at 09:35 (entry) and EOD (settlement)
  - Optionally: per-anchor intraday IC marks (11:00, 13:00, 15:30, 15:59)
    so we can detect TP/SL events analytically without per-minute simulation.

Outputs WR, P&L distribution, max consecutive losses, exit-reason breakdown.

This is data-quality-robust because it only depends on quotes at a few key
minutes — the multi-anchor backfill at ±500 pts covers every needed strike.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import defaultdict
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import BarLoader


QuoteKey = Tuple[str, float, bool]  # (minute_et, strike, is_call)


def _mid(bid: float, ask: float) -> Optional[float]:
    if bid <= 0 and ask <= 0:
        return None
    if bid <= 0:
        return ask
    if ask <= 0:
        return bid
    return (bid + ask) / 2


def _atm_strike(strikes: set, spot: float, step: float = 5.0) -> float:
    """Round spot to the nearest step that exists in the strikes set."""
    rounded = round(spot / step) * step
    if rounded in strikes:
        return rounded
    # fallback to nearest in set
    return min(strikes, key=lambda k: abs(k - spot))


def _quote_mid(quotes: Dict[QuoteKey, Tuple[float, float]],
               minute: str, strike: float, is_call: bool) -> Optional[float]:
    q = quotes.get((minute, strike, is_call))
    if q is None:
        # search nearby minutes (±5 minutes) — bears for handles where
        # the exact minute happens to be sparse but adjacent is fine
        hh, mm = int(minute[:2]), int(minute[3:5])
        for off in [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
            nmm, nhh = mm + off, hh
            while nmm < 0:
                nmm += 60; nhh -= 1
            while nmm >= 60:
                nmm -= 60; nhh += 1
            if nhh < 0 or nhh > 23:
                continue
            alt = f"{nhh:02d}:{nmm:02d}"
            q = quotes.get((alt, strike, is_call))
            if q is not None:
                break
    if q is None:
        return None
    return _mid(*q)


def _compute_em(quotes, minute: str, spot: float, step: float = 5.0) -> Optional[float]:
    """ATM straddle mid at given minute. None if either leg missing."""
    strikes_at = {k[1] for k in quotes.keys() if k[0] == minute}
    if not strikes_at:
        return None
    atm = _atm_strike(strikes_at, spot, step)
    c_mid = _quote_mid(quotes, minute, atm, True)
    p_mid = _quote_mid(quotes, minute, atm, False)
    if c_mid is None or p_mid is None:
        return None
    return c_mid + p_mid


def _ic_credit(quotes, minute: str, short_put: float, long_put: float,
               short_call: float, long_call: float,
               slippage_per_spread: float = 0.10) -> Optional[Dict]:
    """Returns credit info for IC at given minute, or None if any leg missing."""
    sp = _quote_mid(quotes, minute, short_put, False)
    lp = _quote_mid(quotes, minute, long_put, False)
    sc = _quote_mid(quotes, minute, short_call, True)
    lc = _quote_mid(quotes, minute, long_call, True)
    if None in (sp, lp, sc, lc):
        return None
    put_credit = sp - lp - slippage_per_spread
    call_credit = sc - lc - slippage_per_spread
    if put_credit <= 0 or call_credit <= 0:
        return None
    return {
        "put_credit": put_credit,
        "call_credit": call_credit,
        "total_credit": put_credit + call_credit,
        "sp_mid": sp, "lp_mid": lp, "sc_mid": sc, "lc_mid": lc,
    }


def _ic_mark(quotes, minute: str, sp_k: float, lp_k: float,
             sc_k: float, lc_k: float) -> Optional[float]:
    """Current IC value (cost to close) at given minute."""
    sp = _quote_mid(quotes, minute, sp_k, False)
    lp = _quote_mid(quotes, minute, lp_k, False)
    sc = _quote_mid(quotes, minute, sc_k, True)
    lc = _quote_mid(quotes, minute, lc_k, True)
    if None in (sp, lp, sc, lc):
        return None
    put_mark = max(sp - lp, 0)
    call_mark = max(sc - lc, 0)
    return put_mark + call_mark


def _max_consec_losses(seq: List[float]) -> int:
    best = cur = 0
    for p in seq:
        if p < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _max_drawdown(seq: List[float]) -> float:
    """Max peak-to-trough drawdown on running cumulative P&L."""
    cum = 0.0
    peak = 0.0
    worst_dd = 0.0
    for p in seq:
        cum += p
        if cum > peak:
            peak = cum
        dd = cum - peak  # negative or zero
        if dd < worst_dd:
            worst_dd = dd
    return worst_dd


def _evaluate(
    em_mult: float,
    wing_width: float,
    tp_pct: float,
    sl_mult: float,
    anchors: List[str],
    by_date: Dict[date, Dict],
    multiplier: int = 100,
    filters: Optional[Dict] = None,
    em_mult_call: Optional[float] = None,
    em_mult_put: Optional[float] = None,
) -> Dict:
    """Walk per-day data, applying TP/SL at intraday anchors and EOD exit otherwise.

    filters dict supports:
      skip_cpi_next_day, skip_fomc_next_day (bool)
      skip_trend_sigma (float, e.g. 1.5 — skip if prior day's move > X×prior_EM)
      low_em_pct (float, e.g. 0.80 — skip if today's EM < pct × 5-day EM mean)
      vix9d_max (float, e.g. 30 — skip if prior-day VIX9D > vix9d_max)
      vix_series (dict[date,float] — required if vix9d_max set)
    """
    from src_v2.backtest.engine import _FOMC_DATES, _CPI_DATES
    filters = filters or {}

    pnls = []
    exits = defaultdict(int)
    skips = defaultdict(int)
    # Per-exit-type P&L tracking for the exit-breakdown report
    exit_pnls = defaultdict(list)  # kind -> list of pnl
    credits_collected = []
    dates_sorted = sorted(by_date.keys())

    # Precompute prior-day moves and rolling-5d EM
    prior_move = {}
    prior_em = {}
    em_history = []
    for d in dates_sorted:
        info = by_date[d]
        prior_em_5d_mean = sum(em_history[-5:]) / len(em_history[-5:]) if em_history else None
        prior_em[d] = prior_em_5d_mean
        em_history.append(info["em_0935"])
        prior_move[d] = info["spot_close"] - info["spot_open"]

    skip_cpi_next = filters.get("skip_cpi_next_day", False)
    skip_fomc_next = filters.get("skip_fomc_next_day", False)
    skip_trend_sigma = filters.get("skip_trend_sigma")
    low_em_pct = filters.get("low_em_pct")
    vix9d_max = filters.get("vix9d_max")
    vix_series = filters.get("vix_series") or {}

    for d in dates_sorted:
        info = by_date[d]
        spot_open = info["spot_open"]
        spot_close = info["spot_close"]
        em = info["em_0935"]
        quotes = info["quotes"]
        step = 5.0

        # Filter gates
        from datetime import timedelta as _td
        prev = d - _td(days=1)
        # Find the *last trading day before d* for prior-move/VIX lookups
        prev_trading = None
        for off in range(1, 6):
            cand = d - _td(days=off)
            if cand in by_date:
                prev_trading = cand
                break

        if skip_cpi_next and prev in _CPI_DATES:
            skips["cpi_next_day"] += 1; continue
        if skip_fomc_next and prev in _FOMC_DATES:
            skips["fomc_next_day"] += 1; continue
        if skip_trend_sigma is not None and prev_trading is not None:
            pm = prior_move.get(prev_trading, 0)
            pe = by_date[prev_trading]["em_0935"]
            if pe and abs(pm / pe) > skip_trend_sigma:
                skips["trend_sigma"] += 1; continue
        if low_em_pct is not None and prior_em.get(d) is not None:
            if em < low_em_pct * prior_em[d]:
                skips["low_em"] += 1; continue
        if vix9d_max is not None and prev_trading is not None:
            v = vix_series.get(prev_trading)
            if v is not None and v > vix9d_max:
                skips["high_vix"] += 1; continue

        # Pick strikes by em_mult (snapped to 5-pt grid)
        put_em = em_mult_put if em_mult_put is not None else em_mult
        call_em = em_mult_call if em_mult_call is not None else em_mult
        sp_k = math.floor((spot_open - put_em * em) / step) * step
        lp_k = sp_k - wing_width
        sc_k = math.ceil((spot_open + call_em * em) / step) * step
        lc_k = sc_k + wing_width

        cred = _ic_credit(quotes, "09:35", sp_k, lp_k, sc_k, lc_k)
        if cred is None:
            continue
        total_credit = cred["total_credit"]
        credits_collected.append(total_credit)

        # Walk intraday anchors looking for first TP/SL event
        exit_event = None
        for anchor in anchors:
            if anchor == "09:35":
                continue  # entry minute
            mark = _ic_mark(quotes, anchor, sp_k, lp_k, sc_k, lc_k)
            if mark is None:
                continue
            if mark <= tp_pct * total_credit:
                exit_event = ("tp", anchor, mark)
                break
            if sl_mult > 0 and mark >= sl_mult * total_credit:
                exit_event = ("sl", anchor, mark)
                break

        if exit_event is None:
            # Hold to expiry: P&L from spot_close vs strikes
            put_loss = max(sp_k - spot_close, 0) - max(lp_k - spot_close, 0)
            call_loss = max(spot_close - sc_k, 0) - max(spot_close - lc_k, 0)
            debit = put_loss + call_loss
            pnl = (total_credit - debit) * multiplier
            kind = "expiry"
        else:
            kind, anchor, mark = exit_event
            pnl = (total_credit - mark) * multiplier

        exits[kind] += 1
        exit_pnls[kind].append(pnl)
        pnls.append(pnl)

    if not pnls:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mean = sum(pnls) / len(pnls)
    std = math.sqrt(sum((p - mean) ** 2 for p in pnls) / max(1, len(pnls) - 1))
    return {
        "n": len(pnls),
        "wr": 100 * len(wins) / len(pnls),
        "total": sum(pnls),
        "mean": mean,
        "sharpe_ann": math.sqrt(252) * mean / std if std > 0 else 0,
        "max_consec_L": _max_consec_losses(pnls),
        "max_dd": _max_drawdown(pnls),
        "avg_win": sum(wins) / max(1, len(wins)),
        "avg_loss": sum(losses) / max(1, len(losses)),
        "worst": min(pnls),
        "best": max(pnls),
        "exits": dict(exits),
        "skips": dict(skips),
        "exit_pnls": dict(exit_pnls),
        "avg_credit": (sum(credits_collected) / len(credits_collected)
                       if credits_collected else 0),
    }


def _load_data(cache_dir: Path, bars_path: str, start: date, end: date,
               anchors: List[str]) -> Dict[date, Dict]:
    """Build per-day data dict {date: {spot_open, spot_close, em_0935, quotes}}."""
    loader = BarLoader(bars_path, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))

    by_date = {}
    n_missing_cbbo = 0
    n_missing_em = 0
    n_missing_bars = 0
    for d in days:
        if not d.bars:
            n_missing_bars += 1
            continue
        b_0935 = next((b for b in d.bars if b["ts_et"].time() == time(9, 35)), None)
        b_eod = next((b for b in reversed(d.bars)
                      if b["ts_et"].time() >= time(15, 55)), None)
        if b_0935 is None or b_eod is None:
            n_missing_bars += 1
            continue
        pkl = cache_dir / f"{d.session_date.isoformat()}.pkl"
        if not pkl.exists():
            n_missing_cbbo += 1
            continue
        with open(pkl, "rb") as f:
            quotes = pickle.load(f)
        em = _compute_em(quotes, "09:35", b_0935["open"])
        if em is None or em <= 0:
            n_missing_em += 1
            continue
        by_date[d.session_date] = {
            "spot_open": b_0935["open"],
            "spot_close": b_eod["close"],
            "em_0935": em,
            "quotes": quotes,
        }
    return by_date, dict(
        missing_bars=n_missing_bars,
        missing_cbbo=n_missing_cbbo,
        missing_em=n_missing_em,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/cbbo_cache_5anchor")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--start", default="2022-04-18")
    p.add_argument("--end", default="2026-02-28")
    p.add_argument("--anchors", default="09:35,11:00,13:00,15:30,15:59")
    p.add_argument("--em-mults", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    p.add_argument("--wing-widths", default="20,25,30")
    p.add_argument("--tp-pct", type=float, default=0.50)
    p.add_argument("--sl-mult", type=float, default=2.0)
    # When set (single em_mult + single wing), print the per-exit-type loss
    # breakdown for that config instead of the full sweep grid.
    p.add_argument("--exit-breakdown", action="store_true",
                   help="print TP/SL/expiry loss breakdown for one config "
                        "(use with single --em-mults and --wing-widths)")
    p.add_argument("--em-mult-put", type=float, default=None,
                   help="asymmetric put-side em_mult (overrides --em-mults for put)")
    p.add_argument("--em-mult-call", type=float, default=None,
                   help="asymmetric call-side em_mult (overrides --em-mults for call)")
    args = p.parse_args()

    anchors = [a.strip() for a in args.anchors.split(",")]
    by_date, miss = _load_data(
        Path(args.cache), args.bars,
        date.fromisoformat(args.start), date.fromisoformat(args.end),
        anchors,
    )
    print(f"Days loaded: {len(by_date)}")
    print(f"Missing: {miss}\n")

    em_mults = [float(x) for x in args.em_mults.split(",")]
    wings = [float(x) for x in args.wing_widths.split(",")]

    if args.exit_breakdown:
        _print_exit_breakdown(
            em_mults[0], wings[0], args.tp_pct, args.sl_mult, anchors, by_date,
            args.em_mult_put, args.em_mult_call,
        )
        return

    print(f"TP={args.tp_pct} × credit, SL={args.sl_mult} × credit, anchors={anchors}")
    print(f"{'em':>5} {'wing':>5} {'n':>4} {'WR%':>6} {'total':>10} "
          f"{'Sharpe':>7} {'max_L':>5} {'max_DD':>8} {'avg_W':>7} {'avg_L':>7}")
    print("─" * 95)
    for em in em_mults:
        for w in wings:
            r = _evaluate(em, w, args.tp_pct, args.sl_mult, anchors, by_date,
                          em_mult_put=args.em_mult_put, em_mult_call=args.em_mult_call)
            if r["n"] == 0:
                continue
            print(f"{em:>5.2f} {w:>5.0f} {r['n']:>4d} {r['wr']:>5.1f}% "
                  f"${r['total']:>+9,.0f} "
                  f"{r['sharpe_ann']:>+7.2f} {r['max_consec_L']:>5d} "
                  f"${r['max_dd']:>+7,.0f} "
                  f"${r['avg_win']:>+5.0f} ${r['avg_loss']:>+5.0f}")


def _print_exit_breakdown(em, wing, tp_pct, sl_mult, anchors, by_date,
                          em_put=None, em_call=None):
    """Per-exit-type P&L + loss breakdown for a single config."""
    r = _evaluate(em, wing, tp_pct, sl_mult, anchors, by_date,
                  em_mult_put=em_put, em_mult_call=em_call)
    if r["n"] == 0:
        print("NO TRADES")
        return
    put_lbl = em_put if em_put is not None else em
    call_lbl = em_call if em_call is not None else em
    avg_credit = r["avg_credit"]
    print(f"Config: em_put={put_lbl} em_call={call_lbl} wing={wing} "
          f"TP={tp_pct}×credit SL={sl_mult}×credit")
    print(f"  Avg credit collected: ${avg_credit:.2f}  (×100 = ${avg_credit*100:,.0f})")
    print(f"  1× credit loss ≈ ${avg_credit*100:,.0f}  (this is what a full 2×-credit SL costs)\n")
    print(f"  Overall: n={r['n']}  WR={r['wr']:.1f}%  total=${r['total']:+,.0f}  "
          f"Sharpe={r['sharpe_ann']:.2f}  maxDD=${r['max_dd']:+,.0f}  avg_loss=${r['avg_loss']:+.0f}\n")
    print(f"  {'exit':>8} {'n':>5} {'total_pnl':>12} {'#loss':>6} {'avg_loss':>10} {'worst':>10}")
    print("  " + "─" * 56)
    all_losses = []
    for kind in ("tp", "sl", "expiry"):
        seq = r["exit_pnls"].get(kind, [])
        losses = [p for p in seq if p < 0]
        all_losses += losses
        al = sum(losses) / len(losses) if losses else 0
        wl = min(losses) if losses else 0
        print(f"  {kind:>8} {len(seq):>5d} ${sum(seq):>+10,.0f} {len(losses):>6d} "
              f"${al:>+8.0f} ${wl:>+8.0f}")
    blended = sum(all_losses) / len(all_losses) if all_losses else 0
    print(f"\n  Blended avg loss across all exit types: ${blended:+.0f}")


if __name__ == "__main__":
    main()
