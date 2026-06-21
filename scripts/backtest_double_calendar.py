"""
Double Calendar Spread Backtest — SPX Weekly-to-Weekly
Based on: https://youtu.be/olVPaP7OSOM  ("How This Trader Makes 100%+ Returns with Double Calendars")

Strategy rules (reconstructed from transcript):
  Entry day: Tuesday or Wednesday
  VIX filter: prior close < 20 (avoid entering when VIX already elevated)
  No FOMC/CPI within the trade window
  Call calendar: sell front-Friday, buy back-Friday at spot + 100 (call OTM)
  Put calendar: sell front-Friday, buy back-Friday at spot − 100 (put OTM)
  Front expiry: the NEXT Friday (9–10 calendar days from a Tue/Wed entry)
  Back expiry:  the FOLLOWING Friday (7 calendar days after front, so ~16 CD from entry)
  Alt back:     the Monday after front (3 CD gap = "Friday/Monday" tight structure)

  Hold rules (checked at 10:00 ET each session):
    (a) Target: P&L ≥ +30% of initial debit → take profit
    (b) Touch: spot crosses either strike → exit
    (c) Stop:  P&L ≤ −20% of initial debit
    (d) Time:  exit 2 trading days before front expiry (regardless of P&L)

  Pricing: Black-Scholes, sigma = VIX daily close / 100, r = 0.04

Comparison baseline: Open IC on same sessions (0DTE, EM-centred, from canonical_ic.py).

Usage:
    python scripts/backtest_double_calendar.py
    python scripts/backtest_double_calendar.py --wide        # use Fri/Mon tight back
    python scripts/backtest_double_calendar.py --vix-max 25 # relax VIX filter
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.engine import _FOMC_DATES, _CPI_DATES
from src_v2.backtest.vix_loader import load_vix_series, lookup_prior_close
from src_v2.backtest.dc_time_machine.iv import bs_price

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CBBO_DIR = Path("data/cbbo_cache")
VIX_CSV = Path("data/vix_cboe.csv")
RESULTS_DIR = Path("results/double_calendar")
MULT = 100   # SPX = 100× multiplier

# ---------------------------------------------------------------------------
# Event sets (re-use from engine.py)
# ---------------------------------------------------------------------------
_EVENTS: frozenset[date] = frozenset(_FOMC_DATES | _CPI_DATES)


@dataclass
class DcParams:
    # Entry / Mode selection
    # Mode 1 (vix_prev < vix_max): Fri/Fri wide calendar — long vega
    # Mode 2 (vix_max <= vix_prev <= vix_hi_cap): Wed/Fri tight calendar — quasi-negative vega
    # Above vix_hi_cap: skip (too volatile)
    vix_max: float = 20.0       # VIX threshold dividing Mode 1 from Mode 2
    vix_hi_cap: float = 40.0    # VIX above this: skip Mode 2 as well
    entry_days: tuple = (1, 2)       # Mon=0, Tue=1, Wed=2 — enter on Tue/Wed
    entry_minute: str = "10:00"

    # Strikes
    offset_pts: float = 100.0        # call at spot+100, put at spot-100

    # Expiry (Mode 1 only; Mode 2 always uses Wed/Fri)
    tight_back: bool = False         # Mode 1 only: True = Fri/Mon (3 CD); False = Fri/Fri

    # Hold rules
    tp_pct: float = 0.30             # take profit at 30% of debit
    sl_pct: float = 0.20             # stop loss at -20% of debit
    td_before_front: int = 2         # exit N trading days before front expiry

    # BS
    rate: float = 0.04
    slip_per_leg: float = 0.10       # $/share per leg (fill imperfection)

    # Events
    block_events: bool = True        # skip entry if FOMC/CPI within first 3 CD


@dataclass
class TradeResult:
    session: date
    outcome: str                     # target | stop | touch | time | no_entry
    skip_reason: str = ""
    mode: str = ""                   # low_vix (Fri/Fri) | high_vix (Wed/Fri)
    front_expiry: Optional[date] = None
    back_expiry: Optional[date] = None
    spot: float = 0.0
    k_put: float = 0.0
    k_call: float = 0.0
    vix: float = 0.0
    entry_debit: float = 0.0        # $/share (per calendar unit)
    exit_value: float = 0.0
    pnl_share: float = 0.0          # $/share
    pnl_usd: float = 0.0            # × MULT × 1 contract
    hold_days: int = 0
    exit_date: Optional[date] = None
    pnl_pct: float = 0.0            # as fraction of debit


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _bs_call(S, K, T, sigma, r):
    return bs_price(S, K, T, sigma, r, True)

def _bs_put(S, K, T, sigma, r):
    return bs_price(S, K, T, sigma, r, False)

def _calendar_value(S: float, K_put: float, K_call: float,
                    T_front: float, T_back: float,
                    sigma: float, r: float) -> float:
    """Double calendar value: (back_put - front_put) + (back_call - front_call)."""
    if T_front <= 0:
        T_front = 1e-6
    if T_back <= T_front:
        T_back = T_front + 1 / 365
    put_cal = _bs_put(S, K_put, T_back, sigma, r) - _bs_put(S, K_put, T_front, sigma, r)
    call_cal = _bs_call(S, K_call, T_back, sigma, r) - _bs_call(S, K_call, T_front, sigma, r)
    return max(put_cal, 0) + max(call_cal, 0)


def _next_friday_min_cd(d: date, min_cd: int = 7) -> date:
    """First Friday at least min_cd calendar days after d.

    For a Wednesday entry, the immediate Friday is only 2 CD away (too soon).
    The video targets 'following week's Friday' ≈ 9 CD from a Tue/Wed entry.
    min_cd=7 ensures we skip this week's Friday when entry is mid-week.
    """
    days_ahead = (4 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    f = d + timedelta(days=days_ahead)
    # If the nearest Friday is too close, advance to the next one
    while (f - d).days < min_cd:
        f += timedelta(days=7)
    return f

def _following_friday(d: date) -> date:
    """The Friday 7 days after the given Friday (next week)."""
    return d + timedelta(days=7)

def _following_monday(d: date) -> date:
    """The Monday after a given Friday."""
    return d + timedelta(days=3)

def _next_wednesday_min_cd(d: date, min_cd: int = 3) -> date:
    """First Wednesday at least min_cd calendar days after d.
    From a Tuesday entry, skips tomorrow's Wednesday and lands on the following week's.
    """
    days_ahead = (2 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    w = d + timedelta(days=days_ahead)
    while (w - d).days < min_cd:
        w += timedelta(days=7)
    return w

def _trading_days_between(d1: date, d2: date, cal: List[date]) -> int:
    """Number of trading days in cal strictly between d1 and d2 (exclusive of both)."""
    return sum(1 for d in cal if d1 < d < d2)

def _trading_day_offset(ref: date, n: int, cal: List[date]) -> Optional[date]:
    """Return the trading day that is n calendar steps after ref, or None."""
    future = [d for d in cal if d > ref]
    if len(future) < n:
        return None
    return future[n - 1]


def _event_in_window(start: date, end: date, events: frozenset[date]) -> bool:
    """True if any major event date falls in [start, end]."""
    cur = start
    while cur <= end:
        if cur in events:
            return True
        cur += timedelta(days=1)
    return False


def _derive_spot(cb: CbboStore, minute: str) -> Optional[float]:
    """Derive SPX spot via put-call parity at nearest ATM strike."""
    if not cb._strikes:
        return None
    ref = sorted(cb._strikes)[len(cb._strikes) // 2]
    return cb.derive_spot(minute, ref)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def simulate_dc(params: DcParams, start: Optional[date] = None,
                end: Optional[date] = None) -> List[TradeResult]:

    vix = load_vix_series(VIX_CSV)
    cb = CbboStore(CBBO_DIR, cache_dir=CBBO_DIR)

    # Build trading calendar from available CBBO dates
    trading_cal = sorted(
        date.fromisoformat(p.stem) for p in CBBO_DIR.glob("*.pkl")
    )
    if start:
        trading_cal = [d for d in trading_cal if d >= start]
    if end:
        trading_cal = [d for d in trading_cal if d <= end]

    results: List[TradeResult] = []
    # Track open trades: {session: (front, back, K_put, K_call, debit, spot_entry, sigma)}
    open_trades: Dict[date, dict] = {}

    def _skip(session, reason):
        results.append(TradeResult(session=session, outcome="no_entry", skip_reason=reason))

    for session in trading_cal:
        # ----- Check + close open trades -----
        to_close: List[date] = []
        for entry_date, t in open_trades.items():
            front, back, K_p, K_c = t["front"], t["back"], t["K_put"], t["K_call"]
            sigma0, debit, r = t["sigma"], t["debit"], params.rate

            # Load today's session
            cb.load_day(session)
            spot = _derive_spot(cb, params.entry_minute)
            if spot is None:
                continue

            vix_today = lookup_prior_close(vix, session) or sigma0 * 100
            sigma = vix_today / 100

            hold = sum(1 for d in trading_cal if entry_date < d <= session)
            T_front = max((front - session).days, 0) / 365
            T_back = max((back - session).days, 0) / 365

            cal_value = _calendar_value(spot, K_p, K_c, T_front, T_back, sigma, r)
            pnl_pct = (cal_value - debit) / debit if debit > 0 else 0.0

            # Exit conditions (ordered by priority)
            td_to_front = sum(1 for d in trading_cal if session < d <= front)
            outcome = None

            if pnl_pct >= params.tp_pct:
                outcome = "target"
            elif spot <= K_p or spot >= K_c:
                outcome = "touch"
            elif pnl_pct <= -params.sl_pct:
                outcome = "stop"
            elif td_to_front <= params.td_before_front:
                outcome = "time"

            if outcome:
                pnl_s = cal_value - debit
                results.append(TradeResult(
                    session=entry_date, outcome=outcome,
                    mode=t.get("mode", ""),
                    front_expiry=front, back_expiry=back,
                    spot=t["spot_entry"], k_put=K_p, k_call=K_c,
                    vix=sigma0 * 100, entry_debit=debit,
                    exit_value=cal_value, pnl_share=pnl_s,
                    pnl_usd=round(pnl_s * MULT, 2),
                    hold_days=hold, exit_date=session, pnl_pct=pnl_pct,
                ))
                to_close.append(entry_date)

        for k in to_close:
            del open_trades[k]

        # ----- Consider new entry -----
        # Only enter on Tue/Wed
        if session.weekday() not in params.entry_days:
            continue
        # Must not already have an open trade
        if open_trades:
            continue

        # VIX filter + mode selection
        vix_prev = lookup_prior_close(vix, session)
        if vix_prev is None:
            _skip(session, "no VIX")
            continue
        if vix_prev > params.vix_hi_cap:
            _skip(session, f"VIX {vix_prev:.1f} > hi_cap {params.vix_hi_cap:.0f}")
            continue

        # Get spot
        cb.load_day(session)
        spot = _derive_spot(cb, params.entry_minute)
        if spot is None:
            _skip(session, "no spot")
            continue

        # Mode 1 (low VIX, long vega): sell next-week Friday, buy following Friday
        # Mode 2 (high VIX, quasi-negative vega): sell next Wednesday, buy Friday right after
        if vix_prev < params.vix_max:
            mode = "low_vix"
            front_date = _next_friday_min_cd(session, min_cd=7)
            back_date = _following_monday(front_date) if params.tight_back else _following_friday(front_date)
        else:
            mode = "high_vix"
            front_date = _next_wednesday_min_cd(session, min_cd=3)
            back_date = front_date + timedelta(days=2)  # Friday right after that Wednesday

        # Event filter: block only if event falls within 3 calendar days of entry
        # (video: "FOMC in a few days" = don't enter RIGHT BEFORE an event)
        # Calendars held through events are FINE — VIX spikes help them
        near_end = session + timedelta(days=3)
        if params.block_events and _event_in_window(session, near_end, _EVENTS):
            _skip(session, "event in window")
            continue

        # Strikes
        K_c = math.ceil((spot + params.offset_pts) / 5) * 5
        K_p = math.floor((spot - params.offset_pts) / 5) * 5

        sigma = vix_prev / 100
        T_front_entry = max((front_date - session).days, 1) / 365
        T_back_entry = max((back_date - session).days, 1) / 365
        debit = _calendar_value(spot, K_p, K_c, T_front_entry, T_back_entry, sigma, params.rate)

        # Add slippage: 4 legs × slip_per_leg
        debit += 4 * params.slip_per_leg

        if debit <= 0:
            _skip(session, "zero debit")
            continue

        open_trades[session] = {
            "front": front_date, "back": back_date,
            "K_put": K_p, "K_call": K_c,
            "sigma": sigma, "debit": debit,
            "spot_entry": spot, "mode": mode,
        }

    # Any still-open trades → close on last available day
    for entry_date, t in open_trades.items():
        front, back, K_p, K_c = t["front"], t["back"], t["K_put"], t["K_call"]
        sigma0, debit, r = t["sigma"], t["debit"], params.rate
        last = trading_cal[-1]
        cb.load_day(last)
        spot = _derive_spot(cb, params.entry_minute) or t["spot_entry"]
        vix_last = lookup_prior_close(vix, last) or sigma0 * 100
        sigma = vix_last / 100
        T_front = max((front - last).days, 0) / 365
        T_back = max((back - last).days, 0) / 365
        cal_val = _calendar_value(spot, K_p, K_c, T_front, T_back, sigma, r)
        pnl_s = cal_val - debit
        hold = sum(1 for d in trading_cal if entry_date < d <= last)
        results.append(TradeResult(
            session=entry_date, outcome="time",
            mode=t.get("mode", ""),
            front_expiry=front, back_expiry=back,
            spot=t["spot_entry"], k_put=K_p, k_call=K_c,
            vix=sigma0 * 100, entry_debit=debit,
            exit_value=cal_val, pnl_share=pnl_s,
            pnl_usd=round(pnl_s * MULT, 2),
            hold_days=hold, exit_date=last, pnl_pct=pnl_s / debit if debit else 0,
        ))

    # Return only real trades (exclude no_entry)
    return [r for r in results if r.outcome != "no_entry"]


def _run_open_ic_baseline(trading_cal: List[date]) -> Dict[date, float]:
    """Quick 0DTE Open IC run on same sessions for comparison.
    Returns {session_date: pnl_usd}.
    Returns empty dict if cbbo data is unavailable.
    """
    from src_v2.backtest.cbbo_store import CbboStore
    from src_v2.backtest.canonical_ic import price_ic

    cb = CbboStore(CBBO_DIR, cache_dir=CBBO_DIR)
    vix = load_vix_series(VIX_CSV)
    pnl_by_day: Dict[date, float] = {}

    for session in trading_cal:
        n_loaded = cb.load_day(session)
        if n_loaded == 0:
            continue
        spot = _derive_spot(cb, "10:00")
        if spot is None:
            continue
        vix_val = lookup_prior_close(vix, session)
        em = (vix_val / 100) * spot * math.sqrt(1 / 252) if vix_val else None
        if em is None:
            continue
        # Build chain dict for canonical_ic
        chain = {}
        for (k, is_call), v in {(k[1], k[2]): v
                                  for k, v in cb._quotes.items() if k[0] == "10:00"}.items():
            chain[(k, is_call)] = v

        # Try settlement from EOD
        settle_spot = _derive_spot(cb, "16:00")

        result = price_ic(chain, spot, em, em_mult=1.0, wing=25, step=5,
                          slippage=0.10, settle_spot=settle_spot, mult=MULT)
        if result and result.get("pnl") is not None:
            pnl_by_day[session] = result["pnl"]

    return pnl_by_day


def _sharpe(pnls: List[float]) -> float:
    if len(pnls) < 2:
        return float("nan")
    arr = np.array(pnls)
    std = arr.std()
    return (arr.mean() / std) * math.sqrt(252) if std > 0 else float("nan")


def _profit_factor(pnls: List[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = sum(-p for p in pnls if p < 0)
    return wins / losses if losses > 0 else float("inf")


def _report(label: str, trades: List, pnl_field="pnl_usd"):
    if not trades:
        print(f"{label}: no trades")
        return
    pnls = [getattr(t, pnl_field) if hasattr(t, pnl_field) else t for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins) / n * 100
    total = sum(pnls)
    avg = total / n
    pf = _profit_factor(pnls)
    sh = _sharpe(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:        {n}")
    print(f"  Win rate:      {wr:.1f}%")
    print(f"  Total P&L:     ${total:,.0f}")
    print(f"  Avg/trade:     ${avg:,.0f}")
    print(f"  Avg win:       ${avg_win:,.0f}")
    print(f"  Avg loss:      ${avg_loss:,.0f}")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Sharpe (ann):  {sh:.2f}")

    if hasattr(trades[0], 'outcome'):
        outcomes = {}
        for t in trades:
            outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
        print(f"  Outcomes:      {outcomes}")
        pcts = [t.pnl_pct * 100 for t in trades if hasattr(t, 'pnl_pct')]
        if pcts:
            print(f"  Avg P&L%:      {sum(pcts)/len(pcts):.1f}%")
            print(f"  Median P&L%:   {sorted(pcts)[len(pcts)//2]:.1f}%")
        vix_vals = [t.vix for t in trades if t.vix > 0]
        if vix_vals:
            print(f"  Avg entry VIX: {sum(vix_vals)/len(vix_vals):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tight", action="store_true",
                    help="Mode 1 only: use Fri/Mon back expiry instead of Fri/Fri")
    ap.add_argument("--vix-max", type=float, default=20.0,
                    help="VIX threshold: below=Mode1(Fri/Fri), above=Mode2(Wed/Fri)")
    ap.add_argument("--vix-hi-cap", type=float, default=40.0,
                    help="VIX above this: skip Mode 2 as well (extreme vol)")
    ap.add_argument("--tp", type=float, default=0.30, help="Take-profit fraction of debit")
    ap.add_argument("--sl", type=float, default=0.20, help="Stop-loss fraction of debit")
    ap.add_argument("--offset", type=float, default=100.0, help="OTM offset in SPX pts")
    ap.add_argument("--no-events", action="store_false", dest="block_events")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="results/double_calendar/trades.csv")
    args = ap.parse_args()

    params = DcParams(
        vix_max=args.vix_max,
        vix_hi_cap=args.vix_hi_cap,
        tight_back=args.tight,
        tp_pct=args.tp,
        sl_pct=args.sl,
        offset_pts=args.offset,
        block_events=args.block_events,
    )

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    print(f"Running Double Calendar backtest (dual-mode):")
    print(f"  Mode 1 (VIX<{params.vix_max}): Fri/{'Mon' if params.tight_back else 'Fri'} long-vega calendar")
    print(f"  Mode 2 ({params.vix_max}<=VIX<={params.vix_hi_cap}): Wed/Fri quasi-negative-vega calendar")
    print(f"  offset={params.offset_pts}pts, TP={params.tp_pct:.0%}, SL={params.sl_pct:.0%}")

    trades = simulate_dc(params, start=start, end=end)

    _report("Double Calendar — Combined (BS simulation)", trades)

    mode1 = [t for t in trades if t.mode == "low_vix"]
    mode2 = [t for t in trades if t.mode == "high_vix"]
    if mode1:
        _report(f"  Mode 1 (VIX<{params.vix_max}, Fri/Fri)", mode1)
    if mode2:
        _report(f"  Mode 2 (VIX>={params.vix_max}, Wed/Fri)", mode2)

    if trades:
        # Write CSV
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            fields = list(asdict(trades[0]).keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow(asdict(t))
        print(f"\n  Saved trades → {out}")

    # Open IC baseline on same sessions
    print("\nRunning Open IC baseline on same sessions…")
    session_dates = [t.session for t in trades]
    ic_pnl = _run_open_ic_baseline(session_dates)
    if ic_pnl:
        ic_vals = list(ic_pnl.values())
        _report("Open IC baseline (same sessions)", ic_vals)
    else:
        print("  No IC baseline data available")


if __name__ == "__main__":
    main()
