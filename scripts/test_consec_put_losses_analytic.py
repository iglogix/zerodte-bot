"""
Analytical (no-simulation) test of consecutive put-credit-spread losses.

We don't need per-minute CBBO if we only care about expiry-WR at no-SL hold:
  - EM at 09:35 := ATM_call_mid + ATM_put_mid
  - K_short    := spot_open − em_mult × EM
  - Win   ⇐  spot_close > K_short  (put OTM at expiry → expires worthless)
  - Loss  ⇐  spot_close ≤ K_short  (put ITM at expiry)

This bypasses the CBBO sparseness for far-OTM strikes entirely — we only need
the ATM straddle (always present in the cache) and the 1-min underlying bars.

Counts consecutive-loss streaks at multiple em_mult levels.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import SpyLoader


def _atm_straddle_mid(quotes: Dict, target_minute: str, spot: float) -> Optional[float]:
    """Get ATM straddle mid at given minute. Walks back up to 5 min if exact minute missing."""
    available_strikes = {k[1] for k in quotes.keys() if k[0] == target_minute}
    if not available_strikes:
        # Try previous minutes
        hh, mm = int(target_minute[:2]), int(target_minute[3:5])
        for off in range(1, 6):
            new_mm = mm - off
            new_hh = hh
            while new_mm < 0:
                new_mm += 60
                new_hh -= 1
            alt = f"{new_hh:02d}:{new_mm:02d}"
            available_strikes = {k[1] for k in quotes.keys() if k[0] == alt}
            if available_strikes:
                target_minute = alt
                break
        if not available_strikes:
            return None
    atm = min(available_strikes, key=lambda k: abs(k - spot))
    call = quotes.get((target_minute, atm, True))
    put = quotes.get((target_minute, atm, False))
    if call is None or put is None:
        return None
    call_mid = (call[0] + call[1]) / 2 if call[0] > 0 or call[1] > 0 else 0
    put_mid = (put[0] + put[1]) / 2 if put[0] > 0 or put[1] > 0 else 0
    if call_mid <= 0 or put_mid <= 0:
        return None
    return call_mid + put_mid


def _open_close(bars: List[Dict]) -> Optional[Tuple[float, float, float]]:
    """Return (spot_at_0935, spot_close, day_low) from 1-min bars. None if insufficient."""
    bar_0935 = None
    bar_eod = None
    for b in bars:
        if b["ts_et"].time() == time(9, 35):
            bar_0935 = b
        if b["ts_et"].time() >= time(15, 55):
            bar_eod = b  # keep walking to last
    if bar_0935 is None or bar_eod is None:
        return None
    day_low = min(b["low"] for b in bars
                  if time(9, 35) <= b["ts_et"].time() <= time(16, 0))
    return bar_0935["open"], bar_eod["close"], day_low


def _max_consec_losses(seq: List[bool]) -> int:
    """seq is True for win, False for loss. Return max consecutive False."""
    best = cur = 0
    for w in seq:
        if not w:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main():
    cbbo_dir = Path("data/cbbo_cache")
    print("Loading bars + CBBO straddles...")
    loader = SpyLoader("data/spy_1m.csv", price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=date(2022, 4, 18), end_date=date(2026, 2, 28)))

    # Pre-compute (spot_open, spot_close, day_low, em) for each day with valid data
    by_date: Dict[date, Tuple[float, float, float, float]] = {}
    missing = Counter()
    for d in days:
        sd = d.session_date
        bars = d.bars
        if not bars:
            missing["no_bars"] += 1
            continue
        oc = _open_close(bars)
        if oc is None:
            missing["no_open_or_close"] += 1
            continue
        spot_open, spot_close, day_low = oc

        pkl = cbbo_dir / f"{sd.isoformat()}.pkl"
        if not pkl.exists():
            missing["no_cbbo_file"] += 1
            continue
        with open(pkl, "rb") as f:
            quotes = pickle.load(f)
        em = _atm_straddle_mid(quotes, "09:35", spot_open)
        if em is None or em <= 0:
            missing["no_atm_straddle"] += 1
            continue
        by_date[sd] = (spot_open, spot_close, day_low, em)

    print(f"  Valid days: {len(by_date)} / {len(days)}")
    print(f"  Missing breakdown: {dict(missing)}")
    print()

    # Sort by date for chronological streak analysis
    dates_sorted = sorted(by_date.keys())

    print("Analytical put-credit-spread (hold to expiry, EOD-close decides win/loss):")
    print(f"{'em_mult':>8}  {'n':>5}  {'WR%':>6}  {'max_consec_L':>13}  "
          f"{'avg_EM':>7}  {'breach_via_low%':>15}")
    print("─" * 80)
    for em_mult in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 1.5, 2.0]:
        seq = []
        breach_via_low_only = 0  # close > K but intraday low went below
        ems = []
        for d in dates_sorted:
            spot_open, spot_close, day_low, em = by_date[d]
            k_short = spot_open - em_mult * em
            win_at_close = spot_close > k_short
            if win_at_close and day_low <= k_short:
                breach_via_low_only += 1
            seq.append(win_at_close)
            ems.append(em)
        wins = sum(seq)
        total = len(seq)
        wr = 100 * wins / total if total else 0
        avg_em = sum(ems) / len(ems) if ems else 0
        breach_pct = 100 * breach_via_low_only / total if total else 0
        print(f"{em_mult:>8.2f}  {total:>5d}  {wr:>5.1f}%  "
              f"{_max_consec_losses(seq):>13d}  "
              f"{avg_em:>6.1f}  {breach_pct:>14.1f}%")

    # Detailed at em=0.30 (closest to 25Δ)
    print()
    print("=" * 70)
    print("Detail: em_mult=0.30 — losing streaks (close-based)")
    print("=" * 70)
    em_mult = 0.30
    streaks = []
    cur_run = []
    for d in dates_sorted:
        spot_open, spot_close, day_low, em = by_date[d]
        k_short = spot_open - em_mult * em
        if spot_close <= k_short:
            cur_run.append((d, spot_open, spot_close, em, k_short))
        else:
            if len(cur_run) >= 3:
                streaks.append(cur_run)
            cur_run = []
    if len(cur_run) >= 3:
        streaks.append(cur_run)

    streaks.sort(key=lambda s: -len(s))
    for s in streaks[:5]:
        print(f"  Streak length {len(s)}:")
        for d, so, sc, em, k in s:
            move = sc - so
            mult = move / em if em else 0
            print(f"    {d}: open={so:.1f} close={sc:.1f} "
                  f"move={move:+.1f} ({mult:+.2f}×EM) K_short={k:.1f}")


if __name__ == "__main__":
    main()
