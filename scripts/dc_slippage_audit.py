#!/usr/bin/env python
"""DC Time Machine — slippage & clean-P&L audit (2026-06-09 follow-up).

Re-examines the "loses heavily at cross fills" finding and shows it was two artifacts:
  1. stale cache quotes inflate the transform's bid/ask slippage S_t, and
  2. the mental stop was evaluated at the adverse fill, so DCs stopped on bar 1.

With a mid-monitored stop and spreads capped to a realistic % of mid, the strategy is
~flat, not a blow-up. Reproduces the numbers in README "Addendum (2026-06-09)".

    PYTHONPATH=. .venv/bin/python scripts/dc_slippage_audit.py --cache data/dc_cache_full
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

from zerodte.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from zerodte.backtest.dc_time_machine.run_backtest import _sessions_in_cache
from zerodte.backtest.dc_time_machine.simulator import (
    resolve_expiries, select_strikes, session_minutes, _panel_ref, run_session)
from zerodte.backtest.dc_time_machine.fills import fill_price, net_combo_credit
from zerodte.backtest.dc_time_machine.pricing import entry_debit, ic_terminal_pnl
from zerodte.backtest.dc_time_machine.strategy import DcParams


class CappedStore(MultiExpiryStore):
    """Clamp each quote's spread to <= cap_frac of mid (mid preserved), so stale/thin
    cache quotes don't dominate the cross-fill cost."""
    def __init__(self, *a, cap_frac: float = 0.02, **kw):
        super().__init__(*a, **kw)
        self.cap_frac = cap_frac

    def lookup(self, *a, **kw):
        q = super().lookup(*a, **kw)
        if q is None:
            return None
        bid, ask = q
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return q
        cap = self.cap_frac * mid
        return (mid - cap / 2, mid + cap / 2) if (ask - bid) > cap else q


def transform_slippage(cache: Path) -> None:
    """Per-transform combo slippage S_t (= credit lost mid→cross) and its distribution."""
    store = MultiExpiryStore(cache_dir=cache)
    p = DcParams(fill_k=0.0)
    W = p.wing_width
    S, leg_pct = [], []
    for s in _sessions_in_cache(cache):
        r = run_session(store, p, s)
        if r is None or r.outcome != "transformed":
            continue
        store.load_session(s)
        m = r.transform_minute
        legs = [store.lookup(r.back, m, r.k_put, False), store.lookup(r.back, m, r.k_call, True),
                store.lookup(r.front, m, r.k_put - W, False), store.lookup(r.front, m, r.k_call + W, True)]
        S.append(sum((a - b) / 2 for b, a in legs))
        leg_pct += [100 * (a - b) / ((a + b) / 2) for b, a in legs if (a + b) > 0]
    clean = [x for x in S if x < 5]
    print("== transform slippage S_t (sum of 4 half-spreads) ==")
    print(f"  raw mean   {mean(S):.2f}/sh = ${100*mean(S):.0f}/contract  (inflated by stale quotes)")
    print(f"  median     {median(S):.2f}/sh = ${100*median(S):.0f}/contract")
    print(f"  clean (<5) {mean(clean):.2f}/sh = ${100*mean(clean):.0f}/contract   n={len(clean)}/{len(S)}")
    print(f"  per-leg spread % of mid: median {median(leg_pct):.1f}%  "
          f"(≤2%: {sum(x<=2 for x in leg_pct)}/{len(leg_pct)},  >10% stale: {sum(x>10 for x in leg_pct)}/{len(leg_pct)})")


def dc_profit_at_transform(cache: Path) -> None:
    store = MultiExpiryStore(cache_dir=cache)
    p = DcParams(fill_k=0.0)
    W = p.wing_width
    prof, frac = [], []
    for s in _sessions_in_cache(cache):
        r = run_session(store, p, s)
        if r is None or r.outcome != "transformed":
            continue
        store.load_session(s)
        m = r.transform_minute
        vc = (fill_price(*store.lookup(r.back, m, r.k_put, False), "sell", 0.0)
              + fill_price(*store.lookup(r.back, m, r.k_call, True), "sell", 0.0)
              - fill_price(*store.lookup(r.front, m, r.k_put, False), "buy", 0.0)
              - fill_price(*store.lookup(r.front, m, r.k_call, True), "buy", 0.0))
        prof.append(vc - r.entry_debit)
        frac.append((vc - r.entry_debit) / r.entry_debit if r.entry_debit else 0)
    print("\n== DC profit at the transform moment ==")
    print(f"  median {median(prof):+.2f}/sh = ${100*median(prof):+.0f}/contract  "
          f"(= {100*median(frac):+.0f}% of entry debit)")


def simulate_split(store, p, s, k_stop, k_tx):
    """Entry & stop marked at MID; the TRANSFORM pays k_tx of the (capped) spread."""
    store.load_session(s)
    front, back = resolve_expiries(store, s, p)
    if front is None or back is None:
        return None
    spot = store.derive_spot(front, p.entry_minute, ref=_panel_ref(store, front))
    if spot is None:
        return ("no_entry", 0.0)
    kp, kc = select_strikes(store, s, front, p, spot)
    if kp is None or kc is None:
        return ("no_entry", 0.0)
    W, em = p.wing_width, p.entry_minute
    q = [store.lookup(front, em, kp, False), store.lookup(front, em, kc, True),
         store.lookup(back, em, kp, False), store.lookup(back, em, kc, True)]
    if not all(q):
        return ("no_entry", 0.0)
    D = entry_debit(fill_price(*q[0], "sell", 0.0), fill_price(*q[1], "sell", 0.0),
                    fill_price(*q[2], "buy", 0.0), fill_price(*q[3], "buy", 0.0))
    thr, vc = D + W + p.transform_epsilon, 0.0
    for m in [x for x in session_minutes(store, front) if x > em]:
        bp, bc = store.lookup(back, m, kp, False), store.lookup(back, m, kc, True)
        fp, fc = store.lookup(front, m, kp, False), store.lookup(front, m, kc, True)
        if not all((bp, bc, fp, fc)):
            continue
        vc = (fill_price(*bp, "sell", k_stop) + fill_price(*bc, "sell", k_stop)
              - fill_price(*fp, "buy", k_stop) - fill_price(*fc, "buy", k_stop))
        if D > 0 and (vc - D) <= -p.stop_loss_frac * D:
            return ("stopped", round((vc - D) * 100, 2))
        pw, cw = store.lookup(front, m, kp - W, False), store.lookup(front, m, kc + W, True)
        if pw is None or cw is None:
            continue
        C = net_combo_credit([(*bp, "sell"), (*bc, "sell"), (*pw, "buy"), (*cw, "buy")], k_tx)
        if C >= thr:
            bank, S = C - D, store.settlement(front)
            pnl = bank * 100 if S is None else ic_terminal_pnl(S, kp, kc, W, bank) * 100
            return ("transformed", round(pnl, 2))
    return ("closed_eod", round((vc - D) * 100, 2))


def clean_pnl(cache: Path, cap_frac: float = 0.02) -> None:
    sess = _sessions_in_cache(cache)
    print(f"\n== clean P&L: stop @ MID, spreads capped {int(cap_frac*100)}%, transform @ k_tx ==")
    print(f"  {'k_tx':>5} {'n':>4} {'tf':>4} {'win%':>6} {'totPnL':>9} {'avg/tr':>8}  outcomes")
    for k_tx in (0.0, 0.5, 1.0):
        store = CappedStore(cache_dir=cache, cap_frac=cap_frac)
        res = [r for s in sess if (r := simulate_split(store, DcParams(), s, 0.0, k_tx)) is not None]
        real = [r for r in res if r[0] != "no_entry"]
        oc = dict(Counter(r[0] for r in real))
        pnls = [r[1] for r in real]
        tf = sum(1 for r in real if r[0] == "transformed")
        wins = sum(1 for x in pnls if x > 0)
        print(f"  {k_tx:>5.1f} {len(real):>4} {tf:>4} {100*wins/len(real):>6.1f} "
              f"{sum(pnls):>9.0f} {sum(pnls)/len(real):>8.1f}  {oc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/dc_cache_full")
    ap.add_argument("--cap", type=float, default=0.02, help="spread cap as fraction of mid")
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    transform_slippage(cache)
    dc_profit_at_transform(cache)
    clean_pnl(cache, a.cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
