"""As-taught IC placement: SYMMETRIC (spot +/- k*EM) vs Pulver's GAP-ADJUSTED
EXPECTED-MOVE BOX (top = open + k*EM, bottom = prev_close - k*EM).

Self-contained as-taught backtest (no stop, hold to settlement) priced from the
CBBO cache, so the only difference between arms is strike placement. EM = ATM
straddle mid at 09:35 (causal). Compares both arms across k and wing width.

Usage: python -m scripts.em_box_compare [--start ..] [--end ..]
"""
from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zerodte.backtest.cbbo_store import CbboStore

STEP = 5.0
SLIP = 0.10          # per spread
COMM = 1.0           # per leg
MULT = 100
ENTRY, EXIT, OPEN_MIN = "09:35", "15:59", "09:31"


def spread_mid(store, minute, short_k, long_k, is_call):
    s = store.lookup(minute, short_k, is_call)
    l = store.lookup(minute, long_k, is_call)
    if s is None or l is None:
        return None
    return (s[0] + s[1]) / 2 - (l[0] + l[1]) / 2


def ic_pnl(store, Kc, Kp, W):
    lc, lp = Kc + W, Kp - W
    cc_in = spread_mid(store, ENTRY, Kc, lc, True)
    pc_in = spread_mid(store, ENTRY, Kp, lp, False)
    if cc_in is None or pc_in is None:
        return None
    cc_in -= SLIP
    pc_in -= SLIP
    if cc_in < 0.05 or pc_in < 0.05:
        return None
    cc_out = spread_mid(store, EXIT, Kc, lc, True)
    pc_out = spread_mid(store, EXIT, Kp, lp, False)
    if cc_out is None or pc_out is None:
        return None
    credit = cc_in + pc_in
    debit = max(0.0, cc_out) + max(0.0, pc_out)
    return (credit - debit) * MULT - 4 * COMM


def stats(pnls, label):
    n = len(pnls)
    if n == 0:
        print(f"  {label:<28} no trades")
        return
    total = sum(pnls); mean = total / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0
    sharpe = mean / sd * math.sqrt(252) if sd > 0 else 0
    wins = sum(1 for p in pnls if p > 0)
    gw = sum(p for p in pnls if p > 0); gl = -sum(p for p in pnls if p < 0)
    pf = gw / gl if gl > 0 else float("inf")
    print(f"  {label:<28} n={n:>4} Sharpe {sharpe:>5.2f}  ${total:>+8.0f}  "
          f"WR {wins/n*100:4.1f}%  PF {pf:4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-05-11")
    ap.add_argument("--end", default="2026-06-11")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    mem = CbboStore.preload_cache("data/cbbo_cache", start=start, end=end)
    store = CbboStore("data/cbbo_cache", preloaded=mem)
    days = sorted(mem)
    print(f"  {len(days)} cbbo days {days[0]}..{days[-1]}\n")

    for k in (0.75, 1.0, 1.25):
        for W in (10, 20):
            sym, box = [], []
            prev_close = None
            hint = 4000.0  # SPX level hint for ATM-strike selection (parity is strike-independent)
            for d in days:
                store.load_day(d)
                spot = store.derive_spot(ENTRY, prev_close or hint)
                if spot:
                    hint = spot
                em = store.atm_straddle(ENTRY, spot) if spot else None
                opn = store.derive_spot(OPEN_MIN, prev_close or hint)
                close = store.settlement_forward()
                if spot and em and em > 0:
                    # symmetric
                    Kc = math.ceil((spot + k * em) / STEP) * STEP
                    Kp = math.floor((spot - k * em) / STEP) * STEP
                    r = ic_pnl(store, Kc, Kp, W)
                    if r is not None:
                        sym.append(r)
                    # gap-adjusted box (needs open + prev close)
                    if opn and prev_close:
                        Kc2 = math.ceil((opn + k * em) / STEP) * STEP
                        Kp2 = math.floor((prev_close - k * em) / STEP) * STEP
                        r2 = ic_pnl(store, Kc2, Kp2, W)
                        if r2 is not None:
                            box.append(r2)
                prev_close = close if close else prev_close
            print(f"k={k}  wing={W}")
            stats(sym, "  symmetric (spot+/-kEM)")
            stats(box, "  gap-box (open+kEM/pc-kEM)")
            print()


if __name__ == "__main__":
    main()
