#!/usr/bin/env python3
"""Confirmation layer: price the canonical single-entry 0DTE IC each day, split P&L by
regime (suppression S>flip vs amplification S<flip, from the standing all-expiry GEX),
report the spread net of fills. Descriptive robustness check on the Rung 1 result.

Entry = 11:00 ET, full-chain cbbo_cache_meic.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest import gex_regime as gr
from src_v2.backtest.gex import iv_from_em
from src_v2.backtest.canonical_ic import price_ic

ENTRY = "11:00"


def entry_bar(day, entry):
    eh, em = int(entry[:2]), int(entry[3:5])
    return next((b for b in day.bars if b["ts_et"].hour == eh and b["ts_et"].minute == em), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oi-dir", default="data/oi_cache_multi")
    ap.add_argument("--bars", default="data/spy_1m.csv")
    ap.add_argument("--cbbo", default="data/cbbo_cache_meic")
    ap.add_argument("--start", default="2022-05-11")
    ap.add_argument("--end", default="2026-06-13")
    ap.add_argument("--em-mult", type=float, default=1.0)
    ap.add_argument("--wing", type=int, default=60)
    ap.add_argument("--slippage", type=float, default=0.10)
    a = ap.parse_args()

    loader = BarLoader(a.bars, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=date.fromisoformat(a.start),
                                 end_date=date.fromisoformat(a.end)))
    cal = [x.session_date for x in days]
    store = CbboStore(a.cbbo, cache_dir=a.cbbo)
    supp, ampl = [], []
    for day in days:
        d = day.session_date
        pk = Path(a.oi_dir) / f"{d.isoformat()}.pkl"
        if not pk.exists():
            continue
        moi = pickle.load(open(pk, "rb"))
        if not moi or store.load_day(d) == 0:
            continue
        b = entry_bar(day, ENTRY)
        if b is None:
            continue
        spot = store.derive_spot(ENTRY, float(b["open"])) or float(b["open"])
        atm = round(spot / 5) * 5
        cq = store.lookup(ENTRY, float(atm), True)
        pq = store.lookup(ENTRY, float(atm), False)
        if not cq or not pq:
            continue
        em = (cq[0] + cq[1]) / 2 + (pq[0] + pq[1]) / 2
        if em <= 0:
            continue
        sigma = iv_from_em(spot, em, gr.years_to_expiry(ENTRY, 0))
        if sigma <= 0:
            continue
        t_by = {e: gr.years_to_expiry(ENTRY, len([x for x in cal if d < x <= e]))
                for e in gr.expiries(moi)}
        feat = gr.standing_regime(moi, spot, em, sigma, t_by)
        if feat.dist_flip_em is None:
            continue
        chain = {}
        for k in range(int(spot - 400), int(spot + 401), 5):
            for ic in (True, False):
                q = store.lookup(ENTRY, float(k), ic)
                if q:
                    chain[(float(k), ic)] = q
        settle = store.settlement_forward()
        r = price_ic(chain, spot, em, a.em_mult, a.wing, 5, a.slippage, settle)
        if r is None or r["pnl"] is None:
            continue
        (supp if feat.dist_flip_em > 0 else ampl).append(r["pnl"])

    def line(name, xs):
        if not xs:
            print(f"  {name}: n=0")
            return
        arr = np.array(xs)
        print(f"  {name}: n={len(arr)} mean=${arr.mean():+.1f} median=${np.median(arr):+.1f} "
              f"win={(arr > 0).mean():.1%} total=${arr.sum():+.0f}")
    print("[CANONICAL IC P&L BY REGIME - net of slippage+commission, 1.0xEM short, 60pt wing]")
    line("suppression (S>flip)", supp)
    line("amplification (S<flip)", ampl)
    if supp and ampl:
        from scipy import stats
        t, p = stats.ttest_ind(supp, ampl, equal_var=False)
        print(f"  spread (supp-ampl) mean: ${np.mean(supp) - np.mean(ampl):+.1f}/trade  "
              f"(Welch t={t:+.2f}, p={p:.3f})")


if __name__ == "__main__":
    main()
