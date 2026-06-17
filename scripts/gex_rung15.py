#!/usr/bin/env python3
"""Rung 1.5: (a) does the 1DTE-as-0DTE proxy predict realized move; (b) OI-evolution
stability across each expiry's 2DTE->1DTE->0DTE-prior-close snapshots -> the gate
deciding whether Rung 2 (intraday trade reconstruction) is needed.

Entry = 11:00 ET (full-chain cbbo coverage). Stability tracks the expiry that is 0DTE
*today* (= session date d) as it appeared on the caches of d-2, d-1, d.
"""
from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest import gex_regime as gr
from src_v2.backtest.gex import iv_from_em
from src_v2.backtest.gex_dependent import realized_move
from src_v2.backtest.gex_stats import partial_effect, oos_partial_effect

ENTRY = "11:00"


def entry_bar(day, entry):
    eh, em = int(entry[:2]), int(entry[3:5])
    return next((b for b in day.bars if b["ts_et"].hour == eh and b["ts_et"].minute == em), None)


def load_moi(oi_dir, d):
    pk = Path(oi_dir) / f"{d.isoformat()}.pkl"
    if not pk.exists():
        return None
    with open(pk, "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oi-dir", default="data/oi_cache_multi")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--cbbo", default="data/cbbo_cache_meic")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-06-13")
    args = p.parse_args()

    loader = BarLoader(args.bars, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=date.fromisoformat(args.start),
                                 end_date=date.fromisoformat(args.end)))
    store = CbboStore(args.cbbo, cache_dir=args.cbbo)

    proxy_rows, stab = [], []
    for i, day in enumerate(days):
        d = day.session_date
        moi = load_moi(args.oi_dir, d)
        if not moi:
            continue
        next_exps = [e for e in gr.expiries(moi) if e > d]
        if not next_exps:
            continue
        e1 = next_exps[0]
        if store.load_day(d) == 0:
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
        feat = gr.proxy_regime(moi, e1, spot, em, sigma, ENTRY)
        rm = realized_move(day.bars, ENTRY, em)
        if feat.dist_flip_em is not None and rm is not None:
            proxy_rows.append({"date": d, "dist_flip_em": feat.dist_flip_em, "em": em,
                               "excursion": rm["max_excursion_em"]})

        # OI-evolution: the expiry that is 0DTE today (=d) across its prior snapshots
        snaps = []
        for back in (2, 1, 0):
            j = i - back
            if j < 0:
                continue
            m2 = load_moi(args.oi_dir, days[j].session_date)
            if not m2:
                continue
            oi_e = gr.slice_expiry(m2, d)
            if oi_e:
                snaps.append((days[j].session_date, oi_e))
        if len(snaps) >= 2:
            stab.append(gr.evolution_stability(snaps, spot, em, sigma,
                                               gr.years_to_expiry(ENTRY, 0)))

    print(f"proxy rows: {len(proxy_rows)}  stability samples: {len(stab)}")
    if len(proxy_rows) >= 100:
        proxy_rows.sort(key=lambda r: r["date"])
        y = np.array([r["excursion"] for r in proxy_rows])
        regime = np.array([r["dist_flip_em"] for r in proxy_rows])
        em = np.array([r["em"] for r in proxy_rows])
        full = partial_effect(y, regime, em)
        oos = oos_partial_effect(y, regime, em, n_folds=4)
        print(f"[PROXY] full coef={full['regime_coef']:+.4f} t={full['regime_t']:+.2f} "
              f"p={full['regime_p']:.4f}  OOS consistent={oos['consistent_majority']} "
              f"(pos_sig={oos['n_pos_sig']}, neg_sig={oos['n_neg_sig']})")

    if stab:
        sign_stable = sum(1 for s in stab if s["sign_stable"]) / len(stab)
        drifts = [s["max_flip_drift_em"] for s in stab if s["max_flip_drift_em"] is not None]
        med_drift = float(np.median(drifts)) if drifts else float("nan")
        n3 = sum(1 for s in stab if len(s["signs"]) >= 3)
        print(f"\n[OI-EVOLUTION STABILITY]  (samples={len(stab)}, with 3 snapshots={n3})")
        print(f"  sign stable across final days: {sign_stable:.1%}")
        print(f"  median flip drift: {med_drift:.2f} EM")
        gate = ("STICKY -> Rung 2 likely UNNECESSARY"
                if (sign_stable >= 0.7 and med_drift < 0.5)
                else "FLOW-DRIVEN -> Rung 2 REQUIRED")
        print(f"  GATE: {gate}")


if __name__ == "__main__":
    main()
