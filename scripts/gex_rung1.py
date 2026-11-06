#!/usr/bin/env python3
"""Rung 1: does the standing all-expiry GEX regime predict vol-normalized realized
move, after controlling for EM? Pre-registered single hypothesis, sequential OOS.

Entry = 11:00 ET (first window with full-chain cbbo coverage; MEIC-style entry).
Regime variable X = (S - gamma_flip)/EM from the standing all-expiry surface.
Dependent Y = max intraday excursion / EM (primary) or terminal |move| / EM.
"""
from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest import gex_regime as gr
from zerodte.backtest.gex import iv_from_em
from zerodte.backtest.gex_dependent import realized_move
from zerodte.backtest.gex_stats import partial_effect, tercile_buckets, oos_partial_effect

ENTRY = "11:00"


def entry_bar(day, entry):
    eh, em = int(entry[:2]), int(entry[3:5])
    return next((b for b in day.bars if b["ts_et"].hour == eh and b["ts_et"].minute == em), None)


def sessions_after(calendar, d, expiry) -> int:
    return len([x for x in calendar if d < x <= expiry])


def build_rows(oi_dir, bars, cbbo_dir, start, end):
    loader = BarLoader(bars, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    calendar = [x.session_date for x in days]
    store = CbboStore(cbbo_dir, cache_dir=cbbo_dir)
    rows = []
    for day in days:
        d = day.session_date
        oi_pkl = Path(oi_dir) / f"{d.isoformat()}.pkl"
        if not oi_pkl.exists():
            continue
        with open(oi_pkl, "rb") as f:
            moi = pickle.load(f)
        if not moi:
            continue
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
        t_by = {e: gr.years_to_expiry(ENTRY, sessions_after(calendar, d, e))
                for e in gr.expiries(moi)}
        feat = gr.standing_regime(moi, spot, em, sigma, t_by)
        if feat.dist_flip_em is None:
            continue
        rm = realized_move(day.bars, ENTRY, em)
        if rm is None:
            continue
        rows.append({"date": d, "dist_flip_em": feat.dist_flip_em, "sign": feat.sign,
                     "em": em, "excursion": rm["max_excursion_em"],
                     "terminal": rm["terminal_abs_em"]})
    rows.sort(key=lambda r: r["date"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oi-dir", default="data/oi_cache_multi")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--cbbo", default="data/cbbo_cache_meic")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-06-13")
    p.add_argument("--y", choices=["excursion", "terminal"], default="excursion")
    args = p.parse_args()

    rows = build_rows(args.oi_dir, args.bars, args.cbbo,
                      date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"rows: {len(rows)}  (Y={args.y})")
    if len(rows) < 100:
        print("INSUFFICIENT DATA")
        return
    y = np.array([r[args.y] for r in rows])
    regime = np.array([r["dist_flip_em"] for r in rows])
    em = np.array([r["em"] for r in rows])

    full = partial_effect(y, regime, em)
    print(f"\n[FULL SAMPLE] regime coef={full['regime_coef']:+.4f} "
          f"t={full['regime_t']:+.2f} p={full['regime_p']:.4f} "
          f"interaction t={full['interaction_t']:+.2f}  n={full['n']}")

    b = tercile_buckets(em)
    print("\n[BY EM TERCILE]")
    for tb in (0, 1, 2):
        m = b == tb
        if m.sum() < 30:
            continue
        pe = partial_effect(y[m], regime[m], em[m])
        print(f"  tercile {tb} (n={int(m.sum())}, EM {em[m].min():.0f}-{em[m].max():.0f}): "
              f"coef={pe['regime_coef']:+.4f} t={pe['regime_t']:+.2f} p={pe['regime_p']:.4f}")

    oos = oos_partial_effect(y, regime, em, n_folds=4)
    print("\n[SEQUENTIAL OOS]")
    for f in oos["folds"]:
        if f.get("thin"):
            print(f"  fold {f['fold']}: thin (n={f['n']})")
            continue
        print(f"  fold {f['fold']} (n={f['n']}): coef={f['regime_coef']:+.4f} "
              f"t={f['regime_t']:+.2f} p={f['regime_p']:.4f}")
    print(f"\nVERDICT: consistent_majority={oos['consistent_majority']} "
          f"(pos_sig={oos['n_pos_sig']}, neg_sig={oos['n_neg_sig']}, valid={oos['n_valid']})")
    # Descriptive: mean Y by regime sign
    for s, lab in [(1, "suppression S>flip"), (-1, "amplification S<flip")]:
        msk = np.array([r["sign"] for r in rows]) == s
        if msk.sum():
            print(f"  mean {args.y}/EM [{lab}]: {y[msk].mean():.3f} (n={int(msk.sum())})")


if __name__ == "__main__":
    main()
