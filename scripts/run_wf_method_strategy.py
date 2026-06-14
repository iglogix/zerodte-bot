#!/usr/bin/env python3
"""Phase 2: three-tier walk-forward METHOD validation over Power Hour
STRATEGY parameters (docs/wf-cv.md protocol; combos pre-simulated by
gen_ph_combo_logs.py).

Question answered: does periodically re-optimizing credit_scale x stop_scale
x ratchet on the trailing window beat trading the as-taught constants
(credit 1.0 / stop 1.0 / ratchet on) forever? Verdict comes from the stitched
UNTOUCHED test tier only.

  fold k: train [a,b) -> pick combo by account-level Sharpe
          val   [b,c) -> WFE = val/train Sharpe (doc step 5); below the floor
                         the wfe-policy falls back to the as-taught combo
          test  [c,d) -> untouched, recorded
  shift = test length, so test blocks tile (doc step 7).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/power_hour_wf_method"
BASELINE = "c1.0_s1.0_r1"   # as-taught Version A


def sharpe(p):
    return float(p.mean() / (p.std() + 1e-9) * np.sqrt(252))


def load_matrix(out_dir=None):
    series = {}
    for f in sorted((out_dir or OUT).glob("combo_*.csv")):
        df = pd.read_csv(f)
        df = df[df["status"] == "closed"]
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
        series[f.stem.replace("combo_", "")] = (
            df.groupby("session_date")["realized_pnl_usd"].sum())
    mat = pd.DataFrame(series).fillna(0.0).sort_index()  # no-trade day = $0
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=130)
    ap.add_argument("--test-days", type=int, default=65)
    global BASELINE
    ap.add_argument("--wfe-floor", type=float, default=0.3)
    ap.add_argument("--dir", default=str(OUT),
                    help="directory of combo_*.csv logs")
    ap.add_argument("--baseline", default=BASELINE,
                    help="combo tag of the no-optimization baseline")
    args = ap.parse_args()
    BASELINE = args.baseline
    T, S = args.train_days, args.test_days
    V = S

    mat = load_matrix(Path(args.dir))
    assert BASELINE in mat.columns, f"missing baseline log {BASELINE}"
    n = len(mat)
    n_folds = (n - T - V) // S
    off = n - (T + V + n_folds * S)   # end-anchored: newest data always tested
    dates = np.array(mat.index)
    print(f"{len(mat.columns)} combos, {n} trade days "
          f"{dates[0]}..{dates[-1]}; train={T} val={V} test={S} "
          f"-> {n_folds} folds (end-anchored)\n")

    segs = {"always": [], "wfe": [], "base": []}
    winners = []
    for k in range(n_folds):
        a = off + k * S
        b, c, d = a + T, a + T + V, min(a + T + V + S, n)
        tr, va, te = mat.iloc[a:b], mat.iloc[b:c], mat.iloc[c:d]
        scores = {col: sharpe(tr[col].to_numpy()) for col in mat.columns}
        win = max(scores, key=scores.get)
        winners.append(win)
        tr_sh = scores[win]
        va_sh = sharpe(va[win].to_numpy())
        wfe = va_sh / tr_sh if tr_sh > 0 else float("nan")
        dep = win if (np.isfinite(wfe) and wfe >= args.wfe_floor) else BASELINE
        segs["always"].append(te[win].to_numpy())
        segs["wfe"].append(te[dep].to_numpy())
        segs["base"].append(te[BASELINE].to_numpy())
        print(f"Fold {k + 1:2d}: test {dates[c]}..{dates[d - 1]}  "
              f"winner {win:18} trainSh {tr_sh:5.2f}  WFE {wfe:5.2f}"
              f"{'  -> fallback' if dep != win else ''}  "
              f"test$ {te[win].sum():>8,.0f} (base {te[BASELINE].sum():>8,.0f})")

    print("\n=== Stitched UNTOUCHED test tier ===")
    for name, label in (("always", "method (deploy fold winner)"),
                        ("wfe", "method (WFE-gated, fallback=as-taught)"),
                        ("base", "as-taught constants (no optimization)")):
        ser = np.concatenate(segs[name])
        eq = ser.cumsum()
        print(f"  {label:42} total ${ser.sum():>9,.0f}  Sharpe {sharpe(ser):5.2f}  "
              f"maxDD ${(eq - np.maximum.accumulate(eq)).min():>9,.0f}  ({len(ser)}d)")

    print("\nWinner stability (doc: heavy fluctuation = curve fitting):")
    for w, cnt in pd.Series(winners).value_counts().items():
        print(f"  {w}: {cnt}/{n_folds}")


if __name__ == "__main__":
    main()
