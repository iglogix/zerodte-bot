"""Treat (training length, re-optimisation frequency) as META-PARAMETERS and ask
whether the best cell beats chance.

Per docs/wf-cv.md the optimisation method is what WF validates. But the method
is specified by more than its selection rule: how much history it trains on and
how often it re-runs are choices too. Sweeping them and quoting the best cell is
optimisation one level up, and it inherits the same overfitting risk it was
meant to detect.

So this does two things:

  1. Sweeps train length x re-opt frequency, honouring the doc's floor
     (data points >= parameters x 10 = 50 for the 5 parameters here).
  2. Tests the best cell against a PERMUTATION NULL. Row order of the P&L
     matrix is shuffled, destroying any train->test predictability while
     preserving each combo's return distribution and the cross-sectional
     correlation within a day. Re-running the whole sweep on shuffled data
     gives the distribution of "best cell t-stat" under no skill. If the real
     best cell sits inside that distribution, the sweep found noise.

Selection uses train only; the test day is never consulted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_PARAMS = 5           # ew, width, tp, min_em, hold_breach
MIN_TRAIN = N_PARAMS * 10   # docs/wf-cv.md rule of thumb


def load(out_dir: Path) -> pd.DataFrame:
    series = {}
    for f in sorted(out_dir.glob("combo_*.csv")):
        df = pd.read_csv(f)
        df = df[df["status"] == "closed"]
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
        series[f.stem.replace("combo_", "")] = (
            df.groupby("session_date")["realized_pnl_usd"].sum())
    return pd.DataFrame(series).fillna(0.0).sort_index()


def sweep(M: np.ndarray, trains, freqs):
    """OOS series for every (train, freq). Selection = argmax train Sharpe.

    Vectorised: one mean/std over the train block scores all combos at once,
    which is what makes the permutation null affordable.
    """
    n = M.shape[0]
    out = {}
    for T in trains:
        for F in freqs:
            if T + F > n:
                continue
            pnl, params = [], []
            cur = None
            for start in range(0, n - T, F):
                blk = M[start:start + T]
                sd = blk.std(axis=0, ddof=1)
                sh = np.divide(blk.mean(axis=0), sd, out=np.zeros_like(sd),
                               where=sd > 0)
                cur = int(np.argmax(sh))
                lo, hi = start + T, min(start + T + F, n)
                pnl.append(M[lo:hi, cur])
                params.append(cur)
            if not pnl:
                continue
            ser = np.concatenate(pnl)
            if len(ser) < 30:
                continue
            s = ser.std(ddof=1)
            t = float(ser.mean() / s * np.sqrt(len(ser))) if s > 0 else 0.0
            out[(T, F)] = dict(total=float(ser.sum()), t=t, n=len(ser),
                               churn=len(set(params)) / max(len(params), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mat = load(Path(args.dir))
    M = mat.values
    n, k = M.shape
    years = (mat.index[-1] - mat.index[0]).days / 365.25
    ann = np.sqrt(n / years)

    trains = [t for t in (50, 65, 80, 100, 115, 130, 145, 160, 180, 200)
              if t >= MIN_TRAIN]
    freqs = [1, 5, 10, 20, 40, 65]
    print(f"matrix {n} rows x {k} combos over {years:.2f}y | annualise x{ann:.2f}")
    print(f"train floor = {N_PARAMS} params x 10 = {MIN_TRAIN} rows")
    print(f"meta-grid: {len(trains)} train lengths x {len(freqs)} frequencies\n")

    res = sweep(M, trains, freqs)
    print(f"  {'train':>5} " + " ".join(f"{('f=%d' % f):>9}" for f in freqs))
    for T in trains:
        row = f"  {T:>5} "
        for F in freqs:
            r = res.get((T, F))
            row += f"{(f'{r['t']:.2f}' if r else '--'):>9} " if r else f"{'--':>9} "
        print(row)
    print("  (cells are OOS t-stats; selection = argmax train Sharpe)\n")

    best = max(res.items(), key=lambda kv: kv[1]["t"])
    (bT, bF), b = best
    print(f"BEST CELL: train={bT} freq={bF}  t={b['t']:.2f}  "
          f"total ${b['total']:,.0f}  OOS n={b['n']}  "
          f"Sharpe(ann) {b['t']/np.sqrt(b['n'])*ann:.2f}")
    print(f"  cells searched: {len(res)}\n")

    # --- permutation null on the SAME search -------------------------------
    rng = np.random.default_rng(args.seed)
    null = []
    for i in range(args.perms):
        Mp = M[rng.permutation(n)]          # break time order, keep distributions
        rp = sweep(Mp, trains, freqs)
        if rp:
            null.append(max(v["t"] for v in rp.values()))
    null = np.array(null)
    p = float((null >= b["t"]).mean())
    print(f"PERMUTATION NULL ({len(null)} shuffles of the same meta-search):")
    print(f"  best-cell t under no skill: mean {null.mean():.2f}  "
          f"p50 {np.percentile(null,50):.2f}  p95 {np.percentile(null,95):.2f}  "
          f"max {null.max():.2f}")
    print(f"  observed best-cell t = {b['t']:.2f}  ->  p = {p:.3f}")
    print(f"  => {'beats chance' if p < 0.05 else 'INDISTINGUISHABLE FROM CHANCE'}")


if __name__ == "__main__":
    main()
