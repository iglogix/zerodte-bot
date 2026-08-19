"""Nested walk-forward: the META-parameters are chosen out-of-sample too.

The operator's point: if (training length, re-optimisation frequency) are
choices, they are parameters, and quoting the best (T,F) cell found on the whole
sample is optimisation one level up from the thing WF was meant to police.

So this nests the two loops:

  OUTER  at each rebalance, using ONLY prior rows, sweep (T,F) and pick the
         best-scoring pair. Then trade the next block with it.
  INNER  within a block, the ordinary WF method: every F rows, retrain on the
         trailing T rows and take the argmax-train-Sharpe combo.

Nothing at or after a trade day is ever used to choose that day's parameters —
neither the strategy parameters nor the meta-parameters. This is stricter than
a single frozen holdout and needs no new data, and unlike a one-shot holdout it
tests the SELECTION PROCEDURE rather than one lucky pair.

Reported against two controls on the identical folds:
  * random combo each rebalance (does optimising beat not optimising?)
  * fixed (T,F) chosen with hindsight (what the contaminated number looked like)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_PARAMS = 5
MIN_TRAIN = N_PARAMS * 10


def load(out_dir: Path) -> pd.DataFrame:
    series = {}
    for f in sorted(out_dir.glob("combo_*.csv")):
        df = pd.read_csv(f)
        df = df[df["status"] == "closed"]
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
        series[f.stem.replace("combo_", "")] = (
            df.groupby("session_date")["realized_pnl_usd"].sum())
    return pd.DataFrame(series).fillna(0.0).sort_index()


def _sharpe_cols(blk):
    sd = blk.std(axis=0, ddof=1)
    return np.divide(blk.mean(axis=0), sd, out=np.zeros_like(sd), where=sd > 0)


def trade_span(M, T, F, lo, hi, rng=None):
    """P&L over rows [lo,hi) applying the inner method with (T,F).

    Training always looks strictly backwards from the rebalance point, so it may
    reach before `lo` — that is history, not lookahead.
    """
    out = []
    pos = lo
    while pos < hi:
        if pos - T < 0:
            return None
        if rng is None:
            col = int(np.argmax(_sharpe_cols(M[pos - T:pos])))
        else:
            col = int(rng.integers(M.shape[1]))
        nxt = min(pos + F, hi)
        out.append(M[pos:nxt, col])
        pos = nxt
    return np.concatenate(out) if out else None


def score(ser):
    if ser is None or len(ser) < 20:
        return -np.inf
    s = ser.std(ddof=1)
    return float(ser.mean() / s * np.sqrt(len(ser))) if s > 0 else 0.0


def meta_select(M, end, trains, freqs, warm):
    """Best (T,F) judged only on rows[0:end].

    Evaluated from row T (the first point with a full training window) up to
    `end`, so every meta-candidate is scored on as much prior history as it can
    use — and never on anything at or after `end`.
    """
    best, best_s = None, -np.inf
    for T in trains:
        for F in freqs:
            lo = T
            if end - lo < 40:
                continue
            s = score(trade_span(M, T, F, lo, end))
            if s > best_s:
                best, best_s = (T, F), s
    return best, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--outer", type=int, default=40, help="OOS block length")
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()

    mat = load(Path(args.dir))
    M = mat.values
    n = M.shape[0]
    years = (mat.index[-1] - mat.index[0]).days / 365.25
    ann = np.sqrt(n / years)
    trains = [t for t in (80, 100, 130, 145, 160, 180) if t >= MIN_TRAIN]
    freqs = [1, 5, 10, 20, 40, 65]

    print(f"matrix {n} rows over {years:.2f}y | annualise x{ann:.2f}")
    print(f"meta-grid {len(trains)}x{len(freqs)}; outer block {args.outer}; "
          f"warmup {args.warmup}\n")

    live, ctrl_rand, ctrl_fixed, chosen = [], [], [], []
    rng = np.random.default_rng(0)
    start = args.warmup
    while start + args.outer <= n:
        sel, s = meta_select(M, start, trains, freqs, args.warmup)
        if sel is None:
            start += args.outer; continue
        T, F = sel
        seg = trade_span(M, T, F, start, start + args.outer)
        rnd = trade_span(M, T, F, start, start + args.outer, rng=rng)
        fix = trade_span(M, 145, 40, start, start + args.outer)   # hindsight pair
        if seg is not None:
            live.append(seg); ctrl_rand.append(rnd); ctrl_fixed.append(fix)
            chosen.append((mat.index[start], T, F, float(seg.sum())))
            print(f"  block {mat.index[start]}..{mat.index[start+args.outer-1]}  "
                  f"meta-pick T={T:3d} F={F:<3d} (trainScore {s:5.2f})  "
                  f"OOS ${seg.sum():>7,.0f}")
        start += args.outer

    if not live:
        print("  not enough data for a nested test"); return

    print("\n=== NESTED OOS (meta-parameters chosen out-of-sample) ===")
    for name, segs in (("nested method", live),
                       ("random combo (control)", ctrl_rand),
                       ("fixed T=145/F=40 (hindsight pair)", ctrl_fixed)):
        ser = np.concatenate([s for s in segs if s is not None])
        t = score(ser)
        print(f"  {name:36} total ${ser.sum():>8,.0f}  "
              f"Sharpe {t/np.sqrt(len(ser))*ann:5.2f}  t {t:5.2f}  n={len(ser)}")

    picks = {(T, F) for _, T, F, _ in chosen}
    wins = sum(1 for *_, p in chosen if p > 0)
    print(f"\n  meta-pick stability: {len(picks)} distinct (T,F) over "
          f"{len(chosen)} blocks -> {sorted(picks)}")
    print(f"  blocks profitable: {wins}/{len(chosen)}")


if __name__ == "__main__":
    main()
