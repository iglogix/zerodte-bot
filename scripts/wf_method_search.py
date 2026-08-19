"""Search for an optimization METHOD that survives walk-forward, per docs/wf-cv.md.

WF does not ask "which parameters are good" — it asks whether a *procedure* for
choosing parameters produces OOS performance in every period. The procedure is
held constant across folds; the parameters it picks are free to vary. A method
that works in some folds and not others means no method was found.

What this adds over run_wf_method_strategy.py, which tests exactly one crude
method (bare argmax of train Sharpe):

  * STEP 5 IS ACTUALLY IMPLEMENTED. wf-cv.md step 5 says that when validation
    diverges from training you "adjust training parameters to avoid overfitting
    and run training again" — a loop. The old runner instead computed a WFE
    ratio and, on failure, fell back to a fixed baseline. That is not step 5,
    and the baseline it fell back to ranks #4 of 96 on FULL-SAMPLE P&L, so the
    fallback leaks hindsight into a supposedly out-of-sample tier.
  * Selection rules more robust than argmax: plateau (a cell is only as good as
    its neighbourhood), and a hard floor on trade count.
  * Failure is FLAT, never a hindsight-selected baseline.

Everything here uses train (and, for step 5 only, validation). Test is never
consulted for selection.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Ordered parameter grids — adjacency is what makes "plateau" meaningful.
GRID = {
    "ew": [0.4, 0.5, 0.6, 0.75],
    "w": [10, 15, 20],
    "tp": [0.5, 0.75],
    "me": [0, 20],
    "hpb": [0, 1],
}
TAG_RE = re.compile(r"ew([\d.]+)_w(\d+)_tp([\d.]+)_me(\d+)_hpb(\d+)_(.+)")


def sharpe_per_period(p: np.ndarray) -> float:
    """Per-period Sharpe. Deliberately NOT annualised: annualising by sqrt(252)
    a series sampled on trade days only inflates it (~1.63x on this grid)."""
    s = p.std(ddof=1)
    return float(p.mean() / s) if s > 0 else 0.0


def t_stat(p: np.ndarray) -> float:
    return sharpe_per_period(p) * np.sqrt(len(p)) if len(p) > 1 else float("nan")


def parse_tag(tag: str):
    m = TAG_RE.match(tag)
    if not m:
        return None
    ew, w, tp, me, hpb, retest = m.groups()
    try:
        return (GRID["ew"].index(float(ew)), GRID["w"].index(int(w)),
                GRID["tp"].index(float(tp)), GRID["me"].index(int(me)),
                GRID["hpb"].index(int(hpb)), retest)
    except ValueError:
        return None


def build_index(cols):
    idx = {}
    for c in cols:
        p = parse_tag(c)
        if p:
            idx[c] = p
    return idx


def neighbourhood(coord, idx, radius: int):
    """Cells within `radius` steps (Manhattan, in grid-index space)."""
    out = []
    for tag, other in idx.items():
        if other[5] != coord[5]:
            continue
        d = sum(abs(a - b) for a, b in zip(other[:5], coord[:5]))
        if d <= radius:
            out.append(tag)
    return out


def load_matrix(out_dir: Path):
    series, trades = {}, {}
    for f in sorted(out_dir.glob("combo_*.csv")):
        df = pd.read_csv(f)
        df = df[df["status"] == "closed"]
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
        tag = f.stem.replace("combo_", "")
        series[tag] = df.groupby("session_date")["realized_pnl_usd"].sum()
        trades[tag] = len(df)
    mat = pd.DataFrame(series).fillna(0.0).sort_index()
    return mat, trades


# ---------------------------------------------------------------------------
# Selection rules. Each takes the TRAIN slice and returns a combo tag or None.
# ---------------------------------------------------------------------------

def sel_argmax(tr, cols, idx, radius=0, min_trades=0, counts=None):
    """The existing runner's rule: single best train cell. The control."""
    return max(cols, key=lambda c: sharpe_per_period(tr[c].to_numpy()))


def sel_plateau(tr, cols, idx, radius=1, min_trades=0, counts=None):
    """Score a cell by its NEIGHBOURHOOD, not itself.

    An isolated spike in a noisy objective is the classic overfit signature;
    requiring the neighbours to agree is the standard defence, and it is a
    legitimate "adjust training parameters" move under wf-cv.md step 5.
    """
    scores = {c: sharpe_per_period(tr[c].to_numpy()) for c in cols}
    best, best_score = None, -np.inf
    for c in cols:
        if c not in idx:
            continue
        nb = [n for n in neighbourhood(idx[c], idx, radius) if n in scores]
        if not nb:
            continue
        s = float(np.mean([scores[n] for n in nb]))
        if s > best_score:
            best, best_score = c, s
    return best or sel_argmax(tr, cols, idx)


def sel_plateau_mintrades(tr, cols, idx, radius=1, min_trades=60, counts=None):
    """Plateau, but only over cells that actually traded enough to be measured."""
    keep = [c for c in cols if (counts or {}).get(c, 0) >= min_trades]
    if not keep:
        keep = list(cols)
    return sel_plateau(tr, keep, idx, radius=radius)


METHODS = {
    "argmax (control)": dict(fn=sel_argmax, step5=False),
    "plateau r=1": dict(fn=sel_plateau, step5=False),
    "plateau r=2": dict(fn=lambda *a, **k: sel_plateau(*a, **{**k, "radius": 2}),
                        step5=False),
    "plateau r=1 + min60 trades": dict(fn=sel_plateau_mintrades, step5=False),
    "argmax + step5": dict(fn=sel_argmax, step5=True),
    "plateau r=1 + step5": dict(fn=sel_plateau, step5=True),
}


def choose(method, tr, va, cols, idx, counts, wfe_floor, max_radius=3):
    """One fold's parameter choice.

    With step5 on, this is wf-cv.md steps 4-5 as written: optimise on train,
    check validation, and if validation diverges ADJUST THE TRAINING PROCEDURE
    (here: widen the plateau radius, i.e. demand a more robust optimum) and
    optimise again. Exhausting the adjustments means this fold produces no
    tradeable parameter set — the method goes FLAT rather than falling back to
    a hindsight-selected baseline.
    """
    fn, step5 = method["fn"], method["step5"]
    if not step5:
        return fn(tr, cols, idx, counts=counts), 0

    for radius in range(0, max_radius + 1):
        pick = (sel_plateau(tr, cols, idx, radius=radius, counts=counts)
                if radius else fn(tr, cols, idx, counts=counts))
        if pick is None:
            continue
        tr_sh = sharpe_per_period(tr[pick].to_numpy())
        va_sh = sharpe_per_period(va[pick].to_numpy())
        if tr_sh <= 0:
            continue
        wfe = va_sh / tr_sh
        if wfe >= wfe_floor:          # validation agrees with training
            return pick, radius
    return None, max_radius + 1        # no acceptable optimum -> flat


def run(mat, counts, T, S, method, wfe_floor, V=None, shift=None):
    """Walk forward with train=T, val=V, test=S, advancing by `shift` days.

    shift=1 with S=1 is the deployment the operator actually wants: re-optimise
    EVERY day on the most recent history and trade the next session. It is also
    the strictest test available here — one OOS observation per day instead of
    per fold, so hundreds of OOS points rather than 1-6.
    """
    idx = build_index(mat.columns)
    n = len(mat)
    V = S if V is None else V
    shift = S if shift is None else shift
    n_folds = (n - T - V - S) // shift + 1
    if n_folds < 1:
        return None
    off = n - (T + V + S + (n_folds - 1) * shift)
    segs, picks, flats = [], [], 0
    for k in range(n_folds):
        a = off + k * shift
        b, c, d = a + T, a + T + V, min(a + T + V + S, n)
        tr, va, te = mat.iloc[a:b], mat.iloc[b:c], mat.iloc[c:d]
        pick, radius = choose(method, tr, va, list(mat.columns), idx, counts,
                              wfe_floor)
        if pick is None:
            flats += 1
            segs.append(np.zeros(len(te)))       # flat fold = no position
            picks.append(("FLAT", radius))
        else:
            segs.append(te[pick].to_numpy())
            picks.append((pick, radius))
    ser = np.concatenate(segs)
    fold_totals = [float(s.sum()) for s in segs]
    return dict(total=float(ser.sum()), sharpe=sharpe_per_period(ser),
                t=t_stat(ser), n=len(ser), folds=fold_totals, picks=picks,
                flats=flats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--wfe-floor", type=float, default=0.5)
    args = ap.parse_args()

    mat, counts = load_matrix(Path(args.dir))
    n = len(mat)
    years = (mat.index[-1] - mat.index[0]).days / 365.25
    rate = n / years
    ann = np.sqrt(rate)
    print(f"matrix {n} rows over {years:.2f}y = {rate:.1f}/yr "
          f"-> annualise x{ann:.2f} (NOT sqrt(252)={np.sqrt(252):.2f})")
    print(f"methods under test: {len(METHODS)}  (multiple-testing exposure)\n")

    # (train, test, val, shift). The DAILY rows are the operator's stated
    # deployment: re-optimise every day on the most recent history.
    cadences = [(80, 40, 40, 40), (100, 50, 50, 50), (130, 65, 65, 65),
                (160, 80, 80, 80),
                (80, 1, 27, 1), (100, 1, 33, 1), (130, 1, 43, 1),
                (145, 1, 48, 1), (160, 1, 53, 1), (180, 1, 60, 1)]
    verdicts = {}
    for name, method in METHODS.items():
        print(f"=== {name} ===")
        ok_all, rows = True, []
        for T, S, V, shift in cadences:
            r = run(mat, counts, T, S, method, args.wfe_floor, V=V, shift=shift)
            if r is None:
                continue
            losing = sum(1 for f in r["folds"] if f <= 0)
            # Per-DAY folds: "no losing fold" is impossible for a real strategy,
            # so daily rows are judged on stitched OOS sign and significance.
            ok = (r["total"] > 0 and r["t"] > 2.0) if shift == 1 else (
                r["total"] > 0 and losing == 0)
            ok_all &= ok
            rows.append((T, S, r, losing, ok))
            tag = f"{T}/{S}" + ("*daily" if shift == 1 else "")
            print(f"  {tag:12} total ${r['total']:>8,.0f}  Sh {r['sharpe']*ann:5.2f}"
                  f"  t {r['t']:5.2f}  folds {len(r['folds']):3d}"
                  f"  losing {losing:3d}  flat {r['flats']:3d}  {'PASS' if ok else 'fail'}")
        uniq = {p for _, _, r, _, _ in rows for p, _ in r["picks"]}
        print(f"  distinct parameter picks across all folds: {len(uniq)}")
        verdicts[name] = ok_all
        print(f"  -> {'METHOD VALIDATED' if ok_all else 'not validated'}\n")

    print("=== summary (pre-registered: positive in ALL cadences, NO losing fold) ===")
    for k, v in verdicts.items():
        print(f"  {k:30} {'PASS' if v else 'fail'}")
    n_pass = sum(verdicts.values())
    print(f"\n  {n_pass}/{len(METHODS)} methods passed.")
    if n_pass:
        print("  NOTE: passing after searching several methods is itself selection;")
        print("        a passing method needs confirmation on data used in no search.")


if __name__ == "__main__":
    main()
