#!/usr/bin/env python3
"""Three-tier walk-forward METHOD validation per docs/wf-cv.md.

Validates the optimization METHOD (re-run a full param sweep periodically,
deploy the fresh winner), not any fixed parameter set:

  fold k:  train [k*S, k*S+T)   -> grid-search, pick best by train objective
           val   [k*S+T, k*S+T+V) -> consistency check (walk-forward efficiency
                                     = val Sharpe / train Sharpe); doc step 5
           test  [k*S+T+V, ...+S) -> UNTOUCHED: record the deployed winner's
                                     performance; this is the only OOS
  shift = S (= V), so test blocks tile a continuous OOS span (doc step 7).

Instantiated for the Markov regime gate over a frozen Power Hour trade log:
the method's grid is gate rule x window x threshold (+ "none" baseline), the
unit of P&L is one trading day, skipped days score $0 (account-level view).

Two deployment policies are reported:
  always : deploy the train winner unconditionally
  wfe    : deploy the train winner only if WFE >= --wfe-floor, else fall back
           to baseline (no gate) for that fold — the automated stand-in for
           the doc's "adjust and retrain" loop.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / ".claude/skills/markov-2-hedge-fund-method"))
from markov2 import SIDEWAYS, BULL, BEAR, label_states

ROOT = Path(__file__).resolve().parent.parent

RULES = ("none", "non_sw", "excl_stale_sw", "fresh_bear_only")
WINDOWS = (10, 15, 20, 25, 30)
THRESHOLDS = (0.03, 0.04, 0.05, 0.06, 0.07)


def load_spx_states():
    d = json.load(open("/tmp/gspc.json"))["chart"]["result"][0]
    dates = np.array(pd.to_datetime(d["timestamp"], unit="s").date)
    closes = np.array(d["indicators"]["quote"][0]["close"], dtype=float)
    ok = ~np.isnan(closes)
    return dates[ok], closes[ok]


def gate_masks(trade_dates, spx_dates, closes):
    """Precompute, per (rule, window, thr), the allow-mask over trade days."""
    ix = {dt: i for i, dt in enumerate(spx_dates)}
    masks = {("none", 0, 0.0): np.ones(len(trade_dates), bool)}
    for w in WINDOWS:
        for t in THRESHOLDS:
            states = label_states(closes, w, t, -t)
            s_c = np.array([states[ix[d] - 1] if d in ix and ix[d] >= 2 * w + 1 else -9
                            for d in trade_dates])
            s_p = np.array([states[ix[d] - 1 - w] if d in ix and ix[d] >= 2 * w + 1 else -9
                            for d in trade_dates])
            valid = s_c >= 0
            for rule in RULES[1:]:
                if rule == "non_sw":
                    m = s_c != SIDEWAYS
                elif rule == "excl_stale_sw":
                    m = ~((s_c == SIDEWAYS) & (s_p == SIDEWAYS))
                else:  # fresh_bear_only
                    m = (s_c != SIDEWAYS) | (s_p == BEAR)
                masks[(rule, w, t)] = m & valid | ~valid  # unlabelable -> allow
    return masks


def sharpe(pnl):
    return float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252))


def perf(pnl, mask):
    ser = np.where(mask, pnl, 0.0)
    eq = ser.cumsum()
    wins, losses = ser[ser > 0].sum(), -ser[ser < 0].sum()
    return dict(total=float(ser.sum()), sharpe=sharpe(ser),
                pf=float(min(wins / max(losses, 1e-9), 99.0)),
                max_dd=float((eq - np.maximum.accumulate(eq)).min()),
                days=int(mask.sum()), ser=ser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="results/power_hour_spx/trades_A_triggerplus.csv")
    ap.add_argument("--train-days", type=int, default=390)
    ap.add_argument("--wfe-floor", type=float, default=0.3,
                    help="min val/train Sharpe ratio to deploy the winner")
    args = ap.parse_args()

    spx_dates, closes = load_spx_states()
    df = pd.read_csv(ROOT / args.trades)
    df = df[df["status"] == "closed"]
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    daily = df.groupby("session_date")["realized_pnl_usd"].sum().sort_index()
    trade_dates = np.array(daily.index)
    pnl = daily.to_numpy(dtype=float)
    masks = gate_masks(trade_dates, spx_dates, closes)

    T = args.train_days
    V = S = T // 3
    n = len(pnl)
    n_folds = (n - T - V) // S
    # anchor the fold ladder at the END of the data: leftover days are dropped
    # from the oldest end so the final test block covers the newest sessions
    off = n - (T + V + n_folds * S)
    print(f"{args.trades}: {n} trade days {trade_dates[0]}..{trade_dates[-1]}")
    print(f"Three-tier WF (docs/wf-cv.md): train={T} val={V} test={S} shift={S} "
          f"-> {n_folds} folds, grid={len(masks)} combos\n")

    test_always, test_wfe, test_base = [], [], []
    winners = []
    for k in range(n_folds):
        a, b, c, d = (off + k * S, off + k * S + T,
                      off + k * S + T + V, off + k * S + T + V + S)
        tr, va, te = slice(a, b), slice(b, c), slice(c, min(d, n))
        scores = {key: sharpe(np.where(m[tr], pnl[tr], 0.0))
                  for key, m in masks.items()}
        win = max(scores, key=scores.get)
        m = masks[win]
        tr_sh = scores[win]
        va_sh = sharpe(np.where(m[va], pnl[va], 0.0))
        wfe = va_sh / tr_sh if tr_sh > 0 else float("nan")
        deploy_wfe = win if (np.isfinite(wfe) and wfe >= args.wfe_floor) else ("none", 0, 0.0)
        winners.append(win)

        p_win = perf(pnl[te], masks[win][te])
        p_wfe = perf(pnl[te], masks[deploy_wfe][te])
        p_base = perf(pnl[te], masks[("none", 0, 0.0)][te])
        test_always.append(p_win["ser"]); test_wfe.append(p_wfe["ser"])
        test_base.append(p_base["ser"])

        print(f"Fold {k + 1}: train {trade_dates[a]}..{trade_dates[b - 1]}  "
              f"val ..{trade_dates[c - 1]}  test ..{trade_dates[min(d, n) - 1]}")
        print(f"  winner on train: {win}  trainSh={tr_sh:.2f}  valSh={va_sh:.2f}  "
              f"WFE={wfe:.2f}{'  -> FALLBACK to baseline' if deploy_wfe != win else ''}")
        print(f"  test: winner ${p_win['total']:>9,.0f} (Sh {p_win['sharpe']:5.2f})  "
              f"wfe-policy ${p_wfe['total']:>9,.0f}  baseline ${p_base['total']:>9,.0f}")

    print("\n=== Stitched UNTOUCHED test tier (the method's OOS record) ===")
    for name, segs in (("method (always deploy winner)", test_always),
                       ("method (WFE-gated deploy)", test_wfe),
                       ("no optimization (baseline)", test_base)):
        ser = np.concatenate(segs)
        eq = ser.cumsum()
        print(f"  {name:32} total ${ser.sum():>9,.0f}  Sharpe {sharpe(ser):5.2f}  "
              f"maxDD ${(eq - np.maximum.accumulate(eq)).min():>9,.0f}  "
              f"({len(ser)}d)")
    stab = pd.Series([str(w) for w in winners]).value_counts()
    print("\nParameter stability across folds (doc: fluctuation = curve fitting):")
    for k_, v_ in stab.items():
        print(f"  {k_}: {v_}/{n_folds} folds")


if __name__ == "__main__":
    main()
