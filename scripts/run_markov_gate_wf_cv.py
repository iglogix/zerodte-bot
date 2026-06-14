#!/usr/bin/env python3
"""WF-CV for the Markov regime gates over existing 0DTE backtest trade logs.

Protocol mirrors the project's MEIC WF-CV (scripts/run_v2_wf_cv.py output):
expanding-origin, val_len = N // 9 trade days (5:1 train:val on fold 1),
4 sequential validation blocks, gate selected on train by Sharpe, judged OOS
on the untouched validation block.

Gates only decide WHICH days the strategy trades; skipped days score $0 and
stay in the daily series, so low-activity gates are not flattered. All gate
inputs are causal (prior-close state; state 20 sessions before that).

Candidate family (declared up front; includes the in-sample winner — WF-CV
tests whether train-only data would have picked it and whether it held up):
  baseline        trade every day
  non_sw          skip SIDEWAYS days
  excl_stale_sw   skip days where SIDEWAYS now AND 20 sessions ago
  fresh_bear_only trade non-SIDEWAYS plus SIDEWAYS-arrived-from-BEAR
  excl_bear       skip BEAR days (the conventional filter, as control)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / ".claude/skills/markov-2-hedge-fund-method"))
from markov2 import SIDEWAYS, BULL, BEAR, label_states

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 20
FOLDS = 4
MIN_ACTIVE_FRAC = 0.05  # a gate must trade >=5% of train days to be selectable

STRATEGIES = {
    "power_hour_A": ROOT / "results/power_hour_spx/trades_A_triggerplus.csv",
    "meic": ROOT / "results/meic_spx/trades.csv",
}

GATES = {
    "baseline": lambda f: pd.Series(True, f.index),
    "non_sw": lambda f: f["gate_state"] != SIDEWAYS,
    "excl_stale_sw": lambda f: ~((f["gate_state"] == SIDEWAYS) & (f["prev20"] == SIDEWAYS)),
    "fresh_bear_only": lambda f: (f["gate_state"] != SIDEWAYS)
                                 | (f["prev20"] == BEAR),
    "excl_bear": lambda f: f["gate_state"] != BEAR,
}


def build_frame(csv, dates, states, date_ix):
    df = pd.read_csv(csv)
    df = df[df["status"] == "closed"]
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    pnl = df.groupby("session_date")["realized_pnl_usd"].sum().sort_index()
    rows = []
    for d, x in pnl.items():
        i = date_ix.get(d)
        if i is None or i < WINDOW + 1:
            continue
        s_c, s_p = states[i - 1], states[i - 1 - WINDOW]
        if s_c < 0 or s_p < 0:
            continue
        rows.append((d, x, s_c, s_p))
    return pd.DataFrame(rows, columns=["date", "pnl", "gate_state", "prev20"])


def perf(pnl_series, mask):
    """Account-level stats: skipped days = $0, kept in the series."""
    ser = pnl_series.where(mask, 0.0)
    eq = ser.cumsum()
    wins, losses = ser[ser > 0].sum(), -ser[ser < 0].sum()
    return dict(days=int(mask.sum()), total=ser.sum(),
                sharpe=ser.mean() / (ser.std() + 1e-9) * np.sqrt(252),
                pf=min(wins / max(losses, 1e-9), 99.0),
                max_dd=(eq - eq.cummax()).min())


def main():
    d = json.load(open("/tmp/gspc.json"))["chart"]["result"][0]
    dates = np.array(pd.to_datetime(d["timestamp"], unit="s").date)
    closes = np.array(d["indicators"]["quote"][0]["close"], dtype=float)
    ok = ~np.isnan(closes)
    dates, closes = dates[ok], closes[ok]
    states = label_states(closes, WINDOW, 0.05, -0.05)
    date_ix = {dt: i for i, dt in enumerate(dates)}

    for name, csv in STRATEGIES.items():
        f = build_frame(csv, dates, states, date_ix)
        n = len(f)
        val_len = n // 9
        train_end0 = 5 * val_len
        print(f"\n{'=' * 72}\n{name}: {n} trade days "
              f"({f['date'].iloc[0]} .. {f['date'].iloc[-1]}), "
              f"WF-CV ratio=5:1 folds={FOLDS} val_len={val_len}d\n{'=' * 72}")
        oos = {g: [] for g in GATES}
        winners = []
        for k in range(FOLDS):
            t1 = train_end0 + k * val_len
            t2 = min(t1 + val_len, n)
            tr, va = f.iloc[:t1], f.iloc[t1:t2]
            scores = {}
            for g, fn in GATES.items():
                m = fn(tr)
                if g != "baseline" and m.mean() < MIN_ACTIVE_FRAC:
                    continue
                scores[g] = perf(tr["pnl"], m)["sharpe"]
            ranked = sorted(scores, key=scores.get, reverse=True)
            best = ranked[0]
            winners.append(best)
            print(f"\nFold {k + 1}: train {tr['date'].iloc[0]}..{tr['date'].iloc[-1]} "
                  f"({len(tr)}d)  val {va['date'].iloc[0]}..{va['date'].iloc[-1]} ({len(va)}d)")
            print(f"  train ranking by Sharpe: "
                  + " > ".join(f"{g}({scores[g]:.2f})" for g in ranked))
            print(f"  {'gate':18}{'val days':>9}{'val total$':>12}{'Sharpe':>8}{'PF':>7}{'maxDD$':>10}")
            for g, fn in GATES.items():
                p = perf(va["pnl"], fn(va))
                oos[g].append(p)
                tag = " <- picked on train" if g == best else ""
                print(f"  {g:18}{p['days']:9d}{p['total']:12,.0f}{p['sharpe']:8.2f}"
                      f"{p['pf']:7.2f}{p['max_dd']:10,.0f}{tag}")

        print(f"\n--- {name} OOS summary (4 validation blocks stitched) ---")
        print(f"  {'gate':18}{'days':>6}{'total$':>12}{'mean Sharpe':>12}"
              f"{'folds>baseline$':>17}{'picked':>8}")
        base_tot = [p["total"] for p in oos["baseline"]]
        for g in GATES:
            ps = oos[g]
            beats = sum(p["total"] > b for p, b in zip(ps, base_tot))
            print(f"  {g:18}{sum(p['days'] for p in ps):6d}"
                  f"{sum(p['total'] for p in ps):12,.0f}"
                  f"{np.mean([p['sharpe'] for p in ps]):12.2f}"
                  f"{beats:13d}/{FOLDS}{winners.count(g):8d}")

    print("\nNOTE: the candidate family includes the in-sample discovery "
          "(excl_stale_sw); WF-CV here tests selection stability + OOS holdup, "
          "not de-novo discovery. Sharpe is account-level (skipped days = $0).")


if __name__ == "__main__":
    main()
