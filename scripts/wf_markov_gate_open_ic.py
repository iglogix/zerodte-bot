"""Walk-forward Markov regime FILTER on the optimized open-IC daily P&L.

For each WF fold (same splits as run_wf_cv_open_ic_pulver), LEARN the gate rule
on the train window (which prior-day Markov states have positive mean P&L,
min sample), then APPLY it out-of-sample to the val window. Reports ungated vs
gated OOS Sharpe/PnL per fold and aggregate. Everything causal: the state for
day d uses SPX closes through d-1 only (trailing 20d return, BULL/BEAR/SW).

Also reports a few FIXED a-priori rules across folds for robustness.

Usage: python -m scripts.wf_markov_gate_open_ic results/open_ic_pulver_opt/trades.csv
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / ".claude/skills/markov-2-hedge-fund-method"))
from markov2 import BEAR, BULL, SIDEWAYS, label_states  # noqa: E402

WINDOW, BULL_THR, BEAR_THR = 20, 0.05, -0.05
NAME = {BULL: "BULL", BEAR: "BEAR", SIDEWAYS: "SW", -1: "?"}

# val windows from the ratio=4 folds=5 WF-CV (train = start..val_start)
FOLDS = [
    (date(2024, 2, 29), date(2024, 8, 12)),
    (date(2024, 8, 12), date(2025, 1, 24)),
    (date(2025, 1, 24), date(2025, 7, 9)),
    (date(2025, 7, 9), date(2025, 12, 17)),
    (date(2025, 12, 17), date(2026, 6, 2)),
]
MIN_STATE_N = 8   # train sample floor to trust a state's sign


def spx_states():
    by_day = {}
    with open("data/spy_1m.csv") as f:
        for row in csv.DictReader(f):
            ts = row["ts_event"]
            d = date.fromisoformat(ts[:10])
            hhmm = ts[11:16]
            if hhmm <= "20:05":
                prev = by_day.get(d)
                if prev is None or hhmm >= prev[0]:
                    by_day[d] = (hhmm, float(row["close"]) * 10.0)
    days = sorted(by_day)
    closes = np.array([by_day[d][1] for d in days])
    states = label_states(closes, WINDOW, BULL_THR, BEAR_THR)
    idx = {d: i for i, d in enumerate(days)}
    return days, states, idx


def daily_pnl(path):
    out, legs = defaultdict(float), defaultdict(int)
    with open(path) as f:
        for row in csv.DictReader(f):
            p = row.get("realized_pnl_usd", "")
            if p in ("", None):
                continue
            d = date.fromisoformat(row["session_date"])
            out[d] += float(p)
            legs[d] += 1
    return {d: v for d, v in out.items() if legs[d] == 2}


def sharpe(pnls):
    n = len(pnls)
    if n < 2:
        return 0.0, sum(pnls), (sum(1 for p in pnls if p > 0) / n if n else 0), n
    mean = sum(pnls) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1))
    s = (mean / sd * math.sqrt(252)) if sd > 0 else 0.0
    return s, sum(pnls), sum(1 for p in pnls if p > 0) / n, n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/open_ic_pulver_opt/trades.csv"
    days, states, idx = spx_states()
    dp = daily_pnl(path)
    tdays = sorted(dp)

    def prior_state(d):
        i = idx.get(d)
        if i is None or i == 0:
            return None
        return states[i - 1]

    print(f"=== WF Markov FILTER on {path} ===")
    print(f"  causal prior-day state (trailing {WINDOW}d, bull>+{BULL_THR:.0%}/bear<{BEAR_THR:.0%})\n")

    agg_ung, agg_gate = [], []
    print(f"  {'fold val window':<26} {'rule (trade states)':<18} "
          f"{'ungated OOS':<24} {'gated OOS':<24}")
    for (vs, ve) in FOLDS:
        train = [d for d in tdays if d < vs]
        val = [d for d in tdays if vs <= d < ve]
        # learn: states with positive train mean P&L and enough sample
        by_state = defaultdict(list)
        for d in train:
            st = prior_state(d)
            if st is not None and st >= 0:
                by_state[st].append(dp[d])
        keep = {st for st, v in by_state.items()
                if len(v) >= MIN_STATE_N and (sum(v) / len(v)) > 0}
        if not keep:  # fallback: trade all if nothing qualifies
            keep = {BULL, BEAR, SIDEWAYS}
        rule = "+".join(NAME[s] for s in sorted(keep))
        ung = [dp[d] for d in val]
        gate = [dp[d] for d in val if prior_state(d) in keep]
        su, tu, wu, nu = sharpe(ung)
        sg, tg, wg, ng = sharpe(gate)
        agg_ung += ung
        agg_gate += gate
        print(f"  {str(vs)+'..'+str(ve):<26} {rule:<18} "
              f"S={su:>5.2f} ${tu:>+7.0f} n={nu:<4} | "
              f"S={sg:>5.2f} ${tg:>+7.0f} n={ng}")

    su, tu, wu, nu = sharpe(agg_ung)
    sg, tg, wg, ng = sharpe(agg_gate)
    print(f"\n  AGGREGATE OOS (learned gate):")
    print(f"    ungated : Sharpe {su:>5.2f}  PnL ${tu:>+8.0f}  WR {wu*100:4.1f}%  n={nu}")
    print(f"    gated   : Sharpe {sg:>5.2f}  PnL ${tg:>+8.0f}  WR {wg*100:4.1f}%  n={ng}")

    # fixed a-priori rules across all OOS days (robustness check)
    print(f"\n  FIXED rules across ALL OOS days (no learning):")
    alloos = [d for d in tdays if d >= FOLDS[0][0]]
    fixed = {
        "all (ungated)": lambda d: True,
        "skip SW": lambda d: prior_state(d) != SIDEWAYS,
        "skip BEAR": lambda d: prior_state(d) != BEAR,
        "BULL only": lambda d: prior_state(d) == BULL,
        "skip BEAR+SW": lambda d: prior_state(d) not in (BEAR, SIDEWAYS),
    }
    for name, keep in fixed.items():
        sel = [dp[d] for d in alloos if keep(d)]
        s, t, w, n = sharpe(sel)
        print(f"    {name:<16} Sharpe {s:>5.2f}  PnL ${t:>+8.0f}  WR {w*100:4.1f}%  n={n}")


if __name__ == "__main__":
    main()
