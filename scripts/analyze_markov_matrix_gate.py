#!/usr/bin/env python3
"""Does the Markov MATRIX add value beyond the bare state label for 0DTE gating?

Three causal gates compared on the same evaluable days (after a 150-trade-day
walk-forward warmup, expanding estimates only):

  STATE-EV  : trade day d iff mean past P&L of days sharing d's prior-close
              state is > 0. Uses only the label — no matrix.
  MATRIX-EV : trade iff sum_j P(s_d = j | s_c) * mu_j > 0, where the row comes
              from the expanding daily transition matrix at the prior close c
              and mu_j is the (shrunk) mean past P&L of days whose own-day
              state was j. This is the matrix doing real forecasting work.
  NON-SW    : fixed rule from the overlay study (skip SIDEWAYS) for reference.

Also prints the in-sample (state 20d ago -> state now) pair table: if arrival
path matters, there is exploitable structure beyond the first-order label.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / ".claude/skills/markov-2-hedge-fund-method"))
from markov2 import SIDEWAYS, BULL, BEAR, STATE_NAMES, label_states, transition_matrix

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 20
WARMUP_TRADE_DAYS = 150
SHRINK_K = 10  # pseudo-observations pulling per-state means toward the overall mean

STRATEGIES = {
    "power_hour_A": ROOT / "results/power_hour_spx/trades_A_triggerplus.csv",
    "meic": ROOT / "results/meic_spx/trades.csv",
    "ric_nat_scale": ROOT / "results/ric_spx/trades_nat_scale.csv",
}


def load_spx():
    d = json.load(open("/tmp/gspc.json"))["chart"]["result"][0]
    dates = np.array(pd.to_datetime(d["timestamp"], unit="s").date)
    closes = np.array(d["indicators"]["quote"][0]["close"], dtype=float)
    ok = ~np.isnan(closes)
    dates, closes = dates[ok], closes[ok]
    return dates, label_states(closes, WINDOW, 0.05, -0.05)


def daily_pnl(csv):
    df = pd.read_csv(csv)
    df = df[df["status"] == "closed"]
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    return df.groupby("session_date")["realized_pnl_usd"].sum().sort_index()


def stats(pnl):
    pnl = np.asarray(pnl, dtype=float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = pnl.cumsum()
    return dict(days=len(pnl), total=pnl.sum(),
                pf=min(wins / max(losses, 1e-9), 99.0),
                sharpe=pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252),
                max_dd=(eq - np.maximum.accumulate(eq)).min() if len(pnl) else 0.0)


def fmt(label, s):
    return (f"{label:24}{s['days']:6d}{s['total']:12,.0f}{s['pf']:7.2f}"
            f"{s['sharpe']:8.2f}{s['max_dd']:12,.0f}")


def main():
    dates, states = load_spx()
    date_ix = {d: i for i, d in enumerate(dates)}

    for name, csv in STRATEGIES.items():
        pnl = daily_pnl(csv)
        rows = []  # (pnl, gate_state, own_state, ev_state, ev_matrix, evaluable)
        sum_gate = np.zeros(3); cnt_gate = np.zeros(3)   # mu by prior-close state
        sum_own = np.zeros(3); cnt_own = np.zeros(3)     # mu by own-day state
        seen = 0
        for d, x in pnl.items():
            i = date_ix.get(d)
            if i is None or i < 1:
                continue
            s_c, s_d = states[i - 1], states[i]
            if s_c < 0 or s_d < 0:
                continue
            evaluable = seen >= WARMUP_TRADE_DAYS
            ev_state = ev_matrix = np.nan
            if evaluable:
                overall = (sum_gate.sum() / max(cnt_gate.sum(), 1))
                mu_gate = (sum_gate + SHRINK_K * overall) / (cnt_gate + SHRINK_K)
                mu_own = (sum_own + SHRINK_K * overall) / (cnt_own + SHRINK_K)
                ev_state = mu_gate[s_c]
                row = transition_matrix(states[:i], 1)[s_c]  # P(s_d | s_c), data <= c
                ev_matrix = float(row @ mu_own)
            rows.append((x, s_c, s_d, ev_state, ev_matrix, evaluable))
            sum_gate[s_c] += x; cnt_gate[s_c] += 1
            sum_own[s_d] += x; cnt_own[s_d] += 1
            seen += 1

        df = pd.DataFrame(rows, columns=["pnl", "gate_state", "own_state",
                                         "ev_state", "ev_matrix", "evaluable"])
        ev = df[df["evaluable"]]
        b_state = ev["ev_state"] > 0
        b_matrix = ev["ev_matrix"] > 0
        disagree = (b_state != b_matrix).mean()
        print(f"\n=== {name} — {len(ev)} evaluable days "
              f"(after {WARMUP_TRADE_DAYS}-day walk-forward warmup) ===")
        print(f"{'gate':24}{'days':>6}{'total$':>12}{'PF':>7}{'Sharpe':>8}{'maxDD$':>12}")
        print(fmt("ALL (baseline)", stats(ev["pnl"])))
        print(fmt("STATE-EV > 0", stats(ev.loc[b_state, "pnl"])))
        print(fmt("MATRIX-EV > 0", stats(ev.loc[b_matrix, "pnl"])))
        print(fmt("NON-SIDEWAYS (fixed)", stats(ev.loc[ev["gate_state"] != SIDEWAYS, "pnl"])))
        print(f"STATE-EV vs MATRIX-EV decisions disagree on {disagree:.1%} of days")

        # arrival-path table (in-sample, descriptive): state 20d before gate -> gate state
        df["prev20"] = df["gate_state"].copy()
        prev = []
        for d in pnl.index[:len(df)]:
            i = date_ix.get(d)
            prev.append(states[i - 1 - WINDOW] if i is not None and i - 1 - WINDOW >= 0 else -1)
        df["prev20"] = prev[:len(df)]
        print("Arrival path (state 20d earlier -> gate state), avg $/day [n]:")
        for cur in (SIDEWAYS, BULL, BEAR):
            cells = []
            for pv in (SIDEWAYS, BULL, BEAR):
                sub = df[(df["gate_state"] == cur) & (df["prev20"] == pv)]
                cells.append(f"from {STATE_NAMES[pv][:4]:4} {sub['pnl'].mean() if len(sub) else float('nan'):>8,.0f} [{len(sub):3d}]")
            print(f"  now {STATE_NAMES[cur]:8}: " + "   ".join(cells))

    print("\nCAVEAT: gates are walk-forward causal but strategy choice/thresholds "
          "are in-sample; WF-CV before any config change.")


if __name__ == "__main__":
    main()
