#!/usr/bin/env python3
"""Markov 2.0 regime overlay vs existing 0DTE backtests.

For each strategy's backtest trade log, condition daily P&L on the prior-close
Markov state (trailing 20d SPX return, BULL/BEAR/SIDEWAYS) and on the
walk-forward stride-sampled signal P(bull)-P(bear). Everything is causal: the
state and signal for day d are computed from closes through d-1 only.

CAVEAT printed with results: this is in-sample conditioning (we look at which
gate would have helped after seeing the P&L). Anything promising needs WF-CV
before it touches a config.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / ".claude/skills/markov-2-hedge-fund-method"))
from markov2 import (BEAR, BULL, SIDEWAYS, STATE_NAMES, label_states,
                     signal_from, transition_matrix, verify_labels)

ROOT = Path(__file__).resolve().parent.parent
WINDOW, BULL_THR, BEAR_THR = 20, 0.05, -0.05

STRATEGIES = {
    "power_hour_A": ROOT / "results/power_hour_spx/trades_A_triggerplus.csv",
    "meic": ROOT / "results/meic_spx/trades.csv",
    "ric_nat_scale": ROOT / "results/ric_spx/trades_nat_scale.csv",
}


def spx_states():
    d = json.load(open("/tmp/gspc.json"))["chart"]["result"][0]
    px = pd.DataFrame({
        "date": pd.to_datetime(d["timestamp"], unit="s").date,
        "open": d["indicators"]["quote"][0]["open"],
        "high": d["indicators"]["quote"][0]["high"],
        "low": d["indicators"]["quote"][0]["low"],
        "close": d["indicators"]["quote"][0]["close"],
        "adj_close": d["indicators"]["quote"][0]["close"],
        "volume": d["indicators"]["quote"][0]["volume"],
    }).dropna().reset_index(drop=True)
    closes = px["close"].to_numpy()
    states = label_states(closes, WINDOW, BULL_THR, BEAR_THR)
    problems = verify_labels(px, states, WINDOW, BULL_THR, BEAR_THR)
    if problems:
        sys.exit("label self-check FAILED: " + "; ".join(problems))
    print(f"[FIX 2] SPX label self-check passed ({STATE_NAMES})")
    # walk-forward stride signal known at the close of day i (usable on day i+1)
    sig = np.full(len(closes), np.nan)
    for i in range(WINDOW * 12, len(closes)):
        if states[i] < 0:
            continue
        m = transition_matrix(states[: i + 1], WINDOW)
        sig[i] = signal_from(m, states[i])
    return pd.DataFrame({"date": px["date"], "state": states, "signal": sig})


def daily_pnl(csv: Path) -> pd.Series:
    df = pd.read_csv(csv)
    df = df[df["status"] == "closed"]
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    return df.groupby("session_date")["realized_pnl_usd"].sum()


def stats(pnl: pd.Series) -> dict:
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    return dict(days=len(pnl), total=pnl.sum(), avg=pnl.mean(),
                pf=wins.sum() / max(-losses.sum(), 1e-9),
                sharpe=pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252),
                max_dd=dd)


def fmt(label, s):
    pf = f"{min(s['pf'], 99.0):7.2f}" if np.isfinite(s["pf"]) else "    inf"
    return (f"{label:28}{s['days']:6d}{s['total']:12,.0f}{s['avg']:9,.0f}"
            f"{pf}{s['sharpe']:8.2f}{s['max_dd']:12,.0f}")


def main():
    regime = spx_states()
    # gate day d with information from the prior close
    regime["gate_state"] = regime["state"].shift(1)
    regime["gate_signal"] = regime["signal"].shift(1)
    reg = regime.set_index("date")

    hdr = f"{'gate':28}{'days':>6}{'total$':>12}{'avg$/d':>9}{'PF':>7}{'Sharpe':>8}{'maxDD$':>12}"
    for name, csv in STRATEGIES.items():
        pnl = daily_pnl(csv)
        j = pd.DataFrame({"pnl": pnl}).join(reg[["gate_state", "gate_signal"]])
        missing = j["gate_state"].isna().sum()
        j = j.dropna(subset=["gate_state"])
        print(f"\n=== {name} ({csv.name}) — {len(j)} trade days"
              + (f", {missing} unmatched dates dropped" if missing else "") + " ===")
        print(hdr)
        print(fmt("ALL DAYS (baseline)", stats(j["pnl"])))
        for st in (SIDEWAYS, BULL, BEAR):
            sub = j[j["gate_state"] == st]
            if len(sub):
                print(fmt(f"  state {STATE_NAMES[st]} only", stats(sub["pnl"])))
        for label, mask in (
            ("FILTER: non-SIDEWAYS only", j["gate_state"] != SIDEWAYS),
            ("FILTER: exclude BEAR", j["gate_state"] != BEAR),
            ("FILTER: exclude BULL", j["gate_state"] != BULL),
            ("FILTER: SIDEWAYS only", j["gate_state"] == SIDEWAYS),
            ("FILTER: signal > 0", j["gate_signal"] > 0),
            ("FILTER: |signal| < 0.10", j["gate_signal"].abs() < 0.10),
        ):
            print(fmt(label, stats(j.loc[mask, "pnl"])))

    print("\n=== Year-by-year: non-SIDEWAYS gate vs baseline (stability check) ===")
    for name, csv in STRATEGIES.items():
        pnl = daily_pnl(csv)
        j = pd.DataFrame({"pnl": pnl}).join(reg[["gate_state"]]).dropna()
        j["year"] = pd.to_datetime(pd.Series(j.index, index=j.index)).dt.year
        rows = []
        for y, g in j.groupby("year"):
            ns = g[g["gate_state"] != SIDEWAYS]
            rows.append(f"  {y}: all {g['pnl'].sum():>9,.0f} on {len(g):3d}d | "
                        f"non-SW {ns['pnl'].sum():>9,.0f} on {len(ns):3d}d "
                        f"({(ns['pnl'].sum() / g['pnl'].sum() if abs(g['pnl'].sum()) > 1 else float('nan')):5.0%} of P&L on "
                        f"{len(ns) / max(len(g), 1):4.0%} of days)")
        print(f"{name}:\n" + "\n".join(rows))

    print("\nCAVEAT: in-sample conditioning with multiple comparisons — any gate "
          "that looks good here must survive WF-CV (the project's 5:1 ratio, "
          "4-fold protocol) before entering a config.")
    print('Backtests flatter. The fixed matrix shows uglier, truer numbers — '
          'those are the only ones worth trading.')


if __name__ == "__main__":
    main()
