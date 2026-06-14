"""Markov-2.0 regime overlay on an open-IC trades.csv (causal, prior-close).

Derives SPX daily closes from data/spy_1m.csv (x10), labels each day's trailing
state (BULL/BEAR/SIDEWAYS) and a walk-forward stride signal P(bull)-P(bear),
then conditions the strategy's per-day P&L on the prior-day state/signal.
Reports ungated vs gated (skip SIDEWAYS / skip stale-SIDEWAYS / signal floor)
Sharpe + PnL. IN-SAMPLE conditioning — confirm with WF-CV before deploying.

Usage: python -m scripts.markov_overlay_open_ic results/<dir>/trades.csv
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
from markov2 import (BEAR, BULL, SIDEWAYS, STATE_NAMES, label_states,  # noqa: E402
                     signal_from, transition_matrix)

WINDOW, BULL_THR, BEAR_THR = 20, 0.05, -0.05


def spx_daily_closes() -> tuple[list[date], np.ndarray]:
    """Regular-session daily close from spy_1m.csv x10 (UTC ts; last <=20:00Z bar)."""
    by_day: dict[date, tuple[str, float]] = {}
    with open("data/spy_1m.csv") as f:
        for row in csv.DictReader(f):
            ts = row["ts_event"]
            d = date.fromisoformat(ts[:10])
            hhmm = ts[11:16]
            # regular session close ~20:00Z (16:00 ET, std) / 19:xx in DST; take last bar <= 20:05Z
            if hhmm <= "20:05":
                prev = by_day.get(d)
                if prev is None or hhmm >= prev[0]:
                    by_day[d] = (hhmm, float(row["close"]) * 10.0)
    days = sorted(by_day)
    closes = np.array([by_day[d][1] for d in days])
    return days, closes


def daily_pnl(path: str) -> dict[date, float]:
    out = defaultdict(float)
    legs = defaultdict(int)
    with open(path) as f:
        for row in csv.DictReader(f):
            p = row.get("realized_pnl_usd", "")
            if p in ("", None):
                continue
            d = date.fromisoformat(row["session_date"])
            out[d] += float(p)
            legs[d] += 1
    return {d: v for d, v in out.items() if legs[d] == 2}


def sharpe(pnls: list[float]) -> tuple[float, float, float, int]:
    n = len(pnls)
    if n < 2:
        return 0.0, 0.0, 0.0, n
    mean = sum(pnls) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1))
    s = (mean / sd * math.sqrt(252)) if sd > 0 else 0.0
    wr = sum(1 for p in pnls if p > 0) / n
    return s, sum(pnls), wr, n


def main():
    path = sys.argv[1]
    days, closes = spx_daily_closes()
    states = label_states(closes, WINDOW, BULL_THR, BEAR_THR)
    # walk-forward stride signal known at close of day i (usable day i+1)
    sig = np.full(len(closes), np.nan)
    for i in range(WINDOW * 12, len(closes)):
        if states[i] < 0:
            continue
        m = transition_matrix(states[: i + 1], WINDOW)
        sig[i] = signal_from(m, states[i])
    state_by_day = {days[i]: states[i] for i in range(len(days))}
    sig_by_day = {days[i]: sig[i] for i in range(len(days))}
    # prior-trading-day index for causal lookup
    idx = {d: i for i, d in enumerate(days)}

    dp = daily_pnl(path)
    trade_days = sorted(dp)

    def prior_state_sig(d):
        i = idx.get(d)
        if i is None or i == 0:
            return None, None
        return states[i - 1], sig[i - 1]

    base = [dp[d] for d in trade_days]
    bs, bt, bw, bn = sharpe(base)
    print(f"=== Markov overlay: {path} ===")
    print(f"  SPX closes {days[0]}..{days[-1]} ({len(days)}d), states {STATE_NAMES}")
    print(f"\n  UNGATED            : Sharpe {bs:>6.2f}  PnL ${bt:>+9.0f}  WR {bw*100:4.1f}%  n={bn}")

    # condition by prior-day state
    print("\n  P&L by PRIOR-day state:")
    for st_name, st in [("BULL", BULL), ("BEAR", BEAR), ("SIDEWAYS", SIDEWAYS)]:
        sel = [dp[d] for d in trade_days if prior_state_sig(d)[0] == st]
        if sel:
            s, t, w, n = sharpe(sel)
            print(f"    prior={st_name:<8} n={n:>4} Sharpe {s:>6.2f}  PnL ${t:>+9.0f}  WR {w*100:4.1f}%")

    # gates
    gates = {
        "skip prior=SIDEWAYS": lambda d: prior_state_sig(d)[0] != SIDEWAYS,
        "skip prior=BEAR": lambda d: prior_state_sig(d)[0] != BEAR,
        "skip prior in {BEAR,SW}": lambda d: prior_state_sig(d)[0] not in (BEAR, SIDEWAYS),
        "signal>=0 (P(bull)>=P(bear))": lambda d: (prior_state_sig(d)[1] is not None
                                                   and not math.isnan(prior_state_sig(d)[1])
                                                   and prior_state_sig(d)[1] >= 0),
    }
    print("\n  GATED variants:")
    for name, keep in gates.items():
        sel = [dp[d] for d in trade_days if keep(d)]
        s, t, w, n = sharpe(sel)
        print(f"    {name:<32} n={n:>4} Sharpe {s:>6.2f}  PnL ${t:>+9.0f}  WR {w*100:4.1f}%")


if __name__ == "__main__":
    main()
