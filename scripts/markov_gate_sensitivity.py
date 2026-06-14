#!/usr/bin/env python3
"""Sensitivity grid for the excl_stale_sw gate's own hyperparameters.

The gate rule is FIXED (skip day d iff SPX state at prior close is SIDEWAYS
and was also SIDEWAYS `window` sessions before that); only window and the
symmetric bull/bear threshold vary. Evaluated on the frozen Power Hour A
trade log over the SAME stitched OOS region as the WF-CV run (the 4
validation blocks, no per-cell selection anywhere). A real effect shows up
as a plateau of positive cells around the (20, 5%) point, not a spike.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / ".claude/skills/markov-2-hedge-fund-method"))
from markov2 import SIDEWAYS, label_states

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = [10, 15, 20, 25, 30]
THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07]

d = json.load(open("/tmp/gspc.json"))["chart"]["result"][0]
dates = np.array(pd.to_datetime(d["timestamp"], unit="s").date)
closes = np.array(d["indicators"]["quote"][0]["close"], dtype=float)
ok = ~np.isnan(closes)
dates, closes = dates[ok], closes[ok]
date_ix = {dt: i for i, dt in enumerate(dates)}

df = pd.read_csv(ROOT / "results/power_hour_spx/trades_A_triggerplus.csv")
df = df[df["status"] == "closed"]
df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
pnl = df.groupby("session_date")["realized_pnl_usd"].sum().sort_index()

# evaluable trade days must exist for the LARGEST window so every cell sees
# the identical day set; OOS region matches the WF-CV folds (n//9 ratio 5:1)
max_w = max(WINDOWS)
days = [(dt, x, date_ix[dt]) for dt, x in pnl.items()
        if dt in date_ix and date_ix[dt] >= max_w + 1]
n = len(days)
val_len = n // 9
oos = days[5 * val_len: 5 * val_len + 4 * val_len]
oos_pnl = np.array([x for _, x, _ in oos])
print(f"Power Hour A — OOS region {oos[0][0]} .. {oos[-1][0]} "
      f"({len(oos)} trade days), baseline total {oos_pnl.sum():,.0f}$")

def grid(metric):
    hdr = f"{'window':>8} | " + "".join(f"{t:>12.0%}" for t in THRESHOLDS)
    print(hdr + "\n" + "-" * len(hdr))
    for w in WINDOWS:
        cells = []
        for t in THRESHOLDS:
            states = label_states(closes, w, t, -t)
            keep = np.array([not (states[i - 1] == SIDEWAYS
                                  and states[i - 1 - w] == SIDEWAYS)
                             for _, _, i in oos])
            ser = np.where(keep, oos_pnl, 0.0)
            if metric == "total":
                cells.append(f"{ser.sum():>9,.0f}[{keep.sum():3d}]")
            else:
                cells.append(f"{ser.mean() / (ser.std() + 1e-9) * np.sqrt(252):>12.2f}")
        print(f"{w:>8} | " + "".join(cells))

print("\nOOS total $ [days traded] per cell (baseline trades all "
      f"{len(oos)} days for {oos_pnl.sum():,.0f}$):")
grid("total")
print("\nOOS account-level Sharpe (skipped days = $0; baseline "
      f"{oos_pnl.mean() / (oos_pnl.std() + 1e-9) * np.sqrt(252):.2f}):")
grid("sharpe")
