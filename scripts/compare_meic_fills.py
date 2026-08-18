#!/usr/bin/env python3
"""Compare today's live MEIC entry pricing/fills vs the backtest's assumptions.

Backtest entry model (engine.py): collect the MID credit, then charge
slippage_per_spread (0.10) per credit spread = 0.20/IC over the two sides
(~0.05/leg). The manual run logs each short leg's real bid/ask every poll, so
we can measure the actual marketable cost and test whether 0.10/spread is
realistic or optimistic — the sensitivity the WF study flagged as load-bearing
(MEIC's edge lives within ~0.15/spread of entry quality).
"""
import json
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

def _meic_log(root, day):
    """execution_comparison/meic/<date>.jsonl, falling back to the pre-split
    flat path for files written before strategies were namespaced."""
    from pathlib import Path as _P
    ns = _P(root) / "logs/execution_comparison/meic" / f"{day}.jsonl"
    return ns if ns.exists() else _P(root) / "logs/execution_comparison" / f"{day}.jsonl"

date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-15"
log = _meic_log(".", date)
SLIP = 0.10  # backtest slippage_per_spread

entries, quotes = {}, defaultdict(list)
for line in log.read_text().splitlines():
    r = json.loads(line)
    if r.get("kind") == "entry":
        entries[r["trade_id"]] = r
    elif r.get("type") == "quote":
        quotes[(r["trade_id"], r["role"])].append(r)


def ts(s):
    return datetime.fromisoformat(s)


def nearest(qs, t):
    return min(qs, key=lambda q: abs((ts(q["ts"]) - t).total_seconds())) if qs else None


print(f"{'trade':13}{'leg':11}{'assumed':>9}{'bid':>7}{'ask':>7}{'mid':>7}{'spread':>8}{'mid-drift':>10}")
tot_half = n = 0
per_ic = {}
for tid, e in sorted(entries.items()):
    t = ts(e["ts"])
    ic_half = 0.0
    for role, leg in (("short_put", e["legs"]["short_put"]),
                      ("short_call", e["legs"]["short_call"])):
        q = nearest(quotes.get((tid, role), []), t)
        if not q:
            continue
        bid, ask = q["real_bid"], q["real_ask"]
        spread = ask - bid
        mid = (bid + ask) / 2
        half = spread / 2
        ic_half += half
        tot_half += half
        n += 1
        drift = mid - leg["short_mid"]
        print(f"{tid:13}{role:11}{leg['short_mid']:>9.2f}{bid:>7.2f}{ask:>7.2f}"
              f"{mid:>7.2f}{spread:>8.2f}{drift:>+10.3f}")
    per_ic[tid] = ic_half

avg_half = tot_half / n if n else 0
avg_ic = sum(per_ic.values()) / len(per_ic) if per_ic else 0
print(f"\nmean short-leg half-spread:        {avg_half:.3f}  "
      f"(backtest assumes ~{SLIP/2:.3f}/leg)")
print(f"mean per-IC cost, 2 SHORT legs:    {avg_ic:.3f}")
print(f"backtest slippage, 2 sides/IC:     {2*SLIP:.3f}  (all 4 legs)")
print(f"projected per-IC marketable cost:  ~{2*avg_ic:.3f}  "
      f"(short legs ×2, assuming longs ~similar)")
verdict = ("REALISTIC — backtest 0.10/spread brackets the measured cost"
           if 2 * avg_ic <= 2 * SLIP * 1.3 else
           "OPTIMISTIC — real spreads exceed the backtest assumption")
print(f"\nVERDICT: {verdict}")
print(f"edge-sensitivity: backtest breaks even ~0.15/spread; measured "
      f"half-spread/leg = {avg_half:.3f}")
