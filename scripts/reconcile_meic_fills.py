#!/usr/bin/env python3
"""Reconcile actual broker fills (fetch_meic_fills.py) against the bot's
assumed MEIC entry credits and the backtest pricing model.

backtest collects (mid_credit - slippage_per_spread) per side = assumed-0.20/IC.
Ground truth = your real executed net credit per opening IC.
"""
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

def _meic_log(root, day):
    """execution_comparison/meic/<date>.jsonl, falling back to the pre-split
    flat path for files written before strategies were namespaced."""
    from pathlib import Path as _P
    ns = _P(root) / "logs/execution_comparison/meic" / f"{day}.jsonl"
    return ns if ns.exists() else _P(root) / "logs/execution_comparison" / f"{day}.jsonl"

ROOT = Path(__file__).resolve().parent.parent
DAY = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
SLIP = 0.10

fills = json.loads((ROOT / f"results/meic_paper/fills_{DAY}.json").read_text())
# group fills by order (same executed_at)
orders = defaultdict(list)
for f in fills:
    orders[f["executed_at"]].append(f)

# bot assumed entries by strike-set
assumed = {}
for line in _meic_log(ROOT, DAY).read_text().splitlines():
    r = json.loads(line)
    if r.get("kind") != "entry":
        continue
    sp, sc = r["legs"]["short_put"], r["legs"]["short_call"]
    key = (sp["short_strike"], sp["long_strike"], sc["short_strike"], sc["long_strike"])
    assumed[key] = r


def strike(sym):
    return float(sym[-8:]) / 1000


print(f"{'time(UTC)':17}{'type':12}{'put credit':>11}{'call credit':>12}"
      f"{'NET fill':>10}{'bot mid':>9}{'BT model':>9}{'fill vs BT':>11}")
for t, legs in sorted(orders.items()):
    opens = [l for l in legs if "OPEN" in l["action"]]
    if len(opens) != 4:
        kind = "MANAGE" if any("CLOSE" in l["action"] for l in legs) else "partial"
        print(f"{t[11:19]:17}{kind:12}{'(intraday adjust — not a clean IC entry)':>53}")
        continue
    # net credit = sells - buys
    pc = cc = 0.0
    ks = {}
    for l in opens:
        k = strike(l["symbol"]); is_call = "C" in l["symbol"][-9]
        sgn = 1 if "SELL" in l["action"] else -1
        (cc_add := 0)
        if is_call:
            cc += sgn * l["price"]; ks["c_s" if sgn > 0 else "c_l"] = k
        else:
            pc += sgn * l["price"]; ks["p_s" if sgn > 0 else "p_l"] = k
    net = pc + cc
    key = (ks.get("p_s"), ks.get("p_l"), ks.get("c_s"), ks.get("c_l"))
    a = assumed.get(key)
    bot_mid = a["total_credit"] if a else float("nan")
    bt = bot_mid - 2 * SLIP if a else float("nan")   # backtest collects mid-0.20/IC
    vs_bt = net - bt if a else float("nan")
    print(f"{t[11:19]:17}{'IC entry':12}{pc:>11.2f}{cc:>12.2f}{net:>10.2f}"
          f"{bot_mid:>9.2f}{bt:>9.2f}{vs_bt:>+11.2f}")

print(f"\nbot mid = bot's assumed credit (at signal time); BT model = bot_mid - 0.20/IC "
      f"(backtest slippage).\n'fill vs BT' > 0 means your real fill BEAT the backtest assumption.")
