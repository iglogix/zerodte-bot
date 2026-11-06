"""Run a single Burrito Butterfly config over the EOD SPXW cache. Writes
trades.csv + metrics.json under results/burrito/. Usage:

    .venv/bin/python scripts/backtest_burrito.py --dte 3 --signal hull \
        --lock 0.07 --stop 0.07 --pt 0.10 --slippage 1.0 --start 20200101
"""
import argparse
import csv
import json
import os
from dataclasses import asdict

from zerodte.backtest.iron_duck import DuckStore
from zerodte.backtest.burrito_core import BurritoConfig, run_history, summarize_burrito


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/iron_duck_eod/SPXW")
    ap.add_argument("--dte", type=int, default=3)
    ap.add_argument("--signal", default="hull")
    ap.add_argument("--lock", type=float, default=0.07)
    ap.add_argument("--stop", type=float, default=0.07)
    ap.add_argument("--pt", type=float, default=0.10)
    ap.add_argument("--body", type=float, default=1.0, help="body offset in EM multiples")
    ap.add_argument("--wing", type=float, default=15.0, help="fly wing width (pts)")
    ap.add_argument("--link", default="outer")
    ap.add_argument("--defense", default="")  # comma list
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99999999)
    ap.add_argument("--out", default="results/burrito")
    a = ap.parse_args()

    cfg = BurritoConfig(
        entry_dte=a.dte, direction_signal=a.signal, lock_trigger_pct=a.lock,
        stop_pct=a.stop, pt_pct=a.pt, body_offset=a.body, fly_wing_width=a.wing,
        spread_link=a.link, slippage_k=a.slippage, gates_on=not a.no_gates,
        defense_policy=tuple(x for x in a.defense.split(",") if x),
    )
    store = DuckStore(a.cache)
    trades = run_history(store, cfg, a.start, a.end)
    metrics = summarize_burrito(trades)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics}, f, indent=2, default=str)
    with open(os.path.join(a.out, "trades.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_date", "expiry", "side", "dte", "debit", "extra_cost",
                    "exit_date", "exit_reason", "pnl_usd"])
        for t in trades:
            w.writerow([t.entry_date, t.expiry, t.side, t.dte, round(t.debit, 2),
                        round(t.extra_cost, 2), t.exit_date, t.exit_reason,
                        None if t.pnl_usd is None else round(t.pnl_usd, 2)])
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
