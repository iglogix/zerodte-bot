#!/usr/bin/env python3
"""Feasibility probe: do WIDER MEIC shorts/wings even produce tradeable spreads?

Far-OTM 0DTE options go thin/bidless, and the credit-to-width gate
(min_credit_width_ratio, default 0.08) rejects spreads whose credit is too small
for their width. Before sweeping wide-short + stop combos, this checks: at
short_em_mult {1.0,1.5,2.0} x wing_em_frac {0.5,1.0}, how many days actually
trade, and at what credit / width / ratio? Single 11:00 slot, hold-to-expiry,
uniform static sizing. Run across vol regimes (low-vol 2023 is the stress case).

Usage:
  python scripts/probe_meic_strikes.py --start 2023-05-01 --end 2023-07-31
  python scripts/probe_meic_strikes.py --start 2025-09-01 --end 2025-11-30
"""
from __future__ import annotations
import argparse
import copy
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.engine import BacktestEngine
from src_v2.strategies.meic import MeicStrategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--shorts", default="1.0,1.5,2.0")
    ap.add_argument("--wings", default="0.5,1.0")
    a = ap.parse_args()

    cfg0 = yaml.safe_load(Path("config/config_meic_spx.yaml").read_text())
    st = cfg0["strategy"]
    st["exit_mode"] = "hold_to_expiry"
    st["sizing_mode"] = "static"
    st["tranche_sizing"] = {}
    st["default_tranche_qty"] = 1
    cfg0["entry"]["schedule"] = ["11:00"]

    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    loader = BarLoader(cfg0["backtest"]["data_file"],
                       price_scale=cfg0["backtest"]["price_scale"], symbol="SPX")
    days = list(loader.iter_days(start_date=s, end_date=e))
    cbbo_dir = cfg0["backtest"]["cbbo_cache_dir"]
    mem = CbboStore.preload_cache(cbbo_dir, start=days[0].session_date,
                                  end=days[-1].session_date)
    print(f"{len(days)} days {s}..{e}  cache={cbbo_dir}  gate min_credit_width_ratio="
          f"{st.get('min_credit_width_ratio', 0.08)}\n")
    print(f"{'short':>5} {'wing':>5} | {'trades':>6} {'days_ok':>7} {'no_trade':>8} "
          f"{'avgCr$':>7} {'avgWidth':>8} {'cr/width':>8}")
    for sm in [float(x) for x in a.shorts.split(",")]:
        for wf in [float(x) for x in a.wings.split(",")]:
            cfg = copy.deepcopy(cfg0)
            cfg["strategy"]["short_em_mult"] = sm
            cfg["strategy"]["wing_em_frac"] = wf
            eng = BacktestEngine(cfg, MeicStrategy())
            eng._cbbo = CbboStore(cbbo_dir, preloaded=mem)
            res = eng.run(days)
            closed = [r for r in res if r.status == "closed"]
            notrade = sum(1 for r in res
                          if getattr(r, "skip_reason", None) == "no_meic_trades")
            days_ok = len({r.session_date for r in closed})
            if closed:
                cr = np.mean([r.credit_received for r in closed])
                wd = np.mean([r.spread_width for r in closed])
                crw = np.mean([r.credit_received / r.spread_width
                               for r in closed if r.spread_width > 0])
            else:
                cr = wd = crw = 0.0
            print(f"{sm:>5.1f} {wf:>5.1f} | {len(closed):>6} {days_ok:>7} {notrade:>8} "
                  f"{cr:>7.2f} {wd:>8.1f} {crw:>8.3f}")


if __name__ == "__main__":
    main()
