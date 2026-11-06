"""Full delta-range tradeoff for the open IC: sweep em_mult from near-ATM out to
~7-delta, report the REALIZED avg short-call delta (BS, IV from ATM straddle),
win rate, and Sharpe — for BOTH frameworks:
  (A) Pulver as-taught: no stop, hold to settlement
  (B) leg-out: per-side 2x SL + 50% TP + gates

Answers "did you cover the whole delta range, and what's WR at 10-delta?".

Usage: python -m scripts.delta_tradeoff
"""
from __future__ import annotations

import copy
import math
from datetime import date
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.orchestrator import BacktestOrchestrator
from zerodte.strategies.open_ic import OpenIcStrategy
from es_8am.backtest.pricing import bs_call_delta

EM_MULTS = [0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
WING = 30
T_YEARS = (6 + 25 / 60) / (365 * 24)   # 09:35 -> 16:00


def _set(cfg, dotted, v):
    d = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v


def _iv_from_straddle(straddle, spot):
    # straddle ~= 0.8 * spot * iv * sqrt(T)  -> iv
    denom = 0.8 * spot * math.sqrt(T_YEARS)
    return straddle / denom if denom > 0 else None


def _sharpe(pnls):
    n = len(pnls)
    if n < 2:
        return 0.0, sum(pnls), 0.0, n, 0.0
    mean = sum(pnls) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1))
    s = mean / sd * math.sqrt(252) if sd > 0 else 0.0
    wins = sum(1 for p in pnls if p > 0)
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq); dd = max(dd, peak - eq)
    return s, sum(pnls), wins / n, n, dd


def run(cfg, cbbo, days):
    res = BacktestOrchestrator(cfg, OpenIcStrategy, cbbo).run(days)
    by = {}
    for r in res:
        if r.status != "closed":
            continue
        rec = by.setdefault(r.session_date, {"ic": 0.0, "Kc": None, "spot": None, "legs": 0})
        rec["ic"] += r.realized_pnl_usd or 0.0
        rec["legs"] += 1
        if r.underlying_at_entry:
            rec["spot"] = r.underlying_at_entry
        if r.spread_type == "call_credit_spread":
            rec["Kc"] = r.short_strike
    return {d: v for d, v in by.items() if v["legs"] == 2 and v["Kc"] and v["spot"]}


def avg_delta(ic, cbbo):
    ds = []
    for d, rec in ic.items():
        cbbo.load_day(d)
        spot, Kc = rec["spot"], rec["Kc"]
        straddle = cbbo.atm_straddle("09:35", spot)
        if not straddle:
            continue
        iv = _iv_from_straddle(straddle, spot)
        if not iv or iv <= 0:
            continue
        ds.append(bs_call_delta(spot, Kc, T_YEARS, iv))
    return sum(ds) / len(ds) if ds else float("nan")


def main():
    with open("config/config_open_ic_pulver.yaml") as f:
        base = yaml.safe_load(f)
    _set(base, "options.wing_width", WING)
    _set(base, "options.strike_mode", "em_mult")
    _set(base, "options.max_credit_imbalance", 99.0)
    _set(base, "options.min_credit_per_spread", 0.02)

    bt = base["backtest"]
    start, end = date.fromisoformat("2022-05-11"), date.fromisoformat("2026-06-11")
    mem = CbboStore.preload_cache(bt["cbbo_cache_dir"], start=start, end=end)
    cbbo = CbboStore(bt["cbbo_cache_dir"], preloaded=mem)
    loader = BarLoader(bt["data_file"], price_scale=bt["price_scale"], symbol=base["bot"]["symbol"])
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(mem)} cbbo days, {len(days)} bar-days, wing={WING}\n")

    frameworks = {
        "A_as-taught (no stop, hold)": {
            "options.tp_pct_of_credit": 0.0, "options.sl_mult_of_credit": 0.0,
            "options.sl_scope": "total", "strategy.skip_fomc_next_day": False,
            "strategy.skip_nfp_fridays": False, "strategy.skip_weekdays": [],
        },
        "B_leg-out (per-side SL + TP + gates)": {
            "options.tp_pct_of_credit": 0.5, "options.tp_mode": "total",
            "options.sl_mult_of_credit": 2.0, "options.sl_scope": "per_side",
            "strategy.skip_fomc_next_day": True, "strategy.skip_nfp_fridays": True,
            "strategy.skip_weekdays": ["Fri"],
        },
    }

    for fname, fover in frameworks.items():
        print(f"=== {fname} ===")
        print(f"  {'em_mult':>8}{'~delta':>8}{'WR%':>7}{'Sharpe':>8}{'total$':>10}{'n':>5}{'maxDD$':>9}")
        for em in EM_MULTS:
            cfg = copy.deepcopy(base)
            for k, v in fover.items():
                _set(cfg, k, v)
            _set(cfg, "options.em_mult", em)
            ic = run(cfg, cbbo, days)
            if len(ic) < 20:
                print(f"  {em:>8}{'-':>8}{'-':>7}{'-':>8}{'(too few trades: '+str(len(ic))+')':>10}")
                continue
            dlt = avg_delta(ic, cbbo)
            pnls = [ic[d]["ic"] for d in sorted(ic)]
            s, t, w, n, dd = _sharpe(pnls)
            print(f"  {em:>8}{dlt*100:>7.0f}d{w*100:>7.1f}{s:>8.2f}{t:>+10.0f}{n:>5}{dd:>9.0f}")
        print()


if __name__ == "__main__":
    main()
