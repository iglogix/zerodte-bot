"""Joint optimizer: open IC (real engine, per-side leg-out + gates) + broken-wing
DEFENSE overlay with a TUNABLE outer wing. The defense's only drag is the >outer
tail (Pulver's fixed +25 is calibrated to far strikes); here we sweep it.

For each IC config (em_mult x wing) we run the engine once, then overlay the
defense for several outer offsets (and 'off'), pricing entry from CBBO at 09:50
and exit at intrinsic settlement. Ranks all (config x defense) by Sharpe.

Usage: python -m scripts.optimize_ic_plus_defense [--scope per_side|total]
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.orchestrator import BacktestOrchestrator
from zerodte.strategies.open_ic import OpenIcStrategy

ENTRY = "09:50"
SLIP, COMM, MULT = 0.10, 1.0, 100
EM_MULTS = [0.20, 0.30, 0.50, 0.75, 1.00, 1.25]
WINGS = [10, 20, 30]
OUTER_OFFSETS = [15, 20, 25, 30, 40]   # BWB upper/lower long offset from short


def _set(cfg, dotted, v):
    d = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v


def _mid(store, minute, k, is_call):
    q = store.lookup(minute, float(k), is_call)
    return None if q is None else (q[0] + q[1]) / 2


def _bwb_entry_mark(store, low, mid_k, high, is_call):
    a = _mid(store, ENTRY, low, is_call)
    b = _mid(store, ENTRY, mid_k, is_call)
    c = _mid(store, ENTRY, high, is_call)
    if a is None or b is None or c is None:
        return None
    return a - 2 * b + c


def _bwb_intrinsic(settle, low, mid_k, high, is_call):
    f = (lambda k: max(0.0, settle - k)) if is_call else (lambda k: max(0.0, k - settle))
    return f(low) - 2 * f(mid_k) + f(high)


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
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return s, sum(pnls), wins / n, n, dd


def run_ic(cfg, cbbo, days):
    """Return per-day dict: date -> {ic, Kc, Kp, spot}."""
    res = BacktestOrchestrator(cfg, OpenIcStrategy, cbbo).run(days)
    by_day = {}
    for r in res:
        if r.status != "closed":
            continue
        d = r.session_date
        rec = by_day.setdefault(d, {"ic": 0.0, "Kc": None, "Kp": None, "spot": None, "legs": 0})
        rec["ic"] += r.realized_pnl_usd or 0.0
        rec["legs"] += 1
        if r.underlying_at_entry:
            rec["spot"] = r.underlying_at_entry
        if r.spread_type == "call_credit_spread":
            rec["Kc"] = r.short_strike
        elif r.spread_type == "put_credit_spread":
            rec["Kp"] = r.short_strike
    return {d: v for d, v in by_day.items() if v["legs"] == 2 and v["Kc"] and v["Kp"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="per_side", choices=["per_side", "total"])
    ap.add_argument("--start", default="2022-05-11")
    ap.add_argument("--end", default="2026-06-11")
    args = ap.parse_args()

    with open("config/config_open_ic_pulver.yaml") as f:
        base = yaml.safe_load(f)
    # winning management + gates
    _set(base, "options.tp_pct_of_credit", 0.50)
    _set(base, "options.tp_mode", "total")
    _set(base, "options.sl_mult_of_credit", 2.0)
    _set(base, "options.sl_scope", args.scope)
    _set(base, "options.strike_mode", "em_mult")
    _set(base, "options.max_credit_imbalance", 99.0)
    _set(base, "strategy.skip_fomc_next_day", True)
    _set(base, "strategy.skip_nfp_fridays", True)
    _set(base, "strategy.skip_weekdays", ["Fri"])

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    bt = base["backtest"]
    print(f"Preloading CBBO {start}..{end} ...")
    mem = CbboStore.preload_cache(bt["cbbo_cache_dir"], start=start, end=end)
    cbbo = CbboStore(bt["cbbo_cache_dir"], preloaded=mem)
    loader = BarLoader(bt["data_file"], price_scale=bt["price_scale"], symbol=base["bot"]["symbol"])
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(mem)} cbbo days, {len(days)} bar-days, scope={args.scope}\n")

    rows = []
    for em in EM_MULTS:
        for W in WINGS:
            cfg = copy.deepcopy(base)
            _set(cfg, "options.em_mult", em)
            _set(cfg, "options.wing_width", W)
            ic = run_ic(cfg, cbbo, days)
            if len(ic) < 30:
                continue
            tdays = sorted(ic)
            ic_only = [ic[d]["ic"] for d in tdays]
            s0, t0, w0, n0, dd0 = _sharpe(ic_only)
            rows.append((em, W, "off", s0, t0, w0, n0, dd0))
            # precompute settlement + entry inner marks per day, then per outer
            for outer in OUTER_OFFSETS:
                tot = []
                for d in tdays:
                    rec = ic[d]
                    cbbo.load_day(d)
                    Kc, Kp = rec["Kc"], rec["Kp"]
                    settle = cbbo.derive_spot("15:59", rec["spot"] or Kc) or cbbo.settlement_forward()
                    dfn = 0.0
                    if settle is not None:
                        mc = _bwb_entry_mark(cbbo, Kc - 5, Kc + 5, Kc + outer, True)
                        if mc is not None:
                            dfn += (_bwb_intrinsic(settle, Kc - 5, Kc + 5, Kc + outer, True) - mc) * MULT - SLIP * MULT - 3 * COMM
                        mp = _bwb_entry_mark(cbbo, Kp - outer, Kp - 5, Kp + 5, False)
                        if mp is not None:
                            dfn += (_bwb_intrinsic(settle, Kp - outer, Kp - 5, Kp + 5, False) - mp) * MULT - SLIP * MULT - 3 * COMM
                    tot.append(rec["ic"] + dfn)
                s, t, w, n, dd = _sharpe(tot)
                rows.append((em, W, f"out{outer}", s, t, w, n, dd))
            print(f"  done em={em} wing={W} (IC-only Sharpe {s0:.2f})")

    rows.sort(key=lambda r: -r[3])
    print(f"\n  {'em':>5}{'wing':>5}{'defense':>9}{'Sharpe':>8}{'total$':>10}{'WR%':>6}{'n':>5}{'maxDD$':>9}")
    print("  " + "-" * 60)
    for em, W, dfn, s, t, w, n, dd in rows[:30]:
        print(f"  {em:>5}{W:>5}{dfn:>9}{s:>8.2f}{t:>+10.0f}{w*100:>6.1f}{n:>5}{dd:>9.0f}")


if __name__ == "__main__":
    main()
