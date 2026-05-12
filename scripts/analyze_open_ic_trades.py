"""
Dissect open IC PnL by day-of-week, month, VIX regime, calendar proximity,
intraday range. Run both the WF-CV winner and the canonical TT config so
their patterns can be compared side-by-side.
"""

from __future__ import annotations

import copy
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import SpyLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.engine import _FOMC_DATES, _CPI_DATES, _is_nfp_friday
from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.backtest.vix_loader import load_vix_series, lookup_prior_close
from src_v2.strategies.open_ic import OpenIcStrategy


CONFIGS = {
    "WF-CV v5 winner (with new gates)": {
        "options.target_short_delta": 0.25,
        "options.wing_width": 25,
        "options.tp_pct_of_credit": 0.25,
        "options.tp_mode": "total",
        "options.sl_mult_of_credit": 2.0,
        "options.hold_put_breach": True,
        "options.hold_call_breach": True,
        "entry.opening_range_filter.enabled": True,
        "entry.opening_range_filter.max_open_range_em_ratio": 0.5,
        "strategy.skip_fomc_next_day": True,
        "strategy.skip_cpi_next_day": False,
    },
}

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _set_dotted(cfg, k, v):
    parts = k.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v


def _proximity(d: date) -> str:
    """Classify d's relation to FOMC/CPI/NFP days."""
    if _is_nfp_friday(d):
        return "NFP day"
    if d in _FOMC_DATES:
        return "FOMC day"
    if d in _CPI_DATES:
        return "CPI day"
    for off in (-1, 1):
        adj = d + timedelta(days=off)
        if _is_nfp_friday(adj):
            return f"NFP {'eve' if off == 1 else 'next-day'}"
        if adj in _FOMC_DATES:
            return f"FOMC {'eve' if off == 1 else 'next-day'}"
        if adj in _CPI_DATES:
            return f"CPI {'eve' if off == 1 else 'next-day'}"
    return "normal"


def _vix_bucket(vix: float | None) -> str:
    if vix is None:
        return "?"
    if vix < 13:
        return "low (<13)"
    if vix < 17:
        return "mid (13-17)"
    if vix < 22:
        return "hi (17-22)"
    return "xhi (22+)"


def _intraday_range_bucket(day_bars) -> str:
    if not day_bars:
        return "?"
    hi = max(b["high"] for b in day_bars)
    lo = min(b["low"] for b in day_bars)
    rng = hi - lo
    if rng < 25:
        return "tight (<25 pts)"
    if rng < 50:
        return "med (25-50)"
    if rng < 80:
        return "wide (50-80)"
    return "vol (80+)"


def _run_one(label: str, overrides: dict, days, cbbo, base_cfg, vix_series):
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        _set_dotted(cfg, k, v)
    orch = BacktestOrchestrator(cfg, OpenIcStrategy, cbbo)
    results = orch.run(days)

    # Map session_date → day_bars
    day_bars_by_date = {d.session_date: d.bars for d in days}

    by_day = defaultdict(lambda: {"pnl": 0.0, "legs": []})
    for r in results:
        if r.status == "closed":
            by_day[r.session_date]["pnl"] += r.realized_pnl_usd or 0.0
            by_day[r.session_date]["legs"].append(r.exit_reason)

    complete = []
    for d, v in by_day.items():
        if len(v["legs"]) != 2:
            continue
        vix = lookup_prior_close(vix_series, d - timedelta(days=1))
        complete.append({
            "date": d,
            "pnl": v["pnl"],
            "weekday": WEEKDAY_NAMES[d.weekday()],
            "month": d.month,
            "year": d.year,
            "proximity": _proximity(d),
            "vix_bucket": _vix_bucket(vix),
            "range_bucket": _intraday_range_bucket(day_bars_by_date.get(d, [])),
        })

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}")
    total_pnl = sum(r["pnl"] for r in complete)
    wins = [r for r in complete if r["pnl"] > 0]
    losses = [r for r in complete if r["pnl"] < 0]
    print(f"  Days={len(complete)}  Total=${total_pnl:+,.0f}  "
          f"WR={100*len(wins)/len(complete):.1f}%  "
          f"avg_win=${sum(r['pnl'] for r in wins)/max(1,len(wins)):+.0f}  "
          f"avg_loss=${sum(r['pnl'] for r in losses)/max(1,len(losses)):+.0f}")
    print()

    for dim, sort_key in [
        ("weekday", lambda k: WEEKDAY_NAMES.index(k)),
        ("vix_bucket", lambda k: ["low (<13)", "mid (13-17)", "hi (17-22)", "xhi (22+)", "?"].index(k)),
        ("range_bucket", lambda k: ["tight (<25 pts)", "med (25-50)", "wide (50-80)", "vol (80+)", "?"].index(k)),
        ("proximity", None),
        ("year", None),
        ("month", None),
    ]:
        groups = defaultdict(list)
        for r in complete:
            groups[r[dim]].append(r["pnl"])
        keys = sorted(groups.keys(), key=sort_key) if sort_key else sorted(groups.keys(), key=lambda k: -sum(groups[k]))
        print(f"  By {dim}:")
        print(f"    {'bucket':<22} {'n':>4} {'WR%':>6} {'total':>10} {'mean':>8} {'worst':>9} {'best':>8}")
        for k in keys:
            pnls = groups[k]
            n_win = sum(1 for p in pnls if p > 0)
            print(f"    {str(k):<22} {len(pnls):>4d} "
                  f"{100*n_win/len(pnls):>5.1f}% "
                  f"{sum(pnls):>+10.0f} "
                  f"{sum(pnls)/len(pnls):>+8.0f} "
                  f"{min(pnls):>+9.0f} "
                  f"{max(pnls):>+8.0f}")
        print()


def main():
    with open("config/config_spx_0dte_open_ic.yaml") as f:
        base_cfg = yaml.safe_load(f)
    bt = base_cfg.get("backtest", {})
    start = date(2022, 4, 18)
    end = date(2026, 2, 28)

    print(f"Preloading CBBO {start} → {end}…")
    mem = CbboStore.preload_cache(bt.get("cbbo_cache_dir", "data/cbbo_cache"),
                                  start=start, end=end)
    cbbo = CbboStore(bt.get("cbbo_cache_dir"), preloaded=mem)
    loader = SpyLoader(bt.get("data_file"), price_scale=bt.get("price_scale", 10.0),
                       symbol=base_cfg["bot"]["symbol"])
    days = list(loader.iter_days(start_date=start, end_date=end))
    vix_series = load_vix_series("data/vix9d_daily.csv")
    print(f"  {len(mem)} cbbo, {len(days)} bar-days, {len(vix_series)} vix rows")

    for label, overrides in CONFIGS.items():
        _run_one(label, overrides, days, cbbo, base_cfg, vix_series)


if __name__ == "__main__":
    main()
