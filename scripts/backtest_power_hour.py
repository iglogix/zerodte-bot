#!/usr/bin/env python3
"""
Backtest: 0-DTE SPX "Power Hour" iron condors (course strategy spec, not published).

Three staggered fixed-credit tranches (14:55/$4, 15:15/$3, 15:25/$2) with dollar
stops + one-time ratchet on the combined short strangle; forced close 15:55 ET.

Usage:
    .venv/bin/python scripts/backtest_power_hour.py                       # Version A
    .venv/bin/python scripts/backtest_power_hour.py --ratchet-mode entry_plus
    .venv/bin/python scripts/backtest_power_hour.py --overlay             # Version C
    .venv/bin/python scripts/backtest_power_hour.py --stop-exec quote
    .venv/bin/python scripts/backtest_power_hour.py --final-exit-mode expire
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zerodte.backtest.cbbo_store import MergedCbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.power_hour import PowerHourEngine, PowerHourResult


def load_vix(path: str = "data/vix_daily.csv") -> dict:
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        date_col = next((cols[c] for c in ("date", "ts_event", "timestamp") if c in cols), None)
        close_col = next((cols[c] for c in ("close", "vix", "adj close") if c in cols), None)
        if not date_col or not close_col:
            return out
        for row in reader:
            try:
                d = date.fromisoformat(str(row[date_col])[:10])
                out[d] = float(row[close_col])
            except (ValueError, KeyError):
                continue
    return out


def write_trades_csv(results: list[PowerHourResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(PowerHourResult)]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"\nTrades written to {out_path}")


def _stats(pnls: list[float]) -> dict:
    if not pnls:
        return {}
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    total = sum(pnls)
    gross_w = sum(winners)
    gross_l = abs(sum(losers))
    peak = eq = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    n = len(pnls)
    mean = total / n
    var = sum((p - mean) ** 2 for p in pnls) / max(n - 1, 1)
    std = math.sqrt(var)
    return {
        "n": n,
        "win_rate": len(winners) / n * 100,
        "total": total,
        "avg": mean,
        "avg_win": gross_w / len(winners) if winners else 0.0,
        "avg_loss": -gross_l / len(losers) if losers else 0.0,
        "max_win": max(pnls),
        "max_loss": min(pnls),
        "pf": gross_w / gross_l if gross_l > 0 else float("inf"),
        "max_dd": dd,
        "std": std,
    }


def print_report(results: list[PowerHourResult], vix: dict, title: str) -> None:
    closed = [r for r in results if r.status == "closed"]
    skipped = [r for r in results if r.status == "skipped"]
    sep = "=" * 78
    print(f"\n{sep}\n  {title}\n{sep}")

    if not closed:
        print("  No closed trades.")
        _print_skips(skipped)
        return

    # Daily P&L (all tranches summed) for day-level stats
    daily = defaultdict(float)
    for r in closed:
        daily[r.session_date] += r.realized_pnl_usd or 0.0
    day_pnls = [daily[d] for d in sorted(daily)]
    ds = _stats(day_pnls)
    sharpe = (ds["avg"] / ds["std"] * math.sqrt(252)) if ds["std"] > 0 else 0.0

    ts = _stats([r.realized_pnl_usd for r in closed])
    print(f"  Trades: {ts['n']}   Days: {ds['n']}   "
          f"Trade win rate: {ts['win_rate']:.1f}%   Day win rate: {ds['win_rate']:.1f}%")
    print(f"  Total P&L:      ${ts['total']:>+12,.0f}")
    print(f"  Avg/trade:      ${ts['avg']:>+12,.0f}    Avg/day: ${ds['avg']:>+9,.0f}")
    print(f"  Avg win/loss:   ${ts['avg_win']:>+12,.0f} / ${ts['avg_loss']:,.0f}")
    print(f"  Max win/loss:   ${ts['max_win']:>+12,.0f} / ${ts['max_loss']:,.0f}")
    print(f"  Profit factor:  {ts['pf']:>13.2f}")
    print(f"  Max DD (daily): ${ds['max_dd']:>12,.0f}")
    print(f"  Sharpe (daily): {sharpe:>13.2f}")

    print("\n  By tranche:")
    hdr = (f"    {'':<4} {'n':>5} {'WR%':>6} {'total':>12} {'avg':>8} "
           f"{'avgW':>8} {'avgL':>8} {'maxL':>9} {'PF':>6} {'stop%':>6}")
    print(hdr)
    for t in ("T1", "T2", "T3"):
        sub = [r for r in closed if r.tranche == t]
        if not sub:
            continue
        s = _stats([r.realized_pnl_usd for r in sub])
        stops = len([r for r in sub if r.exit_reason == "stopped"]) / len(sub) * 100
        pf = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"
        print(f"    {t:<4} {s['n']:>5} {s['win_rate']:>6.1f} {s['total']:>+12,.0f} "
              f"{s['avg']:>+8,.0f} {s['avg_win']:>+8,.0f} {s['avg_loss']:>+8,.0f} "
              f"{s['max_loss']:>+9,.0f} {pf:>6} {stops:>6.1f}")

    print("\n  By year:")
    for yr in sorted({r.session_date.year for r in closed}):
        sub = [r.realized_pnl_usd for r in closed if r.session_date.year == yr]
        s = _stats(sub)
        print(f"    {yr}  n={s['n']:>4}  WR {s['win_rate']:>5.1f}%  "
              f"total ${s['total']:>+11,.0f}  avg ${s['avg']:>+7,.0f}")

    if vix:
        print("\n  By VIX bucket (prior close):")
        buckets = [(0, 13, "<13"), (13, 15, "13-15"), (15, 18, "15-18"),
                   (18, 22, "18-22"), (22, 99, ">22")]
        for lo, hi, label in buckets:
            sub = [r.realized_pnl_usd for r in closed
                   if lo <= vix.get(r.session_date, -1) < hi]
            if not sub:
                continue
            s = _stats(sub)
            print(f"    VIX {label:<6} n={s['n']:>4}  WR {s['win_rate']:>5.1f}%  "
                  f"total ${s['total']:>+11,.0f}  avg ${s['avg']:>+7,.0f}")

    print("\n  By exit reason:")
    for reason, count in Counter(r.exit_reason for r in closed).most_common():
        pnl = sum(r.realized_pnl_usd for r in closed if r.exit_reason == reason)
        print(f"    {reason or '?':<16} {count:>5}  ${pnl:>+12,.0f}")

    if any(r.day_class for r in closed):
        print("\n  By day class (overlay):")
        for dc, count in Counter(r.day_class for r in closed).most_common():
            pnl = sum(r.realized_pnl_usd for r in closed if r.day_class == dc)
            sub = [r.realized_pnl_usd for r in closed if r.day_class == dc]
            s = _stats(sub)
            print(f"    {dc or '?':<8} {count:>5}  WR {s['win_rate']:>5.1f}%  ${pnl:>+12,.0f}")

    n_edge = len([r for r in closed if r.edge_flag])
    n_ratchet = len([r for r in closed if r.ratcheted])
    print(f"\n  Ratcheted: {n_ratchet}/{len(closed)}   chain-edge flagged: {n_edge}")
    _print_skips(skipped)
    print(sep)


def _print_skips(skipped) -> None:
    if not skipped:
        return
    print("\n  Skip reasons (tranche-level):")
    for reason, count in Counter(r.skip_reason for r in skipped).most_common():
        print(f"    {reason:<28} {count:>5}")


def main() -> int:
    ap = argparse.ArgumentParser(description="0DTE SPX Power Hour backtest")
    ap.add_argument("--config", default="config/config_power_hour_spx.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--ratchet-mode", choices=["trigger_plus", "entry_plus"], default=None)
    ap.add_argument("--stop-exec", choices=["trigger", "quote"], default=None)
    ap.add_argument("--final-exit-mode", choices=["close", "expire"], default=None)
    ap.add_argument("--overlay", action="store_true", help="enable Version C regime overlay")
    ap.add_argument("--version-b", action="store_true",
                    help="Version B: 20-delta shorts, 70pt wings, trailing stop, scale-outs")
    ap.add_argument("--t2-entry", default=None, help='e.g. "15:14"')
    ap.add_argument("--out", default="results/power_hour_spx/trades.csv")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    strat = config["strategy"]
    if args.ratchet_mode:
        strat["ratchet_mode"] = args.ratchet_mode
    if args.stop_exec:
        strat["stop_execution"] = args.stop_exec
    if args.final_exit_mode:
        strat["final_exit_mode"] = args.final_exit_mode
    if args.overlay:
        strat["regime_overlay"]["enabled"] = True
    if args.version_b:
        strat["version"] = "B"
        for t in strat["tranches"]:
            t["qty"] = strat.get("version_b", {}).get("qty", 4)
    if args.t2_entry:
        for t in strat["tranches"]:
            if t["name"] == "T2":
                t["entry"] = args.t2_entry

    bt = config["backtest"]
    start = date.fromisoformat(args.start or bt["start_date"])
    end = date.fromisoformat(args.end or bt["end_date"])

    store = MergedCbboStore(bt["cbbo_cache_dir"], bt.get("cbbo_overlay_dir", ""))
    cache_dates = store.available_dates()

    print(f"Loading bars from {bt['data_file']} (x{bt.get('price_scale', 10.0)})...")
    loader = BarLoader(bt["data_file"], price_scale=bt.get("price_scale", 10.0),
                       symbol="SPX")
    days_data = [(day.session_date, day.bars)
                 for day in loader.iter_days(start_date=start, end_date=end)
                 if day.session_date in cache_dates]
    print(f"  {len(days_data)} days with bars + quotes")

    engine = PowerHourEngine(config)
    engine.set_cbbo_store(store)
    results = engine.run(days_data)

    if strat.get("version") == "B":
        variant = "B (20-delta, trailing, scale-outs)"
    elif strat["regime_overlay"].get("enabled"):
        variant = "C (regime overlay)"
    else:
        variant = "A"
    title = (f"Power Hour SPX 0DTE — Version {variant} | "
             f"ratchet={strat['ratchet_mode']} stop={strat['stop_execution']} "
             f"exit={strat['final_exit_mode']} | {start}..{end}")
    print_report(results, load_vix(), title)

    if not args.no_csv:
        write_trades_csv(results, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
