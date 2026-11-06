#!/usr/bin/env python3
"""
Backtest: 0-DTE SPX "Reverse Iron Condor" (docs/0DTE-Reverse-Iron-Condor-Strategy.md).

Long asymmetric condor near the open on gap-up (>= 0.25%) days: buy the
50-delta call (+20pt short wing) and the 35-delta put (-35pt short wing) for a
net debit; resting profit-target limit at +25-30% of the debit, NO stop, let
the rest expire (cash settle) or close near the bell.

Usage:
    .venv/bin/python scripts/backtest_reverse_ic.py                 # defaults
    .venv/bin/python scripts/backtest_reverse_ic.py --tp 0.30
    .venv/bin/python scripts/backtest_reverse_ic.py --entry-time 09:45
    .venv/bin/python scripts/backtest_reverse_ic.py --scale-out
    .venv/bin/python scripts/backtest_reverse_ic.py --final-exit-mode close
    .venv/bin/python scripts/backtest_reverse_ic.py --gap-min 0.25 --gap-min-mon 0.10
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zerodte.backtest.cbbo_store import MergedCbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.reverse_ic import RicEngine, RicResult


def load_vix(path: str = "data/vix_cboe.csv") -> dict:
    """Parse VIX daily closes; handles both plain Date/Close CSVs and the
    Yahoo multi-row-header export (Price,Close,... / Ticker,... / Date,,,)."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        rows = list(csv.reader(f))
    if not rows:
        return out
    header = [c.lower() for c in rows[0]]
    if header[0] == "price":          # Yahoo format: col0=date, col1=close
        date_i, close_i, data = 0, header.index("close"), rows[3:]
    else:
        date_i = next((i for i, c in enumerate(header)
                       if c in ("date", "ts_event", "timestamp")), 0)
        close_i = next((i for i, c in enumerate(header)
                        if c in ("close", "vix", "adj close")), 1)
        data = rows[1:]
    for row in data:
        try:
            out[date.fromisoformat(str(row[date_i])[:10])] = float(row[close_i])
        except (ValueError, IndexError):
            continue
    return out


def write_trades_csv(results: list[RicResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(RicResult)]
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
        "std": math.sqrt(var),
    }


def _bucket_line(label: str, pnls: list[float]) -> str:
    s = _stats(pnls)
    if not s:
        return f"    {label:<10}   (no trades)"
    return (f"    {label:<10} n={s['n']:>4}  WR {s['win_rate']:>5.1f}%  "
            f"total ${s['total']:>+11,.0f}  avg ${s['avg']:>+7,.0f}")


def print_report(results: list[RicResult], vix: dict, title: str) -> None:
    closed = [r for r in results if r.status == "closed"]
    skipped = [r for r in results if r.status == "skipped"]
    sep = "=" * 78
    print(f"\n{sep}\n  {title}\n{sep}")

    if not closed:
        print("  No closed trades.")
        _print_skips(skipped)
        return

    pnls = [r.realized_pnl_usd for r in closed]
    s = _stats(pnls)
    sharpe = (s["avg"] / s["std"] * math.sqrt(252)) if s["std"] > 0 else 0.0
    # one trade per day -> trade stats are day stats, but annualize on the
    # qualified-day frequency (gap-up days only)
    n_days = len({r.session_date for r in results})

    print(f"  Trades: {s['n']} on {n_days} sessions "
          f"({s['n'] / max(n_days, 1) * 100:.0f}% of days qualify)")
    print(f"  Win rate:       {s['win_rate']:>13.1f}%")
    print(f"  Total P&L:      ${s['total']:>+12,.0f}")
    print(f"  Avg/trade:      ${s['avg']:>+12,.0f}")
    print(f"  Avg win/loss:   ${s['avg_win']:>+12,.0f} / ${s['avg_loss']:,.0f}"
          f"   (W/L ratio {abs(s['avg_win'] / s['avg_loss']) if s['avg_loss'] else float('inf'):.2f})")
    print(f"  Max win/loss:   ${s['max_win']:>+12,.0f} / ${s['max_loss']:,.0f}")
    print(f"  Profit factor:  {s['pf']:>13.2f}")
    print(f"  Max drawdown:   ${s['max_dd']:>12,.0f}")
    print(f"  Sharpe (per-trade, ann. 252): {sharpe:>5.2f}")
    debits = [r.debit for r in closed if r.debit]
    print(f"  Avg debit:      {sum(debits) / len(debits):>13.2f} pts "
          f"(max loss avg ${sum(r.max_loss_usd for r in closed) / len(closed):,.0f})")

    print("\n  By exit reason:")
    for reason, count in Counter(r.exit_reason for r in closed).most_common():
        sub = [r.realized_pnl_usd for r in closed if r.exit_reason == reason]
        st = _stats(sub)
        print(f"    {reason or '?':<16} n={st['n']:>4}  WR {st['win_rate']:>5.1f}%  "
              f"total ${st['total']:>+11,.0f}  avg ${st['avg']:>+7,.0f}")

    # §6 / §14.6 — the edge test: how do losers distribute vs the full debit?
    losers = [r for r in closed if (r.realized_pnl_usd or 0) <= 0]
    if losers:
        print("\n  Loss distribution (loss as % of max loss):")
        buckets = [(0.0, 0.25, "0-25%"), (0.25, 0.50, "25-50%"),
                   (0.50, 0.75, "50-75%"), (0.75, 0.95, "75-95%"),
                   (0.95, 9.99, "full")]
        for lo, hi, label in buckets:
            n = len([r for r in losers
                     if lo <= -r.realized_pnl_usd / max(r.max_loss_usd, 1) < hi])
            print(f"    {label:<8} {n:>4}  ({n / len(losers) * 100:.0f}% of losers)")

    print("\n  By year:")
    for yr in sorted({r.session_date.year for r in closed}):
        print(_bucket_line(str(yr),
                           [r.realized_pnl_usd for r in closed
                            if r.session_date.year == yr]))

    if vix:
        print("\n  By VIX bucket (prior close):")
        for lo, hi, label in [(0, 13, "<13"), (13, 15, "13-15"), (15, 18, "15-18"),
                              (18, 22, "18-22"), (22, 99, ">22")]:
            sub = [r.realized_pnl_usd for r in closed
                   if lo <= vix.get(r.session_date, -1) < hi]
            if sub:
                print(_bucket_line(f"VIX {label}", sub))

    print("\n  By weekday:")
    for wd in ("mon", "tue", "wed", "thu", "fri"):
        sub = [r.realized_pnl_usd for r in closed if r.weekday == wd]
        if sub:
            print(_bucket_line(wd, sub))

    print("\n  By gap size:")
    for lo, hi, label in [(0.25, 0.40, "0.25-0.4"), (0.40, 0.60, "0.4-0.6"),
                          (0.60, 1.00, "0.6-1.0"), (1.00, 99, ">1.0")]:
        sub = [r.realized_pnl_usd for r in closed
               if r.gap_pct is not None and lo <= r.gap_pct < hi]
        if sub:
            print(_bucket_line(f"{label}%", sub))

    n_edge = len([r for r in closed if r.edge_flag])
    n_spy = len([r for r in closed if r.gap_source == "spy"])
    print(f"\n  Chain-edge flagged: {n_edge}   SPY-fallback gaps: {n_spy}")
    _print_skips(skipped)
    print(sep)


def _print_skips(skipped) -> None:
    if not skipped:
        return
    print("\n  Skip reasons:")
    for reason, count in Counter(r.skip_reason for r in skipped).most_common():
        print(f"    {reason:<28} {count:>5}")


def main() -> int:
    ap = argparse.ArgumentParser(description="0DTE SPX Reverse Iron Condor backtest")
    ap.add_argument("--config", default="config/config_ric_spx.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--entry-time", default=None, help='e.g. "09:45"')
    ap.add_argument("--tp", type=float, default=None, help="profit target pct, e.g. 0.30")
    ap.add_argument("--tp-fill", choices=["limit", "mid"], default=None)
    ap.add_argument("--gap-min", type=float, default=None)
    ap.add_argument("--gap-min-mon", type=float, default=None,
                    help="Monday gap override, e.g. 0.10")
    ap.add_argument("--setup-min", type=float, default=None)
    ap.add_argument("--no-setup-filter", action="store_true")
    ap.add_argument("--scale-out", action="store_true")
    ap.add_argument("--final-exit-mode", choices=["expire", "close"], default=None)
    ap.add_argument("--qty", type=int, default=None)
    ap.add_argument("--pricing", choices=["mid", "natural"], default=None,
                    help="natural = pay ask / hit bid on every traded leg")
    ap.add_argument("--vix-floor", type=float, default=None,
                    help="skip days whose prior VIX close is below this")
    ap.add_argument("--slip", type=float, default=None,
                    help="slippage per spread (e.g. 0 with --pricing natural)")
    ap.add_argument("--out", default="results/ric_spx/trades.csv")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    strat = config["strategy"]
    if args.entry_time:
        strat["entry_time"] = args.entry_time
    if args.tp is not None:
        strat["profit_target_pct"] = args.tp
    if args.tp_fill:
        strat["tp_fill"] = args.tp_fill
    if args.gap_min is not None:
        strat["gap_min_pct"] = args.gap_min
    if args.gap_min_mon is not None:
        strat.setdefault("gap_min_pct_by_weekday", {})["mon"] = args.gap_min_mon
    if args.setup_min is not None:
        strat["setup_min_upside_pct"] = args.setup_min
    if args.no_setup_filter:
        strat["setup_min_upside_pct"] = -99.0
    if args.scale_out:
        strat.setdefault("scale_out", {})["enabled"] = True
        strat["qty"] = max(int(strat.get("qty", 1)), 4)
    if args.final_exit_mode:
        strat["final_exit_mode"] = args.final_exit_mode
    if args.qty is not None:
        strat["qty"] = args.qty
    if args.pricing:
        strat["pricing"] = args.pricing
    if args.slip is not None:
        config["options"]["slippage_per_spread"] = args.slip
    if args.vix_floor is not None:
        strat["vix_floor"] = args.vix_floor

    bt = config["backtest"]
    start = date.fromisoformat(args.start or bt["start_date"])
    end = date.fromisoformat(args.end or bt["end_date"])

    store = MergedCbboStore(bt["cbbo_cache_dir"], bt.get("cbbo_overlay_dir", ""))

    print(f"Loading bars from {bt['data_file']} (x{bt.get('price_scale', 10.0)})...")
    loader = BarLoader(bt["data_file"], price_scale=bt.get("price_scale", 10.0),
                       symbol="SPX")
    # Pass ALL sessions (uncached days skip inside the engine) so the gap
    # reference is always the true previous trading session.
    days_data = [(day.session_date, day.bars)
                 for day in loader.iter_days(start_date=start, end_date=end)]
    print(f"  {len(days_data)} sessions loaded")

    engine = RicEngine(config)
    engine.set_cbbo_store(store)
    vix = load_vix()
    engine.set_vix(vix)
    results = engine.run(days_data)

    so = strat.get("scale_out", {}).get("enabled")
    title = (f"Reverse IC SPX 0DTE — entry {strat['entry_time']} "
             f"gap>={strat['gap_min_pct']}% tp={strat['profit_target_pct']:.0%} "
             f"{'scale-out ' if so else ''}exit={strat['final_exit_mode']} "
             f"pricing={strat.get('pricing', 'mid')} | {start}..{end}")
    print_report(results, vix, title)

    if not args.no_csv:
        write_trades_csv(results, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
