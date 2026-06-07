# src_v2/backtest/dc_time_machine/run_backtest.py
"""CLI for the DC Time Machine backtest.

Subcommands:
  backfill       Fetch multi-expiry full-day cbbo-1m into a cache dir (needs Databento).
  study-spreads  Print the empirical half-spread table from a cache dir.
  simulate       Run a config over a cache dir → trades.csv + summary.
  scan           Reconstructed Flux scan over a cache dir.
  wf-cv          Walk-forward CV over a param grid.
  crosscheck     Simulate Feb–Apr 2026 and print aggregate stats vs his published numbers.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import TradeResult, run_session
from src_v2.backtest.dc_time_machine.strategy import DcParams

_CSV_FIELDS = ["session", "front", "back", "k_put", "k_call", "outcome", "entry_debit",
               "credit", "bank", "transform_minute", "settlement", "pnl"]


def write_trades_csv(trades: List[TradeResult], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for t in trades:
            w.writerow({k: getattr(t, k) for k in _CSV_FIELDS})


def summarize(trades: List[TradeResult]) -> Dict[str, float]:
    real = [t for t in trades if t.outcome != "no_entry"]
    wins = [t.pnl for t in real if t.pnl > 0]
    losses = [t.pnl for t in real if t.pnl < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n": len(real),
        "win_rate": (len(wins) / len(real)) if real else 0.0,
        "total_pnl": sum(t.pnl for t in real),
        "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf"),
        "n_transformed": sum(1 for t in real if t.outcome == "transformed"),
    }


def _sessions_in_cache(cache_dir: Path) -> List[date]:
    return sorted(date.fromisoformat(p.stem) for p in cache_dir.glob("*.pkl"))


def _simulate_range(cache_dir: Path, params: DcParams,
                    lo: date | None = None, hi: date | None = None) -> List[TradeResult]:
    store = MultiExpiryStore(cache_dir=cache_dir)
    out: List[TradeResult] = []
    for s in _sessions_in_cache(cache_dir):
        if (lo and s < lo) or (hi and s > hi):
            continue
        res = run_session(store, params, s)
        if res is not None:
            out.append(res)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dc-backtest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate")
    sim.add_argument("--cache", required=True)
    sim.add_argument("--out", default="results/dc_time_machine/trades.csv")
    sim.add_argument("--fill-k", type=float, default=1.0)

    cc = sub.add_parser("crosscheck")
    cc.add_argument("--cache", required=True)
    cc.add_argument("--fill-k", type=float, default=1.0)

    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        params = DcParams(fill_k=args.fill_k)
        trades = _simulate_range(Path(args.cache), params)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_trades_csv(trades, out)
        print(summarize(trades))
        return 0

    if args.cmd == "crosscheck":
        params = DcParams(fill_k=args.fill_k)
        trades = _simulate_range(Path(args.cache), params, date(2026, 2, 1), date(2026, 4, 30))
        s = summarize(trades)
        print("Feb–Apr 2026 cross-check:", s)
        print("His published: ~106 trades, ~63% win, ~25% max-profit-hit")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
