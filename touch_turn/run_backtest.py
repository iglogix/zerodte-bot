"""
Touch and Turn Scalper — Backtest Runner

Usage:
  python -m touch_turn.run_backtest
  python -m touch_turn.run_backtest --config touch_turn/config.yaml
  python -m touch_turn.run_backtest --start 2023-01-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_8am.backtest.data_loader import DatabentoLoader
from touch_turn.backtest.engine import BacktestReport, TouchTurnBacktestEngine
from touch_turn.backtest.equity_loader import EquityLoader


def _print_summary(report: BacktestReport, elapsed: float) -> None:
    trades = report.trades
    n = len(trades)
    print(f"\n{'=' * 60}")
    print("  Touch & Turn Scalper — Backtest Results")
    print(f"{'=' * 60}")
    print(f"  Days processed : {report.total_days}")
    print(f"  Days skipped   : {report.skipped_days}")
    print(f"  Trades taken   : {n}")
    print(f"  Elapsed        : {elapsed:.1f}s")

    if n == 0:
        print("  No trades to report.")
        return

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total_pnl = sum(t.pnl_usd for t in trades)
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    win_rate = len(wins) / n

    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    tp_trades = [t for t in trades if t.exit_reason == "tp"]
    sl_trades = [t for t in trades if t.exit_reason == "sl"]
    eod_trades = [t for t in trades if t.exit_reason == "eod"]

    # Drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl_usd
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    # Sharpe (daily P&L approximation)
    import math
    pnls = [t.pnl_usd for t in trades]
    avg_pnl = total_pnl / n
    if n > 1:
        std_pnl = math.sqrt(sum((p - avg_pnl) ** 2 for p in pnls) / (n - 1))
        sharpe = (avg_pnl / std_pnl) * math.sqrt(252) if std_pnl > 0 else 0
    else:
        sharpe = 0

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print("\n  --- Performance ---")
    print(f"  Total P&L      : ${total_pnl:>10,.2f}")
    print(f"  Avg trade       : ${avg_pnl:>10,.2f}")
    print(f"  Win rate        : {win_rate * 100:>8.1f}%")
    print(f"  Profit factor   : {pf:>8.2f}")
    print(f"  Sharpe (ann.)   : {sharpe:>8.2f}")
    print(f"  Max drawdown    : ${max_dd:>10,.2f}")
    print(f"  Gross profit    : ${gross_profit:>10,.2f}")
    print(f"  Gross loss      : ${gross_loss:>10,.2f}")

    print("\n  --- Breakdown ---")
    print(f"  Longs / Shorts  : {len(longs)} / {len(shorts)}")
    print(f"  TP / SL / EOD   : {len(tp_trades)} / {len(sl_trades)} / {len(eod_trades)}")

    # Best / worst trades
    best = max(trades, key=lambda t: t.pnl_usd)
    worst = min(trades, key=lambda t: t.pnl_usd)
    print(f"\n  Best trade      : ${best.pnl_usd:>10,.2f}  ({best.session_date} {best.direction})")
    print(f"  Worst trade     : ${worst.pnl_usd:>10,.2f}  ({worst.session_date} {worst.direction})")
    print(f"{'=' * 60}\n")


def _save_trades_csv(report: BacktestReport, path: str) -> None:
    trades = report.trades
    if not trades:
        return
    fieldnames = list(asdict(trades[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow(asdict(t))
    print(f"  Trades saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Touch & Turn Scalper backtest")
    parser.add_argument("--config", default="touch_turn/config.yaml")
    parser.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Auto-derive output dir from config name: config_nvda.yaml -> results/touch_turn_nvda
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stem = Path(args.config).stem.removeprefix("config_").strip("_") or "touch_turn"
        out_dir = Path(f"results/{stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "backtest.log", mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = config["backtest"]["data_file"]
    data_source = config.get("backtest", {}).get("data_source", "databento")
    start = date.fromisoformat(args.start or config["backtest"]["start_date"])
    end = date.fromisoformat(args.end or config["backtest"]["end_date"])

    symbol = config.get("bot", {}).get("symbol", "ES")
    print(f"Loading {symbol} 1-min bars from {data_path} ...")
    if data_source == "equity":
        loader = EquityLoader(data_path)
    else:
        loader = DatabentoLoader(data_path)

    engine = TouchTurnBacktestEngine(config)

    print(f"Running backtest: {start} → {end}")
    t0 = time.time()
    report = engine.run(loader.iter_days(start_date=start, end_date=end))
    elapsed = time.time() - t0

    _print_summary(report, elapsed)
    _save_trades_csv(report, str(out_dir / "trades.csv"))


if __name__ == "__main__":
    main()
