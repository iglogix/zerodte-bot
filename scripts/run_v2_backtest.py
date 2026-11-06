"""
Run zerodte backtest with current config and print report.

Uses the real OrbEngine + PositionMonitor against historical ES bars + CBBO.

Usage:
  python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml
  python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml --start 2025-10-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.orchestrator import BacktestOrchestrator
from zerodte.backtest.report import (
    generate_report, print_report, print_trades, export_trades_csv,
)


def _strategy_cls(config):
    """Return the strategy class from config."""
    name = config.get("bot", {}).get("strategy", "orb_em_ic")
    if name == "orb_em_ic":
        from zerodte.strategies.orb_em_ic import OrbEmIronCondorStrategy
        return OrbEmIronCondorStrategy
    if name == "orb_em_directional":
        from zerodte.strategies.orb_em_directional import OrbEmDirectionalStrategy
        return OrbEmDirectionalStrategy
    if name == "open_ic":
        from zerodte.strategies.open_ic import OpenIcStrategy
        return OpenIcStrategy
    raise ValueError(f"Unsupported strategy for backtest: {name}")


def _results_dir(config_path: str) -> str:
    stem = Path(config_path).stem
    name = stem.removeprefix("config_").strip("_")
    return f"results/{name or 'default'}"


def main():
    parser = argparse.ArgumentParser(description="zerodte Backtest Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bt = config.get("backtest", {})
    data_file = bt.get("data_file", "data/spy_1m.csv")
    price_scale = bt.get("price_scale", 10.0)
    start = date.fromisoformat(args.start) if args.start else date.fromisoformat(bt.get("start_date", "2025-09-22"))
    end = date.fromisoformat(args.end) if args.end else date.fromisoformat(bt.get("end_date", "2026-03-20"))
    symbol = config.get("bot", {}).get("symbol", "SPX")

    # Preload CBBO cache
    cbbo_dir = bt.get("cbbo_cache_dir", "")
    if not cbbo_dir:
        print("ERROR: backtest.cbbo_cache_dir not set in config")
        return
    print(f"Preloading CBBO cache from {cbbo_dir}...")
    mem = CbboStore.preload_cache(cbbo_dir)
    print(f"  {len(mem)} days loaded into memory")
    cbbo = CbboStore(cbbo_dir, preloaded=mem)

    # Load bars
    print(f"Loading bars from {data_file} (scale={price_scale})...")
    loader = BarLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} trading days")

    # Create orchestrator
    # Run — MEIC uses the inline BacktestEngine (scheduled tranche path);
    # other strategies use the real OrbEngine+PositionMonitor orchestrator.
    strategy_name = config.get("bot", {}).get("strategy", "orb_em_ic")
    print(f"Running backtest {start} -> {end}...")
    if strategy_name == "meic":
        from zerodte.backtest.engine import BacktestEngine
        from zerodte.strategies.meic import MeicStrategy
        engine = BacktestEngine(config, MeicStrategy())
        engine._cbbo = cbbo  # use preloaded store instead of the lazy one
        results = engine.run(days)
    else:
        strategy_class = _strategy_cls(config)
        orchestrator = BacktestOrchestrator(config, strategy_class, cbbo)
        results = orchestrator.run(days)

    # Report
    report = generate_report(results)
    title = f"zerodte Backtest: {Path(args.config).stem}"
    print_report(report, title=title)
    print_trades(results)

    # CSV export
    out_dir = _results_dir(args.config)
    csv_path = f"{out_dir}/trades.csv"
    export_trades_csv(results, csv_path)


if __name__ == "__main__":
    main()
