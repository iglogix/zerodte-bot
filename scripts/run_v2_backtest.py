"""
Run src_v2 backtest with current config and print report.

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

from src_v2.backtest.data_loader import SpyLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.engine import BacktestEngine
from src_v2.backtest.report import generate_report, print_report


def _load_strategy(config):
    """Instantiate the strategy from config."""
    name = config.get("bot", {}).get("strategy", "orb_em_ic")
    if name == "orb_em_ic":
        from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy
        return OrbEmIronCondorStrategy()
    else:
        raise ValueError(f"Unsupported strategy for backtest: {name}")


def main():
    parser = argparse.ArgumentParser(description="src_v2 Backtest Runner")
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
    mem = None
    if cbbo_dir:
        print(f"Preloading CBBO cache from {cbbo_dir}...")
        mem = CbboStore.preload_cache(cbbo_dir)
        print(f"  {len(mem)} days loaded into memory")

    # Load bars
    print(f"Loading bars from {data_file} (scale={price_scale})...")
    loader = SpyLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} trading days")

    # Create strategy + engine
    strategy = _load_strategy(config)
    engine = BacktestEngine(config, strategy)

    # If we preloaded CBBO, inject into engine's store
    if mem is not None:
        engine._cbbo = CbboStore(cbbo_dir, preloaded=mem)

    # Run
    print(f"Running backtest {start} -> {end}...")
    results = engine.run(days)

    # Report
    report = generate_report(results)
    print_report(report, title=f"src_v2 Backtest: {Path(args.config).stem}")

    return report


if __name__ == "__main__":
    main()
