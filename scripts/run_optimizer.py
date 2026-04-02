"""
Walk-forward CV + OOS optimizer runner.

Supports both ES (Databento) and SPY/SPX (MarketData.app) data sources.
The data source is auto-detected from the config's backtest.data_source setting.

ES data split (5 years: 2021-03-01 -> 2026-02-26):
  CV window  : 2021-03-01 -> 2024-12-31  (~4 years, 4 folds of ~1 year)
  OOS window : 2025-01-01 -> 2026-02-26  (~14 months, never seen during optimisation)

SPX data split (SPX 0DTE from 2022-05-11):
  CV window  : 2022-05-11 -> 2024-12-31  (~2.6 years, 4 folds)
  OOS window : 2025-01-01 -> 2026-02-28  (~14 months, never seen during optimisation)

Usage:
  python -m scripts.run_optimizer
  python -m scripts.run_optimizer --config config/config_8am_spx_options.yaml
  python -m scripts.run_optimizer --n-workers 4
  python -m scripts.run_optimizer --folds 4 --top-n 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import yaml

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_8am.backtest.data_loader import DatabentoLoader
from src_8am.backtest.optimizer import DEFAULT_GRID, EM_GRID, QUANTITY_GRID, GridSearchOptimizer
from src_8am.backtest.spy_loader import SpyLoader


# ---------------------------------------------------------------------------
# Date boundaries (ES / Databento)
# ---------------------------------------------------------------------------

ES_CV_START  = date(2021, 3,  1)
ES_CV_END    = date(2024, 12, 31)
ES_OOS_START = date(2025, 1,  1)
ES_OOS_END   = date(2026, 2, 26)

# ---------------------------------------------------------------------------
# Date boundaries (SPX / SPY via MarketData.app)
# ---------------------------------------------------------------------------

SPX_CV_START  = date(2022, 5, 11)  # SPX 0DTE daily expirations started
SPX_CV_END    = date(2024, 12, 31)
SPX_OOS_START = date(2025, 1,  1)
SPX_OOS_END   = date(2026, 2, 28)


def _strategy_name_from_config(config_path: str) -> str:
    """Derive strategy subdirectory name from config filename.

    config/config_8am_es_options.yaml   -> 8am_es_options
    config/config_8am_spx_options.yaml  -> 8am_spx_options
    config/config.yaml                  -> default
    """
    stem = Path(config_path).stem  # e.g. "config_8am_es_options"
    name = stem.removeprefix("config_").strip("_")
    return name or "default"


def main():
    parser = argparse.ArgumentParser(description="Walk-forward CV + OOS optimizer")
    parser.add_argument("--config",    default="config/config_8am_es_options.yaml")
    parser.add_argument("--folds",     type=int, default=4,  help="Number of CV folds")
    parser.add_argument("--top-n",     type=int, default=5,  help="Top-N combos per fold for voting")
    parser.add_argument("--min-trades",type=int, default=20, help="Min trades to count a combo")
    parser.add_argument("--n-workers", type=int, default=0,  help="Worker processes (0=all cores)")
    parser.add_argument("--account-size", type=float, default=100_000, help="Account size for ARR calc")
    parser.add_argument("--min-arr",   type=float, default=0.70, help="Min annual return rate (0.70=70%%)")
    parser.add_argument("--max-contracts", type=int, default=50, help="Max contracts cap (OI/liquidity limit)")
    parser.add_argument("--train-val-ratio", type=float, default=5.0, help="Train:val ratio (5=5:1, 0=equal folds)")
    parser.add_argument("--sweep-qty", action="store_true", help="Include quantity in grid sweep (values: 1,5,10,25,50)")
    parser.add_argument("--strategy",  default=None, choices=["delta", "em"],
                        help="Strategy type (default: from config, or 'delta')")
    parser.add_argument("--out-dir",   default=None,         help="Directory for CSV output (default: results/<strategy>)")
    parser.add_argument("--log-level", default="INFO",       help="DEBUG, INFO, WARNING")
    args = parser.parse_args()

    # Auto-derive output directory from config name if not explicitly set
    if args.out_dir is None:
        strategy = _strategy_name_from_config(args.config)
        args.out_dir = f"results/{strategy}"

    # --- Logging to file + stderr ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "optimizer_debug.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    logging.getLogger(__name__).info("Logging to %s", log_path)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = config["backtest"]["data_file"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Auto-detect data source and set date boundaries
    # -----------------------------------------------------------------------
    data_source = config.get("backtest", {}).get("data_source", "databento")
    is_spy = data_source == "spy"

    if is_spy:
        CV_START, CV_END = SPX_CV_START, SPX_CV_END
        OOS_START, OOS_END = SPX_OOS_START, SPX_OOS_END
        price_scale = config.get("backtest", {}).get("price_scale", 10.0)
        equity_symbol = config.get("bot", {}).get("symbol", "SPY")
        loader = SpyLoader(data_path, price_scale=price_scale, symbol=equity_symbol)
        product_label = equity_symbol + " (equity)"
    else:
        CV_START, CV_END = ES_CV_START, ES_CV_END
        OOS_START, OOS_END = ES_OOS_START, ES_OOS_END
        loader = DatabentoLoader(data_path)
        product_label = "ES"

    # Optional second data file (continuation) — merge into same loader
    data_path_2 = config.get("backtest", {}).get("data_file_2", "")
    loader_2 = None
    if data_path_2 and Path(data_path_2).exists():
        loader_2 = DatabentoLoader(data_path_2)

    # Allow config to override date boundaries
    bt = config.get("backtest", {})
    if bt.get("start_date"):
        cfg_start = date.fromisoformat(bt["start_date"])
        CV_START = max(CV_START, cfg_start)
    if bt.get("end_date"):
        cfg_end = date.fromisoformat(bt["end_date"])
        OOS_END = min(OOS_END, cfg_end)
    # Allow config to override CV_END and OOS boundaries
    if bt.get("cv_end_date"):
        CV_END = date.fromisoformat(bt["cv_end_date"])
    if bt.get("oos_start_date"):
        OOS_START = date.fromisoformat(bt["oos_start_date"])
    if bt.get("oos_end_date"):
        OOS_END = date.fromisoformat(bt["oos_end_date"])

    print("=" * 70)
    print(f"  ORB Options Optimizer — Walk-Forward CV + OOS ({product_label})")
    print("=" * 70)
    print(f"\nData : {data_path}")
    if loader_2:
        print(f"Data2: {data_path_2}")
    print(f"CV   : {CV_START} -> {CV_END}  ({args.folds} folds)")
    print(f"OOS  : {OOS_START} -> {OOS_END}  (held out)")
    print(f"Acct : ${args.account_size:,.0f}  min ARR={args.min_arr*100:.0f}%  max_contracts={args.max_contracts}")
    print()

    print("Loading CV bars...", flush=True)
    t0 = time.time()
    cv_days = list(loader.iter_days(start_date=CV_START, end_date=CV_END))
    if loader_2:
        cv_days_2 = list(loader_2.iter_days(start_date=CV_START, end_date=CV_END))
        # Merge: add days from file 2 that aren't already in file 1
        existing_dates = {(d.session_date, d.symbol) for d in cv_days}
        cv_days.extend(d for d in cv_days_2 if (d.session_date, d.symbol) not in existing_dates)
    print(f"  {len(cv_days)} day-symbol chunks in {time.time()-t0:.1f}s")

    print("Loading OOS bars...", flush=True)
    t0 = time.time()
    oos_days = list(loader.iter_days(start_date=OOS_START, end_date=OOS_END))
    if loader_2:
        oos_days_2 = list(loader_2.iter_days(start_date=OOS_START, end_date=OOS_END))
        existing_dates = {(d.session_date, d.symbol) for d in oos_days}
        oos_days.extend(d for d in oos_days_2 if (d.session_date, d.symbol) not in existing_dates)
    print(f"  {len(oos_days)} day-symbol chunks in {time.time()-t0:.1f}s")

    # For CBBO strategy: pre-filter to only days with CBBO cache data
    cbbo_cache = config.get("options", {}).get("cbbo_cache_dir", "")
    if cbbo_cache and Path(cbbo_cache).exists():
        cbbo_dates = {date.fromisoformat(f.stem) for f in Path(cbbo_cache).glob("*.pkl")}
        cv_before = len(cv_days)
        cv_days = [d for d in cv_days if d.session_date in cbbo_dates]
        oos_before = len(oos_days)
        oos_days = [d for d in oos_days if d.session_date in cbbo_dates]
        print(f"  CBBO filter: CV {cv_before} -> {len(cv_days)}, OOS {oos_before} -> {len(oos_days)}")
    print()

    # -----------------------------------------------------------------------
    # 2. Walk-forward CV
    # -----------------------------------------------------------------------
    risk_cfg = config.get("risk", {})
    max_bp_pct = risk_cfg.get("max_buying_power_pct", 50.0) / 100.0

    es_mult = config.get("options", {}).get("contract_multiplier", 50)
    opt = GridSearchOptimizer(
        base_config=config,
        zip_path=data_path,
        account_size=args.account_size,
        min_arr=args.min_arr,
        es_multiplier=es_mult,
        max_bp_pct=max_bp_pct,
        max_contracts_cap=args.max_contracts,
    )

    # --- Strategy selection ---
    strategy_type = args.strategy or config.get("options", {}).get("strategy_type", "delta")
    config.setdefault("options", {})["strategy_type"] = strategy_type

    if strategy_type == "em":
        param_grid = dict(EM_GRID)
        print(f"Strategy: Expected Move (EM)")
    elif strategy_type == "cbbo":
        from src_8am.backtest.strategy_cbbo import CbboDeltaStrategy
        param_grid = CbboDeltaStrategy().grid_params()
        print(f"Strategy: CBBO Delta (real SPX quotes)")
    elif strategy_type == "cbbo_em":
        from src_8am.backtest.strategy_cbbo_em import CbboEMStrategy
        param_grid = CbboEMStrategy().grid_params()
        print(f"Strategy: CBBO EM Iron Condor (real SPX quotes)")
    else:
        param_grid = dict(DEFAULT_GRID)
        print(f"Strategy: Delta-targeted")

    # Adjust grid for small-priced equities (IWM ~$200 vs ES ~$5500)
    symbol = config.get("bot", {}).get("symbol", "ES")
    if symbol in ("IWM",):
        param_grid["spread_width_pts"] = [1, 2, 3, 4, 5]
        param_grid["min_orb_range_pts"] = [0.10, 0.20, 0.30]
        param_grid["midline_tolerance"] = [0.10, 0.25, 0.50]

    # For ANN pricing, skip BS/bachelier sweep — ANN replaces the pricing model
    if config.get("options", {}).get("iv_source") == "price_ann":
        param_grid["pricing_model"] = ["bs"]  # BS used only for delta targeting, not pricing

    if args.sweep_qty:
        param_grid["quantity"] = QUANTITY_GRID
        print(f"Quantity sweep enabled: {QUANTITY_GRID}")
        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        print(f"Total grid: {total_combos} combos\n")

    best_params, fold_results = opt.run_walk_forward_cv(
        all_days   = cv_days,
        param_grid = param_grid,
        n_folds    = args.folds,
        top_n      = args.top_n,
        min_trades = args.min_trades,
        n_workers  = args.n_workers,
        train_val_ratio = args.train_val_ratio,
    )

    # Save per-fold top-val rows to CSV
    for fr in fold_results:
        fold_rows = fr.get("top_val_params", [])
        if fold_rows:
            path = out_dir / f"fold_{fr['fold']}_val_top{args.top_n}.csv"
            opt.save_csv(fold_rows, str(path))

    # -----------------------------------------------------------------------
    # 3. OOS final test on winning params
    # -----------------------------------------------------------------------
    oos_report: dict | None = None
    if best_params:
        print(f"\nRunning OOS test with best params...")
        oos_report = opt.eval_on_oos(best_params, oos_days)
        oos_path = out_dir / "oos_result.csv"
        opt.save_csv([{**best_params, **oos_report}], str(oos_path))

    # -----------------------------------------------------------------------
    # 4. Summary
    # -----------------------------------------------------------------------
    opt.print_cv_summary(fold_results, best_params, oos_report)

    # Also run a full-CV-window grid search with the best params for reference table
    if best_params:
        print("\nFull CV-window backtest with winning params (reference):")
        full_report = opt.eval_on_oos(best_params, cv_days)
        print(f"  trades={full_report.get('trades')}  "
              f"sharpe={full_report.get('sharpe', 0):.2f}  "
              f"WR={full_report.get('win_rate', 0)*100:.1f}%  "
              f"total_pnl=${full_report.get('total_pnl_usd', 0):,.0f}")

    # -----------------------------------------------------------------------
    # 5. Save optimized params to config file
    # -----------------------------------------------------------------------
    if best_params:
        opt_path = str(out_dir / "optimized_params.yaml")
        opt.save_optimized_config(
            best_params,
            oos_report=oos_report,
            fold_results=fold_results,
            output_path=opt_path,
        )


if __name__ == "__main__":
    main()
