"""
Walk-Forward Cross-Validation optimizer for the Iron Fly strategy.

Reuses the same WF-CV logic as the ORB optimizer (ratio-based folds,
rank_score formula, majority vote winner selection) but runs IronFlyEngine
instead of OrbOptionsBacktestEngine.

Usage:
    from src_8am.backtest.iron_fly_optimizer import IronFlyOptimizer
    opt = IronFlyOptimizer(config, cbbo_mem, days_data, iv_by_date)
    best, folds = opt.run_walk_forward_cv()
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import time as time_mod
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pathlib import Path

from src_8am.backtest.iron_fly_engine import IronFlyEngine, IronFlyResult
from src_8am.backtest.strategy_iron_fly import IronFlyStrategy
from src_8am.backtest.cbbo_store import CbboStore

logger = logging.getLogger(__name__)


def _run_combo_inproc(params, base_config, days, iv, cbbo_store,
                      min_trades, account_size, es_multiplier, num_years):
    """Run one param combo in-process. Returns result row or None."""
    strategy = IronFlyStrategy()
    cfg = strategy.apply_params(base_config, params)
    strategy.configure(cfg)

    engine = IronFlyEngine(strategy, cfg)
    if cbbo_store is not None:
        engine.set_cbbo_store(cbbo_store)

    results = engine.run(days, iv)
    report = _generate_report(results)

    if report["trades_placed"] < min_trades:
        return None

    return _score_report(params, report, min_trades, account_size,
                         es_multiplier, num_years)


# ---------------------------------------------------------------------------
# Report generation (compatible with optimizer scorer)
# ---------------------------------------------------------------------------

def _generate_report(results: List[IronFlyResult]) -> Dict[str, Any]:
    """Generate report dict from IronFlyResult list."""
    closed = [r for r in results if r.status == "closed"]
    skipped = [r for r in results if r.status == "skipped"]

    if not closed:
        return {"trades_placed": 0, "win_rate": 0, "total_pnl_usd": 0,
                "avg_pnl_per_trade_usd": 0, "profit_factor": 0,
                "max_drawdown_usd": 0, "sharpe_annualised": 0,
                "avg_credit_received": 0, "avg_short_delta": 0}

    pnls = [r.realized_pnl_usd for r in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    # Sharpe (annualised)
    daily_pnl: Dict[str, float] = defaultdict(float)
    for r in closed:
        daily_pnl[str(r.session_date)] += r.realized_pnl_usd
    daily_returns = list(daily_pnl.values())
    sharpe = 0.0
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        var = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0

    credits = [r.credit_received for r in closed if r.credit_received]
    avg_credit = sum(credits) / len(credits) if credits else 0

    return {
        "trades_placed": len(closed),
        "win_rate": round(win_rate, 4),
        "total_pnl_usd": round(total_pnl, 2),
        "avg_pnl_per_trade_usd": round(total_pnl / len(closed), 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "max_drawdown_usd": round(max_dd, 2),
        "sharpe_annualised": round(sharpe, 2),
        "avg_credit_received": round(avg_credit, 3),
        "avg_short_delta": 0.5,  # ATM iron fly
    }


# ---------------------------------------------------------------------------
# Scoring (same formula as ORB optimizer)
# ---------------------------------------------------------------------------

def _score_report(params, report, min_trades, account_size, es_multiplier,
                  num_years):
    """Score a report. Returns result row dict or None."""
    pf = report.get("profit_factor", 0)
    if pf == "inf":
        pf = 999.0

    sharpe = report.get("sharpe_annualised", 0)
    wr = report.get("win_rate", 0)
    dd = report.get("max_drawdown_usd", 0)
    trades = report["trades_placed"]

    # rank_score = 0.50×sharpe + 0.30×PF + 0.20×WR×10
    capped_sharpe = min(sharpe, 20.0)
    score = 0.50 * capped_sharpe + 0.30 * min(float(pf), 10.0) + 0.20 * wr * 10

    if trades < 30:
        score *= trades / 30.0

    # DD penalty
    if account_size > 0:
        dd_pct = dd / account_size
        if dd_pct > 0.10:
            score *= 0.9
        if dd_pct > 0.20:
            score *= 0.7
        if dd_pct > 0.35:
            score *= 0.5
        if dd_pct > 0.50:
            score *= 0.3

    row = {
        **params,
        "trades": trades,
        "win_rate": wr,
        "total_pnl_usd": report["total_pnl_usd"],
        "avg_pnl_usd": report["avg_pnl_per_trade_usd"],
        "profit_factor": pf,
        "max_dd_usd": dd,
        "sharpe": sharpe,
        "avg_credit": report["avg_credit_received"],
        "rank_score": round(score, 4),
    }
    return row


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------

class IronFlyOptimizer:
    """Walk-forward CV optimizer for Iron Fly strategy."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        cbbo_mem: dict,
        days_data: List[Tuple[date, List[Dict]]],
        iv_by_date: Dict[date, float],
        account_size: float = 100_000,
        es_multiplier: float = 100,
    ):
        self._base_config = base_config
        self._cbbo_mem = cbbo_mem
        self._days_data = sorted(days_data, key=lambda x: x[0])
        self._iv_by_date = iv_by_date
        self._account_size = account_size
        self._es_multiplier = es_multiplier

    def run_walk_forward_cv(
        self,
        param_grid: Optional[Dict[str, List]] = None,
        n_folds: int = 4,
        top_n: int = 5,
        min_trades: int = 15,
        n_workers: int = 0,
        train_val_ratio: float = 5.0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Walk-forward CV with ratio-based expanding window.

        Returns (best_params, fold_results).
        """
        if param_grid is None:
            strat = IronFlyStrategy()
            param_grid = strat.grid_params()

        keys = list(param_grid.keys())
        combos = [dict(zip(keys, c)) for c in itertools.product(*param_grid.values())]
        print(f"Grid: {len(combos)} combos, {len(self._days_data)} days, "
              f"{n_folds} folds (ratio {train_val_ratio}:1)")

        # Sort days by date
        all_dates = [d[0] for d in self._days_data]
        n_total = len(all_dates)

        # Ratio-based fold splitting (expanding train window)
        date_step = n_total // (train_val_ratio + n_folds)
        date_step = max(date_step, 1)

        votes: Dict[str, List[float]] = {}
        fold_results = []

        for k in range(n_folds):
            val_start_idx = int((train_val_ratio + k) * date_step)
            val_end_idx = min(int((train_val_ratio + k + 1) * date_step), n_total)
            train_end_idx = val_start_idx

            if val_start_idx >= n_total or val_end_idx > n_total:
                break

            train_days = self._days_data[:train_end_idx]
            val_days = self._days_data[val_start_idx:val_end_idx]

            if not train_days or not val_days:
                continue

            train_dates = [d[0] for d in train_days]
            val_dates = [d[0] for d in val_days]

            print(f"\n--- Fold {k+1}/{n_folds}: train {train_dates[0]}..{train_dates[-1]} "
                  f"({len(train_days)}d) | val {val_dates[0]}..{val_dates[-1]} ({len(val_days)}d)")

            # Train sweep
            train_iv = {d: self._iv_by_date.get(d, 0.15) for d, _ in train_days}
            train_years = self._compute_num_years(train_days)

            print(f"  Training {len(combos)} combos...", flush=True)
            t0 = time_mod.time()

            train_rows = self._run_all(combos, train_days, train_iv,
                                        train_years, min_trades)
            train_rows.sort(key=lambda r: r["rank_score"], reverse=True)

            elapsed = time_mod.time() - t0
            print(f"  {len(train_rows)} valid combos in {elapsed:.1f}s "
                  f"(best score={train_rows[0]['rank_score']:.2f})" if train_rows else "  0 valid")

            if not train_rows:
                continue

            # Validate top-N
            candidates = train_rows[:top_n]
            val_iv = {d: self._iv_by_date.get(d, 0.15) for d, _ in val_days}
            val_years = self._compute_num_years(val_days)

            print(f"  Validating top-{len(candidates)} on val window...", flush=True)

            val_combos = [{k2: r[k2] for k2 in keys} for r in candidates]
            val_rows = self._run_all(val_combos, val_days, val_iv,
                                     val_years, max(min_trades // 3, 3))
            val_rows.sort(key=lambda r: r["rank_score"], reverse=True)

            print(f"  {len(val_rows)} valid val combos")

            for r in val_rows[:top_n]:
                pkey = self._param_key(r, keys)
                votes.setdefault(pkey, []).append(r["rank_score"])

            best_train = train_rows[0] if train_rows else {}
            best_val = val_rows[0] if val_rows else {}
            fold_results.append({
                "fold": k + 1,
                "train_start": str(train_dates[0]),
                "train_end": str(train_dates[-1]),
                "val_start": str(val_dates[0]),
                "val_end": str(val_dates[-1]),
                "train_best_score": best_train.get("rank_score", 0),
                "val_best_score": best_val.get("rank_score", 0),
            })

        # Select winner: most appearances, tie-break by mean val score
        if not votes:
            print("\nNo valid combinations found across folds!")
            return {}, fold_results

        winner_key = max(votes, key=lambda k: (len(votes[k]),
                                                 sum(votes[k]) / len(votes[k])))
        # Reconstruct params from key
        best_params = dict(zip(keys, [self._parse_val(v) for v in winner_key.split("|")]))

        appearances = len(votes[winner_key])
        avg_score = sum(votes[winner_key]) / appearances

        print(f"\n{'='*60}")
        print(f"  WF-CV Winner: appeared in {appearances}/{n_folds} folds, "
              f"avg val score={avg_score:.2f}")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
        print(f"{'='*60}")

        return best_params, fold_results

    def _run_all(self, combos, days, iv, num_years, min_trades):
        """Run all combos in-process (avoids pickling 625MB CBBO to workers)."""
        cbbo = CbboStore("", preloaded=self._cbbo_mem) if self._cbbo_mem else None
        rows = []
        for params in combos:
            result = _run_combo_inproc(
                params, self._base_config, days, iv, cbbo,
                min_trades, self._account_size, self._es_multiplier, num_years,
            )
            if result is not None:
                rows.append(result)
        return rows

    @staticmethod
    def _param_key(row, keys):
        return "|".join(str(row[k]) for k in keys)

    @staticmethod
    def _parse_val(s):
        """Parse string back to typed value."""
        if s in ("True", "False"):
            return s == "True"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s

    @staticmethod
    def _compute_num_years(days):
        if not days:
            return 1.0
        dates = [d[0] for d in days]
        span = (max(dates) - min(dates)).days
        return max(span / 365.25, 0.5)

    def save_optimized_params(self, best_params, fold_results, output_path):
        """Save winner to YAML."""
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dump = {
            "optimized_at": now_str,
            "options": {k: v for k, v in best_params.items()
                        if k in ("em_wing_fraction", "tp_fixed_dollars",
                                 "sl_fixed_dollars", "slippage_per_leg")},
            "strategy": {},
            "_cv_folds": len(fold_results),
        }
        if "macro_filter_enabled" in best_params:
            dump["strategy"]["macro_filter"] = {"enabled": best_params["macro_filter_enabled"]}
        if "orb_filter_tolerance_pct" in best_params:
            dump["strategy"]["orb_filter_tolerance_pct"] = best_params["orb_filter_tolerance_pct"]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"# Iron Fly optimized params (WF-CV)\n# Generated: {now_str}\n\n")
            yaml.dump(dump, f, default_flow_style=False, sort_keys=False)
        print(f"Saved to {output_path}")
