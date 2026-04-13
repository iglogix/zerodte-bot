"""
GridSearchOptimizer — sweep strategy parameters over a grid, rank by objective.

Preloads CBBO + bars once, runs a BacktestOrchestrator per combo.
Returns (params, metrics) pairs sorted by objective metric.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional

from src_v2.backtest.orchestrator import BacktestOrchestrator
from src_v2.backtest.report import generate_report

logger = logging.getLogger(__name__)


@dataclass
class OptResult:
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    seconds: float = 0.0


class GridSearchOptimizer:
    """Grid search over strategy params using BacktestOrchestrator.

    The param grid is a dict mapping dotted config paths to value lists.
    Supported paths:
      - "options.em_with_trend"
      - "options.em_against_trend"
      - "options.width_with_trend"
      - "options.width_against_trend"
      - "options.tp_pct_of_credit"
      - "strategy.midline_tolerance_pts"
      - "strategy.orb_range_filter.min_pts"
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        strategy_cls,
        cbbo_store,
        days: List[Any],
        param_grid: Dict[str, List[Any]],
        objective: str = "sharpe_annualised",
    ):
        self._base = base_config
        self._strategy_cls = strategy_cls
        self._cbbo = cbbo_store
        self._days = days
        self._grid = param_grid
        self._objective = objective

    def run(self, progress_cb: Optional[Callable[[int, int, OptResult], None]] = None) -> List[OptResult]:
        """Run all grid combos sequentially. Returns results sorted by objective (desc)."""
        combos = self._enumerate_combos()
        total = len(combos)
        logger.info("GridSearch: %d combos", total)

        results: List[OptResult] = []
        for i, combo in enumerate(combos, 1):
            t0 = time.time()
            cfg = self._apply_combo(copy.deepcopy(self._base), combo)

            orchestrator = BacktestOrchestrator(cfg, self._strategy_cls, self._cbbo)
            bt_results = orchestrator.run(self._days)
            report = generate_report(bt_results)

            metrics = {
                "trades": report.get("trades_placed", 0),
                "win_rate": report.get("win_rate", 0),
                "total_pnl": report.get("total_pnl_usd", 0),
                "sharpe_annualised": report.get("sharpe_annualised", 0),
                "profit_factor": report.get("profit_factor", 0),
                "max_drawdown": report.get("max_drawdown_usd", 0),
                "avg_pnl_per_trade": report.get("avg_pnl_per_trade_usd", 0),
            }
            result = OptResult(params=combo, metrics=metrics, seconds=time.time() - t0)
            results.append(result)

            if progress_cb:
                progress_cb(i, total, result)

        # Sort by objective (descending; for drawdown we'd ascend — handle as-needed)
        def _key(r: OptResult):
            v = r.metrics.get(self._objective, 0)
            # Handle "inf" string for profit_factor
            if isinstance(v, str):
                return float("inf") if v == "inf" else 0.0
            return v or 0.0

        return sorted(results, key=_key, reverse=True)

    def _enumerate_combos(self) -> List[Dict[str, Any]]:
        keys = list(self._grid.keys())
        value_lists = [self._grid[k] for k in keys]
        combos = []
        for values in product(*value_lists):
            combos.append(dict(zip(keys, values)))
        return combos

    @staticmethod
    def _apply_combo(config: Dict[str, Any], combo: Dict[str, Any]) -> Dict[str, Any]:
        """Set nested config values via dotted paths (e.g. 'options.em_with_trend')."""
        for path, value in combo.items():
            parts = path.split(".")
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value
        return config
