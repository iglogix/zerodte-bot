"""
MeicStrategy — SPX 0DTE Multiple-Entries Iron Condor.

Implements LiveStrategy for tranche-based IC entries with:
  - AM/PM premium-band candidate search across 80-100pt wings
  - per-tranche contract sizing from a config map
  - two-spread routing (independent put/call credit spreads)
  - stop-market on short legs at % of total credit
  - AM/PM scale-out tiers via partial-close ExitActions
  - leftover-long abandonment below a price threshold

Runs on ScheduledEngine (denser schedule than the base IC).
"""

from __future__ import annotations

import math
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src_v2.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.strategies.utils import mid_price as _mid

ET = ZoneInfo("America/New_York")


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


class MeicStrategy(LiveStrategy):

    def __init__(self):
        # (trade_id, side) -> set of fired tier indices
        self._scaleout_state: Dict[Tuple[str, str], set] = {}
        # Injectable clock for testability; tests override this.
        self._now_fn = lambda: datetime.now(ET)

    # ------------------------------------------------------------------
    # Configure
    # ------------------------------------------------------------------

    def configure(self, config: Dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self._am_pm_cutoff = _parse_hhmm(strat.get("am_pm_cutoff", "12:30"))
        self._am_min_credit = strat.get("am_min_credit", 1.25)
        self._am_max_credit = strat.get("am_max_credit", 1.65)
        self._pm_min_credit = strat.get("pm_min_credit", 0.80)
        self._pm_max_credit = strat.get("pm_max_credit", 1.30)
        self._wing_widths = list(strat.get("wing_widths", [80, 90, 100]))
        self._stop_pct = strat.get("stop_pct_of_total_credit", 0.70)
        if not (0.65 <= self._stop_pct <= 0.90):
            raise ValueError(
                f"stop_pct_of_total_credit must be in [0.65, 0.90], got {self._stop_pct}"
            )
        self._am_tiers = list(strat.get("am_scale_out_tiers", [0.55, 0.65, 0.75, 0.85]))
        self._am_fractions = list(strat.get("am_scale_out_fractions", [0.25, 0.25, 0.25, 0.25]))
        self._pm_tiers = list(strat.get("pm_scale_out_tiers", [0.65, 0.85]))
        self._pm_fractions = list(strat.get("pm_scale_out_fractions", [0.50, 0.50]))
        self._abandon_long_below = strat.get("abandon_long_below", 0.05)
        self._credit_equality_threshold = strat.get("credit_equality_threshold", 0.30)
        self._default_tranche_qty = strat.get("default_tranche_qty", 3)
        self._tranche_sizing = dict(strat.get("tranche_sizing", {}))

    # ------------------------------------------------------------------
    # Simple methods
    # ------------------------------------------------------------------

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    def compute_trailing_stop(
        self, trade: Dict[str, Any], quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        return None

    def create_engine(self, deps: "EngineDeps") -> "BaseEngine":
        from src_v2.agent.scheduled_engine import ScheduledEngine
        return ScheduledEngine(deps, self)

    # ------------------------------------------------------------------
    # Stubs filled by later tasks
    # ------------------------------------------------------------------

    async def search_candidates(self, chain, spot):
        raise NotImplementedError

    async def validate(self, setup, chain):
        raise NotImplementedError

    def build_order(self, setup):
        raise NotImplementedError

    def compute_stop_loss(self, setup):
        raise NotImplementedError

    def build_stop_orders(self, setup, stop_info):
        raise NotImplementedError

    def check_exit(self, trade, quotes):
        raise NotImplementedError
