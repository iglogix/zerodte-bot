"""
IronFlyStrategy — live trading strategy for 0DTE ATM Iron Fly.

Based on the backtest strategy (src_8am/backtest/strategy_iron_fly.py) but
adapted for live trading (async, option chain from API, not CBBO store).

Entry: ATM center strike (first above spot), wings at em_wing_fraction × EM.
TP: fixed dollar amount (tp_fixed_dollars) per contract.
SL: underlying crosses wing strike.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src_v2.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.strategies.utils import mid_price as _mid


class IronFlyStrategy(LiveStrategy):

    def configure(self, config: Dict[str, Any]) -> None:
        opt = config.get("options", {})
        self._strike_step = opt.get("strike_step_pts", 5.0)
        self._em_wing_fraction = opt.get("em_wing_fraction", 0.50)
        self._tp_fixed_dollars = opt.get("tp_fixed_dollars", 200.0)
        self._slippage_per_leg = opt.get("slippage_per_leg", 0.05)
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)
        self._em_source = opt.get("em_source", "chain")  # "chain" or "vix"

    # ------------------------------------------------------------------
    # Entry: candidate search
    # ------------------------------------------------------------------

    async def search_candidates(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
    ) -> Dict[str, Any]:
        if not chain:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        step = self._strike_step

        # Center strike = first strike strictly above spot
        center_k = math.ceil(spot / step) * step
        if center_k == spot:
            center_k += step

        # Expected move from ATM straddle in chain
        em = self._compute_em_from_chain(chain, spot)
        if em is None or em <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        # Wing strikes
        wing_dist = em * self._em_wing_fraction
        long_call_k = math.ceil((center_k + wing_dist) / step) * step
        long_put_k = math.floor((center_k - wing_dist) / step) * step

        if long_call_k <= center_k:
            long_call_k = center_k + step
        if long_put_k >= center_k:
            long_put_k = center_k - step

        # Price from chain
        chain_by = {
            (float(o.get("strike", 0)), o.get("option_type", "").upper()): o
            for o in chain
        }

        sc_opt = chain_by.get((center_k, "CALL"))
        sp_opt = chain_by.get((center_k, "PUT"))
        lc_opt = chain_by.get((long_call_k, "CALL"))
        lp_opt = chain_by.get((long_put_k, "PUT"))

        if not all([sc_opt, sp_opt, lc_opt, lp_opt]):
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        sc_mid = _mid(sc_opt)
        sp_mid = _mid(sp_opt)
        lc_mid = _mid(lc_opt)
        lp_mid = _mid(lp_opt)

        if None in (sc_mid, sp_mid, lc_mid, lp_mid):
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        call_credit = round(sc_mid - lc_mid, 4)  # type: ignore
        put_credit = round(sp_mid - lp_mid, 4)  # type: ignore
        total_slippage = self._slippage_per_leg * 4
        total_credit = round(call_credit + put_credit - total_slippage, 2)

        if total_credit <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        setup = LiveTradeSetup(
            strategy_type="iron_fly",
            legs=[
                OrderLeg("", center_k, "call", "sell", self._quantity),
                OrderLeg("", long_call_k, "call", "buy", self._quantity),
                OrderLeg("", center_k, "put", "sell", self._quantity),
                OrderLeg("", long_put_k, "put", "buy", self._quantity),
            ],
            total_credit=total_credit,
            strikes={
                "center": center_k,
                "long_call": long_call_k,
                "long_put": long_put_k,
            },
            extra={
                "call_credit": call_credit,
                "put_credit": put_credit,
                "em_pts": round(em, 2),
                "put_width": center_k - long_put_k,
                "call_width": long_call_k - center_k,
            },
        )

        return {
            "candidates_found": 1,
            "top_candidate": setup,
            "candidates": [setup],
        }

    # ------------------------------------------------------------------
    # Entry: validation
    # ------------------------------------------------------------------

    async def validate(
        self,
        setup: LiveTradeSetup,
        chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        checks = []
        failures = []
        strikes = setup.strikes

        center = strikes.get("center", 0)
        long_put = strikes.get("long_put", 0)
        long_call = strikes.get("long_call", 0)

        # Wing ordering
        if long_put >= center:
            checks.append({"check": "wing_order_put", "passed": False})
            failures.append("wing_order_put")
        else:
            checks.append({"check": "wing_order_put", "passed": True})

        if long_call <= center:
            checks.append({"check": "wing_order_call", "passed": False})
            failures.append("wing_order_call")
        else:
            checks.append({"check": "wing_order_call", "passed": True})

        # Credit positive
        if setup.total_credit <= 0:
            checks.append({"check": "positive_credit", "passed": False})
            failures.append("positive_credit")
        else:
            checks.append({"check": "positive_credit", "passed": True})

        return {
            "passed": len(failures) == 0,
            "checks": checks,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # Entry: order building
    # ------------------------------------------------------------------

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        center = setup.strikes["center"]
        return {
            "order_type": "iron_condor",
            "short_call_strike": center,
            "short_put_strike": center,
            "long_call_strike": setup.strikes["long_call"],
            "long_put_strike": setup.strikes["long_put"],
            "quantity": self._quantity,
        }

    # ------------------------------------------------------------------
    # Stop-loss
    # ------------------------------------------------------------------

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        return {
            "long_call_stop_price": setup.strikes["long_call"],
            "long_put_stop_price": setup.strikes["long_put"],
            "mode": "wing_breach",
        }

    def build_stop_orders(
        self,
        setup: LiveTradeSetup,
        stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Iron fly uses price-based monitoring, not limit stop orders.
        # The position monitor checks underlying price vs wing strikes.
        return []

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        credit = trade.get("credit_received", 0)
        center = trade.get("center_strike", 0)
        long_put = trade.get("long_put", 0)
        long_call = trade.get("long_call", 0)

        # Check underlying price for wing breach
        underlying = quotes.get("_underlying", {})
        price = underlying.get("price", 0)
        if price > 0:
            if price >= long_call:
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=self._all_leg_symbols(trade),
                    reason=f"call wing breach: underlying {price} >= {long_call}",
                )
            if price <= long_put:
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=self._all_leg_symbols(trade),
                    reason=f"put wing breach: underlying {price} <= {long_put}",
                )

        # TP check: mark-to-market the iron fly
        mark = self._compute_mark(trade, quotes)
        if mark is not None and credit > 0:
            tp_pts = self._tp_fixed_dollars / (self._multiplier * self._quantity)
            tp_debit = credit - tp_pts
            if mark <= tp_debit:
                return ExitAction(
                    action="take_profit",
                    legs_to_close=self._all_leg_symbols(trade),
                    reason=f"mark {mark:.2f} <= tp_debit {tp_debit:.2f}",
                )

        return ExitAction(action="hold")

    def compute_trailing_stop(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        # Iron fly doesn't use trailing stops — exit is wing breach or TP.
        return None

    def create_engine(self, deps: "EngineDeps") -> "BaseEngine":
        from src_v2.agent.scheduled_engine import ScheduledEngine
        return ScheduledEngine(deps, self)

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_em_from_chain(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Optional[float]:
        """ATM straddle price = expected move."""
        strikes = set(float(o.get("strike", 0)) for o in chain)
        if not strikes:
            return None
        atm_k = min(strikes, key=lambda k: abs(k - spot))

        chain_by = {
            (float(o.get("strike", 0)), o.get("option_type", "").upper()): o
            for o in chain
        }
        put_opt = chain_by.get((atm_k, "PUT"))
        call_opt = chain_by.get((atm_k, "CALL"))
        if not put_opt or not call_opt:
            return None

        put_mid = _mid(put_opt)
        call_mid = _mid(call_opt)
        if put_mid is None or call_mid is None:
            return None

        return put_mid + call_mid

    def _compute_mark(
        self, trade: Dict[str, Any], quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        """Mark-to-market: sum of both spread values (cost to close)."""
        sc_sym = trade.get("short_call_symbol")
        lc_sym = trade.get("long_call_symbol")
        sp_sym = trade.get("short_put_symbol")
        lp_sym = trade.get("long_put_symbol")

        sc_mid = _mid(quotes.get(sc_sym)) if sc_sym else None
        lc_mid = _mid(quotes.get(lc_sym)) if lc_sym else None
        sp_mid = _mid(quotes.get(sp_sym)) if sp_sym else None
        lp_mid = _mid(quotes.get(lp_sym)) if lp_sym else None

        if None in (sc_mid, lc_mid, sp_mid, lp_mid):
            return None

        call_val = sc_mid - lc_mid  # type: ignore
        put_val = sp_mid - lp_mid  # type: ignore
        return round(call_val + put_val, 2)

    @staticmethod
    def _all_leg_symbols(trade: Dict[str, Any]) -> List[str]:
        syms = []
        for key in ("short_call_symbol", "long_call_symbol",
                     "short_put_symbol", "long_put_symbol"):
            s = trade.get(key)
            if s:
                syms.append(s)
        return syms
