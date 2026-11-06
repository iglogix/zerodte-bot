"""
IronCondorStrategy — extracted from tool_handlers.py, risk_manager.py, position_monitor.py.

Implements LiveStrategy ABC for 0DTE iron condor trades:
  - search_candidates: balanced call/put credit spreads at target delta
  - validate: strike order, delta range, credit range, credit equality
  - build_order: 4-leg IC order payload
  - check_exit: per-leg TP at $0.05 threshold
  - stop_loss: credit-based with OCO stop structure
  - trailing_stop: tighten as long options decay
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from zerodte.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from zerodte.agent.base_engine import BaseEngine, EngineDeps
from zerodte.strategies.utils import mid_price as _mid


class IronCondorStrategy(LiveStrategy):

    def configure(self, config: Dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        stop = config.get("stop_loss", {})
        risk = config.get("risk", {})

        self._credit_equality_threshold = strat.get("credit_equality_threshold", 0.20)
        self._skew_threshold = strat.get("skew_threshold", 1.15)
        self._min_short_delta = strat.get("min_short_delta", 0.10)
        self._max_short_delta = strat.get("max_short_delta", 0.15)
        self._min_side_credit = strat.get("min_side_credit", 1.00)
        self._max_side_credit = strat.get("max_side_credit", 2.00)
        self._wing_width = strat.get("wing_width", 30)

        self._stop_limit_gap = stop.get("stop_limit_gap", 40)
        self._stop_market_gap = stop.get("stop_market_gap", 30)
        self._take_profit_threshold = risk.get("take_profit_threshold", 0.05)

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

        calls = sorted(
            [o for o in chain if o.get("option_type", "").upper() in ("C", "CALL")],
            key=lambda o: o.get("strike", 0),
        )
        puts = sorted(
            [o for o in chain if o.get("option_type", "").upper() in ("P", "PUT")],
            key=lambda o: o.get("strike", 0),
            reverse=True,
        )

        chain_by_strike: Dict[Tuple[float, str], Dict] = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }

        # Filter to target delta range (with buffer)
        search_min = max(0.01, self._min_short_delta - 0.05)
        search_max = self._max_short_delta + 0.05

        sc_candidates = [
            c for c in calls
            if search_min <= abs(c.get("delta", 0)) <= search_max
        ]
        sp_candidates = [
            p for p in puts
            if search_min <= abs(p.get("delta", 0)) <= search_max
        ]

        w = self._wing_width
        widths_to_try = sorted({max(5, w - 5), w, w + 5})
        candidates: List[LiveTradeSetup] = []

        for sc in sc_candidates:
            sc_mid_val = _mid(sc)
            if sc_mid_val is None:
                continue
            sc_strike = float(sc.get("strike", 0))

            for sp in sp_candidates:
                sp_mid_val = _mid(sp)
                if sp_mid_val is None:
                    continue
                sp_strike = float(sp.get("strike", 0))

                for wing_w in widths_to_try:
                    lc_strike = sc_strike + wing_w
                    lp_strike = sp_strike - wing_w

                    lc_opt = chain_by_strike.get((lc_strike, "call"))
                    lp_opt = chain_by_strike.get((lp_strike, "put"))
                    if lc_opt is None or lp_opt is None:
                        continue

                    lc_mid_val = _mid(lc_opt)
                    lp_mid_val = _mid(lp_opt)
                    if lc_mid_val is None or lp_mid_val is None:
                        continue

                    call_credit = round(sc_mid_val - lc_mid_val, 4)
                    put_credit = round(sp_mid_val - lp_mid_val, 4)

                    if call_credit < self._min_side_credit or call_credit > self._max_side_credit:
                        continue
                    if put_credit < self._min_side_credit or put_credit > self._max_side_credit:
                        continue

                    credit_diff = abs(call_credit - put_credit)
                    if credit_diff > self._credit_equality_threshold:
                        continue

                    total_credit = round(call_credit + put_credit, 4)
                    candidates.append(LiveTradeSetup(
                        strategy_type="iron_condor",
                        legs=[
                            OrderLeg("", sc_strike, "call", "sell", 1),
                            OrderLeg("", lc_strike, "call", "buy", 1),
                            OrderLeg("", sp_strike, "put", "sell", 1),
                            OrderLeg("", lp_strike, "put", "buy", 1),
                        ],
                        total_credit=total_credit,
                        strikes={
                            "short_call": sc_strike,
                            "long_call": lc_strike,
                            "short_put": sp_strike,
                            "long_put": lp_strike,
                        },
                        extra={
                            "wing_width": wing_w,
                            "call_credit": call_credit,
                            "put_credit": put_credit,
                            "credit_diff": round(credit_diff, 4),
                            "call_delta": round(abs(sc.get("delta", 0)), 4),
                            "put_delta": round(abs(sp.get("delta", 0)), 4),
                            "total_premium": total_credit,
                        },
                    ))

        candidates.sort(key=lambda c: c.extra["credit_diff"])
        candidates = candidates[:5]
        top = candidates[0] if candidates else None

        return {
            "candidates_found": len(candidates),
            "top_candidate": top,
            "candidates": candidates,
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

        sc = strikes.get("short_call", 0)
        lc = strikes.get("long_call", 0)
        sp = strikes.get("short_put", 0)
        lp = strikes.get("long_put", 0)

        # Strike ordering
        if lc <= sc:
            checks.append({"check": "strike_order_call", "passed": False})
            failures.append("strike_order_call")
        else:
            checks.append({"check": "strike_order_call", "passed": True})

        if sp <= lp:
            checks.append({"check": "strike_order_put", "passed": False})
            failures.append("strike_order_put")
        else:
            checks.append({"check": "strike_order_put", "passed": True})

        # Wing symmetry
        call_width = lc - sc
        put_width = sp - lp
        if call_width > 0 and put_width > 0 and abs(call_width - put_width) > 5:
            checks.append({"check": "wing_symmetry", "passed": False})
            failures.append("wing_symmetry")
        else:
            checks.append({"check": "wing_symmetry", "passed": True})

        # Delta check from chain
        chain_by_strike = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }
        for label, strike, otype in [("short_call", sc, "call"), ("short_put", sp, "put")]:
            opt = chain_by_strike.get((strike, otype))
            if opt:
                delta = abs(opt.get("delta", 0))
                if delta < self._min_short_delta or delta > self._max_short_delta:
                    checks.append({"check": f"delta_{label}", "passed": False})
                    failures.append(f"delta_{label}")
                else:
                    checks.append({"check": f"delta_{label}", "passed": True})

        # Credit check
        call_credit = setup.extra.get("call_credit")
        put_credit = setup.extra.get("put_credit")
        if call_credit is not None:
            if call_credit < self._min_side_credit or call_credit > self._max_side_credit:
                checks.append({"check": "call_credit_range", "passed": False})
                failures.append("call_credit_range")
            else:
                checks.append({"check": "call_credit_range", "passed": True})
        if put_credit is not None:
            if put_credit < self._min_side_credit or put_credit > self._max_side_credit:
                checks.append({"check": "put_credit_range", "passed": False})
                failures.append("put_credit_range")
            else:
                checks.append({"check": "put_credit_range", "passed": True})

        # Credit equality
        if call_credit is not None and put_credit is not None:
            diff = abs(call_credit - put_credit)
            if diff > self._credit_equality_threshold:
                checks.append({"check": "credit_equality", "passed": False})
                failures.append("credit_equality")
            else:
                checks.append({"check": "credit_equality", "passed": True})

        return {
            "passed": len(failures) == 0,
            "checks": checks,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # Entry: order building
    # ------------------------------------------------------------------

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        return {
            "order_type": "iron_condor",
            "short_call_strike": setup.strikes["short_call"],
            "long_call_strike": setup.strikes["long_call"],
            "short_put_strike": setup.strikes["short_put"],
            "long_put_strike": setup.strikes["long_put"],
            "quantity": 1,
        }

    # ------------------------------------------------------------------
    # Stop-loss
    # ------------------------------------------------------------------

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        total_premium = setup.extra.get("total_premium", setup.total_credit)
        stop_dollars = total_premium * 100
        return {
            "short_call_stop": stop_dollars,
            "short_put_stop": stop_dollars,
            "total_premium": total_premium,
            "stop_dollars": stop_dollars,
        }

    def build_stop_orders(
        self,
        setup: LiveTradeSetup,
        stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        orders = []
        for side, credit_key in [("short_call", "call_credit"), ("short_put", "put_credit")]:
            stop_amount = stop_info.get(f"{side}_stop", 0)
            short_price = setup.extra.get(credit_key, 0)
            trigger = short_price + stop_amount
            orders.append({
                "side": side,
                "stop_limit": {
                    "trigger_price": trigger,
                    "limit_price": trigger + self._stop_limit_gap,
                    "trigger_gap": self._stop_limit_gap,
                },
                "stop_market": {
                    "trigger_price": trigger + self._stop_market_gap,
                    "additional_gap": self._stop_market_gap,
                },
            })
        return orders

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        legs_to_close = []
        for short_key in ("short_call_symbol", "short_put_symbol"):
            symbol = trade.get(short_key)
            if not symbol:
                continue
            q = quotes.get(symbol)
            if q is None:
                continue
            # Use ask price (cost to buy back), matching original position_monitor
            price = q.get("price") or q.get("ask") or 0.0
            if price > 0 and price <= self._take_profit_threshold:
                legs_to_close.append(symbol)

        if legs_to_close:
            return ExitAction(
                action="take_profit",
                legs_to_close=legs_to_close,
                reason=f"short leg(s) at/below ${self._take_profit_threshold}",
            )
        return ExitAction(action="hold")

    def compute_trailing_stop(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        original_stop = trade.get("original_stop_loss")
        if original_stop is None:
            return None

        total_decay = 0.0
        for sym_key, orig_key in [
            ("long_call_symbol", "original_long_call_value"),
            ("long_put_symbol", "original_long_put_value"),
        ]:
            orig_val = trade.get(orig_key, 0)
            symbol = trade.get(sym_key)
            q = quotes.get(symbol) if symbol else None
            if q and orig_val > 0:
                current = _mid(q) or 0
                decay = orig_val - current
                if decay > 0:
                    total_decay += decay

        if total_decay <= 0:
            return None
        new_stop = original_stop - total_decay
        return max(new_stop, 0.01)

    def create_engine(self, deps: "EngineDeps") -> "BaseEngine":
        from zerodte.agent.scheduled_engine import ScheduledEngine
        return ScheduledEngine(deps, self)

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]
