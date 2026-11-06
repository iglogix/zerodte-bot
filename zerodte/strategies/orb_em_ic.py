"""
OrbEmIronCondorStrategy — ORB-based Expected Move iron condor for 0DTE SPX.

Entry: bias-adaptive IC using ATM straddle as expected move.
  - With-trend leg placed at em_with_trend × EM (closer, more premium)
  - Against-trend leg placed at em_against_trend × EM (further, less risk)

TP: 50% of credit per spread.
SL: configurable multiplier of credit (0 = disabled, exit on breach only).
Breach: put breaches can be held (hold_put_breach=true).

Optimized params (125-day SPX CBBO backtest, put-call parity spot):
  93.1% WR, Sharpe 14.44, PF 4.35, 5 breaches on 72 trades.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from zerodte.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from zerodte.agent.base_engine import BaseEngine, EngineDeps
from zerodte.strategies.utils import mid_price as _mid


class OrbEmIronCondorStrategy(LiveStrategy):

    def configure(self, config: Dict[str, Any]) -> None:
        opt = config.get("options", {})
        self._strike_step = opt.get("strike_step_pts", 5.0)
        self._em_with_trend = opt.get("em_with_trend", 0.50)
        self._em_against_trend = opt.get("em_against_trend", 0.75)
        self._width_with_trend = opt.get("width_with_trend", 5)
        self._width_against_trend = opt.get("width_against_trend", 10)
        self._min_credit = opt.get("min_credit_per_spread", 0.05)
        self._tp_pct = opt.get("tp_pct_of_credit", 0.50)
        self._sl_mult = opt.get("sl_mult_of_credit", 0.0)
        self._hold_put_breach = opt.get("hold_put_breach", True)
        self._slippage = opt.get("slippage_per_spread", 0.10)
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)

    def create_engine(self, deps: EngineDeps) -> BaseEngine:
        from zerodte.agent.orb_engine import OrbEngine
        return OrbEngine(deps, self)

    # ------------------------------------------------------------------
    # Entry: candidate search
    # ------------------------------------------------------------------

    async def search_candidates(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
    ) -> Dict[str, Any]:
        if not chain or spot <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        em = self._compute_em_from_chain(chain, spot)
        if em is None or em <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        # ORB bias is injected via extra context — default to "long" if not set
        orb_bias = getattr(self, "_orb_bias", "long")

        # With-trend: put if long bias, call if short bias
        # Against-trend: the opposite side
        put_is_with = orb_bias == "long"

        put_em_mult = self._em_with_trend if put_is_with else self._em_against_trend
        call_em_mult = self._em_against_trend if put_is_with else self._em_with_trend
        put_width = self._width_with_trend if put_is_with else self._width_against_trend
        call_width = self._width_against_trend if put_is_with else self._width_with_trend

        step = self._strike_step

        # Put credit spread
        put_short_k = math.floor((spot - put_em_mult * em) / step) * step
        put_long_k = put_short_k - put_width

        # Call credit spread
        call_short_k = math.ceil((spot + call_em_mult * em) / step) * step
        call_long_k = call_short_k + call_width

        # Price from chain
        chain_by = self._chain_map(chain)

        put_short_opt = chain_by.get((put_short_k, "put"))
        put_long_opt = chain_by.get((put_long_k, "put"))
        call_short_opt = chain_by.get((call_short_k, "call"))
        call_long_opt = chain_by.get((call_long_k, "call"))

        if not all([put_short_opt, put_long_opt, call_short_opt, call_long_opt]):
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        ps_mid = _mid(put_short_opt)
        pl_mid = _mid(put_long_opt)
        cs_mid = _mid(call_short_opt)
        cl_mid = _mid(call_long_opt)

        if None in (ps_mid, pl_mid, cs_mid, cl_mid):
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        put_credit = round(ps_mid - pl_mid - self._slippage, 2) # type: ignore
        call_credit = round(cs_mid - cl_mid - self._slippage, 2) # type: ignore

        if put_credit < self._min_credit or call_credit < self._min_credit:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        total_credit = round(put_credit + call_credit, 2)

        setup = LiveTradeSetup(
            strategy_type="orb_em_ic",
            legs=[
                OrderLeg("", put_short_k, "put", "sell", self._quantity),
                OrderLeg("", put_long_k, "put", "buy", self._quantity),
                OrderLeg("", call_short_k, "call", "sell", self._quantity),
                OrderLeg("", call_long_k, "call", "buy", self._quantity),
            ],
            total_credit=total_credit,
            strikes={
                "short_put": put_short_k,
                "long_put": put_long_k,
                "short_call": call_short_k,
                "long_call": call_long_k,
            },
            extra={
                "put_credit": put_credit,
                "call_credit": call_credit,
                "em_pts": round(em, 2),
                "put_em_mult": put_em_mult,
                "call_em_mult": call_em_mult,
                "put_width": put_width,
                "call_width": call_width,
                "orb_bias": orb_bias,
                "put_trend": "with" if put_is_with else "against",
                "call_trend": "against" if put_is_with else "with",
                "put_delta": abs(put_short_opt.get("delta", 0)), # type: ignore
                "call_delta": abs(call_short_opt.get("delta", 0)), # type: ignore
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
        s = setup.strikes

        # Strike ordering
        if s["short_put"] <= s["long_put"]:
            checks.append({"check": "put_strike_order", "passed": False})
            failures.append("put_strike_order")
        else:
            checks.append({"check": "put_strike_order", "passed": True})

        if s["long_call"] <= s["short_call"]:
            checks.append({"check": "call_strike_order", "passed": False})
            failures.append("call_strike_order")
        else:
            checks.append({"check": "call_strike_order", "passed": True})

        # Positive credit per side
        for side in ("put_credit", "call_credit"):
            val = setup.extra.get(side, 0)
            if val < self._min_credit:
                checks.append({"check": f"{side}_min", "passed": False})
                failures.append(f"{side}_min")
            else:
                checks.append({"check": f"{side}_min", "passed": True})

        return {
            "passed": len(failures) == 0,
            "checks": checks,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # Entry: order building
    # ------------------------------------------------------------------

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        s = setup.strikes
        return {
            "order_type": "iron_condor",
            "short_put_strike": s["short_put"],
            "long_put_strike": s["long_put"],
            "short_call_strike": s["short_call"],
            "long_call_strike": s["long_call"],
            "quantity": self._quantity,
        }

    # ------------------------------------------------------------------
    # Stop-loss
    # ------------------------------------------------------------------

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        if self._sl_mult <= 0:
            return {"mode": "breach_only", "hold_put_breach": self._hold_put_breach}
        put_credit = setup.extra.get("put_credit", 0)
        call_credit = setup.extra.get("call_credit", 0)
        return {
            "mode": "credit_mult",
            "sl_mult": self._sl_mult,
            "put_sl_debit": round(put_credit * self._sl_mult, 2),
            "call_sl_debit": round(call_credit * self._sl_mult, 2),
            "hold_put_breach": self._hold_put_breach,
        }

    def build_stop_orders(
        self,
        setup: LiveTradeSetup,
        stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # EM IC uses mark-based monitoring, not limit stop orders.
        return []

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        # Per-side TP check: spread mark <= tp_debit
        for side, short_key, long_key in [
            ("put", "short_put_symbol", "long_put_symbol"),
            ("call", "short_call_symbol", "long_call_symbol"),
        ]:
            short_sym = trade.get(short_key)
            long_sym = trade.get(long_key)
            if not short_sym or not long_sym:
                continue
            s_mid = _mid(quotes.get(short_sym))
            l_mid = _mid(quotes.get(long_sym))
            if s_mid is None or l_mid is None:
                continue
            mark = s_mid - l_mid
            credit = trade.get(f"{side}_credit", 0)
            tp_debit = credit * self._tp_pct
            if credit > 0 and mark <= tp_debit:
                return ExitAction(
                    action="take_profit",
                    legs_to_close=[short_sym, long_sym],
                    reason=f"{side} TP: mark={mark:.2f} <= tp={tp_debit:.2f}",
                )

        # Breach check via underlying price
        underlying = quotes.get("_underlying", {})
        price = underlying.get("price", 0)
        if price > 0:
            short_call = trade.get("short_call_strike", 0)
            short_put = trade.get("short_put_strike", 0)
            if short_call and price >= short_call:
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=self._call_leg_symbols(trade),
                    reason=f"call breach: {price} >= {short_call}",
                )
            if short_put and price <= short_put and not self._hold_put_breach:
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=self._put_leg_symbols(trade),
                    reason=f"put breach: {price} <= {short_put}",
                )

        return ExitAction(action="hold")

    def compute_trailing_stop(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        return None

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
        step = self._strike_step
        atm_k = round(spot / step) * step
        chain_by = self._chain_map(chain)
        put_opt = chain_by.get((atm_k, "put"))
        call_opt = chain_by.get((atm_k, "call"))
        if not put_opt or not call_opt:
            # Try nearest strike
            atm_k = min(strikes, key=lambda k: abs(k - spot))
            put_opt = chain_by.get((atm_k, "put"))
            call_opt = chain_by.get((atm_k, "call"))
        if not put_opt or not call_opt:
            return None
        p_mid = _mid(put_opt)
        c_mid = _mid(call_opt)
        if p_mid is None or c_mid is None:
            return None
        return p_mid + c_mid

    @staticmethod
    def _chain_map(chain: List[Dict[str, Any]]) -> Dict[Tuple[float, str], Dict]:
        return {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }

    @staticmethod
    def _call_leg_symbols(trade: Dict[str, Any]) -> List[str]:
        return [s for s in [trade.get("short_call_symbol"), trade.get("long_call_symbol")] if s]

    @staticmethod
    def _put_leg_symbols(trade: Dict[str, Any]) -> List[str]:
        return [s for s in [trade.get("short_put_symbol"), trade.get("long_put_symbol")] if s]
