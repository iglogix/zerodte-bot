"""
OrbEmDirectionalStrategy — single with-trend credit spread after ORB retest.

Entry: After ORB breakout + midline retest, sells a single credit spread
on the WITH-TREND side (the one protected by the trend direction):
  - Long breakout (bullish) → sell PCS below spot
  - Short breakout (bearish) → sell CCS above spot

Strike placement: short strike at em_multiplier × ATM_straddle from spot.

Reuses OrbEngine for entry (same three-phase flow as IC variant). The
difference is only in strike selection (single side) and exit logic
(one spread to monitor).

Rationale: in OOS WF-CV, the against-trend leg of the IC contributed
<5% of PnL but ~50% of commission. This removes it entirely.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src_v2.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.strategies.utils import mid_price as _mid


class OrbEmDirectionalStrategy(LiveStrategy):

    def configure(self, config: Dict[str, Any]) -> None:
        opt = config.get("options", {})
        self._strike_step = opt.get("strike_step_pts", 5.0)
        # Reuse the same param name as IC's with-trend params for easy comparison
        self._em_multiplier = opt.get("em_with_trend", 0.5)
        self._width = opt.get("width_with_trend", 15)
        self._min_credit = opt.get("min_credit_per_spread", 0.05)
        self._tp_pct = opt.get("tp_pct_of_credit", 0.75)
        self._sl_mult = opt.get("sl_mult_of_credit", 0.0)
        self._hold_put_breach = opt.get("hold_put_breach", False)
        self._slippage = opt.get("slippage_per_spread", 0.10)
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)
        # Filter: skip days where ATM straddle is too small (calm days have tight
        # strikes that breach easily). IC variant gets this for free because its
        # far-OTM against-trend strike often isn't quoted on calm days.
        self._min_em_pts = opt.get("min_em_pts", 0.0)

    def create_engine(self, deps: EngineDeps) -> BaseEngine:
        from src_v2.agent.orb_engine import OrbEngine
        return OrbEngine(deps, self)

    # ------------------------------------------------------------------
    # Entry: candidate search (single spread)
    # ------------------------------------------------------------------

    async def search_candidates(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
    ) -> Dict[str, Any]:
        empty = {"candidates_found": 0, "top_candidate": None, "candidates": []}
        if not chain or spot <= 0:
            return empty

        em = self._compute_em_from_chain(chain, spot)
        if em is None or em <= 0:
            return empty
        if em < self._min_em_pts:
            return empty

        orb_bias = getattr(self, "_orb_bias", "long")
        step = self._strike_step

        if orb_bias == "long":
            # PCS: short put below spot, long put further below
            short_k = math.floor((spot - self._em_multiplier * em) / step) * step
            long_k = short_k - self._width
            side = "put"
        else:
            # CCS: short call above spot, long call further above
            short_k = math.ceil((spot + self._em_multiplier * em) / step) * step
            long_k = short_k + self._width
            side = "call"

        chain_by = self._chain_map(chain)
        short_opt = chain_by.get((short_k, side))
        long_opt = chain_by.get((long_k, side))
        if not short_opt or not long_opt:
            return empty

        s_mid = _mid(short_opt)
        l_mid = _mid(long_opt)
        if s_mid is None or l_mid is None:
            return empty

        credit = round(s_mid - l_mid - self._slippage, 2)
        if credit < self._min_credit:
            return empty

        strikes = {f"short_{side}": short_k, f"long_{side}": long_k}
        setup = LiveTradeSetup(
            strategy_type="orb_em_directional",
            legs=[
                OrderLeg("", short_k, side, "sell", self._quantity),
                OrderLeg("", long_k, side, "buy", self._quantity),
            ],
            total_credit=credit,
            strikes=strikes,
            extra={
                f"{side}_credit": credit,
                "em_pts": round(em, 2),
                "em_mult": self._em_multiplier,
                "width": self._width,
                "orb_bias": orb_bias,
                "side": side,
                f"{side}_delta": abs(short_opt.get("delta", 0)),  # type: ignore
            },
        )
        return {"candidates_found": 1, "top_candidate": setup, "candidates": [setup]}

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
        side = setup.extra.get("side", "put")
        short_k = setup.strikes.get(f"short_{side}", 0)
        long_k = setup.strikes.get(f"long_{side}", 0)

        # Strike ordering
        if side == "put":
            ok = short_k > long_k
        else:
            ok = long_k > short_k
        checks.append({"check": f"{side}_strike_order", "passed": ok})
        if not ok:
            failures.append(f"{side}_strike_order")

        # Min credit
        credit = setup.extra.get(f"{side}_credit", 0)
        ok = credit >= self._min_credit
        checks.append({"check": f"{side}_min_credit", "passed": ok})
        if not ok:
            failures.append(f"{side}_min_credit")

        return {
            "passed": len(failures) == 0,
            "checks": checks,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # Entry: order building
    # ------------------------------------------------------------------

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        side = setup.extra.get("side", "put")
        order_type = "put_credit_spread" if side == "put" else "call_credit_spread"
        return {
            "order_type": order_type,
            f"short_{side}_strike": setup.strikes[f"short_{side}"],
            f"long_{side}_strike": setup.strikes[f"long_{side}"],
            "quantity": self._quantity,
        }

    # ------------------------------------------------------------------
    # Stop-loss
    # ------------------------------------------------------------------

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        if self._sl_mult <= 0:
            return {"mode": "breach_only", "hold_put_breach": self._hold_put_breach}
        return {
            "mode": "credit_mult",
            "sl_mult": self._sl_mult,
            "sl_debit": round(setup.total_credit * self._sl_mult, 2),
            "hold_put_breach": self._hold_put_breach,
        }

    def build_stop_orders(
        self,
        setup: LiveTradeSetup,
        stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    # Exit — same shape as IC but only one side present in trade dict
    # ------------------------------------------------------------------

    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        # Iterate both sides — only the side with symbols will be evaluated
        for side, short_key, long_key in [
            ("put", "short_put_symbol", "long_put_symbol"),
            ("call", "short_call_symbol", "long_call_symbol"),
        ]:
            short_sym = trade.get(short_key)
            long_sym = trade.get(long_key)
            if not short_sym or not long_sym:
                continue
            s_mid_q = _mid(quotes.get(short_sym))
            l_mid_q = _mid(quotes.get(long_sym))
            if s_mid_q is None or l_mid_q is None:
                continue
            mark = s_mid_q - l_mid_q
            credit = trade.get(f"{side}_credit", 0)
            tp_debit = credit * self._tp_pct
            if credit > 0 and mark <= tp_debit:
                return ExitAction(
                    action="take_profit",
                    legs_to_close=[short_sym, long_sym],
                    reason=f"{side} TP: mark={mark:.2f} <= tp={tp_debit:.2f}",
                )

        # Breach check — only the present side will have a non-zero strike
        underlying = quotes.get("_underlying", {})
        price = underlying.get("price", 0)
        if price > 0:
            short_call = trade.get("short_call_strike", 0)
            short_put = trade.get("short_put_strike", 0)
            if short_call and price >= short_call:
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=[s for s in [trade.get("short_call_symbol"),
                                                trade.get("long_call_symbol")] if s],
                    reason=f"call breach: {price} >= {short_call}",
                )
            if short_put and price <= short_put and not self._hold_put_breach:
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=[s for s in [trade.get("short_put_symbol"),
                                                trade.get("long_put_symbol")] if s],
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
    # Helpers (shared with IC variant)
    # ------------------------------------------------------------------

    def _compute_em_from_chain(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Optional[float]:
        strikes = set(float(o.get("strike", 0)) for o in chain)
        if not strikes:
            return None
        step = self._strike_step
        atm_k = round(spot / step) * step
        chain_by = self._chain_map(chain)
        put_opt = chain_by.get((atm_k, "put"))
        call_opt = chain_by.get((atm_k, "call"))
        if not put_opt or not call_opt:
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
