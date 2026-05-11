"""
OpenIcStrategy — 0DTE iron condor entered at a fixed time near the open.

Selects short strikes by target Black-Scholes delta calibrated against the
ATM straddle (so IV matches observed CBBO premiums). Wing width is fixed via
config. Designed for short delta and wing width to be optimized via WF-CV.

Exit logic mirrors OrbEmIronCondorStrategy:
  - per-side TP at tp_pct_of_credit
  - breach handling per hold_put_breach / hold_call_breach
"""

from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

from src_v2.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.strategies.utils import mid_price as _mid
from src_8am.backtest.pricing import (
    bs_call, bs_put, bs_call_delta, bs_put_delta,
)


class OpenIcStrategy(LiveStrategy):

    def configure(self, config: Dict[str, Any]) -> None:
        opt = config.get("options", {})
        self._strike_step = opt.get("strike_step_pts", 5.0)
        self._target_short_delta = opt.get("target_short_delta", 0.15)
        self._wing_width = opt.get("wing_width", 25)
        self._min_credit = opt.get("min_credit_per_spread", 0.10)
        self._max_credit_imbalance = opt.get("max_credit_imbalance", 1.00)
        # tp_pct_of_credit is the EXIT THRESHOLD as a fraction of total credit:
        #   value <= tp_pct_of_credit × credit  →  exit
        # So tp_pct=0.75 means "exit when value drops to 75% of credit collected"
        # which captures 25% of max profit (tastytrade's "25% TP" rule).
        # And tp_pct=0.25 means "let value drop to 25% of credit" → capture 75% of max.
        self._tp_pct = opt.get("tp_pct_of_credit", 0.50)
        self._hold_put_breach = opt.get("hold_put_breach", False)
        self._hold_call_breach = opt.get("hold_call_breach", False)
        self._slippage = opt.get("slippage_per_spread", 0.10)
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)
        self._sl_mult = opt.get("sl_mult_of_credit", 0.0)
        # tp_mode: "per_side" closes one side when that side's spread mark <= tp_pct*side_credit.
        #          "total"     closes the WHOLE IC when (call_mark+put_mark) <= tp_pct*total_credit
        #                      (tastytrade-style: close at X% of max profit).
        self._tp_mode = opt.get("tp_mode", "per_side")
        # If True, closing one side via SL or breach also closes the other side
        # (strict tastytrade: trade is one IC, not two independent spreads).
        self._close_remaining_on_first_stop = opt.get("close_remaining_on_first_stop", False)

        market_close = config.get("market", {}).get("market_close", "16:00")
        h, m = [int(x) for x in market_close.split(":")]
        self._market_close = time(h, m)

        self._now_dt: Optional[datetime] = None  # injected by engine
        self._last_em_pts: Optional[float] = None

    def create_engine(self, deps: EngineDeps) -> BaseEngine:
        from src_v2.agent.window_entry_engine import WindowEntryEngine
        return WindowEntryEngine(deps, self)

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

        t = self._time_to_expiration_years()
        if t <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        chain_by = self._chain_map(chain)

        iv = self._derive_iv_from_atm(chain_by, spot, t)
        if iv is None:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        em = iv * spot * math.sqrt(t)
        self._last_em_pts = em

        # Find short strikes by target delta on each side.
        short_call_k = self._find_strike_for_delta(
            chain_by, spot, self._target_short_delta, "call", t, iv,
        )
        short_put_k = self._find_strike_for_delta(
            chain_by, spot, self._target_short_delta, "put", t, iv,
        )
        if short_call_k is None or short_put_k is None:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        long_call_k = short_call_k + self._wing_width
        long_put_k = short_put_k - self._wing_width

        sc = chain_by.get((short_call_k, "call"))
        lc = chain_by.get((long_call_k, "call"))
        sp = chain_by.get((short_put_k, "put"))
        lp = chain_by.get((long_put_k, "put"))

        if not all([sc, lc, sp, lp]):
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        sc_mid = _mid(sc); lc_mid = _mid(lc)
        sp_mid = _mid(sp); lp_mid = _mid(lp)
        if None in (sc_mid, lc_mid, sp_mid, lp_mid):
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        call_credit = round(sc_mid - lc_mid - self._slippage, 2)  # type: ignore
        put_credit = round(sp_mid - lp_mid - self._slippage, 2)  # type: ignore

        if call_credit < self._min_credit or put_credit < self._min_credit:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}
        if abs(call_credit - put_credit) > self._max_credit_imbalance:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        total_credit = round(call_credit + put_credit, 2)
        call_delta = bs_call_delta(spot, short_call_k, t, iv)
        put_delta = abs(bs_put_delta(spot, short_put_k, t, iv))

        setup = LiveTradeSetup(
            strategy_type="open_ic",
            legs=[
                OrderLeg("", short_put_k, "put", "sell", self._quantity),
                OrderLeg("", long_put_k, "put", "buy", self._quantity),
                OrderLeg("", short_call_k, "call", "sell", self._quantity),
                OrderLeg("", long_call_k, "call", "buy", self._quantity),
            ],
            total_credit=total_credit,
            strikes={
                "short_put": short_put_k,
                "long_put": long_put_k,
                "short_call": short_call_k,
                "long_call": long_call_k,
            },
            extra={
                "put_credit": put_credit,
                "call_credit": call_credit,
                "iv": round(iv, 4),
                "em_pts": round(em, 2),
                "wing_width": self._wing_width,
                "target_short_delta": self._target_short_delta,
                "call_delta": round(call_delta, 4),
                "put_delta": round(put_delta, 4),
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
        checks, failures = [], []
        s = setup.strikes

        if s["short_put"] <= s["long_put"]:
            failures.append("put_strike_order")
        if s["long_call"] <= s["short_call"]:
            failures.append("call_strike_order")

        for side in ("put_credit", "call_credit"):
            if setup.extra.get(side, 0) < self._min_credit:
                failures.append(f"{side}_min")

        return {"passed": len(failures) == 0, "checks": checks, "failures": failures}

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
            return {
                "mode": "breach_only",
                "hold_put_breach": self._hold_put_breach,
                "hold_call_breach": self._hold_call_breach,
            }
        return {
            "mode": "credit_mult",
            "sl_mult": self._sl_mult,
            "put_sl_debit": round(setup.extra.get("put_credit", 0) * self._sl_mult, 2),
            "call_sl_debit": round(setup.extra.get("call_credit", 0) * self._sl_mult, 2),
            "hold_put_breach": self._hold_put_breach,
            "hold_call_breach": self._hold_call_breach,
        }

    def build_stop_orders(self, setup, stop_info) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        # Total-credit TP: close ALL legs when combined mark <= tp_pct * total_credit.
        if self._tp_mode == "total":
            put_mark = self._spread_mark(trade, "put", quotes)
            call_mark = self._spread_mark(trade, "call", quotes)
            if put_mark is not None and call_mark is not None:
                total_credit = trade.get("put_credit", 0) + trade.get("call_credit", 0)
                if total_credit > 0:
                    tp_debit = total_credit * self._tp_pct
                    if (put_mark + call_mark) <= tp_debit:
                        legs = self._all_leg_symbols(trade)
                        return ExitAction(
                            action="take_profit",
                            legs_to_close=legs,
                            reason=f"total TP: mark={put_mark + call_mark:.2f} <= tp={tp_debit:.2f}",
                        )

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

            # Per-side TP (only in per_side mode)
            if self._tp_mode == "per_side":
                tp_debit = credit * self._tp_pct
                if credit > 0 and mark <= tp_debit:
                    return ExitAction(
                        action="take_profit",
                        legs_to_close=[short_sym, long_sym],
                        reason=f"{side} TP: mark={mark:.2f} <= tp={tp_debit:.2f}",
                    )

            # Credit-multiple stop loss: mark >= sl_mult * credit (always per-side)
            if self._sl_mult > 0 and credit > 0:
                sl_debit = credit * self._sl_mult
                if mark >= sl_debit:
                    legs = (self._all_leg_symbols(trade)
                            if self._close_remaining_on_first_stop
                            else [short_sym, long_sym])
                    return ExitAction(
                        action="stop_loss",
                        legs_to_close=legs,
                        reason=f"{side} SL: mark={mark:.2f} >= sl={sl_debit:.2f}",
                    )

        underlying = quotes.get("_underlying", {})
        price = underlying.get("price", 0)
        if price > 0:
            short_call = trade.get("short_call_strike", 0)
            short_put = trade.get("short_put_strike", 0)
            call_legs = self._call_leg_symbols(trade)
            put_legs = self._put_leg_symbols(trade)
            if (short_call and call_legs and price >= short_call
                    and not self._hold_call_breach):
                legs = (self._all_leg_symbols(trade)
                        if self._close_remaining_on_first_stop
                        else call_legs)
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=legs,
                    reason=f"call breach: {price} >= {short_call}",
                )
            if (short_put and put_legs and price <= short_put
                    and not self._hold_put_breach):
                legs = (self._all_leg_symbols(trade)
                        if self._close_remaining_on_first_stop
                        else put_legs)
                return ExitAction(
                    action="stop_loss",
                    legs_to_close=legs,
                    reason=f"put breach: {price} <= {short_put}",
                )

        return ExitAction(action="hold")

    def compute_trailing_stop(self, trade, quotes) -> Optional[float]:
        return None

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _time_to_expiration_years(self) -> float:
        now = self._now_dt
        if now is None:
            return 0.0
        close_dt = datetime(
            now.year, now.month, now.day,
            self._market_close.hour, self._market_close.minute,
            tzinfo=now.tzinfo,
        )
        seconds = max(0.0, (close_dt - now).total_seconds())
        return seconds / (365.0 * 24.0 * 3600.0)

    def _derive_iv_from_atm(
        self, chain_by: Dict, spot: float, t: float,
    ) -> Optional[float]:
        atm_k = round(spot / self._strike_step) * self._strike_step
        call = chain_by.get((atm_k, "call"))
        put = chain_by.get((atm_k, "put"))
        if not call or not put:
            return None
        c_mid = _mid(call)
        p_mid = _mid(put)
        if c_mid is None or p_mid is None:
            return None
        target = c_mid + p_mid
        if target <= 0 or t <= 0:
            return None
        lo, hi = 0.02, 4.0
        for _ in range(50):
            mid = (lo + hi) / 2
            s = bs_call(spot, atm_k, t, mid) + bs_put(spot, atm_k, t, mid)
            if s < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _find_strike_for_delta(
        self,
        chain_by: Dict,
        spot: float,
        target_delta: float,
        side: str,
        t: float,
        iv: float,
    ) -> Optional[float]:
        step = self._strike_step
        best_k, best_diff = None, 1.0

        if side == "call":
            k = math.ceil(spot / step) * step
            limit = spot + 250
            while k < limit:
                if (k, "call") in chain_by:
                    d = bs_call_delta(spot, k, t, iv)
                    diff = abs(d - target_delta)
                    if diff < best_diff:
                        best_diff = diff
                        best_k = k
                    if d < target_delta * 0.3:
                        break
                k += step
        else:
            k = math.floor(spot / step) * step
            limit = spot - 250
            while k > limit:
                if (k, "put") in chain_by:
                    d = abs(bs_put_delta(spot, k, t, iv))
                    diff = abs(d - target_delta)
                    if diff < best_diff:
                        best_diff = diff
                        best_k = k
                    if d < target_delta * 0.3:
                        break
                k -= step
        return best_k

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

    @staticmethod
    def _all_leg_symbols(trade: Dict[str, Any]) -> List[str]:
        return [s for s in [
            trade.get("short_put_symbol"), trade.get("long_put_symbol"),
            trade.get("short_call_symbol"), trade.get("long_call_symbol"),
        ] if s]

    @staticmethod
    def _spread_mark(
        trade: Dict[str, Any], side: str, quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        short_sym = trade.get(f"short_{side}_symbol")
        long_sym = trade.get(f"long_{side}_symbol")
        if not short_sym or not long_sym:
            return None
        s_mid = _mid(quotes.get(short_sym))
        l_mid = _mid(quotes.get(long_sym))
        if s_mid is None or l_mid is None:
            return None
        return s_mid - l_mid
