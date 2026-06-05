"""
MeicStrategy — SPX 0DTE Multiple-Entries Iron Condor.

Implements LiveStrategy for tranche-based IC entries with:
  - expected-move-relative strike selection: EM = ATM straddle (delta-free);
    shorts at short_em_mult x EM OTM, wings at wing_em_frac x EM, snapped to
    the strike step; a credit-to-width ratio gate replaces absolute bands
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
        self._short_em_mult = strat.get("short_em_mult", 1.0)
        self._wing_em_frac = strat.get("wing_em_frac", 0.5)
        self._min_credit_width_ratio = strat.get("min_credit_width_ratio", 0.08)
        self._strike_step = strat.get("strike_step_pts", 5)
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
        # Dislocation guard: reject an entry priced off a leg whose quote spread
        # is both wide (> _disloc_min_spread_pts) AND a large fraction of its mid
        # (> _disloc_spread_mult x mid), e.g. a stale/locked 0.05/45.00 market.
        self._disloc_min_spread_pts = strat.get("dislocation_min_spread_pts", 1.0)
        self._disloc_spread_mult = strat.get("dislocation_spread_mult", 1.5)
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
        if deps.clock is not None:
            self._now_fn = lambda: deps.clock.now(ET)
        return ScheduledEngine(deps, self)

    # ------------------------------------------------------------------
    # Stubs filled by later tasks
    # ------------------------------------------------------------------

    def _tranche_qty(self) -> int:
        slot = self._now_fn().strftime("%H:%M")
        return int(self._tranche_sizing.get(slot, self._default_tranche_qty))

    def _is_dislocated(self, opt: Optional[Dict[str, Any]]) -> bool:
        """A quote is dislocated when its spread is both wide in absolute terms
        AND a large fraction of its mid (e.g. 0.05/45.00). The spread mid of such
        a quote is not a fill you could actually achieve, so the entry is skipped."""
        if not opt:
            return False
        bid = opt.get("bid", 0)
        ask = opt.get("ask", 0)
        if bid <= 0 or ask <= 0:
            return False
        spread = ask - bid
        mid = (bid + ask) / 2
        return (spread > self._disloc_min_spread_pts
                and spread > self._disloc_spread_mult * mid)

    def _compute_em(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Optional[float]:
        """Expected move = ATM straddle (call_mid + put_mid at nearest strike)."""
        if not chain:
            return None
        step = self._strike_step
        by = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }
        strikes = [float(o.get("strike", 0)) for o in chain]
        atm_k = round(spot / step) * step
        cp = by.get((float(atm_k), "call"))
        pp = by.get((float(atm_k), "put"))
        if not cp or not pp:
            atm_k = min(set(strikes), key=lambda k: abs(k - spot))
            cp = by.get((float(atm_k), "call"))
            pp = by.get((float(atm_k), "put"))
        if not cp or not pp:
            return None
        cm = _mid(cp)
        pm = _mid(pp)
        if cm is None or pm is None:
            return None
        return round(cm + pm, 4)

    async def search_candidates(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Dict[str, Any]:
        empty = {"candidates_found": 0, "top_candidate": None, "candidates": []}
        if not chain:
            return empty
        em = self._compute_em(chain, spot)
        if em is None or em <= 0:
            return empty

        step = self._strike_step
        qty = self._tranche_qty()
        by = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }

        short_call = math.ceil((spot + self._short_em_mult * em) / step) * step
        short_put = math.floor((spot - self._short_em_mult * em) / step) * step
        wing_pts = max(step, int(round(self._wing_em_frac * em / step)) * step)
        long_call = short_call + wing_pts
        long_put = short_put - wing_pts

        sc_mid = _mid(by.get((float(short_call), "call")))
        lc_mid = _mid(by.get((float(long_call), "call")))
        sp_mid = _mid(by.get((float(short_put), "put")))
        lp_mid = _mid(by.get((float(long_put), "put")))
        if sc_mid is None or lc_mid is None or sp_mid is None or lp_mid is None:
            return empty

        # Reject if any traded leg's quote is dislocated — its mid is a fantasy
        # fill (e.g. 2025-04-09 13:20 short put 0.05/45.00 on the tariff spike).
        if any(self._is_dislocated(by.get((float(k), t))) for k, t in (
                (short_call, "call"), (long_call, "call"),
                (short_put, "put"), (long_put, "put"))):
            return empty

        call_credit = round(sc_mid - lc_mid, 4)
        put_credit = round(sp_mid - lp_mid, 4)
        if call_credit <= 0 or put_credit <= 0:
            return empty

        call_cw = call_credit / wing_pts
        put_cw = put_credit / wing_pts
        if call_cw < self._min_credit_width_ratio or put_cw < self._min_credit_width_ratio:
            return empty

        total_credit = round(call_credit + put_credit, 4)
        setup = LiveTradeSetup(
            strategy_type="meic",
            legs=[
                OrderLeg("", float(short_call), "call", "sell", qty),
                OrderLeg("", float(long_call), "call", "buy", qty),
                OrderLeg("", float(short_put), "put", "sell", qty),
                OrderLeg("", float(long_put), "put", "buy", qty),
            ],
            total_credit=total_credit,
            strikes={
                "short_call": float(short_call),
                "long_call": float(long_call),
                "short_put": float(short_put),
                "long_put": float(long_put),
            },
            extra={
                "em_pts": round(em, 2),
                "wing_pts": wing_pts,
                "call_credit": call_credit,
                "put_credit": put_credit,
                "credit_diff": round(abs(call_credit - put_credit), 4),
                "total_premium": total_credit,
                "short_call_mid": sc_mid,
                "short_put_mid": sp_mid,
                "call_credit_width": round(call_cw, 4),
                "put_credit_width": round(put_cw, 4),
                "qty": qty,
            },
        )
        return {"candidates_found": 1, "top_candidate": setup, "candidates": [setup]}

    async def validate(
        self, setup: LiveTradeSetup, chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        failures: List[str] = []

        def record(name: str, ok: bool):
            checks.append({"check": name, "passed": ok})
            if not ok:
                failures.append(name)

        st = setup.strikes
        sc, lc = st["short_call"], st["long_call"]
        sp, lp = st["short_put"], st["long_put"]

        record("strike_order_call", lc > sc)
        record("strike_order_put", sp > lp)
        record("wing_symmetry", abs((lc - sc) - (sp - lp)) <= self._strike_step)

        ccw = setup.extra.get("call_credit_width")
        pcw = setup.extra.get("put_credit_width")
        record("credit_to_width_call",
               ccw is not None and ccw >= self._min_credit_width_ratio)
        record("credit_to_width_put",
               pcw is not None and pcw >= self._min_credit_width_ratio)
        record("qty", setup.extra.get("qty", 0) >= 1)

        return {"passed": len(failures) == 0, "checks": checks, "failures": failures}

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        qty = int(setup.extra.get("qty", 1))
        st = setup.strikes
        stop = self.compute_stop_loss(setup)
        return {
            "order_type": "meic_two_spreads",
            "quantity": qty,
            "total_credit": setup.total_credit,
            "stop_amount": stop["stop_amount"],
            "put_spread": {
                "short_strike": st["short_put"],
                "long_strike": st["long_put"],
                "credit": setup.extra.get("put_credit"),
                "short_mid": setup.extra.get("short_put_mid"),
                "stop_trigger": stop["short_put_stop_price"],
                "quantity": qty,
            },
            "call_spread": {
                "short_strike": st["short_call"],
                "long_strike": st["long_call"],
                "credit": setup.extra.get("call_credit"),
                "short_mid": setup.extra.get("short_call_mid"),
                "stop_trigger": stop["short_call_stop_price"],
                "quantity": qty,
            },
        }

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        cc = setup.extra.get("call_credit", 0.0)
        pc = setup.extra.get("put_credit", 0.0)
        total_credit = round(cc + pc, 4)
        stop_amount = round(self._stop_pct * total_credit, 4)
        sc_mid = setup.extra.get("short_call_mid", 0.0)
        sp_mid = setup.extra.get("short_put_mid", 0.0)
        return {
            "total_credit": total_credit,
            "stop_pct": self._stop_pct,
            "stop_amount": stop_amount,
            "short_call_stop_price": round(sc_mid + stop_amount, 4),
            "short_put_stop_price": round(sp_mid + stop_amount, 4),
            "order_type": "stop_market",
        }

    def build_stop_orders(
        self, setup: LiveTradeSetup, stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "side": "short_call",
                "stop_market": {"trigger_price": stop_info["short_call_stop_price"]},
            },
            {
                "side": "short_put",
                "stop_market": {"trigger_price": stop_info["short_put_stop_price"]},
            },
        ]

    def _tiers_for(self, trade: Dict[str, Any]):
        entry = trade.get("entry_time_et")
        is_am = entry is not None and entry.time() < self._am_pm_cutoff
        if is_am:
            return self._am_tiers, self._am_fractions
        return self._pm_tiers, self._pm_fractions

    def check_exit(
        self, trade: Dict[str, Any], quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        trade_id = trade.get("trade_id", "")
        tiers, fractions = self._tiers_for(trade)

        legs_to_close: List[str] = []
        leg_qty: Dict[str, int] = {}
        reasons: List[str] = []
        stop_hit = False

        sides = [
            ("call", "short_call_symbol", "long_call_symbol",
             "call_credit", "call_open_qty", "short_call_stop_price"),
            ("put", "short_put_symbol", "long_put_symbol",
             "put_credit", "put_open_qty", "short_put_stop_price"),
        ]

        for side, short_key, long_key, credit_key, qty_key, stop_key in sides:
            short_sym = trade.get(short_key)
            long_sym = trade.get(long_key)
            if not short_sym:
                continue
            sq = quotes.get(short_sym)
            if sq is None:
                continue
            short_ask = sq.get("ask") or sq.get("price") or 0.0
            credit = trade.get(credit_key, 0.0)
            open_qty = int(trade.get(qty_key, 0))
            if short_ask <= 0 or credit <= 0 or open_qty <= 0:
                continue

            # --- Stop-loss check (short ask >= stop trigger) ---
            stop_price = trade.get(stop_key)
            if stop_price is not None and short_ask >= stop_price:
                stop_hit = True
                legs_to_close.append(short_sym)
                leg_qty[short_sym] = open_qty
                if long_sym:
                    legs_to_close.append(long_sym)
                    leg_qty[long_sym] = open_qty
                reasons.append(f"{side}_stop@{short_ask:.2f}")
                continue

            # --- Scale-out tiers ---
            pnl_pct = (credit - short_ask) / credit
            fired = self._scaleout_state.setdefault((trade_id, side), set())
            contracts = 0
            for i, tier in enumerate(tiers):
                if i in fired:
                    continue
                if pnl_pct >= tier:
                    fired.add(i)
                    contracts += int(math.floor(open_qty * fractions[i]))
            if contracts <= 0:
                continue
            contracts = min(contracts, open_qty)

            legs_to_close.append(short_sym)
            leg_qty[short_sym] = contracts
            reasons.append(f"{side}_scaleout x{contracts}")

            # Leftover long: close paired long unless bidless
            if long_sym:
                lq = quotes.get(long_sym)
                long_bid = lq.get("bid", 0.0) if lq else 0.0
                if long_bid >= self._abandon_long_below:
                    legs_to_close.append(long_sym)
                    leg_qty[long_sym] = contracts
                else:
                    reasons.append(f"{side}_abandon_long")

        if not legs_to_close:
            return ExitAction(action="hold")

        return ExitAction(
            action="stop_loss" if stop_hit else "take_profit",
            legs_to_close=legs_to_close,
            leg_quantities=leg_qty,
            reason="; ".join(reasons),
        )
