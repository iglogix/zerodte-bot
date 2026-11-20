"""
PowerHourStrategy — SPX 0DTE "Power Hour" staggered iron condors (live).

Implements LiveStrategy for the strategy in docs/0DTE-Power-Hour-Strategy.md
(Version A), running on ScheduledEngine with entries at 14:55 / 15:15 / 15:25 ET:

  - short strikes by FIXED CREDIT per side (closest mid to the tranche target:
    $4 / $3 / $2), never inverted — the floor is the ATM straddle
  - long wings = nearest strike beyond the short with ask <= wing_target (~$0.05);
    insurance only, abandoned after entry
  - stop on the COMBINED short strangle at entry credit + stop dollars
    ($4 / $3 / $3), evaluated each PositionMonitor poll on live quotes
  - one-time ratchet: at 40% profit, stop moves to price-at-trigger + add
    ($3 / $2 / none)
  - profit target 90% of credit; PositionMonitor's eod_close_time (15:55)
    handles the forced close

Run with config/config_power_hour_live.yaml. NOTE: TastyTrade cert (sandbox)
cannot trade SPXW 0DTE — use manual mode for paper validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from zerodte.agent.base_engine import BaseEngine, EngineDeps
from zerodte.strategies.base import (
    ExitAction,
    LiveStrategy,
    LiveTradeSetup,
    OrderLeg,
)
from zerodte.strategies.utils import mid_price as _mid

ET = ZoneInfo("America/New_York")


class PowerHourStrategy(LiveStrategy):

    def __init__(self):
        self._now_fn = lambda: datetime.now(ET)
        # trade_id -> ratcheted stop level (combined strangle price), once set
        self._ratchet_state: Dict[str, float] = {}
        # trade_ids whose exit (stop/TP) was already emitted — the wings stay
        # on the trade row until expiry, so without this the monitor would
        # re-fire the same exit every poll
        self._exited: set = set()

    # ------------------------------------------------------------------

    def configure(self, config: Dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self._tranches: Dict[str, dict] = {
            t["entry"]: t for t in strat.get("tranches", [])
        }
        self._ratchet_mode = strat.get("ratchet_mode", "trigger_plus")
        self._ratchet_trigger = float(strat.get("ratchet_trigger_profit_pct", 0.40))
        self._tp_pct = float(strat.get("profit_target_pct", 0.90))
        self._wing_target = float(strat.get("wing_target_price", 0.05))
        self._disloc_min_spread = float(strat.get("dislocation_min_spread_pts", 1.0))
        self._disloc_mult = float(strat.get("dislocation_spread_mult", 1.5))
        self._multiplier = config.get("options", {}).get("contract_multiplier", 100)

    # ------------------------------------------------------------------

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    def compute_trailing_stop(self, trade, quotes) -> Optional[float]:
        return None

    def create_engine(self, deps: "EngineDeps") -> "BaseEngine":
        from zerodte.agent.scheduled_engine import ScheduledEngine
        if deps.clock is not None:
            self._now_fn = lambda: deps.clock.now(ET)
        return ScheduledEngine(deps, self)

    # ------------------------------------------------------------------

    def _current_tranche(self) -> Optional[dict]:
        slot = self._now_fn().strftime("%H:%M")
        if slot in self._tranches:
            return self._tranches[slot]
        # tolerate the engine firing a minute late
        h, m = int(slot[:2]), int(slot[3:5])
        prev = f"{h:02d}:{m - 1:02d}" if m else f"{h - 1:02d}:59"
        return self._tranches.get(prev)

    def _is_dislocated(self, opt: Optional[Dict[str, Any]]) -> bool:
        if not opt:
            return False
        bid, ask = opt.get("bid", 0), opt.get("ask", 0)
        if bid <= 0 or ask <= 0:
            return False
        spread = ask - bid
        return (spread > self._disloc_min_spread
                and spread > self._disloc_mult * ((bid + ask) / 2))

    # ------------------------------------------------------------------

    async def search_candidates(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Dict[str, Any]:
        empty = {"candidates_found": 0, "top_candidate": None, "candidates": []}
        tranche = self._current_tranche()
        if not chain or tranche is None:
            return empty
        target = float(tranche["target_credit"])
        qty = int(tranche.get("qty", 1))

        by: Dict[Tuple[float, str], dict] = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }
        # Shorts need two-sided quotes (mid); wings only need an ask — they are
        # usually zero-bid lottery tickets, which mid_price() rejects by design.
        calls = {k: o for (k, t), o in by.items() if t == "call" and _mid(o)}
        puts = {k: o for (k, t), o in by.items() if t == "put" and _mid(o)}
        calls_ask = {k: o for (k, t), o in by.items()
                     if t == "call" and (o.get("ask") or 0) > 0}
        puts_ask = {k: o for (k, t), o in by.items()
                    if t == "put" and (o.get("ask") or 0) > 0}
        if not calls or not puts:
            return empty

        def closest(side: Dict[float, dict], prefer_high: bool) -> float:
            return min(side, key=lambda k: (abs(_mid(side[k]) - target),
                                            -k if prefer_high else k))

        k_call = closest(calls, True)
        k_put = closest(puts, False)

        # Never invert — floor at the ATM straddle.
        if k_put > k_call:
            strikes = sorted({k for k, _ in by})
            atm = min(strikes, key=lambda k: abs(k - spot))
            if (atm, "call") not in by or (atm, "put") not in by:
                return empty
            k_put = k_call = atm

        sc, sp = by.get((k_call, "call")), by.get((k_put, "put"))
        if sc is None or sp is None:
            return empty
        if self._is_dislocated(sc) or self._is_dislocated(sp):
            return empty

        # Wings: nearest strike beyond the short with ask <= wing_target.
        def pick_wing(side: Dict[float, dict], is_call: bool) -> Optional[float]:
            beyond = {k: o for k, o in side.items()
                      if (k > k_call if is_call else k < k_put)}
            cheap = [k for k in beyond
                     if beyond[k]["ask"] <= self._wing_target + 1e-9]
            if cheap:
                return min(cheap) if is_call else max(cheap)
            if not beyond:
                return None
            return min(beyond, key=lambda k: (beyond[k]["ask"],
                                              -k if is_call else k))

        k_wc = pick_wing(calls_ask, True)
        k_wp = pick_wing(puts_ask, False)
        if k_wc is None or k_wp is None:
            return empty

        call_mid = _mid(sc)
        put_mid = _mid(sp)
        wc_ask = by[(k_wc, "call")].get("ask") or 0.0
        wp_ask = by[(k_wp, "put")].get("ask") or 0.0
        strangle_credit = round(call_mid + put_mid, 4)
        total_credit = round(strangle_credit - wc_ask - wp_ask, 4)
        if total_credit <= 0:
            return empty

        setup = LiveTradeSetup(
            strategy_type="power_hour",
            legs=[
                OrderLeg("", float(k_call), "call", "sell", qty),
                OrderLeg("", float(k_wc), "call", "buy", qty),
                OrderLeg("", float(k_put), "put", "sell", qty),
                OrderLeg("", float(k_wp), "put", "buy", qty),
            ],
            total_credit=total_credit,
            strikes={
                "short_call": float(k_call), "long_call": float(k_wc),
                "short_put": float(k_put), "long_put": float(k_wp),
            },
            extra={
                "tranche": tranche.get("name"),
                "spot": spot,
                "target_credit": target,
                "call_credit": round(call_mid - wc_ask, 4),
                "put_credit": round(put_mid - wp_ask, 4),
                "strangle_credit": strangle_credit,
                "short_call_mid": call_mid,
                "short_put_mid": put_mid,
                "wing_cost": round(wc_ask + wp_ask, 4),
                "stop_dollars": float(tranche["stop"]),
                "ratchet_add": tranche.get("ratchet_add"),
                "qty": qty,
            },
        )
        return {"candidates_found": 1, "top_candidate": setup, "candidates": [setup]}

    # ------------------------------------------------------------------

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
        record("strike_order_call", st["long_call"] > st["short_call"])
        record("strike_order_put", st["short_put"] > st["long_put"])
        record("not_inverted", st["short_put"] <= st["short_call"])
        record("positive_credit", setup.total_credit > 0)
        record("qty", setup.extra.get("qty", 0) >= 1)
        # credits should be in the neighbourhood of the tranche target (sanity,
        # wide tolerance — low IV legitimately floors at the ATM straddle)
        tgt = setup.extra.get("target_credit", 0)
        sc_mid = setup.extra.get("short_call_mid") or 0
        sp_mid = setup.extra.get("short_put_mid") or 0
        record("credit_sane", 0 < sc_mid + sp_mid < 6 * max(tgt, 1.0))
        return {"passed": not failures, "checks": checks, "failures": failures}

    # ------------------------------------------------------------------

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        qty = int(setup.extra.get("qty", 1))
        st = setup.strikes
        stop = self.compute_stop_loss(setup)
        return {
            "order_type": "power_hour_ic",
            "quantity": qty,
            "total_credit": setup.total_credit,
            "stop_amount": stop["stop_amount"],
            "tranche": setup.extra.get("tranche"),
            "underlying_at_entry": setup.extra.get("spot"),
            "strangle_credit": setup.extra.get("strangle_credit"),
            "ratchet_add": setup.extra.get("ratchet_add"),
            "wings_first": True,   # buy wings, then sell shorts with stop attached
            "put_spread": {
                "short_strike": st["short_put"], "long_strike": st["long_put"],
                "credit": setup.extra.get("put_credit"),
                "short_mid": setup.extra.get("short_put_mid"),
                "quantity": qty,
            },
            "call_spread": {
                "short_strike": st["short_call"], "long_strike": st["long_call"],
                "credit": setup.extra.get("call_credit"),
                "short_mid": setup.extra.get("short_call_mid"),
                "quantity": qty,
            },
            "strangle_stop_price": stop["strangle_stop_price"],
        }

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        """Stop is on the COMBINED short strangle price (entry + stop dollars)."""
        strangle = setup.extra.get("strangle_credit", 0.0)
        stop_dollars = setup.extra.get("stop_dollars", 0.0)
        return {
            "total_credit": strangle,
            "stop_amount": stop_dollars,
            "strangle_stop_price": round(strangle + stop_dollars, 4),
            # per-side backstops for broker-side stop-markets (split the dollar
            # stop across sides; the real trigger is the combined check_exit)
            "short_call_stop_price": round(
                (setup.extra.get("short_call_mid") or 0) + stop_dollars, 4),
            "short_put_stop_price": round(
                (setup.extra.get("short_put_mid") or 0) + stop_dollars, 4),
            "order_type": "monitor_combined",
        }

    def build_stop_orders(
        self, setup: LiveTradeSetup, stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Broker-side per-leg backstops; primary management is check_exit's
        # combined-strangle logic on each monitor poll.
        return [
            {"side": "short_call",
             "stop_market": {"trigger_price": stop_info["short_call_stop_price"]}},
            {"side": "short_put",
             "stop_market": {"trigger_price": stop_info["short_put_stop_price"]}},
        ]

    # ------------------------------------------------------------------

    def check_exit(
        self, trade: Dict[str, Any], quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        trade_id = trade.get("trade_id", "")
        if trade_id in self._exited:
            return ExitAction(action="hold")
        short_call = trade.get("short_call_symbol")
        short_put = trade.get("short_put_symbol")
        if not short_call or not short_put:
            return ExitAction(action="hold")
        cq, pq = quotes.get(short_call), quotes.get(short_put)
        if not cq or not pq:
            return ExitAction(action="hold")

        def quote_mid(q: Dict[str, Any]) -> Optional[float]:
            bid, ask = q.get("bid"), q.get("ask")
            if bid is None or ask is None or ask <= 0:
                return q.get("price")
            return (bid + ask) / 2

        cm, pm = quote_mid(cq), quote_mid(pq)
        if cm is None or pm is None:
            return ExitAction(action="hold")
        strangle = cm + pm

        entry = (trade.get("call_credit", 0.0) or 0.0) + (trade.get("put_credit", 0.0) or 0.0)
        # call/put_credit on the trade are net of wings in some flows; prefer the
        # stored strangle credit when present.
        entry = trade.get("strangle_credit") or entry
        stop_dollars = trade.get("stop_dollars")
        ratchet_add = trade.get("ratchet_add")
        if entry <= 0 or stop_dollars is None:
            return ExitAction(action="hold")

        stop_level = self._ratchet_state.get(trade_id, entry + float(stop_dollars))
        qty = int(trade.get("call_open_qty") or trade.get("put_open_qty") or 1)
        all_legs = [s for s in (
            short_call, short_put,
        ) if s]
        leg_qty = {s: qty for s in all_legs}

        # 1) combined-strangle stop
        if strangle >= stop_level:
            self._exited.add(trade_id)
            return ExitAction(
                action="stop_loss", legs_to_close=all_legs, leg_quantities=leg_qty,
                reason=f"strangle_stop {strangle:.2f} >= {stop_level:.2f}",
            )

        # 2) 90% profit target
        if strangle <= entry * (1.0 - self._tp_pct):
            self._exited.add(trade_id)
            return ExitAction(
                action="take_profit", legs_to_close=all_legs, leg_quantities=leg_qty,
                reason=f"profit_target {strangle:.2f} <= {entry * (1 - self._tp_pct):.2f}",
            )

        # 3) one-time ratchet at 40% profit
        if (trade_id not in self._ratchet_state and ratchet_add is not None
                and strangle <= entry * (1.0 - self._ratchet_trigger)):
            if self._ratchet_mode == "entry_plus":
                new_stop = entry + float(ratchet_add)
            else:
                new_stop = strangle + float(ratchet_add)
            self._ratchet_state[trade_id] = round(new_stop, 4)

        return ExitAction(action="hold")
