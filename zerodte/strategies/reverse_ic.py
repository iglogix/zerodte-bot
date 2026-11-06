"""
ReverseIcStrategy — SPX/XSP 0DTE "Reverse Iron Condor" (live).

Implements LiveStrategy for docs/0DTE-Reverse-Iron-Condor-Strategy.md on
ScheduledEngine with a single entry window near the open (config
entry.schedule, e.g. ["09:33"]):

  - entry only when SPX gaps up >= gap_min_pct vs the prior daily close
    (per-weekday overrides), the prior VIX close is >= vix_floor, and the
    day is not FOMC
  - structure: BUY the ~50-delta call with a short call +20 pts above, BUY
    the ~35-delta put with a short put -35 pts below (net debit, long gamma)
  - "sets up" filter: modeled upside profit (call_width - debit) must be
    >= setup_min_upside_pct of the debit, else skip (rejects pumped IV)
  - exit: profit target at debit x (1 + profit_target_pct), NO stop loss;
    optional scale-out tiers (fractions at +20% / +30%, runner to expiry).
    Position rides to cash settlement otherwise — set the monitor's
    eod_close_time to 15:59 (or later) so it is not force-closed early.

Backtest counterpart: zerodte/backtest/reverse_ic.py (validated 2022-2026;
the scale-out + vix_floor 15 variant is the production candidate).
Run with config/config_ric_live.yaml. NOTE: TastyTrade cert (sandbox)
cannot trade SPXW 0DTE — use manual mode for paper validation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from zerodte.agent.base_engine import BaseEngine, EngineDeps
from zerodte.backtest.engine import _FOMC_DATES
from zerodte.backtest.reverse_ic import bs_delta
from zerodte.strategies.base import (
    ExitAction,
    LiveStrategy,
    LiveTradeSetup,
    OrderLeg,
)
from zerodte.strategies.utils import mid_price as _mid

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

_WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class ReverseIcStrategy(LiveStrategy):

    def __init__(self):
        self._now_fn = lambda: datetime.now(ET)
        self._deps: Optional[EngineDeps] = None
        # trade_id -> set of tier indices already filled (scale-out state)
        self._tiers_filled: Dict[str, set] = {}

    # ------------------------------------------------------------------

    def configure(self, config: Dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self._gap_min = float(strat.get("gap_min_pct", 0.25))
        self._gap_min_wd = {
            str(k).lower()[:3]: float(v)
            for k, v in (strat.get("gap_min_pct_by_weekday") or {}).items()
        }
        vf = strat.get("vix_floor")
        self._vix_floor = float(vf) if vf is not None else None
        self._call_delta = float(strat.get("long_call_delta", 0.50))
        self._put_delta = float(strat.get("long_put_delta", 0.35))
        self._call_wing = float(strat.get("call_wing_offset_pts", 20))
        self._put_wing = float(strat.get("put_wing_offset_pts", 35))
        self._tp_pct = float(strat.get("profit_target_pct", 0.30))
        self._setup_min = float(strat.get("setup_min_upside_pct", 0.25))
        self._skip_fomc = bool(strat.get("skip_fomc_days", True))
        self._qty = int(strat.get("qty", 1))
        so = strat.get("scale_out") or {}
        self._scale_enabled = bool(so.get("enabled", False))
        self._scale_tiers = so.get("tiers", [
            {"profit": 0.20, "close_frac": 0.50},
            {"profit": 0.30, "close_frac": 0.25},
        ])
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
        self._deps = deps
        if deps.clock is not None:
            self._now_fn = lambda: deps.clock.now(ET)
        return ScheduledEngine(deps, self)

    # ------------------------------------------------------------------
    # Entry gates
    # ------------------------------------------------------------------

    async def _prior_daily_close(self) -> Optional[float]:
        """Prior session's SPX daily close via the API client."""
        api = getattr(self._deps.tool_handlers, "api_client", None) if self._deps else None
        if api is None:
            return None
        try:
            # from_time must be explicit: get_candles' default lookback only
            # understands minute intervals and would filter daily bars out.
            from datetime import timedelta
            candles = await api.get_candles(
                "SPX", interval="1d", count=5,
                from_time=datetime.now() - timedelta(days=7))
        except Exception:
            logger.exception("RIC: daily candle fetch failed")
            return None
        today = self._now_fn().date()

        def session_day(c) -> str:
            # dxfeed daily candles start at 00:00 UTC; the client's local-time
            # "timestamp" label lands on the prior ET evening. The raw epoch
            # interpreted as a UTC date IS the session date — use it so
            # today's live candle is never mistaken for the prior close.
            ts = c.get("_ts")
            if ts:
                from datetime import timezone
                return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
            return str(c.get("timestamp", ""))[:10]

        closes = [c for c in (candles or []) if session_day(c) < today.isoformat()]
        if not closes:
            return None
        return float(closes[-1].get("close") or 0) or None

    async def _vix_quote(self) -> Optional[float]:
        api = getattr(self._deps.tool_handlers, "api_client", None) if self._deps else None
        if api is None:
            return None
        try:
            q = await api.get_quote("VIX")
        except Exception:
            logger.exception("RIC: VIX quote fetch failed")
            return None
        if not isinstance(q, dict):
            return None
        for key in ("last", "price", "mark", "close"):
            v = q.get(key)
            if v:
                return float(v)
        bid, ask = q.get("bid") or 0, q.get("ask") or 0
        if bid > 0 and ask > 0:
            return (float(bid) + float(ask)) / 2
        return None

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
        if not chain or spot <= 0:
            return empty
        now = self._now_fn()

        if self._skip_fomc and now.date() in _FOMC_DATES:
            logger.info("RIC: FOMC day — skipping")
            return empty

        if self._vix_floor is not None:
            vix = await self._vix_quote()
            if vix is None or vix < self._vix_floor:
                logger.info("RIC: VIX %s below floor %.1f — skipping",
                            f"{vix:.1f}" if vix else "unavailable", self._vix_floor)
                return empty

        prior_close = await self._prior_daily_close()
        if not prior_close:
            logger.warning("RIC: no prior close available — skipping (fail closed)")
            return empty
        gap_pct = (spot / prior_close - 1) * 100
        gap_min = self._gap_min_wd.get(_WEEKDAY_KEYS[now.weekday()], self._gap_min)
        if gap_pct <= 0:
            logger.info("RIC: gap down %.2f%% — skipping", gap_pct)
            return empty
        if gap_pct < gap_min:
            logger.info("RIC: gap +%.2f%% below %.2f%% — skipping", gap_pct, gap_min)
            return empty

        # ---- Leg selection: ~50-delta long call, ~35-delta long put
        by: Dict[Tuple[float, str], dict] = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }
        calls = {k: o for (k, t), o in by.items() if t == "call" and _mid(o)}
        puts = {k: o for (k, t), o in by.items() if t == "put" and _mid(o)}
        if not calls or not puts:
            return empty

        minutes_left = max((16 - now.hour) * 60 - now.minute, 1)
        t_years = minutes_left / (60 * 24 * 365)

        def leg_delta(opt: dict, k: float, is_call: bool) -> Optional[float]:
            d = opt.get("delta")
            if d:
                return float(d)
            mid = _mid(opt)
            if mid is None or mid <= 0.05:
                return None
            return bs_delta(spot, k, t_years, mid, is_call)

        def pick(side: Dict[float, dict], is_call: bool, target: float) -> Optional[float]:
            best, best_err = None, None
            for k, o in side.items():
                if not is_call and k >= spot:
                    continue
                d = leg_delta(o, k, is_call)
                if d is None:
                    continue
                err = abs(abs(d) - target)
                if best_err is None or err < best_err:
                    best, best_err = k, err
            return best

        k_lc = pick(calls, True, self._call_delta)
        k_lp = pick(puts, False, self._put_delta)
        if k_lc is None or k_lp is None:
            return empty

        # Wings: fixed point offsets snapped to the nearest available strike
        # strictly beyond the long. Wings are SOLD, so they need a bid.
        call_ks = sorted(k for k in calls if k > k_lc and calls[k].get("bid", 0) > 0)
        put_ks = sorted(k for k in puts if k < k_lp and puts[k].get("bid", 0) > 0)
        if not call_ks or not put_ks:
            return empty
        k_sc = min(call_ks, key=lambda k: abs(k - (k_lc + self._call_wing)))
        k_sp = min(put_ks, key=lambda k: abs(k - (k_lp - self._put_wing)))

        legs = {
            "long_call": by.get((k_lc, "call")), "short_call": by.get((k_sc, "call")),
            "long_put": by.get((k_lp, "put")), "short_put": by.get((k_sp, "put")),
        }
        if any(o is None or _mid(o) is None for o in legs.values()):
            return empty
        if any(self._is_dislocated(o) for o in legs.values()):
            logger.info("RIC: dislocated leg quote — skipping")
            return empty

        debit = round((_mid(legs["long_call"]) - _mid(legs["short_call"]))
                      + (_mid(legs["long_put"]) - _mid(legs["short_put"])), 4)
        if debit <= 0:
            return empty

        # ---- "Sets up" filter: upside profit ramp must be worth having
        call_width = k_sc - k_lc
        upside = call_width - debit
        if upside < self._setup_min * debit:
            logger.info("RIC: setup filter — upside %.2f < %.0f%% of debit %.2f",
                        upside, self._setup_min * 100, debit)
            return empty

        qty = self._qty
        setup = LiveTradeSetup(
            strategy_type="reverse_ic",
            legs=[
                OrderLeg("", float(k_lc), "call", "buy", qty),
                OrderLeg("", float(k_sc), "call", "sell", qty),
                OrderLeg("", float(k_lp), "put", "buy", qty),
                OrderLeg("", float(k_sp), "put", "sell", qty),
            ],
            total_credit=-debit,     # negative = net debit
            strikes={
                "long_call": float(k_lc), "short_call": float(k_sc),
                "long_put": float(k_lp), "short_put": float(k_sp),
            },
            extra={
                "net_debit": debit,
                "gap_pct": round(gap_pct, 3),
                "call_width": call_width,
                "put_width": k_lp - k_sp,
                "upside_pct": round(upside / debit, 3),
                "tp_value": round(debit * (1.0 + self._tp_pct), 4),
                "max_loss_usd": round(debit * self._multiplier * qty, 2),
                "qty": qty,
                "scale_out": self._scale_enabled,
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
        debit = setup.extra.get("net_debit", 0)
        record("strike_order_call", st["short_call"] > st["long_call"])
        record("strike_order_put", st["short_put"] < st["long_put"])
        record("put_below_call", st["long_put"] < st["long_call"])
        record("positive_debit", debit > 0)
        record("debit_below_call_width",
               debit < setup.extra.get("call_width", 0))
        record("upside_sets_up",
               setup.extra.get("upside_pct", 0) >= self._setup_min)
        record("qty", setup.extra.get("qty", 0) >= 1)
        return {"passed": not failures, "checks": checks, "failures": failures}

    # ------------------------------------------------------------------

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        qty = int(setup.extra.get("qty", 1))
        st = setup.strikes
        debit = setup.extra.get("net_debit", 0.0)
        return {
            "order_type": "reverse_ic_debit",
            "quantity": qty,
            "net_debit": debit,
            "tp_value": setup.extra.get("tp_value"),
            "call_spread": {     # long debit spread
                "long_strike": st["long_call"], "short_strike": st["short_call"],
                "quantity": qty,
            },
            "put_spread": {      # long debit spread
                "long_strike": st["long_put"], "short_strike": st["short_put"],
                "quantity": qty,
            },
        }

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        """No stop — defined risk, max loss is the debit paid (size for it)."""
        return {
            "order_type": "none",
            "stop_amount": None,
            "max_loss_usd": setup.extra.get("max_loss_usd"),
        }

    def build_stop_orders(
        self, setup: LiveTradeSetup, stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------

    def check_exit(
        self, trade: Dict[str, Any], quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        trade_id = trade.get("trade_id", "")
        syms = [trade.get(k) for k in ("long_call_symbol", "short_call_symbol",
                                       "long_put_symbol", "short_put_symbol")]
        if not all(syms):
            return ExitAction(action="hold")
        qs = [quotes.get(s) for s in syms]
        if not all(qs):
            return ExitAction(action="hold")

        def quote_mid(q: Dict[str, Any]) -> Optional[float]:
            bid, ask = q.get("bid"), q.get("ask")
            if bid is None or ask is None or ask <= 0:
                return q.get("price")
            return (bid + ask) / 2

        mids = [quote_mid(q) for q in qs]
        if any(m is None for m in mids):
            return ExitAction(action="hold")
        lc, sc, lp, sp = mids
        value = max(0.0, (lc - sc) + (lp - sp))

        debit = trade.get("net_debit") or abs(trade.get("total_credit") or 0)
        if not debit:
            return ExitAction(action="hold")
        qty = int(trade.get("call_open_qty") or trade.get("put_open_qty")
                  or trade.get("quantity") or 1)
        profit_pct = value / debit - 1.0

        if trade.get("scale_out") or self._scale_enabled:
            filled = self._tiers_filled.setdefault(trade_id, set())
            base_qty = int(trade.get("quantity") or qty)
            for i, tier in enumerate(self._scale_tiers):
                if i in filled or profit_pct < float(tier["profit"]):
                    continue
                filled.add(i)
                n_close = min(qty, max(1, round(float(tier["close_frac"]) * base_qty)))
                if n_close <= 0:
                    continue
                return ExitAction(
                    action="take_profit",
                    legs_to_close=syms,
                    leg_quantities={s: n_close for s in syms},
                    reason=(f"scale_tier_{i + 1} value {value:.2f} "
                            f">= +{float(tier['profit']):.0%} of {debit:.2f}"),
                )
            return ExitAction(action="hold")   # the runner rides to settlement

        if profit_pct >= self._tp_pct:
            return ExitAction(
                action="take_profit",
                legs_to_close=syms,
                leg_quantities={s: qty for s in syms},
                reason=f"profit_target {value:.2f} >= {debit * (1 + self._tp_pct):.2f}",
            )

        # No stop loss — hold to target or cash settlement.
        return ExitAction(action="hold")
