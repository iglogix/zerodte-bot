"""
OpenStrangleStrategy — 0DTE short strangle entered at a fixed time near the open.

Forked from OpenIcStrategy. Differences:
  - 2 legs (short put + short call); no protective wings
  - total_credit = put_mid + call_mid - 2*slippage (gross premium)
  - sl_dollars (absolute $-SL) replaces sl_mult_of_credit. Set per-trade from
    a pre-extracted IC SL-by-day artifact: sl_dollars[d] = min(IC_d, IC_p75).
  - No breach handling (no defined wings to breach)
  - Designed for eod_force_close earlier than 15:59 to avoid assignment risk
"""

from __future__ import annotations

import csv
import math
from datetime import date as date_cls
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zerodte.agent.base_engine import BaseEngine, EngineDeps
from zerodte.strategies.base import (
    ExitAction,
    LiveStrategy,
    LiveTradeSetup,
    OrderLeg,
)
from zerodte.strategies.utils import mid_price as _mid


class OpenStrangleStrategy(LiveStrategy):

    def configure(self, config: Dict[str, Any]) -> None:
        opt = config.get("options", {})
        self._strike_step = opt.get("strike_step_pts", 5.0)
        # "delta" mode is supported for parity with OpenIcStrategy but is
        # unreliable at 0DTE. Default is em_mult.
        self._strike_mode = opt.get("strike_mode", "em_mult")
        self._target_short_delta = opt.get("target_short_delta", 0.15)
        self._em_mult = opt.get("em_mult", 0.30)
        self._em_mult_put = opt.get("em_mult_put", None)
        self._em_mult_call = opt.get("em_mult_call", None)
        self._min_credit = opt.get("min_credit_per_spread", 0.05)
        self._max_credit_imbalance = opt.get("max_credit_imbalance", 1.00)
        self._tp_pct = opt.get("tp_pct_of_credit", 0.50)
        self._tp_pct_end = opt.get("tp_pct_of_credit_end", None)
        # tp_mode: "total" closes both sides when (put+call mark) <= tp_pct*total;
        #          "per_side" closes one side when its mark <= tp_pct*side_credit.
        self._tp_mode = opt.get("tp_mode", "total")
        self._slippage = opt.get("slippage_per_spread", 0.10)
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)

        # SL artifact: per-day $-SL from the matched IC. Set None to disable.
        self._sl_csv_path = opt.get("sl_dollars_csv", None)
        # Fallback $-SL when a date isn't in the artifact (also serves as the
        # p75 ceiling for matched-day values). None means no SL configured.
        fallback_val = opt.get("sl_dollars_fallback")
        self._sl_fallback: Optional[float] = (
            float(fallback_val) if fallback_val is not None else None
        )
        self._sl_by_date: Dict[date_cls, float] = {}
        self._sl_p75: float = self._sl_fallback if self._sl_fallback is not None else 0.0
        if self._sl_csv_path:
            self._load_sl_csv(self._sl_csv_path)

        market_close = config.get("market", {}).get("market_close", "16:00")
        h, m = [int(x) for x in market_close.split(":")]
        self._market_close = time(h, m)

        eod = config.get("strategy", {}).get("eod_force_close", "15:30")
        eh, em_ = [int(x) for x in eod.split(":")]
        self._eod_time = time(eh, em_)

        self._now_dt: Optional[datetime] = None
        self._last_em_pts: Optional[float] = None

    # ------------------------------------------------------------------
    # SL artifact loader
    # ------------------------------------------------------------------

    def _load_sl_csv(self, path: str) -> None:
        """Load IC per-day $-SL from a CSV produced by
        scripts/extract_ic_sl_dollars.py. Expected columns:
            trade_date (YYYY-MM-DD), ic_sl_dollars.
        Also reads <path>.meta.json for the p75 ceiling.
        """
        import json

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"sl_dollars_csv not found: {p}")
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = date_cls.fromisoformat(row["trade_date"])
                self._sl_by_date[d] = float(row["ic_sl_dollars"])
        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._sl_p75 = float(meta.get("p75", self._sl_fallback))
        elif self._sl_by_date:
            # Compute p75 from data if meta missing.
            import numpy as np

            self._sl_p75 = float(np.percentile(list(self._sl_by_date.values()), 75))

    def _sl_dollars_for(self, when: Optional[datetime]) -> float:
        """Return the per-trade $-SL for the given date.

        Resolution rules:
          - If artifact loaded and date present: min(per_day, p75).
          - If artifact loaded but date missing: p75 (ceiling fallback).
          - If artifact NOT loaded and `sl_dollars_fallback` set: fallback value.
          - If neither artifact nor fallback: 0.0, meaning NO SL is configured
            and check_exit must treat this as "do not fire SL".

        The caller (check_exit) is expected to gate SL firing on `sl_dollars > 0`.
        """
        if not self._sl_by_date:
            return self._sl_fallback if self._sl_fallback is not None else 0.0
        if when is None:
            return self._sl_p75
        d = when.date() if isinstance(when, datetime) else when
        per_day = self._sl_by_date.get(d)
        if per_day is None:
            return self._sl_p75
        return min(per_day, self._sl_p75)

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    def create_engine(self, deps: EngineDeps) -> BaseEngine:
        from zerodte.agent.window_entry_engine import WindowEntryEngine
        from zerodte.backtest.sim_clock import WallClock

        self._clock_ref = deps.clock or WallClock()
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

        chain_by = self._chain_map(chain)
        em = self._compute_em_from_straddle(chain_by, spot)
        if em is None or em <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}
        self._last_em_pts = em

        put_mult = None
        call_mult = None
        if self._strike_mode == "em_mult":
            put_mult = self._em_mult_put if self._em_mult_put is not None else self._em_mult
            call_mult = self._em_mult_call if self._em_mult_call is not None else self._em_mult
            short_call_k = (
                math.ceil((spot + call_mult * em) / self._strike_step) * self._strike_step
            )
            short_put_k = math.floor((spot - put_mult * em) / self._strike_step) * self._strike_step
        else:
            # Delta mode (BS inversion — kept for parity; not recommended at 0DTE)
            t = self._time_to_expiration_years()
            if t <= 0:
                return {"candidates_found": 0, "top_candidate": None, "candidates": []}
            iv = self._derive_iv_from_atm(chain_by, spot, t)
            if iv is None:
                return {"candidates_found": 0, "top_candidate": None, "candidates": []}
            short_call_k = self._find_strike_for_delta(
                chain_by,
                spot,
                self._target_short_delta,
                "call",
                t,
                iv,
            )
            short_put_k = self._find_strike_for_delta(
                chain_by,
                spot,
                self._target_short_delta,
                "put",
                t,
                iv,
            )

        if short_call_k is None or short_put_k is None:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        sc = chain_by.get((short_call_k, "call"))
        sp = chain_by.get((short_put_k, "put"))
        if not sc or not sp:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        sc_mid = _mid(sc)
        sp_mid = _mid(sp)
        if sc_mid is None or sp_mid is None:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        call_credit = round(sc_mid - self._slippage, 2)
        put_credit = round(sp_mid - self._slippage, 2)

        if call_credit < self._min_credit or put_credit < self._min_credit:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}
        if abs(call_credit - put_credit) > self._max_credit_imbalance:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        total_credit = round(call_credit + put_credit, 2)

        # Resolve today's sl_dollars from artifact (used by check_exit).
        sl_dollars = self._sl_dollars_for(self._now_dt)

        setup = LiveTradeSetup(
            strategy_type="open_strangle",
            legs=[
                OrderLeg("", short_put_k, "put", "sell", self._quantity),
                OrderLeg("", short_call_k, "call", "sell", self._quantity),
            ],
            total_credit=total_credit,
            strikes={
                "short_put": short_put_k,
                "short_call": short_call_k,
            },
            extra={
                "put_credit": put_credit,
                "call_credit": call_credit,
                "em_pts": round(em, 2),
                "strike_mode": self._strike_mode,
                "em_mult_used": self._em_mult,
                "em_mult_put_used": (put_mult if self._strike_mode == "em_mult" else None),
                "em_mult_call_used": (call_mult if self._strike_mode == "em_mult" else None),
                "sl_dollars": sl_dollars,
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
        # search_candidates already enforces min_credit and credit_imbalance.
        # Live-data validation happens at order-build time downstream.
        return {"passed": True, "checks": [], "failures": []}

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        s = setup.strikes
        # sl_dollars is resolved at setup-time from the IC artifact; pass it
        # through so check_exit can read it off the trade dict.
        return {
            "order_type": "strangle",
            "short_put_strike": s["short_put"],
            "short_call_strike": s["short_call"],
            "quantity": self._quantity,
            "sl_dollars": setup.extra.get("sl_dollars", 0.0),
        }

    # ------------------------------------------------------------------
    # Stop-loss (placeholder — real SL fires in check_exit)
    # ------------------------------------------------------------------

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        return {
            "mode": "sl_dollars",
            "sl_dollars": setup.extra.get("sl_dollars", 0.0),
        }

    def build_stop_orders(self, setup, stop_info) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    # Exit — implemented in Task 5
    # ------------------------------------------------------------------

    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        tp_pct = self._current_tp_pct(trade)

        put_sym = trade.get("short_put_symbol")
        call_sym = trade.get("short_call_symbol")

        put_mark = _mid(quotes.get(put_sym)) if put_sym else None
        call_mark = _mid(quotes.get(call_sym)) if call_sym else None

        legs_all = [s for s in (put_sym, call_sym) if s]
        total_credit = trade.get("put_credit", 0) + trade.get("call_credit", 0)

        # Total-mode TP/SL.
        if self._tp_mode == "total" and put_mark is not None and call_mark is not None:
            total_mark = put_mark + call_mark
            if total_credit > 0:
                tp_debit = total_credit * tp_pct
                if total_mark <= tp_debit:
                    return ExitAction(
                        action="take_profit",
                        legs_to_close=legs_all,
                        reason=f"total TP: mark={total_mark:.2f} <= tp={tp_debit:.2f}",
                    )
                sl_dollars_raw = trade.get("sl_dollars")
                if sl_dollars_raw is None:
                    sl_dollars_raw = self._sl_fallback
                sl_dollars = float(sl_dollars_raw) if sl_dollars_raw is not None else 0.0
                if sl_dollars > 0:
                    # Per-contract loss in dollars: (mark - credit) × multiplier.
                    loss_dollars = (total_mark - total_credit) * self._multiplier
                    if loss_dollars >= sl_dollars:
                        return ExitAction(
                            action="stop_loss",
                            legs_to_close=legs_all,
                            reason=(
                                f"sl_dollars: loss=${loss_dollars:.0f} " f">= cap=${sl_dollars:.0f}"
                            ),
                        )

        # Per-side TP (does not fire in tp_mode=total).
        if self._tp_mode == "per_side":
            for side, sym, credit_key in (
                ("put", put_sym, "put_credit"),
                ("call", call_sym, "call_credit"),
            ):
                if not sym:
                    continue
                m = _mid(quotes.get(sym))
                if m is None:
                    continue
                credit = trade.get(credit_key, 0)
                if credit > 0 and m <= credit * tp_pct:
                    return ExitAction(
                        action="take_profit",
                        legs_to_close=[sym],
                        reason=f"{side} TP: mark={m:.2f} <= tp={credit * tp_pct:.2f}",
                    )

        return ExitAction(action="hold")

    def _current_tp_pct(self, trade: Dict[str, Any]) -> float:
        """Time-ramp tp_pct from initial → end over the trade's life."""
        if self._tp_pct_end is None or self._tp_pct_end == self._tp_pct:
            return self._tp_pct
        clock = getattr(self, "_clock_ref", None)
        entry_time = trade.get("entry_time")
        if clock is None or entry_time is None:
            return self._tp_pct
        from zoneinfo import ZoneInfo

        et_tz = ZoneInfo("America/New_York")
        now = clock.now(et_tz)
        eod_dt = datetime(
            now.year,
            now.month,
            now.day,
            self._eod_time.hour,
            self._eod_time.minute,
            tzinfo=now.tzinfo,
        )
        if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_dt = entry_time.replace(tzinfo=now.tzinfo)
        else:
            entry_dt = entry_time
        total = (eod_dt - entry_dt).total_seconds()
        elapsed = (now - entry_dt).total_seconds()
        if total <= 0:
            return self._tp_pct_end
        f = max(0.0, min(1.0, elapsed / total))
        return self._tp_pct + (self._tp_pct_end - self._tp_pct) * f

    def compute_trailing_stop(self, trade, quotes) -> Optional[float]:
        return None

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        # Strangle has no long legs; PositionMonitor skips when long is None.
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    # ------------------------------------------------------------------
    # Internal helpers (copied from OpenIcStrategy for self-containment)
    # ------------------------------------------------------------------

    def _time_to_expiration_years(self) -> float:
        now = self._now_dt
        if now is None:
            return 0.0
        close_dt = datetime(
            now.year,
            now.month,
            now.day,
            self._market_close.hour,
            self._market_close.minute,
            tzinfo=now.tzinfo,
        )
        seconds = max(0.0, (close_dt - now).total_seconds())
        return seconds / (365.0 * 24.0 * 3600.0)

    def _compute_em_from_straddle(
        self,
        chain_by: Dict,
        spot: float,
    ) -> Optional[float]:
        step = self._strike_step
        atm_k = round(spot / step) * step
        call = chain_by.get((atm_k, "call"))
        put = chain_by.get((atm_k, "put"))
        if not call or not put:
            strikes = sorted({k for (k, _) in chain_by.keys()})
            if not strikes:
                return None
            atm_k = min(strikes, key=lambda k: abs(k - spot))
            call = chain_by.get((atm_k, "call"))
            put = chain_by.get((atm_k, "put"))
        if not call or not put:
            return None
        c_mid = _mid(call)
        p_mid = _mid(put)
        if c_mid is None or p_mid is None:
            return None
        return c_mid + p_mid

    def _derive_iv_from_atm(self, chain_by, spot, t):
        from es_8am.backtest.pricing import bs_call, bs_put

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

    def _find_strike_for_delta(self, chain_by, spot, target_delta, side, t, iv):
        from es_8am.backtest.pricing import bs_call_delta, bs_put_delta

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
        return {(float(o.get("strike", 0)), o.get("option_type", "").lower()): o for o in chain}
