"""
CBBO + Expected Move strategy — EM-based strike selection with real SPX quotes.

Strike placement uses expected move (vol-adaptive distance), not BS delta.
Pricing uses actual CBBO bid/ask instead of model prices.
Supports iron condor mode (both legs placed regardless of bias).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional
from datetime import datetime

from src_8am.backtest.strategy_base import Strategy, TradeSetup
from src_8am.backtest.strategy_em import EMStrategy
from src_8am.backtest.pricing import bs_put_delta, bs_call_delta


class CbboEMStrategy(EMStrategy):
    """EM-based strike selection with CBBO market pricing."""

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self._cbbo_store = None
        self._pricing_uncertainty = config.get("options", {}).get("pricing_uncertainty_per_leg", 0.0)
        # Asymmetric IC: per-side overrides (fall back to shared params)
        opt = config.get("options", {})
        self._em_mult_put = opt.get("em_multiplier_put", self._em_multiplier)
        self._em_mult_call = opt.get("em_multiplier_call", self._em_multiplier)
        self._width_put = opt.get("spread_width_put", self._spread_width)
        self._width_call = opt.get("spread_width_call", self._spread_width)
        # Bias-adaptive IC: with-trend leg closer (more premium), against-trend further (less risk)
        self._em_with_trend = opt.get("em_with_trend", 0.0)
        self._em_against_trend = opt.get("em_against_trend", 0.0)
        self._width_with_trend = opt.get("width_with_trend", 0)
        self._width_against_trend = opt.get("width_against_trend", 0)

    def set_cbbo_store(self, store) -> None:
        self._cbbo_store = store

    def _minute_key(self, dt: datetime) -> str:
        return dt.strftime("%H:%M")

    def select_trade(
        self,
        spot: float,
        bias: str,
        entry_time: datetime,
        session_iv: float,
        tte_years: float,
        tte_hours: float,
        engine: Any,
    ) -> Optional[TradeSetup]:
        spread_type = "put_credit_spread" if bias == "long" else "call_credit_spread"
        iv_adj = session_iv * self._iv_correction
        minute_et = self._minute_key(entry_time)

        # EM from ATM straddle (real 0DTE expected move), fallback to VIX-based
        em = None
        if self._cbbo_store is not None:
            em = self._cbbo_store.atm_straddle(minute_et, spot)
        if em is None or em <= 0:
            em = self.compute_em(spot, iv_adj, tte_hours)

        # Determine EM and width based on ORB breakout direction + spread side
        # In IC mode, bias is synthetic ("long"/"short" per leg) — use actual ORB bias
        orb_bias = getattr(engine, '_orb_bias', bias)
        # Long ORB breakout: market is bullish → put is with-trend, call is against-trend
        # Short ORB breakout: market is bearish → call is with-trend, put is against-trend
        is_with_trend = (
            (spread_type == "put_credit_spread" and orb_bias == "long") or
            (spread_type == "call_credit_spread" and orb_bias == "short")
        )

        if self._em_with_trend > 0 and self._em_against_trend > 0:
            # Bias-adaptive mode
            em_mult = self._em_with_trend if is_with_trend else self._em_against_trend
            width = (self._width_with_trend if is_with_trend else self._width_against_trend) or self._spread_width
        elif spread_type == "put_credit_spread":
            em_mult = self._em_mult_put
            width = self._width_put
        else:
            em_mult = self._em_mult_call
            width = self._width_call

        if spread_type == "put_credit_spread":
            raw_short = spot - em_mult * em
            short_k = math.floor(raw_short / self._strike_step) * self._strike_step
            long_k = short_k - width
        else:
            raw_short = spot + em_mult * em
            short_k = math.ceil(raw_short / self._strike_step) * self._strike_step
            long_k = short_k + width

        is_call = spread_type == "call_credit_spread"
        minute_et = self._minute_key(entry_time)

        # CBBO pricing
        credit = None
        if self._cbbo_store is not None:
            short_q = self._cbbo_store.lookup(minute_et, short_k, is_call)
            long_q = self._cbbo_store.lookup(minute_et, long_k, is_call)
            if short_q is not None and long_q is not None:
                short_mid = (short_q[0] + short_q[1]) / 2
                long_mid = (long_q[0] + long_q[1]) / 2
                credit = round(short_mid - long_mid, 2)

        # Fallback to BS
        if credit is None:
            is_call_flag = 0 if spread_type == "put_credit_spread" else 1
            from src_8am.backtest.pricing import spread_credit
            iv_short = engine._get_strike_iv(iv_adj, spot, short_k, tte_years, is_call_flag)
            credit = spread_credit(spot, short_k, long_k, tte_years, iv_short,
                                   spread_type, model=self._pricing_model)

        credit = round(credit - self._slippage, 2)

        if credit < self._min_credit:
            return None

        # Delta for reporting only
        short_delta = abs(bs_put_delta(spot, short_k, tte_years, iv_adj)) if not is_call \
            else bs_call_delta(spot, short_k, tte_years, iv_adj)

        return TradeSetup(
            spread_type=spread_type,
            short_strike=short_k,
            long_strike=long_k,
            spread_width=width,
            credit=credit,
            short_delta=round(short_delta, 4),
            iv_used=session_iv,
            extra={"pricing": "cbbo" if self._cbbo_store else "bs",
                   "em": round(em, 2), "em_mult": em_mult},
        )

    def compute_exit_value(
        self,
        spot: float,
        short_k: float,
        long_k: float,
        tte_years: float,
        tte_hours: float,
        spread_type: str,
        session_iv: float,
        engine: Any,
    ) -> float:
        """Mark-to-market using CBBO mid prices, fallback to BS."""
        is_call = spread_type == "call_credit_spread"

        if self._cbbo_store is not None:
            minute_et = getattr(engine, '_last_bar_time', None)
            if minute_et is not None:
                short_q = self._cbbo_store.lookup(minute_et, short_k, is_call)
                long_q = self._cbbo_store.lookup(minute_et, long_k, is_call)
                if short_q is not None and long_q is not None:
                    short_mid = (short_q[0] + short_q[1]) / 2
                    long_mid = (long_q[0] + long_q[1]) / 2
                    debit = short_mid - long_mid
                    return max(0.0, round(debit, 4))

        # Fallback to BS
        from src_8am.backtest.pricing import spread_value
        iv = engine._get_strike_iv(session_iv, spot, short_k, tte_years,
                                    0 if not is_call else 1)
        return spread_value(spot, short_k, long_k, tte_years, iv,
                           spread_type, model=self._pricing_model)

    def grid_params(self) -> Dict[str, List[Any]]:
        return {
            "em_with_trend":         [0.5, 0.625, 0.75, 1.0],
            "em_against_trend":      [0.75, 1.0, 1.25, 1.5],
            "width_with_trend":      [5, 10],
            "width_against_trend":   [5, 10],
            "tp_pct_of_credit":      [0.25, 0.50, 0.75],
            "min_orb_range_pts":     [2.0, 3.0],
            "midline_tolerance":     [0.25, 0.50],
            "hold_put_breach":       [True],
        }

    def apply_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(config)
        cfg.setdefault("options", {})
        cfg.setdefault("strategy", {})
        cfg["strategy"].setdefault("orb_range_filter", {})
        # Bias-adaptive params
        if "em_with_trend" in params:
            cfg["options"]["em_with_trend"] = params["em_with_trend"]
            cfg["options"]["em_against_trend"] = params["em_against_trend"]
            cfg["options"]["width_with_trend"] = params["width_with_trend"]
            cfg["options"]["width_against_trend"] = params["width_against_trend"]
            cfg["options"]["em_multiplier"] = params["em_with_trend"]
            cfg["options"]["spread_width_pts"] = params["width_with_trend"]
        # Legacy per-side params
        if "em_multiplier_put" in params:
            cfg["options"]["em_multiplier_put"] = params["em_multiplier_put"]
            cfg["options"]["em_multiplier_call"] = params["em_multiplier_call"]
            cfg["options"]["em_multiplier"] = params["em_multiplier_put"]
            cfg["options"]["spread_width_pts"] = params.get("spread_width_put", 5)
        cfg["options"]["tp_pct_of_credit"] = params["tp_pct_of_credit"]
        cfg["strategy"]["midline_tolerance_pts"] = params["midline_tolerance"]
        cfg["strategy"]["orb_range_filter"]["min_pts"] = params["min_orb_range_pts"]
        if "hold_put_breach" in params:
            cfg["options"]["hold_put_breach"] = params["hold_put_breach"]
        if "quantity" in params:
            cfg["options"]["quantity"] = params["quantity"]
            mult = cfg.get("options", {}).get("contract_multiplier", 100)
            width = max(params.get("width_with_trend", 5), params.get("width_against_trend", 5))
            cfg.setdefault("risk", {})["max_loss_per_trade_usd"] = width * mult * params["quantity"]
        cfg["options"]["pricing_model"] = "bs"
        return cfg
