"""
Delta-based strike selection strategy (original/production strategy).

Selects short strike at target BS delta, prices via BS/Bachelier or ANN.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from datetime import datetime

from es_8am.backtest.strategy_base import Strategy, TradeSetup
from es_8am.backtest.pricing import (
    select_strike, spread_credit, spread_value,
    bs_put_delta, bs_call_delta,
    bachelier_put_delta, bachelier_call_delta,
)


class DeltaStrategy(Strategy):

    def configure(self, config: Dict[str, Any]) -> None:
        opt = config.get("options", {})
        self._target_delta = opt.get("target_short_delta", 0.16)
        self._spread_width = opt.get("spread_width_pts", 5.0)
        self._strike_step = opt.get("strike_step_pts", 5.0)
        self._min_credit = opt.get("min_credit_per_spread", 0.50)
        self._slippage = opt.get("slippage_per_spread", 0.0)
        self._pricing_uncertainty = opt.get("pricing_uncertainty_per_leg", 0.0)
        self._spread_mode = opt.get("spread_mode", "directional")
        self._pricing_model = opt.get("pricing_model", "bs")
        self._iv_correction = opt.get("iv_correction_factor", 1.0)

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

        short_k = select_strike(
            spot, self._target_delta, iv_adj, tte_years,
            spread_type, self._strike_step, model=self._pricing_model,
        )
        long_k = (short_k - self._spread_width if spread_type == "put_credit_spread"
                  else short_k + self._spread_width)

        is_call_flag = 0 if spread_type == "put_credit_spread" else 1

        # Price the spread: ANN or BS
        # ANN path applies pricing_uncertainty internally
        if engine._price_ann is not None:
            credit = engine._ann_spread_credit(spot, short_k, long_k, tte_hours, spread_type)
        else:
            iv_short = engine._get_strike_iv(iv_adj, spot, short_k, tte_years, is_call_flag)
            credit = spread_credit(spot, short_k, long_k, tte_years, iv_short,
                                   spread_type, model=self._pricing_model)
            # Worst-case: short leg lower, long leg higher -> 2x uncertainty
            credit = max(0.0, credit - 2 * self._pricing_uncertainty)
        credit = round(credit - self._slippage, 2)

        if credit < self._min_credit:
            return None

        # Compute delta for reporting
        if self._pricing_model == "bachelier":
            _put_d, _call_d = bachelier_put_delta, bachelier_call_delta
        else:
            _put_d, _call_d = bs_put_delta, bs_call_delta
        if spread_type == "put_credit_spread":
            short_delta = abs(_put_d(spot, short_k, tte_years, iv_adj))
        else:
            short_delta = _call_d(spot, short_k, tte_years, iv_adj)

        return TradeSetup(
            spread_type=spread_type,
            short_strike=short_k,
            long_strike=long_k,
            spread_width=self._spread_width,
            credit=credit,
            short_delta=round(short_delta, 4),
            iv_used=session_iv,
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
        is_call_flag = 0 if spread_type == "put_credit_spread" else 1
        # ANN path applies pricing_uncertainty internally
        if engine._price_ann is not None:
            return engine._ann_spread_value(spot, short_k, long_k, tte_hours, spread_type)
        iv = engine._get_strike_iv(session_iv, spot, short_k, tte_years, is_call_flag)
        val = spread_value(spot, short_k, long_k, tte_years, iv,
                           spread_type, model=self._pricing_model)
        # Worst-case MTM: costs more to close (inflated by 2x uncertainty)
        return val + 2 * self._pricing_uncertainty

    def grid_params(self) -> Dict[str, List[Any]]:
        return {
            "target_short_delta": [0.16, 0.20, 0.25],
            "spread_width_pts":   [5, 10, 15, 20, 25],
            "tp_pct_of_credit":   [0.25, 0.50, 0.75],
            "min_orb_range_pts":  [2.0, 3.0, 5.0],
            "midline_tolerance":  [0.25, 0.50, 1.0],
            "hold_put_breach":    [False, True],
            "pricing_model":      ["bs", "bachelier"],
        }

    def apply_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(config)
        cfg.setdefault("options", {})
        cfg.setdefault("strategy", {})
        cfg["strategy"].setdefault("orb_range_filter", {})
        cfg["options"]["target_short_delta"] = params["target_short_delta"]
        cfg["options"]["spread_width_pts"] = params["spread_width_pts"]
        cfg["options"]["tp_pct_of_credit"] = params["tp_pct_of_credit"]
        cfg["strategy"]["midline_tolerance_pts"] = params["midline_tolerance"]
        cfg["strategy"]["orb_range_filter"]["min_pts"] = params["min_orb_range_pts"]
        if "hold_put_breach" in params:
            cfg["options"]["hold_put_breach"] = params["hold_put_breach"]
        if "pricing_model" in params:
            cfg["options"]["pricing_model"] = params["pricing_model"]
        if "quantity" in params:
            cfg["options"]["quantity"] = params["quantity"]
            mult = cfg.get("options", {}).get("contract_multiplier", 50)
            width = params["spread_width_pts"]
            cfg.setdefault("risk", {})["max_loss_per_trade_usd"] = width * mult * params["quantity"]
        if "assumed_iv" in params:
            cfg["options"]["assumed_iv"] = params["assumed_iv"]
        if "pricing_uncertainty_per_leg" in params:
            cfg["options"]["pricing_uncertainty_per_leg"] = params["pricing_uncertainty_per_leg"]
        if "spread_mode" in params:
            cfg["options"]["spread_mode"] = params["spread_mode"]
        return cfg
