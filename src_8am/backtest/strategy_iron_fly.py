"""
Iron Fly strategy — 0DTE SPX Iron Fly with ATM short strikes and EM-based wings.

Implements the Strategy ABC for use with IronFlyEngine (iron_fly_engine.py).

Strategy logic (from bot_prompt_iron_fly.md):
  1. Short put + short call at the FIRST strike strictly above current price
     (e.g. SPX at 6848 → center strike = 6850).  Intentionally leverages
     the pricing bias of put skew.
  2. Long wings placed at em_wing_fraction × Expected Move from center,
     rounded to strike_step_pts.  EM is the ATM straddle price from CBBO
     (em_source="straddle") or VIX-based estimate (em_source="vix").
  3. Entry at 10:00 AM ET, only if price is within the 30-min opening range.
  4. TP: fixed dollar amount per contract (tp_fixed_dollars).
  5. SL: underlying crosses wing strike → immediate close (wing-based).
     Optional value-based SL (sl_fixed_dollars) exits when mark-to-market
     loss exceeds threshold; disabled by default (0).

Config parameters (all under `options:` in YAML):
  strike_step_pts      (float)  Strike grid spacing.  Default: 5
  em_wing_fraction     (float)  Wing distance as fraction of EM.
                                1.0 = full EM (~30-40pt wings, max loss ~$1000+).
                                0.5 = half EM (~15-20pt wings, max loss ~$350).
                                Default: 1.0
  tp_fixed_dollars     (float)  Take-profit per contract in dollars.  Default: 75
  sl_fixed_dollars     (float)  Value-based stop-loss per contract in dollars.
                                0 = disabled (exit only on wing breach).  Default: 0
  slippage_per_leg     (float)  Slippage deducted per leg at entry (×4 total).
                                Default: 0.05
  contract_multiplier  (int)    Dollar value per point.  Default: 100 (SPX)
  quantity             (int)    Number of contracts.  Default: 1
  pricing_model        (str)    "bs" (Black-Scholes) or "bachelier".  Default: "bs"
  em_source            (str)    "straddle" (CBBO ATM straddle) or "vix" (IV-based).
                                Default: "straddle"

Optimizer integration:
  grid_params()   — returns parameter grid for WF-CV sweep.
  apply_params()  — maps optimizer params to config dict.
  To optimize a new parameter: add it to grid_params(), apply_params(),
  and configure() — no engine changes required.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from src_8am.backtest.strategy_base import Strategy, TradeSetup
from src_8am.backtest.pricing import (
    bs_call, bs_put, bs_call_delta, bs_put_delta,
    spread_credit, spread_value,
)


class IronFlyStrategy(Strategy):
    """ATM Iron Fly: short straddle at center strike, wings at EM distance.

    All config is read from the YAML config dict — see module docstring
    for the full parameter list.
    """

    # ------------------------------------------------------------------
    # Strategy ABC: configure
    # ------------------------------------------------------------------

    def configure(self, config: Dict[str, Any]) -> None:
        """Read all strategy parameters from config.

        Every parameter has a sensible default so the strategy works
        with a minimal config.  See module docstring for descriptions.
        """
        opt = config.get("options", {})
        self._strike_step = opt.get("strike_step_pts", 5.0)
        self._slippage_per_leg = opt.get("slippage_per_leg", 0.05)
        self._tp_fixed_dollars = opt.get("tp_fixed_dollars", 75.0)
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)
        self._pricing_model = opt.get("pricing_model", "bs")
        self._em_source = opt.get("em_source", "straddle")
        self._em_wing_fraction = opt.get("em_wing_fraction", 1.0)

        # CBBO store reference (set externally via set_cbbo_store)
        self._cbbo_store = None

    def set_cbbo_store(self, store) -> None:
        """Inject CbboStore for real SPX option quote pricing."""
        self._cbbo_store = store

    # ------------------------------------------------------------------
    # Strategy ABC: select_trade
    # ------------------------------------------------------------------

    def select_trade(
        self,
        spot: float,
        bias: str,               # ignored — iron fly is non-directional
        entry_time: datetime,
        session_iv: float,
        tte_years: float,
        tte_hours: float,
        engine: Any,
    ) -> Optional[TradeSetup]:
        """Select iron fly strikes and compute entry credit.

        1. Center strike = first strike strictly above spot.
        2. Wings = center ± (em_wing_fraction × EM), snapped to strike grid.
        3. Price via CBBO mid (preferred) or BS fallback.
        4. Deduct slippage (4 legs).

        Returns TradeSetup with spread_type="iron_fly" and all 4-leg
        details packed into the `extra` dict, or None if no valid trade.
        """
        # Center strike = first strike strictly above spot
        center_k = math.ceil(spot / self._strike_step) * self._strike_step
        if center_k == spot:
            center_k += self._strike_step

        minute_et = entry_time.strftime("%H:%M")

        # Expected move from ATM straddle (CBBO) or VIX-based fallback
        em = self._get_expected_move(spot, session_iv, tte_hours, minute_et)
        if em is None or em <= 0:
            return None

        # Wing strikes: distance = em × wing fraction, rounded to strike step
        wing_dist = em * self._em_wing_fraction
        long_call_k = math.ceil((center_k + wing_dist) / self._strike_step) * self._strike_step
        long_put_k = math.floor((center_k - wing_dist) / self._strike_step) * self._strike_step

        # Ensure wings are at least 1 step away
        if long_call_k <= center_k:
            long_call_k = center_k + self._strike_step
        if long_put_k >= center_k:
            long_put_k = center_k - self._strike_step

        put_width = center_k - long_put_k
        call_width = long_call_k - center_k

        # Price all 4 legs (CBBO mid with BS fallback)
        put_credit, call_credit = self._price_iron_fly(
            spot, center_k, long_put_k, long_call_k,
            session_iv, tte_years, minute_et,
        )
        if put_credit is None or call_credit is None:
            return None

        total_slippage = self._slippage_per_leg * 4
        total_credit = round(put_credit + call_credit - total_slippage, 2)

        if total_credit <= 0:
            return None

        put_delta = abs(bs_put_delta(spot, center_k, tte_years, session_iv))
        call_delta = bs_call_delta(spot, center_k, tte_years, session_iv)

        return TradeSetup(
            spread_type="iron_fly",
            short_strike=center_k,
            long_strike=long_put_k,   # ABC compat; full info in extra
            spread_width=max(put_width, call_width),
            credit=total_credit,
            short_delta=round((put_delta + call_delta) / 2, 4),
            iv_used=session_iv,
            extra={
                "center_strike": center_k,
                "long_put": long_put_k,
                "long_call": long_call_k,
                "put_width": put_width,
                "call_width": call_width,
                "put_credit": round(put_credit, 2),
                "call_credit": round(call_credit, 2),
                "total_credit": total_credit,
                "em_pts": round(em, 2),
            },
        )

    # ------------------------------------------------------------------
    # Strategy ABC: compute_exit_value
    # ------------------------------------------------------------------

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
        """Mark-to-market the full iron fly (put spread + call spread).

        short_k = center strike.  Actual wing strikes are read from
        engine._current_trade_extra (set by IronFlyEngine before calling).
        """
        extra = getattr(engine, "_current_trade_extra", {})
        long_put_k = extra.get("long_put", short_k - 25)
        long_call_k = extra.get("long_call", short_k + 25)

        put_val = spread_value(
            spot, short_k, long_put_k, tte_years, session_iv,
            "put_credit_spread", model=self._pricing_model,
        )
        call_val = spread_value(
            spot, short_k, long_call_k, tte_years, session_iv,
            "call_credit_spread", model=self._pricing_model,
        )
        return round(put_val + call_val, 2)

    # ------------------------------------------------------------------
    # Strategy ABC: grid_params / apply_params (optimizer integration)
    # ------------------------------------------------------------------

    def grid_params(self) -> Dict[str, List[Any]]:
        """Parameter grid for WF-CV optimization.

        To sweep a new parameter: add it here and in apply_params().
        Locked values use single-element lists (e.g. [0.50]).
        """
        return {
            "em_wing_fraction": [0.50],             # locked: caps max loss ~$350
            "tp_fixed_dollars": [50, 75, 100, 125, 150, 200],
            "sl_fixed_dollars": [0],                # wing-only (narrow wings = natural SL)
            "slippage_per_leg": [0.05, 0.10],
            "macro_filter_enabled": [True, False],
            "orb_filter_tolerance_pct": [0.0, 0.05, 0.10],
            "skip_events": [True, False],           # NFP/FOMC/CPI calendar filter
        }

    def apply_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Map optimizer params to config dict for a single backtest run."""
        cfg = copy.deepcopy(config)
        cfg.setdefault("options", {})
        cfg.setdefault("strategy", {}).setdefault("macro_filter", {})
        for key in ("tp_fixed_dollars", "slippage_per_leg", "em_wing_fraction",
                     "sl_fixed_dollars"):
            if key in params:
                cfg["options"][key] = params[key]
        if "macro_filter_enabled" in params:
            cfg["strategy"]["macro_filter"]["enabled"] = params["macro_filter_enabled"]
        if "orb_filter_tolerance_pct" in params:
            cfg["strategy"]["orb_filter_tolerance_pct"] = params["orb_filter_tolerance_pct"]
        if "skip_events" in params:
            skip = params["skip_events"]
            cfg["strategy"]["skip_nfp_fridays"] = skip
            cfg["strategy"]["skip_fomc_days"] = skip
            cfg["strategy"]["skip_cpi_days"] = skip
        return cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_expected_move(
        self, spot: float, session_iv: float, tte_hours: float, minute_et: str,
    ) -> Optional[float]:
        """Get expected move in points.

        Primary: ATM straddle price from CBBO (em_source="straddle").
        This is the market's real 0DTE expected move — no model needed.

        Fallback: VIX-based estimate = spot × IV/sqrt(252) × sqrt(tte/6.5).
        Used when CBBO data is unavailable or em_source="vix".
        """
        if self._em_source == "straddle" and self._cbbo_store is not None:
            straddle = self._cbbo_store.atm_straddle(minute_et, spot)
            if straddle is not None and straddle > 0:
                return straddle

        daily_em = spot * session_iv / math.sqrt(252)
        intraday_fraction = tte_hours / 6.5
        return daily_em * math.sqrt(max(intraday_fraction, 0.001))

    def _price_iron_fly(
        self,
        spot: float,
        center_k: float,
        long_put_k: float,
        long_call_k: float,
        session_iv: float,
        tte_years: float,
        minute_et: str,
    ) -> tuple[Optional[float], Optional[float]]:
        """Price put and call credit spreads of the iron fly.

        Returns (put_credit, call_credit).
        Uses CBBO mid prices when available, Black-Scholes fallback otherwise.
        """
        put_credit = None
        call_credit = None

        if self._cbbo_store is not None:
            # Put spread: sell center put, buy wing put
            q = self._cbbo_store.mid_spread_quote(
                minute_et, center_k, long_put_k, is_call=False,
            )
            if q is not None:
                put_credit = q[0]

            # Call spread: sell center call, buy wing call
            q = self._cbbo_store.mid_spread_quote(
                minute_et, center_k, long_call_k, is_call=True,
            )
            if q is not None:
                call_credit = q[0]

        # BS fallback for missing legs
        if put_credit is None:
            put_credit = spread_credit(
                spot, center_k, long_put_k, tte_years, session_iv,
                "put_credit_spread", model=self._pricing_model,
            )
        if call_credit is None:
            call_credit = spread_credit(
                spot, center_k, long_call_k, tte_years, session_iv,
                "call_credit_spread", model=self._pricing_model,
            )

        return put_credit, call_credit
