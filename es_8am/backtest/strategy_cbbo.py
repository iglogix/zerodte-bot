"""
CBBO-based pricing strategy — uses real SPX bid/ask for credit spread pricing.

Strike selection still uses BS delta (we need delta to find the target strike),
but entry credit and exit mark-to-market use actual CBBO quotes.

Falls back to BS pricing when CBBO quotes are unavailable for a strike/minute.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from es_8am.backtest.pricing import (
    bs_call_delta,
    bs_put_delta,
    select_strike,
    spread_credit,
)
from es_8am.backtest.strategy_base import TradeSetup
from es_8am.backtest.strategy_delta import DeltaStrategy


class CbboDeltaStrategy(DeltaStrategy):
    """Delta-based strike selection with CBBO market pricing."""

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        # CBBO store is injected by the engine after construction
        self._cbbo_store = None

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

        # Use BS delta to find the target strike (same as DeltaStrategy)
        short_k = select_strike(
            spot, self._target_delta, iv_adj, tte_years,
            spread_type, self._strike_step, model=self._pricing_model,
        )
        long_k = (short_k - self._spread_width if spread_type == "put_credit_spread"
                  else short_k + self._spread_width)

        is_call = spread_type == "call_credit_spread"
        minute_et = self._minute_key(entry_time)

        # Try CBBO pricing first
        credit = None
        if self._cbbo_store is not None:
            short_q = self._cbbo_store.lookup(minute_et, short_k, is_call)
            long_q = self._cbbo_store.lookup(minute_et, long_k, is_call)
            if short_q is not None and long_q is not None:
                # Entry: sell short at mid, buy long at mid (realistic fill)
                short_mid = (short_q[0] + short_q[1]) / 2
                long_mid = (long_q[0] + long_q[1]) / 2
                credit = round(short_mid - long_mid, 2)

        # Fallback to BS if CBBO unavailable
        if credit is None:
            is_call_flag = 0 if spread_type == "put_credit_spread" else 1
            iv_short = engine._get_strike_iv(iv_adj, spot, short_k, tte_years, is_call_flag)
            credit = spread_credit(spot, short_k, long_k, tte_years, iv_short,
                                   spread_type, model=self._pricing_model)

        credit = round(credit - self._slippage, 2)

        if credit < self._min_credit:
            return None

        # Compute delta for reporting
        if spread_type == "put_credit_spread":
            short_delta = abs(bs_put_delta(spot, short_k, tte_years, iv_adj))
        else:
            short_delta = bs_call_delta(spot, short_k, tte_years, iv_adj)

        return TradeSetup(
            spread_type=spread_type,
            short_strike=short_k,
            long_strike=long_k,
            spread_width=self._spread_width,
            credit=credit,
            short_delta=round(short_delta, 4),
            iv_used=session_iv,
            extra={"pricing": "cbbo" if self._cbbo_store else "bs"},
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
            # Get current time from engine's last bar
            # Engine passes tte_hours, we can derive minute from that
            # But we need the actual timestamp — use engine._last_bar_time
            minute_et = getattr(engine, '_last_bar_time', None)
            if minute_et is not None:
                short_q = self._cbbo_store.lookup(minute_et, short_k, is_call)
                long_q = self._cbbo_store.lookup(minute_et, long_k, is_call)
                if short_q is not None and long_q is not None:
                    # Exit: buy back short at mid, sell long at mid
                    short_mid = (short_q[0] + short_q[1]) / 2
                    long_mid = (long_q[0] + long_q[1]) / 2
                    debit = short_mid - long_mid
                    return max(0.0, round(debit, 4))

        # Fallback to BS
        return super().compute_exit_value(
            spot, short_k, long_k, tte_years, tte_hours,
            spread_type, session_iv, engine,
        )

    def grid_params(self) -> Dict[str, List[Any]]:
        return {
            "target_short_delta": [0.14, 0.16, 0.20, 0.25],
            "spread_width_pts":   [5, 10, 15, 20, 25],
            "tp_pct_of_credit":   [0.25, 0.50, 0.75],
            "min_orb_range_pts":  [2.0, 3.0, 5.0],
            "midline_tolerance":  [0.25, 0.50, 1.0],
            "hold_put_breach":    [False, True],
        }

    def apply_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = super().apply_params(config, params)
        # No pricing_model sweep — CBBO is the pricing model
        cfg["options"]["pricing_model"] = "cbbo"
        return cfg
