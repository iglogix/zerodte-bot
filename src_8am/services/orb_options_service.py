"""
OrbOptionsService — SPX 0DTE option chain fetching and ORB-aware strike selection.

Strategy logic:
  LONG bias  (bullish breakout) → sell PUT credit spread below ORB low
  SHORT bias (bearish breakout) → sell CALL credit spread above ORB high

Strike placement rules:
  Short strike: nearest strike at or below target delta (default ~0.16)
  Long strike:  short_strike - spread_width (puts) / short_strike + spread_width (calls)
  Minimum credit filter applied before accepting strikes.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src_8am.config_keys import require

logger = logging.getLogger(__name__)


class OrbOptionsService:
    """
    Wraps TastyTradeClient option chain calls with ORB-specific strike selection.
    """

    def __init__(self, tt_client, config: Dict[str, Any]):
        self._tt = tt_client
        self._cfg = config
        self._opt_cfg = config.get("options", {})
        self._strategy_cfg = config.get("strategy", {})
        self._product_code: str = config.get("bot", {}).get("symbol", "ES")

        self._spread_width: float = self._opt_cfg.get("spread_width_pts", 5.0)
        self._target_delta: float = self._opt_cfg.get("target_short_delta", 0.16)
        self._max_delta: float = self._opt_cfg.get("max_short_delta", 0.25)
        self._min_credit: float = self._opt_cfg.get("min_credit_per_spread", 0.50)
        self._tp_pct: float = self._opt_cfg.get("tp_pct_of_credit", 0.50)
        self._sl_mult: float = self._opt_cfg.get("sl_mult_of_credit", 0)
        self._atm_filter_width: float = self._opt_cfg.get("atm_filter_width_pts", 150.0)
        self._multiplier: float = self._opt_cfg.get("contract_multiplier", 100)

        # Straddle-EM strike selection (replaces delta-based when enabled)
        self._use_em: bool = self._opt_cfg.get("use_straddle_em", False)
        self._em_with_trend: float = self._opt_cfg.get("em_with_trend", 0.50)
        self._em_against_trend: float = self._opt_cfg.get("em_against_trend", 0.75)
        self._width_with_trend: float = self._opt_cfg.get("width_with_trend", 5.0)
        self._width_against_trend: float = self._opt_cfg.get("width_against_trend", 10.0)
        self._spread_mode: str = self._opt_cfg.get("spread_mode", "directional")
        self._strike_step: float = self._opt_cfg.get("strike_step_pts", 5.0)

    # -------------------------------------------------------------------------
    # Option chain
    # -------------------------------------------------------------------------

    async def get_0dte_chain(self, underlying_price: float) -> List[Dict[str, Any]]:
        """
        Fetch 0DTE option chain filtered to near-ATM strikes (±150 pts).
        Supports both futures options (ES/EW) and equity/index options (SPX/SPXW).
        """
        # Resolved outside the try: a missing key must fail loudly, not as [].
        chain_type = require(self._cfg, "options.chain_type")
        chain_symbol = require(self._cfg, "options.chain_symbol") if chain_type == "index" else None
        try:
            if chain_type == "index":
                # Index options — use equity option chain
                chain = await self._tt.get_option_chain(chain_symbol)
            else:
                # Futures options (ES)
                chain = await self._tt.get_future_option_chain(
                    self._product_code, underlying_price=underlying_price
                )
            today = date.today().isoformat()
            chain = [o for o in chain if str(o.get("expiration", "")).startswith(today)]
            if underlying_price:
                chain = [o for o in chain if abs(o.get("strike", 0) - underlying_price) <= self._atm_filter_width]
            return chain
        except Exception:
            logger.exception("Failed to fetch %s 0DTE chain", self._product_code)
            return []

    # -------------------------------------------------------------------------
    # Strike selection
    # -------------------------------------------------------------------------

    def select_put_spread(
        self,
        chain: List[Dict[str, Any]],
        orb_low: float,
        spx_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Select PUT credit spread strikes for a LONG bias trade.
        Short put placed at or just below ORB low at ~target_delta.
        Returns strike selection dict or None if no valid spread found.
        """
        puts = sorted(
            [o for o in chain if o.get("option_type") == "put" and o.get("bid", 0) > 0],
            key=lambda o: o["strike"],
            reverse=True,
        )
        if not puts:
            return None

        # Prefer strike nearest to ORB low with delta ≤ max_delta
        candidates = [
            p for p in puts
            if p["strike"] <= orb_low and abs(p.get("delta", 1)) <= self._max_delta
        ]
        if not candidates:
            # Relax: any put below current price with delta <= max_delta
            candidates = [
                p for p in puts
                if p["strike"] < spx_price and abs(p.get("delta", 1)) <= self._max_delta
            ]
        if not candidates:
            return None

        # Pick closest to target delta
        short_put = min(candidates, key=lambda p: abs(abs(p.get("delta", 0)) - self._target_delta))
        long_strike = short_put["strike"] - self._spread_width

        long_candidates = [
            p for p in puts
            if abs(p["strike"] - long_strike) <= 1.0
        ]
        if not long_candidates:
            # Construct synthetic long put at spread_width below short
            long_put = {
                "strike": long_strike,
                "bid": 0.0,
                "ask": 0.0,
                "symbol": self._build_symbol("P", long_strike),
                "option_type": "put",
            }
        else:
            long_put = min(long_candidates, key=lambda p: abs(p["strike"] - long_strike))

        credit = self._mid(short_put) - self._mid(long_put)
        if credit < self._min_credit:
            logger.debug(
                "Put spread credit %.2f below min %.2f at short=%s",
                credit, self._min_credit, short_put["strike"],
            )
            return None

        return self._spread_result("put_credit_spread", short_put, long_put, credit)

    def select_call_spread(
        self,
        chain: List[Dict[str, Any]],
        orb_high: float,
        spx_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Select CALL credit spread strikes for a SHORT bias trade.
        Short call placed at or just above ORB high at ~target_delta.
        Returns strike selection dict or None if no valid spread found.
        """
        calls = sorted(
            [o for o in chain if o.get("option_type") == "call" and o.get("bid", 0) > 0],
            key=lambda o: o["strike"],
        )
        if not calls:
            return None

        candidates = [
            c for c in calls
            if c["strike"] >= orb_high and abs(c.get("delta", 1)) <= self._max_delta
        ]
        if not candidates:
            # Widen: any call above spx_price with valid delta
            candidates = [
                c for c in calls
                if c["strike"] > spx_price and abs(c.get("delta", 1)) <= self._max_delta
            ]
        if not candidates:
            # Last resort: take the most OTM call with any delta (sorted ascending by delta)
            candidates = [c for c in calls if c["strike"] > spx_price]
            candidates = sorted(candidates, key=lambda c: abs(c.get("delta", 1)))[:5]
        if not candidates:
            return None

        short_call = min(candidates, key=lambda c: abs(abs(c.get("delta", 0)) - self._target_delta))
        long_strike = short_call["strike"] + self._spread_width

        long_candidates = [c for c in calls if abs(c["strike"] - long_strike) <= 1.0]
        if not long_candidates:
            long_call = {
                "strike": long_strike,
                "bid": 0.0,
                "ask": 0.0,
                "symbol": self._build_symbol("C", long_strike),
                "option_type": "call",
            }
        else:
            long_call = min(long_candidates, key=lambda c: abs(c["strike"] - long_strike))

        credit = self._mid(short_call) - self._mid(long_call)
        if credit < self._min_credit:
            logger.debug(
                "Call spread credit %.2f below min %.2f at short=%s",
                credit, self._min_credit, short_call["strike"],
            )
            return None

        return self._spread_result("call_credit_spread", short_call, long_call, credit)

    # -------------------------------------------------------------------------
    # Straddle-EM strike selection
    # -------------------------------------------------------------------------

    def derive_spot_from_chain(self, chain: List[Dict[str, Any]], es_price: float) -> Optional[float]:
        """Derive index spot price from put-call parity at ATM strikes.

        For 0DTE: S ≈ K + C_mid - P_mid at the ATM strike.
        Returns None if chain lacks paired put/call data.
        """
        # Group by strike
        puts_by_k: Dict[float, Dict] = {}
        calls_by_k: Dict[float, Dict] = {}
        for o in chain:
            k = o.get("strike", 0)
            if o.get("option_type") == "put" and o.get("bid", 0) > 0:
                puts_by_k[k] = o
            elif o.get("option_type") == "call" and o.get("bid", 0) > 0:
                calls_by_k[k] = o

        # Find strikes with both put and call
        common = sorted(set(puts_by_k) & set(calls_by_k))
        if not common:
            return None

        # Pick the strike nearest to ES price as starting ATM guess
        atm_k = min(common, key=lambda k: abs(k - es_price))

        # Derive spot: S = K + C_mid - P_mid
        c_mid = self._mid(calls_by_k[atm_k])
        p_mid = self._mid(puts_by_k[atm_k])
        spot = atm_k + c_mid - p_mid

        logger.info(
            "Derived SPX spot=%.2f from put-call parity at K=%s (C=%.2f P=%.2f, ES=%.2f, basis=%.1f)",
            spot, atm_k, c_mid, p_mid, es_price, es_price - spot,
        )
        return round(spot, 2)

    def atm_straddle(self, chain: List[Dict[str, Any]], spot: float) -> float:
        """Compute ATM straddle price from chain — this IS the 0DTE expected move."""
        import math
        atm_strike = round(spot / self._strike_step) * self._strike_step
        atm_put = next((o for o in chain if o.get("option_type") == "put"
                        and abs(o["strike"] - atm_strike) < self._strike_step), None)
        atm_call = next((o for o in chain if o.get("option_type") == "call"
                         and abs(o["strike"] - atm_strike) < self._strike_step), None)
        if atm_put and atm_call:
            return self._mid(atm_put) + self._mid(atm_call)
        # Fallback: use closest available
        puts = [o for o in chain if o.get("option_type") == "put" and o.get("bid", 0) > 0]
        calls = [o for o in chain if o.get("option_type") == "call" and o.get("bid", 0) > 0]
        if puts and calls:
            best_put = min(puts, key=lambda o: abs(o["strike"] - spot))
            best_call = min(calls, key=lambda o: abs(o["strike"] - spot))
            return self._mid(best_put) + self._mid(best_call)
        return 0.0

    def select_em_spread(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
        spread_type: str,
        orb_bias: str,
    ) -> Optional[Dict[str, Any]]:
        """Select spread using straddle-EM distance instead of BS delta.

        Args:
            chain: 0DTE option chain with bid/ask
            spot: current underlying price
            spread_type: "put_credit_spread" or "call_credit_spread"
            orb_bias: actual ORB breakout direction ("long"/"short")
        """
        import math

        em = self.atm_straddle(chain, spot)
        if em <= 0:
            logger.warning("ATM straddle=0, cannot compute EM")
            return None

        # Determine if this leg is with-trend or against-trend
        is_with_trend = (
            (spread_type == "put_credit_spread" and orb_bias == "long") or
            (spread_type == "call_credit_spread" and orb_bias == "short")
        )
        em_mult = self._em_with_trend if is_with_trend else self._em_against_trend
        width = self._width_with_trend if is_with_trend else self._width_against_trend

        is_put = spread_type == "put_credit_spread"
        side = "put" if is_put else "call"

        # Compute target strike
        if is_put:
            raw_short = spot - em_mult * em
            short_strike = math.floor(raw_short / self._strike_step) * self._strike_step
            long_strike = short_strike - width
        else:
            raw_short = spot + em_mult * em
            short_strike = math.ceil(raw_short / self._strike_step) * self._strike_step
            long_strike = short_strike + width

        # Find options at these strikes
        options = [o for o in chain if o.get("option_type") == side]
        short_opt = next((o for o in options if abs(o["strike"] - short_strike) < 1.0), None)
        long_opt = next((o for o in options if abs(o["strike"] - long_strike) < 1.0), None)

        if not short_opt:
            short_opt = {
                "strike": short_strike, "bid": 0.0, "ask": 0.0,
                "symbol": self._build_symbol("P" if is_put else "C", short_strike),
                "option_type": side, "delta": 0.0,
            }
        if not long_opt:
            long_opt = {
                "strike": long_strike, "bid": 0.0, "ask": 0.0,
                "symbol": self._build_symbol("P" if is_put else "C", long_strike),
                "option_type": side, "delta": 0.0,
            }

        credit = self._mid(short_opt) - self._mid(long_opt)
        if credit < self._min_credit:
            logger.debug(
                "EM spread credit %.2f below min %.2f: %s short=%s (%.1f×EM=%.0f)",
                credit, self._min_credit, spread_type, short_strike, em_mult, em_mult * em,
            )
            return None

        trend_label = "with-trend" if is_with_trend else "against-trend"
        logger.info(
            "EM spread: %s %s short=%s em=%.1f em_mult=%.2f dist=%.0f width=%.0f credit=%.2f",
            spread_type, trend_label, short_strike, em, em_mult, em_mult * em, width, credit,
        )

        result = self._spread_result(spread_type, short_opt, long_opt, credit)
        result["em_distance"] = round(em_mult * em, 1)
        result["em_multiplier"] = em_mult
        result["atm_straddle"] = round(em, 2)
        result["trend_type"] = trend_label
        return result

    # -------------------------------------------------------------------------
    # Position management helpers
    # -------------------------------------------------------------------------

    def tp_price(self, credit: float) -> float:
        """Price to buy back spread at TP (50% of credit by default)."""
        return round(credit * (1 - self._tp_pct), 2)

    def sl_price(self, credit: float) -> float:
        """Price to buy back spread at SL (2x credit by default)."""
        return round(credit * self._sl_mult, 2)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _mid(opt: Dict[str, Any]) -> float:
        b, a = opt.get("bid", 0.0), opt.get("ask", 0.0)
        if b and a:
            return (b + a) / 2
        return b or a or 0.0

    def _spread_result(
        self,
        spread_type: str,
        short_opt: Dict[str, Any],
        long_opt: Dict[str, Any],
        credit: float,
    ) -> Dict[str, Any]:
        width = abs(short_opt["strike"] - long_opt["strike"])
        return {
            "spread_type": spread_type,
            "short_strike": short_opt["strike"],
            "long_strike": long_opt["strike"],
            "spread_width": width,
            "short_symbol": short_opt.get("symbol", self._build_symbol(
                "P" if spread_type == "put_credit_spread" else "C", short_opt["strike"]
            )),
            "long_symbol": long_opt.get("symbol", self._build_symbol(
                "P" if spread_type == "put_credit_spread" else "C", long_opt["strike"]
            )),
            "short_delta": short_opt.get("delta", 0.0),
            "short_bid": short_opt.get("bid", 0.0),
            "short_ask": short_opt.get("ask", 0.0),
            "long_bid": long_opt.get("bid", 0.0),
            "long_ask": long_opt.get("ask", 0.0),
            "credit": round(credit, 2),
            "max_profit_dollars": round(credit * self._multiplier, 2),
            "max_loss_dollars": round((width - credit) * self._multiplier, 2),
            "tp_debit": self.tp_price(credit),
            "sl_debit": self.sl_price(credit),
            "expiration": date.today().isoformat(),
        }

    def _build_symbol(self, side: str, strike: float) -> str:
        """Build a futures option symbol for today's expiry.
        Uses the option_root from config (default 'EW' for ES weekly options).
        """
        option_root = self._opt_cfg.get("option_root", "EW")
        today = date.today().strftime("%y%m%d")
        strike_int = int(strike)
        return f"{option_root}{today}{side}{strike_int:08d}"
