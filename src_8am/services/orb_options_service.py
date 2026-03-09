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

        self._spread_width: float = self._opt_cfg.get("spread_width_pts", 5.0)
        self._target_delta: float = self._opt_cfg.get("target_short_delta", 0.16)
        self._max_delta: float = self._opt_cfg.get("max_short_delta", 0.25)
        self._min_credit: float = self._opt_cfg.get("min_credit_per_spread", 0.50)
        self._tp_pct: float = self._opt_cfg.get("tp_pct_of_credit", 0.50)
        self._sl_mult: float = self._opt_cfg.get("sl_mult_of_credit", 2.0)

    # -------------------------------------------------------------------------
    # Option chain
    # -------------------------------------------------------------------------

    async def get_0dte_chain(self, spx_price: float) -> List[Dict[str, Any]]:
        """
        Fetch SPX 0DTE option chain filtered to near-ATM strikes (±150 pts).
        Returns list enriched with bid/ask/delta.
        """
        try:
            chain = await self._tt.get_option_chain("SPX")
            today = date.today().isoformat()
            chain = [o for o in chain if str(o.get("expiration", "")).startswith(today)]
            if spx_price:
                chain = [o for o in chain if abs(o.get("strike", 0) - spx_price) <= 150]
            return chain
        except Exception:
            logger.exception("Failed to fetch SPX 0DTE chain")
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
            "max_profit_dollars": round(credit * 100, 2),
            "max_loss_dollars": round((width - credit) * 100, 2),
            "tp_debit": self.tp_price(credit),
            "sl_debit": self.sl_price(credit),
            "expiration": date.today().isoformat(),
        }

    @staticmethod
    def _build_symbol(side: str, strike: float) -> str:
        """Build a SPXW OCC symbol for today's expiry."""
        today = date.today().strftime("%y%m%d")
        strike_int = int(strike)
        return f"SPXW{today}{side}{strike_int:08d}"
