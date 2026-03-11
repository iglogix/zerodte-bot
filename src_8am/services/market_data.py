"""
OrbMarketData — async wrapper around TastyTradeClient's DXLink streamer
for all live ORB market data needs.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src_8am.utils.contract_utils import get_dxlink_symbol, get_contract_info

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class OrbMarketData:
    """
    Live market data service for the ORB bot, built on top of TastyTradeClient.

    - get_orb_candle()        → 08:00–08:15 15-min bar via DXLink
    - get_recent_5min_candles() → last N 5-min bars via DXLink
    - get_current_price()     → live mark price via TT REST (for midline polling)
    """

    def __init__(self, tt_client, config: Dict[str, Any]):
        self._tt = tt_client
        self._config = config

        # Auto-detect front-month contract unless explicitly overridden in config
        configured_symbol = config.get("market_data", {}).get("dxlink_symbol")
        if configured_symbol and configured_symbol != "auto":
            self._dxlink_symbol = configured_symbol
            logger.info("Using configured DXLink symbol: %s", self._dxlink_symbol)
        else:
            self._dxlink_symbol = get_dxlink_symbol()
            contract_info = get_contract_info()
            logger.info(
                "Auto-detected front-month: %s (%s, expires %s, %d days)",
                self._dxlink_symbol,
                contract_info["month_name"],
                contract_info["expiration"],
                contract_info["days_to_expiry"],
            )

        self._orb_candle_cache: Optional[Dict[str, Any]] = None
        self._orb_candle_date: Optional[date] = None

    # -------------------------------------------------------------------------
    # ORB candle (08:00–08:15 15-min bar)
    # -------------------------------------------------------------------------

    async def get_orb_candle(self, for_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch the 08:00–08:15 ET 15-min candle for `for_date` (defaults to today).
        Result is cached for the session day.
        """
        target = for_date or date.today()
        if self._orb_candle_cache and self._orb_candle_date == target:
            return self._orb_candle_cache

        from_dt = datetime.combine(target, time(8, 0), tzinfo=ET)
        try:
            candles = await self._tt.get_candles(
                symbol=self._dxlink_symbol,
                interval="15m",
                from_time=from_dt,
                count=1,
            )
            if candles:
                self._orb_candle_cache = candles[0]
                self._orb_candle_date = target
                logger.info(
                    "ORB candle fetched: H=%.2f L=%.2f mid=%.2f",
                    candles[0]["high"],
                    candles[0]["low"],
                    (candles[0]["high"] + candles[0]["low"]) / 2,
                )
                return candles[0]
            logger.warning("No ORB candle returned for %s", target)
            return None
        except Exception:
            logger.exception("Failed to fetch ORB candle for %s", target)
            return None

    def invalidate_orb_cache(self):
        """Clear the cached ORB candle (call at start of each session)."""
        self._orb_candle_cache = None
        self._orb_candle_date = None

    # -------------------------------------------------------------------------
    # 5-min candles for regime classifier
    # -------------------------------------------------------------------------

    async def get_recent_5min_candles(self, count: int = 10) -> List[Dict[str, Any]]:
        """Return the last `count` 5-min bars for the DXLink symbol."""
        try:
            candles = await self._tt.get_candles(
                symbol=self._dxlink_symbol,
                interval="5m",
                count=count,
            )
            return candles or []
        except Exception:
            logger.exception("Failed to fetch 5-min candles")
            return []

    # -------------------------------------------------------------------------
    # Live price (TT REST snapshot — for midline polling)
    # -------------------------------------------------------------------------

    async def get_current_price(self) -> Optional[float]:
        """
        Return the current mark price of the ES future via TT REST snapshot.
        Suitable for polling every ~1s for midline touch detection.
        """
        try:
            data = await self._tt.get_future_quote(self._dxlink_symbol)
            if data:
                mark = data.get("mark") or data.get("last") or data.get("bid")
                return float(mark) if mark is not None else None
            return None
        except Exception:
            logger.exception("Failed to fetch current price")
            return None

    # -------------------------------------------------------------------------
    # Breakout detection helper
    # -------------------------------------------------------------------------

    async def get_breakout_status(
        self,
        orb_high: float,
        orb_low: float,
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Check whether price has broken above ORB high (long bias) or below ORB low (short bias).
        Uses current_price if provided, otherwise fetches fresh.
        """
        price = current_price
        if price is None:
            price = await self.get_current_price()

        if price is None:
            return {"bias": None, "price": None, "status": "no_data"}

        if price > orb_high:
            return {"bias": "long", "price": price, "status": "breakout", "level": orb_high}
        if price < orb_low:
            return {"bias": "short", "price": price, "status": "breakout", "level": orb_low}
        return {"bias": None, "price": price, "status": "inside_range"}
