"""
ReplayTTClient — replays historical ES bars + SPX CBBO quotes for bot testing.

Serves real data from Databento ES OHLCV and CBBO pickle cache, allowing
the full bot pipeline to run against known historical days.

Usage:
    client = ReplayTTClient(
        bars=day_bars.bars,           # list of ES 1-min bar dicts
        cbbo_cache=mem[trading_date], # CBBO QuoteMap for one day
        config=config,
    )
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

from es_8am.config_keys import require
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class ReplayTTClient:
    """Replays historical data as if it were a live TastyTrade connection."""

    def __init__(
        self,
        bars: List[Dict[str, Any]],
        cbbo_quotes: Dict = None,
        config: Dict[str, Any] = None,
    ):
        self._config = config or {}
        self._bars = sorted(bars, key=lambda b: b["ts_et"])
        self._cbbo = cbbo_quotes or {}  # {(minute_str, strike, is_call): (bid, ask)}
        self._option_root = require(self._config, "options.option_root")
        self._strike_step = self._config.get("options", {}).get("strike_step_pts", 5)
        self._atm_filter = self._config.get("options", {}).get("atm_filter_width_pts", 150)

        # Current position in bar replay
        self._bar_idx = 0
        self._current_time: Optional[datetime] = None

    def set_time(self, t: datetime):
        """Advance replay clock to time t. Updates bar index."""
        self._current_time = t
        # Advance bar index to match time
        while (self._bar_idx < len(self._bars) - 1
               and self._bars[self._bar_idx + 1]["ts_et"] <= t):
            self._bar_idx += 1

    def _current_bar(self) -> Optional[Dict]:
        if not self._bars:
            return None
        return self._bars[min(self._bar_idx, len(self._bars) - 1)]

    def _current_price(self) -> float:
        bar = self._current_bar()
        return bar["close"] if bar else 0.0

    def _minute_key(self) -> str:
        if self._current_time:
            return self._current_time.strftime("%H:%M")
        bar = self._current_bar()
        if bar:
            return bar["ts_et"].strftime("%H:%M")
        return "09:30"

    # -------------------------------------------------------------------------
    # Connection stubs
    # -------------------------------------------------------------------------

    async def connect(self): pass
    async def disconnect(self): pass

    # -------------------------------------------------------------------------
    # Market data — replays ES bars
    # -------------------------------------------------------------------------

    async def get_candles(
        self, symbol: str, interval: str = "5m", from_time=None, count: int = 1
    ) -> List[Dict]:
        interval_mins = {"1m": 1, "5m": 5, "15m": 15}.get(interval, 5)

        if from_time:
            if isinstance(from_time, str):
                from_time = datetime.fromisoformat(from_time)
            if from_time.tzinfo is None:
                from_time = from_time.replace(tzinfo=ET)

        # Find bars matching the requested window
        result = []
        for bar in self._bars:
            ts = bar["ts_et"]
            if from_time and ts < from_time:
                continue
            result.append({
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar.get("volume", 0),
                "timestamp": ts.isoformat(),
            })

        # Aggregate to requested interval if needed
        if interval_mins > 1 and result:
            agg = self._aggregate_bars(result, interval_mins)
            return agg[-count:] if len(agg) > count else agg

        return result[-count:] if len(result) > count else result

    def _aggregate_bars(self, bars: List[Dict], interval_mins: int) -> List[Dict]:
        """Aggregate 1-min bars into N-minute bars."""
        if not bars:
            return []
        agg = []
        chunk = []
        for bar in bars:
            chunk.append(bar)
            if len(chunk) >= interval_mins:
                agg.append({
                    "open": chunk[0]["open"],
                    "high": max(b["high"] for b in chunk),
                    "low": min(b["low"] for b in chunk),
                    "close": chunk[-1]["close"],
                    "volume": sum(b.get("volume", 0) for b in chunk),
                    "timestamp": chunk[0]["timestamp"],
                })
                chunk = []
        return agg

    async def get_future_quote(self, symbol: str) -> Dict:
        price = self._current_price()
        return {
            "symbol": symbol,
            "bid": round(price - 0.25, 2),
            "ask": round(price + 0.25, 2),
            "last": round(price, 2),
            "mark": round(price, 2),
        }

    async def get_quote(self, symbol: str) -> Dict:
        return await self.get_future_quote(symbol)

    # -------------------------------------------------------------------------
    # Options chain — serves CBBO quotes
    # -------------------------------------------------------------------------

    async def get_future_option_chain(
        self, product_code: str, underlying_price: float = 0.0
    ) -> List[Dict]:
        return self._build_chain(underlying_price)

    async def get_option_chain(self, symbol: str) -> List[Dict]:
        return self._build_chain(self._current_price())

    def _build_chain(self, spot: float) -> List[Dict]:
        """Build option chain from CBBO quotes at current minute."""
        minute = self._minute_key()
        today = date.today().isoformat()
        root = self._option_root
        dt_str = today.replace("-", "")[2:]

        chain = []
        seen = set()

        for (m, strike, is_call), (bid, ask) in self._cbbo.items():
            if m != minute:
                continue
            if abs(strike - spot) > self._atm_filter:
                continue

            side = "call" if is_call else "put"
            key = (strike, side)
            if key in seen:
                continue
            seen.add(key)

            # Compute approximate delta from moneyness (for reporting only)
            moneyness = strike / spot if spot > 0 else 1.0
            if is_call:
                delta = round(max(0.01, 1.0 - moneyness) * 2, 4) if moneyness < 1.0 else round(0.5 * (1.0 / moneyness), 4)
            else:
                delta = round(min(-0.01, moneyness - 1.0) * 2, 4) if moneyness > 1.0 else round(-0.5 * moneyness, 4)

            sym_side = "C" if is_call else "P"
            chain.append({
                "strike": strike,
                "option_type": side,
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "expiration": today,
                "symbol": f"{root}{dt_str}{sym_side}{int(strike):08d}",
                "streamer_symbol": f".{root}{dt_str}{sym_side}{int(strike)}",
            })

        return chain

    # -------------------------------------------------------------------------
    # Account / order stubs
    # -------------------------------------------------------------------------

    async def get_account_info(self) -> Dict:
        return {"equity": 100000.0, "buying_power": 50000.0, "cash_balance": 100000.0}

    async def place_order(self, order: Dict) -> Dict:
        logger.info(
            "[REPLAY] Order: %s short=%s long=%s credit=%s qty=%s",
            order.get("order_type"), order.get("short_strike"),
            order.get("long_strike"), order.get("credit"), order.get("quantity"),
        )
        return {"order_id": "REPLAY-001", "status": "filled"}
