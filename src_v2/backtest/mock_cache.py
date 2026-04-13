"""Mock cache serving CBBO quotes for backtesting."""

from __future__ import annotations
from typing import Any, Dict, Optional


class BacktestCache:
    """Serves option quotes from CBBO store, keyed by simulated time."""

    def __init__(self, cbbo_store, clock):
        self._cbbo = cbbo_store
        self._clock = clock

    async def get_cached_quote(self, symbol: str, max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """Parse OCC symbol, look up CBBO at current minute."""
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        minute = self._clock.now(ET).strftime("%H:%M")
        strike, is_call = self._parse_symbol(symbol)
        if strike is None:
            return None
        q = self._cbbo.lookup(minute, strike, is_call)
        if q is None:
            return None
        bid, ask = q
        return {"bid": bid, "ask": ask, "price": round((bid + ask) / 2, 4)}

    @staticmethod
    def _parse_symbol(sym: str) -> tuple:
        """Extract (strike, is_call) from OCC symbol."""
        try:
            cp_char = sym[12]
            strike_raw = int(sym[13:])
            return float(strike_raw) / 1000.0, cp_char == "C"
        except (IndexError, ValueError):
            return None, None
