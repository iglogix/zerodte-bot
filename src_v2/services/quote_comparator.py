"""QuoteComparator — manual-mode execution comparison via real quotes.

In manual mode no orders are submitted, so there are no fills to compare. This
service instead tracks each MEIC entry's SHORT legs and polls their REAL
(production) bid/ask each interval, logging where the short's ask is relative to
the credit-relative stop trigger. The report then shows where a stop-market
would actually have filled vs the trigger the backtest assumed.

It depends only on a quote source (cache/api) and the ExecutionLog — no DB.
"""
from __future__ import annotations

import logging
from datetime import time as dt_time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_EOD = dt_time(15, 55)


class QuoteComparator:
    def __init__(
        self,
        cache=None,
        api_client=None,
        execution_log=None,
        poll_interval_seconds: int = 15,
        eod_time: dt_time = _EOD,
        clock=None,
    ):
        self._cache = cache
        self._api = api_client
        self._log = execution_log
        self._interval = poll_interval_seconds
        self._eod = eod_time
        self._active: Dict[str, List[Dict[str, Any]]] = {}
        self._running = False
        from src_v2.backtest.sim_clock import WallClock
        self._clock = clock or WallClock()

    def register(self, trade_id: str, payload: Dict[str, Any],
                 put_short_sym: str, call_short_sym: str) -> None:
        """Register a MEIC entry's two short legs for quote tracking."""
        put = payload.get("put_spread", {})
        call = payload.get("call_spread", {})
        self._active[trade_id] = [
            {"symbol": put_short_sym, "role": "short_put",
             "trigger": put.get("stop_trigger"), "credit": put.get("credit"),
             "short_mid": put.get("short_mid")},
            {"symbol": call_short_sym, "role": "short_call",
             "trigger": call.get("stop_trigger"), "credit": call.get("credit"),
             "short_mid": call.get("short_mid")},
        ]
        logger.info("QuoteComparator tracking %s (%s, %s)",
                    trade_id, put_short_sym, call_short_sym)

    async def start(self):
        self._running = True
        logger.info("QuoteComparator started — poll %ds", self._interval)
        while self._running:
            try:
                now = self._clock.now()
                if now.time() >= self._eod:
                    self._active.clear()  # 0DTE legs expire at the close
                else:
                    await self._poll_once()
            except Exception:
                logger.exception("QuoteComparator poll error")
            await self._clock.sleep(self._interval)

    async def stop(self):
        self._running = False

    async def _poll_once(self):
        for trade_id, legs in list(self._active.items()):
            for leg in legs:
                q = await self._get_quote(leg["symbol"])
                if not q:
                    continue
                if self._log is not None:
                    self._log.log_quote(trade_id, leg, q.get("bid"), q.get("ask"))

    async def _get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        if self._cache is not None:
            try:
                q = await self._cache.get_cached_quote(symbol, max_age_seconds=30)
                if q:
                    return q
            except Exception:
                pass
        if self._api is not None:
            try:
                return await self._api.get_quote(symbol)
            except Exception:
                logger.debug("QuoteComparator: no quote for %s", symbol)
        return None
