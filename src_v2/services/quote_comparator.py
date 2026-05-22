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
        tool_handlers=None,
        execution_log=None,
        poll_interval_seconds: int = 15,
        eod_time: dt_time = _EOD,
        clock=None,
    ):
        # tool_handlers gives the same option-chain source the strategy uses to
        # build orders — reliable bid/ask by strike, unlike per-symbol quote
        # lookups which break on OCC-vs-DXLink symbology.
        self._th = tool_handlers
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
            {"symbol": put_short_sym, "role": "short_put", "is_call": False,
             "strike": float(put.get("short_strike")),
             "trigger": put.get("stop_trigger"), "credit": put.get("credit"),
             "short_mid": put.get("short_mid")},
            {"symbol": call_short_sym, "role": "short_call", "is_call": True,
             "strike": float(call.get("short_strike")),
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
        if not self._active or self._th is None:
            return
        by = await self._chain_by_strike()
        if not by:
            return
        for trade_id, legs in list(self._active.items()):
            for leg in legs:
                opt = by.get((leg["strike"], bool(leg["is_call"])))
                if not opt:
                    continue
                bid = opt.get("bid")
                ask = opt.get("ask")
                # Skip invalid/unpriced quotes (e.g. chain enrichment failure
                # returns 0.0) so we never log garbage zeros.
                if not bid or not ask or bid <= 0 or ask <= 0:
                    continue
                if self._log is not None:
                    self._log.log_quote(trade_id, leg, bid, ask)

    async def _chain_by_strike(self) -> Dict[tuple, Dict[str, Any]]:
        """Fetch the 0DTE chain and key it by (strike, is_call)."""
        try:
            chain = await self._th.get_option_chain(expiration="0DTE")
        except Exception:
            logger.debug("QuoteComparator: option chain fetch failed")
            return {}
        out: Dict[tuple, Dict[str, Any]] = {}
        for o in chain or []:
            ot = str(o.get("option_type", "")).upper()
            is_call = ot in ("C", "CALL")
            out[(float(o.get("strike", 0)), is_call)] = o
        return out
