"""Alpaca paper execution client — the fills half of the hybrid broker setup.

Alpaca has no futures, so it cannot supply the ES opening-range signal this
strategy trades off. It *can* trade SPXW 0DTE multi-leg spreads in paper
(index options landed 2026-07-23, options level 3 required). So the split is:

    market data  -> TastyTrade / DXLink   (ES ORB candles, SPXW chain, spot)
    execution    -> Alpaca paper          (place / cancel / fill reconciliation)

`AlpacaExecutionClient` is a decorator over `TastyTradeClient`: every read
delegates through, and only the writes are re-implemented against Alpaca. That
keeps `CachedMarketData`, `ToolHandlers` and `PositionMonitor` untouched — they
all talk to one injected `api_client`.

Safety property worth preserving: reads fall through via `__getattr__`, but
EVERY known write is overridden explicitly. A write method added to
`TastyTradeClient` later must not silently reach a real broker from here, so
the unsupported ones raise rather than delegate.

Conventions this file pins:
  - Alpaca wants the compact OCC symbol (`SPXW260818P07620000`); `format_osi`
    emits TastyTrade's space-padded 21-char form (`SPXW  260818P07620000`).
  - On a multi-leg order a net CREDIT is a NEGATIVE limit price.
  - `qty` is the number of spreads; `ratio_qty` is the per-leg ratio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

import aiohttp

from src_v2.backtest.osi_symbol import format_osi

logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
TICK = 0.05


# ---------------------------------------------------------------------------
# Pure translation helpers (unit-tested without network)
# ---------------------------------------------------------------------------

def to_alpaca_symbol(osi: str) -> str:
    """TastyTrade's space-padded 21-char OSI -> Alpaca's compact OCC symbol."""
    return (osi or "").replace(" ", "")


def _round_to_tick(price: float) -> float:
    return round(round(float(price) / TICK) * TICK, 2)


def _credit_from_order(data: Dict[str, Any]) -> Optional[float]:
    """Net credit actually filled. Alpaca reports a credit as a negative avg."""
    avg = (data or {}).get("filled_avg_price")
    if avg in (None, ""):
        return None
    try:
        return abs(float(avg))
    except (TypeError, ValueError):
        return None


def build_credit_spread_order(
    order: Dict[str, Any], expiry: Optional[date] = None
) -> Dict[str, Any]:
    """`{put,call}_credit_spread` payload -> Alpaca multi-leg order.

    The net credit becomes a negative limit price, rounded to the 0.05 tick so
    the limit matches what the TastyTrade path would have sent.
    """
    expiry = expiry or date.today()
    qty = int(order.get("quantity", 1) or 1)
    is_call = "call" in order.get("order_type", "")

    if order.get("short_symbol") and order.get("long_symbol"):
        short_sym, long_sym = order["short_symbol"], order["long_symbol"]
    else:
        short_sym = format_osi(expiry, order["short_strike"], is_call=is_call)
        long_sym = format_osi(expiry, order["long_strike"], is_call=is_call)

    credit = _round_to_tick(order.get("credit") or 0.0)
    if credit <= 0:
        raise ValueError(
            f"refusing to submit a credit spread with credit={credit} — a zero "
            "limit would rest forever or fill at any price"
        )

    return {
        "order_class": "mleg",
        "qty": str(qty),
        "type": "limit",
        "time_in_force": "day",
        "limit_price": f"{-credit:.2f}",  # negative == net credit
        "legs": [
            {
                "symbol": to_alpaca_symbol(short_sym),
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_open",
            },
            {
                "symbol": to_alpaca_symbol(long_sym),
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_open",
            },
        ],
    }


def build_fill_ladder(
    credit: float, base_slippage: float, rungs: List[float]
) -> List[float]:
    """Net-credit limits to walk, from the modelled price down to the floor.

    The backtest prices entry at mid − `options.slippage_per_spread` ($0.10), so
    `credit` already embeds that concession and rung 1 reproduces exactly what
    the walk-forward study measured. Later rungs give up more, and the last one
    is the validated ceiling: the WF fill-stress matrix holds at $0.25/spread
    (Sharpe 2.24 with full exit crossing) and the edge disappears near $0.50.

    Spending the budget on PRICE rather than TIME is deliberate. A credit-spread
    limit left resting fills when the spread's value rises — i.e. when the
    underlying moves toward the short strike — so waiting fills you on the days
    the trade immediately goes against you and misses the days it works. That
    adverse selection is not in the model; a bounded price concession is.

    Returns strictly decreasing, tick-rounded, positive limits.
    """
    if not rungs:
        return [_round_to_tick(credit)] if credit > 0 else []

    mid_credit = float(credit) + float(base_slippage)
    ladder: List[float] = []
    for slip in rungs:
        limit = _round_to_tick(mid_credit - float(slip))
        if limit <= 0:
            continue
        if ladder and limit >= ladder[-1]:
            continue  # collapsed onto the same tick, or went backwards
        ladder.append(limit)
    return ladder


def build_close_order(order: Dict[str, Any], held_qty: float) -> Dict[str, Any]:
    """`close_market` / `close_limit` payload -> single-leg closing order.

    The closing SIDE depends on which way the leg is held, and a credit spread
    holds one of each: the short leg is bought to close, the long leg is SOLD to
    close. Hardcoding buy-to-close closes shorts fine and makes every long-leg
    close fail with 422 "position intent mismatch" — which silently breaks
    take-profit, breach exits and the EOD force-close, since all three close
    both legs.

    `held_qty` comes from the broker's own position (negative = short), so the
    direction can never disagree with reality. `position_intent` is sent
    explicitly so a double-close is rejected rather than silently OPENING a new
    position in the opposite direction.
    """
    symbol = to_alpaca_symbol(order.get("symbol", ""))
    if not symbol:
        raise ValueError("close order has no symbol — cannot close an unnamed leg")
    if not held_qty:
        raise ValueError(f"no open position in {symbol} — nothing to close")

    is_short = held_qty < 0
    want = int(order.get("quantity", 1) or 1)
    return {
        "symbol": symbol,
        # Never try to close more than is actually held.
        "qty": str(min(want, int(abs(held_qty)))),
        "side": "buy" if is_short else "sell",
        "type": "market",
        "time_in_force": "day",
        "position_intent": "buy_to_close" if is_short else "sell_to_close",
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AlpacaExecutionClient:
    """Writes go to Alpaca; every read delegates to the wrapped data client."""

    #: Writes that must never fall through to the wrapped (real-money) client.
    _UNSUPPORTED_WRITES = ("place_oco_order", "place_stop_market")

    def __init__(
        self,
        data_client,
        key_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: str = PAPER_URL,
        fill_timeout_seconds: float = 30.0,
        entry_fill: Optional[Dict[str, Any]] = None,
    ):
        entry_fill = entry_fill or {}
        self._entry_rungs = list(entry_fill.get("rungs") or [])
        self._rung_seconds = float(entry_fill.get("rung_seconds", 20.0))
        self._entry_base_slippage = float(entry_fill.get("base_slippage", 0.10))
        self._data = data_client
        self._key = key_id or os.environ.get("APCA_API_KEY_ID", "")
        self._secret = secret_key or os.environ.get("APCA_API_SECRET_KEY", "")
        self.base_url = (base_url or PAPER_URL).rstrip("/")
        self.fill_timeout_seconds = fill_timeout_seconds
        self._http: Optional[aiohttp.ClientSession] = None
        self._account_number: Optional[str] = None

        if not self._key or not self._secret:
            raise ValueError(
                "Alpaca credentials missing — set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY (see .env)"
            )
        if "paper" not in self.base_url:
            logger.warning("Alpaca base_url is NOT a paper endpoint: %s", self.base_url)

    # -- delegation -------------------------------------------------------

    def __getattr__(self, name: str):
        """Reads fall through to the wrapped market-data client.

        Writes are overridden as real methods above, so they never reach here.
        Anything in _UNSUPPORTED_WRITES is refused rather than delegated.
        """
        if name in self._UNSUPPORTED_WRITES:
            raise NotImplementedError(
                f"{name} is not implemented for Alpaca execution and must not "
                "fall through to the market-data broker"
            )
        return getattr(self._data, name)

    # -- lifecycle --------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Content-Type": "application/json",
        }

    async def connect(self):
        """Connect the data client, then verify the Alpaca account is usable."""
        await self._data.connect()
        self._http = aiohttp.ClientSession()
        info = await self.get_account_info()
        level = info.get("options_trading_level", 0)
        if level < 3:
            raise RuntimeError(
                f"Alpaca account options_trading_level={level}; multi-leg spreads "
                "need level 3"
            )
        logger.info(
            "Alpaca execution ready — account=%s equity=$%.0f options_level=%s (%s)",
            info.get("account_number"), info.get("equity", 0), level, self.base_url,
        )

    async def disconnect(self):
        if self._http and not self._http.closed:
            await self._http.close()
        await self._data.disconnect()

    async def _request(self, method: str, path: str, **kw) -> Any:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        async with self._http.request(
            method, f"{self.base_url}{path}", headers=self._headers(), **kw
        ) as resp:
            body = await resp.text()
            if resp.status >= 400:
                logger.error("Alpaca %s %s -> %s: %s", method, path, resp.status, body)
                resp.raise_for_status()
            return json.loads(body) if body else {}

    # -- account ----------------------------------------------------------

    async def get_account_info(self) -> Dict[str, Any]:
        data = await self._request("GET", "/v2/account")
        self._account_number = data.get("account_number")
        return {
            "account_number": data.get("account_number"),
            "equity": float(data.get("equity", 0) or 0),
            "buying_power": float(data.get("options_buying_power", 0) or 0),
            "cash_balance": float(data.get("cash", 0) or 0),
            "day_trading_buying_power": float(data.get("daytrading_buying_power", 0) or 0),
            "maintenance_requirement": float(data.get("maintenance_margin", 0) or 0),
            "options_trading_level": int(data.get("options_trading_level", 0) or 0),
        }

    async def get_account_number(self) -> str:
        if self._account_number is None:
            await self.get_account_info()
        return self._account_number or ""

    # -- writes -----------------------------------------------------------

    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        order_type = order.get("order_type", "")
        if order_type in ("call_credit_spread", "put_credit_spread"):
            return await self._place_credit_spread(order)
        if order_type.startswith("close_"):
            return await self._place_close(order)
        raise ValueError(f"Unknown order_type for Alpaca execution: {order_type!r}")

    async def place_oco_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "OCO stops are not implemented for Alpaca execution "
            "(this strategy runs sl_mult_of_credit=0, i.e. no stop orders)"
        )

    async def place_stop_market(self, *a, **kw) -> Dict[str, Any]:
        raise NotImplementedError("stop-market is not implemented for Alpaca execution")

    async def _place_credit_spread(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Walk the fill ladder: submit, wait, cancel, concede, repeat.

        Bounded in price (the last rung is the validated $0.25/spread ceiling)
        rather than in time, and every rung's outcome is reported so the run
        measures what it actually cost to get filled.
        """
        modelled_credit = float(order.get("credit") or 0.0)
        ladder = build_fill_ladder(
            modelled_credit, self._entry_base_slippage, self._entry_rungs
        )
        if not ladder:
            raise ValueError(f"no valid limit for credit={modelled_credit}")

        attempts: List[Dict[str, Any]] = []
        for i, limit in enumerate(ladder):
            is_last = i == len(ladder) - 1
            payload = build_credit_spread_order({**order, "credit": limit})
            logger.info(
                "Alpaca mleg credit spread rung %d/%d @ $%.2f credit: %s",
                i + 1, len(ladder), limit, json.dumps(payload),
            )
            data = await self._request("POST", "/v2/orders", json=payload)
            order_id = str(data.get("id", ""))
            timeout = self.fill_timeout_seconds if is_last else self._rung_seconds
            fill = await self.wait_for_fill(order_id, timeout=timeout)
            attempts.append({
                "rung": i + 1, "limit_credit": limit, "order_id": order_id,
                "status": fill.get("status"),
            })

            if fill.get("filled"):
                return self._fill_result(order_id, i + 1, limit, modelled_credit,
                                         fill.get("fill_credit"), attempts, True)

            # Cancel before conceding. If the cancel lost the race the order is
            # already working against us as a real position — laddering on would
            # open a SECOND spread.
            cancel = await self.cancel_order(order_id)
            if cancel.get("status") == "filled":
                late_credit = _credit_from_order(await self.get_order(order_id))
                logger.warning(
                    "Rung %d filled during cancel — stopping the ladder at $%s",
                    i + 1, late_credit,
                )
                return self._fill_result(order_id, i + 1, limit, modelled_credit,
                                         late_credit, attempts, True)

        logger.warning(
            "No fill after %d rungs down to $%.2f (modelled $%.2f) — entry abandoned",
            len(ladder), ladder[-1], modelled_credit,
        )
        return self._fill_result(attempts[-1]["order_id"], len(ladder), ladder[-1],
                                 modelled_credit, None, attempts, False)

    def _fill_result(self, order_id, rung, limit, modelled, fill_credit,
                     attempts, filled) -> Dict[str, Any]:
        """Normalized entry outcome, carrying the slippage actually paid."""
        mid_credit = modelled + self._entry_base_slippage
        slippage = (mid_credit - float(fill_credit)) if fill_credit else None
        if filled and fill_credit:
            logger.info(
                "Alpaca entry FILLED on rung %d at $%.2f credit "
                "(mid $%.2f, modelled $%.2f, slippage $%.2f/spread)",
                rung, float(fill_credit), mid_credit, modelled, slippage,
            )
        elif filled:
            # Terminal-filled but the broker has not published a price yet.
            # Report the fill; the limit is the worst case it can have been.
            logger.warning(
                "Alpaca entry filled on rung %d but no fill price reported — "
                "assuming the rung limit $%.2f for slippage accounting",
                rung, limit,
            )
            fill_credit = limit
            slippage = mid_credit - limit
        return {
            "order_id": order_id,
            "status": "filled" if filled else "unfilled",
            "filled": filled,
            "rung": rung,
            "requested_credit": limit,
            "modelled_credit": modelled,
            "mid_credit": round(mid_credit, 2),
            "fill_credit": fill_credit,
            "slippage_per_spread": round(slippage, 2) if slippage is not None else None,
            "attempts": attempts,
            "broker": "alpaca",
        }

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """One open position, or None if flat. 404 is the normal 'flat' answer."""
        try:
            return await self._request(
                "GET", f"/v2/positions/{to_alpaca_symbol(symbol)}")
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                return None
            raise

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Open positions at the EXECUTION broker.

        Overridden rather than left to __getattr__: delegating this to the
        wrapped market-data client would answer with the TastyTrade account,
        which holds none of these positions.
        """
        return await self._request("GET", "/v2/positions") or []

    async def _place_close(self, order: Dict[str, Any]) -> Dict[str, Any]:
        symbol = order.get("symbol", "")
        position = await self.get_position(symbol)
        if position is None:
            # Already flat — a no-op, not an error. Exit paths can legitimately
            # fire twice (e.g. take-profit then EOD sweep).
            logger.info("Close skipped — no open position in %s", symbol)
            return {"symbol": to_alpaca_symbol(symbol), "status": "no_position",
                    "filled": False, "skipped": True, "broker": "alpaca"}

        payload = build_close_order(order, held_qty=float(position.get("qty", 0)))
        logger.info("Alpaca close: %s", json.dumps(payload))
        data = await self._request("POST", "/v2/orders", json=payload)
        order_id = str(data.get("id", ""))
        fill = await self.wait_for_fill(order_id)
        return {
            "order_id": order_id,
            "symbol": payload["symbol"],
            "status": fill.get("status"),
            "filled": fill.get("filled", False),
            "fill_price": fill.get("fill_price"),
            "broker": "alpaca",
        }

    async def cancel_order(self, order_id: str, confirm_timeout: float = 5.0
                           ) -> Dict[str, Any]:
        """Cancel an order and CONFIRM it reached a terminal state.

        Alpaca processes cancels asynchronously — the DELETE returns before the
        order is actually dead. Returning "cancelled" on the DELETE alone is
        unsafe here: the entry path voids the local trade when an order does not
        fill, so an order that slips through and fills anyway would leave a live
        position nothing is monitoring. Poll until it is genuinely terminal and
        report loudly if it filled instead.
        """
        await self._request("DELETE", f"/v2/orders/{order_id}")
        terminal = {"canceled", "cancelled", "filled", "rejected", "expired"}
        waited, status = 0.0, ""
        while waited < confirm_timeout:
            data = await self.get_order(order_id)
            status = (data.get("status") or "").lower()
            if status in terminal:
                break
            await asyncio.sleep(0.5)
            waited += 0.5

        if status == "filled":
            logger.error(
                "Order %s FILLED despite the cancel — there is now an open "
                "position that the local trade record does not cover",
                order_id,
            )
        elif status not in terminal:
            logger.warning(
                "Order %s not confirmed terminal within %.0fs (status=%s)",
                order_id, confirm_timeout, status or "unknown",
            )
        return {"status": status or "cancelled", "order_id": order_id,
                "confirmed": status in terminal}

    # -- fill reconciliation ---------------------------------------------

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v2/orders/{order_id}")

    async def wait_for_fill(
        self, order_id: str, timeout: Optional[float] = None, poll: float = 1.0
    ) -> Dict[str, Any]:
        """Poll an order to resolve what ACTUALLY filled.

        The strategy persists its trade at the *assumed* credit before the order
        is sent. Against a real book the fill may differ or never happen, so the
        caller needs the real number — that reconciliation is the whole point of
        running on a broker rather than paper-tracking locally.

        Returns filled=False on timeout WITHOUT cancelling; the caller decides.
        """
        timeout = self.fill_timeout_seconds if timeout is None else timeout
        terminal = {"filled", "canceled", "cancelled", "rejected", "expired"}
        waited, data = 0.0, {}
        while True:  # always poll at least once, so timeout=0 is a status read
            data = await self.get_order(order_id)
            status = (data.get("status") or "").lower()
            if status in terminal or waited >= timeout:
                break
            await asyncio.sleep(poll)
            waited += poll

        status = (data.get("status") or "").lower()
        filled = status == "filled"
        avg = data.get("filled_avg_price")
        fill_price = float(avg) if avg not in (None, "") else None
        # A filled credit spread reports a negative average price; the strategy
        # thinks in positive credit.
        fill_credit = abs(fill_price) if fill_price is not None else None

        if not filled:
            logger.warning(
                "Alpaca order %s did not fill within %.0fs (status=%s)",
                order_id, timeout, status or "unknown",
            )
        else:
            logger.info("Alpaca order %s FILLED at %s", order_id, fill_price)

        return {
            "status": status,
            "filled": filled,
            "fill_price": fill_price,
            "fill_credit": fill_credit,
            "raw": data,
        }

    # -- quotes (cross-check only) ---------------------------------------

    async def get_alpaca_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Alpaca's own view of a set of contracts, for fill cross-checking.

        NOT used for strike selection — TastyTrade remains the quote source of
        record. Note the chain-snapshot endpoint ignores `expiration_date` and
        only returns the SPX monthly root, so SPXW dailies must be requested by
        explicit symbol here.
        """
        if not symbols:
            return {}
        syms = ",".join(to_alpaca_symbol(s) for s in symbols)
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        url = f"{DATA_URL}/v1beta1/options/quotes/latest?symbols={syms}"
        async with self._http.get(url, headers=self._headers()) as resp:
            if resp.status >= 400:
                logger.warning("Alpaca quote fetch failed (%s)", resp.status)
                return {}
            return (await resp.json()).get("quotes", {})
