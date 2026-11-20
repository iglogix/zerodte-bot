"""
Tool handler implementations for the LLM agent.
Each method corresponds to a tool the LLM can call.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from zerodte.backtest.osi_symbol import format_osi

logger = logging.getLogger(__name__)


class ToolHandlers:
    """Handles all tool calls dispatched by the trading agent."""

    def __init__(
        self,
        cache=None,
        api_client=None,
        db=None,
        mode: Optional[str] = None,
        dry_run: Optional[bool] = None,
        execution_log=None,
        on_entry=None,
        es_roll_days_before_expiry: int = 2,
    ):
        self._execution_log = execution_log
        self._on_entry = on_entry
        # ES is used here only as the ORB *signal* (the traded instrument is
        # SPXW), so we roll the futures contract off the front month a couple of
        # trading days before expiry — its DXLink quotes go silent as liquidity
        # rolls to the next cycle. Strike alignment is irrelevant (options are SPX).
        self._es_roll_days_before_expiry = es_roll_days_before_expiry
        if mode is not None:
            self.mode = mode
        elif dry_run is not None:
            self.mode = "dry_run" if dry_run else "live"
        else:
            self.mode = "dry_run"
        self.cache = cache
        self.api_client = api_client
        self.db = db
        self.dry_run = self.mode == "dry_run"
        self.manual = self.mode == "manual"
        from zerodte.backtest.sim_clock import WallClock
        self._clock = WallClock()
        # ORB state
        self._orb_levels: Optional[Dict] = None
        self._breakout_bias: Optional[str] = None
        self._breakout_price: Optional[float] = None
        self._breakout_time: Optional[datetime] = None

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call by name."""
        handler = getattr(self, tool_name, None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return await handler(**arguments)
        except Exception as exc:
            logger.exception("Tool %s raised an error", tool_name)
            return {"error": str(exc)}

    # -------------------------------------------------------------------------
    # Market data tools
    # -------------------------------------------------------------------------

    def _is_simulated(self) -> bool:
        """True only in dry_run — market data is also simulated."""
        return self.mode == "dry_run"

    def _block_submission(self) -> bool:
        """True in dry_run and manual — orders must not be submitted."""
        return self.mode in ("dry_run", "manual")

    def _manual_order_result(self, tool_name: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Build a manual-mode result: order fully generated but not submitted."""
        logger.info(
            "[MANUAL] Order generated (NOT submitted) — %s: %s",
            tool_name,
            details,
        )
        return {
            **details,
            "manual": True,
            "submitted": False,
            "note": "Order generated from live data but NOT submitted (manual mode)",
        }

    async def get_spx_price(self) -> Dict[str, Any]:
        """Return current SPX price, using cache when valid."""
        try:
            cached = await self.cache.get_cached_quote("SPX", max_age_seconds=60)
            if cached is not None:
                return cached
            data = await self.api_client.get_quote("SPX")
            await self.cache.cache_quote("SPX", data)
            return data
        except Exception as exc:
            return {"error": str(exc)}

    async def get_futures_price(self, symbol: str = "ES") -> Dict[str, Any]:
        """Return current ES futures price via DXLink quote."""
        try:
            if symbol == "ES":
                from es_8am.utils.contract_utils import get_dxlink_symbol
                symbol = get_dxlink_symbol(
                    roll_days_before_expiry=self._es_roll_days_before_expiry
                )
            cached = await self.cache.get_cached_quote(symbol, max_age_seconds=10)
            if cached is not None:
                return cached
            data = await self.api_client.get_future_quote(symbol)
            await self.cache.cache_quote(symbol, data)
            return data
        except Exception as exc:
            return {"error": str(exc)}

    async def get_option_chain(
        self,
        expiration: str = "0DTE",
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        target_credit: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return 0DTE option chain filtered to today's expiry, optionally by delta range.

        If target_credit is provided, results are sorted by proximity to that mid-price
        so the closest match appears first.
        """
        from datetime import date
        chain = await self.cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)

        # Filter to today's expiry (0DTE) if options have expiration data
        has_expiration = any(opt.get("expiration") for opt in chain)
        if has_expiration:
            today = date.today().isoformat()
            chain = [opt for opt in chain if str(opt.get("expiration", "")).startswith(today)]

        # Only keep enriched strikes (bid > 0 means Greeks were fetched)
        enriched = [opt for opt in chain if opt.get("bid", 0) > 0]
        if enriched:
            chain = enriched

        if min_delta is not None or max_delta is not None:
            filtered = []
            for opt in chain:
                abs_delta = abs(opt.get("delta", 0))
                if min_delta is not None and abs_delta < min_delta:
                    continue
                if max_delta is not None and abs_delta > max_delta:
                    continue
                filtered.append(opt)
            chain = filtered

        if target_credit is not None:
            chain = sorted(
                chain,
                key=lambda o: abs(((o.get("bid", 0) + o.get("ask", 0)) / 2) - target_credit),
            )

        return chain

    async def get_5min_candles(self, count: int = 6) -> List[Dict[str, Any]]:
        """Return the last N 5-minute SPX candles."""
        return await self.cache.get_5min_candles("SPX", count=count)

    async def get_daily_closes(self, symbol: str = "SPX", count: int = 50) -> List[float]:
        """Return the last `count` PRIOR-session daily closes, oldest first.

        Today's (partial) candle is excluded so the result is identical at
        every entry window of the day — used by causal day-level gates.
        """
        from datetime import timedelta
        # get_candles sorts ascending and truncates at `count` (keeping the
        # OLDEST), so let from_time bound the window, take everything it holds,
        # and tail the newest ourselves.
        candles = await self.api_client.get_candles(
            symbol, interval="1d",
            from_time=datetime.now() - timedelta(days=int(count * 1.7) + 30),
            count=100_000,
        )
        today = date.today().isoformat()
        return [c["close"] for c in candles
                if c.get("close") and not str(c.get("timestamp", "")).startswith(today)][-count:]

    async def get_expected_move(self) -> Dict[str, Any]:
        """Return the 0DTE expected daily move derived from the ATM straddle price.

        The ATM straddle (call_mid + put_mid at delta ~0.50) is the market's
        direct estimate of expected move — no model assumptions needed.
        It naturally incorporates skew, so put-heavy vol shows up correctly.

        Falls back to IV formula (spx × iv / √252) if straddle price unavailable,
        then to a conservative hardcoded fallback.

        Returns:
            {
                "expected_daily_move": float,   # pts
                "atm_call_mid": float | None,
                "atm_put_mid": float | None,
                "atm_iv": float | None,         # decimal, e.g. 0.18
                "spx_price": float | None,
                "source": "straddle" | "iv_formula" | "fallback",
            }
        """
        import math

        spx_price: Optional[float] = None
        try:
            quote = await self.get_spx_price()
            spx_price = quote.get("price")
        except Exception:
            pass

        def _mid(opt: Dict) -> Optional[float]:
            bid, ask = opt.get("bid", 0), opt.get("ask", 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return None

        # --- Primary: ATM straddle price (delta 0.40-0.60 band) ---
        for min_d, max_d in [(0.40, 0.60), (0.30, 0.60)]:
            try:
                chain = await self.get_option_chain(
                    expiration="0DTE",
                    min_delta=min_d,
                    max_delta=max_d,
                )
                calls = [o for o in chain if o.get("option_type", "").upper() in ("C", "CALL")]
                puts  = [o for o in chain if o.get("option_type", "").upper() in ("P", "PUT")]
                if not calls or not puts:
                    continue

                best_call = min(calls, key=lambda o: abs(abs(o.get("delta", 0)) - 0.50))
                best_put  = min(puts,  key=lambda o: abs(abs(o.get("delta", 0)) - 0.50))

                call_mid = _mid(best_call)
                put_mid  = _mid(best_put)

                if call_mid and put_mid:
                    expected_move = round(call_mid + put_mid, 2)
                    atm_iv = best_call.get("iv") or best_call.get("implied_volatility")
                    logger.info(
                        "Expected move from straddle: call_mid=%.2f + put_mid=%.2f = %.2f pts",
                        call_mid, put_mid, expected_move,
                    )
                    return {
                        "expected_daily_move": expected_move,
                        "atm_call_mid": call_mid,
                        "atm_put_mid": put_mid,
                        "atm_iv": float(atm_iv) if atm_iv else None,
                        "spx_price": spx_price,
                        "source": "straddle",
                    }
            except Exception:
                continue

        # --- Fallback 1: IV formula ---
        for min_d, max_d in [(0.40, 0.60), (0.20, 0.60)]:
            try:
                chain = await self.get_option_chain(
                    expiration="0DTE",
                    min_delta=min_d,
                    max_delta=max_d,
                )
                calls = [o for o in chain if o.get("option_type", "").upper() in ("C", "CALL")]
                if not calls:
                    continue
                best = min(calls, key=lambda o: abs(abs(o.get("delta", 0)) - 0.50))
                iv_raw = best.get("iv") or best.get("implied_volatility")
                if iv_raw and float(iv_raw) > 0:
                    atm_iv = float(iv_raw)
                    price = spx_price or 5500.0
                    expected_move = round(price * atm_iv / math.sqrt(252), 2)
                    logger.warning(
                        "Expected move from IV formula (no straddle prices): iv=%.1f%%, em=%.1f pts",
                        atm_iv * 100, expected_move,
                    )
                    return {
                        "expected_daily_move": expected_move,
                        "atm_call_mid": None,
                        "atm_put_mid": None,
                        "atm_iv": atm_iv,
                        "spx_price": spx_price,
                        "source": "iv_formula",
                    }
            except Exception:
                continue

        # --- Fallback 2: hardcoded conservative IV ---
        if spx_price:
            fallback_iv = 0.15
            expected_move = round(spx_price * fallback_iv / math.sqrt(252), 2)
            logger.warning(
                "Expected move hardcoded fallback: iv=15%%, em=%.1f pts", expected_move,
            )
            return {
                "expected_daily_move": expected_move,
                "atm_call_mid": None,
                "atm_put_mid": None,
                "atm_iv": fallback_iv,
                "spx_price": spx_price,
                "source": "fallback",
            }

        return {
            "expected_daily_move": None,
            "atm_call_mid": None,
            "atm_put_mid": None,
            "atm_iv": None,
            "spx_price": None,
            "source": "unavailable",
        }

    async def get_atm_iv(self) -> Dict[str, Any]:
        """Alias for get_expected_move() — kept for backwards compatibility."""
        return await self.get_expected_move()

    async def get_account_info(self) -> Dict[str, Any]:
        """Return current account info."""
        return await self.cache.get_account_info_with_fallback(max_age_seconds=60)

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return all open iron condor positions."""
        trades = await self.db.get_open_trades()
        return trades if trades else []

    async def get_broker_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Open option positions as the BROKER sees them, or None if unknowable.

        Distinct from get_open_positions(), which returns local *trade rows* and
        carries no per-leg `symbol`. Callers asking "is this leg still open?"
        need the broker's answer; checking a leg symbol against trade rows
        matches nothing and reports every leg as closed.

        Returns None (not []) when there is no broker to ask — dry_run/manual,
        or a client without a positions endpoint — so callers can tell "flat"
        apart from "cannot verify".
        """
        if self._block_submission() or self.api_client is None:
            return None
        getter = getattr(self.api_client, "get_positions", None)
        if getter is None:
            return None
        try:
            return await getter() or []
        except Exception:
            logger.exception("Failed to read broker positions")
            return None

    async def check_buying_power(
        self,
        required_amount: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check available buying power against 50% account limit."""
        account = await self.cache.get_cached_account_info(max_age_seconds=60)
        if account is None:
            account = await self.cache.get_account_info_with_fallback()

        equity = account.get("equity", 0)
        max_allowed = equity * 0.5
        available = account.get("buying_power") or max_allowed

        result = {
            "available": available,
            "max_allowed": max_allowed,
            "equity": equity,
        }

        if required_amount is not None:
            if required_amount > max_allowed:
                result["allowed"] = False
                result["reason"] = (
                    f"Required ${required_amount:,.0f} exceeds 50% buying power limit "
                    f"(${max_allowed:,.0f})"
                )
            else:
                result["allowed"] = True

        return result

    # -------------------------------------------------------------------------
    # Order execution tools
    # -------------------------------------------------------------------------

    async def place_iron_condor(
        self,
        short_call_strike: float,
        long_call_strike: float,
        short_put_strike: float,
        long_put_strike: float,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Place a full 4-leg iron condor. Validates strike order first."""
        if long_call_strike <= short_call_strike:
            return {
                "error": "Invalid strike order: long_call_strike must be > short_call_strike"
            }
        if short_put_strike >= short_call_strike:
            return {
                "error": "Invalid strike order: short_put_strike must be < short_call_strike"
            }
        if long_put_strike >= short_put_strike:
            return {
                "error": "Invalid strike order: long_put_strike must be < short_put_strike"
            }

        if self._block_submission():
            details = {
                "short_call_strike": short_call_strike,
                "long_call_strike": long_call_strike,
                "short_put_strike": short_put_strike,
                "long_put_strike": long_put_strike,
                "quantity": quantity,
            }
            if self.dry_run:
                order_id = f"DRY-IC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                logger.info("[DRY RUN] Simulated iron condor: %s", order_id)
                return {"order_id": order_id, "dry_run": True, "simulated": True, **details}
            return self._manual_order_result("place_iron_condor", details)

        return await self.api_client.place_order({
            "order_type": "iron_condor",
            "short_call_strike": short_call_strike,
            "long_call_strike": long_call_strike,
            "short_put_strike": short_put_strike,
            "long_put_strike": long_put_strike,
            "quantity": quantity,
        })

    async def place_two_spread_ic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Place a MEIC iron condor as two independent credit spreads.

        Submits the put credit spread, then the call credit spread, each at
        the tranche quantity. In dry_run/manual the fills are simulated.
        """
        qty = int(payload.get("quantity", 1))
        put = payload["put_spread"]
        call = payload["call_spread"]

        exp = date.today()
        put_short_sym = format_osi(exp, put["short_strike"], is_call=False)
        call_short_sym = format_osi(exp, call["short_strike"], is_call=True)
        trade_id = datetime.now().strftime("meic-%H%M%S")

        if self.dry_run:
            logger.info("[DRY RUN] Simulated MEIC two-spread IC + stops: %s", trade_id)
            return {
                "order_id": f"DRY-{trade_id}", "trade_id": trade_id,
                "dry_run": True, "simulated": True,
                "quantity": qty, "put_spread": put, "call_spread": call,
                "put_stop_result": {"order_id": "DRY-STOP-P", "symbol": put_short_sym,
                                    "trigger_price": put.get("stop_trigger")},
                "call_stop_result": {"order_id": "DRY-STOP-C", "symbol": call_short_sym,
                                     "trigger_price": call.get("stop_trigger")},
            }

        # Manual + live: record the assumed entry (credits + stop triggers) and
        # register the trade so the quote comparator can track its real quotes.
        if self._execution_log is not None:
            try:
                self._execution_log.log_assumed_entry(
                    trade_id, payload, put_short_sym, call_short_sym)
            except Exception:
                logger.exception("Failed to log assumed entry for execution comparison")
        if self._on_entry is not None:
            try:
                self._on_entry(trade_id, payload, put_short_sym, call_short_sym)
            except Exception:
                logger.exception("on_entry hook raised")

        if self._block_submission():  # manual — orders generated but NOT submitted
            res = self._manual_order_result("place_two_spread_ic", payload)
            res["trade_id"] = trade_id
            return res

        # Live: submit the two credit spreads at their net-credit limit prices.
        put_result = await self.place_credit_spread(
            "put", put["short_strike"], put["long_strike"], qty, credit=put.get("credit"))
        call_result = await self.place_credit_spread(
            "call", call["short_strike"], call["long_strike"], qty, credit=call.get("credit"))

        # Rest a stop-market on each short leg (MEIC: stops on shorts only).
        put_stop_result = None
        call_stop_result = None
        if put.get("stop_trigger") is not None:
            try:
                put_stop_result = await self.api_client.place_stop_market(
                    put_short_sym, put["stop_trigger"], qty)
            except Exception:
                logger.exception("Put short stop submission failed (spread is unprotected)")
        if call.get("stop_trigger") is not None:
            try:
                call_stop_result = await self.api_client.place_stop_market(
                    call_short_sym, call["stop_trigger"], qty)
            except Exception:
                logger.exception("Call short stop submission failed (spread is unprotected)")

        return {
            "order_type": "meic_two_spreads",
            "trade_id": trade_id,
            "quantity": qty,
            "put_result": put_result,
            "call_result": call_result,
            "put_stop_result": put_stop_result,
            "call_stop_result": call_stop_result,
            "put_spread": put,
            "call_spread": call,
        }

    async def place_power_hour_ic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Place a Power Hour iron condor (docs/0DTE-Power-Hour-Strategy.md).

        In ALL modes the assumed fill is persisted to the trades DB (with the
        leg symbols + stop/ratchet params in strategy_data) so PositionMonitor
        runs the combined-strangle stop, the 40% ratchet, and the 15:55 forced
        close — in manual mode this is the paper lifecycle that the 2026-06-11
        review found missing. The comparator hooks mirror place_two_spread_ic.
        """
        qty = int(payload.get("quantity", 1))
        put = payload["put_spread"]
        call = payload["call_spread"]
        tranche = payload.get("tranche") or "T"

        exp = date.today()
        put_short_sym = format_osi(exp, put["short_strike"], is_call=False)
        put_long_sym = format_osi(exp, put["long_strike"], is_call=False)
        call_short_sym = format_osi(exp, call["short_strike"], is_call=True)
        call_long_sym = format_osi(exp, call["long_strike"], is_call=True)
        trade_id = datetime.now().strftime(f"ph-{tranche}-%H%M%S")

        # Execution-comparison hooks (same contract as the MEIC path).
        if self._execution_log is not None:
            try:
                self._execution_log.log_assumed_entry(
                    trade_id, payload, put_short_sym, call_short_sym)
            except Exception:
                logger.exception("Failed to log assumed entry for execution comparison")
        if self._on_entry is not None:
            try:
                self._on_entry(trade_id, payload, put_short_sym, call_short_sym)
            except Exception:
                logger.exception("on_entry hook raised")

        # Persist the assumed fill so the monitor owns the exit lifecycle.
        if self.db is not None:
            strategy_data = {
                "short_put_symbol": put_short_sym,
                "long_put_symbol": put_long_sym,
                "short_call_symbol": call_short_sym,
                "long_call_symbol": call_long_sym,
                "strangle_credit": payload.get("strangle_credit"),
                "stop_dollars": payload.get("stop_amount"),
                "ratchet_add": payload.get("ratchet_add"),
                "put_credit": put.get("credit"),
                "call_credit": call.get("credit"),
                "put_open_qty": qty,
                "call_open_qty": qty,
                "tranche": tranche,
            }
            try:
                await self.db.create_trade({
                    "trade_id": trade_id,
                    "entry_time": datetime.now(),
                    "expiration_date": exp,
                    "underlying": "SPX",
                    "underlying_price_at_entry": float(payload.get("underlying_at_entry") or 0.0),
                    "short_call_strike": call["short_strike"],
                    "long_call_strike": call["long_strike"],
                    "short_put_strike": put["short_strike"],
                    "long_put_strike": put["long_strike"],
                    "call_spread_premium": call.get("credit") or 0.0,
                    "put_spread_premium": put.get("credit") or 0.0,
                    "total_premium": payload.get("total_credit") or 0.0,
                    "call_wing_width": int(call["long_strike"] - call["short_strike"]),
                    "put_wing_width": int(put["short_strike"] - put["long_strike"]),
                    "stop_loss_amount": payload.get("stop_amount"),
                    "status": "open",
                    "strategy_data": json.dumps(strategy_data),
                })
                logger.info("Power Hour trade persisted: %s (strangle %.2f, stop +%.2f, ratchet %s)",
                            trade_id, payload.get("strangle_credit") or 0.0,
                            payload.get("stop_amount") or 0.0, payload.get("ratchet_add"))
            except Exception:
                logger.exception("Failed to persist power_hour trade %s", trade_id)

        if self._block_submission():
            if self.dry_run:
                logger.info("[DRY RUN] Simulated Power Hour IC: %s", trade_id)
                return {"order_id": f"DRY-{trade_id}", "trade_id": trade_id,
                        "dry_run": True, "simulated": True, **payload}
            res = self._manual_order_result("place_power_hour_ic", payload)
            res["trade_id"] = trade_id
            return res

        # LIVE: wings are part of each credit-spread order (defined risk per
        # side); the combined-strangle stop is monitor-driven, so no broker
        # stop-markets are rested here.
        put_result = await self.place_credit_spread(
            "put", put["short_strike"], put["long_strike"], qty, credit=put.get("credit"))
        call_result = await self.place_credit_spread(
            "call", call["short_strike"], call["long_strike"], qty, credit=call.get("credit"))
        return {
            "order_type": "power_hour_ic",
            "trade_id": trade_id,
            "quantity": qty,
            "put_result": put_result,
            "call_result": call_result,
            "put_spread": put,
            "call_spread": call,
        }

    async def place_credit_spread(
        self,
        spread_type: str,
        short_strike: float,
        long_strike: float,
        quantity: int = 1,
        credit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a single credit spread (call or put side) at a net-credit limit."""
        if self._block_submission():
            details = {"spread_type": spread_type, "short_strike": short_strike, "long_strike": long_strike, "quantity": quantity, "credit": credit}
            if self.dry_run:
                return {"order_id": f"DRY-CS-{datetime.now().strftime('%Y%m%d%H%M%S')}", "dry_run": True, "simulated": True}
            return self._manual_order_result("place_credit_spread", details)

        return await self.api_client.place_order({
            "order_type": f"{spread_type}_credit_spread",
            "short_strike": short_strike,
            "long_strike": long_strike,
            "quantity": quantity,
            "credit": credit,
        })

    async def place_stop_order(
        self,
        symbol: str,
        stop_loss_amount: float,
        stop_limit_gap: Optional[int] = None,
        stop_market_gap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place OCO stop-loss: stop-limit + stop-market (gaps from config)."""
        stop_limit_gap = stop_limit_gap if stop_limit_gap is not None else self.stop_limit_gap
        stop_market_gap = stop_market_gap if stop_market_gap is not None else self.stop_market_gap
        if self._block_submission():
            details = {
                "symbol": symbol,
                "stop_loss_amount": stop_loss_amount,
                "stop_limit_gap": stop_limit_gap,
                "stop_market_gap": stop_market_gap,
                "stop_limit_trigger": stop_loss_amount,
                "stop_limit_price": stop_loss_amount + stop_limit_gap,
                "stop_market_trigger": stop_loss_amount + stop_market_gap,
            }
            if self.dry_run:
                return {
                    "stop_limit_order_id": f"DRY-SL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "stop_market_order_id": f"DRY-SM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "dry_run": True,
                    "simulated": True,
                }
            return self._manual_order_result("place_stop_order", details)

        return await self.api_client.place_oco_order({
            "symbol": symbol,
            "stop_loss_amount": stop_loss_amount,
            "stop_limit_gap": stop_limit_gap,
            "stop_market_gap": stop_market_gap,
        })

    async def close_position(
        self,
        symbol: Optional[str] = None,
        trade_id: Optional[str] = None,
        quantity: int = 1,
        order_type: str = "market",
        close_all: bool = False,
        reason: str = "",
        exit_quote: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Close a position or specific leg.

        `reason` is passed by PositionMonitor._close_leg on every take-profit /
        stop-loss close; it is recorded, not acted on. It was previously absent
        from this signature, so those calls raised a TypeError that the caller's
        `except Exception` swallowed — positions never took profit and instead
        rode to the EOD force-close.
        """
        details = {"symbol": symbol, "trade_id": trade_id, "quantity": quantity,
                   "order_type": order_type, "close_all": close_all, "reason": reason}
        if self._block_submission():
            if self.dry_run:
                return {"order_id": f"DRY-CL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "dry_run": True, "simulated": True, **details}
            return self._manual_order_result("close_position", details)

        # close_all carries only a trade_id, so resolve the trade's actual legs
        # here (this layer owns the DB) and close each explicitly. Sending
        # close_trade downstream produced an order with an EMPTY symbol.
        if close_all and trade_id:
            legs = await self._resolve_trade_legs(trade_id)
            if not legs:
                logger.error(
                    "close_all: no legs resolved for trade %s — nothing submitted",
                    trade_id,
                )
                return {"error": "no_legs_resolved", "trade_id": trade_id}
            results = []
            for leg_symbol, leg_qty in legs:
                res = await self.api_client.place_order({
                    "order_type": f"close_{order_type}",
                    "symbol": leg_symbol,
                    "quantity": leg_qty,
                })
                self._log_exit_fill(trade_id, leg_symbol, reason or "eod_close",
                                    leg_qty, None, res)
                # Stamp the leg we asked to close so the caller can book P&L
                # without depending on the broker's symbol spelling.
                results.append({**res, "leg_symbol": leg_symbol}
                               if isinstance(res, dict) else res)
            return {"trade_id": trade_id, "closed_legs": len(results), "results": results}

        if not symbol:
            logger.error("close_position called without a symbol — nothing submitted")
            return {"error": "no_symbol"}

        result = await self.api_client.place_order({
            "order_type": f"close_{order_type}",
            "symbol": symbol,
            "quantity": quantity,
        })
        self._log_exit_fill(trade_id, symbol, reason, quantity, exit_quote, result)
        return result

    def _log_exit_fill(self, trade_id, symbol, reason, quantity, exit_quote, result
                       ) -> None:
        """Record what this leg actually cost to close, vs its quote."""
        if self._execution_log is None:
            return
        try:
            self._execution_log.log_exit_fill(
                trade_id=trade_id or "", symbol=symbol, reason=reason,
                quantity=quantity, exit_quote=exit_quote,
                result=result if isinstance(result, dict) else {},
            )
        except Exception:
            logger.exception("Failed to write exit-fill record for %s", symbol)

    async def _resolve_trade_legs(self, trade_id: str) -> List[tuple]:
        """Open option legs of a persisted trade as [(symbol, quantity), ...].

        Reads the symbols the entry path stored in `strategy_data`, covering
        both the single-spread (ORB directional) and two-sided (IC/MEIC) shapes.
        """
        if self.db is None:
            return []
        try:
            trade = await self.db.get_trade_by_id(trade_id)
        except Exception:
            logger.exception("Failed to load trade %s for close_all", trade_id)
            return []
        if not trade:
            return []

        data = trade.get("strategy_data") or "{}"
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (ValueError, TypeError):
                data = {}

        # DatabaseService flattens the stored strategy_data JSON into the trade
        # dict; other sources hand back the raw JSON under "strategy_data".
        # Read through one merged view so both shapes resolve.
        fields = {**trade, **(data if isinstance(data, dict) else {})}

        legs: List[tuple] = []
        for side in ("put", "call"):
            qty = int(fields.get(f"{side}_open_qty", 0) or 0)
            if qty <= 0:
                continue
            for role in ("short", "long"):
                sym = fields.get(f"{role}_{side}_symbol")
                if sym:
                    legs.append((sym, qty))
        return legs

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order."""
        if self._block_submission():
            if self.dry_run:
                return {"status": "cancelled", "order_id": order_id, "dry_run": True}
            return self._manual_order_result("cancel_order", {"order_id": order_id, "status": "would_cancel"})

        return await self.api_client.cancel_order(order_id)

    async def place_order(self, order_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generic order placement — routes strategy-built payloads to the API."""
        if order_payload.get("order_type") == "meic_two_spreads":
            return await self.place_two_spread_ic(order_payload)
        if order_payload.get("order_type") == "power_hour_ic":
            return await self.place_power_hour_ic(order_payload)
        if order_payload.get("order_type") in ("put_credit_spread", "call_credit_spread"):
            return await self._place_single_spread(order_payload)
        if self._block_submission():
            if self.dry_run:
                order_id = f"DRY-{order_payload.get('order_type', 'ORD').upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                logger.info("[DRY RUN] Simulated order: %s", order_id)
                return {"order_id": order_id, "dry_run": True, "simulated": True, **order_payload}
            return self._manual_order_result("place_order", order_payload)

        return await self.api_client.place_order(order_payload)

    async def _place_single_spread(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Place (or paper-track) a single directional credit spread (PCS/CCS).

        Persists an assumed-fill trade so PositionMonitor owns the exit lifecycle
        (TP / breach / EOD) — the same paper-lifecycle pattern as place_power_hour_ic.
        Used by the ORB directional strategy. In manual/dry_run the order is NOT
        submitted but the trade is tracked; in live the spread is submitted.
        """
        side = payload.get("side") or (
            "put" if payload.get("order_type") == "put_credit_spread" else "call"
        )
        is_call = side == "call"
        qty = int(payload.get("quantity", 1))
        short_k = payload[f"short_{side}_strike"]
        long_k = payload[f"long_{side}_strike"]
        credit = float(payload.get("credit") or 0.0)
        exp = date.today()
        short_sym = format_osi(exp, short_k, is_call=is_call)
        long_sym = format_osi(exp, long_k, is_call=is_call)
        trade_id = datetime.now().strftime(f"orbdir-{side}-%H%M%S")
        width = int(abs(long_k - short_k))

        # Persist the assumed fill so the monitor owns the exit lifecycle. The
        # absent side's strikes are 0 so the strategy's check_exit only evaluates
        # the present spread.
        if self.db is not None:
            strategy_data = {
                f"short_{side}_symbol": short_sym,
                f"long_{side}_symbol": long_sym,
                f"{side}_credit": credit,
                f"{side}_open_qty": qty,
                "side": side,
                "strategy_type": "orb_em_directional",
            }
            try:
                await self.db.create_trade({
                    "trade_id": trade_id,
                    "entry_time": datetime.now(),
                    "expiration_date": exp,
                    "underlying": "SPX",
                    "underlying_price_at_entry": float(payload.get("underlying_at_entry") or 0.0),
                    "short_call_strike": short_k if is_call else 0,
                    "long_call_strike": long_k if is_call else 0,
                    "short_put_strike": 0 if is_call else short_k,
                    "long_put_strike": 0 if is_call else long_k,
                    "call_spread_premium": credit if is_call else 0.0,
                    "put_spread_premium": 0.0 if is_call else credit,
                    "total_premium": credit,
                    "call_wing_width": width if is_call else 0,
                    "put_wing_width": 0 if is_call else width,
                    "status": "open",
                    "strategy_data": json.dumps(strategy_data),
                })
                logger.info(
                    "ORB directional trade persisted: %s (%s spread %.2f credit, short %s)",
                    trade_id, side, credit, short_sym,
                )
            except Exception:
                logger.exception("Failed to persist orb directional trade %s", trade_id)

        if self._block_submission():
            if self.dry_run:
                logger.info("[DRY RUN] Simulated %s: %s", payload.get("order_type"), trade_id)
                return {"order_id": f"DRY-{trade_id}", "trade_id": trade_id,
                        "dry_run": True, "simulated": True, **payload}
            res = self._manual_order_result("place_order", payload)
            res["trade_id"] = trade_id
            return res

        # LIVE: submit the single credit spread, walking the fill ladder.
        started = datetime.now()
        result = await self.place_credit_spread(side, short_k, long_k, qty, credit=credit)
        result["trade_id"] = trade_id

        # Counterfactual record — written for fills AND no-fills, because "could
        # not fill at the modelled price" is the answer to the open question,
        # not a missing data point.
        if self._execution_log is not None:
            try:
                self._execution_log.log_entry_fill(
                    trade_id=trade_id,
                    side=side,
                    strikes={f"short_{side}": short_k, f"long_{side}": long_k},
                    short_quote=payload.get("short_quote") or {},
                    long_quote=payload.get("long_quote") or {},
                    modelled_credit=credit,
                    result=result,
                    elapsed_seconds=(datetime.now() - started).total_seconds(),
                )
            except Exception:
                logger.exception("Failed to write entry-fill record for %s", trade_id)

        await self._reconcile_entry_fill(trade_id, side, credit, result)
        return result

    async def _reconcile_entry_fill(
        self, trade_id: str, side: str, assumed_credit: float, result: Dict[str, Any],
    ) -> None:
        """Replace the assumed entry credit with what the broker actually filled.

        The trade is persisted at the assumed credit *before* submission so the
        monitor can own the exit lifecycle. Against a real book that number is a
        guess: the fill can be worse, or never happen. Producing the real one is
        the entire reason for routing orders to a broker instead of tracking
        them locally, so an unfilled order voids the trade rather than leaving a
        phantom position for the monitor to manage.
        """
        if self.db is None or "filled" not in result:
            return

        fill_credit = result.get("fill_credit")
        if result.get("filled") and fill_credit:
            slippage = assumed_credit - float(fill_credit)
            logger.info(
                "[FILL] %s %s spread filled at $%.2f (assumed $%.2f, slippage $%.2f/spread)",
                trade_id, side, float(fill_credit), assumed_credit, slippage,
            )
            try:
                await self.db.update_trade(trade_id, {
                    f"{'call' if side == 'call' else 'put'}_spread_premium": float(fill_credit),
                    "total_premium": float(fill_credit),
                })
            except Exception:
                logger.exception("Failed to write actual fill credit for %s", trade_id)
            return

        # Not filled — cancel any resting order and void the local trade so the
        # monitor does not manage a position that was never opened.
        order_id = result.get("order_id")
        if order_id:
            try:
                await self.api_client.cancel_order(order_id)
            except Exception:
                logger.exception("Failed to cancel unfilled order %s", order_id)
        logger.warning(
            "[NO-FILL] %s did not fill at $%.2f credit — voiding the local trade",
            trade_id, assumed_credit,
        )
        try:
            await self.db.update_trade(trade_id, {
                "status": "cancelled",
                "close_reason": "entry_not_filled",
            })
        except Exception:
            logger.exception("Failed to void unfilled trade %s", trade_id)

    # -------------------------------------------------------------------------
    # IC-specific methods removed — now in IronCondorStrategy
    # Generic tools below retained for backward compat with LLM agent
    # -------------------------------------------------------------------------

    async def validate_ic_params(
        self,
        short_call_strike: float,
        long_call_strike: float,
        short_put_strike: float,
        long_put_strike: float,
    ) -> Dict[str, Any]:
        """Pre-flight check of IC parameters against live chain data and strategy rules.

        Checks (in order):
          1. Strike ordering
          2. Wing width symmetry
          3. Short strike delta within target range
          4. Per-side credit within target range
          5. Credit equality within threshold
        Returns a summary dict with pass/fail per check and an overall valid flag.
        """
        checks: List[Dict[str, Any]] = []
        all_pass = True

        def fail(name: str, detail: str) -> Dict[str, Any]:
            nonlocal all_pass
            all_pass = False
            return {"check": name, "passed": False, "detail": detail}

        def ok(name: str, detail: str) -> Dict[str, Any]:
            return {"check": name, "passed": True, "detail": detail}

        call_width = round(long_call_strike - short_call_strike, 2)
        put_width = round(short_put_strike - long_put_strike, 2)

        if call_width <= 0:
            checks.append(fail("strike_order_call", f"long_call {long_call_strike} must be > short_call {short_call_strike}"))
        else:
            checks.append(ok("strike_order_call", f"call wing width={call_width}pts"))

        if put_width <= 0:
            checks.append(fail("strike_order_put", f"short_put {short_put_strike} must be > long_put {long_put_strike}"))
        else:
            checks.append(ok("strike_order_put", f"put wing width={put_width}pts"))

        if call_width > 0 and put_width > 0:
            width_diff = abs(call_width - put_width)
            if width_diff > 5:
                checks.append(fail(
                    "wing_width_symmetry",
                    f"call_width={call_width}pts vs put_width={put_width}pts (diff={width_diff}pts, max allowed=5pts)",
                ))
            else:
                checks.append(ok("wing_width_symmetry", f"call_width={call_width}pts, put_width={put_width}pts"))

        chain = await self.get_option_chain(expiration="0DTE")
        chain_by_strike: Dict[tuple, Dict] = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o for o in chain
        }

        def _mid(opt: Optional[Dict]) -> Optional[float]:
            if not opt:
                return None
            bid, ask = opt.get("bid", 0), opt.get("ask", 0)
            if bid <= 0 or ask <= 0:
                return None
            return round((bid + ask) / 2, 4)

        sc_opt = chain_by_strike.get((short_call_strike, "call"))
        sp_opt = chain_by_strike.get((short_put_strike, "put"))

        for label, opt, strike in [("short_call", sc_opt, short_call_strike), ("short_put", sp_opt, short_put_strike)]:
            if opt is None:
                checks.append(fail(f"chain_data_{label}", f"Strike {strike} not found in live chain"))
                continue
            delta = abs(opt.get("delta", 0))
            if delta < self.min_short_delta or delta > self.max_short_delta:
                checks.append(fail(
                    f"delta_{label}",
                    f"delta={delta:.3f} outside target [{self.min_short_delta},{self.max_short_delta}]",
                ))
            else:
                checks.append(ok(f"delta_{label}", f"delta={delta:.3f}"))

        call_credit: Optional[float] = None
        put_credit: Optional[float] = None

        lc_opt = chain_by_strike.get((long_call_strike, "call"))
        lp_opt = chain_by_strike.get((long_put_strike, "put"))

        if sc_opt and lc_opt:
            sc_mid = _mid(sc_opt)
            lc_mid = _mid(lc_opt)
            if sc_mid is not None and lc_mid is not None:
                call_credit = round(sc_mid - lc_mid, 4)
                if call_credit < self.min_side_credit or call_credit > self.max_side_credit:
                    checks.append(fail(
                        "call_credit_range",
                        f"call credit=${call_credit:.2f} outside target [${self.min_side_credit:.2f},${self.max_side_credit:.2f}]",
                    ))
                else:
                    checks.append(ok("call_credit_range", f"call credit=${call_credit:.2f}"))
            else:
                checks.append(fail("call_credit_range", "Cannot compute call credit — zero bid/ask in chain"))
        else:
            checks.append(fail("call_credit_range", f"Missing chain data for call legs ({short_call_strike}/{long_call_strike})"))

        if sp_opt and lp_opt:
            sp_mid = _mid(sp_opt)
            lp_mid = _mid(lp_opt)
            if sp_mid is not None and lp_mid is not None:
                put_credit = round(sp_mid - lp_mid, 4)
                if put_credit < self.min_side_credit or put_credit > self.max_side_credit:
                    checks.append(fail(
                        "put_credit_range",
                        f"put credit=${put_credit:.2f} outside target [${self.min_side_credit:.2f},${self.max_side_credit:.2f}]",
                    ))
                else:
                    checks.append(ok("put_credit_range", f"put credit=${put_credit:.2f}"))
            else:
                checks.append(fail("put_credit_range", "Cannot compute put credit — zero bid/ask in chain"))
        else:
            checks.append(fail("put_credit_range", f"Missing chain data for put legs ({short_put_strike}/{long_put_strike})"))

        if call_credit is not None and put_credit is not None:
            credit_diff = abs(call_credit - put_credit)
            if credit_diff > self.credit_equality_threshold:
                checks.append(fail(
                    "credit_equality",
                    f"|call_credit - put_credit|=${credit_diff:.2f} exceeds threshold=${self.credit_equality_threshold:.2f} "
                    f"(call=${call_credit:.2f}, put=${put_credit:.2f})",
                ))
            else:
                checks.append(ok(
                    "credit_equality",
                    f"|call_credit - put_credit|=${credit_diff:.2f} within threshold=${self.credit_equality_threshold:.2f} "
                    f"(call=${call_credit:.2f}, put=${put_credit:.2f})",
                ))

        return {
            "valid": all_pass,
            "credit_equality_threshold": self.credit_equality_threshold,
            "call_credit": call_credit,
            "put_credit": put_credit,
            "call_width": call_width,
            "put_width": put_width,
            "checks": checks,
        }

    async def get_premium_balance_candidates(
        self,
        target_credit_per_side: Optional[float] = None,
        wing_width: int = 30,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """Search the live 0DTE chain for IC configurations balanced within credit_equality_threshold.

        Tries wing widths of [wing_width-5, wing_width, wing_width+5] and returns
        candidates ranked by |call_credit - put_credit|.
        """
        if target_credit_per_side is None:
            target_credit_per_side = round((self.min_side_credit + self.max_side_credit) / 2, 2)

        # Search ±0.05 beyond the strict validator delta range so boundary strikes
        # aren't missed (e.g. a 0.169 delta call when market is at range edge).
        # The credit range filter below does the real tightening.
        search_min_delta = max(0.01, self.min_short_delta - 0.05)
        search_max_delta = self.max_short_delta + 0.05
        chain = await self.get_option_chain(
            expiration="0DTE",
            min_delta=search_min_delta,
            max_delta=search_max_delta,
        )
        if not chain:
            return {"error": "No chain data available in target delta range", "candidates": []}

        calls = sorted(
            [o for o in chain if o.get("option_type", "").upper() in ("C", "CALL")],
            key=lambda o: o.get("strike", 0),
        )
        puts = sorted(
            [o for o in chain if o.get("option_type", "").upper() in ("P", "PUT")],
            key=lambda o: o.get("strike", 0),
            reverse=True,
        )

        full_chain = await self.get_option_chain(expiration="0DTE")
        chain_by_strike: Dict[tuple, Dict] = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o for o in full_chain
        }

        def _mid(opt: Optional[Dict]) -> Optional[float]:
            if not opt:
                return None
            bid, ask = opt.get("bid", 0), opt.get("ask", 0)
            if bid <= 0 or ask <= 0:
                return None
            return round((bid + ask) / 2, 4)

        widths_to_try = sorted({max(5, wing_width - 5), wing_width, wing_width + 5})
        candidates: List[Dict[str, Any]] = []

        for sc in calls:
            sc_mid = _mid(sc)
            if sc_mid is None:
                continue
            sc_strike = float(sc.get("strike", 0))

            for sp in puts:
                sp_mid = _mid(sp)
                if sp_mid is None:
                    continue
                sp_strike = float(sp.get("strike", 0))

                for w in widths_to_try:
                    lc_strike = sc_strike + w
                    lp_strike = sp_strike - w

                    lc_opt = chain_by_strike.get((lc_strike, "call"))
                    lp_opt = chain_by_strike.get((lp_strike, "put"))
                    if lc_opt is None or lp_opt is None:
                        continue

                    lc_mid = _mid(lc_opt)
                    lp_mid = _mid(lp_opt)
                    if lc_mid is None or lp_mid is None:
                        continue

                    call_credit = round(sc_mid - lc_mid, 4)
                    put_credit = round(sp_mid - lp_mid, 4)

                    if call_credit < self.min_side_credit or call_credit > self.max_side_credit:
                        continue
                    if put_credit < self.min_side_credit or put_credit > self.max_side_credit:
                        continue

                    credit_diff = abs(call_credit - put_credit)
                    if credit_diff > self.credit_equality_threshold:
                        continue

                    candidates.append({
                        "short_call_strike": sc_strike,
                        "long_call_strike": lc_strike,
                        "short_put_strike": sp_strike,
                        "long_put_strike": lp_strike,
                        "wing_width": w,
                        "call_credit": call_credit,
                        "put_credit": put_credit,
                        "total_credit": round(call_credit + put_credit, 4),
                        "credit_diff": round(credit_diff, 4),
                        "call_delta": round(abs(sc.get("delta", 0)), 4),
                        "put_delta": round(abs(sp.get("delta", 0)), 4),
                    })

        candidates.sort(key=lambda c: c["credit_diff"])
        candidates = candidates[:max_results]

        top = candidates[0] if candidates else None
        place_instruction = (
            f"Place iron condor: short_call={top['short_call_strike']}, "
            f"long_call={top['long_call_strike']}, "
            f"short_put={top['short_put_strike']}, "
            f"long_put={top['long_put_strike']}. "
            f"Expected call credit ${top['call_credit']:.2f}, "
            f"put credit ${top['put_credit']:.2f}, "
            f"total ${top['total_credit']:.2f}."
            if top else "No balanced configuration found. Accept lower put credit per strategy rules."
        )

        return {
            "target_credit_per_side": target_credit_per_side,
            "credit_equality_threshold": self.credit_equality_threshold,
            "wing_widths_tried": widths_to_try,
            "candidates_found": len(candidates),
            "top_candidate": top,
            "place_instruction": place_instruction,
            "candidates": candidates,
        }

    async def get_skew_snapshot(self, delta_target: float = 0.12) -> Dict[str, Any]:
        """Return put/call skew metrics at the target delta band.

        Measures:
          - call_iv / put_iv ratio at the target delta
          - call_credit / put_credit ratio at the target delta
          - verdict: 'within_threshold' or 'skew_too_high'
        """
        band = 0.03
        chain = await self.get_option_chain(
            expiration="0DTE",
            min_delta=max(0.01, delta_target - band),
            max_delta=delta_target + band,
        )
        if not chain:
            return {"error": "No chain data in target delta range", "verdict": "unknown"}

        calls = [o for o in chain if o.get("option_type", "").upper() in ("C", "CALL")]
        puts = [o for o in chain if o.get("option_type", "").upper() in ("P", "PUT")]

        def _closest(opts: List[Dict], target: float) -> Optional[Dict]:
            if not opts:
                return None
            return min(opts, key=lambda o: abs(abs(o.get("delta", 0)) - target))

        best_call = _closest(calls, delta_target)
        best_put = _closest(puts, delta_target)

        if best_call is None or best_put is None:
            return {"error": "Insufficient call or put data at target delta", "verdict": "unknown"}

        call_iv = best_call.get("iv") or best_call.get("implied_volatility")
        put_iv = best_put.get("iv") or best_put.get("implied_volatility")

        def _mid(opt: Dict) -> Optional[float]:
            bid, ask = opt.get("bid", 0), opt.get("ask", 0)
            if bid <= 0 or ask <= 0:
                return None
            return round((bid + ask) / 2, 4)

        call_credit = _mid(best_call)
        put_credit = _mid(best_put)

        iv_ratio: Optional[float] = None
        credit_ratio: Optional[float] = None

        if call_iv and put_iv and call_iv > 0:
            iv_ratio = round(put_iv / call_iv, 4)

        if call_credit and put_credit and call_credit > 0:
            credit_ratio = round(put_credit / call_credit, 4)

        skew_ratio = iv_ratio if iv_ratio is not None else credit_ratio
        verdict: str
        if skew_ratio is None:
            verdict = "unknown"
        elif skew_ratio <= self.skew_threshold:
            verdict = "balanced"
        else:
            verdict = "skewed_use_candidates"

        return {
            "delta_target": delta_target,
            "skew_threshold": self.skew_threshold,
            "call_strike": best_call.get("strike"),
            "call_delta": round(abs(best_call.get("delta", 0)), 4),
            "call_iv": call_iv,
            "call_credit": call_credit,
            "put_strike": best_put.get("strike"),
            "put_delta": round(abs(best_put.get("delta", 0)), 4),
            "put_iv": put_iv,
            "put_credit": put_credit,
            "iv_ratio": iv_ratio,
            "credit_ratio": credit_ratio,
            "skew_ratio_used": skew_ratio,
            "verdict": verdict,
            "action_hint": (
                "Proceed with entry. Standard wing widths should work."
                if verdict == "balanced"
                else "Call get_premium_balance_candidates to find adjusted wing widths "
                     "that bring both sides within the credit equality threshold. "
                     "If no balanced config exists, prioritize call-side premium and "
                     "accept a lower put credit per strategy rules."
            ),
        }

    # -------------------------------------------------------------------------
    # ORB (Opening Range Breakout) tools
    # -------------------------------------------------------------------------

    async def get_orb_levels(self, symbol: str, start: str, end: str) -> Dict[str, Any]:
        """Fetch ORB candle and compute high/low/midline. Caches for the session."""
        if self._orb_levels is not None:
            return self._orb_levels
        try:
            # Resolve bare "ES" to DXLink futures symbol (e.g. "/ESM26:XCME").
            # Same early-roll as get_futures_price so the ORB levels and the
            # breakout/retest price polling always reference the SAME contract.
            if symbol == "ES":
                from es_8am.utils.contract_utils import get_dxlink_symbol
                symbol = get_dxlink_symbol(
                    roll_days_before_expiry=self._es_roll_days_before_expiry
                )
            candles = await self.cache.get_candles(symbol, interval="5m", start=start, end=end)
            if not candles:
                return {"error": "no_orb_candles"}
            for c in candles:
                logger.debug(
                    "ORB candle: %s  O=%.2f H=%.2f L=%.2f C=%.2f",
                    c.get("timestamp"), c.get("open", 0), c.get("high", 0),
                    c.get("low", 0), c.get("close", 0),
                )
            high = max(c.get("high", 0) for c in candles)
            low = min(c.get("low", float("inf")) for c in candles)
            midline = round((high + low) / 2, 4)
            range_pts = round(high - low, 4)
            self._orb_levels = {
                "high": high, "low": low, "midline": midline, "range_pts": range_pts,
            }
            return self._orb_levels
        except Exception as exc:
            return {"error": str(exc)}

    async def get_breakout_status(self, current_price: float) -> Dict[str, Any]:
        """Check if price has broken out of ORB range. Latches bias once set."""
        if self._orb_levels is None:
            return {"error": "orb_levels_not_set"}
        if self._breakout_bias is not None:
            return {"status": "breakout", "bias": self._breakout_bias,
                    "price": self._breakout_price}
        high = self._orb_levels["high"]
        low = self._orb_levels["low"]
        if current_price > high:
            self._breakout_bias = "long"
            self._breakout_price = current_price
            self._breakout_time = self._clock.now()
            return {"status": "breakout", "bias": "long", "price": current_price}
        if current_price < low:
            self._breakout_bias = "short"
            self._breakout_price = current_price
            self._breakout_time = self._clock.now()
            return {"status": "breakout", "bias": "short", "price": current_price}
        return {"status": "inside_range", "price": current_price}

    def check_midline_touch(self, price: float, midline: float, tolerance: float) -> bool:
        """Check if price is within tolerance of midline."""
        return abs(price - midline) <= tolerance

    async def get_orb_spot(self, chain: list, ref_price: float, index_symbol: str = "") -> Dict[str, Any]:
        """Derive spot from put-call parity, fallback to API quote."""
        from zerodte.strategies.utils import derive_spot_from_chain
        spot = derive_spot_from_chain(chain, ref_price)
        if spot is not None:
            return {"spot": spot, "source": "put_call_parity"}
        if index_symbol:
            try:
                q = await self.api_client.get_quote(index_symbol)
                price = q.get("price") or q.get("mark") or q.get("last") or q.get("bid")
                if price:
                    return {"spot": float(price), "source": "api_quote"}
            except Exception as exc:
                return {"error": f"spot_derivation_failed: {exc}"}
        return {"error": "spot_derivation_failed: no parity and no index_symbol"}

    def reset_orb_state(self) -> None:
        """Clear all ORB state for new session."""
        self._orb_levels = None
        self._breakout_bias = None
        self._breakout_price = None
        self._breakout_time = None
