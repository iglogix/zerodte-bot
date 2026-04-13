"""
BacktestToolHandlers — serves historical data through the same interface as live ToolHandlers.

Used by BacktestOrchestrator to run the REAL OrbEngine against historical bars + CBBO.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime, time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src_v2.strategies.utils import derive_spot_from_chain

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class BacktestToolHandlers:
    """Serves historical ES bars and CBBO quotes through the live ToolHandlers interface."""

    def __init__(self, bars: List[Dict], cbbo_store, clock, db, session_date,
                 slippage: float = 0.10):
        """
        Args:
            bars: day's 1-min ES bars, each with ts_et, open, high, low, close, volume
            cbbo_store: CbboStore with day already loaded
            clock: SimClock for current simulated time
            db: BacktestDb for trade storage
            session_date: date object for this trading day
            slippage: per-spread slippage deduction (default 0.10)
        """
        self._bars = bars
        self._cbbo = cbbo_store
        self._clock = clock
        self._db = db
        self._session_date = session_date
        self._slippage = slippage
        self._bar_idx = 0

        # ORB state (same as live ToolHandlers)
        self._orb_levels: Optional[Dict] = None
        self._breakout_bias: Optional[str] = None
        self._breakout_price: Optional[float] = None
        self._breakout_time: Optional[datetime] = None

        # Exit tracking: symbol -> {bid, ask, time} at close
        self._exit_log: Dict[str, Dict] = {}

    @property
    def current_bar(self) -> Optional[Dict]:
        if 0 <= self._bar_idx < len(self._bars):
            return self._bars[self._bar_idx]
        return None

    def advance_bar(self) -> bool:
        """Advance to next bar. Returns False if no more bars."""
        if self._bar_idx < len(self._bars) - 1:
            self._bar_idx += 1
            self._clock.advance_to(self._bars[self._bar_idx]["ts_et"])
            return True
        return False

    def has_bars_remaining(self) -> bool:
        return self._bar_idx < len(self._bars) - 1

    def set_bar_index(self, idx: int) -> None:
        """Set bar pointer to specific index (used by orchestrator)."""
        self._bar_idx = min(idx, len(self._bars) - 1)

    # ------------------------------------------------------------------
    # ORB state (same logic as live ToolHandlers)
    # ------------------------------------------------------------------

    def reset_orb_state(self) -> None:
        self._orb_levels = None
        self._breakout_bias = None
        self._breakout_price = None
        self._breakout_time = None

    async def get_orb_levels(self, symbol: str, start: str, end: str) -> Dict[str, Any]:
        """Compute ORB from historical bars in [start, end) window."""
        start_t = time(*[int(x) for x in start.split(":")])
        end_t = time(*[int(x) for x in end.split(":")])

        orb_bars = [b for b in self._bars if start_t <= b["ts_et"].time() < end_t]
        if not orb_bars:
            return {"error": "no_orb_candles"}

        high = max(b["high"] for b in orb_bars)
        low = min(b["low"] for b in orb_bars)
        midline = round((high + low) / 2, 4)
        range_pts = round(high - low, 4)

        self._orb_levels = {
            "high": high, "low": low, "midline": midline, "range_pts": range_pts,
        }
        return self._orb_levels

    async def get_breakout_status(self, current_price: float) -> Dict[str, Any]:
        """Same stateful logic as live ToolHandlers — latches bias on first breakout."""
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
        return abs(price - midline) <= tolerance

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    async def get_futures_price(self, symbol: str = "ES") -> Dict[str, Any]:
        bar = self.current_bar
        if bar is None:
            return {"error": "no_bar_data"}
        return {"price": bar["close"]}

    async def get_spx_price(self) -> Dict[str, Any]:
        """Return CBBO-derived spot, or bar close as fallback."""
        bar = self.current_bar
        if bar is None:
            return {"error": "no_bar_data"}
        minute = bar["ts_et"].strftime("%H:%M")
        spot = self._cbbo.derive_spot(minute, bar["close"])
        if spot is not None:
            return {"price": spot}
        return {"price": bar["close"]}

    async def get_5min_candles(self, count: int = 10) -> List[Dict[str, Any]]:
        """Aggregate recent 1-min bars into 5-min candles."""
        end_idx = self._bar_idx + 1
        start_idx = max(0, end_idx - count * 5)
        recent = self._bars[start_idx:end_idx]
        if not recent:
            return []

        candles = []
        for i in range(0, len(recent), 5):
            chunk = recent[i:i + 5]
            if not chunk:
                break
            candles.append({
                "timestamp": chunk[0]["ts_et"].isoformat(),
                "open": chunk[0]["open"],
                "high": max(b["high"] for b in chunk),
                "low": min(b["low"] for b in chunk),
                "close": chunk[-1]["close"],
            })
        return candles[-count:]

    # ------------------------------------------------------------------
    # Option chain
    # ------------------------------------------------------------------

    async def get_option_chain(self, expiration: str = "0DTE", **kwargs) -> List[Dict[str, Any]]:
        """Build option chain from CBBO at current minute."""
        bar = self.current_bar
        if bar is None:
            return []
        minute = bar["ts_et"].strftime("%H:%M")

        # Derive spot for filtering range
        spot = self._cbbo.derive_spot(minute, bar["close"])
        if spot is None:
            spot = bar["close"]

        chain = []
        for strike in sorted(self._cbbo.available_strikes):
            if abs(strike - spot) > 200:
                continue
            for is_call in [False, True]:
                q = self._cbbo.lookup(minute, strike, is_call)
                if q is None:
                    continue
                bid, ask = q
                if bid <= 0 and ask <= 0:
                    continue
                sym = self._make_symbol(strike, is_call)
                chain.append({
                    "strike": strike,
                    "option_type": "call" if is_call else "put",
                    "bid": bid,
                    "ask": ask,
                    "symbol": sym,
                    "delta": 0,
                    "expiration": self._session_date.isoformat(),
                })
        return chain

    async def get_orb_spot(self, chain: list, ref_price: float, index_symbol: str = "") -> Dict[str, Any]:
        """Derive spot from put-call parity — same util as live."""
        # Chain already uses "call"/"put" option_type — pass directly
        spot = derive_spot_from_chain(chain, ref_price)
        if spot is not None:
            return {"spot": spot, "source": "put_call_parity"}
        return {"spot": ref_price, "source": "bar_fallback"}

    # ------------------------------------------------------------------
    # Order execution (simulated)
    # ------------------------------------------------------------------

    async def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record trade in mock DB with full leg symbols and per-side credits."""
        trade_id = f"BT-{self._session_date.isoformat()}-{uuid.uuid4().hex[:6]}"
        bar = self.current_bar
        minute = bar["ts_et"].strftime("%H:%M") if bar else "00:00"
        entry_price = bar["close"] if bar else 0.0

        # Build leg symbols from strikes
        sp_k = payload.get("short_put_strike", 0)
        lp_k = payload.get("long_put_strike", 0)
        sc_k = payload.get("short_call_strike", 0)
        lc_k = payload.get("long_call_strike", 0)

        sp_sym = self._make_symbol(sp_k, False)
        lp_sym = self._make_symbol(lp_k, False)
        sc_sym = self._make_symbol(sc_k, True)
        lc_sym = self._make_symbol(lc_k, True)

        # Compute per-side credits from CBBO at entry time
        def _mid(strike, is_call):
            q = self._cbbo.lookup(minute, strike, is_call)
            return (q[0] + q[1]) / 2 if q else 0.0

        put_credit = round(_mid(sp_k, False) - _mid(lp_k, False) - self._slippage, 2)
        call_credit = round(_mid(sc_k, True) - _mid(lc_k, True) - self._slippage, 2)

        trade = {
            "trade_id": trade_id,
            "status": "open",
            "entry_time": self._clock.now(ET),
            "entry_price": entry_price,
            "short_put_symbol": sp_sym,
            "long_put_symbol": lp_sym,
            "short_call_symbol": sc_sym,
            "long_call_symbol": lc_sym,
            "short_put_strike": sp_k,
            "long_put_strike": lp_k,
            "short_call_strike": sc_k,
            "long_call_strike": lc_k,
            "put_credit": max(put_credit, 0.0),
            "call_credit": max(call_credit, 0.0),
            **payload,
        }
        await self._db.add_trade(trade)
        return {"order_id": trade_id, "status": "filled", "trade_id": trade_id}

    async def close_position(
        self,
        symbol: Optional[str] = None,
        trade_id: Optional[str] = None,
        close_all: bool = False,
        order_type: str = "market",
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Update trade status in mock DB and record exit quotes."""
        now = self._clock.now(ET)
        if close_all and trade_id:
            # EOD close: record exit quotes for all legs in this trade
            trades = await self._db.get_open_trades()
            for t in trades:
                if t.get("trade_id") == trade_id:
                    for key in ("short_put_symbol", "long_put_symbol",
                                "short_call_symbol", "long_call_symbol"):
                        leg_sym = t.get(key)
                        if leg_sym and leg_sym not in self._exit_log:
                            self._record_exit_quote(leg_sym, now)
                    break
            await self._db.update_trade(trade_id, {
                "status": "closed",
                "exit_time": now,
            })
            return {"status": "closed", "trade_id": trade_id}
        if symbol:
            self._record_exit_quote(symbol, now)
        return {"status": "closed", "symbol": symbol}

    def _record_exit_quote(self, symbol: str, now: datetime) -> None:
        """Look up CBBO at current minute and store exit mark for a leg."""
        minute = now.strftime("%H:%M")
        strike, is_call = self._parse_symbol(symbol)
        if strike is not None:
            q = self._cbbo.lookup(minute, strike, is_call)
            if q:
                self._exit_log[symbol] = {"bid": q[0], "ask": q[1], "time": now}

    @staticmethod
    def _parse_symbol(sym: str):
        """Extract (strike, is_call) from OCC symbol."""
        try:
            cp_char = sym[12]
            strike_raw = int(sym[13:])
            return float(strike_raw) / 1000.0, cp_char == "C"
        except (IndexError, ValueError):
            return None, None

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        return await self._db.get_open_trades()

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return {"status": "cancelled", "order_id": order_id}

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def _make_symbol(self, strike: float, is_call: bool) -> str:
        root = "SPXW".ljust(6)
        dt_str = self._session_date.strftime("%y%m%d")
        cp = "C" if is_call else "P"
        strike_str = f"{int(strike * 1000):08d}"
        return f"{root}{dt_str}{cp}{strike_str}"
