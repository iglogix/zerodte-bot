"""
Iron Fly Backtest Engine — simulates 0DTE Iron Fly trades on SPX.

Per-day lifecycle:
  1. Build ORB (9:30–10:00 ET) — high/low/range
  2. Macro filter — skip corrective days (large candles, high VIX)
  3. Entry at 10:00 bar — price must be within ORB
  4. Exit simulation — TP (fixed $), SL (underlying crosses wing), or EOD
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src_8am.config_keys import require
from src_8am.backtest.strategy_iron_fly import IronFlyStrategy
from src_8am.backtest.strategy_base import TradeSetup
from src_8am.backtest.pricing import spread_value

ET = ZoneInfo("America/New_York")


@dataclass
class IronFlyResult:
    """Result of a single-day iron fly simulation."""
    trade_id: str
    session_date: date
    symbol: str

    # ORB
    orb_high: float
    orb_low: float
    orb_midline: float
    orb_range_pts: float

    # Trade structure
    center_strike: Optional[float] = None
    long_put: Optional[float] = None
    long_call: Optional[float] = None
    em_pts: Optional[float] = None

    # Entry
    entry_time: Optional[datetime] = None
    underlying_at_entry: Optional[float] = None
    credit_received: Optional[float] = None
    put_credit: Optional[float] = None
    call_credit: Optional[float] = None
    iv_used: Optional[float] = None

    # Exit
    exit_time: Optional[datetime] = None
    underlying_at_exit: Optional[float] = None
    debit_paid: Optional[float] = None
    exit_reason: Optional[str] = None

    # P&L
    realized_pnl_usd: Optional[float] = None
    realized_pnl_pts: Optional[float] = None  # for report.py compat

    status: str = "skipped"
    skip_reason: Optional[str] = None
    bias: Optional[str] = "neutral"  # for report.py compat


class IronFlyEngine:
    """Simulates 0DTE Iron Fly trades day by day."""

    def __init__(self, strategy: IronFlyStrategy, config: Dict[str, Any]):
        self._strategy = strategy
        self._config = config

        strat = config.get("strategy", {})
        opt = config.get("options", {})
        mkt = config.get("market", {})
        risk = config.get("risk", {})

        # Times
        self._orb_start = self._parse_time(strat.get("orb_window_start", "09:30"))
        self._orb_end = self._parse_time(strat.get("orb_window_end", "10:00"))
        self._entry_time = self._parse_time(strat.get("entry_time", "10:00"))
        self._eod = self._parse_time(strat.get("eod_force_close", "15:50"))

        # Market
        self._market_close = self._parse_time(mkt.get("market_close", "16:00"))
        self._trading_days_year = mkt.get("trading_days_per_year", 252)
        self._trading_hours_day = mkt.get("trading_hours_per_day", 6.5)

        # Options
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)
        self._tp_fixed_dollars = opt.get("tp_fixed_dollars", 75.0)
        self._sl_fixed_dollars = opt.get("sl_fixed_dollars", 0.0)  # 0 = disabled (wing-only)
        self._strike_step = opt.get("strike_step_pts", 5.0)

        # Macro filter
        macro = strat.get("macro_filter", {})
        self._macro_enabled = macro.get("enabled", True)
        self._max_candle_body_pct = macro.get("max_candle_body_pct_of_em", 0.50)
        self._max_vix = macro.get("max_vix", 30.0)

        # Risk
        self._max_loss_usd = risk.get("max_loss_per_trade_usd", 50000)

        # ORB filter tolerance
        self._orb_tolerance_pct = strat.get("orb_filter_tolerance_pct", 0.0)

        # Calendar filters (reuse date sets from ORB engine)
        self._skip_nfp = strat.get("skip_nfp_fridays", False)
        self._skip_fomc = strat.get("skip_fomc_days", False)
        self._skip_cpi = strat.get("skip_cpi_days", False)

        # CBBO store (set externally)
        self._cbbo_store = None

        # Passed to strategy during exit for wing lookups
        self._current_trade_extra: Dict[str, Any] = {}

        # Session IV source
        self._iv_source = opt.get("iv_source", "vix")
        self._assumed_iv = opt.get("assumed_iv", 0.15)

    def set_cbbo_store(self, store) -> None:
        self._cbbo_store = store
        self._strategy.set_cbbo_store(store)

    def run(self, days_data: List[Tuple[date, List[Dict]]],
            iv_by_date: Optional[Dict[date, float]] = None) -> List[IronFlyResult]:
        """Run backtest across all provided days."""
        results = []
        for session_date, bars in days_data:
            if self._cbbo_store is not None:
                self._cbbo_store.load_day(session_date)
            session_iv = (iv_by_date or {}).get(session_date, self._assumed_iv)
            result = self._run_day(session_date, bars, session_iv)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results

    def _run_day(self, session_date: date, bars: List[Dict],
                 session_iv: float) -> IronFlyResult:
        """Simulate one trading day."""
        symbol = require(self._config, "bot.symbol")
        tid = f"IFL-{session_date}-{uuid.uuid4().hex[:6].upper()}"

        # Step 0: Calendar filters
        if self._skip_nfp and self._is_nfp_friday(session_date):
            return self._skip(tid, session_date, symbol, 0, 0, "nfp_friday")
        if self._skip_fomc and self._is_fomc_day(session_date):
            return self._skip(tid, session_date, symbol, 0, 0, "fomc_day")
        if self._skip_cpi and self._is_cpi_day(session_date):
            return self._skip(tid, session_date, symbol, 0, 0, "cpi_day")

        # Step 1: Build ORB (bars within 9:30–10:00)
        orb_bars = [b for b in bars if self._orb_start <= b["ts_et"].time() < self._orb_end]
        if len(orb_bars) < 2:
            return self._skip(tid, session_date, symbol, 0, 0, "insufficient_orb_bars")

        orb_high = max(b["high"] for b in orb_bars)
        orb_low = min(b["low"] for b in orb_bars)
        orb_mid = (orb_high + orb_low) / 2
        orb_range = orb_high - orb_low

        # Step 2: Macro filter
        if self._macro_enabled:
            # Check for large candle bodies during ORB
            entry_bar = next(
                (b for b in bars if b["ts_et"].time() >= self._entry_time), None
            )
            em_estimate = self._estimate_em(orb_mid, session_iv, self._entry_time)
            if em_estimate and em_estimate > 0:
                for b in orb_bars:
                    body = abs(b["close"] - b["open"])
                    if body > self._max_candle_body_pct * em_estimate:
                        return self._skip(tid, session_date, symbol, orb_high, orb_low,
                                          "macro_large_candle")

            # VIX threshold (using session_iv as proxy)
            annualized_vix = session_iv * 100
            if annualized_vix > self._max_vix:
                return self._skip(tid, session_date, symbol, orb_high, orb_low,
                                  "macro_high_vix")

        # Step 3: Find entry bar (10:00 ET)
        entry_bar = None
        for b in bars:
            if b["ts_et"].time() >= self._entry_time:
                entry_bar = b
                break

        if entry_bar is None:
            return self._skip(tid, session_date, symbol, orb_high, orb_low, "no_entry_bar")

        spot = entry_bar["close"]

        # Check spot is within ORB range (with tolerance)
        tolerance = self._orb_tolerance_pct * orb_range
        if spot > orb_high + tolerance or spot < orb_low - tolerance:
            return self._skip(tid, session_date, symbol, orb_high, orb_low,
                              "outside_orb_range")

        # Step 4: Select trade
        entry_time = entry_bar["ts_et"]
        tte_hours = self._tte_hours(entry_time)
        tte_years = tte_hours / (self._trading_days_year * self._trading_hours_day)

        setup = self._strategy.select_trade(
            spot=spot,
            bias="neutral",
            entry_time=entry_time,
            session_iv=session_iv,
            tte_years=tte_years,
            tte_hours=tte_hours,
            engine=self,
        )
        if setup is None:
            return self._skip(tid, session_date, symbol, orb_high, orb_low,
                              "no_trade_setup")

        extra = setup.extra
        self._current_trade_extra = extra

        # Risk check
        max_width = max(extra["put_width"], extra["call_width"])
        max_loss = (max_width - setup.credit) * self._multiplier * self._quantity
        if max_loss > self._max_loss_usd:
            return self._skip(tid, session_date, symbol, orb_high, orb_low,
                              "risk_limit")

        # Step 5: Simulate exit
        post_entry = [b for b in bars if b["ts_et"] > entry_time]
        exit_time, exit_price, exit_reason, debit = self._simulate_exit(
            bars=post_entry,
            entry_bar=entry_bar,
            setup=setup,
            session_iv=session_iv,
        )

        pnl_pts = round(setup.credit - debit, 2)
        pnl_usd = round(pnl_pts * self._multiplier * self._quantity, 2)

        return IronFlyResult(
            trade_id=tid,
            session_date=session_date,
            symbol=symbol,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_midline=round(orb_mid, 2),
            orb_range_pts=round(orb_range, 2),
            center_strike=extra["center_strike"],
            long_put=extra["long_put"],
            long_call=extra["long_call"],
            em_pts=extra["em_pts"],
            entry_time=entry_time,
            underlying_at_entry=spot,
            credit_received=setup.credit,
            put_credit=extra.get("put_credit"),
            call_credit=extra.get("call_credit"),
            iv_used=session_iv,
            exit_time=exit_time,
            underlying_at_exit=exit_price,
            debit_paid=debit,
            exit_reason=exit_reason,
            realized_pnl_usd=pnl_usd,
            realized_pnl_pts=pnl_pts,
            status="closed",
        )

    def _simulate_exit(
        self,
        bars: List[Dict],
        entry_bar: Dict,
        setup: TradeSetup,
        session_iv: float,
    ) -> Tuple[datetime, float, str, float]:
        """Walk bars after entry, checking TP / SL / EOD.

        Returns (exit_time, underlying_price, exit_reason, debit_paid).
        """
        extra = setup.extra
        center_k = extra["center_strike"]
        long_put_k = extra["long_put"]
        long_call_k = extra["long_call"]
        credit = setup.credit

        # TP/SL in points: dollars / (multiplier * qty)
        tp_profit_pts = self._tp_fixed_dollars / (self._multiplier * self._quantity)
        tp_debit = round(credit - tp_profit_pts, 2)

        sl_loss_pts = self._sl_fixed_dollars / (self._multiplier * self._quantity) if self._sl_fixed_dollars > 0 else 0.0
        sl_debit = round(credit + sl_loss_pts, 2) if sl_loss_pts > 0 else 0.0

        check_bars = [entry_bar] + list(bars)

        for bar in check_bars:
            ts = bar["ts_et"]

            # EOD close
            if ts.time() >= self._eod:
                tte_h = self._tte_hours(ts)
                tte_y = tte_h / (self._trading_days_year * self._trading_hours_day)
                debit = self._iron_fly_value(bar["close"], center_k, long_put_k,
                                             long_call_k, tte_y, session_iv)
                return ts, bar["close"], "eod_close", round(debit, 2)

            tte_h = self._tte_hours(ts)
            tte_y = tte_h / (self._trading_days_year * self._trading_hours_day)
            minute_et = ts.strftime("%H:%M")

            # Value-based SL: check at bar close (not high/low — ATM gamma
            # causes large intra-bar swings that recover by close).
            mark_close = self._iron_fly_value_cbbo(
                bar["close"], center_k, long_put_k, long_call_k,
                tte_y, session_iv, minute_et,
            )
            if sl_debit > 0 and mark_close >= sl_debit:
                sl_label = f"sl_{int(self._sl_fixed_dollars)}usd"
                return ts, bar["close"], sl_label, round(sl_debit, 2)

            # Wing-based SL (fallback when value SL disabled or gap through)
            if bar["low"] <= long_put_k:
                debit = self._iron_fly_value_cbbo(
                    long_put_k, center_k, long_put_k, long_call_k,
                    tte_y, session_iv, minute_et,
                )
                return ts, bar["low"], "sl_put_wing", round(debit, 2)

            if bar["high"] >= long_call_k:
                debit = self._iron_fly_value_cbbo(
                    long_call_k, center_k, long_put_k, long_call_k,
                    tte_y, session_iv, minute_et,
                )
                return ts, bar["high"], "sl_call_wing", round(debit, 2)

            # TP check (at close)
            if tp_debit > 0 and mark_close <= tp_debit:
                return ts, bar["close"], f"tp_{int(self._tp_fixed_dollars)}usd", round(tp_debit, 2)

        # Expired worthless (shouldn't happen if EOD bar exists, but safety)
        last = bars[-1] if bars else entry_bar
        return last["ts_et"], last["close"], "expired_worthless", 0.0

    def _iron_fly_value(
        self, spot: float, center_k: float, long_put_k: float,
        long_call_k: float, tte_years: float, iv: float,
    ) -> float:
        """BS mark-to-market of full iron fly (put spread + call spread)."""
        put_val = spread_value(spot, center_k, long_put_k, tte_years, iv,
                               "put_credit_spread", model="bs")
        call_val = spread_value(spot, center_k, long_call_k, tte_years, iv,
                                "call_credit_spread", model="bs")
        return put_val + call_val

    def _iron_fly_value_cbbo(
        self, spot: float, center_k: float, long_put_k: float,
        long_call_k: float, tte_years: float, iv: float, minute_et: str,
    ) -> float:
        """MTM using CBBO quotes when available, BS fallback."""
        if self._cbbo_store is None:
            return self._iron_fly_value(spot, center_k, long_put_k, long_call_k,
                                        tte_years, iv)

        # Put spread debit (cost to close)
        put_q = self._cbbo_store.mid_spread_quote(
            minute_et, center_k, long_put_k, is_call=False,
        )
        call_q = self._cbbo_store.mid_spread_quote(
            minute_et, center_k, long_call_k, is_call=True,
        )

        put_val = put_q[1] if put_q is not None else spread_value(
            spot, center_k, long_put_k, tte_years, iv, "put_credit_spread",
        )
        call_val = call_q[1] if call_q is not None else spread_value(
            spot, center_k, long_call_k, tte_years, iv, "call_credit_spread",
        )
        return put_val + call_val

    def _iron_fly_intrinsic(
        self, breach_price: float, center_k: float,
        long_put_k: float, long_call_k: float,
    ) -> float:
        """Intrinsic value of iron fly at wing breach (worst case).

        At put wing breach: put side = put_width, call side ≈ 0
        At call wing breach: call side = call_width, put side ≈ 0
        """
        put_intrinsic = max(0.0, center_k - breach_price) - max(0.0, long_put_k - breach_price)
        call_intrinsic = max(0.0, breach_price - center_k) - max(0.0, breach_price - long_call_k)
        return max(0.0, put_intrinsic + call_intrinsic)

    def _estimate_em(self, spot: float, iv: float, entry_time) -> Optional[float]:
        """Quick EM estimate for macro filter (VIX-based)."""
        if isinstance(entry_time, time):
            close_h = self._market_close.hour + self._market_close.minute / 60
            entry_h = entry_time.hour + entry_time.minute / 60
            tte_hours = max(close_h - entry_h, 0.1)
        else:
            tte_hours = self._tte_hours(entry_time)
        daily_em = spot * iv / math.sqrt(252)
        return daily_em * math.sqrt(tte_hours / 6.5)

    def _tte_hours(self, ts: datetime) -> float:
        """Trading hours remaining until market close."""
        close_dt = ts.replace(
            hour=self._market_close.hour,
            minute=self._market_close.minute,
            second=0, microsecond=0,
        )
        if ts.tzinfo is None:
            close_dt = close_dt.replace(tzinfo=ET)
            ts = ts.replace(tzinfo=ET)
        remaining = (close_dt - ts).total_seconds() / 3600.0
        return max(remaining, 0.0)

    # ------------------------------------------------------------------
    # Calendar filters (delegate to ORB engine date sets)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_nfp_friday(session_date: date) -> bool:
        return session_date.weekday() == 4 and session_date.day <= 7

    @staticmethod
    def _is_fomc_day(session_date: date) -> bool:
        from src_8am.backtest.options_engine import OrbOptionsBacktestEngine
        return session_date in OrbOptionsBacktestEngine._FOMC_DATES

    @staticmethod
    def _is_cpi_day(session_date: date) -> bool:
        from src_8am.backtest.options_engine import OrbOptionsBacktestEngine
        return session_date in OrbOptionsBacktestEngine._CPI_DATES

    def _skip(self, tid: str, session_date: date, symbol: str,
              orb_high: float, orb_low: float, reason: str) -> IronFlyResult:
        mid = (orb_high + orb_low) / 2 if (orb_high + orb_low) > 0 else 0
        return IronFlyResult(
            trade_id=tid,
            session_date=session_date,
            symbol=symbol,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_midline=round(mid, 2),
            orb_range_pts=round(orb_high - orb_low, 2),
            status="skipped",
            skip_reason=reason,
        )

    @staticmethod
    def _parse_time(s: str) -> time:
        parts = s.split(":")
        return time(int(parts[0]), int(parts[1]))
