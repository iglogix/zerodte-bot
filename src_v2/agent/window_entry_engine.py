"""Single-shot time-window entry engine.

Used by strategies that enter at a configured wall-clock time (e.g. 09:45 ET).
Mirrors ScheduledEngine's deterministic core but fires once per session:

  1. Advance the clock until the configured entry_time.
  2. Stability + large-move guards (volatility-relative, IV from ATM straddle).
  3. Strategy.search_candidates -> validate -> place_order.
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Optional
from zoneinfo import ZoneInfo

from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.agent.decision_engine import CycleResult

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class WindowEntryEngine(BaseEngine):
    """Fires a single entry attempt at the configured time."""

    def __init__(self, deps: EngineDeps, strategy):
        super().__init__(deps, strategy)
        from src_v2.backtest.sim_clock import WallClock
        self._clock = deps.clock or WallClock()

        entry_cfg = deps.config.get("entry", {})
        self._entry_time_str = entry_cfg.get("entry_time", "09:45")
        self._stability_fraction = entry_cfg.get("stability_fraction", 0.15)
        self._single_candle_fraction = entry_cfg.get("single_candle_fraction", 0.10)
        self._fallback_stability_pts = entry_cfg.get("fallback_stability_pts", 8.0)
        self._stability_check_enabled = entry_cfg.get("stability_check", True)

        strat_cfg = deps.config.get("strategy", {})
        eod = strat_cfg.get("eod_force_close", "15:50")
        self._eod = time(*[int(x) for x in eod.split(":")])

        self._poll_interval = 1.0

        # Regime gate (VIX9D, prior close)
        regime_cfg = deps.config.get("regime", {})
        self._regime_enabled = regime_cfg.get("enabled", False)
        self._vix_csv = regime_cfg.get("vix_csv", "data/vix9d_daily.csv")
        self._vix_min = regime_cfg.get("vix_min", None)
        self._vix_max = regime_cfg.get("vix_max", None)

        # Opening-range gate: skip when the first 5-min bar's range exceeds
        # `max_open_range_em_ratio × ATM straddle EM`.
        open_gate = entry_cfg.get("opening_range_filter", {})
        self._open_gate_enabled = open_gate.get("enabled", False)
        self._open_window_start = open_gate.get("window_start", "09:30")
        self._open_window_end = open_gate.get("window_end", "09:35")
        self._open_max_em_ratio = open_gate.get("max_open_range_em_ratio", None)
        self._open_max_pts = open_gate.get("max_open_range_pts", None)

    async def run(self) -> CycleResult:
        et_h, et_m = [int(x) for x in self._entry_time_str.split(":")]
        target = time(et_h, et_m)

        while True:
            now_et = self._clock.now(ET)
            now_t = now_et.time()
            if now_t >= self._eod:
                return CycleResult(outcome="waited", reason="eod_no_entry").finish()
            if now_t >= target:
                break
            await self._clock.sleep(self._poll_interval)

        return await self._execute_entry()

    async def _execute_entry(self) -> CycleResult:
        # Regime gate (prior-close VIX9D)
        if self._regime_enabled and (self._vix_min is not None or self._vix_max is not None):
            block = self._regime_check()
            if block:
                return CycleResult(outcome="waited", reason=block).finish()

        candles = await self.deps.tool_handlers.get_5min_candles(count=10)
        if len(candles) < 3:
            return CycleResult(outcome="waited", reason="insufficient_candles").finish()

        # Stability guards (mirror ScheduledEngine)
        if self._stability_check_enabled:
            block = self._stability_check(candles)
            if block:
                return CycleResult(outcome="waited", reason=block).finish()

        try:
            chain = await self.deps.tool_handlers.get_option_chain(expiration="0DTE")
            spot_data = await self.deps.tool_handlers.get_spx_price()
            spot = spot_data.get("price", 0)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"chain_fetch_failed: {exc}").finish()

        # Inject current time so strategy can compute time-to-expiration.
        self.strategy._now_dt = self._clock.now(ET)

        try:
            candidates = await self.strategy.search_candidates(chain, spot)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"candidates_failed: {exc}").finish()

        # Opening-range gate fires after candidates so we have EM from strategy.
        if self._open_gate_enabled:
            block = self._opening_range_check()
            if block:
                return CycleResult(outcome="waited", reason=block).finish()

        if candidates.get("candidates_found", 0) == 0:
            return CycleResult(outcome="waited", reason="no_candidates").finish()

        top = candidates["top_candidate"]

        try:
            validation = await self.strategy.validate(top, chain)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"validation_error: {exc}").finish()

        if not validation.get("passed", False):
            return CycleResult(
                outcome="waited",
                reason=f"validation_failed: {validation.get('failures', [])}",
            ).finish()

        try:
            order_payload = self.strategy.build_order(top)
            trade_result = await self.deps.tool_handlers.place_order(order_payload)
        except Exception as exc:
            return CycleResult(outcome="error", reason=f"place_order_failed: {exc}").finish()

        logger.info("Window entry placed: strikes=%s", top.strikes)
        return CycleResult(
            outcome="executed",
            reason="window_trade_placed",
            trade=trade_result,
            candidates_found=candidates.get("candidates_found", 0),
        ).finish()

    def _opening_range_check(self) -> Optional[str]:
        """Compute the opening 5-min bar's range and gate entry on it.

        Blocks if range > max_open_range_pts (absolute), or > max_open_range_em_ratio × EM.
        """
        th = self.deps.tool_handlers
        bars = getattr(th, "_bars", None)
        if not bars:
            return None  # live mode: skip silently (could call 5min handler instead)
        from datetime import time as _time
        sh, sm = [int(x) for x in self._open_window_start.split(":")]
        eh, em_ = [int(x) for x in self._open_window_end.split(":")]
        st = _time(sh, sm)
        et = _time(eh, em_)
        window = [b for b in bars if st <= b["ts_et"].time() < et]
        if not window:
            return None
        rng = max(b["high"] for b in window) - min(b["low"] for b in window)

        if self._open_max_pts is not None and rng > self._open_max_pts:
            return f"opening_range_skip: range={rng:.1f}pts > max_pts={self._open_max_pts}"

        if self._open_max_em_ratio is not None:
            em = getattr(self.strategy, "_last_em_pts", None)
            if em and em > 0:
                ratio = rng / em
                if ratio > self._open_max_em_ratio:
                    return (f"opening_range_skip: range={rng:.1f}/EM={em:.1f}="
                            f"{ratio:.2f} > {self._open_max_em_ratio}")
        return None

    def _regime_check(self) -> Optional[str]:
        from src_v2.backtest.vix_loader import load_vix_series, lookup_prior_close
        from datetime import date as _date, timedelta as _timedelta
        series = load_vix_series(self._vix_csv)
        if not series:
            return None  # no data → don't block
        today = self._clock.now(ET).date()
        vix = lookup_prior_close(series, today - _timedelta(days=1))
        if vix is None:
            return None
        if self._vix_min is not None and vix < self._vix_min:
            return f"regime_skip: vix={vix:.1f} < min={self._vix_min}"
        if self._vix_max is not None and vix > self._vix_max:
            return f"regime_skip: vix={vix:.1f} > max={self._vix_max}"
        return None

    def _stability_check(self, candles) -> Optional[str]:
        # Compute expected move from chain via ATM straddle if possible (fallback to fixed pts).
        # We don't have direct EM here; reuse fallback_stability_pts unless strategy injects EM.
        em: Optional[float] = getattr(self.strategy, "_last_em_pts", None)
        if em and em > 0:
            single_thresh = em * self._single_candle_fraction
            stab_thresh = em * self._stability_fraction
        else:
            stab_thresh = self._fallback_stability_pts
            single_thresh = stab_thresh * (self._single_candle_fraction / max(self._stability_fraction, 1e-6))

        prev_close = candles[-2].get("close", 0)
        last_close = candles[-1].get("close", 0)
        if prev_close > 0 and abs(last_close - prev_close) > single_thresh:
            return f"large_move_guard: candle={abs(last_close - prev_close):.1f} > {single_thresh:.1f}"

        closes = [c.get("close", 0) for c in candles[-3:]]
        rng = max(closes) - min(closes)
        if rng > stab_thresh:
            return f"market_unstable: range={rng:.1f} > {stab_thresh:.1f}"
        return None
