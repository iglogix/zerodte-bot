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

from zerodte.agent.base_engine import BaseEngine, EngineDeps
from zerodte.agent.decision_engine import CycleResult

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class WindowEntryEngine(BaseEngine):
    """Fires a single entry attempt at the configured time."""

    def __init__(self, deps: EngineDeps, strategy):
        super().__init__(deps, strategy)
        from zerodte.backtest.sim_clock import WallClock
        self._clock = deps.clock or WallClock()

        entry_cfg = deps.config.get("entry", {})
        # entry_mode:
        #   "fixed_time" — original: enter at exactly entry_time
        #   "post_move"  — watch from watch_start; enter once cumulative move from
        #                  watch_start crosses move_trigger_em_ratio × ATM-straddle EM
        #                  (computed at watch_start). Bails at watch_end.
        self._entry_mode = entry_cfg.get("entry_mode", "fixed_time")
        self._entry_time_str = entry_cfg.get("entry_time", "09:45")

        post_move = entry_cfg.get("post_move", {})
        self._watch_start_str = post_move.get("watch_start", "09:30")
        self._watch_end_str = post_move.get("watch_end", "10:30")
        self._move_trigger_em_ratio = post_move.get("trigger_em_ratio", 0.30)
        self._move_trigger_pts = post_move.get("trigger_pts", None)  # absolute fallback

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
        if self._entry_mode == "post_move":
            return await self._run_post_move()
        return await self._run_fixed_time()

    async def _run_fixed_time(self) -> CycleResult:
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

    async def _run_post_move(self) -> CycleResult:
        """Wait from watch_start. Compute baseline EM at watch_start, then watch
        cumulative high-low range until it crosses trigger_em_ratio × EM.
        Enter on the next bar after trigger.  If we hit watch_end with no
        trigger, skip the day.
        """
        ws_h, ws_m = [int(x) for x in self._watch_start_str.split(":")]
        we_h, we_m = [int(x) for x in self._watch_end_str.split(":")]
        watch_start = time(ws_h, ws_m)
        watch_end = time(we_h, we_m)

        # Sleep to watch_start
        while True:
            now_t = self._clock.now(ET).time()
            if now_t >= self._eod:
                return CycleResult(outcome="waited", reason="eod_no_entry").finish()
            if now_t >= watch_start:
                break
            await self._clock.sleep(self._poll_interval)

        # Capture baseline EM (compute from ATM straddle; retry up to 5 min if
        # the very first bar's chain is incomplete).
        baseline_em = None
        for _ in range(5):
            baseline_em = await self._compute_em()
            if baseline_em and baseline_em > 0:
                break
            await self._clock.sleep(self._poll_interval)
            if self._clock.now(ET).time() >= watch_end:
                break
        if baseline_em is None or baseline_em <= 0:
            return CycleResult(outcome="waited", reason="no_em_at_watch_start").finish()

        threshold_pts = (self._move_trigger_pts
                         if self._move_trigger_pts is not None
                         else self._move_trigger_em_ratio * baseline_em)

        bars = getattr(self.deps.tool_handlers, "_bars", None)

        # Poll bars until trigger or watch_end
        while True:
            now_dt = self._clock.now(ET)
            now_t = now_dt.time()
            if now_t >= watch_end:
                return CycleResult(
                    outcome="waited",
                    reason=f"no_move_trigger (em={baseline_em:.1f} thr={threshold_pts:.1f})",
                ).finish()

            cur_range = self._cumulative_range(bars, watch_start, now_dt)
            if cur_range is not None and cur_range >= threshold_pts:
                await self._clock.sleep(self._poll_interval)
                return await self._execute_entry()

            await self._clock.sleep(self._poll_interval)

    async def _compute_em(self) -> Optional[float]:
        """Derive EM from ATM straddle at current time. Returns absolute pts."""
        try:
            chain = await self.deps.tool_handlers.get_option_chain(expiration="0DTE")
            spot_data = await self.deps.tool_handlers.get_spx_price()
            spot = spot_data.get("price", 0)
        except Exception:
            return None
        if not chain or spot <= 0:
            return None
        # Find ATM strike
        strikes = sorted({float(o.get("strike", 0)) for o in chain if o.get("strike")})
        if not strikes:
            return None
        atm = min(strikes, key=lambda k: abs(k - spot))
        call_mid = put_mid = None
        for o in chain:
            if float(o.get("strike", 0)) != atm:
                continue
            mid = ((o.get("bid", 0) + o.get("ask", 0)) / 2) if o.get("ask") else None
            if mid is None or mid <= 0:
                continue
            if o.get("option_type", "").lower() == "call":
                call_mid = mid
            else:
                put_mid = mid
        if call_mid is None or put_mid is None:
            return None
        return call_mid + put_mid

    @staticmethod
    def _cumulative_range(bars, watch_start: time, now_dt) -> Optional[float]:
        if not bars:
            return None
        # 1-min bars from watch_start through current minute
        cur_t = now_dt.time()
        chunk = [b for b in bars
                 if watch_start <= b["ts_et"].time() <= cur_t]
        if not chunk:
            return None
        return max(b["high"] for b in chunk) - min(b["low"] for b in chunk)

    async def _execute_entry(self) -> CycleResult:
        # Regime gate (prior-close VIX9D)
        if self._regime_enabled and (self._vix_min is not None or self._vix_max is not None):
            block = self._regime_check()
            if block:
                return CycleResult(outcome="waited", reason=block).finish()

        candles = await self.deps.tool_handlers.get_5min_candles(count=10)
        if len(candles) < 3:
            return CycleResult(outcome="waited", reason="insufficient_candles").finish()

        # Stability + opening-range gates only apply in fixed_time mode.
        # post_move mode INVERTS the volatility logic (we WANT the recent move),
        # so these gates would auto-reject valid setups.
        if self._entry_mode != "post_move" and self._stability_check_enabled:
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

        # Opening-range gate only in fixed_time mode (see comment above).
        if self._entry_mode != "post_move" and self._open_gate_enabled:
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
        from datetime import timedelta as _timedelta

        from zerodte.backtest.vix_loader import load_vix_series, lookup_prior_close
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
