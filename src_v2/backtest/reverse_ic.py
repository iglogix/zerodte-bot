"""
0-DTE "Reverse Iron Condor" (RIC) backtest engine — long an asymmetric SPX
iron condor near the open on gap-up days (spec:
docs/0DTE-Reverse-Iron-Condor-Strategy.md, §15).

Structure (net debit, long gamma): buy the 50-delta call with a short call
+20 pts above (20-wide call debit spread) + buy the 35-delta put with a short
put -35 pts below (35-wide put debit spread), same-day expiry.

Entry gates: SPX gaps up >= 0.25% vs the prior session close (per-weekday
overrides supported; parity-derived spot vs prior parity settlement so the
SPYx10 basis never enters), FOMC blackout, and the "sets up" filter — modeled
upside max profit (call_width - debit) must be >= 25% of the debit, which
naturally rejects pumped-IV days.

Exits: resting profit-target limit at debit x (1 + 25-30%), NO stop, and
either cash settlement at the parity forward (default) or a 15:55 close-out.
Optional scale-out variant: fractions at +20% / +30% with a runner to expiry.

Reuses CbboStore quote pickles (1-min OPRA CBBO) and SPY x10 bars for the
session timeline, like the Power Hour engine this mirrors (inverse premium).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.engine import _FOMC_DATES

ET = ZoneInfo("America/New_York")

_WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class RicResult:
    session_date: date
    weekday: str = ""
    status: str = "skipped"            # closed | skipped
    skip_reason: Optional[str] = None

    gap_pct: Optional[float] = None
    gap_source: Optional[str] = None   # parity | spy
    entry_time: Optional[str] = None   # "HH:MM" ET
    spot_at_entry: Optional[float] = None

    long_call: Optional[float] = None
    short_call: Optional[float] = None
    long_put: Optional[float] = None
    short_put: Optional[float] = None
    call_width: Optional[float] = None
    put_width: Optional[float] = None
    call_delta: Optional[float] = None
    put_delta: Optional[float] = None

    qty: int = 0
    debit: Optional[float] = None          # combo debit per 1x (points)
    upside_pct: Optional[float] = None     # (call_width - debit) / debit
    tp_level: Optional[float] = None       # combo value of the resting limit

    tiers_filled: int = 0
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None      # profit_target | expired | time_exit
    exit_value: Optional[float] = None     # combo value at final exit (points)
    settle_spot: Optional[float] = None

    commissions: float = 0.0
    slippage_usd: float = 0.0
    realized_pnl_usd: Optional[float] = None
    max_loss_usd: Optional[float] = None   # debit + entry costs (defined risk)
    edge_flag: bool = False                # strike selection at cached-chain edge


# ---------------------------------------------------------------------------
# Black-Scholes delta implied from the quote mid (r=0), as Power Hour B
# ---------------------------------------------------------------------------

def bs_delta(spot: float, strike: float, t_years: float, mid: float,
             is_call: bool) -> Optional[float]:
    def norm(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def price(sig: float) -> float:
        if sig <= 0 or t_years <= 0:
            return max(0.0, (spot - strike) if is_call else (strike - spot))
        sq = sig * math.sqrt(t_years)
        d1 = (math.log(spot / strike) + 0.5 * sig * sig * t_years) / sq
        d2 = d1 - sq
        if is_call:
            return spot * norm(d1) - strike * norm(d2)
        return strike * norm(-d2) - spot * norm(-d1)

    lo, hi = 1e-4, 5.0
    if not (price(lo) <= mid <= price(hi)):
        return None
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if price(m) < mid:
            lo = m
        else:
            hi = m
    sig = 0.5 * (lo + hi)
    sq = sig * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * sig * sig * t_years) / sq
    return norm(d1) if is_call else norm(d1) - 1.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RicEngine:
    def __init__(self, config: dict):
        strat = config.get("strategy", {})
        opt = config.get("options", {})

        self._entry_time = strat.get("entry_time", "09:33")
        self._gap_minute = strat.get("gap_check_minute", "09:31")
        self._gap_min = float(strat.get("gap_min_pct", 0.25))
        self._gap_min_wd = {
            str(k).lower()[:3]: float(v)
            for k, v in (strat.get("gap_min_pct_by_weekday") or {}).items()
        }
        self._call_delta = float(strat.get("long_call_delta", 0.50))
        self._put_delta = float(strat.get("long_put_delta", 0.35))
        self._call_wing = float(strat.get("call_wing_offset_pts", 20))
        self._put_wing = float(strat.get("put_wing_offset_pts", 35))
        self._tp_pct = float(strat.get("profit_target_pct", 0.25))
        self._tp_fill = strat.get("tp_fill", "limit")
        self._setup_min = float(strat.get("setup_min_upside_pct", 0.25))
        self._final_exit = strat.get("final_exit", "15:55")
        self._final_exit_mode = strat.get("final_exit_mode", "expire")
        self._skip_fomc = bool(strat.get("skip_fomc_days", True))
        self._qty = int(strat.get("qty", 1))
        so = strat.get("scale_out") or {}
        self._scale_enabled = bool(so.get("enabled", False))
        self._scale_tiers = so.get("tiers", [
            {"profit": 0.20, "close_frac": 0.50},
            {"profit": 0.30, "close_frac": 0.25},
        ])
        self._dis_min_spread = float(strat.get("dislocation_min_spread_pts", 1.0))
        self._dis_mult = float(strat.get("dislocation_spread_mult", 1.5))
        # mid: fills at quote mids + slippage (OO-style). natural: pay the ask
        # / hit the bid on every traded leg — the worst-case execution bound.
        self._pricing = strat.get("pricing", "mid")
        # Regime switch (§7/§16): skip when the prior VIX close is below the
        # floor — the long condor needs movement to pay for its debit.
        vf = strat.get("vix_floor")
        self._vix_floor = float(vf) if vf is not None else None
        self._vix: Dict[date, float] = {}

        self._mult = float(opt.get("contract_multiplier", 100))
        self._slip = float(opt.get("slippage_per_spread", 0.10))
        self._comm_open = float(opt.get("commission_open_per_contract_per_leg", 1.0))
        self._comm_close = float(opt.get("commission_close_per_contract_per_leg", 1.0))

        self._store: Optional[CbboStore] = None

    def set_cbbo_store(self, store: CbboStore) -> None:
        self._store = store

    def set_vix(self, vix_by_date: Dict[date, float]) -> None:
        """Daily VIX closes; the floor gate uses the latest close BEFORE the
        session date."""
        self._vix = dict(sorted(vix_by_date.items()))

    def _prior_vix(self, d: date) -> Optional[float]:
        best = None
        for vd, v in self._vix.items():
            if vd >= d:
                break
            best = v
        return best

    # ------------------------------------------------------------------

    def run(self, days_data: List[Tuple[date, list]]) -> List[RicResult]:
        """days_data: chronological (session_date, bars) pairs. Pass ALL bar
        sessions (not just cached ones) so the prior-close reference is the
        true previous trading session."""
        out: List[RicResult] = []
        prev_date: Optional[date] = None
        prev_settle: Optional[float] = None
        prev_bar_close: Optional[float] = None
        for d, bars in days_data:
            res, settle = self._process_day(
                d, bars, prev_date, prev_settle, prev_bar_close)
            out.append(res)
            prev_date = d
            prev_settle = settle
            prev_bar_close = float(bars[-1]["close"]) if bars else None
        return out

    # ------------------------------------------------------------------
    # Quote helpers
    # ------------------------------------------------------------------

    def _mid(self, minute: str, strike: float, is_call: bool,
             stale: int = 3) -> Optional[float]:
        q = self._store.lookup(minute, strike, is_call, max_stale_minutes=stale)
        if q is None or q[1] <= 0:
            return None
        return (q[0] + q[1]) / 2

    def _dislocated(self, q: Optional[Tuple[float, float]]) -> bool:
        if q is None:
            return False
        bid, ask = q
        spread = ask - bid
        mid = (bid + ask) / 2
        return spread > self._dis_min_spread and spread > self._dis_mult * max(mid, 0.01)

    def _entry_chain(self, minute: str, is_call: bool) -> Dict[float, Tuple[float, float]]:
        chain: Dict[float, Tuple[float, float]] = {}
        for k in sorted(self._store.available_strikes):
            q = self._store.lookup(minute, k, is_call, max_stale_minutes=3)
            if q is not None and q[1] > 0:
                chain[k] = q
        return chain

    def _combo_value(self, minute: str, res: RicResult,
                     stale: int = 3) -> Optional[float]:
        """Liquidation value of the combo: mids, or (natural pricing) selling
        the longs at the bid and buying the shorts back at the ask."""
        legs = [(res.long_call, True, +1), (res.short_call, True, -1),
                (res.long_put, False, +1), (res.short_put, False, -1)]
        total = 0.0
        for strike, is_call, sign in legs:
            q = self._store.lookup(minute, strike, is_call, max_stale_minutes=stale)
            if q is None or q[1] <= 0:
                return None
            if self._pricing == "natural":
                px = q[0] if sign > 0 else q[1]
            else:
                px = (q[0] + q[1]) / 2
            total += sign * px
        return max(0.0, total)

    def _settle_value(self, res: RicResult, settle: Optional[float]) -> float:
        if settle is None:
            return 0.0
        call_v = min(max(0.0, settle - res.long_call), res.call_width)
        put_v = min(max(0.0, res.long_put - settle), res.put_width)
        return call_v + put_v

    # ------------------------------------------------------------------
    # Day simulation
    # ------------------------------------------------------------------

    def _process_day(
        self, d: date, bars: list,
        prev_date: Optional[date], prev_settle: Optional[float],
        prev_bar_close: Optional[float],
    ) -> Tuple[RicResult, Optional[float]]:
        res = RicResult(d, weekday=_WEEKDAY_KEYS[d.weekday()])

        if self._store is None or self._store.load_day(d) == 0:
            res.skip_reason = "no_cbbo_data"
            return res, None
        settle = self._store.settlement_forward()

        if self._skip_fomc and d in _FOMC_DATES:
            res.skip_reason = "fomc_day"
            return res, settle

        if self._vix_floor is not None:
            v = self._prior_vix(d)
            if v is None or v < self._vix_floor:
                res.skip_reason = "vix_below_floor"
                return res, settle

        bars_by_minute = {b["ts_et"].strftime("%H:%M"): b for b in bars}

        # ---- Gate 1: gap up vs the prior session close.
        # Primary: parity-derived spot vs prior parity settlement (pure SPX).
        # Fallback: SPYx10 bar vs prior SPYx10 close (scale-consistent).
        stale_ref = prev_date is None or (d - prev_date).days > 7
        gap_bar = bars_by_minute.get(self._gap_minute) or (bars[0] if bars else None)
        gap_pct: Optional[float] = None
        if not stale_ref and gap_bar is not None:
            spot_ref = self._store.derive_spot(self._gap_minute, gap_bar["close"])
            if prev_settle is not None and spot_ref is not None:
                gap_pct = (spot_ref / prev_settle - 1) * 100
                res.gap_source = "parity"
            elif prev_bar_close:
                gap_pct = (float(gap_bar["close"]) / prev_bar_close - 1) * 100
                res.gap_source = "spy"
        if gap_pct is None:
            res.skip_reason = "no_gap_reference"
            return res, settle
        res.gap_pct = round(gap_pct, 3)

        gap_min = self._gap_min_wd.get(res.weekday, self._gap_min)
        if gap_pct <= 0:
            res.skip_reason = "gap_down"
            return res, settle
        if gap_pct < gap_min:
            res.skip_reason = "gap_below_min"
            return res, settle

        # ---- Entry
        entry_min = self._entry_time
        res.entry_time = entry_min
        bar = bars_by_minute.get(entry_min)
        if bar is None:
            res.skip_reason = "no_bar_at_entry"
            return res, settle
        spot = self._store.derive_spot(entry_min, bar["close"]) or float(bar["close"])
        res.spot_at_entry = round(spot, 2)

        calls = self._entry_chain(entry_min, True)
        puts = self._entry_chain(entry_min, False)
        if not calls or not puts:
            res.skip_reason = "no_quotes_at_entry"
            return res, settle

        h, m = int(entry_min[:2]), int(entry_min[3:5])
        t_years = max(16 * 60 - (h * 60 + m), 1) / (60 * 24 * 365)

        def pick_by_delta(chain, is_call, target):
            best, best_err, best_delta = None, None, None
            for k, q in chain.items():
                if not is_call and k >= spot:
                    continue          # the 35-delta long put is OTM by design
                mid = (q[0] + q[1]) / 2
                if mid <= 0.05:
                    continue
                delta = bs_delta(spot, k, t_years, mid, is_call)
                if delta is None:
                    continue
                err = abs(abs(delta) - target)
                if best_err is None or err < best_err:
                    best, best_err, best_delta = k, err, delta
            return best, best_delta

        k_lc, d_lc = pick_by_delta(calls, True, self._call_delta)
        k_lp, d_lp = pick_by_delta(puts, False, self._put_delta)
        if k_lc is None or k_lp is None:
            res.skip_reason = "no_delta_strike"
            return res, settle

        # ---- Wings: fixed point offsets snapped to the nearest cached strike
        # strictly beyond the long.
        strikes = sorted(self._store.available_strikes)
        k_sc = min((k for k in strikes if k > k_lc),
                   key=lambda k: abs(k - (k_lc + self._call_wing)), default=None)
        k_sp = min((k for k in strikes if k < k_lp),
                   key=lambda k: abs(k - (k_lp - self._put_wing)), default=None)
        if k_sc is None or k_sp is None:
            res.skip_reason = "no_wing_available"
            res.edge_flag = True
            return res, settle
        res.edge_flag = (k_sc == strikes[-1] or k_sp == strikes[0])

        legs = [(k_lc, True), (k_sc, True), (k_lp, False), (k_sp, False)]
        quotes = {leg: self._store.lookup(entry_min, leg[0], leg[1],
                                          max_stale_minutes=3) for leg in legs}
        if any(q is None or q[1] <= 0 for q in quotes.values()):
            res.skip_reason = "leg_unquoted"
            return res, settle
        if any(self._dislocated(q) for q in quotes.values()):
            res.skip_reason = "dislocated_quote"
            return res, settle

        if self._pricing == "natural":
            # pay the ask on the longs, receive the bid on the shorts
            px = {leg: (q[1] if leg in ((k_lc, True), (k_lp, False)) else q[0])
                  for leg, q in quotes.items()}
        else:
            px = {leg: (q[0] + q[1]) / 2 for leg, q in quotes.items()}
        debit = ((px[(k_lc, True)] - px[(k_sc, True)])
                 + (px[(k_lp, False)] - px[(k_sp, False)]))
        if debit <= 0:
            res.skip_reason = "non_positive_debit"
            return res, settle

        res.long_call, res.short_call = k_lc, k_sc
        res.long_put, res.short_put = k_lp, k_sp
        res.call_width = round(k_sc - k_lc, 1)
        res.put_width = round(k_lp - k_sp, 1)
        res.call_delta = round(d_lc, 3) if d_lc is not None else None
        res.put_delta = round(d_lp, 3) if d_lp is not None else None
        res.debit = round(debit, 2)

        # ---- Gate 2: the trade must "set up" — upside max profit >= 25% of
        # the debit (a flat/underwater upside ramp means IV is pumped).
        upside = res.call_width - debit
        res.upside_pct = round(upside / debit, 3)
        if upside < self._setup_min * debit:
            res.skip_reason = "setup_filter"
            return res, settle

        # ---- Open the position
        qty = self._qty
        res.qty = qty
        res.status = "closed"
        tp_level = debit * (1.0 + self._tp_pct)
        res.tp_level = round(tp_level, 2)

        commissions = self._comm_open * 4 * qty
        entry_slip = self._slip * 2 * qty            # 2 side-spreads per unit
        slippage_usd = entry_slip * self._mult
        res.max_loss_usd = round((debit * qty + entry_slip) * self._mult
                                 + commissions, 0)

        proceeds = 0.0          # sum of exit combo values x units closed
        open_q = qty
        tiers_done = [False] * len(self._scale_tiers)
        exit_reason: Optional[str] = None
        exit_time: Optional[str] = None

        for mnt in self._iter_minutes(entry_min, self._final_exit)[1:]:
            value = self._combo_value(mnt, res)
            if value is None:
                continue

            if self._scale_enabled:
                profit_pct = value / debit - 1.0
                for i, tier in enumerate(self._scale_tiers):
                    if tiers_done[i] or profit_pct < float(tier["profit"]):
                        continue
                    tiers_done[i] = True
                    n_close = min(open_q,
                                  max(1, round(float(tier["close_frac"]) * qty)))
                    if n_close > 0:
                        proceeds += value * n_close
                        slippage_usd += self._slip * 2 * n_close * self._mult
                        commissions += self._comm_close * 4 * n_close
                        open_q -= n_close
                        res.tiers_filled += 1
                        res.exit_value = round(value, 2)
                        exit_reason, exit_time = "scaled_out", mnt
                if open_q <= 0:
                    break
            elif value >= tp_level:
                # resting limit: fill at the limit level (or the crossing mid)
                fill = tp_level if self._tp_fill == "limit" else value
                proceeds += fill * open_q
                slippage_usd += self._slip * 2 * open_q * self._mult
                commissions += self._comm_close * 4 * open_q
                res.exit_value = round(fill, 2)
                exit_reason, exit_time = "profit_target", mnt
                open_q = 0
                break

        # ---- Remainder: expire (cash settle) or close out at final_exit
        if open_q > 0:
            if self._final_exit_mode == "close":
                value = self._combo_value(self._final_exit, res, stale=5)
                if value is not None:
                    proceeds += value * open_q
                    slippage_usd += self._slip * 2 * open_q * self._mult
                    commissions += self._comm_close * 4 * open_q
                    res.exit_value = round(value, 2)
                    exit_reason = exit_reason or "time_exit"
                    exit_time = self._final_exit
                    open_q = 0
            if open_q > 0:      # expire mode, or no quote at the close
                value = self._settle_value(res, settle)
                proceeds += value * open_q
                res.exit_value = round(value, 2)
                exit_reason = "expired" if exit_reason in (None, "scaled_out") \
                    else exit_reason
                exit_time = "16:00"
                open_q = 0

        # Exit slippage for traded closes is already accumulated; expired
        # units pay no close slippage or commission.
        total_slip_pts = slippage_usd / self._mult
        res.exit_time = exit_time
        res.exit_reason = exit_reason
        res.settle_spot = settle
        res.commissions = round(commissions, 2)
        res.slippage_usd = round(slippage_usd, 2)
        res.realized_pnl_usd = round(
            (proceeds - debit * qty - total_slip_pts) * self._mult - commissions, 2)
        return res, settle

    # ------------------------------------------------------------------

    @staticmethod
    def _iter_minutes(start: str, end: str) -> List[str]:
        h0, m0 = int(start[:2]), int(start[3:5])
        h1, m1 = int(end[:2]), int(end[3:5])
        out = []
        t = h0 * 60 + m0
        while t <= h1 * 60 + m1:
            out.append(f"{t // 60:02d}:{t % 60:02d}")
            t += 1
        return out
