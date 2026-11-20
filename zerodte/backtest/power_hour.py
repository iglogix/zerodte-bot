"""
0-DTE "Power Hour" backtest engine — three staggered fixed-credit SPX iron
condors entered in the final trading hour (spec: docs/0DTE-Power-Hour-Strategy.md).

Version A: short strikes by fixed credit per side ($4/$3/$2), cheap ~$0.05 long
wings, fixed dollar stop on the combined short strangle with a one-time
profit-triggered ratchet, forced close at 15:55 ET (or expiry).

Version C overlay: classify each tranche by SPX %-from-open at its entry time
(up >= +0.5% / down <= -0.6% / normal); skip normal tranches and skew the
put:call ratio 2:1 in the direction of the move.

Reuses CbboStore quote pickles (1-min OPRA CBBO) and SPY x10 bars for the
underlying timeline; spot is derived from put-call parity, and expiry
settlement uses the robust parity forward at the last cached minute.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.engine import _FOMC_DATES

ET = ZoneInfo("America/New_York")

QuoteMap = Dict[Tuple[str, float, bool], Tuple[float, float]]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class PowerHourResult:
    session_date: date
    tranche: str
    status: str = "skipped"            # closed | skipped
    skip_reason: Optional[str] = None

    entry_time: Optional[str] = None   # "HH:MM" ET
    day_class: Optional[str] = None    # up | down | normal (overlay) or None
    spot_at_entry: Optional[float] = None
    pct_from_open: Optional[float] = None

    short_put: Optional[float] = None
    short_call: Optional[float] = None
    long_put: Optional[float] = None
    long_call: Optional[float] = None
    put_width: Optional[float] = None
    call_width: Optional[float] = None

    qty_put: int = 0
    qty_call: int = 0

    put_credit: Optional[float] = None     # short put mid at entry
    call_credit: Optional[float] = None    # short call mid at entry
    strangle_credit: Optional[float] = None  # per 1:1 unit (put+call mids)
    wing_debit: Optional[float] = None       # weighted wing cost
    weighted_credit: Optional[float] = None  # qty-weighted strangle credit (stop ref)

    stop_initial: Optional[float] = None   # strangle-price stop level (weighted)
    ratcheted: bool = False
    ratchet_time: Optional[str] = None
    stop_final: Optional[float] = None

    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None      # stopped | profit_target | time_exit | expired
    exit_debit: Optional[float] = None     # weighted debit paid to close shorts
    wing_salvage: float = 0.0              # weighted wing value realized at settlement
    settle_spot: Optional[float] = None

    commissions: float = 0.0
    slippage_usd: float = 0.0
    realized_pnl_usd: Optional[float] = None
    max_loss_usd: Optional[float] = None   # defined-risk worst case (per position)
    edge_flag: bool = False                # strike selection hit edge of cached chain


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PowerHourEngine:
    def __init__(self, config: dict):
        strat = config.get("strategy", {})
        opt = config.get("options", {})

        self._tranches = strat.get("tranches", [])
        self._ratchet_mode = strat.get("ratchet_mode", "trigger_plus")
        self._ratchet_trigger = float(strat.get("ratchet_trigger_profit_pct", 0.40))
        self._tp_pct = float(strat.get("profit_target_pct", 0.90))
        self._final_exit = strat.get("final_exit", "15:55")
        self._final_exit_mode = strat.get("final_exit_mode", "close")
        self._wing_target = float(strat.get("wing_target_price", 0.05))
        self._stop_execution = strat.get("stop_execution", "trigger")
        self._skip_fomc = bool(strat.get("skip_fomc_days", True))
        self._dis_min_spread = float(strat.get("dislocation_min_spread_pts", 1.0))
        self._dis_mult = float(strat.get("dislocation_spread_mult", 1.5))

        # Version B (lecture 1.10): delta-based shorts, fixed-width wings,
        # trailing stop with scale-outs. Selected via strategy.version: "B".
        self._version = str(strat.get("version", "A")).upper()
        vb = strat.get("version_b", {}) or {}
        self._b_delta = float(vb.get("target_delta", 0.20))
        self._b_wing_width = float(vb.get("wing_width_pts", 70))
        self._b_trail = float(vb.get("trail_pct", 0.60))
        self._b_tiers = vb.get("scale_tiers", [
            {"profit": 0.20, "close_frac": 0.50, "new_trail": 0.30},
            {"profit": 0.40, "close_frac": 0.25},
        ])

        overlay = strat.get("regime_overlay", {}) or {}
        self._overlay_enabled = bool(overlay.get("enabled", False))
        self._overlay_up = float(overlay.get("up_pct", 0.5))
        self._overlay_down = float(overlay.get("down_pct", -0.6))
        self._overlay_skip_normal = bool(overlay.get("skip_normal", True))
        self._overlay_ratio = int(overlay.get("ratio", 2))

        self._mult = float(opt.get("contract_multiplier", 100))
        self._slip = float(opt.get("slippage_per_spread", 0.10))
        self._stop_slip = float(opt.get("stop_slippage_per_spread", 0.25))
        self._comm_open = float(opt.get("commission_open_per_contract_per_leg", 1.0))
        self._comm_close = float(opt.get("commission_close_per_contract_per_leg", 1.0))

        self._store: Optional[CbboStore] = None

    def set_cbbo_store(self, store: CbboStore) -> None:
        self._store = store

    # ------------------------------------------------------------------

    def run(self, days_data: List[Tuple[date, list]]) -> List[PowerHourResult]:
        """days_data: list of (session_date, bars) where bars are BarLoader dicts."""
        out: List[PowerHourResult] = []
        for d, bars in days_data:
            out.extend(self._process_day(d, bars))
        return out

    # ------------------------------------------------------------------

    def _process_day(self, d: date, bars: list) -> List[PowerHourResult]:
        if self._skip_fomc and d in _FOMC_DATES:
            return [PowerHourResult(d, t["name"], skip_reason="fomc_day")
                    for t in self._tranches]

        if self._store is None or self._store.load_day(d) == 0:
            return [PowerHourResult(d, t["name"], skip_reason="no_cbbo_data")
                    for t in self._tranches]

        bars_by_minute = {b["ts_et"].strftime("%H:%M"): b for b in bars}
        settle = self._store.settlement_forward()

        # Day open (for the Version C %-from-open classification). SPY x10 scale
        # is consistent between open and entry bar, so the basis cancels.
        open_bar = next((b for b in bars
                         if b["ts_et"].time() >= datetime(2000, 1, 1, 9, 30).time()), None)
        day_open = float(open_bar["open"]) if open_bar else None

        results: List[PowerHourResult] = []
        for spec in self._tranches:
            if self._version == "B":
                results.append(self._run_tranche_b(d, spec, bars_by_minute, settle))
            else:
                results.append(self._run_tranche(d, spec, bars_by_minute, day_open, settle))
        return results

    # ------------------------------------------------------------------
    # Quote helpers
    # ------------------------------------------------------------------

    def _mid(self, minute: str, strike: float, is_call: bool,
             stale: int = 3) -> Optional[float]:
        q = self._store.lookup(minute, strike, is_call, max_stale_minutes=stale)
        if q is None:
            return None
        bid, ask = q
        if ask <= 0:
            return None
        return (bid + ask) / 2

    def _quote(self, minute: str, strike: float, is_call: bool,
               stale: int = 3) -> Optional[Tuple[float, float]]:
        return self._store.lookup(minute, strike, is_call, max_stale_minutes=stale)

    def _dislocated(self, q: Tuple[float, float]) -> bool:
        bid, ask = q
        spread = ask - bid
        mid = (bid + ask) / 2
        return spread > self._dis_min_spread and spread > self._dis_mult * max(mid, 0.01)

    def _entry_chain(self, minute: str, is_call: bool) -> Dict[float, Tuple[float, float]]:
        """All strikes with a live quote at the entry minute (with staleness)."""
        chain: Dict[float, Tuple[float, float]] = {}
        for k in sorted(self._store.available_strikes):
            q = self._quote(minute, k, is_call)
            if q is not None and q[1] > 0:
                chain[k] = q
        return chain

    # ------------------------------------------------------------------
    # Tranche simulation
    # ------------------------------------------------------------------

    def _run_tranche(self, d: date, spec: dict, bars_by_minute: dict,
                     day_open: Optional[float], settle: Optional[float]) -> PowerHourResult:
        name = spec["name"]
        entry_min = spec["entry"]
        target = float(spec["target_credit"])
        stop_dollars = float(spec["stop"])
        ratchet_add = spec.get("ratchet_add")
        base_qty = int(spec.get("qty", 1))

        res = PowerHourResult(d, name, entry_time=entry_min)

        bar = bars_by_minute.get(entry_min)
        if bar is None:
            res.skip_reason = "no_bar_at_entry"
            return res

        spot = self._store.derive_spot(entry_min, bar["close"]) or bar["close"]
        res.spot_at_entry = round(spot, 2)

        # ---- Version C day classification at this tranche's entry time
        qty_put, qty_call = base_qty, base_qty
        if day_open:
            res.pct_from_open = round((bar["close"] / day_open - 1) * 100, 3)
        if self._overlay_enabled:
            if day_open is None or res.pct_from_open is None:
                res.skip_reason = "no_open_for_overlay"
                return res
            if res.pct_from_open >= self._overlay_up:
                res.day_class = "up"
                qty_put, qty_call = base_qty * self._overlay_ratio, base_qty
            elif res.pct_from_open <= self._overlay_down:
                res.day_class = "down"
                qty_put, qty_call = base_qty, base_qty * self._overlay_ratio
            else:
                res.day_class = "normal"
                if self._overlay_skip_normal:
                    res.skip_reason = "normal_day"
                    return res

        # ---- Strike selection: fixed credit per side, never invert
        calls = self._entry_chain(entry_min, True)
        puts = self._entry_chain(entry_min, False)
        if not calls or not puts:
            res.skip_reason = "no_quotes_at_entry"
            return res

        def closest(chain: Dict[float, Tuple[float, float]], tgt: float,
                    prefer_otm_call: Optional[bool] = None) -> float:
            # min |mid - target|; ties go further OTM (higher call / lower put)
            def key(k):
                mid = (chain[k][0] + chain[k][1]) / 2
                tie = -k if prefer_otm_call else k
                return (abs(mid - tgt), tie)
            return min(chain, key=key)

        k_call = closest(calls, target, prefer_otm_call=True)
        k_put = closest(puts, target, prefer_otm_call=False)

        # Never invert — floor is the ATM straddle.
        if k_put > k_call:
            atm = min(self._store.available_strikes, key=lambda k: abs(k - spot))
            k_put = k_call = atm

        cq, pq = calls.get(k_call), puts.get(k_put)
        if cq is None or pq is None:
            res.skip_reason = "short_leg_unquoted"
            return res
        if self._dislocated(cq) or self._dislocated(pq):
            res.skip_reason = "dislocated_short_quote"
            return res

        call_credit = (cq[0] + cq[1]) / 2
        put_credit = (pq[0] + pq[1]) / 2

        # ---- Wings: nearest strike beyond the short whose ASK <= wing_target
        # (zero-bid insurance is bought at the ask; mid would understate cost).
        # If nothing in the cached chain is cheap enough, take the cheapest ask
        # available (further OTM on ties) and flag the chain edge.
        def pick_wing(chain: Dict[float, Tuple[float, float]], is_call: bool):
            cheap = [k for k, q in chain.items() if q[1] <= self._wing_target + 1e-9]
            if cheap:
                k = min(cheap) if is_call else max(cheap)   # nearest to the short
                return k, chain[k][1], False
            k = min(chain, key=lambda k: (chain[k][1], -k if is_call else k))
            return k, chain[k][1], True

        wing_calls = {k: q for k, q in calls.items() if k > k_call}
        wing_puts = {k: q for k, q in puts.items() if k < k_put}
        if not wing_calls or not wing_puts:
            res.skip_reason = "no_wing_available"
            res.edge_flag = True
            return res
        k_wc, wc_mid, edge_c = pick_wing(wing_calls, True)
        k_wp, wp_mid, edge_p = pick_wing(wing_puts, False)

        strikes = sorted(self._store.available_strikes)
        res.edge_flag = (edge_c or edge_p
                         or k_wc == strikes[-1] or k_wp == strikes[0])

        res.short_put, res.short_call = k_put, k_call
        res.long_put, res.long_call = k_wp, k_wc
        res.put_width = round(k_put - k_wp, 1)
        res.call_width = round(k_wc - k_call, 1)
        res.qty_put, res.qty_call = qty_put, qty_call
        res.put_credit = round(put_credit, 2)
        res.call_credit = round(call_credit, 2)
        res.strangle_credit = round(put_credit + call_credit, 2)

        # ---- Weighted (ratio-aware) entry bookkeeping
        n_spreads = qty_put + qty_call          # side-spreads opened
        scale = n_spreads / 2.0                 # dollar-stop scale vs the 1:1 spec
        ew = qty_put * put_credit + qty_call * call_credit
        wing_cost = qty_put * wp_mid + qty_call * wc_mid
        res.wing_debit = round(wing_cost, 2)
        res.weighted_credit = round(ew, 2)

        entry_slip = self._slip * n_spreads
        comm_open = self._comm_open * 2 * n_spreads      # short+long leg per spread
        commissions = comm_open
        slippage_usd = entry_slip * self._mult

        stop_level = ew + stop_dollars * scale
        res.stop_initial = round(stop_level, 2)
        tp_level = ew * (1.0 - self._tp_pct)
        ratchet_level = ew * (1.0 - self._ratchet_trigger)

        res.status = "closed"

        # Worst case if stops fail: widest side realizes full width minus net credit.
        worst_width = max(qty_put * res.put_width, qty_call * res.call_width)
        res.max_loss_usd = round((worst_width - (ew - wing_cost)) * self._mult, 0)

        # ---- Minute walk
        exit_debit: Optional[float] = None
        exit_reason: Optional[str] = None
        exit_time: Optional[str] = None
        ratcheted = False
        close_comm_spreads = 0  # side-spreads bought back (commission on close)

        minutes = self._iter_minutes(entry_min, self._final_exit)
        for m in minutes[1:]:  # start the minute after entry
            pm = self._mid(m, k_put, False)
            cm = self._mid(m, k_call, True)
            if pm is None or cm is None:
                continue
            mw = qty_put * pm + qty_call * cm

            # 1) stop
            if mw >= stop_level:
                if self._stop_execution == "quote":
                    pa = self._quote(m, k_put, False)
                    ca = self._quote(m, k_call, True)
                    raw = (qty_put * (pa[1] if pa else pm)
                           + qty_call * (ca[1] if ca else cm))
                    exit_debit = raw + self._stop_slip * n_spreads
                else:  # trigger — optimistic: fill at the stop level itself
                    exit_debit = stop_level + self._stop_slip * n_spreads
                slippage_usd += self._stop_slip * n_spreads * self._mult
                exit_reason = "stopped"
                exit_time = m
                close_comm_spreads = n_spreads
                break

            # 2) profit target (90% decay)
            if mw <= tp_level:
                exit_debit = mw + self._slip * n_spreads
                slippage_usd += self._slip * n_spreads * self._mult
                exit_reason = "profit_target"
                exit_time = m
                close_comm_spreads = n_spreads
                break

            # 3) one-time ratchet
            if not ratcheted and ratchet_add is not None and mw <= ratchet_level:
                ratcheted = True
                res.ratcheted = True
                res.ratchet_time = m
                if self._ratchet_mode == "entry_plus":
                    stop_level = ew + float(ratchet_add) * scale
                else:  # trigger_plus (verbal worked example)
                    stop_level = mw + float(ratchet_add) * scale

        res.stop_final = round(stop_level, 2)

        # ---- Time exit / expiry
        if exit_reason is None:
            if self._final_exit_mode == "expire":
                exit_reason = "expired"
                exit_time = self._final_exit
                if settle is not None:
                    intr_put = max(0.0, k_put - settle)
                    intr_call = max(0.0, settle - k_call)
                    exit_debit = qty_put * intr_put + qty_call * intr_call
                else:
                    exit_debit = 0.0
                res.settle_spot = settle
            else:
                m = self._final_exit
                pm = self._mid(m, k_put, False, stale=5)
                cm = self._mid(m, k_call, True, stale=5)
                if pm is None or cm is None:
                    # no quote at the close — fall back to settlement intrinsic
                    exit_reason = "expired"
                    exit_time = m
                    if settle is not None:
                        exit_debit = (qty_put * max(0.0, k_put - settle)
                                      + qty_call * max(0.0, settle - k_call))
                    else:
                        exit_debit = 0.0
                    res.settle_spot = settle
                else:
                    exit_debit = (qty_put * pm + qty_call * cm) + self._slip * n_spreads
                    slippage_usd += self._slip * n_spreads * self._mult
                    exit_reason = "time_exit"
                    exit_time = m
                    close_comm_spreads = n_spreads

        # Wings ride to expiry; salvage their settlement intrinsic (usually 0).
        salvage = 0.0
        if settle is not None:
            salvage = (qty_put * max(0.0, k_wp - settle)
                       + qty_call * max(0.0, settle - k_wc))
        res.wing_salvage = round(salvage, 2)
        res.settle_spot = settle

        commissions += self._comm_close * close_comm_spreads  # shorts closed only
        # Cap the realized loss at defined risk (wings cap the settlement debit;
        # an intraday stop fill past the wings is theoretically possible but the
        # long would then have offsetting value — approximate by the cap).
        max_debit = (qty_put * res.put_width + qty_call * res.call_width) + wing_cost
        exit_debit = min(exit_debit, max_debit)

        res.exit_time = exit_time
        res.exit_reason = exit_reason
        res.exit_debit = round(exit_debit, 2)
        res.commissions = round(commissions, 2)
        res.slippage_usd = round(slippage_usd, 2)
        # Exit slippage is already inside exit_debit; entry slippage applied here.
        res.realized_pnl_usd = round(
            (ew - wing_cost - entry_slip - exit_debit + salvage) * self._mult
            - commissions, 2)
        return res

    # ------------------------------------------------------------------
    # Version B — 20-delta shorts, fixed-width wings, trailing stop, scale-outs
    # ------------------------------------------------------------------

    @staticmethod
    def _bs_delta(spot: float, strike: float, t_years: float, mid: float,
                  is_call: bool) -> Optional[float]:
        """Delta via implied vol bisected from the quote mid (r=0)."""
        import math

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

    def _run_tranche_b(self, d: date, spec: dict, bars_by_minute: dict,
                       settle: Optional[float]) -> PowerHourResult:
        name = spec["name"]
        entry_min = spec["entry"]
        qty = int(spec.get("qty", 4))
        res = PowerHourResult(d, name, entry_time=entry_min)

        bar = bars_by_minute.get(entry_min)
        if bar is None:
            res.skip_reason = "no_bar_at_entry"
            return res
        spot = self._store.derive_spot(entry_min, bar["close"]) or bar["close"]
        res.spot_at_entry = round(spot, 2)

        calls = self._entry_chain(entry_min, True)
        puts = self._entry_chain(entry_min, False)
        if not calls or not puts:
            res.skip_reason = "no_quotes_at_entry"
            return res

        # minutes to the 16:00 cash close, ACT/365 (consistent with the IV inversion)
        h, m = int(entry_min[:2]), int(entry_min[3:5])
        t_years = max((16 * 60 - (h * 60 + m)), 1) / (60 * 24 * 365)

        def pick_by_delta(chain, is_call):
            best, best_err = None, None
            for k, q in chain.items():
                if is_call and k <= spot:
                    continue
                if not is_call and k >= spot:
                    continue
                mid = (q[0] + q[1]) / 2
                if mid <= 0.05:
                    continue
                delta = self._bs_delta(spot, k, t_years, mid, is_call)
                if delta is None:
                    continue
                err = abs(abs(delta) - self._b_delta)
                if best_err is None or err < best_err:
                    best, best_err = k, err
            return best

        k_call = pick_by_delta(calls, True)
        k_put = pick_by_delta(puts, False)
        if k_call is None or k_put is None:
            res.skip_reason = "no_delta_strike"
            return res
        cq, pq = calls[k_call], puts[k_put]
        if self._dislocated(cq) or self._dislocated(pq):
            res.skip_reason = "dislocated_short_quote"
            return res
        call_credit = (cq[0] + cq[1]) / 2
        put_credit = (pq[0] + pq[1]) / 2

        # Fixed-width wings (~70 pts), snapped to the nearest cached strike.
        strikes = sorted(self._store.available_strikes)
        wing_c_target = k_call + self._b_wing_width
        wing_p_target = k_put - self._b_wing_width
        k_wc = min((k for k in strikes if k > k_call),
                   key=lambda k: abs(k - wing_c_target), default=None)
        k_wp = min((k for k in strikes if k < k_put),
                   key=lambda k: abs(k - wing_p_target), default=None)
        if k_wc is None or k_wp is None:
            res.skip_reason = "no_wing_available"
            return res
        wcq = self._quote(entry_min, k_wc, True)
        wpq = self._quote(entry_min, k_wp, False)
        wc_cost = wcq[1] if wcq else 0.05
        wp_cost = wpq[1] if wpq else 0.05
        res.edge_flag = (k_wc == strikes[-1] or k_wp == strikes[0])

        res.short_put, res.short_call = k_put, k_call
        res.long_put, res.long_call = k_wp, k_wc
        res.put_width = round(k_put - k_wp, 1)
        res.call_width = round(k_wc - k_call, 1)
        res.qty_put = res.qty_call = qty
        res.put_credit = round(put_credit, 2)
        res.call_credit = round(call_credit, 2)
        ew = put_credit + call_credit          # per-unit strangle credit
        wing_cost = wp_cost + wc_cost          # per-unit wing debit
        res.strangle_credit = round(ew, 2)
        res.wing_debit = round(wing_cost, 2)
        res.weighted_credit = round(ew * qty, 2)
        res.status = "closed"
        res.max_loss_usd = round(
            (max(res.put_width, res.call_width) - (ew - wing_cost))
            * self._mult * qty, 0)

        trail = self._b_trail
        lwm = ew
        stop_level = lwm + trail * ew
        res.stop_initial = round(stop_level, 2)

        commissions = self._comm_open * 4 * qty
        entry_slip = self._slip * 2 * qty
        slippage_usd = entry_slip * self._mult
        realized_units = 0.0      # accumulated (credit - debit) per closed contract
        open_q = qty
        tiers_done = [False] * len(self._b_tiers)
        exit_reason = exit_time = None

        for mnt in self._iter_minutes(entry_min, self._final_exit)[1:]:
            pm = self._mid(mnt, k_put, False)
            cm = self._mid(mnt, k_call, True)
            if pm is None or cm is None:
                continue
            mw = pm + cm

            if mw < lwm:
                lwm = mw
                stop_level = lwm + trail * ew

            if mw >= stop_level:
                debit = stop_level + self._stop_slip * 2
                slippage_usd += self._stop_slip * 2 * open_q * self._mult
                realized_units += (ew - debit) * open_q
                commissions += self._comm_close * 2 * open_q
                open_q = 0
                exit_reason, exit_time = "stopped", mnt
                res.exit_debit = round(debit, 2)
                break

            profit_pct = 1.0 - mw / ew
            for i, tier in enumerate(self._b_tiers):
                if tiers_done[i] or profit_pct < float(tier["profit"]):
                    continue
                tiers_done[i] = True
                n_close = min(open_q, max(1, round(float(tier["close_frac"]) * qty)))
                if n_close > 0:
                    debit = mw + self._slip * 2
                    slippage_usd += self._slip * 2 * n_close * self._mult
                    realized_units += (ew - debit) * n_close
                    commissions += self._comm_close * 2 * n_close
                    open_q -= n_close
                if "new_trail" in tier and tier["new_trail"] is not None:
                    trail = float(tier["new_trail"])
                    stop_level = lwm + trail * ew
            if open_q <= 0:
                exit_reason, exit_time = "scaled_out", mnt
                res.exit_debit = round(mw, 2)
                break

        if open_q > 0:
            mnt = self._final_exit
            pm = self._mid(mnt, k_put, False, stale=5)
            cm = self._mid(mnt, k_call, True, stale=5)
            if pm is not None and cm is not None:
                debit = (pm + cm) + self._slip * 2
                slippage_usd += self._slip * 2 * open_q * self._mult
                realized_units += (ew - debit) * open_q
                commissions += self._comm_close * 2 * open_q
                exit_reason = exit_reason or "time_exit"
                res.exit_debit = round(debit, 2)
            else:
                intr = 0.0
                if settle is not None:
                    intr = max(0.0, k_put - settle) + max(0.0, settle - k_call)
                realized_units += (ew - intr) * open_q
                exit_reason = exit_reason or "expired"
                res.exit_debit = round(intr, 2)
            exit_time = mnt
            open_q = 0

        salvage = 0.0
        if settle is not None:
            salvage = max(0.0, k_wp - settle) + max(0.0, settle - k_wc)
        res.wing_salvage = round(salvage * qty, 2)
        res.settle_spot = settle
        res.stop_final = round(stop_level, 2)
        res.exit_time = exit_time
        res.exit_reason = exit_reason
        res.commissions = round(commissions, 2)
        res.slippage_usd = round(slippage_usd, 2)
        res.realized_pnl_usd = round(
            (realized_units - (wing_cost - salvage) * qty
             - entry_slip) * self._mult - commissions, 2)
        return res

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
