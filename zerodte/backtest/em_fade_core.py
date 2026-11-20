"""SPX 0DTE expected-move fade — minute-resolution path simulator on real CBBO.

Tests the Jamal Gi "credit spread mean reversion" claim (docs/spx_0dte_em_fade_spec.md):
mark bands at open ± 1 expected move, and when price TOUCHES a band sell a
credit spread on that same side (a fade), TP +50% of credit, stop-market at
-100% of credit, optional trail.

Why this is not built on BacktestOrchestrator / PositionMonitor
---------------------------------------------------------------
The orchestrator prices exits from the FIRST mark whose *mid* crosses the
target, then applies `exit_fill_frac` to that mark. That is failure-catalog B2
(first-favourable-mark exits are glitch-biased) and it cannot express either of
this spec's two load-bearing fill rules:
  * a resting take-profit limit fills when the market ASK reaches the limit,
    and fills AT the limit, not at whatever mark happened to cross it;
  * a stop-MARKET triggers on the package mark and then fills by CROSSING,
    i.e. strictly worse than the level that triggered it.
Modelling the stop fill at mid is the single most likely way to manufacture a
fake 85% win rate, so the path engine is purpose-built and every mark it reads
goes through `zerodte.backtest.noarb.vertical_mark`.

Everything else is reused: `CbboStore` for quotes, `iv_compute`-style per-strike
IV inversion + `bs_delta` for delta targeting (never an ATM IV — 0DTE skew
misstates a 20-delta wing by several delta points).

Invariants asserted in code (enumerated before any result was looked at):
  I1  every quote used is uncrossed, non-negative, finite            (noarb.quote_ok)
  I2  a short vertical's mark lives in [0, width]                    (noarb.vertical_mark)
  I3  booked pnl ∈ [-(width - credit), +credit]                      (noarb.assert_pnl_bounds)
  I4  0 < credit < width at entry
  I5  exit_minute > entry_minute >= touch_minute
  I6  a stop-market's realised debit >= the debit that triggered it
  I7  trades ⊆ touch days ⊆ session days; every non-trade day carries a reason
  I8  entry depth f monotone: higher f never yields a larger credit
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from zerodte.backtest.noarb import (
    PackageMark,
    assert_pnl_bounds,
    quote_ok,
    vertical_mark,
)
from zerodte.flux.iv_compute import compute_iv_rows
from zerodte.flux.iv_compute import parity_spot as _iv_parity_spot

Quote = Tuple[float, float]
QuoteMap = Dict[Tuple[str, float, bool], Quote]

RATE = 0.04
YEAR_MINUTES = 365.0 * 24.0 * 60.0
EXPIRY_MINUTE = 16 * 60          # SPXW 0DTE is PM-settled at 16:00 ET


# ----------------------------------------------------------------------
# minute helpers
# ----------------------------------------------------------------------

def _to_min(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def _to_str(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def minutes_between(start: str, end: str) -> List[str]:
    return [_to_str(m) for m in range(_to_min(start), _to_min(end) + 1)]


def time_to_expiry_years(minute: str) -> float:
    """Calendar-year fraction from `minute` to the 16:00 ET settlement."""
    return max((EXPIRY_MINUTE - _to_min(minute)) / YEAR_MINUTES, 1.0 / YEAR_MINUTES)


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class EmFadeConfig:
    # --- bands -------------------------------------------------------
    em_multiplier: float = 1.0          # band = open ± em_multiplier × EM
    em_scale: float = 1.0               # A1 sensitivity: 0.8 × straddle
    band_shift_pct: float = 0.0         # ± sensitivity, as a % of the open
    anchor_minute: str = "09:31"        # cache's first minute; 09:30 is not quoted
    em_minute: str = "09:32"

    # --- entry -------------------------------------------------------
    entry_start_et: str = "09:35"
    entry_cutoff_et: str = "13:00"
    direction: str = "fade"             # fade | with_trend (mirror control)
    entry_mode: str = "touch"           # touch | fixed | random_minute
    fixed_entry_et: str = "10:00"
    fixed_side: str = "put"             # side for entry_mode="fixed"
    random_seed: int = 7
    entry_delay_min: int = 0            # minutes between observing the touch and filling
    touch_basis: str = "close"          # close | intrabar (SPY high/low + per-minute basis)
    sr_mode: str = "off"                # off | prior_day  (spec A9 strike nudge, proxied)
    sr_buffer_pts: float = 0.0

    # --- structure ---------------------------------------------------
    # How the short strike is chosen. "delta" normalises premium away (a 0.20d
    # strike migrates 31pt -> 18pt OTM across the session, holding the credit
    # nearly constant); the other modes deliberately do not.
    strike_mode: str = "delta"          # delta | distance | credit | em_frac
    target_distance_pts: float = 25.0   # strike_mode="distance"
    target_credit_pts: float = 1.50     # strike_mode="credit"
    target_em_frac: float = 0.75        # strike_mode="em_frac"
    target_delta: float = 0.20
    delta_min: float = 0.10
    delta_max: float = 0.25
    width_pts: float = 10.0
    min_credit: float = 0.05

    # --- management --------------------------------------------------
    tp_pct_of_credit: float = 0.50      # buy back at 0.5 × credit
    sl_mult_of_credit: float = 1.00     # stop at 2 × credit (= -100% of credit)
    trail_enabled: bool = False
    trail_arm_pct: float = 0.25
    trail_stop_pct: float = 0.10
    eod_flat_et: str = "15:50"

    # --- execution ---------------------------------------------------
    breach_stop: bool = False           # exit on the UNDERLYING reaching the short strike,
                                        # instead of on the spread mark (disables sl_mult)
    stop_fill_at_trigger: bool = False   # True = idealised continuous stop (fills AT the level)
    monitor_every_min: int = 1          # observation grid for the exit path
    entry_depth_f: float = 0.7          # 0.5 = mid, 1.0 = far side of the quote
    exit_regime: str = "middle"         # optimistic | middle | pessimistic
    commission_per_spread: float = 2.00  # repo convention: TT SPX $1/leg open only
    contract_multiplier: int = 100
    quantity: int = 1
    rate: float = RATE

    @property
    def entry_depth(self) -> float:
        """Fraction of each leg's half-spread crossed against you at entry."""
        return max(0.0, min(1.0, 2.0 * self.entry_depth_f - 1.0))

    @property
    def exit_depth(self) -> float:
        return {"optimistic": 0.0, "middle": 0.5, "pessimistic": 1.0}[self.exit_regime]

    @property
    def tp_uses_ask(self) -> bool:
        """Pessimistic regime requires the *ask* to reach the limit (leg-sum basis)."""
        return self.exit_regime == "pessimistic"

    def key(self) -> str:
        """Collision-proof identity for this config.

        Must cover EVERY field: an earlier version omitted `band_shift_pct`, so
        the three band-shift sensitivity arms silently collapsed onto one key
        and reported identical results for 3x the available days. A readable
        prefix is kept for eyeballing, but the hash is what makes it unique.
        """
        import hashlib
        blob = repr(sorted(self.__dict__.items())).encode()
        h = hashlib.blake2b(blob, digest_size=5).hexdigest()
        return (f"d{self.target_delta:g}_w{self.width_pts:g}_tp{self.tp_pct_of_credit:g}"
                f"_sl{self.sl_mult_of_credit:g}_tr{int(self.trail_enabled)}"
                f"_em{self.em_multiplier:g}_f{self.entry_depth_f:g}_{self.exit_regime[:4]}"
                f"_{self.direction}_{self.entry_mode}_{h}")


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------

@dataclass
class EmFadeTrade:
    session_date: date
    side: str                  # "put" | "call"
    touch_minute: str
    entry_minute: str
    exit_minute: str
    hold_minutes: int
    open_spot: float
    em_pts: float
    band_level: float
    spot_at_entry: float
    short_strike: float
    long_strike: float
    width: float
    short_delta: float
    credit: float
    tp_debit: float
    sl_trigger_debit: float
    exit_debit: float
    exit_reason: str
    pnl_usd: float
    max_loss_usd: float
    interior_strike: bool
    glitch_minutes: int
    trail_armed: bool
    both_hit_same_minute: bool


@dataclass
class DayOutcome:
    session_date: date
    trade: Optional[EmFadeTrade] = None
    skip_reason: Optional[str] = None
    touched: bool = False
    open_spot: Optional[float] = None
    em_pts: Optional[float] = None


# ----------------------------------------------------------------------
# per-day context (config-independent work done once)
# ----------------------------------------------------------------------

@dataclass
class DayContext:
    session_date: date
    quotes: QuoteMap
    minutes: List[str]
    rows_by_minute: Dict[str, List[Tuple[float, bool, float, float]]]
    spot: Dict[str, float]
    strikes: List[float]
    open_spot: Optional[float] = None
    em_pts: Optional[float] = None
    # intrabar extremes in index points: {minute: (high, low)}, basis-corrected
    intrabar: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # prior regular session (high, low) in index points, for the S/R proxy
    prior_hl: Optional[Tuple[float, float]] = None
    _delta_cache: Dict[str, Dict[Tuple[float, bool], float]] = field(default_factory=dict)

    @classmethod
    def build(cls, session_date: date, quotes: QuoteMap,
              anchor_minute: str = "09:31", em_minute: str = "09:32",
              bars: Optional[Dict[str, Tuple[float, float, float]]] = None,
              prior_hl: Optional[Tuple[float, float]] = None) -> "DayContext":
        rows_by_minute: Dict[str, List[Tuple[float, bool, float, float]]] = {}
        for (minute, strike, is_call), q in quotes.items():
            if not quote_ok(q):
                continue
            rows_by_minute.setdefault(minute, []).append((strike, is_call, q[0], q[1]))
        minutes = sorted(rows_by_minute)
        spot: Dict[str, float] = {}
        for m in minutes:
            s = _parity_spot(rows_by_minute[m])
            if s is not None:
                spot[m] = s
        strikes = sorted({k for (_, k, _) in quotes})
        ctx = cls(session_date, quotes, minutes, rows_by_minute, spot, strikes)
        ctx.open_spot = ctx.spot_at(anchor_minute)
        ctx.em_pts = ctx.atm_straddle(em_minute)
        ctx.prior_hl = prior_hl
        if bars:
            # SPY*10 carries an index-vs-ETF basis, so shift each minute's bar
            # extremes by that minute's OWN measured basis before comparing them
            # to bands defined on the parity spot (failure-catalog A1/A3/F5).
            for m, (hi10, lo10, cl10) in bars.items():
                s_par = spot.get(m)
                if s_par is None:
                    continue
                basis = s_par - cl10
                ctx.intrabar[m] = (hi10 + basis, lo10 + basis)
        return ctx

    # -- spot / EM ----------------------------------------------------

    def spot_at(self, minute: str, max_stale: int = 5) -> Optional[float]:
        m = _to_min(minute)
        for off in range(0, max_stale + 1):
            s = self.spot.get(_to_str(m - off))
            if s is not None:
                return s
        return None

    def lookup(self, minute: str, strike: float, is_call: bool,
               max_stale: int = 5) -> Optional[Quote]:
        m = _to_min(minute)
        for off in range(0, max_stale + 1):
            q = self.quotes.get((_to_str(m - off), strike, is_call))
            if q is not None:
                return q
        return None

    def atm_straddle(self, minute: str) -> Optional[float]:
        spot = self.spot_at(minute)
        if spot is None or not self.strikes:
            return None
        atm_k = min(self.strikes, key=lambda k: abs(k - spot))
        cq = self.lookup(minute, atm_k, True)
        pq = self.lookup(minute, atm_k, False)
        if not quote_ok(cq) or not quote_ok(pq):
            return None
        return (cq[0] + cq[1]) / 2.0 + (pq[0] + pq[1]) / 2.0

    # -- per-strike delta, inverted from that strike's OWN mid --------

    def deltas_at(self, minute: str, rate: float = RATE) -> Dict[Tuple[float, bool], float]:
        if minute in self._delta_cache:
            return self._delta_cache[minute]
        out: Dict[Tuple[float, bool], float] = {}
        rows = self.rows_by_minute.get(minute)
        spot = self.spot_at(minute)
        if rows and spot:
            # Shared definition: per-strike IV inverted from that strike's OWN
            # mid, then bs_delta. Delegated rather than reimplemented — this is
            # the study's most load-bearing quantity and must not fork.
            option_rows, _ = compute_iv_rows(rows, time_to_expiry_years(minute),
                                             spot=spot, rate=rate)
            out = {(r.strike, r.is_call): r.delta for r in option_rows}
        self._delta_cache[minute] = out
        return out


def _parity_spot(rows: Sequence[Tuple[float, bool, float, float]]) -> Optional[float]:
    """S = K + C_mid - P_mid at the strike minimising |C_mid - P_mid| (true ATM).

    Hint-free and independent of any external bar series, so it carries no
    SPY×10 basis (failure-catalog A1/A3). Delegates to the repo's single
    definition in `zerodte.flux.iv_compute` — an earlier version of this module
    reimplemented it, which is the same fork-the-definition failure that
    `noarb.py` exists to prevent (F1).
    """
    return _iv_parity_spot(list(rows))


# ----------------------------------------------------------------------
# fill helpers
# ----------------------------------------------------------------------

def _sell_fill(q: Quote, depth: float) -> float:
    bid, ask = q
    mid = (bid + ask) / 2.0
    return mid - depth * (mid - bid)


def _buy_fill(q: Quote, depth: float) -> float:
    bid, ask = q
    mid = (bid + ask) / 2.0
    return mid + depth * (ask - mid)


def _pkg_fill(mark: PackageMark, depth: float, width: float) -> float:
    """Debit paid to buy the package back, `depth` of the half-spread crossed."""
    px = mark.mid + depth * (mark.ask - mark.mid)
    return max(0.0, min(px, width))


# ----------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------

def simulate_day(ctx: DayContext, cfg: EmFadeConfig) -> DayOutcome:
    sd = ctx.session_date
    out = DayOutcome(session_date=sd)

    open_spot = ctx.spot_at(cfg.anchor_minute)
    if open_spot is None:
        out.skip_reason = "no_open_spot"
        return out
    out.open_spot = open_spot

    straddle = ctx.atm_straddle(cfg.em_minute)
    if straddle is None or straddle <= 0:
        out.skip_reason = "no_em"
        return out
    em = straddle * cfg.em_scale
    out.em_pts = em

    shift = open_spot * cfg.band_shift_pct / 100.0
    upper = open_spot + cfg.em_multiplier * em + shift
    lower = open_spot - cfg.em_multiplier * em - shift

    scan = [m for m in ctx.minutes
            if cfg.entry_start_et <= m <= cfg.entry_cutoff_et]

    # ---- entry trigger ---------------------------------------------
    touch_minute: Optional[str] = None
    touch_dir: Optional[str] = None       # "up" | "down"
    band_level = 0.0
    intrabar = cfg.touch_basis == "intrabar"
    for m in scan:
        s = ctx.spot.get(m)
        if s is None:
            continue
        hi, lo = (ctx.intrabar.get(m, (s, s)) if intrabar else (s, s))
        if hi >= upper:
            touch_minute, touch_dir, band_level = m, "up", upper
            break
        if lo <= lower:
            touch_minute, touch_dir, band_level = m, "down", lower
            break

    out.touched = touch_minute is not None

    if cfg.entry_mode == "touch":
        if touch_minute is None:
            out.skip_reason = "no_touch"
            return out
        entry_minute = touch_minute
    elif cfg.entry_mode == "fixed":
        entry_minute = cfg.fixed_entry_et
        touch_dir = "up" if cfg.fixed_side == "call" else "down"
        touch_minute = entry_minute
        band_level = open_spot
    elif cfg.entry_mode == "random_minute":
        # Same days and same side as the touch rule, but a random minute drawn
        # AT OR AFTER the touch. Drawing from the whole window would let the
        # control enter *before* the touch while using the touch direction to
        # pick the side — future information, which made this control look far
        # worse than it is and flattered the fade by comparison.
        if touch_minute is None:
            out.skip_reason = "no_touch"
            return out
        rng = random.Random(f"{sd.isoformat()}|{cfg.random_seed}")
        cand = [m for m in scan if m >= touch_minute and m in ctx.spot]
        if not cand:
            out.skip_reason = "no_quote_in_window"
            return out
        entry_minute = rng.choice(cand)
    elif cfg.entry_mode == "random_uncond":
        # D9 baseline: no trigger at all — random minute and random side, every day
        rng = random.Random(f"{sd.isoformat()}|u|{cfg.random_seed}")
        cand = [m for m in scan if m in ctx.spot]
        if not cand:
            out.skip_reason = "no_quote_in_window"
            return out
        entry_minute = rng.choice(cand)
        touch_dir = rng.choice(["up", "down"])
        touch_minute = entry_minute
        band_level = open_spot
    else:
        raise ValueError(f"unknown entry_mode {cfg.entry_mode!r}")

    # observing a touch and being filled are not the same instant
    if cfg.entry_delay_min:
        entry_minute = _to_str(_to_min(entry_minute) + cfg.entry_delay_min)
        if entry_minute > cfg.eod_flat_et:
            out.skip_reason = "entry_delay_past_eod"
            return out

    # fade sells on the side price touched; the mirror control sells the other
    if cfg.direction == "fade":
        side = "call" if touch_dir == "up" else "put"
    elif cfg.direction == "with_trend":
        side = "put" if touch_dir == "up" else "call"
    else:
        raise ValueError(f"unknown direction {cfg.direction!r}")

    spot_e = ctx.spot_at(entry_minute)
    if spot_e is None:
        out.skip_reason = "no_quote_at_entry"
        return out

    # ---- strike selection: delta from each strike's own mid ---------
    is_call = side == "call"
    deltas = ctx.deltas_at(entry_minute, cfg.rate)
    if not deltas:
        out.skip_reason = "no_chain_at_entry"
        return out

    otm = [k for (k, kc) in deltas if kc is is_call
           and (k > spot_e if is_call else k < spot_e)]

    cands = []
    if cfg.strike_mode == "delta":
        for (k, kc), d in deltas.items():
            if kc is not is_call or k not in set(otm):
                continue
            ad = abs(d)
            if cfg.delta_min <= ad <= cfg.delta_max:
                cands.append((abs(ad - cfg.target_delta), k, ad))
        if not cands:
            out.skip_reason = "delta_out_of_range"
            return out
    elif cfg.strike_mode in ("distance", "em_frac"):
        want = (cfg.target_distance_pts if cfg.strike_mode == "distance"
                else cfg.target_em_frac * em)
        tgt = spot_e + want if is_call else spot_e - want
        for k in otm:
            cands.append((abs(k - tgt), k, abs(deltas[(k, is_call)])))
        if not cands:
            out.skip_reason = "no_otm_strike"
            return out
    elif cfg.strike_mode == "credit":
        # pick the strike whose ACTUAL entry credit is nearest the target
        for k in otm:
            wk = k + cfg.width_pts if is_call else k - cfg.width_pts
            sq = ctx.lookup(entry_minute, k, is_call)
            lq = ctx.lookup(entry_minute, wk, is_call)
            if not quote_ok(sq) or not quote_ok(lq):
                continue
            cr = _sell_fill(sq, cfg.entry_depth) - _buy_fill(lq, cfg.entry_depth)
            if cr <= 0:
                continue
            cands.append((abs(cr - cfg.target_credit_pts), k, abs(deltas[(k, is_call)])))
        if not cands:
            out.skip_reason = "no_credit_match"
            return out
    else:
        raise ValueError(f"unknown strike_mode {cfg.strike_mode!r}")

    # Spec A9, proxied ex ante: "place the short strike just beyond a recent
    # support/resistance level". Prior regular session's high (calls) / low
    # (puts), plus a buffer. The delta band is NOT relaxed to accommodate it —
    # relaxing would change the instrument rather than the filter (D15), so a
    # day with no qualifying strike inside the band is skipped, exactly as a
    # live trader following both rules would find.
    if cfg.sr_mode == "prior_day":
        if ctx.prior_hl is None:
            out.skip_reason = "no_sr_level"
            return out
        p_hi, p_lo = ctx.prior_hl
        level = (p_hi + cfg.sr_buffer_pts) if is_call else (p_lo - cfg.sr_buffer_pts)
        cands = [c for c in cands if (c[1] >= level if is_call else c[1] <= level)]
        if not cands:
            out.skip_reason = "sr_filter_blocked"
            return out
    elif cfg.sr_mode != "off":
        raise ValueError(f"unknown sr_mode {cfg.sr_mode!r}")

    cands.sort()
    _, short_k, short_delta = cands[0]

    long_k = short_k + cfg.width_pts if is_call else short_k - cfg.width_pts
    short_q = ctx.lookup(entry_minute, short_k, is_call)
    long_q = ctx.lookup(entry_minute, long_k, is_call)
    if not quote_ok(long_q):
        out.skip_reason = "wing_unlisted"
        return out
    if not quote_ok(short_q):
        out.skip_reason = "no_quote_at_entry"
        return out

    entry_mark = vertical_mark(short_q, long_q, cfg.width_pts)
    if entry_mark is None:
        out.skip_reason = "bad_entry_mark"
        return out

    credit = round(_sell_fill(short_q, cfg.entry_depth) - _buy_fill(long_q, cfg.entry_depth), 3)
    if credit < cfg.min_credit:
        out.skip_reason = "credit_below_min"
        return out
    if credit >= cfg.width_pts:                       # I4
        out.skip_reason = "credit_exceeds_width"
        return out

    day_strikes = sorted({k for (k, kc) in deltas if kc is is_call})
    interior = bool(day_strikes) and (min(day_strikes) < min(short_k, long_k)
                                      and max(day_strikes) > max(short_k, long_k))

    tp_debit = credit * cfg.tp_pct_of_credit
    sl_trigger = credit * (1.0 + cfg.sl_mult_of_credit)

    # ---- path management -------------------------------------------
    walk = [m for m in ctx.minutes if entry_minute < m <= cfg.eod_flat_et]
    if cfg.monitor_every_min > 1:
        e0 = _to_min(entry_minute)
        walk = [m for m in walk
                if (_to_min(m) - e0) % cfg.monitor_every_min == 0 or m == walk[-1]]
    exit_minute = None
    exit_debit = None
    exit_reason = None
    glitches = 0
    trail_armed = False
    both_hit = False
    last_valid: Optional[Tuple[str, PackageMark]] = None

    for m in walk:
        sq = ctx.lookup(m, short_k, is_call, max_stale=2)
        lq = ctx.lookup(m, long_k, is_call, max_stale=2)
        mark = vertical_mark(sq, lq, cfg.width_pts)
        if mark is None:
            glitches += 1
            continue
        last_valid = (m, mark)

        stop_level = sl_trigger
        if trail_armed:
            stop_level = min(stop_level, credit * (1.0 - cfg.trail_stop_pct))

        if cfg.breach_stop:
            # The loss diagnostic showed the mark-based stop fires on positions
            # whose short strike is never threatened (2% breach rate among the
            # worst trades). This exits on the UNDERLYING instead: flat only once
            # spot actually reaches the short strike.
            s_now = ctx.spot.get(m)
            stop_hit = s_now is not None and (
                s_now >= short_k if is_call else s_now <= short_k)
            stop_level = mark.mid if stop_hit else cfg.width_pts
        else:
            # stop-market triggers on the package MARK, then fills by crossing
            stop_hit = mark.mid >= stop_level
        tp_ref = mark.ask if cfg.tp_uses_ask else mark.mid
        # tp_pct_of_credit <= 0 disables the take-profit arm entirely
        tp_hit = cfg.tp_pct_of_credit > 0 and tp_ref <= tp_debit

        if stop_hit and tp_hit:
            both_hit = True
        if stop_hit:
            if cfg.stop_fill_at_trigger:
                # Idealised continuous monitoring: the stop is assumed to fill at
                # exactly the level that triggered it. Not attainable for a real
                # stop-MARKET, but it isolates how much of the measured loss is
                # genuine slippage versus an artifact of the 1-minute grid.
                fill = min(max(stop_level, 0.0), cfg.width_pts)
            else:
                fill = _pkg_fill(mark, 1.0 if cfg.exit_regime == "pessimistic"
                                 else cfg.exit_depth, cfg.width_pts)
                fill = max(fill, stop_level)           # I6: never better than the trigger
            exit_minute, exit_debit = m, fill
            exit_reason = "trail_stop" if (trail_armed and stop_level < sl_trigger) else "stop_loss"
            break
        if tp_hit:
            # a resting buy limit fills AT the limit (never at a glitch print)
            exit_minute, exit_debit, exit_reason = m, tp_debit, "take_profit"
            break

        if cfg.trail_enabled and not trail_armed:
            if mark.mid <= credit * (1.0 - cfg.trail_arm_pct):
                trail_armed = True

    if exit_minute is None:
        if last_valid is None:
            out.skip_reason = "no_valid_mark_after_entry"
            return out
        m, mark = last_valid
        exit_minute = cfg.eod_flat_et if m >= cfg.eod_flat_et else m
        exit_debit = _pkg_fill(mark, cfg.exit_depth, cfg.width_pts)
        exit_reason = "eod_flat"

    pnl_pts = credit - exit_debit
    assert_pnl_bounds(pnl_pts, credit, cfg.width_pts, ctx=f"{sd} {side} {short_k}")  # I3

    pnl_usd = round(
        pnl_pts * cfg.contract_multiplier * cfg.quantity - cfg.commission_per_spread, 2
    )
    max_loss_usd = round(
        (cfg.width_pts - credit) * cfg.contract_multiplier * cfg.quantity
        + cfg.commission_per_spread, 2
    )

    out.trade = EmFadeTrade(
        session_date=sd,
        side=side,
        touch_minute=touch_minute or entry_minute,
        entry_minute=entry_minute,
        exit_minute=exit_minute,
        hold_minutes=_to_min(exit_minute) - _to_min(entry_minute),
        open_spot=round(open_spot, 2),
        em_pts=round(em, 2),
        band_level=round(band_level, 2),
        spot_at_entry=round(spot_e, 2),
        short_strike=short_k,
        long_strike=long_k,
        width=cfg.width_pts,
        short_delta=round(short_delta, 4),
        credit=round(credit, 3),
        tp_debit=round(tp_debit, 3),
        sl_trigger_debit=round(sl_trigger, 3),
        exit_debit=round(exit_debit, 3),
        exit_reason=exit_reason,
        pnl_usd=pnl_usd,
        max_loss_usd=max_loss_usd,
        interior_strike=interior,
        glitch_minutes=glitches,
        trail_armed=trail_armed,
        both_hit_same_minute=both_hit,
    )
    return out


__all__ = [
    "DayContext",
    "DayOutcome",
    "EmFadeConfig",
    "EmFadeTrade",
    "minutes_between",
    "simulate_day",
    "time_to_expiry_years",
]
