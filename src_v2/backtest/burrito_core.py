"""Burrito Butterfly backtest — directional OTM butterfly + linked vertical
debit spread on cash-settled SPX, with a path-dependent lock/defense/clawback
lifecycle. Reuses Iron Duck pricing/loader/Smile primitives. Pure logic; data
I/O lives in burrito_clock.py and scripts/. See spec
docs/superpowers/specs/2026-06-14-burrito-butterfly-design.md.

All prices are SPX points (per share); P&L uses the 100 multiplier.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from src_v2.backtest.iron_duck import (
    Smile,
    _fill,
    black_call,
    black_put,
    parity_forward,
    _to_date,
)

Side = Literal["call", "put"]
Right = Literal["C", "P"]


def round_to_step(x: float, step: float) -> float:
    return round(x / step) * step


# ---------------------------------------------------------------- structure


@dataclass(frozen=True)
class Leg:
    qty: int        # +long / -short, in lots
    strike: float
    right: Right


@dataclass
class BurritoConfig:
    # entry
    entry_dte: int = 3
    entry_anchor: str = "15:55"          # EOD for DTE>=1; intraday bar for DTE=0
    direction_signal: str = "hull"       # hull | momentum | both | markov
    body_offset: float = 1.0             # body pushed body_offset*EM OTM from forward
    # geometry
    fly_wing_width: float = 15.0
    spread_width: float = 5.0
    spread_link: str = "outer"           # "outer" (default read) | "inner"
    step: float = 5.0
    multiplier: int = 100
    # lifecycle (fractions of net debit at risk)
    lock_trigger_pct: float = 0.07       # +X% -> lock opposing spread (FLOATING)
    stop_pct: float = 0.07               # Uncle Point: -X% -> close
    pt_pct: float = 0.10                 # profit target on floated structure
    defense_policy: Tuple[str, ...] = () # subset of wing_lock|otm_freeze|choo_choo|clawback
    defense_trigger_pct: float = 0.05    # adverse move fraction that arms defenses
    clawback_last_n: int = 1             # final N marks count as the clawback window
    # entry gates
    gates_on: bool = True
    max_debit_pct: float = 0.40          # debit <= max_debit_pct * fly_wing_width
    # costs
    slippage_k: float = 1.0              # _fill half-spread multiplier
    commission_per_leg: float = 0.0


def _merge(legs: List[Leg]) -> List[Leg]:
    agg: Dict[Tuple[float, Right], int] = defaultdict(int)
    for l in legs:
        agg[(l.strike, l.right)] += l.qty
    return [Leg(q, k, r) for (k, r), q in agg.items() if q != 0]


def build_legs(forward: float, em: float, cfg: BurritoConfig, side: Side) -> List[Leg]:
    """Fly (1-2-1, fly_wing_width wings, body pushed OTM) + linked vertical
    debit spread (spread_width). 'outer' link: spread long reuses the fly's
    far/outer wing. Returns net (merged) legs."""
    right: Right = "C" if side == "call" else "P"
    sign = 1.0 if side == "call" else -1.0
    W, S = cfg.fly_wing_width, cfg.spread_width
    body = round_to_step(forward + sign * cfg.body_offset * em, cfg.step)
    inner = body - sign * W            # ATM-side wing
    outer = body + sign * W            # far wing
    fly = [Leg(1, inner, right), Leg(-2, body, right), Leg(1, outer, right)]
    if cfg.spread_link == "outer":
        spread = [Leg(1, outer, right), Leg(-1, outer + sign * S, right)]
    else:                              # inner link: spread long reuses the ATM-side wing
        spread = [Leg(1, inner, right), Leg(-1, inner + sign * S, right)]
    return _merge(fly + spread)


# ---------------------------------------------------------------- payoff / value


def _intrinsic(leg: Leg, s: float) -> float:
    if leg.right == "C":
        return max(s - leg.strike, 0.0)
    return max(leg.strike - s, 0.0)


def payoff(legs: List[Leg], s: float) -> float:
    """Total per-share intrinsic value of all legs at settlement price s
    (excludes the entry debit; net trade P&L = payoff - debit)."""
    return sum(leg.qty * _intrinsic(leg, s) for leg in legs)


def _leg_model_value(leg: Leg, smile: Smile, F: float) -> float:
    v = smile.vol(leg.strike) or 0.0
    px = (black_call(F, leg.strike, smile.T, v) if leg.right == "C"
          else black_put(F, leg.strike, smile.T, v))
    return leg.qty * px


def position_value(legs: List[Leg], chain: Optional[dict], smile: Optional[Smile] = None,
                   F: Optional[float] = None, slippage_k: float = 1.0) -> Optional[float]:
    """Current per-share value of HOLDING the position (= proceeds to close it).
    Closing a long leg sells (proceeds +), closing a short buys (cost -). Uses
    chain fills where available; falls back to the Black model via `smile`.
    Returns None if a leg can be priced by neither source."""
    total = 0.0
    for leg in legs:
        q = chain.get((leg.strike, leg.right)) if chain else None
        if q is not None and q[1] > 0:
            side = "sell" if leg.qty > 0 else "buy"
            px = _fill(q, side, slippage_k)
            total += abs(leg.qty) * (px if leg.qty > 0 else -px)
        elif smile is not None and F is not None and smile.vol(leg.strike):
            total += _leg_model_value(leg, smile, F)
        else:
            return None
    return total


def entry_debit(legs: List[Leg], chain: dict, slippage_k: float = 1.0,
                commission_per_leg: float = 0.0) -> Optional[float]:
    """Net debit to OPEN (buy longs at +half-spread, sell shorts at -half-spread).
    Returns None if any leg is unpriced. Positive = pay; negative = credit."""
    total = 0.0
    for leg in legs:
        q = chain.get((leg.strike, leg.right))
        if q is None or q[1] <= 0:
            return None
        side = "buy" if leg.qty > 0 else "sell"
        px = _fill(q, side, slippage_k)
        total += abs(leg.qty) * (px if leg.qty > 0 else -px)
        total += abs(leg.qty) * commission_per_leg
    return total


def passes_gates(debit: Optional[float], cfg: BurritoConfig) -> bool:
    if not cfg.gates_on:
        return True
    if debit is None or debit <= 0:
        return False
    return debit <= cfg.max_debit_pct * cfg.fly_wing_width


# ---------------------------------------------------------------- direction


def _wma(vals: List[float]) -> float:
    w = list(range(1, len(vals) + 1))
    return sum(v * wi for v, wi in zip(vals, w)) / sum(w)


def _hull_ma(prices: List[float], n: int) -> Optional[float]:
    half = max(1, n // 2)
    sq = max(1, int(round(math.sqrt(n))))
    if len(prices) < n + sq:
        return None
    raw = []
    for end in range(len(prices) - sq + 1, len(prices) + 1):
        win_full = prices[end - n:end]
        win_half = prices[end - half:end]
        raw.append(2 * _wma(win_half) - _wma(win_full))
    return _wma(raw)


def direction_markov(prices: List[float], lookback: int = 20) -> str:
    """Lightweight regime proxy: sign of the mean of recent daily returns.
    Pluggable — replace with the full markov-2-hedge-fund-method skill later."""
    if len(prices) <= lookback:
        return "skip"
    rets = [prices[i] / prices[i - 1] - 1.0 for i in range(len(prices) - lookback, len(prices))]
    m = sum(rets) / len(rets)
    if abs(m) < 1e-5:
        return "skip"
    return "call" if m > 0 else "put"


def direction_from_series(prices: List[float], signal: str, n: int = 21, mom: int = 10) -> str:
    """Map a price series (oldest..newest) to 'call'|'put'|'both'|'skip'."""
    if signal == "both":
        return "both"
    if signal == "momentum":
        if len(prices) <= mom:
            return "skip"
        return "call" if prices[-1] >= prices[-1 - mom] else "put"
    if signal == "markov":
        return direction_markov(prices)
    # default: hull
    h = _hull_ma(prices, n)
    if h is None:
        return "skip"
    return "call" if prices[-1] >= h else "put"


# ---------------------------------------------------------------- adjustments


def _opp(side: Side) -> Side:
    return "put" if side == "call" else "call"


def _debit_spread_legs(forward: float, em: float, cfg: BurritoConfig, side: Side) -> List[Leg]:
    right: Right = "C" if side == "call" else "P"
    sign = 1.0 if side == "call" else -1.0
    body = round_to_step(forward + sign * cfg.body_offset * em, cfg.step)
    outer = body + sign * cfg.fly_wing_width
    return [Leg(1, outer, right), Leg(-1, outer + sign * cfg.spread_width, right)]


def lock_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Lock into 'free' status: buy the OPPOSING-side debit spread."""
    return _debit_spread_legs(forward, em, cfg, _opp(side))


def wing_lock_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Preemptive wing lock == buy the opposing debit spread (same as lock)."""
    return _debit_spread_legs(forward, em, cfg, _opp(side))


def otm_freeze_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Buy 1 cheap far-OTM option in the TREND (adverse) direction to flatten."""
    opp = _opp(side)
    right: Right = "C" if opp == "call" else "P"
    sign = 1.0 if opp == "call" else -1.0
    k = round_to_step(forward + sign * 3.0 * em, cfg.step)
    return [Leg(1, k, right)]


def _fly_only(forward: float, em: float, cfg: BurritoConfig, side: Side) -> List[Leg]:
    right: Right = "C" if side == "call" else "P"
    sign = 1.0 if side == "call" else -1.0
    W = cfg.fly_wing_width
    body = round_to_step(forward + sign * cfg.body_offset * em, cfg.step)
    return _merge([Leg(1, body - sign * W, right), Leg(-2, body, right),
                   Leg(1, body + sign * W, right)])


def choo_choo_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Add a second fly_wing_width fly into the adverse move (overlapping tent)."""
    return _fly_only(forward, em, cfg, side)


def _nearest_valley(legs: List[Leg], forward: float, cfg: BurritoConfig,
                    debit: float = 0.0) -> Optional[float]:
    """Scan settlement payoff around the forward; return the strike-step center of
    the nearest region where net payoff (payoff - debit) < 0, else None."""
    lo = forward - 6 * cfg.fly_wing_width
    hi = forward + 6 * cfg.fly_wing_width
    s = lo
    best = None
    best_dist = 1e18
    while s <= hi:
        if payoff(legs, s) - debit < 0:
            d = abs(s - forward)
            if d < best_dist:
                best_dist, best = d, round_to_step(s, cfg.step)
        s += cfg.step
    return best


def clawback_legs(legs: List[Leg], forward: float, cfg: BurritoConfig,
                  debit: float = 0.0) -> List[Leg]:
    """Hole-injection: a standard fly centered on the nearest sub-zero valley to
    lift it above breakeven."""
    center = _nearest_valley(legs, forward, cfg, debit)
    if center is None:
        return []
    W = cfg.fly_wing_width
    right: Right = "C" if center >= forward else "P"
    return _merge([Leg(1, center - W, right), Leg(-2, center, right), Leg(1, center + W, right)])


# ---------------------------------------------------------------- lifecycle


@dataclass
class BurritoTrade:
    side: Side
    legs: List[Leg]
    debit: float                  # entry cost at slipped fills (for realized P&L)
    entry_date: int
    expiry: int
    debit_mid: float = 0.0        # entry cost at mid (basis for % decisions)
    state: str = "DIRECTIONAL"
    extra_cost: float = 0.0       # adjustment costs at slipped fills
    extra_mid: float = 0.0        # adjustment costs at mid
    exit_date: Optional[int] = None
    exit_reason: Optional[str] = None
    pnl_pts: Optional[float] = None
    pnl_usd: Optional[float] = None
    dte: int = 0
    days_held: int = 0


@dataclass
class Decision:
    action: str            # hold | lock | defend | clawback | close | settle
    next_state: str = ""
    reason: str = ""


def lifecycle_decision(tr: BurritoTrade, state: str, mtm_value: float, cfg: BurritoConfig,
                       is_last: bool, in_valley: bool = False) -> Decision:
    """Pure decision function for one mark. mtm_value = current per-share MID value
    of holding. Capital basis = mid entry cost (debit_mid + extra_mid)."""
    basis = (tr.debit_mid + tr.extra_mid) or (tr.debit + tr.extra_cost)
    pnl_frac = (mtm_value - basis) / basis if basis > 0 else 0.0
    if is_last:
        return Decision("settle", next_state="CLOSED", reason="expiry")
    if state == "DIRECTIONAL":
        if pnl_frac <= -cfg.stop_pct and "wing_lock" not in cfg.defense_policy:
            return Decision("close", next_state="CLOSED", reason="uncle_stop")
        if pnl_frac <= -cfg.defense_trigger_pct and cfg.defense_policy:
            return Decision("defend", next_state="DEFENDED", reason="adverse")
        if pnl_frac >= cfg.lock_trigger_pct:
            return Decision("lock", next_state="FLOATING", reason="lock")
        return Decision("hold", next_state="DIRECTIONAL")
    # FLOATING / DEFENDED
    if pnl_frac >= cfg.pt_pct:
        return Decision("close", next_state="CLOSED", reason="target")
    if in_valley and "clawback" in cfg.defense_policy:
        return Decision("clawback", next_state=state, reason="valley")
    if pnl_frac <= -cfg.stop_pct:
        return Decision("close", next_state="CLOSED", reason="uncle_stop")
    return Decision("hold", next_state=state)


# ---------------------------------------------------------------- engine


def _em_from_chain(chain: dict, F: float, T: float) -> float:
    """Expected move proxy = ATM straddle mid (per project guidance: ATM straddle,
    not VIX). Cheap — no IV solve. Falls back to 1% of F if unpriceable."""
    strikes = sorted({k for (k, _r) in chain})
    if strikes:
        atm = min(strikes, key=lambda k: abs(k - F))
        c = chain.get((atm, "C"))
        p = chain.get((atm, "P"))
        if c and p and c[1] > 0 and p[1] > 0:
            straddle = (c[0] + c[1]) / 2 + (p[0] + p[1]) / 2
            if straddle > 0:
                return straddle
    return 0.01 * F


def run_trade(store, entry_date: int, expiry: int, side: Side,
              em: Optional[float], cfg: BurritoConfig,
              marks: Optional[list] = None) -> Optional[BurritoTrade]:
    """Build legs at entry, walk marks (EOD by default), apply the lifecycle,
    execute adjustments, and close with realized P&L."""
    ch0 = store.chain(entry_date, expiry)
    if ch0 is None:
        return None
    F0 = parity_forward(ch0)
    if F0 is None:
        return None
    T0 = max((_to_date(expiry) - _to_date(entry_date)).days / 365.0, 1e-6)
    if em is None:
        em = _em_from_chain(ch0, F0, T0)
    legs = build_legs(F0, em, cfg, side)
    debit = entry_debit(legs, ch0, cfg.slippage_k, cfg.commission_per_leg)
    debit_mid = entry_debit(legs, ch0, 0.0, 0.0)
    if debit is None or debit_mid is None or (cfg.gates_on and not passes_gates(debit, cfg)):
        return None
    tr = BurritoTrade(side=side, legs=legs, debit=debit, debit_mid=debit_mid,
                      entry_date=entry_date, expiry=expiry,
                      dte=(_to_date(expiry) - _to_date(entry_date)).days)

    if marks is None:
        from src_v2.backtest.burrito_clock import eod_marks
        marks = list(eod_marks(store, entry_date, expiry))
    if not marks:
        return None

    def _add(add_legs, mark):
        """Price an adjustment and fold its legs into the position."""
        if not add_legs:
            return
        fill = entry_debit(add_legs, mark.chain, cfg.slippage_k, cfg.commission_per_leg)
        mid = entry_debit(add_legs, mark.chain, 0.0, 0.0)
        if fill is None or mid is None:
            return
        tr.extra_cost += fill
        tr.extra_mid += mid
        tr.legs = _merge(tr.legs + add_legs)

    n = len(marks)
    for i, m in enumerate(marks):
        # lazy smile: only build (expensive IV solve) when a leg is unpriced by the chain
        mid_val = position_value(tr.legs, m.chain, None, None, 0.0)
        sm = None
        if mid_val is None and m.forward is not None:
            sm = Smile(m.chain, m.forward, m.T)
            mid_val = position_value(tr.legs, m.chain, sm, m.forward, 0.0)
        if mid_val is None:
            continue
        in_window = (i >= n - cfg.clawback_last_n)
        in_valley = (in_window and m.forward is not None and
                     _nearest_valley(tr.legs, m.forward, cfg,
                                     tr.debit_mid + tr.extra_mid) is not None)
        d = lifecycle_decision(tr, tr.state, mid_val, cfg, m.is_last, in_valley)
        if d.action == "lock":
            _add(lock_legs(tr.side, m.forward, em, cfg), m)
            tr.state = d.next_state
        elif d.action == "defend":
            _add(_apply_defense(tr, m, em, cfg), m)
            tr.state = d.next_state
        elif d.action == "clawback":
            _add(clawback_legs(tr.legs, m.forward, cfg, tr.debit_mid + tr.extra_mid), m)
        elif d.action in ("close", "settle"):
            if d.reason in ("expiry", "settle"):
                _close(tr, m, None, d.reason, cfg, settle_price=_settle_price(store, m))
            else:
                fill_val = position_value(tr.legs, m.chain, sm, m.forward, cfg.slippage_k)
                _close(tr, m, fill_val, d.reason, cfg)
            return tr
    # safety net: settle at last mark (cash settlement)
    last = marks[-1]
    _close(tr, last, None, "expiry", cfg, settle_price=_settle_price(store, last))
    return tr


def _apply_defense(tr: BurritoTrade, m, em: float, cfg: BurritoConfig) -> List[Leg]:
    if "wing_lock" in cfg.defense_policy:
        return wing_lock_legs(tr.side, m.forward, em, cfg)
    if "choo_choo" in cfg.defense_policy:
        return choo_choo_legs(tr.side, m.forward, em, cfg)
    if "otm_freeze" in cfg.defense_policy:
        return otm_freeze_legs(tr.side, m.forward, em, cfg)
    return []


def _settle_price(store, m) -> Optional[float]:
    """Robust expiry settlement: parity forward of the expiry chain, else the
    index spot on the expiry day, else the mark's forward (SPXW is PM cash-settled
    at the expiry-day close)."""
    s = parity_forward(m.chain)
    if s is None and hasattr(store, "spot"):
        s = store.spot(m.date)
    return s if s is not None else m.forward


def _close(tr: BurritoTrade, m, val: Optional[float], reason: str, cfg: BurritoConfig,
           settle_price: Optional[float] = None) -> None:
    if reason in ("expiry", "settle"):
        settle = settle_price if settle_price is not None else (parity_forward(m.chain) or m.forward)
        gross = payoff(tr.legs, settle) if settle is not None else (val or 0.0)
    elif val is not None:
        gross = val                      # slipped proceeds to close early
    else:                                # unpriceable: fall back to intrinsic at forward
        gross = payoff(tr.legs, m.forward) if m.forward is not None else (tr.debit + tr.extra_cost)
    tr.pnl_pts = gross - (tr.debit + tr.extra_cost)
    tr.pnl_usd = tr.pnl_pts * cfg.multiplier
    tr.exit_date = m.date
    tr.exit_reason = reason if reason != "settle" else "expiry"
    tr.state = "CLOSED"
    tr.days_held = (_to_date(m.date) - _to_date(tr.entry_date)).days


# ---------------------------------------------------------------- run loop / stats


def run_history(store, cfg: BurritoConfig, start: int = 0, end: int = 99999999) -> List[BurritoTrade]:
    """Sequential entries: on each eligible quote date pick the expiry nearest
    cfg.entry_dte, derive direction from the spot series, enter, run to close;
    re-enter on the next quote date after the expiry (no overlap)."""
    trades: List[BurritoTrade] = []
    dates = [d for d in store.quote_dates() if start <= d <= end]
    spots = {d: store.spot(d) for d in dates}
    next_ok_idx = 0
    for i, d in enumerate(dates):
        if i < next_ok_idx:
            continue
        if spots.get(d) is None:
            continue
        if cfg.entry_dte > 0:
            cands = store.expiries_in_dte(d, 1, cfg.entry_dte + 2)
        else:
            cands = store.expiries_in_dte(d, 0, 0)
        if not cands:
            continue
        expiry = min(cands, key=lambda e: (abs(e[1] - cfg.entry_dte), e[1]))[0]
        hist = [spots[x] for x in dates[:i + 1] if spots[x] is not None]
        direction = direction_from_series(hist, cfg.direction_signal)
        if direction == "skip":
            continue
        sides: List[Side] = ["call", "put"] if direction == "both" else [direction]  # type: ignore
        for side in sides:
            tr = run_trade(store, d, expiry, side, None, cfg)
            if tr is not None:
                trades.append(tr)
        next_ok_idx = next((j for j, x in enumerate(dates) if x > expiry), len(dates))
    return trades


def summarize_burrito(trades: List[BurritoTrade]) -> Dict:
    closed = [t for t in trades if getattr(t, "pnl_usd", None) is not None]
    if not closed:
        return {"trades": 0}
    pnls = [t.pnl_usd for t in closed]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    rets = []
    for t in closed:
        basis = (getattr(t, "debit", 0.0) + getattr(t, "extra_cost", 0.0)) * 100.0
        rets.append(t.pnl_usd / basis if basis else 0.0)
    mean = sum(rets) / len(rets)
    sd = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    sharpe = (mean / sd * (252 ** 0.5)) if sd > 0 else 0.0
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    by_reason: Dict[str, int] = {}
    for t in closed:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
    return {
        "trades": len(closed),
        "net_pnl": round(sum(pnls), 2),
        "win_rate": round(len(wins) / len(closed), 4),
        "avg_pnl": round(sum(pnls) / len(closed), 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else float("inf"),
        "avg_ret": round(mean, 4),
        "exit_reasons": by_reason,
    }
