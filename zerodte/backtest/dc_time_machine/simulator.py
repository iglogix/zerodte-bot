# zerodte/backtest/dc_time_machine/simulator.py
"""Per-trade minute-stepped lifecycle for the DC Time Machine.

simulate_trade is deliberately given resolved strikes/expiries and an explicit
minute list so it is fully deterministic and unit-testable without Databento.
Higher layers (run_backtest / wf_cv) resolve those via resolve_expiries /
select_strikes / session_minutes and then call this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from zerodte.backtest.dc_time_machine.fills import fill_price, net_combo_credit
from zerodte.backtest.dc_time_machine.policy import entry_allowed, overnight_decision
from zerodte.backtest.dc_time_machine.pricing import (
    entry_debit, ic_terminal_pnl, is_risk_free,
)
from zerodte.backtest.dc_time_machine.strategy import DcParams
from zerodte.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore


@dataclass
class TradeResult:
    session: date
    front: date
    back: date
    k_put: float
    k_call: float
    outcome: str                  # transformed | stopped | closed_eod | held_overnight | no_entry
    entry_debit: float = 0.0
    credit: float = 0.0
    bank: float = 0.0
    transform_minute: Optional[str] = None
    settlement: Optional[float] = None
    pnl: float = 0.0              # final $/contract
    params_wing: float = 0.0
    tags: Dict = field(default_factory=dict)


def _q(store, expiry, minute, strike, is_call):
    return store.lookup(expiry, minute, strike, is_call)


def simulate_trade(store: MultiExpiryStore, params: DcParams, *, session: date,
                   minutes: List[str], k_put: float, k_call: float,
                   front: date, back: date,
                   features: Optional[Dict] = None, is_fri_mon: bool = False,
                   ratio: float = float("nan")) -> TradeResult:
    store.load_session(session)
    wing = params.wing_width
    k = params.fill_k
    feats = dict(features or {})
    feats.setdefault("is_fri_mon", is_fri_mon)
    feats.setdefault("ratio", ratio)
    res = TradeResult(session=session, front=front, back=back, k_put=k_put,
                      k_call=k_call, outcome="no_entry", params_wing=wing, tags=feats)

    if not entry_allowed(params.entry_filter, feats):
        return res

    em = params.entry_minute
    fpb = _q(store, front, em, k_put, False)      # front put (sell)
    fcb = _q(store, front, em, k_call, True)      # front call (sell)
    bpa = _q(store, back, em, k_put, False)       # back put (buy)
    bca = _q(store, back, em, k_call, True)       # back call (buy)
    if not all((fpb, fcb, bpa, bca)):
        return res

    # Entry debit with fills: sell fronts @ sell-fill, buy backs @ buy-fill
    D = entry_debit(
        front_put_bid=fill_price(*fpb, "sell", k),
        front_call_bid=fill_price(*fcb, "sell", k),
        back_put_ask=fill_price(*bpa, "buy", k),
        back_call_ask=fill_price(*bca, "buy", k),
    )
    res.entry_debit = D
    threshold = D + wing + params.transform_epsilon

    later = [m for m in minutes if m > em]
    for m in later:
        # Mark DC to close: sell backs @ sell-fill, buy-to-close fronts @ buy-fill
        bp = _q(store, back, m, k_put, False)
        bc = _q(store, back, m, k_call, True)
        fp = _q(store, front, m, k_put, False)
        fc = _q(store, front, m, k_call, True)
        if not all((bp, bc, fp, fc)):
            continue
        v_close = (fill_price(*bp, "sell", k) + fill_price(*bc, "sell", k)
                   - fill_price(*fp, "buy", k) - fill_price(*fc, "buy", k))
        dc_pnl = v_close - D
        if D > 0 and dc_pnl <= -params.stop_loss_frac * D:
            res.outcome = "stopped"
            res.pnl = round(dc_pnl * 100, 2)
            return res

        # Transform credit (atomic combo): sell backs, buy front wings
        pw = _q(store, front, m, k_put - wing, False)
        cw = _q(store, front, m, k_call + wing, True)
        if pw is None or cw is None:
            continue
        legs = [(*bp, "sell"), (*bc, "sell"), (*pw, "buy"), (*cw, "buy")]
        C = net_combo_credit(legs, k)
        if C >= threshold:
            assert is_risk_free(D, C, wing), "transform fired without risk-free invariant"
            res.outcome = "transformed"
            res.transform_minute = m
            res.credit = C
            res.bank = C - D
            S = store.settlement(front)
            res.settlement = S
            if S is None:
                res.pnl = round(res.bank * 100, 2)   # conservative: no settlement → bank only
            else:
                res.pnl = round(ic_terminal_pnl(S, k_put, k_call, wing, res.bank) * 100, 2)
            return res

    # EOD, not transformed → overnight seam (v1 default: close)
    state = {"dc_pnl_frac": (v_close - D) / D if D else 0.0, "ratio": feats.get("ratio", 0.0)}
    if overnight_decision(params.overnight_policy, state) == "hold":
        res.outcome = "held_overnight"
        res.pnl = round((v_close - D) * 100, 2)   # marked; multi-day hold handled by caller
        return res
    res.outcome = "closed_eod"
    res.pnl = round((v_close - D) * 100, 2)
    return res


# --- append to zerodte/backtest/dc_time_machine/simulator.py ---
from typing import Tuple

from zerodte.backtest.dc_time_machine.iv import bs_delta, implied_vol, front_back_ratio


def resolve_expiries(store: MultiExpiryStore, session: date, params: DcParams,
                     dte_tolerance: int = 2) -> Tuple[Optional[date], Optional[date]]:
    """Pick front/back expiries by TRADING-DAY distance (robust to sparse caches).

    front  = available expiry nearest `front_dte` trading days after session;
    back   = available expiry after front nearest `front_dte + gap` trading days out.
    Returns (None, None) if the nearest match is off by more than `dte_tolerance`
    trading days (no suitable expiry was cached).
    """
    from zerodte.backtest.dc_time_machine import trading_calendar as _cal

    avail = [e for e in store.available_expiries() if e > session]
    if not avail:
        return None, None
    front = min(avail, key=lambda e: abs(_cal.trading_days_after(session, e) - params.front_dte))
    if abs(_cal.trading_days_after(session, front) - params.front_dte) > dte_tolerance:
        return None, None
    later = [e for e in avail if e > front]
    if not later:
        return None, None
    target_back = params.front_dte + params.gap
    back = min(later, key=lambda e: abs(_cal.trading_days_after(session, e) - target_back))
    if abs(_cal.trading_days_after(session, back) - target_back) > dte_tolerance:
        return None, None
    return front, back


def _T(session: date, expiry: date) -> float:
    return max((expiry - session).days, 1) / 365.0


def select_strikes(store: MultiExpiryStore, session: date, front: date,
                   params: DcParams, spot: float) -> Tuple[Optional[float], Optional[float]]:
    """Pick K_p (put, below spot) and K_c (call, above spot) nearest the target abs delta.

    Delta is computed with each strike's own implied vol at the entry minute; falls
    back to scanning available strikes. Returns (None, None) if no usable strikes.
    """
    em = params.entry_minute
    T = _T(session, front)
    ks = sorted(store.strikes(front))
    best_put = best_call = None
    bp_err = bc_err = 1e9
    for K in ks:
        for is_call in (True, False):
            q = store.lookup(front, em, K, is_call)
            if q is None:
                continue
            mid = (q[0] + q[1]) / 2
            iv = implied_vol(mid, spot, K, T, params.rate, is_call)
            if iv != iv:   # NaN
                continue
            d = bs_delta(spot, K, T, iv, params.rate, is_call)
            err = abs(abs(d) - params.target_delta)
            if is_call and K > spot and err < bc_err:
                bc_err, best_call = err, K
            if (not is_call) and K < spot and err < bp_err:
                bp_err, best_put = err, K
    return best_put, best_call


def compute_iv_features(store: MultiExpiryStore, session: date, front: date,
                        back: date, minute: str, spot: float, params: DcParams) -> Dict:
    """Reconstruct the front/back ATM IV ratio (the 'Flux' signal) + day-combo flags."""
    def atm_iv(expiry: date) -> float:
        ks = store.strikes(expiry)
        if not ks:
            return float("nan")
        atm = min(ks, key=lambda x: abs(x - spot))
        ivs = []
        T = _T(session, expiry)
        for is_call in (True, False):
            q = store.lookup(expiry, minute, atm, is_call)
            if q is None:
                continue
            mid = (q[0] + q[1]) / 2
            iv = implied_vol(mid, spot, atm, T, params.rate, is_call)
            if iv == iv:
                ivs.append(iv)
        return sum(ivs) / len(ivs) if ivs else float("nan")

    fr, bk = atm_iv(front), atm_iv(back)
    ratio = front_back_ratio(fr, bk)
    is_fri_mon = (session.weekday() == 4 or front.weekday() == 4) and back.weekday() == 0
    return {
        "front_iv": fr, "back_iv": bk, "ratio": ratio,
        "is_backwardation": (ratio == ratio and ratio > 1.0),
        "is_fri_mon": bool(is_fri_mon),
        "ratio_z": 0.0, "ratio_falling": False,   # filled by flux_scanner history when available
    }


# --- append to zerodte/backtest/dc_time_machine/simulator.py ---

def session_minutes(store: MultiExpiryStore, front: date) -> List[str]:
    """Sorted unique minute strings present for the front expiry of the loaded session."""
    return sorted({k[0] for k in store._panel.get(front, {})})


def run_session(store: MultiExpiryStore, params: DcParams, session: date
                ) -> Optional[TradeResult]:
    """Resolve a single DC trade for `session` and simulate it. None if no expiries."""
    store.load_session(session)
    front, back = resolve_expiries(store, session, params)
    if front is None or back is None:
        return None
    spot = store.derive_spot(front, params.entry_minute, ref=_panel_ref(store, front))
    if spot is None:
        return TradeResult(session=session, front=front, back=back, k_put=0.0,
                           k_call=0.0, outcome="no_entry", params_wing=params.wing_width)
    k_put, k_call = select_strikes(store, session, front, params, spot)
    if k_put is None or k_call is None:
        return TradeResult(session=session, front=front, back=back, k_put=0.0,
                           k_call=0.0, outcome="no_entry", params_wing=params.wing_width)
    feats = compute_iv_features(store, session, front, back, params.entry_minute, spot, params)
    minutes = session_minutes(store, front)
    return simulate_trade(store, params, session=session, minutes=minutes,
                          k_put=k_put, k_call=k_call, front=front, back=back,
                          features=feats, is_fri_mon=feats["is_fri_mon"],
                          ratio=feats["ratio"])


def _panel_ref(store: MultiExpiryStore, front: date) -> float:
    """A rough ATM reference = median available strike for `front` (for spot derivation)."""
    ks = sorted(store.strikes(front))
    return ks[len(ks) // 2] if ks else 0.0
