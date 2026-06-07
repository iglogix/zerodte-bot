# src_v2/backtest/dc_time_machine/simulator.py
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

from src_v2.backtest.dc_time_machine.fills import fill_price, net_combo_credit
from src_v2.backtest.dc_time_machine.policy import entry_allowed, overnight_decision
from src_v2.backtest.dc_time_machine.pricing import (
    entry_debit, ic_terminal_pnl, is_risk_free,
)
from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore


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
