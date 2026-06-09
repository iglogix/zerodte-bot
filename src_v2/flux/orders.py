"""Build the double-calendar order from live chains — strikes at a target delta and a
suggested net limit from the mids. Pure/testable; the Flask layer turns an OrderTicket
into a manual ticket or (in a non-manual mode) a TastyTrade complex order."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Optional, Tuple

from src_v2.flux.models import ChainSnapshot
from src_v2.backtest.osi_symbol import format_osi


@dataclass(frozen=True)
class OrderLeg:
    expiry: date
    strike: float
    is_call: bool
    action: str        # "Sell to Open" | "Buy to Open" | "Sell to Close" | "Buy to Close"
    quantity: int
    limit: float = 0.0  # this leg's mid (used by leg-in execution to price each leg)
    root: str = "SPXW"  # underlying root for the OSI symbol (SPY for the cert proxy)

    @property
    def osi(self) -> str:
        return format_osi(self.expiry, self.strike, self.is_call, root=self.root)

    @property
    def is_long(self) -> bool:
        return self.action.startswith("Buy")


@dataclass(frozen=True)
class OrderTicket:
    kind: str                       # "double_calendar" | "transform"
    legs: Tuple[OrderLeg, ...]
    price: float                    # net limit, $/share (per contract /100)
    price_effect: str               # "Debit" | "Credit"
    est_cost: float                 # price * 100 * qty (signed by effect)
    note: str


def _mid(chain: ChainSnapshot, strike: float, is_call: bool) -> Optional[float]:
    for r in chain.rows:
        if r.strike == strike and r.is_call == is_call:
            return (r.bid + r.ask) / 2.0
    return None


def select_dc_strikes(chain: ChainSnapshot, target_delta: float = 0.35
                      ) -> Tuple[Optional[float], Optional[float]]:
    """K_put nearest −target_delta, K_call nearest +target_delta on `chain`'s skew."""
    calls = [r for r in chain.rows if r.is_call]
    puts = [r for r in chain.rows if not r.is_call]
    if not calls or not puts:
        return None, None
    kc = min(calls, key=lambda r: abs(r.delta - target_delta)).strike
    kp = min(puts, key=lambda r: abs(r.delta + target_delta)).strike
    return kp, kc


def build_dc_order(front_chain: ChainSnapshot, back_chain: ChainSnapshot,
                   k_put: float, k_call: float, qty: int = 1) -> Optional[OrderTicket]:
    """Double calendar: sell front put/call @ K, buy back put/call @ K. Limit = net debit
    at mids = (back puts+calls) − (front puts+calls)."""
    fp, fc = _mid(front_chain, k_put, False), _mid(front_chain, k_call, True)
    bp, bc = _mid(back_chain, k_put, False), _mid(back_chain, k_call, True)
    if None in (fp, fc, bp, bc):
        return None
    debit = (bp + bc) - (fp + fc)
    legs = (
        OrderLeg(front_chain.expiry, k_put, False, "Sell to Open", qty, round(fp, 2)),
        OrderLeg(back_chain.expiry, k_put, False, "Buy to Open", qty, round(bp, 2)),
        OrderLeg(front_chain.expiry, k_call, True, "Sell to Open", qty, round(fc, 2)),
        OrderLeg(back_chain.expiry, k_call, True, "Buy to Open", qty, round(bc, 2)),
    )
    return OrderTicket(
        kind="double_calendar", legs=legs, price=round(debit, 2),
        price_effect="Debit" if debit >= 0 else "Credit",
        est_cost=round(debit * 100 * qty, 2),
        note=f"DC {front_chain.expiry}/{back_chain.expiry} {k_put:.0f}P / {k_call:.0f}C x{qty}")


def spy_proxy(ticket: OrderTicket) -> OrderTicket:
    """Map an SPX/SPXW ticket to a SPY-equivalent proxy (SPY ≈ SPX/10) for the TastyTrade
    cert env, which doesn't list SPXW. Strikes ÷10 → nearest $1 (a real SPY strike); each
    leg's mid and the net limit ÷10. Same expiries/actions/structure, root → SPY."""
    legs = tuple(
        replace(l, strike=float(round(l.strike / 10.0)), limit=round(l.limit / 10.0, 2),
                root="SPY")
        for l in ticket.legs)
    kp = next((l.strike for l in legs if not l.is_call), 0.0)
    kc = next((l.strike for l in legs if l.is_call), 0.0)
    qty = legs[0].quantity if legs else 1
    return OrderTicket(
        kind=ticket.kind, legs=legs, price=round(ticket.price / 10.0, 2),
        price_effect=ticket.price_effect, est_cost=round(ticket.est_cost / 10.0, 2),
        note=f"SPY proxy: {kp:.0f}P / {kc:.0f}C x{qty} (of {ticket.note})")
