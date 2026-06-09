"""Build the double-calendar order from live chains — strikes at a target delta and a
suggested net limit from the mids. Pure/testable; the Flask layer turns an OrderTicket
into a manual ticket or (in a non-manual mode) a TastyTrade complex order."""
from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def osi(self) -> str:
        return format_osi(self.expiry, self.strike, self.is_call)

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
