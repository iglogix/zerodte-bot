"""Detect open double calendars from broker positions, and build the transform order.

The DC Time Machine transform converts an open DC (short front P/C @ K, long back P/C @ K)
into a front-expiry iron condor: SELL the backs (the longs you hold) and BUY front wings at
K_put−W / K_call+W. Pure/testable; the Flask layer feeds it live positions + chains and turns
the OrderTicket into a manual ticket or a TastyTrade order (same submit path as entry)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

from src_v2.flux.models import ChainSnapshot
from src_v2.flux.orders import OrderLeg, OrderTicket
from src_v2.backtest.dc_time_machine.pricing import transform_credit


@dataclass(frozen=True)
class _Leg:
    root: str
    expiry: date
    strike: float
    is_call: bool
    direction: str     # "Long" | "Short"
    qty: int


@dataclass(frozen=True)
class DoubleCalendar:
    root: str
    front_expiry: date   # the SHORT (nearer) expiry
    back_expiry: date    # the LONG (farther) expiry
    k_put: float
    k_call: float
    qty: int


def parse_option_symbol(sym: str) -> Optional[tuple]:
    """OSI 21-char → (root, expiry, strike, is_call); None if not an option symbol."""
    s = sym.strip()
    if len(sym) != 21 or sym[12] not in ("C", "P"):
        return None
    try:
        root = sym[0:6].strip()
        expiry = datetime.strptime(sym[6:12], "%y%m%d").date()
        is_call = sym[12] == "C"
        strike = int(sym[13:21]) / 1000.0
    except ValueError:
        return None
    return root, expiry, strike, is_call


def _legs(positions: List[dict]) -> List[_Leg]:
    out = []
    for p in positions:
        parsed = parse_option_symbol(p.get("symbol", ""))
        if parsed is None:
            continue
        root, expiry, strike, is_call = parsed
        qty = abs(int(p.get("quantity", 0)))
        direction = p.get("direction", "")
        if qty == 0 or direction not in ("Long", "Short"):
            continue
        out.append(_Leg(root, expiry, strike, is_call, direction, qty))
    return out


def _calendars(legs: List[_Leg], is_call: bool):
    """All (root, strike, front, back, qty) calendars on one side: a Short + a Long at the
    same root/strike with short expiry < long expiry."""
    side = [l for l in legs if l.is_call == is_call]
    shorts = [l for l in side if l.direction == "Short"]
    longs = [l for l in side if l.direction == "Long"]
    out = []
    for s in shorts:
        for lg in longs:
            if s.root == lg.root and s.strike == lg.strike and s.expiry < lg.expiry:
                out.append((s.root, s.strike, s.expiry, lg.expiry, min(s.qty, lg.qty)))
    return out


def detect_double_calendars(positions: List[dict]) -> List[DoubleCalendar]:
    """Find DCs in open positions: a put-calendar and a call-calendar that share root +
    front expiry + back expiry. K_put and K_call may differ (delta-selected DC)."""
    legs = _legs(positions)
    puts = _calendars(legs, is_call=False)
    calls = _calendars(legs, is_call=True)
    out = []
    for proot, kp, pf, pb, pq in puts:
        for croot, kc, cf, cb, cq in calls:
            if proot == croot and pf == cf and pb == cb:
                out.append(DoubleCalendar(proot, pf, pb, kp, kc, min(pq, cq)))
    return out


def _mid(chain: ChainSnapshot, strike: float, is_call: bool) -> Optional[float]:
    for r in chain.rows:
        if r.strike == strike and r.is_call == is_call:
            return (r.bid + r.ask) / 2.0
    return None


def build_transform_order(dc: DoubleCalendar, wing: float,
                          front_chain: ChainSnapshot, back_chain: ChainSnapshot
                          ) -> Optional[OrderTicket]:
    """Transform: SELL the back P/C @ K (the longs), BUY front wings @ K_put−W / K_call+W.
    Priced at mids (like the entry ticket). Net is a credit when the wings cost less than the
    backs fetch. Risk-free iff (credit − original_debit) ≥ wing — we surface credit vs wing
    (original debit isn't in the position feed)."""
    pw_k, cw_k = dc.k_put - wing, dc.k_call + wing
    bp = _mid(back_chain, dc.k_put, False)     # sell back put
    bc = _mid(back_chain, dc.k_call, True)     # sell back call
    pw = _mid(front_chain, pw_k, False)        # buy put wing
    cw = _mid(front_chain, cw_k, True)         # buy call wing
    if None in (bp, bc, pw, cw):
        return None
    credit = transform_credit(bp, bc, pw, cw)  # (back bids/mids) − (wing asks/mids)
    legs = (
        OrderLeg(dc.back_expiry, dc.k_put, False, "Sell to Close", dc.qty, round(bp, 2), root=dc.root),
        OrderLeg(dc.back_expiry, dc.k_call, True, "Sell to Close", dc.qty, round(bc, 2), root=dc.root),
        OrderLeg(dc.front_expiry, pw_k, False, "Buy to Open", dc.qty, round(pw, 2), root=dc.root),
        OrderLeg(dc.front_expiry, cw_k, True, "Buy to Open", dc.qty, round(cw, 2), root=dc.root),
    )
    effect = "Credit" if credit >= 0 else "Debit"
    covers = "covers wing" if credit >= wing else "below wing"
    return OrderTicket(
        kind="transform", legs=legs, price=round(abs(credit), 2), price_effect=effect,
        est_cost=round(credit * 100 * dc.qty, 2),
        note=(f"Transform {dc.k_put:.0f}P/{dc.k_call:.0f}C wing {wing:.0f} → "
              f"{effect} {abs(credit):.2f} ({covers})"))
