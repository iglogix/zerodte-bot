"""Detect open double calendars from broker positions, and build the transform order.

The DC Time Machine transform converts an open DC (short front P/C @ K, long back P/C @ K)
into a front-expiry iron condor: SELL the backs (the longs you hold) and BUY front wings at
K_put−W / K_call+W. Pure/testable; the Flask layer feeds it live positions + chains and turns
the OrderTicket into a manual ticket or a TastyTrade order (same submit path as entry)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

from zerodte.backtest.dc_time_machine.pricing import transform_credit
from zerodte.flux.models import ChainSnapshot
from zerodte.flux.orders import OrderLeg, OrderTicket


@dataclass(frozen=True)
class _Leg:
    root: str
    expiry: date
    strike: float
    is_call: bool
    direction: str     # "Long" | "Short"
    qty: int
    open_price: float = 0.0   # avg open price per share (cost basis)


@dataclass(frozen=True)
class DoubleCalendar:
    root: str
    front_expiry: date   # the SHORT (nearer) expiry
    back_expiry: date    # the LONG (farther) expiry
    k_put: float
    k_call: float
    qty: int
    entry_debit: float = 0.0   # net debit paid to open ($/share): longs − shorts; 0 if unknown


@dataclass(frozen=True)
class WingRow:
    """One candidate wing W for transforming a DC. All $/share unless noted."""
    wing: float
    credit: float        # achievable transform credit: backs @ bid − wings @ ask
    banked: float        # credit − entry_debit (cash net of the sunk DC debit)
    floor: float         # banked − wing  → worst-case terminal P&L (deep ITM through a wing)
    ceiling: float       # banked         → best case (settles between the short strikes)
    risk_free: bool      # floor ≥ 0


def parse_option_symbol(sym: str) -> Optional[tuple]:
    """OSI 21-char → (root, expiry, strike, is_call); None if not an option symbol."""
    sym.strip()
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
        out.append(_Leg(root, expiry, strike, is_call, direction, qty,
                        float(p.get("avg_open_price", 0) or 0)))
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


def _open(legs: List[_Leg], root, expiry, strike, is_call) -> float:
    for l in legs:
        if l.root == root and l.expiry == expiry and l.strike == strike and l.is_call == is_call:
            return l.open_price
    return 0.0


def detect_double_calendars(positions: List[dict]) -> List[DoubleCalendar]:
    """Find DCs in open positions: a put-calendar and a call-calendar that share root +
    front expiry + back expiry. K_put and K_call may differ (delta-selected DC). entry_debit
    is reconstructed from cost basis: (long backs) − (short fronts); 0 if cost basis absent."""
    legs = _legs(positions)
    puts = _calendars(legs, is_call=False)
    calls = _calendars(legs, is_call=True)
    out = []
    for proot, kp, pf, pb, pq in puts:
        for croot, kc, cf, cb, cq in calls:
            if proot == croot and pf == cf and pb == cb:
                debit = ((_open(legs, proot, pb, kp, False) + _open(legs, croot, cb, kc, True))
                         - (_open(legs, proot, pf, kp, False) + _open(legs, croot, cf, kc, True)))
                out.append(DoubleCalendar(proot, pf, pb, kp, kc, min(pq, cq), round(debit, 2)))
    return out


def _quote(chain: ChainSnapshot, strike: float, is_call: bool, which: str) -> Optional[float]:
    for r in chain.rows:
        if r.strike == strike and r.is_call == is_call:
            return r.bid if which == "bid" else r.ask
    return None


def analyze_transform(dc: DoubleCalendar, entry_debit: float,
                      front_chain: ChainSnapshot, back_chain: ChainSnapshot,
                      max_wing: Optional[float] = None, buffer: float = 0.0):
    """Sweep listed wings W and report the risk-free frontier of the transform.

    For each W with both K_put−W and K_call+W listed on the front chain, the achievable
    credit is C(W) = (back_put_bid + back_call_bid) − (put_wing_ask + call_wing_ask) — i.e.
    the CROSS-fill credit (= the backtest's k=1 net combo credit). The resulting front IC has
    worst-case floor = (C−D) − W and best-case ceiling = (C−D). `buffer` is the required
    cushion ($/share) over break-even: a wing is risk_free iff floor ≥ buffer, and W* is the
    widest such wing. buffer=0 = bare break-even at cross."""
    bpb = _quote(back_chain, dc.k_put, False, "bid")
    bcb = _quote(back_chain, dc.k_call, True, "bid")
    if bpb is None or bcb is None:
        return [], None
    backs = bpb + bcb
    put_strikes = {r.strike for r in front_chain.rows if not r.is_call}
    call_strikes = {r.strike for r in front_chain.rows if r.is_call}
    cand = sorted({dc.k_put - kp for kp in put_strikes
                   if kp < dc.k_put and (dc.k_call + (dc.k_put - kp)) in call_strikes})
    rows: List[WingRow] = []
    for w in cand:
        if max_wing is not None and w > max_wing:
            continue
        pw = _quote(front_chain, dc.k_put - w, False, "ask")
        cw = _quote(front_chain, dc.k_call + w, True, "ask")
        if pw is None or cw is None:
            continue
        credit = backs - (pw + cw)
        banked = credit - entry_debit
        floor = banked - w
        rows.append(WingRow(wing=w, credit=round(credit, 2), banked=round(banked, 2),
                            floor=round(floor, 2), ceiling=round(banked, 2),
                            risk_free=floor >= buffer))
    max_rf = max((r.wing for r in rows if r.risk_free), default=None)
    return rows, max_rf


def _mid(chain: ChainSnapshot, strike: float, is_call: bool) -> Optional[float]:
    for r in chain.rows:
        if r.strike == strike and r.is_call == is_call:
            return (r.bid + r.ask) / 2.0
    return None


def _legpx(chain: ChainSnapshot, strike: float, is_call: bool, side: str, fill: str
          ) -> Optional[float]:
    """Per-leg price. fill='cross' = worst realistic (sell @ bid, buy @ ask); 'mid' = mid."""
    for r in chain.rows:
        if r.strike == strike and r.is_call == is_call:
            if fill == "mid":
                return (r.bid + r.ask) / 2.0
            return r.bid if side == "sell" else r.ask
    return None


def build_transform_order(dc: DoubleCalendar, wing: float,
                          front_chain: ChainSnapshot, back_chain: ChainSnapshot,
                          entry_debit: float = 0.0, fill: str = "cross") -> Optional[OrderTicket]:
    """Transform: SELL the back P/C @ K (the longs), BUY front wings @ K_put−W / K_call+W.

    Priced at CROSS by default (sell @ bid, buy @ ask) so the submitted limit matches the
    risk-free guarantee the analyzer computes — you can only fill at this credit or better.
    The note reports the mid credit as the price-improvement target, plus banked = credit − D
    and floor = banked − wing (risk-free iff floor ≥ 0)."""
    pw_k, cw_k = dc.k_put - wing, dc.k_call + wing
    bp = _legpx(back_chain, dc.k_put, False, "sell", fill)     # sell back put
    bc = _legpx(back_chain, dc.k_call, True, "sell", fill)     # sell back call
    pw = _legpx(front_chain, pw_k, False, "buy", fill)         # buy put wing
    cw = _legpx(front_chain, cw_k, True, "buy", fill)          # buy call wing
    if None in (bp, bc, pw, cw):
        return None
    credit = transform_credit(bp, bc, pw, cw)                  # at the chosen fill (limit price)
    # mid credit shown as the upside if the combo gets price improvement
    mids = [_mid(back_chain, dc.k_put, False), _mid(back_chain, dc.k_call, True),
            _mid(front_chain, pw_k, False), _mid(front_chain, cw_k, True)]
    credit_mid = (mids[0] + mids[1]) - (mids[2] + mids[3]) if None not in mids else credit
    legs = (
        OrderLeg(dc.back_expiry, dc.k_put, False, "Sell to Close", dc.qty, round(bp, 2), root=dc.root),
        OrderLeg(dc.back_expiry, dc.k_call, True, "Sell to Close", dc.qty, round(bc, 2), root=dc.root),
        OrderLeg(dc.front_expiry, pw_k, False, "Buy to Open", dc.qty, round(pw, 2), root=dc.root),
        OrderLeg(dc.front_expiry, cw_k, True, "Buy to Open", dc.qty, round(cw, 2), root=dc.root),
    )
    effect = "Credit" if credit >= 0 else "Debit"
    banked = credit - entry_debit
    floor = banked - wing
    tag = "RISK-FREE" if floor >= 0 else f"floor {floor:+.2f}"
    return OrderTicket(
        kind="transform", legs=legs, price=round(abs(credit), 2), price_effect=effect,
        est_cost=round(credit * 100 * dc.qty, 2),
        note=(f"Transform {dc.k_put:.0f}P/{dc.k_call:.0f}C W{wing:.0f} [{fill}] → {effect} "
              f"{abs(credit):.2f} (mid {credit_mid:+.2f}) | banked {banked:+.2f} | "
              f"floor {floor:+.2f} ({tag})"))
