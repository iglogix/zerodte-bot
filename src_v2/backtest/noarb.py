"""Structural-bound guards for option quotes and defined-risk package marks.

Promoted to a shared primitive on purpose. The same no-arb check was previously
written inline inside one simulator; the next simulator to price a package was
written without it and booked P&L outside the structural band (failure-catalog
B1 / F1). Every simulator in this repo that marks a vertical MUST route through
`vertical_mark`, and every P&L it books MUST pass `check_pnl_bounds`.

Policy on violation: **skip, never clamp**. A violated bound means the datum is
untrustworthy, so the caller falls through to the next observation. Clamping
manufactures a plausible value that lands exactly on the bound and then looks
like a real extreme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Quote = Tuple[float, float]


def quote_ok(q: Optional[Quote]) -> bool:
    """A quote is usable iff it exists, is finite, non-negative and uncrossed."""
    if q is None:
        return False
    bid, ask = q
    if bid != bid or ask != ask:          # NaN
        return False
    if bid in (float("inf"), float("-inf")) or ask in (float("inf"), float("-inf")):
        return False
    return 0.0 <= bid <= ask


@dataclass(frozen=True)
class PackageMark:
    """Marks for a short vertical (short one leg, long the protective wing).

    bid: proceeds if you SOLD the package again right now (short_bid - long_ask)
    ask: cost to BUY the package back right now  (short_ask - long_bid)
    mid: mid-to-mid net
    """
    bid: float
    ask: float
    mid: float


def vertical_mark(
    short_q: Optional[Quote],
    long_q: Optional[Quote],
    width: float,
) -> Optional[PackageMark]:
    """Mark a short vertical, or None if the quotes violate structure.

    Structural facts used: a short vertical's value lives in [0, width]; a
    quote cannot be crossed; the package ask cannot sit below the package bid.
    """
    if width <= 0:
        return None
    if not quote_ok(short_q) or not quote_ok(long_q):
        return None
    sb, sa = short_q
    lb, la = long_q
    mid = (sb + sa) / 2.0 - (lb + la) / 2.0
    if not (0.0 <= mid <= width):
        return None
    bid = sb - la
    ask = sa - lb
    if ask < bid:
        return None
    return PackageMark(bid=bid, ask=ask, mid=mid)


def check_pnl_bounds(pnl_pts: float, credit: float, width: float, tol: float = 1e-6) -> bool:
    """Defined-risk credit spread: pnl ∈ [-(width - credit), +credit], always."""
    max_loss = width - credit
    return (-max_loss - tol) <= pnl_pts <= (credit + tol)


def assert_pnl_bounds(pnl_pts: float, credit: float, width: float, ctx: str = "") -> None:
    if not check_pnl_bounds(pnl_pts, credit, width):
        raise AssertionError(
            f"no-arb violation{' [' + ctx + ']' if ctx else ''}: "
            f"pnl_pts={pnl_pts:.4f} outside [-{width - credit:.4f}, {credit:.4f}]"
        )
