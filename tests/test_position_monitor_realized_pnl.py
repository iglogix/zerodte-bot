"""Realized-P&L booking and the past-expiry invariant on PositionMonitor.

Both defects these pin were found on 2026-09-08 while trying to answer "what is
ORB's performance so far?" and discovering the answer was not in the database:

  1. `realized_pnl` is declared on IronCondorTrade but was never assigned by any
     code path, so every row in every bot DB was NULL and P&L could only be
     recovered by parsing execution-comparison JSONL and daily logs.
  2. Two ORB trades (`orbdir-call-093125` 07-09→07-10, `orbdir-call-144640`
     07-29→07-30) were held past their own 0DTE expiration into the NEXT
     session and booked as a routine `eod_forced_close`. The 07-09 one was a
     deterministic max loss (short call 7515 vs SPX 7577.98).

The rule for #1 is "never fabricate": a close with no fill price (manual /
dry_run / an already-flat leg) must leave realized_pnl NULL rather than book a
plausible-looking number. A wrong P&L is worse than a missing one.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zerodte.services.position_monitor import PositionMonitor, TradeOutlivedExpiry

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _call_spread(credit: float, short_k: int = 7680, long_k: int = 7700,
                 qty: int = 1, exp: date = date(2026, 8, 26)) -> dict:
    return {
        "trade_id": "t-call",
        "expiration_date": exp,
        "total_premium": credit,
        "short_call_symbol": f"SPXW  260826C0{short_k}000",
        "long_call_symbol": f"SPXW  260826C0{long_k}000",
        "call_open_qty": qty,
    }


def _put_spread(credit: float, short_k: int = 7670, long_k: int = 7650,
                qty: int = 1, exp: date = date(2026, 8, 20)) -> dict:
    return {
        "trade_id": "t-put",
        "expiration_date": exp,
        "total_premium": credit,
        "short_put_symbol": f"SPXW  260820P0{short_k}000",
        "long_put_symbol": f"SPXW  260820P0{long_k}000",
        "put_open_qty": qty,
    }


def _fills(trade: dict, short_px: float, long_px: float) -> list[dict]:
    short_sym = trade.get("short_call_symbol") or trade.get("short_put_symbol")
    long_sym = trade.get("long_call_symbol") or trade.get("long_put_symbol")
    return [
        {"symbol": short_sym, "filled": True, "fill_price": short_px},
        {"symbol": long_sym, "filled": True, "fill_price": long_px},
    ]


# -----------------------------------------------------------------------------
# realized P&L
# -----------------------------------------------------------------------------

def test_call_credit_spread_loss_is_booked():
    """The real 2026-08-26 ORB trade: credit 4.20, breach exit 10.50/3.00."""
    t = _call_spread(4.20)
    pnl = PositionMonitor._realized_pnl(t, _fills(t, 10.50, 3.00))
    assert pnl == pytest.approx(-330.0)


def test_put_credit_spread_win_is_booked():
    """The real 2026-08-20 ORB trade: credit 4.00, TP exit 6.30/3.10."""
    t = _put_spread(4.00)
    pnl = PositionMonitor._realized_pnl(t, _fills(t, 6.30, 3.10))
    assert pnl == pytest.approx(80.0)


def test_quantity_scales_pnl():
    t = _call_spread(4.20, qty=3)
    assert PositionMonitor._realized_pnl(t, _fills(t, 10.50, 3.00)) == pytest.approx(-990.0)


def test_manual_mode_close_books_nothing():
    """Manual mode never submits, so there is no fill and no P&L to book.

    This is the case that produced 26 NULL rows of phantom 'performance'.
    """
    t = _call_spread(4.20)
    short_sym = t["short_call_symbol"]
    long_sym = t["long_call_symbol"]
    results = [
        {"symbol": short_sym, "manual": True, "submitted": False},
        {"symbol": long_sym, "manual": True, "submitted": False},
    ]
    assert PositionMonitor._realized_pnl(t, results) is None


def test_partial_fill_information_books_nothing():
    """One leg filled, the other already flat -> unknowable, must stay None."""
    t = _call_spread(4.20)
    results = [
        {"symbol": t["short_call_symbol"], "filled": True, "fill_price": 10.50},
        {"symbol": t["long_call_symbol"], "status": "no_position", "skipped": True},
    ]
    assert PositionMonitor._realized_pnl(t, results) is None


def test_missing_entry_credit_books_nothing():
    t = _call_spread(4.20)
    t["total_premium"] = None
    assert PositionMonitor._realized_pnl(t, _fills(t, 10.50, 3.00)) is None


def test_duplicate_leg_fills_are_rejected():
    """A leg must appear exactly once; a repeat means retry noise, not a fill.

    The 08-26 log holds 150 repeat close attempts on an already-closed leg.
    Silently taking the first would work by accident; assert instead.
    """
    t = _call_spread(4.20)
    results = _fills(t, 10.50, 3.00) + [
        {"symbol": t["short_call_symbol"], "filled": True, "fill_price": 9.99},
    ]
    assert PositionMonitor._realized_pnl(t, results) is None


def test_pnl_is_bounded_by_the_spread():
    """P&L can never exceed the credit or fall below credit - width."""
    t = _call_spread(4.20)
    pnl = PositionMonitor._realized_pnl(t, _fills(t, 10.50, 3.00))
    width = 7700 - 7680
    assert -(width - 4.20) * 100 <= pnl <= 4.20 * 100


def test_reproduces_the_six_real_alpaca_trades():
    """Regression against the reconstructed live record (2026-08-20..09-04).

    These are the actual fills from
    logs/alpaca/execution_comparison/orb_em_directional/*.jsonl. If this drifts,
    the booking formula changed and every historical number moves with it.
    """
    cases = [
        # credit, short_exit, long_exit, expected
        (4.00, 6.30, 3.10, 80.0),    # 08-20 put  TP
        (4.20, 10.50, 3.00, -330.0),  # 08-26 call breach
        (5.10, 5.40, 1.90, 160.0),   # 08-27 call TP
        (4.40, 7.40, 4.10, 110.0),   # 08-28 put  TP
        (3.75, 3.90, 1.05, 90.0),    # 08-31 call TP
        (4.40, 4.60, 1.20, 100.0),   # 09-04 call TP
    ]
    total = 0.0
    for credit, sx, lx, expected in cases:
        t = _call_spread(credit)
        pnl = PositionMonitor._realized_pnl(t, _fills(t, sx, lx))
        assert pnl == pytest.approx(expected), f"credit {credit}"
        total += pnl
    assert total == pytest.approx(210.0)


# -----------------------------------------------------------------------------
# past-expiry invariant
# -----------------------------------------------------------------------------

def test_same_day_close_is_allowed():
    t = _call_spread(4.20, exp=date(2026, 8, 26))
    PositionMonitor._check_not_past_expiry(t, datetime(2026, 8, 26, 15, 50))


def test_close_before_expiry_is_allowed():
    t = _call_spread(4.20, exp=date(2026, 8, 28))
    PositionMonitor._check_not_past_expiry(t, datetime(2026, 8, 26, 15, 50))


def test_close_after_expiry_raises():
    """The real orbdir-call-093125: 0DTE entered 07-09, closed 07-10 15:50."""
    t = _call_spread(5.30, exp=date(2026, 7, 9))
    with pytest.raises(TradeOutlivedExpiry) as exc:
        PositionMonitor._check_not_past_expiry(t, datetime(2026, 7, 10, 15, 50))
    assert "2026-07-09" in str(exc.value)
    assert "2026-07-10" in str(exc.value)


def test_expiry_accepts_iso_string_dates():
    """SQLite round-trips can hand back a string; the guard must not silently pass."""
    t = _call_spread(5.30, exp="2026-07-09")
    with pytest.raises(TradeOutlivedExpiry):
        PositionMonitor._check_not_past_expiry(t, datetime(2026, 7, 10, 15, 50))


def test_missing_expiry_does_not_raise():
    t = _call_spread(4.20)
    t["expiration_date"] = None
    PositionMonitor._check_not_past_expiry(t, datetime(2026, 8, 26, 15, 50))


# -----------------------------------------------------------------------------
# symbol-format mismatch between the trade record and the broker's echo
# -----------------------------------------------------------------------------

def test_alpaca_symbol_format_still_matches_occ_leg():
    """Alpaca echoes `SPXW260826C07680000`; the trade holds `SPXW  260826C...`.

    Matching on the raw echo would silently book nothing for every Alpaca
    trade — the exact run whose P&L we care about.
    """
    t = _call_spread(4.20)
    results = [
        {"symbol": "SPXW260826C07680000", "filled": True, "fill_price": 10.50},
        {"symbol": "SPXW260826C07700000", "filled": True, "fill_price": 3.00},
    ]
    assert PositionMonitor._realized_pnl(t, results) == pytest.approx(-330.0)


def test_leg_symbol_stamp_wins_over_broker_echo():
    """_close_leg stamps the canonical symbol it was asked to close."""
    t = _call_spread(4.20)
    results = [
        {"leg_symbol": t["short_call_symbol"], "symbol": "WHATEVER",
         "filled": True, "fill_price": 10.50},
        {"leg_symbol": t["long_call_symbol"], "symbol": "WHATEVER2",
         "filled": True, "fill_price": 3.00},
    ]
    assert PositionMonitor._realized_pnl(t, results) == pytest.approx(-330.0)


# -----------------------------------------------------------------------------
# the funnel itself — every close site must reach this
# -----------------------------------------------------------------------------

class _FakeClock:
    def __init__(self, now: datetime):
        self._now = now

    def now(self) -> datetime:
        return self._now


class _FakeDb:
    def __init__(self):
        self.updates: dict = {}

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        self.updates[trade_id] = updates
        return True


def _monitor(now: datetime) -> PositionMonitor:
    m = PositionMonitor.__new__(PositionMonitor)
    m.db = _FakeDb()
    m._clock = _FakeClock(now)
    return m


@pytest.mark.asyncio
async def test_finalize_books_pnl_and_exit_time():
    m = _monitor(datetime(2026, 8, 26, 9, 44))
    t = _call_spread(4.20)
    await m._finalize_trade(t, "call breach", _fills(t, 10.50, 3.00))
    u = m.db.updates["t-call"]
    assert u["status"] == "closed"
    assert u["close_reason"] == "call breach"
    assert u["realized_pnl"] == pytest.approx(-330.0)
    assert u["exit_time"] == datetime(2026, 8, 26, 9, 44)


@pytest.mark.asyncio
async def test_finalize_omits_pnl_when_unbookable():
    """Manual mode must still close the row — just without a fabricated P&L."""
    m = _monitor(datetime(2026, 8, 26, 15, 50))
    t = _call_spread(4.20)
    await m._finalize_trade(t, "eod_forced_close", None)
    u = m.db.updates["t-call"]
    assert u["status"] == "closed"
    assert "realized_pnl" not in u
    assert u["exit_time"] == datetime(2026, 8, 26, 15, 50)


@pytest.mark.asyncio
async def test_finalize_flags_a_past_expiry_close_without_crashing():
    """The 07-09 case: still closes the row, but the reason is marked.

    Raising here would strand every other open position, so the invariant is
    loud (ERROR log + reason tag) rather than fatal.
    """
    m = _monitor(datetime(2026, 7, 10, 15, 50))
    t = _call_spread(5.30, exp=date(2026, 7, 9))
    await m._finalize_trade(t, "eod_forced_close", None)
    u = m.db.updates["t-call"]
    assert u["status"] == "closed"
    assert "PAST_EXPIRY" in u["close_reason"]


@pytest.mark.asyncio
async def test_finalize_marks_expired_status():
    m = _monitor(datetime(2026, 8, 26, 16, 20))
    t = _call_spread(4.20)
    await m._finalize_trade(t, "option_expiry", None, status="expired")
    assert m.db.updates["t-call"]["status"] == "expired"


def test_pnl_is_rounded_to_cents():
    """Option prices are 2dp; (3.90-2.80)*100 must book $110.00, not 109.99999997.

    The real 2026-09-08 ORB trade. Sub-cent float noise in a currency column is
    noise that later reads as precision.
    """
    t = _put_spread(3.90, short_k=7690, long_k=7670)
    pnl = PositionMonitor._realized_pnl(t, _fills(t, 3.10, 0.30))
    assert pnl == 110.0
