"""Tests for the MEIC `take_profit` exit mode (early-entry IV-collapse study).

Semantics being pinned:
  - exit_mode="take_profit": close a side FULLY once its profit reaches
    tp_pct_of_credit; NO stop-loss, NO partial scale-out tiers.
  - A losing side (short_ask >> credit) is NOT stopped — it returns "hold" and
    rides to EOD, where the engine settles it at expiry INTRINSIC (tested at the
    engine level separately; the danger is booking it as a worthless win).
  - hold_to_expiry / managed behaviour is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_v2.strategies.meic import MeicStrategy


def _strategy(exit_mode: str = "take_profit", tp_pct: float = 0.50) -> MeicStrategy:
    s = MeicStrategy()
    s.configure(
        {
            "strategy": {"exit_mode": exit_mode, "tp_pct_of_credit": tp_pct},
            "options": {},
        }
    )
    return s


def _call_trade() -> dict:
    # short call credit 3.0, paired long 15pt OTM
    return {
        "trade_id": "t1",
        "short_call_symbol": "SPXW  240320C05000000",
        "long_call_symbol": "SPXW  240320C05015000",
        "call_credit": 3.0,
        "call_open_qty": 4,
        "short_call_stop_price": 9.0,  # managed would stop at short_ask>=9
        "entry_time_et": None,
    }


def _quotes(short_ask: float, long_bid: float = 0.5) -> dict:
    return {
        "SPXW  240320C05000000": {"bid": short_ask - 0.2, "ask": short_ask},
        "SPXW  240320C05015000": {"bid": long_bid, "ask": long_bid + 0.2},
    }


# --- take-profit fires on a decayed (winning) side ------------------------


def test_take_profit_fires_when_profit_exceeds_threshold():
    s = _strategy(tp_pct=0.50)
    # short_ask 1.0 -> pnl_pct = (3-1)/3 = 0.667 >= 0.50 -> close
    action = s.check_exit(_call_trade(), _quotes(short_ask=1.0, long_bid=0.5))
    assert action.action == "take_profit"
    assert action.leg_quantities["SPXW  240320C05000000"] == 4  # full short close
    assert action.leg_quantities["SPXW  240320C05015000"] == 4  # long closed too
    assert "tp@" in action.reason


def test_take_profit_holds_below_threshold():
    s = _strategy(tp_pct=0.50)
    # short_ask 2.5 -> pnl_pct = (3-2.5)/3 = 0.167 < 0.50 -> hold
    action = s.check_exit(_call_trade(), _quotes(short_ask=2.5))
    assert action.action == "hold"


def test_take_profit_abandons_bidless_long():
    s = _strategy(tp_pct=0.50)
    # long bid below abandon_long_below (0.05 default) -> short closed, long abandoned
    action = s.check_exit(_call_trade(), _quotes(short_ask=1.0, long_bid=0.0))
    assert action.action == "take_profit"
    assert action.leg_quantities["SPXW  240320C05000000"] == 4
    assert "SPXW  240320C05015000" not in action.leg_quantities
    assert "abandon_long" in action.reason


# --- the load-bearing invariant: NO stop in take-profit-only mode ---------


def test_take_profit_does_not_stop_a_losing_side():
    """A breached side (short_ask 10 >= stop 9) must NOT stop under take_profit;
    it holds and will settle at expiry intrinsic in the engine. Booking it as a
    stop here, or worse as a worthless win at EOD, is the fiction we guard."""
    s = _strategy(tp_pct=0.50)
    action = s.check_exit(_call_trade(), _quotes(short_ask=10.0))
    assert action.action == "hold"  # NOT stop_loss


# --- managed mode still stops (no regression) -----------------------------


def test_managed_mode_still_stops():
    s = _strategy(exit_mode="managed")
    action = s.check_exit(_call_trade(), _quotes(short_ask=10.0))
    assert action.action == "stop_loss"


def test_hold_to_expiry_never_exits():
    s = _strategy(exit_mode="hold_to_expiry")
    assert s.check_exit(_call_trade(), _quotes(short_ask=10.0)).action == "hold"
    assert s.check_exit(_call_trade(), _quotes(short_ask=0.1)).action == "hold"


def test_tp_pct_default_and_config():
    assert _strategy(tp_pct=0.70)._tp_pct == 0.70
    s = MeicStrategy()
    s.configure({"strategy": {"exit_mode": "take_profit"}, "options": {}})
    assert s._tp_pct == 0.50


# --- stop_hold: disaster stop + hold-to-expiry settlement, no scale-out ----


def _stop_strategy() -> MeicStrategy:
    s = MeicStrategy()
    s.configure({"strategy": {"exit_mode": "stop_hold"}, "options": {}})
    return s


def test_stop_hold_stops_when_short_breaches_trigger():
    # _call_trade has short_call_stop_price 9.0; short_ask 10 >= 9 -> stop
    action = _stop_strategy().check_exit(_call_trade(), _quotes(short_ask=10.0))
    assert action.action == "stop_loss"
    assert action.leg_quantities["SPXW  240320C05000000"] == 4   # full short close
    assert action.leg_quantities["SPXW  240320C05015000"] == 4   # long closed too


def test_stop_hold_holds_below_trigger():
    action = _stop_strategy().check_exit(_call_trade(), _quotes(short_ask=2.5))
    assert action.action == "hold"


def test_stop_hold_never_takes_profit():
    # deeply profitable (short decayed to 0.1) must still hold — stop_hold only stops
    action = _stop_strategy().check_exit(_call_trade(), _quotes(short_ask=0.1))
    assert action.action == "hold"
