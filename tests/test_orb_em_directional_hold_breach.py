"""Test the hold_put_breach + hold_call_breach config flags.

Pin the symmetric semantics: by default both sides exit on breach; when
hold_put_breach=True, put spreads ride breaches; when hold_call_breach=True,
call spreads ride breaches. The two flags are independent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_v2.strategies.orb_em_directional import OrbEmDirectionalStrategy


def _build_config(hold_put: bool = False, hold_call: bool = False) -> dict:
    return {
        "options": {
            "hold_put_breach": hold_put,
            "hold_call_breach": hold_call,
            "tp_pct_of_credit": 0.75,
        },
        "strategy": {},
    }


def _strategy(hold_put: bool = False, hold_call: bool = False) -> OrbEmDirectionalStrategy:
    s = OrbEmDirectionalStrategy()
    s.configure(_build_config(hold_put, hold_call))
    return s


def _put_trade(short_strike: float = 5000, long_strike: float = 4985) -> dict:
    return {
        "short_put_symbol": "SPXW  240320P05000000",
        "long_put_symbol": "SPXW  240320P04985000",
        "short_put_strike": short_strike,
        "long_put_strike": long_strike,
        "put_credit": 3.0,
    }


def _call_trade(short_strike: float = 5000, long_strike: float = 5015) -> dict:
    return {
        "short_call_symbol": "SPXW  240320C05000000",
        "long_call_symbol": "SPXW  240320C05015000",
        "short_call_strike": short_strike,
        "long_call_strike": long_strike,
        "call_credit": 3.0,
    }


def _quotes_with_breach_below(short_put: float, breach_price: float) -> dict:
    """Build quote dict where price has crashed below short_put — TP NOT hit."""
    return {
        "short_put_symbol_unused": None,
        "SPXW  240320P05000000": {"bid": 10.0, "ask": 10.5},
        "SPXW  240320P04985000": {"bid": 5.0, "ask": 5.5},
        "_underlying": {"price": breach_price},
    }


def _quotes_with_breach_above(short_call: float, breach_price: float) -> dict:
    return {
        "SPXW  240320C05000000": {"bid": 10.0, "ask": 10.5},
        "SPXW  240320C05015000": {"bid": 5.0, "ask": 5.5},
        "_underlying": {"price": breach_price},
    }


# ----------------------------------------------------------------------------
# Put breach
# ----------------------------------------------------------------------------

def test_put_breach_exits_when_hold_put_breach_false():
    s = _strategy(hold_put=False, hold_call=False)
    action = s.check_exit(_put_trade(), _quotes_with_breach_below(5000, 4990))
    assert action.action == "stop_loss"
    assert "put breach" in action.reason


def test_put_breach_holds_when_hold_put_breach_true():
    s = _strategy(hold_put=True, hold_call=False)
    action = s.check_exit(_put_trade(), _quotes_with_breach_below(5000, 4990))
    assert action.action == "hold"


# ----------------------------------------------------------------------------
# Call breach (the new flag)
# ----------------------------------------------------------------------------

def test_call_breach_exits_when_hold_call_breach_false():
    s = _strategy(hold_put=False, hold_call=False)
    action = s.check_exit(_call_trade(), _quotes_with_breach_above(5000, 5010))
    assert action.action == "stop_loss"
    assert "call breach" in action.reason


def test_call_breach_holds_when_hold_call_breach_true():
    s = _strategy(hold_put=False, hold_call=True)
    action = s.check_exit(_call_trade(), _quotes_with_breach_above(5000, 5010))
    assert action.action == "hold"


# ----------------------------------------------------------------------------
# Independence
# ----------------------------------------------------------------------------

def test_hold_call_does_not_affect_put_behavior():
    s = _strategy(hold_put=False, hold_call=True)
    action = s.check_exit(_put_trade(), _quotes_with_breach_below(5000, 4990))
    # Put still exits because hold_put_breach is False
    assert action.action == "stop_loss"
    assert "put breach" in action.reason


def test_hold_put_does_not_affect_call_behavior():
    s = _strategy(hold_put=True, hold_call=False)
    action = s.check_exit(_call_trade(), _quotes_with_breach_above(5000, 5010))
    # Call still exits because hold_call_breach is False
    assert action.action == "stop_loss"
    assert "call breach" in action.reason


# ----------------------------------------------------------------------------
# Default value
# ----------------------------------------------------------------------------

def test_hold_call_breach_defaults_to_false():
    """Backward compatibility: omitting hold_call_breach behaves like the old code."""
    s = OrbEmDirectionalStrategy()
    s.configure({"options": {}, "strategy": {}})
    assert s._hold_call_breach is False
    assert s._hold_put_breach is False
