"""Tests for the live PowerHourStrategy (zerodte/strategies/power_hour.py)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from zerodte.strategies.power_hour import PowerHourStrategy

ET = ZoneInfo("America/New_York")


def make_config():
    return {
        "strategy": {
            "tranches": [
                {"name": "T1", "entry": "14:55", "target_credit": 4.00,
                 "stop": 4.00, "ratchet_add": 3.00, "qty": 1},
                {"name": "T3", "entry": "15:25", "target_credit": 2.00,
                 "stop": 3.00, "ratchet_add": None, "qty": 1},
            ],
            "ratchet_mode": "trigger_plus",
            "ratchet_trigger_profit_pct": 0.40,
            "profit_target_pct": 0.90,
            "wing_target_price": 0.05,
        },
        "options": {"contract_multiplier": 100},
    }


def make_chain(spot=5000.0):
    """Chain where the $4-mid shorts are 4990P / 5010C and wings go zero-bid
    (0 x 0.05) from 40 pts OTM."""
    chain = []
    for k in [spot + d for d in range(-100, 105, 5)]:
        for ot in ("call", "put"):
            otm = (k - spot) if ot == "call" else (spot - k)
            if (k, ot) in ((4990.0, "put"), (5010.0, "call")):
                mid = 4.0
            elif otm <= 0:
                mid = 8.3 - otm * 0.1
            else:
                mid = max(0.025, 7.3 - otm * 0.13)
            if otm >= 40:
                bid, ask = 0.0, 0.05
            else:
                bid, ask = round(mid - 0.05, 3), round(mid + 0.05, 3)
            chain.append({"strike": k, "option_type": ot, "bid": bid, "ask": ask})
    return chain


def make_strategy(at="14:55"):
    s = PowerHourStrategy()
    s.configure(make_config())
    h, m = int(at[:2]), int(at[3:5])
    s._now_fn = lambda: datetime(2026, 6, 9, h, m, tzinfo=ET)
    return s


async def test_search_fixed_credit_and_zero_bid_wings():
    s = make_strategy("14:55")
    out = await s.search_candidates(make_chain(), spot=5000.0)
    setup = out["top_candidate"]
    assert setup is not None
    assert setup.strikes["short_put"] == 4990.0
    assert setup.strikes["short_call"] == 5010.0
    # wings: nearest zero-bid (ask 0.05) strike beyond the short (otm >= 40)
    assert setup.strikes["long_call"] == 5040.0
    assert setup.strikes["long_put"] == 4960.0
    assert setup.extra["tranche"] == "T1"
    assert setup.extra["stop_dollars"] == 4.0
    assert setup.extra["strangle_credit"] == pytest.approx(8.0, abs=0.01)
    val = await s.validate(setup, make_chain())
    assert val["passed"], val["failures"]


async def test_search_uses_tranche_for_current_slot():
    s = make_strategy("15:25")
    out = await s.search_candidates(make_chain(), spot=5000.0)
    setup = out["top_candidate"]
    assert setup.extra["tranche"] == "T3"
    assert setup.extra["ratchet_add"] is None
    assert setup.extra["stop_dollars"] == 3.0


async def test_no_tranche_outside_schedule():
    s = make_strategy("13:00")
    out = await s.search_candidates(make_chain(), spot=5000.0)
    assert out["top_candidate"] is None


def _trade(entry_strangle=8.0, stop_dollars=4.0, ratchet_add=3.0):
    return {
        "trade_id": "t1",
        "short_call_symbol": "SC", "long_call_symbol": "LC",
        "short_put_symbol": "SP", "long_put_symbol": "LP",
        "strangle_credit": entry_strangle,
        "stop_dollars": stop_dollars,
        "ratchet_add": ratchet_add,
        "call_open_qty": 1, "put_open_qty": 1,
    }


def _quotes(call_mid, put_mid):
    return {
        "SC": {"bid": call_mid - 0.05, "ask": call_mid + 0.05},
        "SP": {"bid": put_mid - 0.05, "ask": put_mid + 0.05},
    }


def test_check_exit_stop_at_entry_plus_4():
    s = make_strategy()
    act = s.check_exit(_trade(), _quotes(6.0, 6.1))   # 12.1 >= 12.0
    assert act.action == "stop_loss"
    assert set(act.legs_to_close) == {"SC", "SP"}


def test_check_exit_holds_below_stop():
    s = make_strategy()
    act = s.check_exit(_trade(), _quotes(5.0, 6.0))   # 11.0 < 12.0
    assert act.action == "hold"


def test_check_exit_ratchet_then_stop():
    """Worked example: $8 entry, decay to $4.80 ratchets stop to $7.80."""
    s = make_strategy()
    t = _trade()
    assert s.check_exit(t, _quotes(2.4, 2.4)).action == "hold"   # 4.8 -> ratchet
    assert s._ratchet_state["t1"] == pytest.approx(7.8, abs=0.01)
    act = s.check_exit(t, _quotes(3.95, 3.95))                   # 7.9 >= 7.8
    assert act.action == "stop_loss"


def test_check_exit_no_ratchet_when_disabled():
    s = make_strategy()
    t = _trade(ratchet_add=None, stop_dollars=3.0)
    assert s.check_exit(t, _quotes(2.0, 2.0)).action == "hold"
    assert "t1" not in s._ratchet_state
    # still uses the original entry+3 stop
    act = s.check_exit(t, _quotes(5.6, 5.5))                     # 11.1 >= 11.0
    assert act.action == "stop_loss"


def test_check_exit_profit_target():
    s = make_strategy()
    act = s.check_exit(_trade(), _quotes(0.35, 0.40))            # 0.75 <= 0.8
    assert act.action == "take_profit"
