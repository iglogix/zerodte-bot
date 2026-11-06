"""Manual-mode lifecycle for power_hour_ic: the assumed fill must round-trip
through the DB into a trade dict that PowerHourStrategy.check_exit can manage
(the 2026-06-11 review found this path entirely missing)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from zerodte.agent.tool_handlers import ToolHandlers
from trading_core.services.db_service import DatabaseService
from zerodte.strategies.power_hour import PowerHourStrategy

ET = ZoneInfo("America/New_York")


def make_payload():
    return {
        "order_type": "power_hour_ic",
        "quantity": 1,
        "total_credit": 7.5,
        "stop_amount": 4.0,
        "tranche": "T1",
        "underlying_at_entry": 7368.6,
        "strangle_credit": 7.7,
        "ratchet_add": 3.0,
        "wings_first": True,
        "put_spread": {"short_strike": 7345.0, "long_strike": 7175.0,
                       "credit": 3.45, "short_mid": 3.55, "quantity": 1},
        "call_spread": {"short_strike": 7385.0, "long_strike": 7510.0,
                        "credit": 4.05, "short_mid": 4.15, "quantity": 1},
    }


@pytest.fixture
def db(tmp_path):
    svc = DatabaseService(str(tmp_path / "ph_test.db"))
    svc.initialize()
    return svc


async def test_manual_order_persists_open_trade(db):
    th = ToolHandlers(db=db, mode="manual")
    res = await th.place_order(make_payload())
    assert res["submitted"] is False and res["manual"] is True
    assert res["trade_id"].startswith("ph-T1-")

    trades = await db.get_open_trades()
    assert len(trades) == 1
    t = trades[0]
    # base columns
    assert t["short_put_strike"] == 7345.0
    assert t["short_call_strike"] == 7385.0
    assert t["stop_loss_amount"] == 4.0
    # strategy_data overlay — everything check_exit needs
    assert t["strangle_credit"] == 7.7
    assert t["stop_dollars"] == 4.0
    assert t["ratchet_add"] == 3.0
    assert t["put_open_qty"] == 1 and t["call_open_qty"] == 1
    for key in ("short_put_symbol", "long_put_symbol",
                "short_call_symbol", "long_call_symbol"):
        assert t[key], key
    assert "P07345000" in t["short_put_symbol"]
    assert "C07385000" in t["short_call_symbol"]


async def test_check_exit_runs_on_db_trade_dict(db):
    th = ToolHandlers(db=db, mode="manual")
    await th.place_order(make_payload())
    trade = (await db.get_open_trades())[0]

    s = PowerHourStrategy()
    s.configure({"strategy": {"tranches": [], "ratchet_trigger_profit_pct": 0.40,
                              "profit_target_pct": 0.90}, "options": {}})
    sc, sp = trade["short_call_symbol"], trade["short_put_symbol"]

    # below the 11.70 stop -> hold
    quotes = {sc: {"bid": 5.0, "ask": 5.2}, sp: {"bid": 6.0, "ask": 6.2}}
    assert s.check_exit(trade, quotes).action == "hold"

    # breach 7.7 + 4.0 -> stop, both shorts closed
    quotes = {sc: {"bid": 6.0, "ask": 6.2}, sp: {"bid": 5.8, "ask": 6.0}}
    act = s.check_exit(trade, quotes)
    assert act.action == "stop_loss"
    assert set(act.legs_to_close) == {sc, sp}

    # idempotent: the same breach must not re-fire (wings stay on the row)
    assert s.check_exit(trade, quotes).action == "hold"


async def test_dry_run_also_persists(db):
    th = ToolHandlers(db=db, mode="dry_run")
    res = await th.place_order(make_payload())
    assert res["dry_run"] is True
    assert len(await db.get_open_trades()) == 1
