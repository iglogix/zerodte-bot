"""close_all against the real DatabaseService, not a hand-written fake.

The fakes in the other close-path tests once defined a ``get_trade`` method the
real service never had, so EOD ``close_all`` resolved no legs and submitted
nothing while every test stayed green. These tests wire the real service on a
temporary SQLite file so the handler and the persistence layer cannot drift
apart again.
"""

from __future__ import annotations

import json
from datetime import date, datetime

from trading_core.services.db_service import DatabaseService
from zerodte.agent.tool_handlers import ToolHandlers
from zerodte.services.position_monitor import PositionMonitor

SHORT_PUT = "SPXW  260115P07620000"
LONG_PUT = "SPXW  260115P07600000"


def _trade_row(trade_id: str) -> dict:
    return {
        "trade_id": trade_id,
        "entry_time": datetime(2026, 1, 15, 10, 15),
        "expiration_date": date(2026, 1, 15),
        "underlying": "SPX",
        "underlying_price_at_entry": 7650.0,
        "short_call_strike": 0.0,
        "long_call_strike": 0.0,
        "short_put_strike": 7620.0,
        "long_put_strike": 7600.0,
        "call_spread_premium": 0.0,
        "put_spread_premium": 2.35,
        "total_premium": 2.35,
        "call_wing_width": 0,
        "put_wing_width": 20,
        "status": "open",
        "strategy_data": json.dumps({
            "short_put_symbol": SHORT_PUT,
            "long_put_symbol": LONG_PUT,
            "put_open_qty": 1,
            "side": "put",
        }),
    }


def _real_db(tmp_path) -> DatabaseService:
    db = DatabaseService(str(tmp_path / "trades.db"))
    db.initialize()
    return db


class _RecordingApi:
    def __init__(self):
        self.submitted = []

    async def place_order(self, order):
        self.submitted.append(order)
        return {"order_id": f"ord-{len(self.submitted)}"}


async def test_close_all_submits_every_leg_of_a_persisted_trade(tmp_path):
    db = _real_db(tmp_path)
    await db.create_trade(_trade_row("orbdir-put-101500"))
    api = _RecordingApi()

    handlers = ToolHandlers(cache=None, api_client=api, db=db, mode="live")
    await handlers.close_position(trade_id="orbdir-put-101500", close_all=True,
                                  order_type="market")

    assert {o["symbol"] for o in api.submitted} == {SHORT_PUT, LONG_PUT}


async def test_close_all_for_an_unknown_trade_submits_nothing(tmp_path):
    db = _real_db(tmp_path)
    api = _RecordingApi()

    handlers = ToolHandlers(cache=None, api_client=api, db=db, mode="live")
    result = await handlers.close_position(trade_id="no-such-trade", close_all=True,
                                           order_type="market")

    assert api.submitted == []
    assert result.get("error") or result.get("status")


def test_every_db_method_the_exit_path_calls_exists_on_the_real_service():
    """The exit path may only call methods DatabaseService really defines."""
    for owner in (ToolHandlers, PositionMonitor):
        import inspect
        import re

        called = set(re.findall(r"self\.db\.(\w+)\(", inspect.getsource(owner)))
        missing = sorted(name for name in called if not hasattr(DatabaseService, name))
        assert not missing, f"{owner.__name__} calls undefined DatabaseService methods: {missing}"
