"""Tests for Alpaca paper execution (hybrid: TT market data, Alpaca fills).

Two groups:

1. Payload translation — pure functions, no network. Pins the OSI-symbol
   reshaping (TT space-pads to 21 chars, Alpaca wants the compact OCC form),
   the multi-leg credit-spread structure, and the sign convention that makes a
   net credit a NEGATIVE limit price on Alpaca.

2. Close-path regressions — the exit flow that the paper run exists to exercise.
   Both were live defects found while wiring the adapter:
     - PositionMonitor._close_leg passes reason=, close_position didn't accept
       it, so EVERY take-profit / stop-loss leg close raised a TypeError that
       the caller's `except Exception` swallowed. Positions rode to EOD instead
       of taking profit, in every mode.
     - close_position(close_all=True) emitted an order carrying only trade_id,
       so the broker layer built a leg with an EMPTY symbol.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zerodte.agent.tool_handlers import ToolHandlers
from zerodte.backtest.osi_symbol import format_osi
from zerodte.services.alpaca_client import (
    AlpacaExecutionClient,
    build_close_order,
    build_credit_spread_order,
    build_fill_ladder,
    to_alpaca_symbol,
)

EXP = date(2026, 8, 18)
PAPER_TEST_URL = "https://paper-api.alpaca.markets"


# ---------------------------------------------------------------------------
# 1. Symbol translation
# ---------------------------------------------------------------------------

def test_osi_padding_stripped_for_alpaca():
    """format_osi emits TT's space-padded 21-char OSI; Alpaca wants it compact."""
    tt_symbol = format_osi(EXP, 7620, is_call=False)
    assert tt_symbol == "SPXW  260818P07620000"
    assert len(tt_symbol) == 21
    assert to_alpaca_symbol(tt_symbol) == "SPXW260818P07620000"


def test_alpaca_symbol_is_idempotent():
    """Re-translating an already-compact symbol must not corrupt it."""
    assert to_alpaca_symbol("SPXW260818P07620000") == "SPXW260818P07620000"


# ---------------------------------------------------------------------------
# 2. Credit-spread (entry) translation
# ---------------------------------------------------------------------------

def test_put_credit_spread_builds_two_leg_mleg():
    order = {
        "order_type": "put_credit_spread",
        "short_strike": 7620,
        "long_strike": 7600,
        "quantity": 1,
        "credit": 2.35,
    }
    payload = build_credit_spread_order(order, expiry=EXP)

    assert payload["order_class"] == "mleg"
    assert payload["type"] == "limit"
    assert payload["time_in_force"] == "day"
    assert payload["qty"] == "1"

    legs = payload["legs"]
    assert len(legs) == 2
    short, long_ = legs
    assert short == {
        "symbol": "SPXW260818P07620000",
        "ratio_qty": "1",
        "side": "sell",
        "position_intent": "sell_to_open",
    }
    assert long_ == {
        "symbol": "SPXW260818P07600000",
        "ratio_qty": "1",
        "side": "buy",
        "position_intent": "buy_to_open",
    }


def test_credit_is_a_negative_limit_price():
    """Alpaca expresses a net credit as a negative limit on a multi-leg order."""
    payload = build_credit_spread_order(
        {"order_type": "put_credit_spread", "short_strike": 7620,
         "long_strike": 7600, "quantity": 1, "credit": 2.35},
        expiry=EXP,
    )
    assert float(payload["limit_price"]) < 0
    assert payload["limit_price"] == "-2.35"


def test_credit_limit_rounds_to_the_nickel_tick():
    """Matches the TT path, which rounds the net-credit limit to the 0.05 tick."""
    payload = build_credit_spread_order(
        {"order_type": "put_credit_spread", "short_strike": 7620,
         "long_strike": 7600, "quantity": 1, "credit": 2.37},
        expiry=EXP,
    )
    assert payload["limit_price"] == "-2.35"


def test_call_credit_spread_uses_call_symbols():
    payload = build_credit_spread_order(
        {"order_type": "call_credit_spread", "short_strike": 7800,
         "long_strike": 7820, "quantity": 2, "credit": 1.80},
        expiry=EXP,
    )
    assert payload["qty"] == "2"
    assert [leg["symbol"] for leg in payload["legs"]] == [
        "SPXW260818C07800000",
        "SPXW260818C07820000",
    ]
    # Ratio stays 1:1 per spread — qty carries the multiple.
    assert all(leg["ratio_qty"] == "1" for leg in payload["legs"])


def test_zero_credit_is_rejected_rather_than_sent_as_a_free_spread():
    """A 0.00 limit would rest forever or fill at any price — refuse it."""
    with pytest.raises(ValueError, match="credit"):
        build_credit_spread_order(
            {"order_type": "put_credit_spread", "short_strike": 7620,
             "long_strike": 7600, "quantity": 1, "credit": 0},
            expiry=EXP,
        )


# ---------------------------------------------------------------------------
# 3. Close (exit) translation
# ---------------------------------------------------------------------------

def test_short_leg_is_bought_to_close():
    payload = build_close_order(
        {"order_type": "close_market", "symbol": "SPXW  260818P07620000", "quantity": 1},
        held_qty=-1,
    )
    assert payload == {
        "symbol": "SPXW260818P07620000",
        "qty": "1",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "position_intent": "buy_to_close",
    }


def test_long_leg_is_SOLD_to_close():
    """The live defect: a credit spread holds one short and one LONG leg.
    Buying to close the long 422s ("position intent mismatch"), which silently
    broke take-profit, breach exits and the EOD close — all close both legs."""
    payload = build_close_order(
        {"order_type": "close_market", "symbol": "SPXW  260818C07750000", "quantity": 1},
        held_qty=1,
    )
    assert payload["side"] == "sell"
    assert payload["position_intent"] == "sell_to_close"
    assert payload["symbol"] == "SPXW260818C07750000"


def test_close_never_exceeds_the_quantity_actually_held():
    payload = build_close_order(
        {"order_type": "close_market", "symbol": "SPXW  260818C07750000", "quantity": 5},
        held_qty=-2,
    )
    assert payload["qty"] == "2"


def test_close_order_without_symbol_is_rejected():
    """Guards the close_all path that used to emit an empty-symbol leg."""
    with pytest.raises(ValueError, match="symbol"):
        build_close_order({"order_type": "close_market", "quantity": 1}, held_qty=-1)


def test_close_with_no_position_is_rejected():
    with pytest.raises(ValueError, match="nothing to close"):
        build_close_order(
            {"order_type": "close_market", "symbol": "SPXW  260818C07750000"},
            held_qty=0,
        )


async def test_closing_a_flat_leg_is_a_noop_not_an_error():
    """Exit paths can fire twice (take-profit then the EOD sweep)."""
    client = AlpacaExecutionClient(
        data_client=object(), key_id="k", secret_key="s", base_url=PAPER_TEST_URL)
    posted = []

    async def fake_request(method, path, **kw):
        if method == "POST":
            posted.append(kw); return {"id": "x"}
        return {}

    async def no_position(symbol): return None

    client._request = fake_request
    client.get_position = no_position
    res = await client.place_order(
        {"order_type": "close_market", "symbol": "SPXW  260818C07750000", "quantity": 1})
    assert res["skipped"] is True
    assert posted == [], "must not submit an order when already flat"


# ---------------------------------------------------------------------------
# 4. Close-path regressions (live defects)
# ---------------------------------------------------------------------------

async def test_close_position_accepts_reason_kwarg():
    """PositionMonitor._close_leg always passes reason=; before the fix this
    raised TypeError, which the caller swallowed — so take-profit never fired."""
    th = ToolHandlers(cache=None, api_client=None, db=None, mode="manual")
    result = await th.close_position(
        symbol="SPXW  260818P07620000",
        order_type="market",
        quantity=1,
        reason="take_profit",
    )
    assert result["symbol"] == "SPXW  260818P07620000"
    assert result["reason"] == "take_profit"


async def test_cancel_confirms_terminal_state_not_just_the_delete():
    """Alpaca cancels asynchronously. The entry path voids the local trade on a
    no-fill, so a cancel that is merely *requested* could still fill and leave
    an unmonitored position — cancel_order must poll until genuinely terminal."""
    from zerodte.services.alpaca_client import AlpacaExecutionClient

    client = AlpacaExecutionClient(
        data_client=object(), key_id="k", secret_key="s", base_url=PAPER_TEST_URL
    )
    statuses = iter(["pending_cancel", "pending_cancel", "canceled"])

    async def fake_request(method, path, **kw):
        return {}

    async def fake_get_order(order_id):
        return {"status": next(statuses)}

    client._request = fake_request
    client.get_order = fake_get_order

    result = await client.cancel_order("abc", confirm_timeout=5.0)
    assert result["status"] == "canceled"
    assert result["confirmed"] is True


async def test_cancel_reports_an_order_that_filled_anyway():
    """The dangerous case: cancel lost the race. Must not be reported clean."""
    from zerodte.services.alpaca_client import AlpacaExecutionClient

    client = AlpacaExecutionClient(
        data_client=object(), key_id="k", secret_key="s", base_url=PAPER_TEST_URL
    )

    async def fake_request(method, path, **kw):
        return {}

    async def fake_get_order(order_id):
        return {"status": "filled"}

    client._request = fake_request
    client.get_order = fake_get_order

    result = await client.cancel_order("abc", confirm_timeout=5.0)
    assert result["status"] == "filled", "a late fill must not be masked as cancelled"


async def test_unsupported_writes_never_fall_through_to_the_data_broker():
    """__getattr__ delegates reads to the wrapped client; a write must not
    silently reach the market-data broker just because it wasn't overridden."""
    from zerodte.services.alpaca_client import AlpacaExecutionClient

    class LoudDataClient:
        async def place_oco_order(self, *a, **kw):
            raise AssertionError("write fell through to the market-data broker")

        async def place_stop_market(self, *a, **kw):
            raise AssertionError("write fell through to the market-data broker")

        def get_quote(self, symbol):
            return {"symbol": symbol, "price": 1.23}

    client = AlpacaExecutionClient(
        data_client=LoudDataClient(), key_id="k", secret_key="s",
        base_url=PAPER_TEST_URL,
    )
    # Reads delegate through.
    assert client.get_quote("SPX")["price"] == 1.23

    # Writes are refused, not delegated.
    with pytest.raises(NotImplementedError):
        await client.place_oco_order({"symbol": "X", "stop_loss_amount": 1})
    with pytest.raises(NotImplementedError):
        await client.place_stop_market("X", 1.0)

    # And an unknown order_type is rejected rather than guessed at.
    with pytest.raises(ValueError, match="order_type"):
        await client.place_order({"order_type": "iron_condor"})


async def test_close_all_resolves_legs_from_the_trade_record():
    """close_all carries only a trade_id; the legs must be resolved from the DB
    so the broker receives real symbols instead of an empty one."""
    short_sym = format_osi(EXP, 7620, is_call=False)
    long_sym = format_osi(EXP, 7600, is_call=False)

    class FakeDb:
        async def get_trade(self, trade_id):
            return {
                "trade_id": trade_id,
                "strategy_data": (
                    '{"short_put_symbol": "%s", "long_put_symbol": "%s",'
                    ' "put_open_qty": 1, "side": "put"}' % (short_sym, long_sym)
                ),
            }

    submitted = []

    class FakeApi:
        async def place_order(self, order):
            submitted.append(order)
            return {"order_id": f"ord-{len(submitted)}"}

    th = ToolHandlers(cache=None, api_client=FakeApi(), db=FakeDb(), mode="live")
    await th.close_position(trade_id="orbdir-put-101500", close_all=True,
                            order_type="market")

    assert len(submitted) == 2, "both legs of the spread must be closed"
    symbols = {o["symbol"] for o in submitted}
    assert symbols == {short_sym, long_sym}
    assert all(o["symbol"].strip() for o in submitted), "no empty-symbol legs"


# ---------------------------------------------------------------------------
# 5. Entry fill ladder
# ---------------------------------------------------------------------------
# The backtest prices entry at mid - options.slippage_per_spread ($0.10), and
# the WF fill-stress matrix puts the validated ceiling at $0.25/spread (the edge
# breaks near $0.50). The ladder therefore walks a bounded PRICE concession, not
# an unbounded wait: a resting credit-spread limit fills preferentially when the
# underlying moves toward the short strike, which is adverse selection the
# backtest never modelled.

def test_ladder_walks_from_modelled_credit_down_to_the_validated_floor():
    ladder = build_fill_ladder(credit=2.35, base_slippage=0.10,
                               rungs=[0.10, 0.18, 0.25])
    # mid credit = 2.35 + 0.10 = 2.45; rungs are total slippage off mid.
    assert ladder == [2.35, 2.25, 2.20]


def test_first_rung_equals_the_backtest_assumption():
    """Rung 1 must be exactly what the WF study priced, or the forward record
    is not comparable to Sharpe 3.29."""
    ladder = build_fill_ladder(credit=3.10, base_slippage=0.10, rungs=[0.10, 0.25])
    assert ladder[0] == 3.10


def test_ladder_is_strictly_decreasing_and_deduped():
    """Rungs that collapse onto the same tick must not produce a resubmit."""
    ladder = build_fill_ladder(credit=1.00, base_slippage=0.10,
                               rungs=[0.10, 0.11, 0.12, 0.25])
    assert ladder == sorted(set(ladder), reverse=True)
    assert len(ladder) == len(set(ladder))


def test_ladder_never_bids_below_zero_credit():
    ladder = build_fill_ladder(credit=0.10, base_slippage=0.10,
                               rungs=[0.10, 0.25, 0.50])
    assert all(x > 0 for x in ladder)


def test_no_rungs_configured_means_a_single_submission():
    assert build_fill_ladder(credit=2.35, base_slippage=0.10, rungs=[]) == [2.35]


def _ladder_client(monkeypatch, fill_on_rung=None, fill_during_cancel_rung=None):
    """Client whose submit/poll/cancel are driven by a scripted fill outcome."""
    client = AlpacaExecutionClient(
        data_client=object(), key_id="k", secret_key="s", base_url=PAPER_TEST_URL,
        entry_fill={"rungs": [0.10, 0.18, 0.25], "rung_seconds": 0.01,
                    "base_slippage": 0.10},
    )
    state = {"submits": [], "cancels": []}

    async def fake_request(method, path, **kw):
        if method == "POST":
            state["submits"].append(kw["json"])
            return {"id": f"ord{len(state['submits'])}", "status": "accepted"}
        return {}

    async def fake_wait(order_id, timeout=None, poll=1.0):
        rung = int(order_id.replace("ord", ""))
        if fill_on_rung == rung:
            return {"status": "filled", "filled": True, "fill_price": -2.20,
                    "fill_credit": 2.20}
        return {"status": "accepted", "filled": False, "fill_price": None,
                "fill_credit": None}

    async def fake_cancel(order_id, confirm_timeout=5.0):
        state["cancels"].append(order_id)
        rung = int(order_id.replace("ord", ""))
        if fill_during_cancel_rung == rung:
            return {"status": "filled", "order_id": order_id, "confirmed": True}
        return {"status": "canceled", "order_id": order_id, "confirmed": True}

    async def fake_get_order(order_id):
        return {"status": "filled", "filled_avg_price": "-2.25"}

    client._request = fake_request
    client.wait_for_fill = fake_wait
    client.cancel_order = fake_cancel
    client.get_order = fake_get_order
    return client, state


async def test_ladder_stops_at_the_first_fill(monkeypatch):
    client, state = _ladder_client(monkeypatch, fill_on_rung=2)
    res = await client.place_order({
        "order_type": "put_credit_spread", "short_symbol": "SPXW  260818P07620000",
        "long_symbol": "SPXW  260818P07600000", "quantity": 1, "credit": 2.35,
    })
    assert res["filled"] is True
    assert res["fill_credit"] == 2.20
    assert len(state["submits"]) == 2, "must not keep laddering after a fill"
    assert res["rung"] == 2


async def test_ladder_exhausts_and_reports_no_fill(monkeypatch):
    client, state = _ladder_client(monkeypatch)
    res = await client.place_order({
        "order_type": "put_credit_spread", "short_symbol": "SPXW  260818P07620000",
        "long_symbol": "SPXW  260818P07600000", "quantity": 1, "credit": 2.35,
    })
    assert res["filled"] is False
    assert len(state["submits"]) == 3
    assert len(state["cancels"]) == 3, "every unfilled rung must be cancelled"
    # Limits walked down to the floor and no further.
    limits = [abs(float(s["limit_price"])) for s in state["submits"]]
    assert limits == [2.35, 2.25, 2.20]


async def test_a_fill_discovered_during_cancel_does_not_resubmit(monkeypatch):
    """The cancel race is the double-fill hazard: if the order filled after the
    cancel was requested, laddering on would open a SECOND spread."""
    client, state = _ladder_client(monkeypatch, fill_during_cancel_rung=1)
    res = await client.place_order({
        "order_type": "put_credit_spread", "short_symbol": "SPXW  260818P07620000",
        "long_symbol": "SPXW  260818P07600000", "quantity": 1, "credit": 2.35,
    })
    assert res["filled"] is True
    assert len(state["submits"]) == 1, "must not submit another rung after a late fill"
    assert res["fill_credit"] == 2.25


# ---------------------------------------------------------------------------
# 6. Counterfactual entry log
# ---------------------------------------------------------------------------

def _read_entry_records(tmp_path):
    import json as _json
    from datetime import date as _date
    p = tmp_path / "execution_comparison" / f"{_date.today().isoformat()}.jsonl"
    return [_json.loads(l) for l in p.read_text().splitlines()
            if _json.loads(l).get("type") == "entry_fill"]


async def test_entry_log_records_the_counterfactual_on_a_fill(tmp_path):
    """Pins the three reference prices the entry sits between, so slippage can
    be measured against the model rather than guessed at."""
    from zerodte.services.execution_log import ExecutionLog

    log = ExecutionLog(base_dir=str(tmp_path))

    class FakeApi:
        async def place_order(self, o):
            return {"order_id": "x1", "filled": True, "fill_credit": 2.25,
                    "rung": 2, "requested_credit": 2.25,
                    "slippage_per_spread": 0.20, "attempts": [{"rung": 1}, {"rung": 2}]}

    class FakeDb:
        async def create_trade(self, t): return t
        async def update_trade(self, tid, patch): return None

    th = ToolHandlers(cache=None, api_client=FakeApi(), db=FakeDb(),
                      mode="alpaca_paper", execution_log=log)
    await th.place_order({
        "order_type": "put_credit_spread", "side": "put",
        "short_put_strike": 7620, "long_put_strike": 7600,
        "credit": 2.35, "quantity": 1, "underlying_at_entry": 7728,
        "short_quote": {"bid": 3.00, "ask": 3.20},
        "long_quote": {"bid": 0.50, "ask": 0.60},
    })

    rec = _read_entry_records(tmp_path)[0]
    assert rec["filled"] is True
    assert rec["fill_credit"] == 2.25
    assert rec["modelled_credit"] == 2.35          # what the backtest assumed
    assert rec["mid_credit"] == 2.55               # 3.10 - 0.55
    assert rec["natural_credit"] == 2.40           # short bid - long ask
    assert rec["slippage_per_spread"] == 0.20
    assert rec["rung"] == 2
    assert rec["time_to_fill_seconds"] >= 0


async def test_entry_log_records_no_fills_too(tmp_path):
    """A no-fill is the answer to the open question, not a missing data point."""
    from zerodte.services.execution_log import ExecutionLog

    log = ExecutionLog(base_dir=str(tmp_path))

    class FakeApi:
        async def place_order(self, o):
            return {"order_id": "x1", "filled": False, "fill_credit": None,
                    "rung": 3, "requested_credit": 2.20, "attempts": []}
        async def cancel_order(self, oid): return {"status": "canceled"}

    class FakeDb:
        def __init__(self): self.updates = []
        async def create_trade(self, t): return t
        async def update_trade(self, tid, patch): self.updates.append(patch)

    db = FakeDb()
    th = ToolHandlers(cache=None, api_client=FakeApi(), db=db,
                      mode="alpaca_paper", execution_log=log)
    await th.place_order({
        "order_type": "put_credit_spread", "side": "put",
        "short_put_strike": 7620, "long_put_strike": 7600,
        "credit": 2.35, "quantity": 1, "underlying_at_entry": 7728,
        "short_quote": {"bid": 3.00, "ask": 3.20},
        "long_quote": {"bid": 0.50, "ask": 0.60},
    })

    rec = _read_entry_records(tmp_path)[0]
    assert rec["filled"] is False
    assert rec["fill_credit"] is None
    assert rec["mid_credit"] == 2.55
    # ...and the local trade is voided rather than left for the monitor.
    assert {"status": "cancelled", "close_reason": "entry_not_filled"} in db.updates


# ---------------------------------------------------------------------------
# 7. False stop-out regression (live defect, 2026-08-18)
# ---------------------------------------------------------------------------

async def test_leg_still_held_is_not_reported_as_stopped_out():
    """_is_position_closed compared leg symbols against local TRADE rows, which
    carry no `symbol` key — so the set was {None}, every leg looked closed, and
    the first poll after entry declared a false stop-out and tried to dump the
    hedge leg."""
    from zerodte.services.position_monitor import PositionMonitor

    class TH:
        async def get_broker_positions(self):
            return [{"symbol": "SPXW260818C07730000", "qty": "-1"},
                    {"symbol": "SPXW260818C07750000", "qty": "1"}]

    m = PositionMonitor.__new__(PositionMonitor)
    m.tool_handlers = TH()
    # Padded OSI (as stored on the trade) vs compact (as the broker reports).
    assert await m._is_position_closed("SPXW  260818C07730000") is False
    assert await m._is_position_closed("SPXW  260818C07999000") is True


async def test_unverifiable_position_is_never_called_stopped_out():
    """No broker to ask (dry_run/manual) -> must not claim a stop-out."""
    from zerodte.services.position_monitor import PositionMonitor

    class TH:
        async def get_broker_positions(self): return None

    m = PositionMonitor.__new__(PositionMonitor)
    m.tool_handlers = TH()
    assert await m._is_position_closed("SPXW  260818C07730000") is False


async def test_manual_mode_reports_positions_as_unknowable():
    th = ToolHandlers(cache=None, api_client=object(), db=None, mode="manual")
    assert await th.get_broker_positions() is None


# ---------------------------------------------------------------------------
# 8. Entry concurrency guard
# ---------------------------------------------------------------------------

async def test_entry_is_refused_when_a_trade_is_already_open():
    """The ORB path never consulted the RiskManager, so max_concurrent_trades
    was unenforced. The unit runs Restart=on-failure, so a mid-session crash
    re-runs the entry watch and would stack a second spread on the live one."""
    from zerodte.agent.orb_engine import OrbEngine

    class Deps:
        config = {"risk": {"max_concurrent_trades": 1}}
        class db:
            @staticmethod
            async def get_open_trades(): return [{"trade_id": "orbdir-call-093113"}]

    e = OrbEngine.__new__(OrbEngine)
    e.deps = Deps()
    assert await e._at_trade_limit() is True


async def test_entry_allowed_when_flat():
    from zerodte.agent.orb_engine import OrbEngine

    class Deps:
        config = {"risk": {"max_concurrent_trades": 1}}
        class db:
            @staticmethod
            async def get_open_trades(): return []

    e = OrbEngine.__new__(OrbEngine)
    e.deps = Deps()
    assert await e._at_trade_limit() is False


async def test_unreadable_trade_state_refuses_entry():
    """Fail safe: skipping a signal costs one trade; stacking doubles risk."""
    from zerodte.agent.orb_engine import OrbEngine

    class Deps:
        config = {"risk": {"max_concurrent_trades": 1}}
        class db:
            @staticmethod
            async def get_open_trades(): raise RuntimeError("db locked")

    e = OrbEngine.__new__(OrbEngine)
    e.deps = Deps()
    assert await e._at_trade_limit() is True


async def test_concurrency_guard_keeps_the_session_alive():
    """run_scheduled() stops the PositionMonitor as soon as engine.run()
    returns, so the guard must idle to EOD rather than return immediately —
    otherwise it abandons the very open position it is protecting."""
    from zerodte.agent.orb_engine import OrbEngine

    idled = []

    class Deps:
        config = {"risk": {"max_concurrent_trades": 1}}
        tool_handlers = None
        class db:
            @staticmethod
            async def get_open_trades(): return [{"trade_id": "t1"}]

    e = OrbEngine.__new__(OrbEngine)
    e.deps = Deps()

    async def fake_idle(): idled.append(True)
    e._idle_until_eod = fake_idle

    res = await e._execute_entry(7711.0, "short")
    assert res.reason == "max_concurrent_trades"
    assert idled == [True], "guard must hold the session open until EOD"


# ---------------------------------------------------------------------------
# 9. Take-profit could never fire (live defect, 2026-08-18)
# ---------------------------------------------------------------------------

async def test_leg_quotes_resolve_from_the_chain_so_take_profit_can_evaluate():
    """_get_quote only read an in-memory cache that holds "SPX" and the ES
    future — never an option leg. So _get_all_quotes returned {}, check_exit
    bailed on `mid is None`, and take-profit could not fire in ANY mode."""
    from zerodte.services.position_monitor import PositionMonitor

    short_sym = format_osi(EXP, 7730, is_call=True)
    long_sym = format_osi(EXP, 7750, is_call=True)

    class TH:
        async def get_option_chain(self, expiration="0DTE"):
            return [
                {"strike": 7730, "option_type": "C", "bid": 0.95, "ask": 1.05},
                {"strike": 7750, "option_type": "C", "bid": 0.20, "ask": 0.30},
                {"strike": 7730, "option_type": "P", "bid": 9.0, "ask": 9.2},
            ]
        async def get_spx_price(self): return {"price": 7690.0}

    m = PositionMonitor.__new__(PositionMonitor)
    m.tool_handlers = TH()
    quotes = await m._get_all_quotes({
        "short_call_symbol": short_sym, "long_call_symbol": long_sym,
    })

    assert short_sym in quotes and long_sym in quotes, "both legs must price"
    assert quotes[short_sym]["bid"] == 0.95
    assert quotes[long_sym]["ask"] == 0.30
    assert quotes["_underlying"]["price"] == 7690.0

    # ...and the strategy can now actually compute a take-profit mark.
    from zerodte.strategies.utils import mid_price
    mark = mid_price(quotes[short_sym]) - mid_price(quotes[long_sym])
    assert round(mark, 2) == 0.75


async def test_take_profit_fires_once_legs_are_priceable():
    """End-to-end: with real leg quotes, a spread well past the TP threshold
    must return take_profit rather than hold."""
    from zerodte.strategies.orb_em_directional import OrbEmDirectionalStrategy

    short_sym = format_osi(EXP, 7730, is_call=True)
    long_sym = format_osi(EXP, 7750, is_call=True)
    s = OrbEmDirectionalStrategy()
    s.configure({
        "options": {"tp_pct_of_credit": 0.75, "sl_mult_of_credit": 0,
                    "hold_put_breach": True, "width_with_trend": 20},
    })
    trade = {
        "trade_id": "t1", "short_call_symbol": short_sym,
        "long_call_symbol": long_sym, "short_call_strike": 7730,
        "long_call_strike": 7750, "short_put_strike": 0, "long_put_strike": 0,
        # db_service._trade_to_dict flattens strategy_data to the top level.
        "call_credit": 4.05, "call_open_qty": 1, "side": "call",
        "call_spread_premium": 4.05, "total_premium": 4.05,
    }
    quotes = {
        short_sym: {"bid": 0.95, "ask": 1.05},
        long_sym: {"bid": 0.20, "ask": 0.30},
        "_underlying": {"price": 7690.0},
    }
    action = s.check_exit(trade, quotes)
    assert action.action == "take_profit", f"expected TP, got {action.action}"
    assert set(action.legs_to_close) == {short_sym, long_sym}


# NOTE: an earlier edit "corrected" this threshold to credit*(1-tp_pct).
# REVERTED — the WF study that produced $13,751 / n=233 runs check_exit with
# credit*tp_pct, so changing it would make live diverge from the validated
# config. The parameter is misnamed, not miswired. See the 2026-08-18 audit.

async def test_take_profit_marks_the_trade_closed():
    """TP closed both legs at the broker but never updated the row, so the
    monitor re-fired TP every poll and the entry concurrency guard would see a
    phantom open trade and block the NEXT session's entry."""
    from zerodte.services.position_monitor import PositionMonitor
    from zerodte.strategies.base import ExitAction

    short_sym, long_sym = "SPXW  260818C07730000", "SPXW  260818C07750000"
    updates = {}

    class Db:
        async def update_trade(self, tid, patch): updates[tid] = patch

    class Strategy:
        def check_exit(self, trade, quotes):
            return ExitAction(action="take_profit",
                              legs_to_close=[short_sym, long_sym],
                              reason="call TP: mark=0.62 <= tp=0.99")

    m = PositionMonitor.__new__(PositionMonitor)
    m.db, m.strategy, m.mode = Db(), Strategy(), "alpaca_paper"
    from zerodte.backtest.sim_clock import WallClock
    m._clock = WallClock()

    async def quotes(trade): return {}
    async def close_leg(trade_id, sym, name, reason="", quantity=1, exit_quote=None):
        return {"symbol": sym, "status": "filled", "filled": True}
    m._get_all_quotes, m._close_leg = quotes, close_leg

    await m._check_trade({"trade_id": "t1"})
    assert updates["t1"]["status"] == "closed"
    assert "TP" in updates["t1"]["close_reason"]


async def test_take_profit_leaves_trade_open_if_a_leg_did_not_close():
    """A leg that errored must NOT be recorded as a completed exit."""
    from zerodte.services.position_monitor import PositionMonitor
    from zerodte.strategies.base import ExitAction

    updates = {}

    class Db:
        async def update_trade(self, tid, patch): updates[tid] = patch

    class Strategy:
        def check_exit(self, trade, quotes):
            return ExitAction(action="take_profit", legs_to_close=["A", "B"],
                              reason="tp")

    m = PositionMonitor.__new__(PositionMonitor)
    m.db, m.strategy, m.mode = Db(), Strategy(), "alpaca_paper"
    from zerodte.backtest.sim_clock import WallClock
    m._clock = WallClock()

    async def quotes(trade): return {}
    async def close_leg(trade_id, sym, name, reason="", quantity=1, exit_quote=None):
        return {"filled": True} if sym == "A" else None   # B raised
    m._get_all_quotes, m._close_leg = quotes, close_leg

    await m._check_trade({"trade_id": "t1"})
    assert updates == {}, "must not mark closed while a leg is unconfirmed"


def test_manual_mode_close_counts_as_flat():
    """manual/dry_run return simulated results; the trade must still close."""
    from zerodte.services.position_monitor import PositionMonitor
    assert PositionMonitor._leg_is_flat({"manual": True, "submitted": False})
    assert PositionMonitor._leg_is_flat({"dry_run": True})
    assert PositionMonitor._leg_is_flat({"skipped": True})
    assert PositionMonitor._leg_is_flat({"filled": True})
    assert not PositionMonitor._leg_is_flat(None)
    assert not PositionMonitor._leg_is_flat({"filled": False})


# ---------------------------------------------------------------------------
# 11. Execution log: per-strategy separation + exit records
# ---------------------------------------------------------------------------

def test_strategies_write_to_separate_files(tmp_path):
    """MEIC and ORB both default to log_dir "logs"; a shared file interleaves
    them and each strategy's report then reads the other's rows."""
    from zerodte.services.execution_log import ExecutionLog
    from datetime import date as _d

    meic = ExecutionLog(base_dir=str(tmp_path), strategy="meic")
    orb = ExecutionLog(base_dir=str(tmp_path), strategy="orb_em_directional")
    meic._append({"type": "assumed", "trade_id": "meic-1"})
    orb._append({"type": "entry_fill", "trade_id": "orbdir-1"})

    day = _d.today().isoformat()
    mrows = [json.loads(l) for l in (tmp_path / "execution_comparison" / "meic" / f"{day}.jsonl").read_text().splitlines()]
    orows = [json.loads(l) for l in (tmp_path / "execution_comparison" / "orb_em_directional" / f"{day}.jsonl").read_text().splitlines()]
    assert [r["trade_id"] for r in mrows] == ["meic-1"]
    assert [r["trade_id"] for r in orows] == ["orbdir-1"]
    # every record is self-identifying even if files are later concatenated
    assert mrows[0]["strategy"] == "meic"
    assert orows[0]["strategy"] == "orb_em_directional"


def test_no_strategy_keeps_the_legacy_flat_path(tmp_path):
    """Historical files and their readers must keep working."""
    from zerodte.services.execution_log import ExecutionLog
    from datetime import date as _d
    ExecutionLog(base_dir=str(tmp_path))._append({"type": "assumed"})
    assert (tmp_path / "execution_comparison" / f"{_d.today().isoformat()}.jsonl").exists()


async def test_exit_fill_is_recorded_against_the_quote_it_crossed(tmp_path):
    """Exit realism is the open question for this strategy; a fill with no
    record of the quote it crossed answers nothing."""
    from zerodte.services.execution_log import ExecutionLog
    from datetime import date as _d

    log = ExecutionLog(base_dir=str(tmp_path), strategy="orb_em_directional")

    class Api:
        async def place_order(self, o):
            return {"order_id": "c1", "filled": True, "fill_price": 0.95,
                    "status": "filled"}

    th = ToolHandlers(cache=None, api_client=Api(), db=None,
                      mode="alpaca_paper", execution_log=log)
    await th.close_position(symbol="SPXW  260818C07730000", trade_id="t1",
                            order_type="market", quantity=1, reason="take_profit",
                            exit_quote={"bid": 0.90, "ask": 1.00})

    day = _d.today().isoformat()
    rows = [json.loads(l) for l in
            (tmp_path / "execution_comparison" / "orb_em_directional" / f"{day}.jsonl").read_text().splitlines()]
    r = [x for x in rows if x["type"] == "exit_fill"][0]
    assert r["reason"] == "take_profit"
    assert r["mid"] == 0.95 and r["spread"] == 0.10
    assert r["fill_price"] == 0.95
    assert r["fill_vs_mid"] == 0.0        # filled at mid
    assert r["filled"] is True


async def test_exit_fill_records_a_skipped_close_too(tmp_path):
    """A close that did not happen is itself a finding."""
    from zerodte.services.execution_log import ExecutionLog
    from datetime import date as _d

    log = ExecutionLog(base_dir=str(tmp_path), strategy="orb_em_directional")

    class Api:
        async def place_order(self, o):
            return {"symbol": o["symbol"], "status": "no_position",
                    "filled": False, "skipped": True}

    th = ToolHandlers(cache=None, api_client=Api(), db=None,
                      mode="alpaca_paper", execution_log=log)
    await th.close_position(symbol="SPXW  260818C07730000", trade_id="t1",
                            order_type="market", quantity=1, reason="eod")

    day = _d.today().isoformat()
    rows = [json.loads(l) for l in
            (tmp_path / "execution_comparison" / "orb_em_directional" / f"{day}.jsonl").read_text().splitlines()]
    r = [x for x in rows if x["type"] == "exit_fill"][0]
    assert r["skipped"] is True and r["filled"] is False and r["fill_price"] is None


# ---------------------------------------------------------------------------
# 12. Live/backtest handler contract
# ---------------------------------------------------------------------------

def test_backtest_handlers_accept_every_live_close_kwarg():
    """PositionMonitor drives BOTH handler classes, so their close_position
    signatures must stay compatible. A kwarg added to one side raises TypeError
    on the other, and _close_leg swallows it — silently disabling exit
    management. That exact drift caused two separate live defects (`reason`,
    then `exit_quote`), each invisible until a position needed closing."""
    import inspect
    from zerodte.backtest.bt_tool_handlers import BacktestToolHandlers

    live = inspect.signature(ToolHandlers.close_position).parameters
    bt = inspect.signature(BacktestToolHandlers.close_position).parameters
    missing = [n for n in live if n != "self" and n not in bt]
    assert not missing, (
        f"BacktestToolHandlers.close_position is missing {missing} — "
        "PositionMonitor will TypeError against the backtest handlers"
    )


def test_backtest_close_position_actually_accepts_the_live_call():
    """Signature check plus a real call, so a **kwargs sink can't fake it."""
    import inspect
    from zerodte.backtest.bt_tool_handlers import BacktestToolHandlers
    sig = inspect.signature(BacktestToolHandlers.close_position)
    for name in ("reason", "exit_quote", "trade_id", "close_all", "quantity"):
        assert name in sig.parameters, f"{name} missing from backtest handler"
