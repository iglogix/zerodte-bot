"""Tests for the Live P&L Cockpit: data layer, sim harness, snapshot store, routes."""

from __future__ import annotations

import hashlib
import os
from datetime import date

import pytest

from web import cockpit_data as cd
from web.cockpit_sim import CockpitSim, black_scholes

# --------------------------------------------------------------------------- #
#  cockpit_data — pure functions
# --------------------------------------------------------------------------- #


def test_strategy_label_from_prefix():
    assert cd.strategy_label("meic-100500") == "MEIC"
    assert cd.strategy_label("ph-T1-143000") == "Power Hour"
    assert cd.strategy_label("orbdir-put-094500") == "ORB Directional"
    assert cd.strategy_label("weird-123") == "Other"


def test_parse_osi_symbol():
    from src_v2.backtest.osi_symbol import format_osi

    sym = format_osi(date(2026, 6, 17), 5950.0, is_call=False)
    parsed = cd.parse_osi_symbol(sym)
    assert parsed["right"] == "put"
    assert parsed["strike"] == 5950.0
    assert parsed["expiration"] == date(2026, 6, 17)
    assert cd.parse_osi_symbol("garbage") is None


def test_build_position_meic_ic():
    sim = CockpitSim(spot=6000.0)
    trade = next(t for t in sim.trades() if t["trade_id"] == "meic-100500")
    pos = cd.build_position(trade)
    assert pos["strategy"] == "MEIC"
    # two spreads (put + call), four legs
    assert len(pos["spreads"]) == 2
    assert len(pos["legs"]) == 4
    sides = sorted(leg["side"] for leg in pos["legs"])
    assert sides == ["long", "long", "short", "short"]
    # max profit = (put_credit + call_credit) * qty * 100
    assert pos["max_profit"] == pytest.approx((1.10 + 1.05) * 100)


def test_build_position_orb_single_side():
    sim = CockpitSim(spot=6000.0)
    trade = next(t for t in sim.trades() if t["trade_id"].startswith("orbdir"))
    pos = cd.build_position(trade)
    assert pos["strategy"] == "ORB Directional"
    assert len(pos["spreads"]) == 1
    assert pos["spreads"][0]["right"] == "put"
    assert len(pos["legs"]) == 2


def test_open_pnl_matches_credit_when_worthless():
    # If both legs are worth ~0, you keep the full credit.
    pos = {
        "spreads": [
            {
                "right": "put",
                "credit": 1.10,
                "qty": 1,
                "short": {"symbol": "S", "multiplier": 100},
                "long": {"symbol": "L", "multiplier": 100},
            }
        ],
    }
    marks = {"S": 0.0, "L": 0.0}
    assert cd.open_pnl_for_position(pos, marks) == pytest.approx(110.0)
    # Cost to close 0.40 (short 0.50 / long 0.10) => pnl = (1.10-0.40)*100 = 70
    assert cd.open_pnl_for_position(pos, {"S": 0.50, "L": 0.10}) == pytest.approx(70.0)


def test_open_pnl_none_when_marks_missing():
    pos = {
        "spreads": [
            {
                "right": "put",
                "credit": 1.0,
                "qty": 1,
                "short": {"symbol": "S", "multiplier": 100},
                "long": None,
            }
        ]
    }
    assert cd.open_pnl_for_position(pos, {}) is None


def test_snapshot_request_body_side_mapping():
    sim = CockpitSim(spot=6000.0)
    positions = cd.group_positions(sim.trades())
    body = cd.snapshot_request_body(positions)
    assert body["underlyings"] == ["SPX"]
    assert len(body["legs"]) == sum(len(p["legs"]) for p in positions)
    for leg in body["legs"]:
        assert leg["side"] in ("buy", "sell")
        assert leg["asset_type"] == "option"
        assert leg["multiplier"] == 100


# --------------------------------------------------------------------------- #
#  black-scholes sanity
# --------------------------------------------------------------------------- #


def test_black_scholes_sanity():
    atm = black_scholes(6000, 6000, 0.02, 0.16, True)
    assert atm["price"] > 0
    assert 0 < atm["delta"] < 1
    itm = black_scholes(6100, 6000, 1e-6, 0.16, True)
    assert itm["price"] == pytest.approx(100, abs=0.5)
    otm = black_scholes(6100, 6000, 1e-6, 0.16, False)
    assert otm["price"] < 0.5


# --------------------------------------------------------------------------- #
#  sim snapshot
# --------------------------------------------------------------------------- #


def test_sim_snapshot_prices_all_requested_legs():
    sim = CockpitSim(spot=6000.0)
    sim.seek(360)  # late session: every strategy has entered
    positions = cd.group_positions(sim.trades())
    body = cd.snapshot_request_body(positions)
    snap = sim.snapshot(body)
    assert len(snap["legs"]) == len(body["legs"])
    for leg in snap["legs"]:
        assert leg["status"] == "ok"
        assert leg["mark"] is not None
        assert leg["bid"] <= leg["mark"] <= leg["ask"]
    assert snap["underlyings"]["SPX"]["last"] is not None


def test_sim_quotes_exist_all_day_but_positions_gate_by_entry():
    # Quotes exist for every option all session (a real option always has a
    # quote), but no position is "active" at the open (09:30) yet.
    sim = CockpitSim(spot=6000.0)
    body = cd.snapshot_request_body(cd.group_positions(sim.trades()))
    snap = sim.snapshot(body)  # renders minute 0
    assert all(leg["status"] == "ok" for leg in snap["legs"])
    assert snap["sim"]["active_trade_ids"] == []


def test_sim_active_trade_ids_grow_through_session():
    sim = CockpitSim(spot=6000.0)
    body = cd.snapshot_request_body(cd.group_positions(sim.trades()))
    sim.seek(20)  # after 09:45 ORB entry only
    early = set(sim.snapshot(body)["sim"]["active_trade_ids"])
    sim.seek(360)  # after every entry
    late = set(sim.snapshot(body)["sim"]["active_trade_ids"])
    assert early == {"orbdir-put-094500"}
    assert len(late) == 4 and early.issubset(late)


def test_sim_clock_advances_open_to_close():
    sim = CockpitSim(spot=6000.0, minutes_per_poll=60)
    body = cd.snapshot_request_body(cd.group_positions(sim.trades()))
    minutes = [sim.snapshot(body)["sim"]["minute"] for _ in range(10)]
    assert minutes[0] == 0
    assert minutes[-1] == 390  # holds at the close
    assert sim.snapshot(body)["sim"]["at_close"] is True


def test_sim_spot_moves_between_snapshots():
    sim = CockpitSim(spot=6000.0)
    body = cd.snapshot_request_body(cd.group_positions(sim.trades()))
    a = sim.snapshot(body)["underlyings"]["SPX"]["last"]
    b = sim.snapshot(body)["underlyings"]["SPX"]["last"]
    assert a != b  # spot path advances with the clock


# --------------------------------------------------------------------------- #
#  snapshot store
# --------------------------------------------------------------------------- #


def test_pnl_store_round_trip(tmp_path):
    from src_v2.services.pnl_snapshot import PnlSnapshotStore

    store = PnlSnapshotStore(str(tmp_path / "pnl.db"))
    n = store.record_many(
        [
            {
                "trade_id": "meic-1",
                "strategy": "MEIC",
                "open_pnl": 100.0,
                "underlying_price": 6000.0,
            },
            {
                "trade_id": "orb-1",
                "strategy": "ORB Directional",
                "open_pnl": -30.0,
                "underlying_price": 6000.0,
            },
        ]
    )
    assert n == 2
    rows = store.series_for_date(date.today())
    assert len(rows) == 2
    assert {r["trade_id"] for r in rows} == {"meic-1", "orb-1"}
    assert rows[0]["open_pnl"] in (100.0, -30.0)


# --------------------------------------------------------------------------- #
#  routes (sim mode)
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_WEB_TOKEN", "test-token")
    monkeypatch.setenv("COCKPIT_SIM", "1")
    monkeypatch.setenv("COCKPIT_PNL_DB", str(tmp_path / "cockpit_pnl.db"))
    # reset cockpit singletons so the tmp store path is picked up
    import web.cockpit as ck

    ck._state = ck._State()

    from fastapi.testclient import TestClient
    from web.app import create_app

    app = create_app()
    c = TestClient(app)
    session = hashlib.sha256("test-token".encode()).hexdigest()[:32]
    c.cookies.set("session", session)
    return c


def test_route_positions_sim(client):
    r = client.get("/api/positions")
    assert r.status_code == 200
    data = r.json()
    assert data["sim"] is True
    assert len(data["positions"]) == 4
    strategies = {p["strategy"] for p in data["positions"]}
    assert "MEIC" in strategies and "Power Hour" in strategies


def test_route_quotes_and_history_sim(client):
    positions = client.get("/api/positions").json()["positions"]
    # Jump the simulated session past every entry so all positions are active.
    import web.cockpit as ck

    ck._state.sim.seek(360)
    r = client.post("/api/quotes", json={"positions": positions})
    assert r.status_code == 200
    data = r.json()
    assert data["marks"]
    assert data["underlying"]["last"] is not None
    assert data["sim"]["minute"] >= 360
    assert len(data["pnl"]) == len(positions)
    assert all(p["open_pnl"] is not None for p in data["pnl"])

    # the poll should have persisted a snapshot; history returns it
    hist = client.get("/api/pnl_history").json()
    assert len(hist["samples"]) == len(positions)


def test_route_sim_reset(client):
    import web.cockpit as ck

    ck._state.sim.seek(120)
    r = client.post("/api/sim/reset")
    assert r.status_code == 200
    assert r.json()["minute"] == 0


def test_route_cockpit_page_served(client):
    r = client.get("/cockpit")
    assert r.status_code == 200
    assert "P&amp;L Cockpit" in r.text or "P&L Cockpit" in r.text


def test_route_requires_auth():
    from fastapi.testclient import TestClient
    from web.app import create_app

    os.environ["BOT_WEB_TOKEN"] = "test-token"
    c = TestClient(create_app(), follow_redirects=False)
    r = c.get("/api/positions")
    assert r.status_code in (303, 307)  # redirect to /login
