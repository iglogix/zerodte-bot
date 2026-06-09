from datetime import date, datetime, timezone

import pytest

from src_v2.flux.models import OptionRow, ChainSnapshot
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.app import create_app


class FakeLive(FluxSource):
    """Front (odd-day expiry) IV 0.22, back (even-day) 0.20 → ratio 1.1."""
    def chain(self, expiry, on_date=None):
        iv = 0.22 if expiry.day % 2 == 1 else 0.20
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, 15, tzinfo=timezone.utc),
                             (OptionRow(7000, True, 40, 41, iv, 0.50),
                              OptionRow(7000, False, 40, 41, iv, -0.50)))


def _client():
    return create_app(live_source=FakeLive()).test_client()


def test_live_ratio_endpoint():
    r = _client().get("/api/live_ratio?front=2026-06-17&back=2026-06-18&ref_delta=0.5")
    assert r.status_code == 200
    b = r.get_json()
    assert b["front_iv"] == pytest.approx(0.22) and b["back_iv"] == pytest.approx(0.20)
    assert b["ratio"] == pytest.approx(1.1)


def test_expirations_iv_endpoint():
    r = _client().get("/api/expirations_iv?ref=2026-06-08&min_dte=1&max_dte=3&ref_delta=0.5")
    assert r.status_code == 200
    exps = r.get_json()["expirations"]
    assert len(exps) == 3
    assert "atm_iv" in exps[0] and "dte" in exps[0] and exps[0]["atm_iv"] is not None


def test_live_endpoints_off_when_no_live_source():
    c = create_app().test_client()   # no live_source
    assert "error" in c.get("/api/live_ratio?front=2026-06-17&back=2026-06-18").get_json()
    assert c.get("/api/expirations_iv").get_json()["expirations"] == []


class FakeIntraday:
    def intraday(self, session, front_exp, back_exp, ref_delta, atm_hint):
        from datetime import datetime, timezone
        from src_v2.flux.models import RatioPoint
        return [RatioPoint(datetime(2026, 6, 5, 13, 40, tzinfo=timezone.utc),
                           front_exp, back_exp, ref_delta, 0.22, 0.20, 1.1),
                RatioPoint(datetime(2026, 6, 5, 13, 50, tzinfo=timezone.utc),
                           front_exp, back_exp, ref_delta, 0.23, 0.20, 1.15)]


def test_intraday_endpoint_returns_series():
    c = create_app(intraday_source=FakeIntraday()).test_client()
    r = c.get("/api/intraday?front=2026-06-17&back=2026-06-18&session=2026-06-05&ref_delta=0.5&atm=7000")
    assert r.status_code == 200
    b = r.get_json()
    assert len(b["points"]) == 2 and b["points"][0]["ratio"] == pytest.approx(1.1)
    assert b["meta"]["session"] == "2026-06-05"


def test_intraday_off_when_not_configured():
    c = create_app().test_client()
    assert c.get("/api/intraday?front=2026-06-17&back=2026-06-18").get_json()["points"] == []


class FakeIntradayChains:
    def latest_chain(self, session, expiry, atm_hint=None):
        rich = 2.0 if expiry.day % 2 == 0 else 0.0   # back (even day) richer than front
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc),
                             (OptionRow(7040, True, 29 + rich, 31 + rich, 0.18, 0.35),
                              OptionRow(6960, False, 29 + rich, 31 + rich, 0.18, -0.35)))


def test_dc_order_endpoint_builds_ticket():
    c = create_app(intraday_source=FakeIntradayChains()).test_client()
    r = c.get("/api/dc_order?front=2026-06-17&back=2026-06-18&target_delta=0.35&qty=2")
    assert r.status_code == 200
    t = r.get_json()["ticket"]
    assert t["price_effect"] == "Debit" and t["price"] == pytest.approx(4.0)   # (32+32)-(30+30)
    assert len(t["legs"]) == 4 and all(l["qty"] == 2 for l in t["legs"])
    assert r.get_json()["mode"] == "manual"


def test_dc_submit_refused_in_manual():
    c = create_app(intraday_source=FakeIntradayChains()).test_client()   # no submitter → manual
    b = c.post("/api/dc_submit?front=2026-06-17&back=2026-06-18&execution=combo").get_json()
    assert b["ok"] is False and "manual" in b["detail"].lower()


def test_dc_submit_combo_in_sandbox_passes_qty():
    from src_v2.flux.submit import MockSubmitter
    s = MockSubmitter("sandbox")
    c = create_app(intraday_source=FakeIntradayChains(), submitter=s).test_client()
    b = c.post("/api/dc_submit?front=2026-06-17&back=2026-06-18&execution=combo&qty=3").get_json()
    assert b["ok"] and b["execution"] == "combo" and len(b["order_ids"]) == 1
    assert s.calls[-1][0].legs[0].quantity == 3        # qty propagated to the ticket


def test_dc_submit_legin_places_four():
    from src_v2.flux.submit import MockSubmitter
    c = create_app(intraday_source=FakeIntradayChains(), submitter=MockSubmitter("live")).test_client()
    b = c.post("/api/dc_submit?front=2026-06-17&back=2026-06-18&execution=legin").get_json()
    assert b["ok"] and len(b["order_ids"]) == 4 and b["mode"] == "live"


class FakeTransformChains:
    """Backs (7335/7505) and wings (7310/7530, wing=25) priced on every expiry."""
    def latest_chain(self, session, expiry, atm_hint=None):
        rows = (OptionRow(7335, False, 39, 41, 0.18, -0.35), OptionRow(7505, True, 39, 41, 0.18, 0.35),
                OptionRow(7310, False, 7, 9, 0.18, -0.20), OptionRow(7530, True, 7, 9, 0.18, 0.20))
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), rows)


class FakeDcSubmitter:
    """MockSubmitter-like, plus open_double_calendars()."""
    def __init__(self, mode="live"):
        self.mode = mode
        self.calls = []

    def open_double_calendars(self):
        from src_v2.flux.positions import DoubleCalendar
        return [DoubleCalendar("SPXW", date(2026, 6, 22), date(2026, 6, 23), 7335.0, 7505.0, 1)]

    def working_orders(self):
        return [{"id": "1057169", "status": "Received", "order_type": "Limit", "price": 0.57,
                 "price_effect": "Debit",
                 "legs": [{"symbol": "SPY   260618P00734000", "action": "Sell to Open", "qty": 1}]}]

    def submit(self, ticket, execution="combo"):
        from src_v2.flux.submit import SubmitResult
        self.calls.append((ticket, execution))
        n = len(ticket.legs) if execution == "legin" else 1
        return SubmitResult(True, [f"MOCK-{i+1}" for i in range(n)], f"mock {execution}")


def test_positions_lists_detected_dcs():
    c = create_app(submitter=FakeDcSubmitter()).test_client()
    dcs = c.get("/api/positions").get_json()["dcs"]
    assert len(dcs) == 1 and dcs[0]["k_put"] == 7335.0 and dcs[0]["k_call"] == 7505.0
    assert dcs[0]["front_expiry"] == "2026-06-22"


def test_positions_off_without_broker():
    assert "error" in create_app().test_client().get("/api/positions").get_json()


def test_orders_lists_working_orders():
    c = create_app(submitter=FakeDcSubmitter()).test_client()
    o = c.get("/api/orders").get_json()["orders"]
    assert len(o) == 1 and o[0]["status"] == "Received" and o[0]["id"] == "1057169"


def test_orders_off_without_broker():
    assert "error" in create_app().test_client().get("/api/orders").get_json()


def test_transform_order_builds_credit_ticket():
    c = create_app(intraday_source=FakeTransformChains()).test_client()
    q = "front=2026-06-22&back=2026-06-23&k_put=7335&k_call=7505&wing=25"
    t = c.get(f"/api/transform_order?{q}").get_json()["ticket"]   # default fill = cross
    assert t["kind"] == "transform" and t["price_effect"] == "Credit"
    assert t["price"] == pytest.approx(60.0)               # cross: backs@bid (39+39) − wings@ask (9+9)
    assert len(t["legs"]) == 4
    # mid fill is the optimistic target
    t_mid = c.get(f"/api/transform_order?{q}&fill=mid").get_json()["ticket"]
    assert t_mid["price"] == pytest.approx(64.0)


def test_transform_submit_routes_through_submitter():
    s = FakeDcSubmitter("sandbox")
    c = create_app(intraday_source=FakeTransformChains(), submitter=s).test_client()
    q = "front=2026-06-22&back=2026-06-23&k_put=7335&k_call=7505&wing=25&execution=combo"
    b = c.post(f"/api/transform_submit?{q}").get_json()
    assert b["ok"] and b["execution"] == "combo"
    assert s.calls[-1][0].kind == "transform"
