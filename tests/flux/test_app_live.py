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
