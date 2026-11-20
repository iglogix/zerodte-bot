# tests/flux/test_app.py
from datetime import datetime, timezone

from zerodte.flux.app import create_app
from zerodte.flux.models import ChainSnapshot, OptionRow
from zerodte.flux.sources.base import FluxSource


class FakeSource(FluxSource):
    def chain(self, expiry, on_date=None):
        is_front = expiry.day % 2 == 1
        iv = 0.22 if is_front else 0.20
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc),
                             (OptionRow(7000, True, 1, 1, iv, 0.5),
                              OptionRow(7000, False, 1, 1, iv, -0.5)))


def _client():
    return create_app(FakeSource()).test_client()


def test_ratio_endpoint_returns_series():
    r = _client().get("/api/ratio?front_dte=9&gap=1&ref_delta=0.5&view=5day&end_date=2026-06-10")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["points"]) >= 1
    assert "front_iv" in body["points"][0] and "ratio" in body["points"][0]


def test_scan_endpoint_returns_rows():
    r = _client().get("/api/scan?on_date=2026-06-10&min_dte=8&max_dte=10&gaps=1&ref_delta=0.5")
    assert r.status_code == 200
    assert "rows" in r.get_json()


def test_expirations_endpoint():
    r = _client().get("/api/expirations?ref=2026-06-08&min_dte=1&max_dte=3")
    assert r.status_code == 200
    assert r.get_json()["expirations"] == ["2026-06-09", "2026-06-10", "2026-06-11"]
