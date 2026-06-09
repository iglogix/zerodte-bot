from datetime import date, datetime, timezone

import pytest

from src_v2.flux.models import ChainSnapshot, OptionRow
from src_v2.flux.positions import (
    parse_option_symbol, detect_double_calendars, build_transform_order, DoubleCalendar,
)


def test_parse_option_symbol_round_trips_osi():
    root, exp, k, is_call = parse_option_symbol("SPXW  260622P07335000")
    assert root == "SPXW" and exp == date(2026, 6, 22) and k == 7335.0 and is_call is False


def test_parse_option_symbol_rejects_non_option():
    assert parse_option_symbol("AAPL") is None
    assert parse_option_symbol("SPY   260618X00734000") is None   # bad C/P slot


def _dc_positions():
    """A real DC: short front P/C @ 7335/7505, long back P/C @ same strikes."""
    def osi(exp, k, cp):
        return f"SPXW  {exp}{cp}{int(k*1000):08d}"
    return [
        {"symbol": osi("260622", 7335, "P"), "quantity": 1, "direction": "Short"},
        {"symbol": osi("260622", 7505, "C"), "quantity": 1, "direction": "Short"},
        {"symbol": osi("260623", 7335, "P"), "quantity": 1, "direction": "Long"},
        {"symbol": osi("260623", 7505, "C"), "quantity": 1, "direction": "Long"},
        {"symbol": "SPY", "quantity": 100, "direction": "Long"},   # noise, ignored
    ]


def test_detect_double_calendars_finds_the_dc():
    dcs = detect_double_calendars(_dc_positions())
    assert len(dcs) == 1
    dc = dcs[0]
    assert dc.root == "SPXW" and dc.k_put == 7335.0 and dc.k_call == 7505.0
    assert dc.front_expiry == date(2026, 6, 22) and dc.back_expiry == date(2026, 6, 23)
    assert dc.qty == 1


def test_detect_ignores_incomplete_legs():
    # only a put calendar, no call calendar → not a DC
    pos = [p for p in _dc_positions() if "C0" not in p["symbol"]]
    assert detect_double_calendars(pos) == []


def _chain(expiry, rows):
    return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), tuple(rows))


def test_build_transform_order_sells_backs_buys_wings():
    dc = DoubleCalendar("SPXW", date(2026, 6, 22), date(2026, 6, 23), 7335.0, 7505.0, 1)
    wing = 25.0
    # back chain: backs worth mid 40 each ; front chain: wings cost mid 8 each
    back = _chain(date(2026, 6, 23), [
        OptionRow(7335, False, 39, 41, 0.18, -0.35), OptionRow(7505, True, 39, 41, 0.18, 0.35)])
    front = _chain(date(2026, 6, 22), [
        OptionRow(7310, False, 7, 9, 0.18, -0.20), OptionRow(7530, True, 7, 9, 0.18, 0.20)])
    t = build_transform_order(dc, wing, front, back)
    assert t.kind == "transform" and t.price_effect == "Credit"
    # credit = (40 + 40) − (8 + 8) = 64
    assert t.price == pytest.approx(64.0)
    acts = {(l.expiry, l.is_call, round(l.strike), l.action) for l in t.legs}
    assert (date(2026, 6, 23), False, 7335, "Sell to Close") in acts    # sell back put
    assert (date(2026, 6, 23), True, 7505, "Sell to Close") in acts     # sell back call
    assert (date(2026, 6, 22), False, 7310, "Buy to Open") in acts      # buy put wing K−W
    assert (date(2026, 6, 22), True, 7530, "Buy to Open") in acts       # buy call wing K+W
    assert all(l.quantity == 1 for l in t.legs)


def test_build_transform_none_when_wing_missing():
    dc = DoubleCalendar("SPXW", date(2026, 6, 22), date(2026, 6, 23), 7335.0, 7505.0, 1)
    back = _chain(date(2026, 6, 23), [
        OptionRow(7335, False, 39, 41, 0.18, -0.35), OptionRow(7505, True, 39, 41, 0.18, 0.35)])
    front = _chain(date(2026, 6, 22), [])     # no wing quotes
    assert build_transform_order(dc, 25.0, front, back) is None
