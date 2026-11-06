from datetime import date, datetime, timezone

import pytest

from zerodte.flux.models import ChainSnapshot, OptionRow
from zerodte.flux.orders import (
    select_dc_strikes, build_dc_order, spy_proxy, snap_spy_to_listed, OrderLeg,
)


def _chain(expiry, rows):
    return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), tuple(rows))


def _front():
    # calls at +.30/.35/.45 delta ; puts at -.30/.35/.45 delta
    return _chain(date(2026, 6, 17), [
        OptionRow(7050, True, 24.5, 25.5, 0.18, 0.30), OptionRow(7040, True, 29.5, 30.5, 0.18, 0.35),
        OptionRow(7020, True, 39.5, 40.5, 0.18, 0.45),
        OptionRow(6950, False, 24.5, 25.5, 0.18, -0.30), OptionRow(6960, False, 29.5, 30.5, 0.18, -0.35),
        OptionRow(6980, False, 39.5, 40.5, 0.18, -0.45),
    ])


def _back():
    # same strikes, slightly richer (1-day-later)
    return _chain(date(2026, 6, 18), [
        OptionRow(7040, True, 31.5, 32.5, 0.18, 0.36), OptionRow(6960, False, 31.5, 32.5, 0.18, -0.36),
    ])


def test_select_dc_strikes_picks_nearest_target_delta():
    kp, kc = select_dc_strikes(_front(), target_delta=0.35)
    assert kc == 7040.0 and kp == 6960.0           # nearest +0.35 call / -0.35 put


def test_select_dc_strikes_none_on_empty():
    kp, kc = select_dc_strikes(ChainSnapshot("SPX", date(2026, 6, 17), None, ()), 0.35)
    assert kp is None and kc is None


def test_build_dc_order_legs_and_debit():
    t = build_dc_order(_front(), _back(), k_put=6960.0, k_call=7040.0, qty=2)
    assert t.kind == "double_calendar" and t.price_effect == "Debit"
    # debit = back mids (32 + 32) − front mids (30 + 30) = 4.0
    assert t.price == pytest.approx(4.0)
    assert t.est_cost == pytest.approx(4.0 * 100 * 2)
    # 4 legs: sell front put, buy back put, sell front call, buy back call (2 expiries)
    acts = {(l.expiry, l.is_call, l.action) for l in t.legs}
    assert (date(2026, 6, 17), False, "Sell to Open") in acts
    assert (date(2026, 6, 18), False, "Buy to Open") in acts
    assert (date(2026, 6, 17), True, "Sell to Open") in acts
    assert (date(2026, 6, 18), True, "Buy to Open") in acts
    assert all(l.quantity == 2 for l in t.legs)


def test_order_leg_osi_encodes_expiry_strike_right():
    leg = OrderLeg(date(2026, 6, 18), 7040.0, True, "Buy to Open", 1)
    assert leg.osi == "SPXW  260618C07040000"


def test_build_dc_order_none_when_strike_missing():
    assert build_dc_order(_front(), _back(), k_put=9999.0, k_call=7040.0) is None


def test_spy_proxy_maps_strikes_limits_and_root():
    t = build_dc_order(_front(), _back(), k_put=6960.0, k_call=7040.0, qty=2)
    p = spy_proxy(t)
    # strikes ÷10 → nearest $1; same expiries/actions/qty preserved
    assert {l.strike for l in p.legs} == {696.0, 704.0}
    assert p.price == pytest.approx(0.4)          # 4.0 ÷ 10
    assert p.est_cost == pytest.approx(80.0)      # 800 ÷ 10
    assert all(l.root == "SPY" for l in p.legs) and all(l.quantity == 2 for l in p.legs)
    # OSI now SPY, not SPXW
    putleg = next(l for l in p.legs if not l.is_call and l.action.startswith("Sell"))
    assert putleg.osi == "SPY   260617P00696000"
    # each leg's mid also scaled (front put K6960 mid 30 → 3.0)
    assert putleg.limit == pytest.approx(3.0)


def _spy_chain():
    """SPY lists Fridays only here: 696/700 strikes, two expiries a week apart."""
    def syms(exp, k, cp):
        return f"SPY   {exp.replace('-','')[2:]}{cp}{int(k*1000):08d}"
    out = {}
    for exp in ("2026-06-19", "2026-06-26"):
        out[exp] = {k: {"call": syms(exp, k, "C"), "put": syms(exp, k, "P")}
                    for k in (696.0, 700.0, 705.0)}
    return out


def test_snap_spy_to_listed_uses_real_contracts():
    t = build_dc_order(_front(), _back(), k_put=6960.0, k_call=7040.0, qty=1)  # 696/704 after ÷10
    snapped = snap_spy_to_listed(spy_proxy(t), _spy_chain())
    assert snapped is not None
    # two distinct LISTED expiries (Fri+Fri a week apart), not the synthesized Jun-17/18
    exps = sorted({l.expiry.isoformat() for l in snapped.legs})
    assert exps == ["2026-06-19", "2026-06-26"]
    # call 704 snaps to nearest listed (705); put 696 is listed exactly
    assert {l.strike for l in snapped.legs if l.is_call} == {705.0}
    assert {l.strike for l in snapped.legs if not l.is_call} == {696.0}
    # legs carry the broker symbol verbatim
    assert all(l.osi.startswith("SPY ") for l in snapped.legs)
    assert any(l.osi == "SPY   260619P00696000" for l in snapped.legs)


def test_snap_returns_none_when_under_two_expiries():
    one = {"2026-06-19": {696.0: {"call": "c", "put": "p"}}}
    assert snap_spy_to_listed(spy_proxy(build_dc_order(_front(), _back(), 6960.0, 7040.0)), one) is None
