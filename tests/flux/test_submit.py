from datetime import date

from zerodte.flux.orders import OrderLeg, OrderTicket
from zerodte.flux.submit import combo_legs, legin_groups, MockSubmitter


def _ticket():
    legs = (OrderLeg(date(2026, 6, 22), 7335, False, "Sell to Open", 1, 40.0),
            OrderLeg(date(2026, 6, 23), 7335, False, "Buy to Open", 1, 42.0),
            OrderLeg(date(2026, 6, 22), 7505, True, "Sell to Open", 1, 30.0),
            OrderLeg(date(2026, 6, 23), 7505, True, "Buy to Open", 1, 32.0))
    return OrderTicket("double_calendar", legs, 5.7, "Debit", 570.0, "DC")


def test_combo_legs_are_all_four():
    cl = combo_legs(_ticket())
    assert len(cl) == 4 and cl[0] == ("SPXW  260622P07335000", "Sell to Open", 1)


def test_legin_groups_longs_first_then_shorts():
    g = legin_groups(_ticket())
    assert len(g) == 2
    assert all(l.is_long for l in g[0]) and len(g[0]) == 2          # buy the backs first
    assert all(not l.is_long for l in g[1]) and len(g[1]) == 2      # then sell the fronts


def test_mock_manual_mode_refuses():
    r = MockSubmitter("manual").submit(_ticket(), "combo")
    assert r.ok is False and "manual" in r.detail.lower()


def test_mock_combo_one_order_legin_four_orders():
    s = MockSubmitter("sandbox")
    assert s.submit(_ticket(), "combo").ok and len(s.submit(_ticket(), "combo").order_ids) == 1
    assert len(s.submit(_ticket(), "legin").order_ids) == 4
    assert s.calls[-1][1] == "legin"
