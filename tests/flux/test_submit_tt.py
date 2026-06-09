"""TastyTradeSubmitter routing — exercised against a fake async client (never a real broker).

Validates that combo → one complex order, leg-in → per-leg orders with longs FIRST at each
leg's own mid, manual → refusal, and broker errors surface as ok=False."""
from datetime import date

from src_v2.flux.orders import OrderLeg, OrderTicket
from src_v2.flux.submit import TastyTradeSubmitter


def _ticket():
    legs = (OrderLeg(date(2026, 6, 22), 7335, False, "Sell to Open", 2, 40.0),
            OrderLeg(date(2026, 6, 23), 7335, False, "Buy to Open", 2, 42.0),
            OrderLeg(date(2026, 6, 22), 7505, True, "Sell to Open", 2, 30.0),
            OrderLeg(date(2026, 6, 23), 7505, True, "Buy to Open", 2, 32.0))
    return OrderTicket("double_calendar", legs, 4.0, "Debit", 800.0, "DC")


class FakeClient:
    """Records calls; returns incrementing order ids. No network."""
    def __init__(self):
        self.complex = []
        self.legs = []
        self._n = 0

    def _id(self):
        self._n += 1
        return {"order_id": f"TT-{self._n}"}

    async def place_complex_order(self, leg_specs, price, price_effect="Debit", tif="Day"):
        self.complex.append((list(leg_specs), price, price_effect))
        return self._id()

    async def place_option_limit(self, osi, action, quantity, limit, tif="Day"):
        self.legs.append((osi, action, quantity, limit))
        return self._id()


def test_combo_sends_one_complex_order():
    c = FakeClient()
    res = TastyTradeSubmitter("sandbox", c).submit(_ticket(), "combo")
    assert res.ok and len(res.order_ids) == 1
    assert len(c.complex) == 1 and len(c.complex[0][0]) == 4      # all 4 legs, one order
    assert c.complex[0][1] == 4.0 and c.complex[0][2] == "Debit"  # net price + effect
    assert c.legs == []


def test_legin_buys_longs_first_at_each_leg_mid():
    c = FakeClient()
    res = TastyTradeSubmitter("live", c).submit(_ticket(), "legin")
    assert res.ok and len(res.order_ids) == 4
    actions = [a for (_osi, a, _q, _lim) in c.legs]
    assert actions[:2] == ["Buy to Open", "Buy to Open"]         # longs first — never naked-short
    assert actions[2:] == ["Sell to Open", "Sell to Open"]
    assert all(q == 2 for (_o, _a, q, _l) in c.legs)             # qty propagated
    assert {lim for (_o, _a, _q, lim) in c.legs} == {42.0, 32.0, 40.0, 30.0}  # each leg's own mid


def test_manual_refuses_even_with_client():
    c = FakeClient()
    res = TastyTradeSubmitter("manual", c).submit(_ticket(), "combo")
    assert res.ok is False and "manual" in res.detail.lower()
    assert c.complex == [] and c.legs == []


def test_broker_error_surfaces_as_not_ok():
    class Boom(FakeClient):
        async def place_complex_order(self, *a, **k):
            raise RuntimeError("rejected: insufficient buying power")
    res = TastyTradeSubmitter("sandbox", Boom()).submit(_ticket(), "combo")
    assert res.ok is False and "buying power" in res.detail


def test_spy_proxy_routes_spy_symbols_at_scaled_price():
    c = FakeClient()
    res = TastyTradeSubmitter("sandbox", c, proxy="spy").submit(_ticket(), "combo")
    assert res.ok and "SPY proxy" in res.detail
    osis = [osi for (osi, _a, _q) in c.complex[0][0]]
    assert all(osi.startswith("SPY ") for osi in osis)     # SPXW → SPY
    assert c.complex[0][1] == 0.4                          # net limit 4.0 ÷ 10


def test_spy_proxy_legin_uses_spy_legs():
    c = FakeClient()
    res = TastyTradeSubmitter("sandbox", c, proxy="spy").submit(_ticket(), "legin")
    assert res.ok and len(res.order_ids) == 4
    assert all(osi.startswith("SPY ") for (osi, _a, _q, _l) in c.legs)
