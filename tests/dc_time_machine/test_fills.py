import pytest

from zerodte.backtest.dc_time_machine.fills import fill_price, net_combo_credit


def test_buy_fill_walks_from_mid_to_ask():
    assert fill_price(bid=1.0, ask=2.0, side="buy", k=0.0) == pytest.approx(1.5)  # mid
    assert fill_price(bid=1.0, ask=2.0, side="buy", k=1.0) == pytest.approx(2.0)  # ask
    assert fill_price(bid=1.0, ask=2.0, side="buy", k=0.5) == pytest.approx(1.75)


def test_sell_fill_walks_from_mid_to_bid():
    assert fill_price(bid=1.0, ask=2.0, side="sell", k=0.0) == pytest.approx(1.5)  # mid
    assert fill_price(bid=1.0, ask=2.0, side="sell", k=1.0) == pytest.approx(1.0)  # bid


def test_net_combo_credit_brackets_mid_to_cross():
    legs = [
        (5.4, 5.6, "sell"),  # back put
        (5.4, 5.6, "sell"),  # back call
        (0.3, 0.5, "buy"),   # put wing
        (0.4, 0.6, "buy"),   # call wing
    ]
    assert net_combo_credit(legs, k=0.0) == pytest.approx(10.1)  # net mid
    assert net_combo_credit(legs, k=1.0) == pytest.approx(9.7)   # net cross (−4×0.1)


def test_unknown_side_raises():
    with pytest.raises(ValueError):
        fill_price(1.0, 2.0, side="hold", k=0.0)
