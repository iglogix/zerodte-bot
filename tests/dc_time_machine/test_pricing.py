# tests/dc_time_machine/test_pricing.py
import pytest

from zerodte.backtest.dc_time_machine.pricing import (
    entry_debit,
    ic_terminal_pnl,
    is_risk_free,
    transform_credit,
)


def test_entry_debit_is_backs_minus_fronts():
    # buy backs @ ask (5.0 + 5.2), sell fronts @ bid (4.9 + 5.0)
    assert entry_debit(front_put_bid=4.9, front_call_bid=5.0,
                       back_put_ask=5.0, back_call_ask=5.2) == pytest.approx(0.3)


def test_transform_credit_is_backbids_minus_wingasks():
    # sell back bids (5.5 + 5.6), buy wing asks (0.4 + 0.5)
    assert transform_credit(back_put_bid=5.5, back_call_bid=5.6,
                            put_wing_ask=0.4, call_wing_ask=0.5) == pytest.approx(10.2)


def test_his_example_reconciles():
    # Video: D=10.10, C=15.25, W=5 → bank 5.15; inside tent = 5.15/share, floor = 0.15/share
    D, C, W = 10.10, 15.25, 5.0
    bank = C - D
    assert is_risk_free(D, C, W) is True
    assert ic_terminal_pnl(settlement=5000, k_put=4900, k_call=5100, wing=W, bank=bank) == pytest.approx(5.15)
    # settle far below the put long → floor
    assert ic_terminal_pnl(settlement=4000, k_put=4900, k_call=5100, wing=W, bank=bank) == pytest.approx(0.15)


def test_partial_penetration_is_linear():
    bank = 6.0
    # S between short put (100) and long put (95): loss = 100 - 98 = 2
    assert ic_terminal_pnl(settlement=98, k_put=100, k_call=110, wing=5, bank=bank) == pytest.approx(4.0)


def test_risk_free_requires_bank_ge_wing():
    assert is_risk_free(debit=10.0, credit=14.99, wing=5.0) is False
    assert is_risk_free(debit=10.0, credit=15.00, wing=5.0) is True
