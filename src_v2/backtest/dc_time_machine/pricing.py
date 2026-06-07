# src_v2/backtest/dc_time_machine/pricing.py
"""Pure payoff math for the DC Time Machine. All values are $/share."""
from __future__ import annotations


def entry_debit(front_put_bid: float, front_call_bid: float,
                back_put_ask: float, back_call_ask: float) -> float:
    """Net debit to open the double calendar: buy backs @ ask, sell fronts @ bid."""
    return (back_put_ask + back_call_ask) - (front_put_bid + front_call_bid)


def transform_credit(back_put_bid: float, back_call_bid: float,
                     put_wing_ask: float, call_wing_ask: float) -> float:
    """Quote-achievable transform credit: sell backs @ bid, buy front wings @ ask."""
    return (back_put_bid + back_call_bid) - (put_wing_ask + call_wing_ask)


def is_risk_free(debit: float, credit: float, wing: float) -> bool:
    """The transform is risk-free iff banked cash (C - D) covers the wing width."""
    return (credit - debit) >= wing


def ic_terminal_pnl(settlement: float, k_put: float, k_call: float,
                    wing: float, bank: float) -> float:
    """Terminal P&L ($/share) of the front-expiry IC held to cash settlement.

    bank = C - D (cash already banked). Only one vertical can be ITM (k_put < k_call).
    """
    put_loss = min(max(k_put - settlement, 0.0), wing)
    call_loss = min(max(settlement - k_call, 0.0), wing)
    return bank - put_loss - call_loss
