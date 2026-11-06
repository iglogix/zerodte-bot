# tests/dc_time_machine/test_simulator.py
from datetime import date

import pytest

from zerodte.backtest.dc_time_machine.strategy import DcParams
from zerodte.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from zerodte.backtest.dc_time_machine.simulator import simulate_trade

S = date(2026, 2, 2)
F = date(2026, 2, 11)
B = date(2026, 2, 12)
MIN = "09:40"


def _build_panel(transform_minute="09:42", settle=5000.0):
    """Synthetic panel where the transform credit clears D+W at transform_minute.

    Strikes: K_p=4980 (put, ~35Δ proxy), K_c=5020 (call), wings at 4975 / 5025.
    Two minutes present: entry (09:40) and a later minute where back bids rise.
    """
    # Entry @ 09:40: D = backs@ask − fronts@bid
    entry = {
        # fronts (sold)
        ("09:40", 4980.0, False): (24.9, 25.1),   # front put bid 24.9
        ("09:40", 5020.0, True): (24.9, 25.1),    # front call bid 24.9
        # backs (bought)
        ("09:40", 4980.0, False, ): (24.9, 25.1),
    }
    # Build full quote sets for both expiries at two minutes
    def qm_front(minute, put_bid, call_bid):
        return {
            (minute, 4980.0, False): (put_bid, put_bid + 0.2),     # front put (short)
            (minute, 5020.0, True): (call_bid, call_bid + 0.2),    # front call (short)
            (minute, 4975.0, False): (0.40, 0.60),                 # put wing (buy)
            (minute, 5025.0, True): (0.40, 0.60),                  # call wing (buy)
            (minute, 5000.0, True): (40.0, 41.0),                  # ATM for spot
            (minute, 5000.0, False): (40.0, 41.0),
        }

    def qm_back(minute, put_bid, call_bid):
        return {
            (minute, 4980.0, False): (put_bid, put_bid + 0.2),     # back put (long)
            (minute, 5020.0, True): (call_bid, call_bid + 0.2),    # back call (long)
        }

    panel = {
        S: {
            F: {**qm_front("09:40", 25.0, 25.0), **qm_front("09:42", 25.0, 25.0)},
            B: {**qm_back("09:40", 25.5, 25.5), **qm_back("09:42", 31.0, 31.0)},
        },
        F: {  # settlement panel for F
            F: {
                ("15:59", round(settle), True): (max(settle - 5000.0, 0.0), max(settle - 5000.0, 0.0)),
                ("15:59", round(settle), False): (max(5000.0 - settle, 0.0), max(5000.0 - settle, 0.0)),
                ("15:59", 5000.0, True): (settle - 5000.0 if settle > 5000 else 0.0,) * 2,
                ("15:59", 5000.0, False): (5000.0 - settle if settle < 5000 else 0.0,) * 2,
            }
        },
    }
    return panel


def _params(**kw):
    base = dict(entry_minute=MIN, front_dte=1, gap=1, wing_width=5.0,
                fill_k=0.0, entry_filter="always", overnight_policy="close",
                target_delta=0.35)
    base.update(kw)
    return DcParams(**base)


def test_transform_fires_and_banks_positive():
    store = MultiExpiryStore(preloaded=_build_panel(settle=5000.0))
    res = simulate_trade(store, _params(), session=S, minutes=["09:40", "09:42"],
                         k_put=4980.0, k_call=5020.0, front=F, back=B)
    assert res.outcome == "transformed"
    assert res.transform_minute == "09:42"
    # D = backs@ask(25.7+25.7) − fronts@bid(25.0+25.0) = 1.4 ; C(09:42) net-mid
    #   = back bids(31.0+31.0) − wing asks(0.6+0.6) = 60.8 → bank huge, risk-free
    assert res.bank >= res.params_wing
    assert res.pnl > 0


def test_stop_loss_closes_when_dc_drops_20pct():
    panel = _build_panel()
    # Make the only later minute show a deep DC loss instead of a transform:
    panel[S][B][("09:42", 4980.0, False)] = (5.0, 5.2)   # back put bid collapses
    panel[S][B][("09:42", 5020.0, True)] = (5.0, 5.2)
    store = MultiExpiryStore(preloaded=panel)
    res = simulate_trade(store, _params(), session=S, minutes=["09:40", "09:42"],
                         k_put=4980.0, k_call=5020.0, front=F, back=B)
    assert res.outcome == "stopped"
    assert res.pnl < 0


def test_no_entry_when_filter_blocks():
    store = MultiExpiryStore(preloaded=_build_panel())
    res = simulate_trade(store, _params(entry_filter="fri_mon"), session=S,
                         minutes=["09:40", "09:42"], k_put=4980.0, k_call=5020.0,
                         front=F, back=B, is_fri_mon=False)
    assert res.outcome == "no_entry"
    assert res.pnl == 0.0
