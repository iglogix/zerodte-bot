# tests/dc_time_machine/test_run_session.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import run_session

S = date(2026, 2, 2)
F = date(2026, 2, 3)
B = date(2026, 2, 4)


def _panel():
    def front(minute, pb, cb):
        return {
            (minute, 4980.0, False): (pb, pb + 0.2),
            (minute, 5020.0, True): (cb, cb + 0.2),
            (minute, 4975.0, False): (0.4, 0.6),
            (minute, 5025.0, True): (0.4, 0.6),
            (minute, 5000.0, True): (40.0, 41.0),
            (minute, 5000.0, False): (40.0, 41.0),
            (minute, 4980.0, True): (45.0, 46.0),   # for delta scan
            (minute, 5020.0, False): (45.0, 46.0),
        }

    def back(minute, pb, cb):
        return {(minute, 4980.0, False): (pb, pb + 0.2), (minute, 5020.0, True): (cb, cb + 0.2)}

    return {
        S: {
            F: {**front("09:40", 25.0, 25.0), **front("09:42", 25.0, 25.0)},
            B: {**back("09:40", 25.5, 25.5), **back("09:42", 31.0, 31.0)},
        },
        F: {F: {("15:59", 5000.0, True): (5.0, 5.0), ("15:59", 5000.0, False): (5.0, 5.0)}},
    }


def test_run_session_produces_a_trade():
    store = MultiExpiryStore(preloaded=_panel())
    params = DcParams(entry_minute="09:40", front_dte=1, gap=1, wing_width=5.0, fill_k=0.0)
    res = run_session(store, params, S)
    assert res is not None
    assert res.outcome in {"transformed", "closed_eod", "stopped", "held_overnight"}
    assert res.front == F and res.back == B
