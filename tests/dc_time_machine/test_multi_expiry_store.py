# tests/dc_time_machine/test_multi_expiry_store.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore

S = date(2026, 2, 2)        # session
F = date(2026, 2, 11)       # front expiry
B = date(2026, 2, 12)       # back expiry


def _panel():
    # ATM ~5000; provide call & put at 5000 for spot/settlement parity
    return {
        S: {
            F: {
                ("09:40", 5000.0, True): (40.0, 41.0),
                ("09:40", 5000.0, False): (40.0, 41.0),
                ("09:41", 4980.0, False): (30.0, 31.0),
            },
            B: {
                ("09:40", 5000.0, True): (45.0, 46.0),
            },
        },
        # session == front expiry holds the 0DTE settlement panel for F
        F: {
            F: {
                ("15:59", 5000.0, True): (12.0, 12.0),
                ("15:59", 5000.0, False): (8.0, 8.0),
            },
        },
    }


def test_lookup_exact_and_stale():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    assert store.lookup(F, "09:40", 5000.0, True) == (40.0, 41.0)
    # 09:42 not present → walk back to 09:41
    assert store.lookup(F, "09:42", 4980.0, False, max_stale=5) == (30.0, 31.0)
    assert store.lookup(F, "09:40", 9999.0, True) is None


def test_available_expiries_sorted():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    assert store.available_expiries() == [F, B]


def test_derive_spot_from_parity():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    # call_mid == put_mid at 5000 → spot ≈ 5000
    assert store.derive_spot(F, "09:40", 5000.0) == pytest.approx(5000.0, abs=0.6)


def test_settlement_uses_expiry_own_session_panel():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    # at F's last minute: 5000 + call_mid(12) − put_mid(8) = 5004
    assert store.settlement(F) == pytest.approx(5004.0, abs=0.5)
