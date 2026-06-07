# tests/dc_time_machine/test_resolution.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import (
    resolve_expiries, select_strikes, compute_iv_features,
)

S = date(2026, 2, 2)
EXPS = [date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 11), date(2026, 2, 12)]


def _panel_for_resolution():
    front_strikes = {}
    for kk in range(4900, 5101, 5):
        # symmetric smile-free chain at 18% vol-ish prices is unnecessary; provide
        # monotone put/call mids so delta selection has a unique solution
        front_strikes[("09:40", float(kk), True)] = (max(5000 - kk, 0) + 20.0, max(5000 - kk, 0) + 21.0)
        front_strikes[("09:40", float(kk), False)] = (max(kk - 5000, 0) + 20.0, max(kk - 5000, 0) + 21.0)
    panel = {S: {e: dict(front_strikes) for e in EXPS}}
    return panel


def test_resolve_expiries_picks_front_and_back_by_index():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    # only 4 expiries exist; front_dte=3 → 3rd expiry after S (index 2) = 2026-02-11, gap 1 → 2026-02-12
    front, back = resolve_expiries(store, S, DcParams(front_dte=3, gap=1))
    assert front == date(2026, 2, 11)
    assert back == date(2026, 2, 12)


def test_resolve_expiries_returns_none_when_out_of_range():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    assert resolve_expiries(store, S, DcParams(front_dte=9, gap=1)) == (None, None)


def test_select_strikes_returns_put_below_call_above_spot():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    front = date(2026, 2, 11)
    k_put, k_call = select_strikes(store, S, front, DcParams(target_delta=0.35), spot=5000.0)
    assert k_put is not None and k_call is not None
    assert k_put < 5000.0 < k_call


def test_compute_iv_features_has_ratio_and_flags():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    feats = compute_iv_features(store, S, front=date(2026, 2, 11), back=date(2026, 2, 12),
                                minute="09:40", spot=5000.0, params=DcParams())
    assert "ratio" in feats and "is_fri_mon" in feats
    assert feats["is_fri_mon"] is False   # 2026-02-11 is a Wednesday
