# tests/dc_time_machine/test_spreads.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.spreads import half_spread_stats

S = date(2026, 2, 2)
F = date(2026, 2, 11)   # 9 DTE


def _panel():
    return {S: {F: {
        ("09:40", 5000.0, True): (10.0, 10.4),   # half-spread 0.20
        ("09:40", 5000.0, False): (10.0, 10.6),  # half-spread 0.30
        ("14:00", 5000.0, True): (10.0, 11.0),   # half-spread 0.50
    }}}


def test_half_spread_stats_returns_percentiles_by_bucket():
    store = MultiExpiryStore(preloaded=_panel())
    stats = half_spread_stats(store, sessions=[S])
    key = (9, "open")   # 9 DTE, morning bucket
    assert key in stats
    assert stats[key]["median"] == pytest.approx(0.25, abs=1e-9)   # median of {0.20, 0.30}
    assert stats[key]["n"] == 2
    assert (9, "midday") in stats   # the 14:00 quote
