# tests/dc_time_machine/test_dc_backfill.py
from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from zerodte.backtest.dc_time_machine.dc_backfill import fetch_expiry_session


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    return df.set_index("ts_recv")


def test_fetch_expiry_session_uses_distinct_expiry_and_full_window():
    client = MagicMock()
    client.timeseries.get_range.return_value.to_df.return_value = _df([
        {"ts_recv": "2026-02-02T14:40:00Z", "symbol": "SPXW  260211P05000000", "bid_px_00": 40.0, "ask_px_00": 41.0},
        {"ts_recv": "2026-02-02T14:40:00Z", "symbol": "SPXW  260211C05000000", "bid_px_00": 40.0, "ask_px_00": 41.0},
    ])
    quotes = fetch_expiry_session(
        client, session_date=date(2026, 2, 2), expiry_date=date(2026, 2, 11),
        atm_strike=5000.0, half_width_pts=10, step_pts=5,
        start_et="09:30", end_et="16:00",
    )
    kw = client.timeseries.get_range.call_args.kwargs
    assert kw["dataset"] == "OPRA.PILLAR" and kw["schema"] == "cbbo-1m"
    # expiry (Feb 11) differs from session (Feb 2) → symbols carry 260211
    assert any("260211" in s for s in kw["symbols"])
    # full-day window: 09:30 ET == 14:30 UTC (EST), 16:00 ET == 21:00 UTC
    assert kw["start"] == "2026-02-02T14:30:00"
    assert kw["end"] == "2026-02-02T21:00:00"
    assert quotes[("09:40", 5000.0, False)] == (40.0, 41.0)
