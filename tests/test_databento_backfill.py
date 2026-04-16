from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src_v2.backtest.databento_backfill import (
    fetch_day_quotes,
    build_request_window,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def test_build_request_window_converts_et_to_utc_iso():
    start_iso, end_iso = build_request_window(
        session_date=date(2025, 9, 22),
        anchor_minute_et="09:32",
        before_min=2,
        after_min=10,
    )
    # 09:30 ET == 13:30 UTC (DST), 09:42 ET == 13:42 UTC
    assert start_iso == "2025-09-22T13:30:00"
    assert end_iso == "2025-09-22T13:42:00"


def _fake_dbn_dataframe(rows):
    """Build a DataFrame matching what client.timeseries.get_range(...).to_df() returns
    for an OPRA cbbo-1m request. Index is a DatetimeIndex of ts_recv (UTC)."""
    if not rows:
        # Empty DataFrame with correct index
        df = pd.DataFrame(columns=["symbol", "bid_px_00", "ask_px_00"])
        df.index = pd.DatetimeIndex([], tz="UTC", name="ts_recv")
        return df
    df = pd.DataFrame(rows)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df = df.set_index("ts_recv")
    return df


def test_fetch_day_quotes_parses_into_pickle_format():
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.to_df.return_value = _fake_dbn_dataframe([
        {"ts_recv": "2025-09-22T13:30:00Z", "symbol": "SPXW  250922P06445000", "bid_px_00": 0.10, "ask_px_00": 0.15},
        {"ts_recv": "2025-09-22T13:30:00Z", "symbol": "SPXW  250922C06445000", "bid_px_00": 12.30, "ask_px_00": 12.50},
        {"ts_recv": "2025-09-22T13:31:00Z", "symbol": "SPXW  250922P06445000", "bid_px_00": 0.11, "ask_px_00": 0.16},
    ])
    fake_client.timeseries.get_range.return_value = fake_response

    quotes = fetch_day_quotes(
        client=fake_client,
        session_date=date(2025, 9, 22),
        anchor_minute_et="09:32",
        atm_strike=6445.0,
        half_width_pts=50,
        step_pts=5,
        before_min=2,
        after_min=10,
    )

    assert quotes[("09:30", 6445.0, False)] == (0.10, 0.15)
    assert quotes[("09:30", 6445.0, True)] == (12.30, 12.50)
    assert quotes[("09:31", 6445.0, False)] == (0.11, 0.16)


def test_fetch_day_quotes_passes_correct_request_params():
    fake_client = MagicMock()
    fake_client.timeseries.get_range.return_value.to_df.return_value = _fake_dbn_dataframe([])

    fetch_day_quotes(
        client=fake_client,
        session_date=date(2025, 9, 22),
        anchor_minute_et="09:32",
        atm_strike=6445.0,
        half_width_pts=10,
        step_pts=5,
        before_min=2,
        after_min=10,
    )

    call_kwargs = fake_client.timeseries.get_range.call_args.kwargs
    assert call_kwargs["dataset"] == "OPRA.PILLAR"
    assert call_kwargs["schema"] == "cbbo-1m"
    assert call_kwargs["stype_in"] == "raw_symbol"
    assert call_kwargs["start"] == "2025-09-22T13:30:00"
    assert call_kwargs["end"] == "2025-09-22T13:42:00"
    # 5 strikes (6435..6455 step 5) × 2 sides
    assert len(call_kwargs["symbols"]) == 10
    assert "SPXW  250922P06445000" in call_kwargs["symbols"]
    assert "SPXW  250922C06445000" in call_kwargs["symbols"]
