"""Tests for the targeted 0DTE MEIC CBBO backfill helpers."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from zerodte.backtest.meic_backfill import (
    spot_range_from_bars, meic_day_symbols, meic_window_utc, quotes_from_df,
)
from zerodte.backtest.osi_symbol import format_osi

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _bar(h, m, low, high):
    return {"ts_et": datetime(2023, 6, 15, h, m, tzinfo=ET),
            "low": low, "high": high, "open": low, "close": high}


class TestSpotRange:
    def test_min_low_max_high_within_window(self):
        bars = [
            _bar(10, 30, 6600, 6610),   # before 11:00 -> excluded
            _bar(11, 0, 6650, 6660),
            _bar(13, 0, 6630, 6700),
            _bar(15, 50, 6640, 6655),
            _bar(15, 55, 6500, 6800),   # after 15:50 -> excluded
        ]
        lo, hi = spot_range_from_bars(bars, "11:00", "15:50")
        assert lo == 6630
        assert hi == 6700

    def test_none_when_no_bars_in_window(self):
        bars = [_bar(9, 35, 6600, 6610), _bar(16, 0, 6500, 6510)]
        assert spot_range_from_bars(bars, "11:00", "15:50") is None


class TestDaySymbols:
    def test_spans_range_plus_buffer_both_sides(self):
        syms = meic_day_symbols(date(2023, 6, 15), spot_low=6650, spot_high=6700,
                                buffer_pts=300, step_pts=5)
        # strikes 6350..7000 step 5 -> 131 strikes x 2 (C+P) = 262 symbols
        assert len(syms) == 262
        # lowest put and highest call present, correctly formatted
        assert format_osi(date(2023, 6, 15), 6350.0, is_call=False) in syms
        assert format_osi(date(2023, 6, 15), 7000.0, is_call=True) in syms
        # every symbol is a valid 21-char SPXW OSI string
        assert all(len(s) == 21 and s.startswith("SPXW") for s in syms)

    def test_both_call_and_put_per_strike(self):
        syms = meic_day_symbols(date(2023, 6, 15), 6650, 6650, buffer_pts=5, step_pts=5)
        # strikes 6645,6650,6655 -> 3 x 2 = 6
        assert len(syms) == 6
        for k in (6645.0, 6650.0, 6655.0):
            assert format_osi(date(2023, 6, 15), k, is_call=True) in syms
            assert format_osi(date(2023, 6, 15), k, is_call=False) in syms


class TestWindowUtc:
    def test_summer_edt_offset(self):
        # 2023-06-15 is EDT (UTC-4): 11:00->15:00Z, 15:50->19:50Z
        start, end = meic_window_utc(date(2023, 6, 15), "11:00", "15:50")
        assert start == "2023-06-15T15:00:00"
        assert end == "2023-06-15T19:50:00"

    def test_winter_est_offset(self):
        # 2023-01-04 is EST (UTC-5): 11:00->16:00Z, 15:50->20:50Z
        start, end = meic_window_utc(date(2023, 1, 4), "11:00", "15:50")
        assert start == "2023-01-04T16:00:00"
        assert end == "2023-01-04T20:50:00"


class TestQuotesFromDf:
    def _df(self, rows):
        idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
        return pd.DataFrame(
            {"symbol": [r[1] for r in rows],
             "bid_px_00": [r[2] for r in rows],
             "ask_px_00": [r[3] for r in rows]},
            index=idx,
        )

    def test_parses_and_keys_by_minute_strike_side(self):
        ts = datetime(2023, 6, 15, 15, 0, tzinfo=UTC)  # 11:00 ET
        df = self._df([
            (ts, format_osi(date(2023, 6, 15), 6650.0, is_call=True), 1.40, 1.50),
            (ts, format_osi(date(2023, 6, 15), 6550.0, is_call=False), 1.30, 1.40),
        ])
        q = quotes_from_df(df)
        assert q[("11:00", 6650.0, True)] == (1.40, 1.50)
        assert q[("11:00", 6550.0, False)] == (1.30, 1.40)

    def test_drops_nan_and_nonpositive(self):
        ts = datetime(2023, 6, 15, 15, 0, tzinfo=UTC)
        df = self._df([
            (ts, format_osi(date(2023, 6, 15), 6650.0, is_call=True), float("nan"), 1.50),
            (ts, format_osi(date(2023, 6, 15), 6655.0, is_call=True), 0.0, 0.0),
            (ts, format_osi(date(2023, 6, 15), 6660.0, is_call=True), 0.10, 0.20),
        ])
        q = quotes_from_df(df)
        assert ("11:00", 6650.0, True) not in q   # NaN bid dropped
        assert ("11:00", 6655.0, True) not in q   # both <= 0 dropped
        assert q[("11:00", 6660.0, True)] == (0.10, 0.20)

    def test_empty_df(self):
        assert quotes_from_df(pd.DataFrame()) == {}
