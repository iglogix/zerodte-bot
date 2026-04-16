from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from src_v2.backtest.signal_detector import (
    SignalEvent,
    compute_orb_levels,
    detect_signals_for_day,
)

ET = ZoneInfo("America/New_York")


def _bar(t: str, o: float, h: float, l: float, c: float):
    """Build a one-minute bar dict in the format DatabentoLoader yields."""
    hh, mm = int(t[:2]), int(t[3:5])
    return {
        "ts_et": datetime(2024, 3, 20, hh, mm, tzinfo=ET),
        "open": o, "high": h, "low": l, "close": c, "volume": 100,
    }


def test_compute_orb_levels_takes_window_high_low():
    bars = [
        _bar("08:00", 5000, 5005, 4998, 5003),
        _bar("08:05", 5003, 5010, 5001, 5008),
        _bar("08:14", 5008, 5012, 5007, 5011),
        _bar("08:15", 5011, 5013, 5010, 5012),  # outside window (end exclusive)
    ]
    lo, mid, hi = compute_orb_levels(bars, "08:00", "08:15")
    assert hi == 5012
    assert lo == 4998
    assert mid == pytest.approx(5005.0)


def test_compute_orb_levels_returns_none_if_window_empty():
    bars = [_bar("07:30", 5000, 5005, 4998, 5003)]
    assert compute_orb_levels(bars, "08:00", "08:15") == (None, None, None)


def test_detect_signals_breakout_level_long():
    """Long bias: price breaks ORB high after 09:30, then retests ORB high → signal."""
    bars = (
        # ORB window 08:00-08:14 → high=5012, low=4998
        [_bar(f"08:{m:02d}", 5000, 5012 if m == 7 else 5005, 4998 if m == 3 else 5000, 5005) for m in range(15)]
        # 09:30 onward
        + [_bar("09:30", 5005, 5008, 5004, 5007),
           _bar("09:31", 5007, 5015, 5006, 5014),  # breakout above 5012 → long bias
           _bar("09:32", 5014, 5016, 5012, 5012),  # retest of 5012 → SIGNAL here
           _bar("09:33", 5012, 5018, 5011, 5017)]
    )
    sigs = detect_signals_for_day(
        bars, session_date=date(2024, 3, 20),
        orb_start="08:00", orb_end="08:15", market_open="09:30",
        eod="15:50", retest_mode="breakout_level",
        midline_tol_pts=1.0, min_range_pts=2.0, max_range_pts=30.0,
    )
    assert len(sigs) == 1
    assert sigs[0].bias == "long"
    assert sigs[0].signal_minute_et == "09:32"
    assert sigs[0].es_price_at_signal == 5012.0


def test_detect_signals_skips_when_orb_range_below_min():
    bars = (
        [_bar(f"08:{m:02d}", 5000, 5001, 4999.5, 5000) for m in range(15)]
        + [_bar("09:30", 5000, 5010, 4995, 5008)]  # would-be breakout but range too small
    )
    sigs = detect_signals_for_day(
        bars, date(2024, 3, 20), "08:00", "08:15", "09:30", "15:50",
        retest_mode="breakout_level", midline_tol_pts=1.0,
        min_range_pts=2.0, max_range_pts=30.0,
    )
    assert sigs == []


def test_detect_signals_breakout_level_short():
    bars = (
        [_bar(f"08:{m:02d}", 5005, 5012 if m == 7 else 5008, 4998 if m == 3 else 5005, 5005) for m in range(15)]
        + [_bar("09:30", 5005, 5008, 5004, 5006),
           _bar("09:31", 5006, 5007, 4990, 4995),  # breakout below 4998 → short bias
           _bar("09:32", 4995, 4999, 4994, 4998),  # retest of 4998 → SIGNAL
           _bar("09:33", 4998, 4998, 4985, 4990)]
    )
    sigs = detect_signals_for_day(
        bars, date(2024, 3, 20), "08:00", "08:15", "09:30", "15:50",
        retest_mode="breakout_level", midline_tol_pts=1.0,
        min_range_pts=2.0, max_range_pts=30.0,
    )
    assert len(sigs) == 1
    assert sigs[0].bias == "short"
    assert sigs[0].signal_minute_et == "09:32"
