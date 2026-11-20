"""Tests for zerodte.backtest.data_loader."""

import csv
from datetime import date
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


@pytest.fixture
def spy_csv(tmp_path):
    """Create a minimal SPY 1-min CSV with 2 days of data."""
    csv_path = tmp_path / "spy_1m.csv"
    rows = [
        {"ts_event": "2025-10-01T12:00:00.000000000Z", "symbol": "SPY",
         "open": "570.00", "high": "570.50", "low": "569.50", "close": "570.25", "volume": "1000"},
        {"ts_event": "2025-10-01T12:01:00.000000000Z", "symbol": "SPY",
         "open": "570.25", "high": "570.75", "low": "570.00", "close": "570.50", "volume": "1200"},
        {"ts_event": "2025-10-02T12:00:00.000000000Z", "symbol": "SPY",
         "open": "571.00", "high": "571.50", "low": "570.50", "close": "571.25", "volume": "800"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path)


def test_daybars_repr():
    from zerodte.backtest.data_loader import DayBars
    db = DayBars(date(2025, 10, 1), "SPX", [{"close": 1}])
    assert "2025-10-01" in repr(db)
    assert "1 bars" in repr(db)


def test_bar_loader_iter_days(spy_csv):
    from zerodte.backtest.data_loader import BarLoader
    loader = BarLoader(spy_csv, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days())
    assert len(days) == 2
    d1 = days[0]
    assert d1.session_date == date(2025, 10, 1)
    assert d1.symbol == "SPX"
    assert len(d1.bars) == 2
    assert d1.bars[0]["open"] == pytest.approx(5700.0)
    assert d1.bars[0]["close"] == pytest.approx(5702.5)
    assert d1.bars[0]["ts_et"].tzinfo is not None
    assert d1.bars[0]["ts_et"].hour == 8  # 12:00 UTC = 08:00 ET (Oct, EDT)


def test_bar_loader_date_filter(spy_csv):
    from zerodte.backtest.data_loader import BarLoader
    loader = BarLoader(spy_csv, price_scale=10.0)
    days = list(loader.iter_days(start_date=date(2025, 10, 2)))
    assert len(days) == 1
    assert days[0].session_date == date(2025, 10, 2)


def test_bar_loader_missing_file():
    from zerodte.backtest.data_loader import BarLoader
    with pytest.raises(FileNotFoundError):
        BarLoader("/nonexistent/path.csv")
