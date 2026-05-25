"""Trading-day guard for the live scheduler (skip weekends + market holidays)."""
from datetime import date

from src_v2.agent.scheduled_engine import is_trading_day


def test_weekend_is_not_trading_day():
    assert is_trading_day(date(2026, 5, 23)) is False  # Saturday
    assert is_trading_day(date(2026, 5, 24)) is False  # Sunday


def test_regular_weekday_is_trading_day():
    assert is_trading_day(date(2026, 5, 22)) is True   # Friday


def test_market_holiday_is_not_trading_day():
    assert is_trading_day(date(2026, 5, 25)) is False   # Memorial Day (Mon)
    assert is_trading_day(date(2026, 1, 1)) is False    # New Year's Day
    assert is_trading_day(date(2026, 12, 25)) is False  # Christmas


def test_weekday_short_circuits_without_calendar(monkeypatch):
    # If the calendar import/lookup fails, a weekday degrades to "trading day"
    # (the anomaly guard still backstops actual holidays) and never crashes.
    import src_v2.agent.scheduled_engine as se

    def _boom(*a, **k):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(se, "_nyse_calendar", _boom)
    assert is_trading_day(date(2026, 5, 22)) is True   # weekday → True
    assert is_trading_day(date(2026, 5, 24)) is False  # weekend short-circuit still works
