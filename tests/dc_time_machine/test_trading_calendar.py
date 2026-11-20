from datetime import date

from zerodte.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after,
    sessions_between,
    trading_days_after,
)


def test_sessions_between_skips_weekends():
    s = sessions_between(date(2026, 2, 3), date(2026, 2, 13))
    assert s[0] == date(2026, 2, 3)
    assert date(2026, 2, 7) not in s   # Saturday
    assert date(2026, 2, 8) not in s   # Sunday
    assert date(2026, 2, 11) in s


def test_good_friday_excluded():
    s = sessions_between(date(2022, 4, 14), date(2022, 4, 18))
    assert date(2022, 4, 15) not in s   # Good Friday holiday


def test_nth_trading_day_after():
    # Mon 2026-02-02: 1st session after = 02-03; 9th = 02-13 (02-16 Presidents Day skipped later)
    assert nth_trading_day_after(date(2026, 2, 2), 1) == date(2026, 2, 3)
    assert nth_trading_day_after(date(2026, 2, 2), 9) == date(2026, 2, 13)
    assert nth_trading_day_after(date(2026, 2, 2), 10) == date(2026, 2, 17)  # skips weekend + Presidents Day


def test_trading_days_after_counts_sessions_exclusive_of_start():
    assert trading_days_after(date(2026, 2, 2), date(2026, 2, 3)) == 1
    assert trading_days_after(date(2026, 2, 2), date(2026, 2, 13)) == 9
    assert trading_days_after(date(2026, 2, 2), date(2026, 2, 2)) == 0   # same day
