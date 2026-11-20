# tests/test_study_expiries.py
from datetime import date, timedelta

from zerodte.backtest.study_expiries import study_expiries, third_fridays


def _weekday_calendar(start: date, n: int):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

def test_includes_next_n_sessions():
    cal = _weekday_calendar(date(2024, 1, 2), 200)
    exps = study_expiries(date(2024, 1, 2), cal, n_dailies=5, max_dte_days=120)
    # the entry day + next 4 sessions are present
    assert cal[0] in exps and cal[4] in exps
    # all expiries are within 120 calendar days and on the calendar
    assert all((e - date(2024, 1, 2)).days <= 120 for e in exps)
    assert all(e in set(cal) for e in exps)

def test_includes_a_third_friday():
    cal = _weekday_calendar(date(2024, 1, 2), 200)
    tf = third_fridays(date(2024, 1, 2), date(2024, 1, 2) + timedelta(days=120))
    assert date(2024, 1, 19) in tf   # Jan 2024 third Friday
    exps = study_expiries(date(2024, 1, 2), cal, n_dailies=5, max_dte_days=120)
    assert date(2024, 1, 19) in exps

def test_sorted_and_deduped():
    cal = _weekday_calendar(date(2024, 1, 2), 200)
    exps = study_expiries(date(2024, 1, 2), cal, n_dailies=5, max_dte_days=120)
    assert exps == sorted(set(exps))
