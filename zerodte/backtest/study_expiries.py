# zerodte/backtest/study_expiries.py
"""Select the study's GEX expiries: the near-dated dailies (which carry the 0DTE-like
day-trader gamma, incl. 1DTE/2DTE for the evolution test) plus the standard monthly
third-Fridays and month-end sessions out to max_dte_days. Bounds symbol count/cost
while keeping the dealer-gamma-relevant OI."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Sequence


def third_fridays(start: date, end: date) -> List[date]:
    out: List[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        fridays = [d for d in (date(y, m, day) for day in range(15, 22))
                   if d.weekday() == 4]
        tf = fridays[0]
        if start <= tf <= end:
            out.append(tf)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def month_end_sessions(calendar: Sequence[date], start: date, end: date) -> List[date]:
    by_month: dict = {}
    for d in calendar:
        if start <= d <= end:
            by_month[(d.year, d.month)] = d  # calendar is sorted -> last wins
    return sorted(by_month.values())


def study_expiries(session_date: date, calendar: Sequence[date],
                   n_dailies: int = 5, max_dte_days: int = 120) -> List[date]:
    cal = [d for d in sorted(calendar) if d >= session_date]
    end = session_date + timedelta(days=max_dte_days)
    near = cal[:n_dailies]
    cal_set = set(cal)
    monthlies = [d for d in third_fridays(session_date, end) if d in cal_set]
    eom = [d for d in month_end_sessions(cal, session_date, end)]
    picked = {d for d in (near + monthlies + eom) if session_date <= d <= end}
    return sorted(picked)
