"""
US Economic Calendar: Major market-moving event dates (2021-2026).

Data sources:
  - FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  - CPI:  https://www.bls.gov/schedule/news_release/cpi.htm
  - NFP:  https://www.bls.gov/schedule/news_release/empsit.htm
  - GDP:  https://www.bea.gov/news/schedule
  - PPI:  https://www.bls.gov/schedule/news_release/ppi.htm

Usage:
    from scripts.economic_calendar import (
        FOMC_DATES, CPI_DATES, NFP_DATES, GDP_DATES, PPI_DATES,
        ALL_EVENT_DATES, is_event_day, get_events_on_date,
    )

    import datetime
    d = datetime.date(2023, 6, 13)
    print(is_event_day(d))          # True
    print(get_events_on_date(d))    # ['CPI', 'FOMC']
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# FOMC announcement dates (second day of each two-day meeting = statement day)
# These are the dates the rate decision is announced at 2:00 PM ET.
# ---------------------------------------------------------------------------
FOMC_DATES: List[datetime.date] = [
    # 2021
    datetime.date(2021, 1, 27),
    datetime.date(2021, 3, 17),
    datetime.date(2021, 4, 28),
    datetime.date(2021, 6, 16),
    datetime.date(2021, 7, 28),
    datetime.date(2021, 9, 22),
    datetime.date(2021, 11, 3),
    datetime.date(2021, 12, 15),
    # 2022
    datetime.date(2022, 1, 26),
    datetime.date(2022, 3, 16),
    datetime.date(2022, 5, 4),
    datetime.date(2022, 6, 15),
    datetime.date(2022, 7, 27),
    datetime.date(2022, 9, 21),
    datetime.date(2022, 11, 2),
    datetime.date(2022, 12, 14),
    # 2023
    datetime.date(2023, 2, 1),
    datetime.date(2023, 3, 22),
    datetime.date(2023, 5, 3),
    datetime.date(2023, 6, 14),
    datetime.date(2023, 7, 26),
    datetime.date(2023, 9, 20),
    datetime.date(2023, 11, 1),
    datetime.date(2023, 12, 13),
    # 2024
    datetime.date(2024, 1, 31),
    datetime.date(2024, 3, 20),
    datetime.date(2024, 5, 1),
    datetime.date(2024, 6, 12),
    datetime.date(2024, 7, 31),
    datetime.date(2024, 9, 18),
    datetime.date(2024, 11, 7),
    datetime.date(2024, 12, 18),
    # 2025
    datetime.date(2025, 1, 29),
    datetime.date(2025, 3, 19),
    datetime.date(2025, 5, 7),
    datetime.date(2025, 6, 18),
    datetime.date(2025, 7, 30),
    datetime.date(2025, 9, 17),
    datetime.date(2025, 10, 29),
    datetime.date(2025, 12, 10),
    # 2026
    datetime.date(2026, 1, 28),
    datetime.date(2026, 3, 18),
    datetime.date(2026, 4, 29),
    datetime.date(2026, 6, 17),
    datetime.date(2026, 7, 29),
    datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28),
    datetime.date(2026, 12, 9),
]

# ---------------------------------------------------------------------------
# CPI release dates (8:30 AM ET). The release covers the prior month's data.
# ---------------------------------------------------------------------------
CPI_DATES: List[datetime.date] = [
    # 2021 (release dates for Dec 2020 through Nov 2021 data)
    datetime.date(2021, 1, 13),   # Dec 2020 data
    datetime.date(2021, 2, 10),   # Jan 2021
    datetime.date(2021, 3, 10),   # Feb 2021
    datetime.date(2021, 4, 13),   # Mar 2021
    datetime.date(2021, 5, 12),   # Apr 2021
    datetime.date(2021, 6, 10),   # May 2021
    datetime.date(2021, 7, 13),   # Jun 2021
    datetime.date(2021, 8, 11),   # Jul 2021
    datetime.date(2021, 9, 14),   # Aug 2021
    datetime.date(2021, 10, 13),  # Sep 2021
    datetime.date(2021, 11, 10),  # Oct 2021
    datetime.date(2021, 12, 10),  # Nov 2021
    # 2022 (release dates for Dec 2021 through Nov 2022 data)
    datetime.date(2022, 1, 12),   # Dec 2021
    datetime.date(2022, 2, 10),   # Jan 2022
    datetime.date(2022, 3, 10),   # Feb 2022
    datetime.date(2022, 4, 12),   # Mar 2022
    datetime.date(2022, 5, 11),   # Apr 2022
    datetime.date(2022, 6, 10),   # May 2022
    datetime.date(2022, 7, 13),   # Jun 2022
    datetime.date(2022, 8, 10),   # Jul 2022
    datetime.date(2022, 9, 13),   # Aug 2022
    datetime.date(2022, 10, 13),  # Sep 2022
    datetime.date(2022, 11, 10),  # Oct 2022
    datetime.date(2022, 12, 13),  # Nov 2022
    # 2023 (release dates for Dec 2022 through Nov 2023 data)
    datetime.date(2023, 1, 12),   # Dec 2022
    datetime.date(2023, 2, 14),   # Jan 2023
    datetime.date(2023, 3, 14),   # Feb 2023
    datetime.date(2023, 4, 12),   # Mar 2023
    datetime.date(2023, 5, 10),   # Apr 2023
    datetime.date(2023, 6, 13),   # May 2023
    datetime.date(2023, 7, 12),   # Jun 2023
    datetime.date(2023, 8, 10),   # Jul 2023
    datetime.date(2023, 9, 13),   # Aug 2023
    datetime.date(2023, 10, 12),  # Sep 2023
    datetime.date(2023, 11, 14),  # Oct 2023
    datetime.date(2023, 12, 12),  # Nov 2023
    # 2024 (release dates for Dec 2023 through Nov 2024 data)
    datetime.date(2024, 1, 11),   # Dec 2023
    datetime.date(2024, 2, 13),   # Jan 2024
    datetime.date(2024, 3, 12),   # Feb 2024
    datetime.date(2024, 4, 10),   # Mar 2024
    datetime.date(2024, 5, 15),   # Apr 2024
    datetime.date(2024, 6, 12),   # May 2024
    datetime.date(2024, 7, 11),   # Jun 2024
    datetime.date(2024, 8, 14),   # Jul 2024
    datetime.date(2024, 9, 11),   # Aug 2024
    datetime.date(2024, 10, 10),  # Sep 2024
    datetime.date(2024, 11, 13),  # Oct 2024
    datetime.date(2024, 12, 11),  # Nov 2024
    # 2025 (release dates for Dec 2024 through Nov 2025 data)
    datetime.date(2025, 1, 15),   # Dec 2024
    datetime.date(2025, 2, 12),   # Jan 2025
    datetime.date(2025, 3, 12),   # Feb 2025
    datetime.date(2025, 4, 10),   # Mar 2025
    datetime.date(2025, 5, 13),   # Apr 2025
    datetime.date(2025, 6, 11),   # May 2025
    datetime.date(2025, 7, 15),   # Jun 2025
    datetime.date(2025, 8, 12),   # Jul 2025
    datetime.date(2025, 9, 11),   # Aug 2025
    datetime.date(2025, 10, 24),  # Sep 2025 (delayed due to govt shutdown)
    # Oct 2025 CPI was CANCELED (govt shutdown)
    datetime.date(2025, 12, 18),  # Nov 2025 (rescheduled from Dec 10)
    # 2026 (release dates for Dec 2025 through Nov 2026 data)
    datetime.date(2026, 1, 13),   # Dec 2025
    datetime.date(2026, 2, 13),   # Jan 2026 (rescheduled from Feb 11)
    datetime.date(2026, 3, 11),   # Feb 2026
    datetime.date(2026, 4, 10),   # Mar 2026
    datetime.date(2026, 5, 12),   # Apr 2026
    datetime.date(2026, 6, 10),   # May 2026
    datetime.date(2026, 7, 14),   # Jun 2026
    datetime.date(2026, 8, 12),   # Jul 2026
    datetime.date(2026, 9, 11),   # Aug 2026
    datetime.date(2026, 10, 14),  # Sep 2026
    datetime.date(2026, 11, 10),  # Oct 2026
    datetime.date(2026, 12, 10),  # Nov 2026
]

# ---------------------------------------------------------------------------
# NFP (Nonfarm Payrolls / Employment Situation) release dates (8:30 AM ET).
# Typically first Friday of the month. Some exceptions noted.
# ---------------------------------------------------------------------------
NFP_DATES: List[datetime.date] = [
    # 2021
    datetime.date(2021, 1, 8),
    datetime.date(2021, 2, 5),
    datetime.date(2021, 3, 5),
    datetime.date(2021, 4, 2),
    datetime.date(2021, 5, 7),
    datetime.date(2021, 6, 4),
    datetime.date(2021, 7, 2),
    datetime.date(2021, 8, 6),
    datetime.date(2021, 9, 3),
    datetime.date(2021, 10, 8),
    datetime.date(2021, 11, 5),
    datetime.date(2021, 12, 3),
    # 2022
    datetime.date(2022, 1, 7),
    datetime.date(2022, 2, 4),
    datetime.date(2022, 3, 4),
    datetime.date(2022, 4, 1),
    datetime.date(2022, 5, 6),
    datetime.date(2022, 6, 3),
    datetime.date(2022, 7, 8),
    datetime.date(2022, 8, 5),
    datetime.date(2022, 9, 2),
    datetime.date(2022, 10, 7),
    datetime.date(2022, 11, 4),
    datetime.date(2022, 12, 2),
    # 2023
    datetime.date(2023, 1, 6),
    datetime.date(2023, 2, 3),
    datetime.date(2023, 3, 10),
    datetime.date(2023, 4, 7),
    datetime.date(2023, 5, 5),
    datetime.date(2023, 6, 2),
    datetime.date(2023, 7, 7),
    datetime.date(2023, 8, 4),
    datetime.date(2023, 9, 1),
    datetime.date(2023, 10, 6),
    datetime.date(2023, 11, 3),
    datetime.date(2023, 12, 8),
    # 2024
    datetime.date(2024, 1, 5),
    datetime.date(2024, 2, 2),
    datetime.date(2024, 3, 8),
    datetime.date(2024, 4, 5),
    datetime.date(2024, 5, 3),
    datetime.date(2024, 6, 7),
    datetime.date(2024, 7, 5),
    datetime.date(2024, 8, 2),
    datetime.date(2024, 9, 6),
    datetime.date(2024, 10, 4),
    datetime.date(2024, 11, 1),
    datetime.date(2024, 12, 6),
    # 2025
    datetime.date(2025, 1, 10),
    datetime.date(2025, 2, 7),
    datetime.date(2025, 3, 7),
    datetime.date(2025, 4, 4),
    datetime.date(2025, 5, 2),
    datetime.date(2025, 6, 6),
    datetime.date(2025, 7, 3),
    datetime.date(2025, 8, 1),
    datetime.date(2025, 9, 5),
    # Oct 2025: Sept report delayed to Nov 20 due to govt shutdown
    datetime.date(2025, 11, 20),  # Sep 2025 data (delayed)
    # Oct+Nov 2025 data released together
    datetime.date(2025, 12, 16),  # Oct+Nov 2025 data
    # 2026
    datetime.date(2026, 1, 9),
    datetime.date(2026, 2, 6),
    datetime.date(2026, 3, 6),
    datetime.date(2026, 4, 3),
    datetime.date(2026, 5, 8),
    datetime.date(2026, 6, 5),
    datetime.date(2026, 7, 2),
    datetime.date(2026, 8, 7),
    datetime.date(2026, 9, 4),
    datetime.date(2026, 10, 2),
    datetime.date(2026, 11, 6),
    datetime.date(2026, 12, 4),
]

# ---------------------------------------------------------------------------
# GDP advance/preliminary/final release dates (8:30 AM ET).
# BEA releases ~end of month. These are the "advance" (first) estimates
# which are most market-moving, plus second/third estimates.
# ---------------------------------------------------------------------------
GDP_DATES: List[datetime.date] = [
    # 2021 - advance estimates (most impactful)
    datetime.date(2021, 1, 28),   # Q4 2020 advance
    datetime.date(2021, 4, 29),   # Q1 2021 advance
    datetime.date(2021, 7, 29),   # Q2 2021 advance
    datetime.date(2021, 10, 28),  # Q3 2021 advance
    # 2022
    datetime.date(2022, 1, 27),   # Q4 2021 advance
    datetime.date(2022, 4, 28),   # Q1 2022 advance
    datetime.date(2022, 7, 28),   # Q2 2022 advance
    datetime.date(2022, 10, 27),  # Q3 2022 advance
    # 2023
    datetime.date(2023, 1, 26),   # Q4 2022 advance
    datetime.date(2023, 4, 27),   # Q1 2023 advance
    datetime.date(2023, 7, 27),   # Q2 2023 advance
    datetime.date(2023, 10, 26),  # Q3 2023 advance
    # 2024
    datetime.date(2024, 1, 25),   # Q4 2023 advance
    datetime.date(2024, 4, 25),   # Q1 2024 advance
    datetime.date(2024, 7, 25),   # Q2 2024 advance
    datetime.date(2024, 10, 30),  # Q3 2024 advance
    # 2025
    datetime.date(2025, 1, 30),   # Q4 2024 advance
    datetime.date(2025, 4, 30),   # Q1 2025 advance
    datetime.date(2025, 7, 30),   # Q2 2025 advance
    datetime.date(2025, 10, 29),  # Q3 2025 advance
    # 2026
    datetime.date(2026, 1, 29),   # Q4 2025 advance
    datetime.date(2026, 4, 29),   # Q1 2026 advance
    datetime.date(2026, 7, 29),   # Q2 2026 advance
    datetime.date(2026, 10, 28),  # Q3 2026 advance
]

# ---------------------------------------------------------------------------
# PPI (Producer Price Index) release dates (8:30 AM ET).
# Typically released 1-2 days after CPI each month.
# ---------------------------------------------------------------------------
PPI_DATES: List[datetime.date] = [
    # 2021
    datetime.date(2021, 1, 14),
    datetime.date(2021, 2, 17),
    datetime.date(2021, 3, 12),
    datetime.date(2021, 4, 9),
    datetime.date(2021, 5, 13),
    datetime.date(2021, 6, 15),
    datetime.date(2021, 7, 14),
    datetime.date(2021, 8, 12),
    datetime.date(2021, 9, 10),
    datetime.date(2021, 10, 14),
    datetime.date(2021, 11, 9),
    datetime.date(2021, 12, 14),
    # 2022
    datetime.date(2022, 1, 13),
    datetime.date(2022, 2, 15),
    datetime.date(2022, 3, 15),
    datetime.date(2022, 4, 13),
    datetime.date(2022, 5, 12),
    datetime.date(2022, 6, 14),
    datetime.date(2022, 7, 14),
    datetime.date(2022, 8, 11),
    datetime.date(2022, 9, 14),
    datetime.date(2022, 10, 12),
    datetime.date(2022, 11, 15),
    datetime.date(2022, 12, 9),
    # 2023
    datetime.date(2023, 1, 18),
    datetime.date(2023, 2, 16),
    datetime.date(2023, 3, 15),
    datetime.date(2023, 4, 13),
    datetime.date(2023, 5, 11),
    datetime.date(2023, 6, 14),
    datetime.date(2023, 7, 13),
    datetime.date(2023, 8, 11),
    datetime.date(2023, 9, 14),
    datetime.date(2023, 10, 11),
    datetime.date(2023, 11, 15),
    datetime.date(2023, 12, 13),
    # 2024
    datetime.date(2024, 1, 12),
    datetime.date(2024, 2, 16),
    datetime.date(2024, 3, 14),
    datetime.date(2024, 4, 11),
    datetime.date(2024, 5, 14),
    datetime.date(2024, 6, 13),
    datetime.date(2024, 7, 12),
    datetime.date(2024, 8, 13),
    datetime.date(2024, 9, 12),
    datetime.date(2024, 10, 11),
    datetime.date(2024, 11, 14),
    datetime.date(2024, 12, 12),
    # 2025
    datetime.date(2025, 1, 14),
    datetime.date(2025, 2, 13),
    datetime.date(2025, 3, 13),
    datetime.date(2025, 4, 11),
    datetime.date(2025, 5, 15),
    datetime.date(2025, 6, 12),
    datetime.date(2025, 7, 15),
    datetime.date(2025, 8, 14),
    datetime.date(2025, 9, 12),
    datetime.date(2025, 10, 24),  # delayed (govt shutdown)
    datetime.date(2025, 12, 18),  # rescheduled
    # 2026
    datetime.date(2026, 1, 14),
    datetime.date(2026, 2, 13),
    datetime.date(2026, 3, 12),
    datetime.date(2026, 4, 14),
    datetime.date(2026, 5, 13),
    datetime.date(2026, 6, 11),
    datetime.date(2026, 7, 15),
    datetime.date(2026, 8, 13),
    datetime.date(2026, 9, 15),
    datetime.date(2026, 10, 15),
    datetime.date(2026, 11, 12),
    datetime.date(2026, 12, 11),
]

# ---------------------------------------------------------------------------
# Convenience: combined set and lookup helpers
# ---------------------------------------------------------------------------

# Map: event_type -> set of dates for O(1) lookups
_EVENT_MAP: Dict[str, Set[datetime.date]] = {
    "FOMC": set(FOMC_DATES),
    "CPI": set(CPI_DATES),
    "NFP": set(NFP_DATES),
    "GDP": set(GDP_DATES),
    "PPI": set(PPI_DATES),
}

# Union of all event dates
ALL_EVENT_DATES: Set[datetime.date] = set()
for _dates in _EVENT_MAP.values():
    ALL_EVENT_DATES |= _dates


def is_event_day(d: datetime.date) -> bool:
    """Return True if d is a major economic event day."""
    return d in ALL_EVENT_DATES


def get_events_on_date(d: datetime.date) -> List[str]:
    """Return list of event types occurring on date d."""
    return [name for name, dates in _EVENT_MAP.items() if d in dates]


def get_events_in_range(
    start: datetime.date, end: datetime.date
) -> Dict[datetime.date, List[str]]:
    """Return {date: [event_types]} for all event days in [start, end]."""
    result: Dict[datetime.date, List[str]] = {}
    for d in sorted(ALL_EVENT_DATES):
        if start <= d <= end:
            result[d] = get_events_on_date(d)
    return result


def get_nearby_events(
    d: datetime.date, window_days: int = 1
) -> Dict[datetime.date, List[str]]:
    """Return events within +/- window_days of date d."""
    start = d - datetime.timedelta(days=window_days)
    end = d + datetime.timedelta(days=window_days)
    return get_events_in_range(start, end)


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"FOMC dates: {len(FOMC_DATES)}")
    print(f"CPI dates:  {len(CPI_DATES)}")
    print(f"NFP dates:  {len(NFP_DATES)}")
    print(f"GDP dates:  {len(GDP_DATES)}")
    print(f"PPI dates:  {len(PPI_DATES)}")
    print(f"Total unique event dates: {len(ALL_EVENT_DATES)}")
    print()

    # Example: show all events in March 2026
    march_events = get_events_in_range(
        datetime.date(2026, 3, 1), datetime.date(2026, 3, 31)
    )
    print("March 2026 events:")
    for d, events in sorted(march_events.items()):
        print(f"  {d.isoformat()} ({d.strftime('%A')}): {', '.join(events)}")
