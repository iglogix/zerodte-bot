"""
ES Futures contract utilities — automatic front-month symbol detection.

ES contract months: H=March, M=June, U=September, Z=December
Expiration: 3rd Friday of contract month at 9:30 AM ET (CME)
Roll: Typically 8 trading days before expiration, volume shifts to next contract
"""

from datetime import date, timedelta
from typing import Tuple

# ES quarterly contract months
CONTRACT_MONTHS = {
    3: "H",   # March
    6: "M",   # June
    9: "U",   # September
    12: "Z",  # December
}

# Reverse mapping
MONTH_TO_CODE = {v: k for k, v in CONTRACT_MONTHS.items()}


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month."""
    # First day of month
    first = date(year, month, 1)
    # Days until first Friday (0=Monday, 4=Friday)
    days_until_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_until_friday)
    # Third Friday is 14 days after first Friday
    return first_friday + timedelta(days=14)


def _get_contract_months_for_year(year: int) -> list[Tuple[int, int, date]]:
    """Return list of (year, month, expiration_date) for all ES contracts in a year."""
    return [(year, m, _third_friday(year, m)) for m in sorted(CONTRACT_MONTHS.keys())]


def get_front_month_contract(
    reference_date: date | None = None,
    roll_days_before_expiry: int = 0,
) -> Tuple[str, date]:
    """
    Determine the front-month ES contract symbol based on reference date.

    Args:
        reference_date: Date to calculate from (defaults to today)
        roll_days_before_expiry: Days before expiration to roll to next contract.
                                  Default 0: stay on current contract until expiry
                                  (0DTE options are listed until expiration day).

    Returns:
        Tuple of (contract_code, expiration_date)
        Contract code format: "ESH26" for March 2026

    Example:
        >>> get_front_month_contract(date(2026, 3, 1))
        ('ESH26', date(2026, 3, 20))
        >>> get_front_month_contract(date(2026, 3, 20))  # Expiration day, stays on H
        ('ESH26', date(2026, 3, 20))
        >>> get_front_month_contract(date(2026, 3, 21))  # Day after expiry, rolls to June
        ('ESM26', date(2026, 6, 19))
    """
    if reference_date is None:
        reference_date = date.today()

    # Build list of upcoming contract expirations (this year + next year)
    year = reference_date.year
    contracts = []
    contracts.extend(_get_contract_months_for_year(year))
    contracts.extend(_get_contract_months_for_year(year + 1))

    # Calculate roll date (calendar days approximation for trading days)
    # 8 trading days ≈ 11 calendar days (accounting for weekends)
    roll_calendar_days = int(roll_days_before_expiry * 1.4)

    for yr, month, expiry in contracts:
        roll_date = expiry - timedelta(days=roll_calendar_days)

        # If we're on or before the roll date, this is the front month.
        # 0DTE options trade through expiration day, so stay on current contract.
        if reference_date <= roll_date:
            month_code = CONTRACT_MONTHS[month]
            year_suffix = str(yr)[-2:]  # Last 2 digits: 2026 -> "26"
            return f"ES{month_code}{year_suffix}", expiry

    # Fallback to last contract in list (shouldn't happen)
    yr, month, expiry = contracts[-1]
    month_code = CONTRACT_MONTHS[month]
    year_suffix = str(yr)[-2:]
    return f"ES{month_code}{year_suffix}", expiry


def get_dxlink_symbol(
    reference_date: date | None = None,
    roll_days_before_expiry: int = 0,
) -> str:
    """
    Return the DXLink symbol for the front-month ES contract.

    Format: /ES{month}{year}:XCME
    Example: /ESH26:XCME for March 2026

    roll_days_before_expiry: roll to the next cycle this many trading days
        before expiry. Default 0 keeps the contract through expiry day (correct
        when trading ES options that settle into it). Use a small positive value
        when ES is only a price/signal source — the expiring front month's quotes
        go silent as liquidity rolls to the next cycle.
    """
    contract, _ = get_front_month_contract(reference_date, roll_days_before_expiry)
    return f"/{contract}:XCME"


def get_contract_info(reference_date: date | None = None) -> dict:
    """
    Return detailed info about the current front-month contract.

    Returns:
        {
            "symbol": "ESH26",
            "dxlink_symbol": "/ESH26:XCME",
            "month_code": "H",
            "month_name": "March",
            "expiration": date(2026, 3, 20),
            "days_to_expiry": 11,
        }
    """
    if reference_date is None:
        reference_date = date.today()

    contract, expiry = get_front_month_contract(reference_date)
    month_code = contract[2]  # "ESH26" -> "H"
    month_num = MONTH_TO_CODE[month_code]

    month_names = {3: "March", 6: "June", 9: "September", 12: "December"}

    return {
        "symbol": contract,
        "dxlink_symbol": f"/{contract}:XCME",
        "month_code": month_code,
        "month_name": month_names[month_num],
        "expiration": expiry,
        "days_to_expiry": (expiry - reference_date).days,
    }


if __name__ == "__main__":
    # Quick test
    test_dates = [
        date(2026, 1, 15),   # Should be H (March)
        date(2026, 3, 1),    # Should be H (March)
        date(2026, 3, 10),   # Should roll to M (June) - close to 3rd Friday
        date(2026, 3, 21),   # After H expiry, should be M (June)
        date(2026, 6, 1),    # Should be M (June)
        date(2026, 9, 1),    # Should be U (September)
        date(2026, 12, 1),   # Should be Z (December)
    ]

    for d in test_dates:
        info = get_contract_info(d)
        sym = info["dxlink_symbol"]
        exp = info["expiration"]
        days = info["days_to_expiry"]
        print(f"{d}: {sym} (expires {exp}, {days} days)")
