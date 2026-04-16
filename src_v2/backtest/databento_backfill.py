"""Targeted Databento CBBO-1m backfill.

Fetches quotes for a narrow strike window around an ATM signal price, in a narrow
time window around the signal minute. Returns a dict in the same shape as the
existing data/cbbo_cache pickles, so the existing CbboStore can read the output
without modification.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from src_v2.backtest.osi_symbol import build_strike_window

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]


def build_request_window(
    session_date: date,
    anchor_minute_et: str,
    before_min: int,
    after_min: int,
) -> Tuple[str, str]:
    """Compute (start_iso_utc, end_iso_utc) for a window centered on anchor_minute_et."""
    hh, mm = int(anchor_minute_et[:2]), int(anchor_minute_et[3:5])
    anchor_et = datetime(session_date.year, session_date.month, session_date.day,
                         hh, mm, tzinfo=ET)
    start_et = anchor_et - timedelta(minutes=before_min)
    end_et = anchor_et + timedelta(minutes=after_min)
    start_utc = start_et.astimezone(UTC)
    end_utc = end_et.astimezone(UTC)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return start_utc.strftime(fmt), end_utc.strftime(fmt)


def _parse_symbol(raw: str) -> Optional[Tuple[date, bool, float]]:
    """'SPXW  250922P06445000' → (date(2025,9,22), False, 6445.0). None if malformed."""
    raw = raw.rstrip()
    if len(raw) < 21 or not raw.startswith("SPXW"):
        return None
    body = raw[6:]  # strip 'SPXW  '
    yy, mm, dd = int(body[0:2]), int(body[2:4]), int(body[4:6])
    cp = body[6]
    strike = int(body[7:15]) / 1000.0
    return date(2000 + yy, mm, dd), cp == "C", strike


def fetch_day_quotes(
    client: Any,
    session_date: date,
    anchor_minute_et: str,
    atm_strike: float,
    half_width_pts: int = 50,
    step_pts: int = 5,
    before_min: int = 2,
    after_min: int = 10,
) -> Dict[QuoteKey, QuoteVal]:
    """One Databento call → {(minute_et, strike, is_call): (bid, ask)} dict.

    `client` should be a databento.Historical() instance (or a mock with the same
    interface).
    """
    symbols = build_strike_window(
        expiry=session_date,
        atm_strike=atm_strike,
        half_width_pts=half_width_pts,
        step_pts=step_pts,
    )
    start_iso, end_iso = build_request_window(
        session_date, anchor_minute_et, before_min, after_min
    )

    response = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        schema="cbbo-1m",
        symbols=symbols,
        stype_in="raw_symbol",
        start=start_iso,
        end=end_iso,
    )
    df = response.to_df()

    quotes: Dict[QuoteKey, QuoteVal] = {}
    if df.empty:
        return quotes

    for ts_utc, row in df.iterrows():
        parsed = _parse_symbol(str(row["symbol"]))
        if parsed is None:
            continue
        _expiry, is_call, strike = parsed
        ts_et = ts_utc.tz_convert(ET)
        minute_str = f"{ts_et.hour:02d}:{ts_et.minute:02d}"
        bid = float(row.get("bid_px_00", 0.0) or 0.0)
        ask = float(row.get("ask_px_00", 0.0) or 0.0)
        if bid <= 0 and ask <= 0:
            continue
        quotes[(minute_str, strike, is_call)] = (bid, ask)

    return quotes
