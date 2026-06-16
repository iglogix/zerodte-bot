"""Pure helpers for the targeted 0DTE MEIC CBBO backfill.

Kept separate from the script driver so the symbol/window/quote-parsing logic
is unit-testable without a live Databento client. Unlike the ES-anchored
`databento_backfill`, the ATM here comes DIRECTLY from the (SPY-derived) SPX
spot range of the day — no ES->SPX basis indirection — so it works for months
that have no existing CBBO cache to bootstrap a basis from.
"""
from __future__ import annotations

from datetime import date, datetime as _dt
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src_v2.backtest.osi_symbol import format_osi
from src_v2.backtest.databento_backfill import _parse_symbol, build_request_window

ET = ZoneInfo("America/New_York")

QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]


def _hhmm(s: str) -> Tuple[int, int]:
    return int(s[:2]), int(s[3:5])


def spot_range_from_bars(
    bars: List[Dict[str, Any]], start_et: str = "11:00", end_et: str = "15:50",
) -> Optional[Tuple[float, float]]:
    """(min low, max high) over bars whose ET time is within [start_et, end_et].

    Returns None if no bar falls in the window. Used to size the strike window
    so it covers the whole MEIC monitoring session even on big-move days.
    """
    s = _hhmm(start_et)
    e = _hhmm(end_et)
    lo: Optional[float] = None
    hi: Optional[float] = None
    for b in bars:
        t = b["ts_et"].time()
        tm = (t.hour, t.minute)
        if tm < s or tm > e:
            continue
        bl, bh = b["low"], b["high"]
        lo = bl if lo is None else min(lo, bl)
        hi = bh if hi is None else max(hi, bh)
    if lo is None or hi is None:
        return None
    return lo, hi


def meic_day_symbols(
    session_date: date,
    spot_low: float,
    spot_high: float,
    buffer_pts: int = 300,
    step_pts: int = 5,
    root: str = "SPXW",
) -> List[str]:
    """OSI symbols for 0DTE SPXW spanning [spot_low-buffer, spot_high+buffer].

    Both calls and puts at every step. The buffer must cover the ~10-delta
    short strikes (well OTM) plus the 80-100pt long wings beyond them.
    """
    base_lo = int((spot_low - buffer_pts) // step_pts) * step_pts
    base_hi = int(-(-(spot_high + buffer_pts) // step_pts)) * step_pts  # ceil
    out: List[str] = []
    k = base_lo
    while k <= base_hi:
        out.append(format_osi(session_date, float(k), is_call=False, root=root))
        out.append(format_osi(session_date, float(k), is_call=True, root=root))
        k += step_pts
    return out


def multi_expiry_symbols(expiries, spot_low, spot_high, buffer_pts=150,
                         buffer_down_pts=None, buffer_up_pts=None, step_pts=5,
                         root="SPXW"):
    """OSI symbols for a list of expiries over [spot_low-down, spot_high+up].

    Defaults to a symmetric buffer_pts; pass buffer_down_pts/buffer_up_pts for an
    asymmetric (wider-downside) window so the put-driven gamma flip — which sits well
    below spot — is never truncated. Truncation must not be vol-correlated, so over-fetch
    generously rather than tightly fit (statistics schema is cheap)."""
    down = buffer_down_pts if buffer_down_pts is not None else buffer_pts
    up = buffer_up_pts if buffer_up_pts is not None else buffer_pts
    base_lo = int((spot_low - down) // step_pts) * step_pts
    base_hi = int(-(-(spot_high + up) // step_pts)) * step_pts
    out = []
    for expiry in expiries:
        k = base_lo
        while k <= base_hi:
            out.append(format_osi(expiry, float(k), is_call=False, root=root))
            out.append(format_osi(expiry, float(k), is_call=True, root=root))
            k += step_pts
    return out


def parse_osi_multi(symbol):
    """'SPXW  240119P04700000' -> (date(2024,1,19), 4700.0, False); None on bad input."""
    s = symbol.strip()
    try:
        exp = _dt.strptime(s[-15:-9], "%y%m%d").date()
        is_call = s[-9] == "C"
        strike = int(s[-8:]) / 1000.0
        return exp, strike, is_call
    except (ValueError, IndexError):
        return None


def meic_window_utc(
    session_date: date, start_et: str = "11:00", end_et: str = "15:50",
) -> Tuple[str, str]:
    """(start_iso_utc, end_iso_utc) covering the MEIC monitoring window, DST-correct."""
    s_h, s_m = _hhmm(start_et)
    e_h, e_m = _hhmm(end_et)
    after_min = (e_h * 60 + e_m) - (s_h * 60 + s_m)
    return build_request_window(session_date, start_et, 0, after_min)


def quotes_from_df(df: Any) -> Dict[QuoteKey, QuoteVal]:
    """Parse a Databento cbbo-1m to_df() frame -> {(minute, strike, is_call): (bid, ask)}.

    Drops rows with NaN/None bid or ask, and rows where both sides are <= 0.
    """
    quotes: Dict[QuoteKey, QuoteVal] = {}
    if df is None or getattr(df, "empty", True):
        return quotes
    for ts_utc, row in df.iterrows():
        parsed = _parse_symbol(str(row["symbol"]))
        if parsed is None:
            continue
        _expiry, is_call, strike = parsed
        ts_et = ts_utc.tz_convert(ET)
        minute_str = f"{ts_et.hour:02d}:{ts_et.minute:02d}"
        bid_raw = row.get("bid_px_00")
        ask_raw = row.get("ask_px_00")
        if bid_raw is None or ask_raw is None or pd.isna(bid_raw) or pd.isna(ask_raw):
            continue
        bid = float(bid_raw)
        ask = float(ask_raw)
        if bid <= 0 and ask <= 0:
            continue
        quotes[(minute_str, strike, is_call)] = (bid, ask)
    return quotes
