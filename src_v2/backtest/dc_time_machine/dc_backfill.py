# src_v2/backtest/dc_time_machine/dc_backfill.py
"""Databento backfill for the DC backtest: full-day cbbo-1m for a SPECIFIC expiry.

Generalizes src_v2/backtest/databento_backfill.fetch_day_quotes by decoupling the
fetched `expiry_date` from the trade `session_date`, and widening the time window
to a full session. Writes one pickle per session: {expiry: {(min,strike,call):(bid,ask)}}.
"""
from __future__ import annotations

import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src_v2.backtest.osi_symbol import build_strike_window, parse_osi

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]


def _window_utc(session_date: date, start_et: str, end_et: str) -> Tuple[str, str]:
    def to_utc(hhmm: str) -> str:
        hh, mm = int(hhmm[:2]), int(hhmm[3:5])
        dt = datetime(session_date.year, session_date.month, session_date.day, hh, mm, tzinfo=ET)
        return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return to_utc(start_et), to_utc(end_et)


def fetch_expiry_session(client: Any, session_date: date, expiry_date: date,
                         atm_strike: float, half_width_pts: int = 80, step_pts: int = 5,
                         start_et: str = "09:30", end_et: str = "16:00") -> Dict[QuoteKey, QuoteVal]:
    """One Databento call → {(minute_et, strike, is_call): (bid, ask)} for `expiry_date`."""
    symbols = build_strike_window(expiry=expiry_date, atm_strike=atm_strike,
                                  half_width_pts=half_width_pts, step_pts=step_pts)
    start_iso, end_iso = _window_utc(session_date, start_et, end_et)
    resp = client.timeseries.get_range(
        dataset="OPRA.PILLAR", schema="cbbo-1m", symbols=symbols,
        stype_in="raw_symbol", start=start_iso, end=end_iso,
    )
    df = resp.to_df()
    out: Dict[QuoteKey, QuoteVal] = {}
    if df is None or df.empty:
        return out
    for ts_utc, row in df.iterrows():
        sym = str(row["symbol"]).rstrip()
        try:
            strike, is_call = parse_osi(f"{sym:<21}")
        except ValueError:
            continue
        bid_raw, ask_raw = row.get("bid_px_00"), row.get("ask_px_00")
        if bid_raw is None or ask_raw is None or pd.isna(bid_raw) or pd.isna(ask_raw):
            continue
        bid, ask = float(bid_raw), float(ask_raw)
        if bid <= 0 and ask <= 0:
            continue
        ts_et = ts_utc.tz_convert(ET)
        minute = f"{ts_et.hour:02d}:{ts_et.minute:02d}"
        out[(minute, strike, is_call)] = (bid, ask)
    return out


def backfill_session(client: Any, session_date: date, expiries: Iterable[date],
                     atm_strike: float, out_dir: str | Path,
                     half_width_pts: int = 80, step_pts: int = 5) -> Path:
    """Fetch each expiry's full-day panel and write {expiry: quotemap} to <out_dir>/<session>.pkl."""
    panel: Dict[date, Dict[QuoteKey, QuoteVal]] = {}
    for exp in expiries:
        panel[exp] = fetch_expiry_session(client, session_date, exp, atm_strike,
                                          half_width_pts=half_width_pts, step_pts=step_pts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session_date.isoformat()}.pkl"
    with open(path, "wb") as f:
        pickle.dump(panel, f)
    return path
