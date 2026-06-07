"""ThetaData (Theta Terminal) backfill for the DC backtest.

Uses the local Theta Terminal bulk endpoint to fetch a full-day 1-min NBBO panel
for ALL strikes of one (session, expiry), then filters to the ATM strike window.
Output shape matches dc_backfill.backfill_session exactly: one pickle per session,
{expiry: {(minute, strike, is_call): (bid, ask)}}, so MultiExpiryStore reads it
unchanged.

Requires the OPTION.VALUE entitlement (intraday). Flat-rate — no per-query cost.
"""
from __future__ import annotations

import json
import pickle
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

THETA = "http://127.0.0.1:25510"
QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]


def _http_json(url: str, timeout: int = 180) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def fetch_expiry_session_theta(
    session: date, expiry: date, atm_strike: float,
    half_width_pts: int = 80, base: str = THETA,
    fetch_json: Callable[[str], dict] = _http_json,
) -> Dict[QuoteKey, QuoteVal]:
    """One bulk call → {(minute_et, strike, is_call): (bid, ask)} for `expiry` on `session`."""
    lo, hi = atm_strike - half_width_pts, atm_strike + half_width_pts
    url = (f"{base}/v2/bulk_hist/option/quote?root=SPXW"
           f"&exp={expiry.strftime('%Y%m%d')}"
           f"&start_date={session.strftime('%Y%m%d')}&end_date={session.strftime('%Y%m%d')}"
           f"&ivl=60000&rth=true")
    data = fetch_json(url)
    out: Dict[QuoteKey, QuoteVal] = {}
    fmt = data.get("header", {}).get("format")
    if not fmt:
        return out
    i_ms, i_bid, i_ask = fmt.index("ms_of_day"), fmt.index("bid"), fmt.index("ask")
    for el in data.get("response", []):
        c = el.get("contract", {})
        strike = c.get("strike", 0) / 1000.0
        if strike < lo or strike > hi:
            continue
        is_call = c.get("right") == "C"
        for row in el.get("ticks", []):
            bid, ask = row[i_bid], row[i_ask]
            if bid is None or ask is None:
                continue
            bid, ask = float(bid), float(ask)
            if bid <= 0 and ask <= 0:
                continue
            m = int(row[i_ms]) // 60000
            out[(f"{m // 60:02d}:{m % 60:02d}", strike, is_call)] = (bid, ask)
    return out


def backfill_session_theta(
    session: date, expiries: Iterable[date], atm_strike: float, out_dir: str | Path,
    half_width_pts: int = 80, base: str = THETA,
    fetch_json: Callable[[str], dict] = _http_json,
) -> Path:
    """Fetch each expiry's panel and write {expiry: quotemap} to <out_dir>/<session>.pkl."""
    panel: Dict[date, Dict[QuoteKey, QuoteVal]] = {}
    for exp in expiries:
        panel[exp] = fetch_expiry_session_theta(
            session, exp, atm_strike, half_width_pts=half_width_pts,
            base=base, fetch_json=fetch_json,
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session.isoformat()}.pkl"
    with open(path, "wb") as f:
        pickle.dump(panel, f)
    return path
