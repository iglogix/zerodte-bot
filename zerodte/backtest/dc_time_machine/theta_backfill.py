"""ThetaData (Theta Terminal) backfill for the DC backtest.

Uses the local Theta Terminal bulk endpoint to fetch a full-day 1-min NBBO panel
for ALL strikes of one (session, expiry). The window is centered on the parity
forward DERIVED FROM THE DATA at the entry minute (robust to big gaps), not on an
external SPY reference, then filtered to ±half_width. Output shape matches
dc_backfill.backfill_session: one pickle per session, {expiry: {(minute, strike,
is_call): (bid, ask)}}.

Requires the OPTION.VALUE entitlement (intraday). Flat-rate — no per-query cost.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

THETA = "http://127.0.0.1:25510"
QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]


def _http_json(url: str, timeout: int = 180, retries: int = 3, backoff: float = 1.5) -> dict:
    """GET + parse JSON. Retries only TRANSIENT errors (connection/timeout/5xx); a 4xx
    such as 472 means 'no data for this contract' (e.g. a future or non-existent expiry)
    and is raised immediately so we don't waste backoff on it."""
    import urllib.error
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:      # definitive no-data → fail fast, no retry
                raise
            last = e                     # 5xx → transient
        except Exception as e:           # URLError / socket timeout / JSON
            last = e
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise last if last else RuntimeError("fetch failed")


DEFAULT_IVL_MS = 600_000   # 10-min bars: 54x faster cold-storage reads than 1-min, and
                           # ample resolution for a transform that fires over minutes-to-hours.


def _bulk_url(session: date, expiry: date, base: str, ivl: int = DEFAULT_IVL_MS) -> str:
    return (f"{base}/v2/bulk_hist/option/quote?root=SPXW"
            f"&exp={expiry.strftime('%Y%m%d')}"
            f"&start_date={session.strftime('%Y%m%d')}&end_date={session.strftime('%Y%m%d')}"
            f"&ivl={ivl}&rth=true")


def _parse_bulk(data: dict) -> Dict[QuoteKey, QuoteVal]:
    """Full-chain bulk response → {(minute, strike, is_call): (bid, ask)} (unfiltered)."""
    out: Dict[QuoteKey, QuoteVal] = {}
    fmt = data.get("header", {}).get("format")
    if not fmt:
        return out
    i_ms, i_bid, i_ask = fmt.index("ms_of_day"), fmt.index("bid"), fmt.index("ask")
    for el in data.get("response", []):
        c = el.get("contract", {})
        strike = c.get("strike", 0) / 1000.0
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


def _forward_at(panel: Dict[QuoteKey, QuoteVal], minute: str) -> Optional[float]:
    """Parity forward (strike minimising |call_mid − put_mid|) at `minute`."""
    strikes = {k[1] for k in panel if k[0] == minute}
    best = None
    for K in strikes:
        c = panel.get((minute, K, True))
        p = panel.get((minute, K, False))
        if not c or not p:
            continue
        c_mid = (c[0] + c[1]) / 2
        p_mid = (p[0] + p[1]) / 2
        diff = abs(c_mid - p_mid)
        if best is None or diff < best[0]:
            best = (diff, K + c_mid - p_mid)
    return best[1] if best else None


def _window(full: Dict[QuoteKey, QuoteVal], center: float, half: float) -> Dict[QuoteKey, QuoteVal]:
    lo, hi = center - half, center + half
    return {k: v for k, v in full.items() if lo <= k[1] <= hi}


def fetch_expiry_session_theta(
    session: date, expiry: date, atm_strike: float,
    half_width_pts: int = 200, base: str = THETA,
    fetch_json: Callable[[str], dict] = _http_json, ivl: int = DEFAULT_IVL_MS,
) -> Dict[QuoteKey, QuoteVal]:
    """One bulk call → {(minute, strike, is_call): (bid, ask)} filtered to ±half around atm_strike."""
    full = _parse_bulk(fetch_json(_bulk_url(session, expiry, base, ivl)))
    return _window(full, atm_strike, half_width_pts)


def backfill_session_theta(
    session: date, expiries: Iterable[date], atm_strike: float, out_dir: str | Path,
    half_width_pts: int = 200, ref_minute: str = "09:40", base: str = THETA,
    fetch_json: Callable[[str], dict] = _http_json, ivl: int = DEFAULT_IVL_MS,
) -> Optional[Path]:
    """Fetch each expiry full-chain, center on the data-derived forward, write the panel.

    `atm_strike` is only a fallback if the forward cannot be derived from the data.
    """
    expiries = list(expiries)

    def _safe(exp: date) -> Dict[QuoteKey, QuoteVal]:
        try:                                  # a missing expiry (e.g. a 2022 Thursday) or a
            return _parse_bulk(fetch_json(_bulk_url(session, exp, base, ivl)))   # transient error
        except Exception:                     # must not drop the whole session
            return {}

    fulls: Dict[date, Dict[QuoteKey, QuoteVal]] = {exp: _safe(exp) for exp in expiries}
    # Center: prefer the session's own 0DTE forward; else the first expiry that yields one.
    center: Optional[float] = None
    if session in fulls:
        center = _forward_at(fulls[session], ref_minute)
    if center is None:
        for exp in expiries:
            center = _forward_at(fulls[exp], ref_minute)
            if center is not None:
                break
    if center is None:
        center = atm_strike

    panel = {exp: _window(full, center, half_width_pts) for exp, full in fulls.items()}
    if not any(panel.values()):
        return None   # nothing fetched (terminal overload / all-missing) → leave for resume
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session.isoformat()}.pkl"
    tmp = out_dir / f".{session.isoformat()}.pkl.tmp"   # atomic: write tmp, then rename
    with open(tmp, "wb") as f:
        pickle.dump(panel, f)
    os.replace(tmp, path)
    return path
