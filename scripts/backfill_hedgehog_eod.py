"""Backfill EOD multi-expiry option chains for the Hedge Hog backtest.

Fetches, via a locally running Theta Terminal (VALUE tier, EOD history
2020-01 -> present), the full EOD lifetime of every Thursday/Friday SPY
expiration and stores one compact pickle per expiration:

    data/hedgehog_eod/SPY/<YYYYMMDD>.pkl
        {"expiry": int, "days": {quote_date_int: {(strike, right): (bid, ask, volume)}}}

Strikes are in dollars (Theta's 1/1000 units divided out). bid/ask are the
NBBO snapshot at the 16:00 close. Resumable: completed expirations are
skipped on re-run.

Usage:
    .venv/bin/python scripts/backfill_hedgehog_eod.py [--root SPY] [--workers 2]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "http://127.0.0.1:25510/v2"
HISTORY_START = 20200102  # VALUE-tier EOD boundary (probed 2026-06-11)
LOOKBACK_DAYS = 130       # need quotes from ~115 DTE (hog max) + margin


def _get(url: str, timeout: int = 180) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _to_date(d: int) -> dt.date:
    return dt.date(d // 10000, d // 100 % 100, d % 100)


def _to_int(d: dt.date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def list_expirations(root: str) -> list[int]:
    data = _get(f"{BASE}/list/expirations?root={root}", timeout=60)
    if not data:
        raise RuntimeError("Theta Terminal not reachable or no expirations")
    return sorted(r[0] if isinstance(r, list) else r for r in data["response"])


def target_expirations(root: str, today: dt.date) -> list[int]:
    """Thu/Fri expirations whose quote window overlaps available history."""
    out = []
    for exp in list_expirations(root):
        ed = _to_date(exp)
        if ed.weekday() not in (3, 4):
            continue
        if exp < HISTORY_START:
            continue
        if ed > today + dt.timedelta(days=LOOKBACK_DAYS):
            continue
        out.append(exp)
    return out


def fetch_expiry(root: str, exp: int, today: dt.date) -> dict | None:
    ed = _to_date(exp)
    start = max(HISTORY_START, _to_int(ed - dt.timedelta(days=LOOKBACK_DAYS)))
    end = min(exp, _to_int(today))
    url = (
        f"{BASE}/bulk_hist/option/eod?root={root}&exp={exp}"
        f"&start_date={start}&end_date={end}"
    )
    data = _get(url)
    if not data or "response" not in data:
        return None
    fmt = data["header"]["format"]
    i_bid, i_ask = fmt.index("bid"), fmt.index("ask")
    i_vol, i_date = fmt.index("volume"), fmt.index("date")
    days: dict[int, dict] = {}
    for row in data["response"]:
        c = row["contract"]
        strike = c["strike"] / 1000.0
        right = c["right"]
        for t in row["ticks"]:
            q = days.setdefault(t[i_date], {})
            q[(strike, right)] = (t[i_bid], t[i_ask], t[i_vol])
    return {"expiry": exp, "days": days}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="SPY")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    today = dt.date.today()
    out_dir = Path(args.out or f"data/hedgehog_eod/{args.root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    expirations = target_expirations(args.root, today)
    todo = []
    for exp in expirations:
        path = out_dir / f"{exp}.pkl"
        if path.exists():
            continue  # complete or previously fetched; delete file to refetch
        todo.append(exp)
    print(f"{len(expirations)} target expirations, {len(todo)} to fetch", flush=True)

    failed = []
    t0 = time.time()

    def work(exp: int) -> tuple[int, bool]:
        payload = fetch_expiry(args.root, exp, today)
        if payload is None or not payload["days"]:
            return exp, False
        with open(out_dir / f"{exp}.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return exp, True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, exp): exp for exp in todo}
        done = 0
        for fut in as_completed(futures):
            exp, ok = fut.result()
            done += 1
            if not ok:
                failed.append(exp)
            if done % 10 == 0 or not ok:
                rate = done / max(time.time() - t0, 1)
                eta = (len(todo) - done) / max(rate, 1e-9) / 60
                status = "ok" if ok else "FAIL"
                print(
                    f"[{done}/{len(todo)}] {exp} {status} eta {eta:.0f}m", flush=True
                )

    print(f"done in {(time.time() - t0) / 60:.1f}m, {len(failed)} failures", flush=True)
    if failed:
        print("failed:", failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
