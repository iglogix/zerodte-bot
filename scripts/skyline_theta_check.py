#!/usr/bin/env python3
"""Sanity gate: hand-reconcile Databento cbbo-1m marks against ThetaData NBBO.

Samples random (date, strike, right) rows actually used by the Skyline entry and
compares bid/ask at 09:45 ET between the two independent OPRA feeds.
"""
from __future__ import annotations

import pickle
import random
import sys
import urllib.request
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cbbo_cache_fly_10min"
THETA = "http://127.0.0.1:25510"


def theta_quote(d: date, strike: float, right: str, ms_of_day: int):
    url = (f"{THETA}/v2/hist/option/quote?root=SPXW"
           f"&exp={d.strftime('%Y%m%d')}&strike={int(strike*1000)}"
           f"&right={right}&start_date={d.strftime('%Y%m%d')}"
           f"&end_date={d.strftime('%Y%m%d')}&ivl=60000")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            js = json.load(r)
    except Exception as exc:
        return None, f"http:{exc}"
    if "response" not in js:
        return None, f"err:{str(js)[:90]}"
    fmt = js["header"]["format"]
    i_ms, i_bid, i_ask = fmt.index("ms_of_day"), fmt.index("bid"), fmt.index("ask")
    best = None
    for row in js["response"]:
        if row[i_ms] <= ms_of_day:
            best = (row[i_bid], row[i_ask])
        else:
            break
    return best, None


def main() -> int:
    random.seed(7)
    files = sorted(CACHE.glob("*.pkl"))
    samples = random.sample(files, 12)
    ms945 = (9 * 3600 + 45 * 60) * 1000
    rows, errs = [], []
    for f in samples:
        d = date.fromisoformat(f.stem)
        with open(f, "rb") as fh:
            q = pickle.load(fh)
        keys = [k for k in q if k[0] == "09:45" and q[k][1] > 0]
        if not keys:
            continue
        for k in random.sample(keys, min(3, len(keys))):
            _, strike, is_call = k
            db_bid, db_ask = q[k]
            tq, err = theta_quote(d, strike, "C" if is_call else "P", ms945)
            if err:
                errs.append((d, strike, err))
                continue
            if tq is None:
                errs.append((d, strike, "no-row"))
                continue
            rows.append((d, strike, "C" if is_call else "P",
                         db_bid, db_ask, tq[0], tq[1]))

    print(f"{'date':12} {'strike':>8} {'R':2} {'DB bid':>8} {'DB ask':>8} "
          f"{'TH bid':>8} {'TH ask':>8} {'dBid':>7} {'dAsk':>7}")
    dmax = 0.0
    for d, k, r, bb, ba, tb, ta in rows:
        db_, da_ = bb - tb, ba - ta
        dmax = max(dmax, abs(db_), abs(da_))
        print(f"{d.isoformat():12} {k:8.0f} {r:2} {bb:8.2f} {ba:8.2f} "
              f"{tb:8.2f} {ta:8.2f} {db_:+7.2f} {da_:+7.2f}")
    print(f"\nn={len(rows)}  max abs diff = {dmax:.2f}")
    if errs:
        print(f"theta errors ({len(errs)}): {errs[:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
