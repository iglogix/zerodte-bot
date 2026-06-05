#!/usr/bin/env python3
"""Sanity cross-check: ThetaData NBBO vs project's Databento CBBO pickle for one 0DTE day.

Compares bid/ask for a handful of (strike, right, minute) points so we can confirm the
Theta pipeline returns sane, consistent quotes before building the full MEIC re-engine.
"""
import json
import pickle
import sys
import urllib.request
from datetime import datetime

THETA = "http://127.0.0.1:25510"
DATE = "20240605"
PKL = "~/Dev/0dte_bot/data/cbbo_cache_meic/2024-06-05.pkl"

STRIKES = [5320.0, 5340.0, 5350.0, 5360.0, 5380.0]
RIGHTS = [("C", True), ("P", False)]
# ET minutes to probe (MEIC entry/management window)
MINUTES = ["10:00", "11:00", "13:00", "14:30", "15:00"]


def ms_to_hhmm(ms: int) -> str:
    m = ms // 60000
    return f"{m // 60:02d}:{m % 60:02d}"


def hhmm_to_ms(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return (int(h) * 60 + int(m)) * 60000


def theta_quote_minute(root, exp, strike, right, date):
    """Return {hhmm: (bid, ask)} for the day at 1-min interval."""
    strike_milli = int(round(strike * 1000))
    url = (f"{THETA}/v2/hist/option/quote?root={root}&exp={exp}"
           f"&strike={strike_milli}&right={right}"
           f"&start_date={date}&end_date={date}&ivl=60000&rth=true")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    fmt = data["header"]["format"]
    i_ms, i_bid, i_ask = fmt.index("ms_of_day"), fmt.index("bid"), fmt.index("ask")
    out = {}
    for row in data["response"]:
        out[ms_to_hhmm(row[i_ms])] = (row[i_bid], row[i_ask])
    return out


def main():
    with open(PKL, "rb") as f:
        cache = pickle.load(f)
    print(f"Databento pickle: {len(cache)} keys; sample key types: "
          f"{type(next(iter(cache)))} -> {type(next(iter(cache.values())))}")

    header = f"{'strike':>7} {'R':>1} {'min':>5} | {'theta_bid':>9} {'theta_ask':>9} | {'dbn_bid':>8} {'dbn_ask':>8} | {'d_mid':>7}"
    print("\n" + header)
    print("-" * len(header))

    rows = 0
    mid_abs_diffs = []
    for strike in STRIKES:
        for right_code, is_call in RIGHTS:
            try:
                tq = theta_quote_minute("SPXW", DATE, strike, right_code, DATE)
            except Exception as e:
                print(f"{strike:>7} {right_code:>1}  Theta error: {e}")
                continue
            for mn in MINUTES:
                tb, ta = tq.get(mn, (None, None))
                dbn = cache.get((mn, strike, is_call))
                if dbn is None:
                    db, da = None, None
                else:
                    db, da = dbn
                if None not in (tb, ta, db, da):
                    tmid, dmid = (tb + ta) / 2, (db + da) / 2
                    dmid_diff = tmid - dmid
                    mid_abs_diffs.append(abs(dmid_diff))
                    dstr = f"{dmid_diff:+7.2f}"
                else:
                    dstr = f"{'--':>7}"
                fb = f"{tb:9.2f}" if tb is not None else f"{'--':>9}"
                fa = f"{ta:9.2f}" if ta is not None else f"{'--':>9}"
                gb = f"{db:8.2f}" if db is not None else f"{'--':>8}"
                ga = f"{da:8.2f}" if da is not None else f"{'--':>8}"
                print(f"{strike:>7.0f} {right_code:>1} {mn:>5} | {fb} {fa} | {gb} {ga} | {dstr}")
                rows += 1

    print("-" * len(header))
    if mid_abs_diffs:
        import statistics
        print(f"\nMatched points: {len(mid_abs_diffs)}  |  "
              f"mean |mid diff| = {statistics.mean(mid_abs_diffs):.3f}  "
              f"median = {statistics.median(mid_abs_diffs):.3f}  "
              f"max = {max(mid_abs_diffs):.3f}")
    else:
        print("\nNo overlapping points matched — investigate key format / minute alignment.")


if __name__ == "__main__":
    main()
