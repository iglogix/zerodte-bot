#!/usr/bin/env python3
"""Skyline study — fetch ^GSPC daily OHLC (price index, no dividend adjustment)
into the per-study cache backtest_data/skyline.db.

^GSPC is a PRICE index, so there is no raw-vs-adjusted basis question (A4);
it is used ONLY as an independent cross-check of the parity-derived settlement
and for the prior-close entry filter.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "backtest_data" / "skyline.db"


def fetch(start: date, end: date) -> list[tuple]:
    p1 = int(datetime(start.year, start.month, start.day).timestamp()) - 86400
    p2 = int(datetime(end.year, end.month, end.day).timestamp()) + 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
           f"?period1={p1}&period2={p2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = []
    for t, o, h, l, c in zip(res["timestamp"], q["open"], q["high"],
                             q["low"], q["close"]):
        if c is None:
            continue
        d = datetime.fromtimestamp(t).date()
        rows.append((d.isoformat(), o, h, l, c))
    return rows


def main() -> int:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS gspc_daily(
        d TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS provenance(
        k TEXT PRIMARY KEY, v TEXT)""")
    rows = fetch(date(2022, 4, 1), date.today())
    con.executemany("INSERT OR REPLACE INTO gspc_daily VALUES (?,?,?,?,?)", rows)
    con.execute("INSERT OR REPLACE INTO provenance VALUES (?,?)",
                ("gspc_fetch", f"yahoo chart ^GSPC 1d, fetched {date.today()}"))
    con.execute("INSERT OR REPLACE INTO provenance VALUES (?,?)",
                ("quote_cache", "data/cbbo_cache_fly_10min (Databento OPRA cbbo-1m, "
                 "+/-250pt @5pt around 09:30 open, calls+puts; built by "
                 "scripts/backfill_butterfly_cbbo.py commit fa856f6, fetched 2026-06-03..06-12)"))
    con.commit()
    n = con.execute("SELECT COUNT(*), MIN(d), MAX(d) FROM gspc_daily").fetchone()
    print(f"gspc_daily rows={n[0]} {n[1]}..{n[2]}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
