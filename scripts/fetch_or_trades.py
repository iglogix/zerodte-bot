"""Fetch trade-level data for the opening-range window and cache per-day
penny volume-at-price histograms (for a faithful FRVP, not 1-min smearing).

Source: XNAS.ITCH `trades` (full history 2018+, single-venue Nasdaq; the volume
*profile shape* is what matters, captured faithfully by relative price distn).
Window: 09:30-09:45 ET per session. Output: data/or_trade_vp/<SYM>.pkl
  { 'YYYY-MM-DD': {'p': int16 price-in-cents, 'v': float volume} }

Resumable: skips dates already cached. Run in background.
"""
from __future__ import annotations
import os, sys, pickle, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import pandas as pd
import databento as db

SYM = sys.argv[1] if len(sys.argv) > 1 else "SPY"
CSV = {"SPY": "data/spy_1m.csv", "IWM": "data/iwm_1m.csv", "NVDA": "data/nvda_1m.csv"}[SYM]
OUT = f"data/or_trade_vp/{SYM}.pkl"
OR_START, OR_END = "09:30", "09:45"   # ET opening-range window
DATASET = "XNAS.ITCH"

os.makedirs("data/or_trade_vp", exist_ok=True)
client = db.Historical(os.environ["DATABENTO_API_KEY"])


def trading_dates():
    df = pd.read_csv(CSV, usecols=["ts_event"])
    ts = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("America/New_York")
    return sorted(set(ts.dt.strftime("%Y-%m-%d")))


def utc_window(d: str):
    s = pd.Timestamp(f"{d} {OR_START}", tz="America/New_York").tz_convert("UTC")
    e = pd.Timestamp(f"{d} {OR_END}", tz="America/New_York").tz_convert("UTC")
    return s.strftime("%Y-%m-%dT%H:%M"), e.strftime("%Y-%m-%dT%H:%M")


def fetch_day(d: str):
    s, e = utc_window(d)
    try:
        data = client.timeseries.get_range(
            dataset=DATASET, symbols=[SYM], schema="trades",
            stype_in="raw_symbol", start=s, end=e)
        df = data.to_df()
    except Exception as ex:
        return d, None, str(ex)[:80]
    if df is None or len(df) == 0:
        return d, None, "empty"
    # price in dollars (databento prices are 1e-9 scaled -> to_df gives float)
    cents = np.round(df["price"].to_numpy() * 100).astype(np.int32)
    size = df["size"].to_numpy(float)
    # aggregate volume per penny
    order = np.argsort(cents)
    c_sorted = cents[order]; s_sorted = size[order]
    uniq, idx = np.unique(c_sorted, return_index=True)
    vols = np.add.reduceat(s_sorted, idx)
    return d, {"p": uniq.astype(np.int32), "v": vols.astype(np.float64)}, None


def main():
    cache = {}
    if os.path.exists(OUT):
        with open(OUT, "rb") as f:
            cache = pickle.load(f)
    dates = [d for d in trading_dates() if d not in cache]
    print(f"{SYM}: {len(cache)} cached, {len(dates)} to fetch", flush=True)
    done = err = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_day, d): d for d in dates}
        for i, fut in enumerate(as_completed(futs), 1):
            d, hist, msg = fut.result()
            if hist is not None:
                cache[d] = hist; done += 1
            else:
                err += 1
                if err <= 10:
                    print(f"  miss {d}: {msg}", flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(dates)} (ok={done} miss={err})", flush=True)
                with open(OUT, "wb") as f:
                    pickle.dump(cache, f)
    with open(OUT, "wb") as f:
        pickle.dump(cache, f)
    print(f"DONE {SYM}: {len(cache)} days cached, {err} misses", flush=True)


if __name__ == "__main__":
    main()
