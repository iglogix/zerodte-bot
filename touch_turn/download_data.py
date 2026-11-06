"""
Download equity 1-min candles from MarketData.app for Touch & Turn backtesting.

Usage:
  python -m touch_turn.download_data --symbol NVDA
  python -m touch_turn.download_data --symbol NVDA --from 2022-01-01 --to 2024-12-31
  python -m touch_turn.download_data --symbol AAPL --output data/aapl_1m.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

API_BASE = "https://api.marketdata.app/v1"
CHUNK_DAYS = 180  # stay within 1-year intraday limit


def get_token() -> str:
    """Load MARKET_DATA_TOKEN from environment or options_api .env."""
    token = os.environ.get("MARKET_DATA_TOKEN")
    if token:
        return token

    # Check project .env, then sibling options_api/.env
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / ".env",
        project_root.parent / "options_api" / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("MARKET_DATA_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError(
        "MARKET_DATA_TOKEN not found. Set it as env var or in .env / ../options_api/.env"
    )


def fetch_candles(
    token: str,
    symbol: str,
    from_date: date,
    to_date: date,
) -> list[dict]:
    """Fetch 1-min candles for a date range (RTH only for equities)."""
    url = f"{API_BASE}/stocks/candles/1/{symbol}/"
    params = {
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "extended": "false",  # RTH only for equities
        "adjustsplits": "true",
    }
    headers = {"Authorization": f"Token {token}"}

    resp = requests.get(url, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if data.get("s") != "ok":
        print(f"  API returned: {data.get('s')} — {data.get('errmsg', '')}")
        return []

    bars = []
    timestamps = data.get("t", [])
    opens = data.get("o", [])
    highs = data.get("h", [])
    lows = data.get("l", [])
    closes = data.get("c", [])
    volumes = data.get("v", [])

    for i in range(len(timestamps)):
        ts_utc = datetime.fromtimestamp(timestamps[i], tz=UTC)
        bars.append({
            "ts_event": ts_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "symbol": symbol,
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": volumes[i] if i < len(volumes) else 0,
        })

    return bars


def download(symbol: str, from_date: date, to_date: date, output_path: Path):
    """Download equity 1-min data in chunks and save to CSV."""
    token = get_token()
    all_bars: list[dict] = []
    total_credits = 0

    chunk_start = from_date
    chunk_num = 0

    while chunk_start < to_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), to_date)
        chunk_num += 1

        print(f"  Chunk {chunk_num}: {chunk_start} -> {chunk_end} ...", end=" ", flush=True)
        bars = fetch_candles(token, symbol, chunk_start, chunk_end)
        n = len(bars)
        credits_used = max(1, (n + 999) // 1000)
        total_credits += credits_used
        all_bars.extend(bars)
        print(f"{n:,} bars ({credits_used} credits)")

        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start < to_date:
            time.sleep(0.5)

    if not all_bars:
        print("No data returned!")
        return

    all_bars.sort(key=lambda b: b["ts_event"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ts_event", "symbol", "open", "high", "low", "close", "volume"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bar in all_bars:
            writer.writerow(bar)

    print(f"\nDone! Saved {len(all_bars):,} bars to {output_path}")
    print(f"Total API credits used: {total_credits}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download equity 1-min candles")
    parser.add_argument("--symbol", default="NVDA", help="Ticker symbol (default: NVDA)")
    parser.add_argument("--from", dest="from_date", default="2022-01-01")
    parser.add_argument("--to", dest="to_date", default="2024-12-31")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: data/<symbol>_1m.csv)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)
    output_path = Path(args.output or f"data/{symbol.lower()}_1m.csv")

    print("=" * 60)
    print(f"  {symbol} 1-Min Candle Downloader (MarketData.app)")
    print("=" * 60)
    print(f"  Symbol  : {symbol}")
    print(f"  Range   : {from_date} -> {to_date}")
    print(f"  Output  : {output_path}")
    print()

    download(symbol, from_date, to_date, output_path)


if __name__ == "__main__":
    main()
