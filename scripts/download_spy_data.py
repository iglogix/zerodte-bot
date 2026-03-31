"""
Download SPY 1-min candles from MarketData.app for SPX 0DTE backtesting.

SPX daily expirations (0DTE every day) started 2022-05-11.
Uses SPY as SPX proxy (SPY * 10 ≈ SPX).

API: https://api.marketdata.app/v1/stocks/candles/1/SPY/
  - Max 1 year per intraday request
  - Cost: 1 credit per 1000 candles (~98 credits/year)
  - Starter plan: 10,000 daily credits

Usage:
  python -m scripts.download_spy_data
  python -m scripts.download_spy_data --from 2022-05-11 --to 2026-02-28
  python -m scripts.download_spy_data --extended  # include pre/post market (needed for 8AM ORB)
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
DEFAULT_START = date(2022, 5, 11)  # SPX 0DTE daily expirations started
DEFAULT_END = date(2026, 2, 28)

# Max intraday request span is 1 year; use ~6 months for safety
CHUNK_DAYS = 180


def get_token() -> str:
    """Load MARKET_DATA_TOKEN from environment or options_api .env."""
    token = os.environ.get("MARKET_DATA_TOKEN")
    if token:
        return token

    # Try loading from options_api .env
    env_path = Path(__file__).resolve().parents[2] / "options_api" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("MARKET_DATA_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                return token

    raise RuntimeError(
        "MARKET_DATA_TOKEN not found. Set it as env var or in ../options_api/.env"
    )


def fetch_candles(
    token: str,
    symbol: str,
    from_date: date,
    to_date: date,
    extended: bool = True,
) -> list[dict]:
    """Fetch 1-min candles for a date range (max ~1 year)."""
    url = f"{API_BASE}/stocks/candles/1/{symbol}/"
    params = {
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "extended": str(extended).lower(),
        "adjustsplits": "true",
    }
    headers = {"Authorization": f"Token {token}"}

    resp = requests.get(url, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if data.get("s") != "ok":
        print(f"  API returned status: {data.get('s')} — {data.get('errmsg', 'unknown error')}")
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
        ts_et = ts_utc.astimezone(ET)
        bars.append({
            "ts_event": ts_utc.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "ts_et": ts_et.isoformat(),
            "symbol": symbol,
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": volumes[i] if i < len(volumes) else 0,
        })

    return bars


def download_spy_data(
    from_date: date,
    to_date: date,
    output_path: Path,
    extended: bool = True,
    symbol: str = "SPY",
):
    """Download stock 1-min data in chunks and save to CSV."""
    token = get_token()
    all_bars = []
    total_credits = 0

    # Split into chunks to stay within 1-year intraday limit
    chunk_start = from_date
    chunk_num = 0

    while chunk_start < to_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), to_date)
        chunk_num += 1

        print(f"  Chunk {chunk_num}: {chunk_start} -> {chunk_end} ...", end=" ", flush=True)
        bars = fetch_candles(token, symbol, chunk_start, chunk_end, extended=extended)
        n = len(bars)
        credits_used = max(1, (n + 999) // 1000)
        total_credits += credits_used
        all_bars.extend(bars)
        print(f"{n:,} bars ({credits_used} credits)")

        chunk_start = chunk_end + timedelta(days=1)

        # Respect rate limits — small pause between chunks
        if chunk_start < to_date:
            time.sleep(0.5)

    if not all_bars:
        print("No data returned!")
        return

    # Sort by timestamp
    all_bars.sort(key=lambda b: b["ts_event"])

    # Save as CSV compatible with DatabentoLoader format
    # Columns: ts_event, symbol, open, high, low, close, volume
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts_event", "symbol", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for bar in all_bars:
            writer.writerow({
                "ts_event": bar["ts_event"],
                "symbol": bar["symbol"],
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            })

    # Count unique trading days
    trading_days = len(set(
        datetime.fromisoformat(b["ts_et"]).date() for b in all_bars
    ))

    print(f"\nDone! Saved {len(all_bars):,} bars ({trading_days} trading days) to {output_path}")
    print(f"Total API credits used: {total_credits}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download stock 1-min candles from MarketData.app")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--from", dest="from_date", default=DEFAULT_START.isoformat(),
                        help=f"Start date (default: {DEFAULT_START})")
    parser.add_argument("--to", dest="to_date", default=DEFAULT_END.isoformat(),
                        help=f"End date (default: {DEFAULT_END})")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: data/{symbol}_1m.csv)")
    parser.add_argument("--no-extended", action="store_true",
                        help="Exclude pre/post market hours (default: include for 8AM ORB)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)
    output_path = Path(args.output) if args.output else Path(f"data/{symbol.lower()}_1m.csv")

    print("=" * 60)
    print(f"  {symbol} 1-Min Candle Downloader (MarketData.app)")
    print("=" * 60)
    print(f"  Symbol  : {symbol}")
    print(f"  Range   : {from_date} -> {to_date}")
    print(f"  Extended: {not args.no_extended}")
    print(f"  Output  : {output_path}")
    print()

    download_spy_data(from_date, to_date, output_path, extended=not args.no_extended, symbol=symbol)


if __name__ == "__main__":
    main()
