#!/usr/bin/env python3
"""
Collect real ES 0DTE option quotes via TastyTrade SDK DXLink streamer.

Uses the standard SDK DXLinkStreamer (api token) to subscribe to Quote and
Greeks for near-ATM ES futures options. Saves snapshots to
data/iv_collection/ in JSONL format for ANN training.

Usage:
  python scripts/collect_es_options.py                # collect until 16:15 ET
  python scripts/collect_es_options.py --duration 30   # collect for 30 minutes
"""

from dotenv import load_dotenv; load_dotenv()

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, date, time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.tastytrade_client import TastyTradeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
COLLECT_DIR = Path("data/iv_collection")


class EsOptionsCollector:
    """
    Collects real ES 0DTE option quotes (bid/ask/greeks) via SDK DXLinkStreamer.

    Subscribes to Quote + Greeks for near-ATM options, accumulates data,
    and flushes JSONL records periodically.
    """

    def __init__(self):
        self._tt_client = None
        self._streamer = None
        self._output_path = COLLECT_DIR / f"{date.today().isoformat()}_es.jsonl"
        self._records_written = 0
        self._spot_price = 0.0
        self._es_symbol = ""

    async def initialize(self):
        """Connect to TastyTrade API."""
        COLLECT_DIR.mkdir(parents=True, exist_ok=True)
        self._tt_client = TastyTradeClient(mode="manual")
        await self._tt_client.connect()
        logger.info("TastyTrade API connected")

    async def get_chain_symbols(self) -> list[dict]:
        """Fetch today's ES 0DTE option chain symbols."""
        from src_8am.utils.contract_utils import get_dxlink_symbol
        from tastytrade.instruments import NestedFutureOptionChain

        self._es_symbol = get_dxlink_symbol()
        logger.info("Front-month contract: %s", self._es_symbol)

        q = await self._tt_client.get_quote(self._es_symbol)
        self._spot_price = q.get("price", 0)
        logger.info("ES spot: %.2f", self._spot_price)

        session = self._tt_client._market_sdk_session or self._tt_client._sdk_session
        chain = await NestedFutureOptionChain.a_get(session, "ES")

        today_str = date.today().strftime("%Y-%m-%d")
        options = []
        for subchain in chain.option_chains:
            for expiration in subchain.expirations:
                if str(expiration.expiration_date) != today_str:
                    continue
                for strike in expiration.strikes:
                    k = float(strike.strike_price)
                    if abs(k - self._spot_price) > 150:
                        continue
                    for side in ["call", "put"]:
                        streamer_sym = getattr(strike, f"{side}_streamer_symbol", "")
                        sym = getattr(strike, side, None)
                        if streamer_sym:
                            options.append({
                                "symbol": str(sym) if sym else "",
                                "streamer_symbol": streamer_sym,
                                "strike": k,
                                "option_type": side,
                            })

        logger.info("ES 0DTE chain: %d options (±150 pts of %.2f)",
                    len(options), self._spot_price)
        return options

    async def collect(self, options: list[dict], duration_minutes: int = 0):
        """
        Subscribe to Quote + Greeks via SDK DXLinkStreamer, collect data.
        """
        from tastytrade import DXLinkStreamer
        from tastytrade.dxfeed import Quote, Greeks

        sym_meta = {opt["streamer_symbol"]: opt for opt in options}
        streamer_syms = list(sym_meta.keys())

        session = self._tt_client._market_sdk_session or self._tt_client._sdk_session

        now_et = datetime.now(ET)
        if duration_minutes > 0:
            end_ts = now_et.timestamp() + duration_minutes * 60
        else:
            close = datetime.combine(now_et.date(), time(16, 15), tzinfo=ET)
            end_ts = close.timestamp()

        logger.info("Collecting until %s",
                    datetime.fromtimestamp(end_ts, ET).strftime("%H:%M ET"))

        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Quote, streamer_syms)
            await streamer.subscribe(Greeks, streamer_syms)
            logger.info("Subscribed to %d symbols (Quote + Greeks)", len(streamer_syms))

            quotes = {}   # sym → {bid, ask, bid_size, ask_size}
            greeks = {}   # sym → {price, iv, delta, gamma, theta, vega}
            flush_interval = 60
            last_flush = asyncio.get_event_loop().time()

            while datetime.now(ET).timestamp() < end_ts:
                try:
                    # Read Quote events
                    try:
                        q = await asyncio.wait_for(
                            streamer.get_event(Quote), timeout=2
                        )
                        sym = q.event_symbol
                        quotes[sym] = {
                            "bid": float(q.bid_price or 0),
                            "ask": float(q.ask_price or 0),
                            "bid_size": float(q.bid_size or 0),
                            "ask_size": float(q.ask_size or 0),
                        }
                    except asyncio.TimeoutError:
                        pass

                    # Read Greeks events
                    try:
                        g = await asyncio.wait_for(
                            streamer.get_event(Greeks), timeout=0.5
                        )
                        sym = g.event_symbol
                        greeks[sym] = {
                            "price": float(g.price or 0),
                            "iv": float(g.volatility or 0),
                            "delta": float(g.delta or 0),
                            "gamma": float(g.gamma or 0),
                            "theta": float(g.theta or 0),
                            "vega": float(g.vega or 0),
                        }
                    except asyncio.TimeoutError:
                        pass

                    # Periodic flush
                    now_loop = asyncio.get_event_loop().time()
                    if now_loop - last_flush >= flush_interval:
                        n = self._flush(quotes, greeks, sym_meta)
                        if n > 0:
                            logger.info(
                                "Flushed %d records (total: %d, Q: %d, G: %d)",
                                n, self._records_written,
                                len(quotes), len(greeks),
                            )
                        last_flush = now_loop

                        # Refresh spot price
                        await self._update_spot()

                except Exception as exc:
                    logger.warning("Collection loop error: %s", exc)
                    await asyncio.sleep(1)

        # Final flush
        n = self._flush(quotes, greeks, sym_meta)
        logger.info("Final flush: %d records. Total: %d", n, self._records_written)

    def _flush(self, quotes: dict, greeks: dict, sym_meta: dict) -> int:
        """Write merged quote+greeks records to JSONL."""
        now_str = datetime.now(ET).isoformat()
        count = 0

        with open(self._output_path, "a") as f:
            for sym in quotes:
                q = quotes[sym]
                g = greeks.get(sym, {})
                meta = sym_meta.get(sym, {})

                bid = q.get("bid", 0)
                ask = q.get("ask", 0)
                if not bid or not ask or bid <= 0 or ask <= 0:
                    continue

                mid = (bid + ask) / 2
                record = {
                    "timestamp": now_str,
                    "symbol": "ES",
                    "streamer_symbol": sym,
                    "option_symbol": meta.get("symbol", ""),
                    "strike": meta.get("strike", 0),
                    "option_type": meta.get("option_type", ""),
                    "spot": self._spot_price,
                    "bid": round(bid, 4),
                    "ask": round(ask, 4),
                    "mid": round(mid, 4),
                    "bid_size": q.get("bid_size", 0),
                    "ask_size": q.get("ask_size", 0),
                    "market_iv": g.get("iv", 0),
                    "delta": g.get("delta", 0),
                    "gamma": g.get("gamma", 0),
                    "theta": g.get("theta", 0),
                    "vega": g.get("vega", 0),
                    "theo_price": g.get("price", 0),
                    "source": "dxlink_sdk",
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        self._records_written += count
        return count

    async def _update_spot(self):
        """Refresh ES spot price."""
        try:
            q = await self._tt_client.get_quote(self._es_symbol)
            self._spot_price = q.get("price", self._spot_price)
        except Exception:
            pass

    async def shutdown(self):
        if self._tt_client:
            await self._tt_client.disconnect()


async def main():
    parser = argparse.ArgumentParser(
        description="Collect ES 0DTE option quotes via SDK DXLink"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="Collection duration in minutes (0=until 16:15 ET)",
    )
    args = parser.parse_args()

    collector = EsOptionsCollector()

    try:
        await collector.initialize()
        options = await collector.get_chain_symbols()
        if not options:
            logger.error("No ES 0DTE options found for today")
            return
        await collector.collect(options, duration_minutes=args.duration)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.error("Collection failed: %s", exc, exc_info=True)
    finally:
        await collector.shutdown()

    logger.info("Done. Records: %d → %s",
                collector._records_written, collector._output_path)


if __name__ == "__main__":
    asyncio.run(main())
