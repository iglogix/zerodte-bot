#!/usr/bin/env python3
"""
Multi-symbol 0DTE options data collector for ANN training.

Collects real futures option quotes (bid/ask/greeks/IV) via TastyTrade SDK
DXLinkStreamer for multiple uncorrelated products. Saves to JSONL for
unified ANN training.

Supported symbols: ES, NQ, CL, GC, ZN, RTY

Usage:
  python scripts/collect_options.py                    # all symbols, until 16:15 ET
  python scripts/collect_options.py --symbols ES NQ    # specific symbols
  python scripts/collect_options.py --duration 60      # collect for 60 minutes
  python scripts/collect_options.py --range 100        # override ATM range (pts)
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

# ── Product definitions ──────────────────────────────────────────────
# Each product defines its contract cycle, exchange, and reasonable
# ATM range for strike filtering.
#
# Contract months:
#   ES/NQ: H(Mar) M(Jun) U(Sep) Z(Dec) — quarterly
#   CL:    F G H J K M N Q U V X Z     — monthly
#   GC:    G J M Q Z + F(Feb) (main)    — bimonthly, but we use nearest
#   RTY:   H M U Z                      — quarterly
#   ZN:    H M U Z                      — quarterly

PRODUCTS = {
    "ES": {
        "exchange": "XCME",
        "contract_months": {3: "H", 6: "M", 9: "U", 12: "Z"},
        "atm_range": 150,       # points each side of spot
        "multiplier": 50,
        "description": "E-mini S&P 500",
    },
    "NQ": {
        "exchange": "XCME",
        "contract_months": {3: "H", 6: "M", 9: "U", 12: "Z"},
        "atm_range": 500,       # NQ trades ~20K, wider range needed
        "multiplier": 20,
        "description": "E-mini Nasdaq 100",
    },
    "CL": {
        "exchange": "XNYM",
        "contract_months": {m: c for m, c in zip(
            range(1, 13),
            "F G H J K M N Q U V X Z".split()
        )},
        "atm_range": 5,         # CL trades ~$70, $5 range
        "multiplier": 1000,
        "description": "Crude Oil",
    },
    "GC": {
        "exchange": "XCEC",
        "contract_months": {2: "G", 4: "J", 6: "M", 8: "Q", 12: "Z"},
        "atm_range": 200,       # GC trades ~$4500+, need wider range
        "multiplier": 100,
        "description": "Gold",
    },
    "RTY": {
        "exchange": "XCME",
        "contract_months": {3: "H", 6: "M", 9: "U", 12: "Z"},
        "atm_range": 80,        # RTY trades ~2100, moderate range
        "multiplier": 50,
        "description": "Russell 2000",
    },
    "ZN": {
        "exchange": "XCBT",
        "contract_months": {3: "H", 6: "M", 9: "U", 12: "Z"},
        "atm_range": 3,         # ZN trades ~110, tight range
        "multiplier": 1000,
        "description": "10-Year Treasury Note",
    },
}


def _short_to_dxlink(short_sym: str, exchange: str) -> str:
    """
    Convert short futures symbol to DXLink format.

    /ESH6  → /ESH26:XCME
    /CLK6  → /CLK26:XNYM
    /GCJ6  → /GCJ26:XCEC
    """
    # Short format: /XXY# where Y=month code, #=single digit year
    # DXLink format: /XXY##:EXCHANGE where ##=two digit year
    if len(short_sym) < 4:
        return f"{short_sym}:{exchange}"

    # Extract the year digit(s) at the end
    base = short_sym.rstrip("0123456789")
    year_part = short_sym[len(base):]

    if len(year_part) == 1:
        # Single digit year: 6 → 26 (assume 2020s)
        year_2d = f"2{year_part}"
    elif len(year_part) == 2:
        year_2d = year_part
    else:
        year_2d = year_part[-2:]

    return f"{base}{year_2d}:{exchange}"


async def discover_active_contract(product: str, session) -> tuple[str, str] | None:
    """
    Discover the active futures contract for a product by querying the
    option chain API. Returns the DXLink symbol for the underlying
    that has 0DTE options today.

    Returns:
        (dxlink_symbol, exchange) or None if no options found.
        e.g. ("/ESH26:XCME", "XCME")
    """
    from tastytrade.instruments import NestedFutureOptionChain

    try:
        chain = await NestedFutureOptionChain.a_get_chain(session, product)
    except Exception as exc:
        logger.error("%s: failed to fetch option chain: %s", product, exc)
        return None

    today_str = date.today().strftime("%Y-%m-%d")

    # Walk the chain to find which underlying has 0DTE options today
    best = None  # (short_sym, exchange) for fallback

    for subchain in chain.option_chains:
        for expiration in subchain.expirations:
            short_sym = expiration.underlying_symbol  # e.g. "/ESH6"

            # Get exchange from first streamer symbol
            exchange = PRODUCTS[product]["exchange"]
            if expiration.strikes:
                s0 = expiration.strikes[0]
                ss = getattr(s0, "call_streamer_symbol", "")
                if ":" in ss:
                    exchange = ss.split(":")[-1]

            if str(expiration.expiration_date) == today_str:
                dxlink = _short_to_dxlink(short_sym, exchange)
                logger.info("%s: found 0DTE options under %s → %s",
                            product, short_sym, dxlink)
                return dxlink, exchange

            if best is None:
                best = (short_sym, exchange, expiration.expiration_date)

    if best:
        short_sym, exchange, exp_date = best
        dxlink = _short_to_dxlink(short_sym, exchange)
        logger.warning("%s: no 0DTE today (nearest %s), using %s",
                       product, exp_date, dxlink)
        return dxlink, exchange

    return None


class MultiSymbolCollector:
    """
    Collects 0DTE option quotes for multiple futures products via DXLinkStreamer.

    Uses a single TastyTrade session and streamer connection for all symbols.
    """

    def __init__(self, symbols: list[str], atm_range_override: int | None = None):
        self._symbols = [s.upper() for s in symbols]
        self._atm_range_override = atm_range_override
        self._tt_client = None
        self._output_path = COLLECT_DIR / f"{date.today().isoformat()}_multi.jsonl"
        self._records_written = 0
        self._spot_prices: dict[str, float] = {}   # product → spot
        self._dxlink_symbols: dict[str, str] = {}   # product → dxlink futures sym
        self._total_options = 0

    async def initialize(self):
        """Connect to TastyTrade API."""
        COLLECT_DIR.mkdir(parents=True, exist_ok=True)
        self._tt_client = TastyTradeClient(mode="manual")
        await self._tt_client.connect()
        logger.info("TastyTrade API connected")

    async def get_all_chain_symbols(self) -> list[dict]:
        """Fetch 0DTE option chain symbols for all configured products."""
        from tastytrade.instruments import NestedFutureOptionChain

        session = self._tt_client._market_sdk_session or self._tt_client._sdk_session
        today_str = date.today().strftime("%Y-%m-%d")
        all_options = []

        for product in self._symbols:
            if product not in PRODUCTS:
                logger.warning("Unknown product: %s, skipping", product)
                continue

            info = PRODUCTS[product]
            atm_range = self._atm_range_override or info["atm_range"]

            # Discover active contract from option chain API
            result = await discover_active_contract(product, session)
            if not result:
                logger.warning("%s: no active contract found, skipping", product)
                continue

            dxlink_sym, _ = result
            self._dxlink_symbols[product] = dxlink_sym
            logger.info("%s active contract: %s", product, dxlink_sym)

            # Get spot price via DXLink using the exact contract symbol
            # discovered from the option chain (avoids contract month mismatch).
            try:
                q = await self._tt_client.get_quote(dxlink_sym)
                spot = q.get("price", 0)
                self._spot_prices[product] = spot
                logger.info("%s spot: %.2f (via DXLink %s)", product, spot, dxlink_sym)
            except Exception as exc:
                logger.error("%s: failed to get spot price: %s", product, exc)
                continue

            if spot <= 0:
                logger.error("%s: invalid spot price %.2f, skipping", product, spot)
                continue

            # Fetch option chain and extract 0DTE strikes
            try:
                chain = await NestedFutureOptionChain.a_get_chain(session, product)
            except Exception as exc:
                logger.error("%s: failed to fetch option chain: %s", product, exc)
                continue

            product_options = []
            for subchain in chain.option_chains:
                for expiration in subchain.expirations:
                    if str(expiration.expiration_date) != today_str:
                        continue
                    for strike in expiration.strikes:
                        k = float(strike.strike_price)
                        if abs(k - spot) > atm_range:
                            continue
                        for side in ["call", "put"]:
                            streamer_sym = getattr(strike, f"{side}_streamer_symbol", "")
                            sym = getattr(strike, side, None)
                            if streamer_sym:
                                product_options.append({
                                    "product": product,
                                    "symbol": str(sym) if sym else "",
                                    "streamer_symbol": streamer_sym,
                                    "strike": k,
                                    "option_type": side,
                                })

            logger.info("%s: %d 0DTE options (±%s of %.2f)",
                        product, len(product_options), atm_range, spot)
            all_options.extend(product_options)

        self._total_options = len(all_options)
        logger.info("Total: %d options across %d products",
                    self._total_options, len(self._spot_prices))
        return all_options

    async def collect(self, options: list[dict], duration_minutes: int = 0):
        """Subscribe to Quote + Greeks via SDK DXLinkStreamer, collect data."""
        from tastytrade import DXLinkStreamer
        from tastytrade.dxfeed import Quote, Greeks

        if not options:
            logger.error("No options to collect")
            return

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
                    try:
                        q = await asyncio.wait_for(
                            streamer.get_event(Quote), timeout=2
                        )
                        quotes[q.event_symbol] = {
                            "bid": float(q.bid_price or 0),
                            "ask": float(q.ask_price or 0),
                            "bid_size": float(q.bid_size or 0),
                            "ask_size": float(q.ask_size or 0),
                        }
                    except asyncio.TimeoutError:
                        pass

                    try:
                        g = await asyncio.wait_for(
                            streamer.get_event(Greeks), timeout=0.5
                        )
                        greeks[g.event_symbol] = {
                            "price": float(g.price or 0),
                            "iv": float(g.volatility or 0),
                            "delta": float(g.delta or 0),
                            "gamma": float(g.gamma or 0),
                            "theta": float(g.theta or 0),
                            "vega": float(g.vega or 0),
                        }
                    except asyncio.TimeoutError:
                        pass

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
                        await self._update_spots()

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

                product = meta.get("product", "")
                spot = self._spot_prices.get(product, 0)
                strike = meta.get("strike", 0)
                mid = (bid + ask) / 2

                # Compute ANN features
                moneyness = strike / spot if spot > 0 else 0
                is_call = 1.0 if meta.get("option_type") == "call" else 0.0

                record = {
                    "timestamp": now_str,
                    "symbol": product,
                    "streamer_symbol": sym,
                    "option_symbol": meta.get("symbol", ""),
                    "strike": strike,
                    "option_type": meta.get("option_type", ""),
                    "spot": spot,
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
                    "moneyness": round(moneyness, 6),
                    "is_call": is_call,
                    "multiplier": PRODUCTS.get(product, {}).get("multiplier", 1),
                    "source": "dxlink_sdk",
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        self._records_written += count
        return count

    async def _update_spots(self):
        """Refresh spot prices via DXLink using the exact contract symbol
        discovered from the option chain (avoids contract month mismatch).
        """
        for product, dxlink_sym in self._dxlink_symbols.items():
            try:
                q = await self._tt_client.get_quote(dxlink_sym)
                price = q.get("price", 0)
                if price > 0:
                    self._spot_prices[product] = price
            except Exception:
                pass

    async def shutdown(self):
        if self._tt_client:
            await self._tt_client.disconnect()


async def main():
    parser = argparse.ArgumentParser(
        description="Collect multi-symbol 0DTE option quotes for ANN training"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=list(PRODUCTS.keys()),
        help=f"Products to collect (default: all). Choices: {', '.join(PRODUCTS.keys())}",
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="Collection duration in minutes (0=until 16:15 ET)",
    )
    parser.add_argument(
        "--range", type=int, default=None, dest="atm_range",
        help="Override ATM range for all products (points)",
    )
    args = parser.parse_args()

    collector = MultiSymbolCollector(
        symbols=args.symbols,
        atm_range_override=args.atm_range,
    )

    try:
        await collector.initialize()
        options = await collector.get_all_chain_symbols()
        if not options:
            logger.error("No 0DTE options found for any product today")
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
