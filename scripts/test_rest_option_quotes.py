#!/usr/bin/env python3
"""Test: Can we get real ES option quotes via TastyTrade REST API?

This script tries to fetch bid/ask for specific ES 0DTE option symbols
via the /market-data/quotes endpoint. If it works, we don't need
browser scraping — REST is simpler and faster.

Run during market hours:
  python scripts/test_rest_option_quotes.py
"""
from dotenv import load_dotenv; load_dotenv()
import asyncio, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_ic.services.tastytrade_client import TastyTradeClient


async def test():
    tt = TastyTradeClient(mode="manual")
    await tt.connect()
    print("Auth OK\n")

    # Get current ES price
    q = await tt.get_quote("/ESH26:XCME")
    es_price = q.get("price", 0)
    print(f"ES price: {es_price}")

    # Get the option chain to find real option symbols
    chain = await tt.get_future_option_chain("ES", underlying_price=es_price)
    print(f"Chain: {len(chain)} options\n")

    # Pick a few near-ATM options
    near_atm = [
        o for o in chain
        if abs(o["strike"] - es_price) < 30
    ]
    near_atm.sort(key=lambda x: (x["strike"], x["option_type"]))

    # Try to fetch REST quotes for individual options
    print("=== REST Quotes for Individual ES Options ===\n")
    session = tt._sdk_session or tt._market_sdk_session
    hdrs = tt._market_headers()

    success = 0
    for opt in near_atm[:10]:
        sym = opt.get("symbol", "")
        streamer_sym = opt.get("streamer_symbol", "")
        strike = opt["strike"]
        otype = opt["option_type"]

        # Try different symbol formats
        for try_sym in [sym, streamer_sym]:
            if not try_sym:
                continue
            try:
                # Use the REST endpoint directly
                import aiohttp
                http, base = tt._market_http_and_url()
                rest_sym = try_sym.lstrip("/").lstrip(".")
                url = f"{base}/market-data/quotes/{rest_sym}"
                async with http.get(url, headers=hdrs) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        item = data.get("data", {})
                        bid = item.get("bid", 0)
                        ask = item.get("ask", 0)
                        last = item.get("last", 0)
                        iv = item.get("implied-volatility", None)
                        delta = item.get("delta", None)
                        print(
                            f"  {otype:4s} K={strike:7.0f} "
                            f"bid={bid} ask={ask} last={last} "
                            f"iv={iv} delta={delta} "
                            f"[{try_sym}]"
                        )
                        success += 1
                        break
                    else:
                        text = await resp.text()
                        print(f"  {otype:4s} K={strike:7.0f} — {resp.status} [{try_sym}] {text[:100]}")
            except Exception as e:
                print(f"  {otype:4s} K={strike:7.0f} — Error [{try_sym}]: {e}")

    print(f"\nSuccess: {success}/{min(10, len(near_atm))}")
    await tt.disconnect()
    print("Done")


loop = asyncio.new_event_loop()
loop.set_exception_handler(lambda l, c: None)
try:
    loop.run_until_complete(test())
finally:
    loop.close()
