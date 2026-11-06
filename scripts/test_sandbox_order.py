#!/usr/bin/env python3
"""
Quick sandbox smoke test: connect → fetch ES chain → build order → submit.
Run: python scripts/test_sandbox_order.py
"""

import asyncio
import sys
import os
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from agentic_ic.services.tastytrade_client import TastyTradeClient


async def main():
    # Load config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config_8am_es_options.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    option_root = config.get("options", {}).get("option_root", "EW")
    product_code = config.get("bot", {}).get("symbol", "ES")

    print(f"=== Sandbox Order Smoke Test ===")
    print(f"Product: {product_code}, Option root: {option_root}")

    tt = TastyTradeClient(mode="sandbox")
    await tt.connect()
    print("✓ Connected to sandbox")

    try:
        # 1. Fetch current ES price
        from es_8am.utils.contract_utils import get_dxlink_symbol
        dxlink_sym = get_dxlink_symbol()
        print(f"\nFetching quote for {dxlink_sym}...")
        quote = await tt.get_quote(dxlink_sym)
        es_price = quote["price"]
        print(f"✓ ES price: {es_price:.2f}")

        # 2. Fetch futures option chain
        print(f"\nFetching {product_code} 0DTE futures option chain...")
        chain = await tt.get_future_option_chain(product_code, underlying_price=es_price)
        print(f"✓ Got {len(chain)} options")

        if not chain:
            print("✗ Empty chain — no 0DTE options available today?")
            return

        # Filter to today
        from datetime import date
        today = date.today().isoformat()
        chain = [o for o in chain if str(o.get("expiration", "")).startswith(today)]
        print(f"  0DTE options: {len(chain)}")

        # 3. Find a far-OTM put spread (safe to submit — won't fill)
        puts = [o for o in chain if o["option_type"] == "put" and o["strike"] < es_price - 100]
        puts.sort(key=lambda o: o["strike"], reverse=True)

        if len(puts) < 2:
            print("✗ Not enough OTM puts to build a test spread")
            return

        short_put = puts[0]
        # Find long put 5 pts below
        spread_width = config.get("options", {}).get("spread_width_pts", 5)
        long_target = short_put["strike"] - spread_width
        long_candidates = [p for p in puts if abs(p["strike"] - long_target) <= 1]
        if not long_candidates:
            long_put = puts[1]
        else:
            long_put = long_candidates[0]

        print(f"\n  Test spread: sell {short_put['strike']}P / buy {long_put['strike']}P")
        print(f"  Short symbol: {short_put['symbol']}")
        print(f"  Long symbol:  {long_put['symbol']}")
        print(f"  Short delta:  {short_put.get('delta', 'N/A')}")

        # 4. Build and submit order
        order = {
            "order_type": "put_credit_spread",
            "short_strike": short_put["strike"],
            "long_strike": long_put["strike"],
            "short_symbol": short_put["symbol"],
            "long_symbol": long_put["symbol"],
            "instrument_type": "Future Option",
            "quantity": 1,
        }

        # Estimate credit from bid/ask
        short_mid = (short_put.get("bid", 0) + short_put.get("ask", 0)) / 2
        long_mid = (long_put.get("bid", 0) + long_put.get("ask", 0)) / 2
        credit = round(max(0.05, short_mid - long_mid), 2)

        order["credit"] = credit
        print(f"\nSubmitting order to sandbox (credit={credit})...")
        print(f"  Order payload: {order}")

        try:
            result = await tt.place_order(order)
            print(f"✓ Order accepted! Result: {result}")
        except Exception as exc:
            print(f"✗ Order rejected: {exc}")

    finally:
        await tt.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    asyncio.run(main())
