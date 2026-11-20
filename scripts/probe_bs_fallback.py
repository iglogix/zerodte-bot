#!/usr/bin/env python3
"""Test: BS fallback greeks for ES futures options."""
from dotenv import load_dotenv; load_dotenv()
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_ic.services.tastytrade_client import TastyTradeClient

async def test():
    tt = TastyTradeClient(mode="manual")
    await tt.connect()
    print("Auth OK", flush=True)

    # Get ATM IV via REST (this works for /ES)
    atm_iv = await tt._get_atm_iv_rest("/ES")
    print(f"ATM IV (REST): {atm_iv}", flush=True)

    # Get ES price from SPY (works via DXLink) and scale
    try:
        q = await tt.get_quote("SPY")
        spy_price = q.get("price", 0)
        es_price = spy_price * 10  # approximate ES from SPY
        print(f"SPY={spy_price:.2f} → ES≈{es_price:.2f}", flush=True)
    except Exception:
        es_price = 5700.0
        print(f"Using default ES={es_price}", flush=True)

    # Get chain — DXLink will fail, BS fallback should kick in
    print("Fetching chain (expect BS fallback)...", flush=True)
    chain = await tt.get_future_option_chain("ES", underlying_price=es_price)
    print(f"Chain: {len(chain)} options", flush=True)

    # Check if BS enrichment worked
    with_delta = [o for o in chain if o.get("delta", 0) != 0]
    print(f"Options with delta: {with_delta and len(with_delta) or 0}/{len(chain)}", flush=True)

    # Show near-ATM
    near = [o for o in chain if abs(o["strike"] - es_price) < 30]
    print(f"\nNear-ATM ({len(near)} options):", flush=True)
    for o in sorted(near, key=lambda x: (x["strike"], x["option_type"]))[:10]:
        print(
            f"  {o['option_type']:4s} K={o['strike']:7.0f} "
            f"delta={o['delta']:+.4f} gamma={o['gamma']:.6f} "
            f"theta={o['theta']:.4f} vega={o['vega']:.4f} "
            f"bid={o['bid']:.2f} ask={o['ask']:.2f} iv={o['iv']:.4f}",
            flush=True,
        )

    await tt.disconnect()
    print("\nDone", flush=True)

loop = asyncio.new_event_loop()
def _suppress(loop, ctx):
    msg = ctx.get("message", "")
    if "SSL" in msg or "Bad file" in msg or "cancel" in msg.lower():
        return
    loop.default_exception_handler(ctx)
loop.set_exception_handler(_suppress)
try:
    loop.run_until_complete(test())
finally:
    loop.close()
