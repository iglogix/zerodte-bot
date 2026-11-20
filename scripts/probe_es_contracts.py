#!/usr/bin/env python3
"""Check ESH6 vs ESM26 prices."""
from dotenv import load_dotenv; load_dotenv()
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agentic_ic.services.tastytrade_client import TastyTradeClient

async def test():
    tt = TastyTradeClient(mode="manual")
    await tt.connect()
    for sym in ["/ESH26:XCME", "/ESM26:XCME", "/ESH6:XCME", "/ESM6:XCME"]:
        try:
            q = await tt.get_quote(sym)
            print(f"{sym}: bid={q.get('bid')}, ask={q.get('ask')}, mid={q.get('price')}")
        except Exception as e:
            print(f"{sym}: {e}")
    await tt.disconnect()

loop = asyncio.new_event_loop()
loop.set_exception_handler(lambda l, c: None)
try:
    loop.run_until_complete(test())
finally:
    loop.close()
