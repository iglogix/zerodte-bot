#!/usr/bin/env python3
"""Pull TODAY's actual SPX option fills from the TastyTrade account and
reconcile them against the bot's assumed MEIC entry credits.

Ground-truth fill quality: your real executed prices vs the mid-based credits
the bot logged, and vs the backtest's 0.10/spread slippage assumption.
"""
import asyncio
import sys
import json
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from tastytrade.account import Account
from src_v2.services.tastytrade_client import TastyTradeClient

DAY = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()


async def main():
    client = TastyTradeClient(mode="manual")
    await client.connect()
    session = await client.get_session()
    acct_num = await client.get_account_number()
    acct = await Account.get(session, acct_num)   # fetch full account object
    res = acct.get_history(session, start_date=DAY, underlying_symbol="SPX")
    txns = await res if asyncio.iscoroutine(res) else res
    opt = [t for t in txns
           if str(getattr(t, "instrument_type", "")).lower().find("option") >= 0
           and getattr(t, "price", None) is not None]
    print(f"{len(opt)} SPX option fills on {DAY}\n")
    print(f"{'executed':20}{'action':16}{'symbol':22}{'qty':>4}{'price':>9}")
    fills = []
    for t in sorted(opt, key=lambda x: str(x.executed_at)):
        sym = (t.symbol or "").strip()
        print(f"{str(t.executed_at)[:19]:20}{str(t.action):16}{sym:22}"
              f"{int(t.quantity):>4}{float(t.price):>9.2f}")
        fills.append({"executed_at": str(t.executed_at), "action": str(t.action),
                      "symbol": sym, "qty": int(t.quantity), "price": float(t.price)})

    out = Path(f"results/meic_paper/fills_{DAY}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fills, indent=1))
    print(f"\nsaved -> {out}")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
