"""
Sandbox smoke test — verifies connectivity to TastyTrade cert environment.

Checks:
  1. OAuth2 authentication (OAuthSession)
  2. Account info (buying power, equity)
  3. SPX quote fetch
  4. 0DTE option chain (first 5 strikes)
  5. Open positions list

Run:
  BOT_MODE=sandbox python scripts/sandbox_smoke_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")


async def main():
    from src.services.tastytrade_client import TastyTradeClient

    mode = os.environ.get("BOT_MODE", "sandbox")
    client = TastyTradeClient(mode=mode)

    logger.info("=" * 60)
    logger.info("TastyTrade Sandbox Smoke Test")
    logger.info("=" * 60)
    logger.info("provider_secret : %s...%s", client.provider_secret[:6], client.provider_secret[-4:])
    logger.info("refresh_token   : %s...%s", client.refresh_token[:10], client.refresh_token[-6:])
    logger.info("account_number  : %s", client.account_number)
    logger.info("base_url        : %s", client.base_url)
    logger.info("")

    # ------------------------------------------------------------------ #
    # 1. Authenticate
    # ------------------------------------------------------------------ #
    logger.info("Step 1: Authenticating via OAuthSession...")
    await client.connect()
    logger.info("  ✓ Authenticated — access_token: %s...%s",
                client._access_token[:10], client._access_token[-6:])

    # ------------------------------------------------------------------ #
    # 2. Account info
    # ------------------------------------------------------------------ #
    logger.info("Step 2: Fetching account info...")
    try:
        info = await client.get_account_info()
        logger.info("  ✓ Account  : %s", info["account_number"])
        logger.info("    Equity   : $%.2f", info["equity"])
        logger.info("    BuyPower : $%.2f", info["buying_power"])
        logger.info("    Cash     : $%.2f", info["cash_balance"])
    except Exception as e:
        logger.error("  ✗ Account info failed: %s", e)

    # ------------------------------------------------------------------ #
    # 3. SPX quote
    # ------------------------------------------------------------------ #
    logger.info("Step 3: Fetching SPX quote...")
    try:
        quote = await client.get_quote("SPX")
        logger.info("  ✓ SPX price : $%.2f  bid=%.2f  ask=%.2f",
                    quote["price"], quote["bid"], quote["ask"])
    except Exception as e:
        logger.error("  ✗ SPX quote failed: %s", e)

    # ------------------------------------------------------------------ #
    # 4. 0DTE option chain
    # ------------------------------------------------------------------ #
    logger.info("Step 4: Fetching 0DTE SPXW option chain...")
    try:
        chain = await client.get_option_chain("SPXW")
        if chain:
            logger.info("  ✓ Option chain: %d strikes returned", len(chain))
            for opt in chain[:3]:
                logger.info("    %s  strike=%.0f  bid=%.2f  ask=%.2f  delta=%.3f",
                            opt["option_type"].upper(), opt["strike"],
                            opt["bid"], opt["ask"], opt["delta"])
        else:
            logger.warning("  ⚠ Option chain returned empty (market may be closed)")
    except Exception as e:
        logger.error("  ✗ Option chain failed: %s", e)

    # ------------------------------------------------------------------ #
    # 5. Open positions
    # ------------------------------------------------------------------ #
    logger.info("Step 5: Fetching open positions...")
    try:
        async with client._http.get(
            f"{client.base_url}/accounts/{client.account_number}/positions",
            headers=client._headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            positions = data.get("data", {}).get("items", [])
            if positions:
                logger.info("  ✓ Open positions: %d", len(positions))
                for p in positions[:5]:
                    logger.info("    %s  qty=%s  direction=%s",
                                p.get("symbol"), p.get("quantity"), p.get("quantity-direction"))
            else:
                logger.info("  ✓ No open positions (sandbox account is clean)")
    except Exception as e:
        logger.error("  ✗ Positions fetch failed: %s", e)

    await client.disconnect()
    logger.info("")
    logger.info("=" * 60)
    logger.info("Smoke test complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
