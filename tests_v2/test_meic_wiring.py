"""MEIC wiring: tool handler + config routing."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src_v2.agent.tool_handlers import ToolHandlers


@pytest.mark.asyncio
async def test_place_two_spread_ic_dry_run_simulates():
    cache = MagicMock()
    api = MagicMock()
    db = MagicMock()
    th = ToolHandlers(cache=cache, api_client=api, db=db, mode="dry_run")

    payload = {
        "order_type": "meic_two_spreads",
        "quantity": 4,
        "put_spread": {"short_strike": 5750, "long_strike": 5660,
                       "credit": 1.40, "quantity": 4},
        "call_spread": {"short_strike": 5950, "long_strike": 6040,
                        "credit": 1.40, "quantity": 4},
    }
    result = await th.place_two_spread_ic(payload)
    assert result["dry_run"] is True
    assert result["quantity"] == 4
    # Both spread legs reported
    assert result["put_spread"]["short_strike"] == 5750
    assert result["call_spread"]["short_strike"] == 5950
