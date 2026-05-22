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


def test_main_routes_meic_strategy():
    import yaml
    from src_v2.main import BotOrchestrator
    from src_v2.strategies.meic import MeicStrategy

    with open("config/config_meic_spx.yaml") as f:
        config = yaml.safe_load(f)

    orch = BotOrchestrator(config=config, mode="dry_run")
    strategy = orch._create_strategy("meic")
    assert isinstance(strategy, MeicStrategy)
    # configured from YAML
    assert strategy._stop_pct == 0.70
    assert strategy._wing_em_frac == 0.5


@pytest.mark.asyncio
async def test_place_two_spread_ic_live_submits_short_stops():
    """Live MEIC entry submits the 2 spreads AND a stop-market on each short."""
    from datetime import date
    from src_v2.backtest.osi_symbol import format_osi

    api = MagicMock()
    api.place_order = AsyncMock(return_value={"order_id": "SPREAD"})
    api.place_stop_market = AsyncMock(
        side_effect=lambda sym, trig, q: {"order_id": "STOP", "symbol": sym, "trigger_price": trig})
    th = ToolHandlers(cache=MagicMock(), api_client=api, db=MagicMock(), mode="live")

    payload = {
        "order_type": "meic_two_spreads", "quantity": 3,
        "put_spread": {"short_strike": 5750, "long_strike": 5660,
                       "credit": 1.40, "stop_trigger": 3.41, "quantity": 3},
        "call_spread": {"short_strike": 5950, "long_strike": 6040,
                        "credit": 1.40, "stop_trigger": 3.41, "quantity": 3},
    }
    result = await th.place_two_spread_ic(payload)

    assert api.place_order.await_count == 2          # put + call spreads
    assert api.place_stop_market.await_count == 2    # short put + short call stops
    exp = date.today()
    stop_syms = {c.args[0] for c in api.place_stop_market.await_args_list}
    assert format_osi(exp, 5750, is_call=False) in stop_syms   # short put OCC
    assert format_osi(exp, 5950, is_call=True) in stop_syms    # short call OCC
    assert result["put_stop_result"]["order_id"] == "STOP"
    assert result["call_stop_result"]["order_id"] == "STOP"
