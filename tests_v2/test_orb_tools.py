"""Tests for ORB tools on ToolHandlers."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src_v2.agent.tool_handlers import ToolHandlers


def _make_handlers(candles=None, api_quote=None, api_error=None):
    cache = AsyncMock()
    cache.get_candles = AsyncMock(return_value=candles or [])
    api = AsyncMock()
    if api_error:
        api.get_quote = AsyncMock(side_effect=api_error)
    else:
        api.get_quote = AsyncMock(return_value=api_quote or {})
    return ToolHandlers(cache=cache, api_client=api, mode="manual")


@pytest.mark.asyncio
async def test_get_orb_levels_computes_high_low_midline():
    candles = [{"high": 5860, "low": 5845, "open": 5850, "close": 5855},
               {"high": 5865, "low": 5840, "open": 5855, "close": 5850}]
    h = _make_handlers(candles=candles)
    result = await h.get_orb_levels("ES", "08:00", "08:15")
    assert result["high"] == 5865
    assert result["low"] == 5840
    assert result["midline"] == (5865 + 5840) / 2
    assert result["range_pts"] == 25


@pytest.mark.asyncio
async def test_get_orb_levels_caches():
    candles = [{"high": 100, "low": 90, "open": 95, "close": 96}]
    h = _make_handlers(candles=candles)
    r1 = await h.get_orb_levels("ES", "08:00", "08:15")
    r2 = await h.get_orb_levels("ES", "08:00", "08:15")
    assert r1 is r2
    h.cache.get_candles.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_orb_levels_no_candles():
    h = _make_handlers(candles=[])
    result = await h.get_orb_levels("ES", "08:00", "08:15")
    assert "error" in result


@pytest.mark.asyncio
async def test_breakout_long():
    h = _make_handlers()
    h._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
    result = await h.get_breakout_status(101)
    assert result["status"] == "breakout"
    assert result["bias"] == "long"


@pytest.mark.asyncio
async def test_breakout_short():
    h = _make_handlers()
    h._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
    result = await h.get_breakout_status(89)
    assert result["status"] == "breakout"
    assert result["bias"] == "short"


@pytest.mark.asyncio
async def test_breakout_inside_range():
    h = _make_handlers()
    h._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
    result = await h.get_breakout_status(95)
    assert result["status"] == "inside_range"


@pytest.mark.asyncio
async def test_breakout_bias_latches():
    h = _make_handlers()
    h._orb_levels = {"high": 100, "low": 90, "midline": 95, "range_pts": 10}
    await h.get_breakout_status(101)  # long breakout
    result = await h.get_breakout_status(89)  # would be short, but latched
    assert result["bias"] == "long"


def test_midline_touch_within():
    h = _make_handlers()
    assert h.check_midline_touch(95.1, 95.0, 0.25) is True


def test_midline_touch_outside():
    h = _make_handlers()
    assert h.check_midline_touch(96.0, 95.0, 0.25) is False


@pytest.mark.asyncio
async def test_get_orb_spot_parity():
    h = _make_handlers()
    chain = [
        {"strike": 5850, "option_type": "call", "bid": 7.4, "ask": 7.6},
        {"strike": 5850, "option_type": "put", "bid": 7.4, "ask": 7.6},
    ]
    result = await h.get_orb_spot(chain, 5880.0)
    assert result["source"] == "put_call_parity"
    assert result["spot"] == 5850.0  # K + C_mid - P_mid = 5850 + 7.5 - 7.5


@pytest.mark.asyncio
async def test_get_orb_spot_api_fallback():
    h = _make_handlers(api_quote={"price": 5848.5})
    result = await h.get_orb_spot([], 5880.0, index_symbol="SPX")
    assert result["source"] == "api_quote"
    assert result["spot"] == 5848.5


@pytest.mark.asyncio
async def test_get_orb_spot_both_fail():
    h = _make_handlers()
    result = await h.get_orb_spot([], 5880.0)
    assert "error" in result


def test_reset_orb_state():
    h = _make_handlers()
    h._orb_levels = {"high": 100}
    h._breakout_bias = "long"
    h._breakout_price = 101.0
    h.reset_orb_state()
    assert h._orb_levels is None
    assert h._breakout_bias is None
    assert h._breakout_price is None
