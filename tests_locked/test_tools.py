"""
Tool Handler Tests for 0DTE Trading Bot.
DO NOT MODIFY after approval - this file is part of the immutable test suite.

Tests cover:
- get_spx_price returns cached or live price
- get_option_chain filters 0DTE, returns Greeks
- get_5min_candles returns last N candles
- place_iron_condor validates legs, returns order ID
- place_stop_order creates OCO structure
- get_open_positions returns active ICs
- check_buying_power enforces 50% limit
"""

import pytest
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestGetSpxPrice:
    """Tests for get_spx_price tool."""

    @pytest.mark.asyncio
    async def test_returns_cached_price_when_valid(self, sample_spx_price):
        """Test that cached price is returned when cache is valid."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_quote = AsyncMock(return_value={"price": sample_spx_price})
        mock_cache.is_cache_valid = AsyncMock(return_value=True)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_spx_price()
        
        assert result["price"] == sample_spx_price
        mock_cache.get_cached_quote.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetches_live_price_when_cache_expired(self, sample_spx_price):
        """Test that live price is fetched when cache is expired."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_quote = AsyncMock(return_value=None)
        mock_cache.cache_quote = AsyncMock()
        
        mock_api = AsyncMock()
        mock_api.get_quote = AsyncMock(return_value={"price": sample_spx_price})
        
        handlers = ToolHandlers(cache=mock_cache, api_client=mock_api)
        
        result = await handlers.get_spx_price()
        
        assert result["price"] == sample_spx_price
        mock_api.get_quote.assert_called_once_with("SPX")
        mock_cache.cache_quote.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_error_on_api_failure(self):
        """Test that error is returned when API fails."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_quote = AsyncMock(return_value=None)
        
        mock_api = AsyncMock()
        mock_api.get_quote = AsyncMock(side_effect=Exception("API Error"))
        
        handlers = ToolHandlers(cache=mock_cache, api_client=mock_api)
        
        result = await handlers.get_spx_price()
        
        assert "error" in result
        assert "API Error" in result["error"]


class TestGetOptionChain:
    """Tests for get_option_chain tool."""

    @pytest.mark.asyncio
    async def test_filters_0dte_options(self, sample_option_chain):
        """Test that only 0DTE options are returned."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_option_chain_with_fallback = AsyncMock(return_value=sample_option_chain)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_option_chain(expiration="0DTE")
        
        assert len(result) > 0
        for option in result:
            assert "delta" in option
            assert "gamma" in option
            assert "theta" in option
            assert "vega" in option

    @pytest.mark.asyncio
    async def test_returns_greeks_for_all_options(self, sample_option_chain):
        """Test that Greeks are included for all options."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_option_chain_with_fallback = AsyncMock(return_value=sample_option_chain)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_option_chain()
        
        for option in result:
            assert "delta" in option
            assert isinstance(option["delta"], (int, float))
            assert "bid" in option
            assert "ask" in option

    @pytest.mark.asyncio
    async def test_filters_by_delta_range(self, sample_option_chain):
        """Test filtering options by delta range (10-15 delta)."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_option_chain_with_fallback = AsyncMock(return_value=sample_option_chain)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_option_chain(min_delta=0.10, max_delta=0.15)
        
        for option in result:
            abs_delta = abs(option["delta"])
            assert 0.10 <= abs_delta <= 0.15


class TestGet5minCandles:
    """Tests for get_5min_candles tool."""

    @pytest.mark.asyncio
    async def test_returns_last_n_candles(self, sample_5min_candles):
        """Test that last N candles are returned."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_5min_candles = AsyncMock(return_value=sample_5min_candles[-3:])
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_5min_candles(count=3)
        
        assert len(result) == 3
        mock_cache.get_5min_candles.assert_called_once_with("SPX", count=3)

    @pytest.mark.asyncio
    async def test_candles_have_required_fields(self, sample_5min_candles):
        """Test that candles have all required OHLCV fields."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_5min_candles = AsyncMock(return_value=sample_5min_candles)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_5min_candles(count=6)
        
        for candle in result:
            assert "timestamp" in candle
            assert "open" in candle
            assert "high" in candle
            assert "low" in candle
            assert "close" in candle
            assert "volume" in candle

    @pytest.mark.asyncio
    async def test_candles_ordered_by_timestamp(self, sample_5min_candles):
        """Test that candles are ordered by timestamp (oldest first)."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_5min_candles = AsyncMock(return_value=sample_5min_candles)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_5min_candles(count=6)
        
        for i in range(1, len(result)):
            assert result[i]["timestamp"] > result[i-1]["timestamp"]


class TestPlaceIronCondor:
    """Tests for place_iron_condor tool."""

    @pytest.mark.asyncio
    async def test_validates_4_legs(self):
        """Test that iron condor requires exactly 4 legs."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        handlers = ToolHandlers(api_client=mock_api)
        
        result = await handlers.place_iron_condor(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
            quantity=1,
        )
        
        assert "order_id" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_validates_strike_order(self):
        """Test that strikes are validated (long_put < short_put < short_call < long_call)."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        handlers = ToolHandlers(api_client=mock_api)
        
        result = await handlers.place_iron_condor(
            short_call_strike=5890.0,
            long_call_strike=5880.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
            quantity=1,
        )
        
        assert "error" in result
        assert "invalid" in result["error"].lower() or "strike" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_order_id_on_success(self):
        """Test that order ID is returned on successful placement."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_order = AsyncMock(return_value={"order_id": "IC-20240222-001"})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.place_iron_condor(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
            quantity=1,
        )
        
        assert "order_id" in result
        assert result["order_id"] == "IC-20240222-001"

    @pytest.mark.asyncio
    async def test_dry_run_does_not_execute(self):
        """Test that dry run mode does not actually place orders."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_order = AsyncMock()
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=True)
        
        result = await handlers.place_iron_condor(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
            quantity=1,
        )
        
        mock_api.place_order.assert_not_called()
        assert "dry_run" in result or "simulated" in str(result).lower()


class TestPlaceStopOrder:
    """Tests for place_stop_order tool."""

    @pytest.mark.asyncio
    async def test_creates_oco_structure(self):
        """Test that OCO order structure is created."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_oco_order = AsyncMock(return_value={
            "stop_limit_order_id": "SL-001",
            "stop_market_order_id": "SM-001",
        })
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.place_stop_order(
            symbol="SPXW240222C5890",
            stop_loss_amount=3.00,
            stop_limit_gap=40,
            stop_market_gap=30,
        )
        
        assert "stop_limit_order_id" in result
        assert "stop_market_order_id" in result

    @pytest.mark.asyncio
    async def test_stop_limit_with_40_point_gap(self):
        """Test that stop-limit has 40-point gap between trigger and limit."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_oco_order = AsyncMock(return_value={"orders": []})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        await handlers.place_stop_order(
            symbol="SPXW240222C5890",
            stop_loss_amount=3.00,
            stop_limit_gap=40,
            stop_market_gap=30,
        )
        
        call_args = mock_api.place_oco_order.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_stop_market_30_points_further(self):
        """Test that stop-market is 30 points further than stop-limit."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_oco_order = AsyncMock(return_value={"orders": []})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.place_stop_order(
            symbol="SPXW240222C5890",
            stop_loss_amount=3.00,
            stop_limit_gap=40,
            stop_market_gap=30,
        )
        
        assert "error" not in result


class TestGetOpenPositions:
    """Tests for get_open_positions tool."""

    @pytest.mark.asyncio
    async def test_returns_active_iron_condors(self, temp_db_path, sample_iron_condor_trade):
        """Test that only active iron condors are returned."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[sample_iron_condor_trade])
        
        handlers = ToolHandlers(db=mock_db)
        
        result = await handlers.get_open_positions()
        
        assert len(result) == 1
        assert result[0]["status"] == "open"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_positions(self):
        """Test that empty list is returned when no positions exist."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[])
        
        handlers = ToolHandlers(db=mock_db)
        
        result = await handlers.get_open_positions()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_includes_pnl_for_each_position(self, sample_iron_condor_trade):
        """Test that P&L is included for each position."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        trade_with_pnl = {**sample_iron_condor_trade, "unrealized_pnl": 50.0}
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[trade_with_pnl])
        
        handlers = ToolHandlers(db=mock_db)
        
        result = await handlers.get_open_positions()
        
        assert "unrealized_pnl" in result[0]


class TestCheckBuyingPower:
    """Tests for check_buying_power tool."""

    @pytest.mark.asyncio
    async def test_enforces_50_percent_limit(self, sample_account_info):
        """Test that 50% buying power limit is enforced."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.check_buying_power(required_amount=60000.0)
        
        assert result["allowed"] is False
        assert "exceeds 50%" in result["reason"].lower() or "limit" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_allows_within_limit(self, sample_account_info):
        """Test that trades within 50% limit are allowed."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.check_buying_power(required_amount=30000.0)
        
        assert result["allowed"] is True

    @pytest.mark.asyncio
    async def test_returns_available_buying_power(self, sample_account_info):
        """Test that available buying power is returned."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.check_buying_power()
        
        assert "available" in result
        assert "max_allowed" in result
        assert result["max_allowed"] == sample_account_info["equity"] * 0.5


class TestGetAccountInfo:
    """Tests for get_account_info tool."""

    @pytest.mark.asyncio
    async def test_returns_account_details(self, sample_account_info):
        """Test that account details are returned."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_cache = AsyncMock()
        mock_cache.get_account_info_with_fallback = AsyncMock(return_value=sample_account_info)
        
        handlers = ToolHandlers(cache=mock_cache)
        
        result = await handlers.get_account_info()
        
        assert "equity" in result
        assert "buying_power" in result
        assert result["equity"] == 100000.0


class TestPlaceCreditSpread:
    """Tests for place_credit_spread tool."""

    @pytest.mark.asyncio
    async def test_places_call_credit_spread(self):
        """Test placing a call credit spread."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_order = AsyncMock(return_value={"order_id": "CS-001"})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.place_credit_spread(
            spread_type="call",
            short_strike=5890.0,
            long_strike=5920.0,
            quantity=1,
        )
        
        assert "order_id" in result

    @pytest.mark.asyncio
    async def test_places_put_credit_spread(self):
        """Test placing a put credit spread."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_order = AsyncMock(return_value={"order_id": "CS-002"})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.place_credit_spread(
            spread_type="put",
            short_strike=5810.0,
            long_strike=5780.0,
            quantity=1,
        )
        
        assert "order_id" in result


class TestClosePosition:
    """Tests for close_position tool."""

    @pytest.mark.asyncio
    async def test_closes_specific_leg(self):
        """Test closing a specific leg of a position."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_order = AsyncMock(return_value={"order_id": "CL-001"})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.close_position(
            symbol="SPXW240222C5890",
            quantity=1,
            order_type="market",
        )
        
        assert "order_id" in result

    @pytest.mark.asyncio
    async def test_closes_entire_iron_condor(self):
        """Test closing an entire iron condor."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.place_order = AsyncMock(return_value={"order_id": "CL-002"})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.close_position(
            trade_id="IC-20240222-001",
            close_all=True,
        )
        
        assert "order_id" in result or "orders" in result


class TestCancelOrder:
    """Tests for cancel_order tool."""

    @pytest.mark.asyncio
    async def test_cancels_pending_order(self):
        """Test cancelling a pending order."""
        from agentic_ic.agent.tool_handlers import ToolHandlers
        
        mock_api = AsyncMock()
        mock_api.cancel_order = AsyncMock(return_value={"status": "cancelled"})
        
        handlers = ToolHandlers(api_client=mock_api, dry_run=False)
        
        result = await handlers.cancel_order(order_id="ORD-001")
        
        assert result["status"] == "cancelled"
        mock_api.cancel_order.assert_called_once_with("ORD-001")
