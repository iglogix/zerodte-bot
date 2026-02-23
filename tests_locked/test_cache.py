"""
Market Data Cache Tests for 0DTE Trading Bot.
DO NOT MODIFY after approval - this file is part of the immutable test suite.

Tests cover:
- Cache write: option prices (60s TTL), 5-min candles (permanent)
- Cache read: retrieve by symbol + timestamp
- Cache expiry: TTL-based invalidation for options
- Permanent candle storage: no expiry, append-only
- Rate limit simulation: cache hit prevents API call
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import json
import asyncio


class TestCacheWrite:
    """Tests for writing data to cache."""

    @pytest.mark.asyncio
    async def test_cache_write_option_prices(self, temp_cache_db_path, sample_option_chain):
        """Test writing option prices with 60s TTL."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_option_chain("SPX", sample_option_chain)
        
        cached = await cache.get_cached_option_chain("SPX")
        assert cached is not None
        assert len(cached) == len(sample_option_chain)
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_write_5min_candles_permanent(self, temp_cache_db_path, sample_5min_candles):
        """Test writing 5-min candles with permanent storage (no TTL)."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        for candle in sample_5min_candles:
            await cache.cache_5min_candle("SPX", candle)
        
        candles = await cache.get_5min_candles("SPX", count=6)
        assert len(candles) == 6
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_write_spx_quote(self, temp_cache_db_path, sample_spx_price):
        """Test writing SPX quote with 60s TTL."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_quote("SPX", {"price": sample_spx_price, "timestamp": datetime.now().isoformat()})
        
        cached = await cache.get_cached_quote("SPX")
        assert cached is not None
        assert cached["price"] == sample_spx_price
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_write_account_info(self, temp_cache_db_path, sample_account_info):
        """Test writing account info with 60s TTL."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_account_info(sample_account_info)
        
        cached = await cache.get_cached_account_info()
        assert cached is not None
        assert cached["equity"] == 100000.0
        
        await cache.close()


class TestCacheRead:
    """Tests for reading data from cache."""

    @pytest.mark.asyncio
    async def test_cache_read_by_symbol(self, temp_cache_db_path, sample_option_chain):
        """Test reading cached data by symbol."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_option_chain("SPX", sample_option_chain)
        await cache.cache_option_chain("SPY", [{"symbol": "SPY240222C580", "strike": 580.0}])
        
        spx_chain = await cache.get_cached_option_chain("SPX")
        spy_chain = await cache.get_cached_option_chain("SPY")
        
        assert len(spx_chain) == len(sample_option_chain)
        assert len(spy_chain) == 1
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_read_by_timestamp_range(self, temp_cache_db_path, sample_5min_candles):
        """Test reading cached candles by timestamp range."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        for candle in sample_5min_candles:
            await cache.cache_5min_candle("SPX", candle)
        
        start_time = sample_5min_candles[2]["timestamp"]
        end_time = sample_5min_candles[5]["timestamp"]
        
        candles = await cache.get_5min_candles_range("SPX", start_time, end_time)
        assert len(candles) == 4
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_read_returns_none_when_empty(self, temp_cache_db_path):
        """Test that cache returns None when no data exists."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        result = await cache.get_cached_option_chain("NONEXISTENT")
        assert result is None
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_read_latest_entry(self, temp_cache_db_path):
        """Test that cache returns the latest entry when multiple exist."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_quote("SPX", {"price": 5800.0, "timestamp": datetime.now().isoformat()})
        await asyncio.sleep(0.01)
        await cache.cache_quote("SPX", {"price": 5850.0, "timestamp": datetime.now().isoformat()})
        await asyncio.sleep(0.01)
        await cache.cache_quote("SPX", {"price": 5900.0, "timestamp": datetime.now().isoformat()})
        
        cached = await cache.get_cached_quote("SPX")
        assert cached["price"] == 5900.0
        
        await cache.close()


class TestCacheExpiry:
    """Tests for TTL-based cache invalidation."""

    @pytest.mark.asyncio
    async def test_option_chain_expires_after_60s(self, temp_cache_db_path, sample_option_chain):
        """Test that option chain cache expires after 60 seconds."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_option_chain("SPX", sample_option_chain, ttl_seconds=60)
        
        cached = await cache.get_cached_option_chain("SPX", max_age_seconds=60)
        assert cached is not None
        
        cached_expired = await cache.get_cached_option_chain("SPX", max_age_seconds=0)
        assert cached_expired is None
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_quote_expires_after_60s(self, temp_cache_db_path, sample_spx_price):
        """Test that quote cache expires after 60 seconds."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_quote("SPX", {"price": sample_spx_price}, ttl_seconds=60)
        
        cached = await cache.get_cached_quote("SPX", max_age_seconds=60)
        assert cached is not None
        
        cached_expired = await cache.get_cached_quote("SPX", max_age_seconds=0)
        assert cached_expired is None
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_is_cache_valid_method(self, temp_cache_db_path):
        """Test the is_cache_valid helper method."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        await cache.cache_quote("SPX", {"price": 5850.0}, ttl_seconds=60)
        
        is_valid = await cache.is_cache_valid("SPX", "quote", max_age_seconds=60)
        assert is_valid is True
        
        is_valid_strict = await cache.is_cache_valid("SPX", "quote", max_age_seconds=0)
        assert is_valid_strict is False
        
        await cache.close()


class TestPermanentCandleStorage:
    """Tests for permanent 5-min candle storage."""

    @pytest.mark.asyncio
    async def test_candles_never_expire(self, temp_cache_db_path, sample_5min_candles):
        """Test that 5-min candles never expire (permanent storage)."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        for candle in sample_5min_candles:
            await cache.cache_5min_candle("SPX", candle, permanent=True)
        
        candles = await cache.get_5min_candles("SPX", count=100, ignore_ttl=True)
        assert len(candles) == 6
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_candles_append_only(self, temp_cache_db_path):
        """Test that candles are append-only (no overwrites)."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        candle1 = {"timestamp": datetime(2024, 2, 22, 10, 0, 0), "open": 5850.0, "close": 5852.0}
        candle2 = {"timestamp": datetime(2024, 2, 22, 10, 5, 0), "open": 5852.0, "close": 5855.0}
        candle3 = {"timestamp": datetime(2024, 2, 22, 10, 10, 0), "open": 5855.0, "close": 5853.0}
        
        await cache.cache_5min_candle("SPX", candle1)
        await cache.cache_5min_candle("SPX", candle2)
        await cache.cache_5min_candle("SPX", candle3)
        
        candles = await cache.get_5min_candles("SPX", count=10)
        assert len(candles) == 3
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_candles_deduplicated_by_timestamp(self, temp_cache_db_path):
        """Test that duplicate candles (same timestamp) are not stored twice."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        candle = {"timestamp": datetime(2024, 2, 22, 10, 0, 0), "open": 5850.0, "close": 5852.0}
        
        await cache.cache_5min_candle("SPX", candle)
        await cache.cache_5min_candle("SPX", candle)
        await cache.cache_5min_candle("SPX", candle)
        
        candles = await cache.get_5min_candles("SPX", count=10)
        assert len(candles) == 1
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_candles_queryable_for_backtesting(self, temp_cache_db_path):
        """Test that historical candles can be queried for backtesting."""
        from src.services.market_data import CachedMarketData
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path)
        await cache.initialize()
        
        base_time = datetime(2024, 2, 22, 9, 30, 0)
        for i in range(100):
            candle = {
                "timestamp": base_time + timedelta(minutes=5 * i),
                "open": 5850.0 + i,
                "high": 5852.0 + i,
                "low": 5848.0 + i,
                "close": 5851.0 + i,
                "volume": 100000 + i * 1000,
            }
            await cache.cache_5min_candle("SPX", candle)
        
        all_candles = await cache.get_5min_candles("SPX", count=1000)
        assert len(all_candles) == 100
        
        start = base_time + timedelta(minutes=50)
        end = base_time + timedelta(minutes=100)
        range_candles = await cache.get_5min_candles_range("SPX", start, end)
        assert len(range_candles) == 11
        
        await cache.close()


class TestRateLimitPrevention:
    """Tests for rate limit prevention via caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_prevents_api_call(self, temp_cache_db_path, sample_option_chain):
        """Test that a cache hit prevents an API call."""
        from src.services.market_data import CachedMarketData
        
        mock_api = AsyncMock()
        mock_api.get_option_chain = AsyncMock(return_value=sample_option_chain)
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path, api_client=mock_api)
        await cache.initialize()
        
        await cache.cache_option_chain("SPX", sample_option_chain, ttl_seconds=60)
        
        result = await cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)
        
        mock_api.get_option_chain.assert_not_called()
        assert len(result) == len(sample_option_chain)
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_miss_triggers_api_call(self, temp_cache_db_path, sample_option_chain):
        """Test that a cache miss triggers an API call."""
        from src.services.market_data import CachedMarketData
        
        mock_api = AsyncMock()
        mock_api.get_option_chain = AsyncMock(return_value=sample_option_chain)
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path, api_client=mock_api)
        await cache.initialize()
        
        result = await cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)
        
        mock_api.get_option_chain.assert_called_once_with("SPX")
        assert len(result) == len(sample_option_chain)
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_expired_cache_triggers_api_call(self, temp_cache_db_path, sample_option_chain):
        """Test that an expired cache triggers an API call."""
        from src.services.market_data import CachedMarketData
        
        mock_api = AsyncMock()
        mock_api.get_option_chain = AsyncMock(return_value=sample_option_chain)
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path, api_client=mock_api)
        await cache.initialize()
        
        await cache.cache_option_chain("SPX", [{"old": "data"}], ttl_seconds=60)
        
        result = await cache.get_option_chain_with_fallback("SPX", max_age_seconds=0)
        
        mock_api.get_option_chain.assert_called_once()
        assert len(result) == len(sample_option_chain)
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_api_result_cached_after_fetch(self, temp_cache_db_path, sample_option_chain):
        """Test that API results are cached after fetching."""
        from src.services.market_data import CachedMarketData
        
        mock_api = AsyncMock()
        mock_api.get_option_chain = AsyncMock(return_value=sample_option_chain)
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path, api_client=mock_api)
        await cache.initialize()
        
        await cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)
        
        await cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)
        
        assert mock_api.get_option_chain.call_count == 1
        
        await cache.close()

    @pytest.mark.asyncio
    async def test_multiple_symbols_cached_independently(self, temp_cache_db_path):
        """Test that different symbols are cached independently."""
        from src.services.market_data import CachedMarketData
        
        spx_chain = [{"symbol": "SPX", "strike": 5850.0}]
        spy_chain = [{"symbol": "SPY", "strike": 585.0}]
        
        mock_api = AsyncMock()
        mock_api.get_option_chain = AsyncMock(side_effect=[spx_chain, spy_chain])
        
        cache = CachedMarketData(cache_db_path=temp_cache_db_path, api_client=mock_api)
        await cache.initialize()
        
        spx_result = await cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)
        spy_result = await cache.get_option_chain_with_fallback("SPY", max_age_seconds=60)
        
        assert spx_result[0]["symbol"] == "SPX"
        assert spy_result[0]["symbol"] == "SPY"
        
        spx_cached = await cache.get_option_chain_with_fallback("SPX", max_age_seconds=60)
        assert mock_api.get_option_chain.call_count == 2
        
        await cache.close()
