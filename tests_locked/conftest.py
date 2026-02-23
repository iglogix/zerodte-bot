"""
Pytest fixtures for 0DTE trading bot tests.
DO NOT MODIFY after approval - this file is part of the immutable test suite.
"""

import pytest
import asyncio
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_0dte_bot.db")


@pytest.fixture
def temp_cache_db_path(tmp_path):
    """Create a temporary cache database path."""
    return str(tmp_path / "test_market_cache.db")


@pytest.fixture
def sample_spx_price() -> float:
    """Sample SPX price for testing."""
    return 5850.25


@pytest.fixture
def sample_option_chain() -> List[Dict[str, Any]]:
    """Sample 0DTE SPX option chain with Greeks."""
    spx_price = 5850.0
    return [
        # Call options
        {"symbol": "SPXW240222C5880", "strike": 5880.0, "option_type": "call", 
         "delta": 0.12, "gamma": 0.002, "theta": -0.85, "vega": 0.15,
         "bid": 1.80, "ask": 2.00, "last": 1.90, "volume": 1500, "open_interest": 3200},
        {"symbol": "SPXW240222C5890", "strike": 5890.0, "option_type": "call",
         "delta": 0.10, "gamma": 0.0018, "theta": -0.75, "vega": 0.12,
         "bid": 1.20, "ask": 1.40, "last": 1.30, "volume": 2100, "open_interest": 4500},
        {"symbol": "SPXW240222C5900", "strike": 5900.0, "option_type": "call",
         "delta": 0.08, "gamma": 0.0015, "theta": -0.65, "vega": 0.10,
         "bid": 0.80, "ask": 1.00, "last": 0.90, "volume": 1800, "open_interest": 3800},
        {"symbol": "SPXW240222C5910", "strike": 5910.0, "option_type": "call",
         "delta": 0.06, "gamma": 0.0012, "theta": -0.55, "vega": 0.08,
         "bid": 0.50, "ask": 0.70, "last": 0.60, "volume": 1200, "open_interest": 2900},
        {"symbol": "SPXW240222C5920", "strike": 5920.0, "option_type": "call",
         "delta": 0.04, "gamma": 0.0010, "theta": -0.45, "vega": 0.06,
         "bid": 0.30, "ask": 0.50, "last": 0.40, "volume": 900, "open_interest": 2100},
        # Put options
        {"symbol": "SPXW240222P5820", "strike": 5820.0, "option_type": "put",
         "delta": -0.12, "gamma": 0.002, "theta": -0.85, "vega": 0.15,
         "bid": 1.80, "ask": 2.00, "last": 1.90, "volume": 1600, "open_interest": 3400},
        {"symbol": "SPXW240222P5810", "strike": 5810.0, "option_type": "put",
         "delta": -0.10, "gamma": 0.0018, "theta": -0.75, "vega": 0.12,
         "bid": 1.20, "ask": 1.40, "last": 1.30, "volume": 2200, "open_interest": 4700},
        {"symbol": "SPXW240222P5800", "strike": 5800.0, "option_type": "put",
         "delta": -0.08, "gamma": 0.0015, "theta": -0.65, "vega": 0.10,
         "bid": 0.80, "ask": 1.00, "last": 0.90, "volume": 1900, "open_interest": 4000},
        {"symbol": "SPXW240222P5790", "strike": 5790.0, "option_type": "put",
         "delta": -0.06, "gamma": 0.0012, "theta": -0.55, "vega": 0.08,
         "bid": 0.50, "ask": 0.70, "last": 0.60, "volume": 1300, "open_interest": 3100},
        {"symbol": "SPXW240222P5780", "strike": 5780.0, "option_type": "put",
         "delta": -0.04, "gamma": 0.0010, "theta": -0.45, "vega": 0.06,
         "bid": 0.30, "ask": 0.50, "last": 0.40, "volume": 1000, "open_interest": 2300},
    ]


@pytest.fixture
def sample_5min_candles() -> List[Dict[str, Any]]:
    """Sample 5-minute SPX candles for stability detection."""
    base_time = datetime(2024, 2, 22, 10, 0, 0)
    base_price = 5850.0
    return [
        {"timestamp": base_time - timedelta(minutes=25), "open": 5848.0, "high": 5852.0, "low": 5847.0, "close": 5850.0, "volume": 125000},
        {"timestamp": base_time - timedelta(minutes=20), "open": 5850.0, "high": 5853.0, "low": 5849.0, "close": 5851.0, "volume": 118000},
        {"timestamp": base_time - timedelta(minutes=15), "open": 5851.0, "high": 5854.0, "low": 5850.0, "close": 5850.5, "volume": 132000},
        {"timestamp": base_time - timedelta(minutes=10), "open": 5850.5, "high": 5852.0, "low": 5849.5, "close": 5850.0, "volume": 145000},
        {"timestamp": base_time - timedelta(minutes=5), "open": 5850.0, "high": 5851.5, "low": 5849.0, "close": 5850.25, "volume": 138000},
        {"timestamp": base_time, "open": 5850.25, "high": 5852.0, "low": 5849.5, "close": 5850.50, "volume": 142000},
    ]


@pytest.fixture
def sample_volatile_candles() -> List[Dict[str, Any]]:
    """Sample volatile 5-minute candles (should NOT trigger entry)."""
    base_time = datetime(2024, 2, 22, 10, 0, 0)
    return [
        {"timestamp": base_time - timedelta(minutes=25), "open": 5820.0, "high": 5825.0, "low": 5815.0, "close": 5823.0, "volume": 250000},
        {"timestamp": base_time - timedelta(minutes=20), "open": 5823.0, "high": 5835.0, "low": 5820.0, "close": 5832.0, "volume": 280000},
        {"timestamp": base_time - timedelta(minutes=15), "open": 5832.0, "high": 5845.0, "low": 5830.0, "close": 5842.0, "volume": 310000},
        {"timestamp": base_time - timedelta(minutes=10), "open": 5842.0, "high": 5855.0, "low": 5840.0, "close": 5852.0, "volume": 295000},
        {"timestamp": base_time - timedelta(minutes=5), "open": 5852.0, "high": 5865.0, "low": 5850.0, "close": 5862.0, "volume": 320000},
        {"timestamp": base_time, "open": 5862.0, "high": 5875.0, "low": 5860.0, "close": 5872.0, "volume": 340000},
    ]


@pytest.fixture
def sample_account_info() -> Dict[str, Any]:
    """Sample account information."""
    return {
        "account_number": "TEST123456",
        "equity": 100000.0,
        "buying_power": 80000.0,
        "cash_balance": 50000.0,
        "day_trading_buying_power": 200000.0,
        "maintenance_requirement": 20000.0,
    }


@pytest.fixture
def sample_iron_condor_trade() -> Dict[str, Any]:
    """Sample iron condor trade data."""
    return {
        "trade_id": "IC-20240222-001",
        "entry_time": datetime(2024, 2, 22, 9, 45, 0),
        "expiration_date": date(2024, 2, 22),
        "underlying": "SPX",
        "underlying_price_at_entry": 5850.25,
        "short_call_strike": 5890.0,
        "long_call_strike": 5920.0,
        "short_put_strike": 5810.0,
        "long_put_strike": 5780.0,
        "call_spread_premium": 1.50,
        "put_spread_premium": 1.50,
        "total_premium": 3.00,
        "call_wing_width": 30,
        "put_wing_width": 30,
        "status": "open",
    }


@pytest.fixture
def sample_llm_decision_enter() -> Dict[str, Any]:
    """Sample LLM decision to enter a trade."""
    return {
        "action": "enter_trade",
        "timing": "now",
        "order_sequence": ["call_spread", "put_spread"],
        "short_call_strike": 5890.0,
        "long_call_strike": 5920.0,
        "short_put_strike": 5810.0,
        "long_put_strike": 5780.0,
        "call_wing_width": 30,
        "put_wing_width": 30,
        "expected_call_premium": 1.50,
        "expected_put_premium": 1.50,
        "stop_loss_amount": 3.00,
        "reasoning": "Market stable, 3 consecutive candles at same level, delta targets met",
    }


@pytest.fixture
def sample_llm_decision_wait() -> Dict[str, Any]:
    """Sample LLM decision to wait (no trade)."""
    return {
        "action": "wait",
        "timing": None,
        "order_sequence": None,
        "reasoning": "Market too volatile, large directional move in progress",
    }


@pytest.fixture
def mock_dxlink_streamer():
    """Mock DXLink streamer for market data."""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value=True)
    mock.subscribe = AsyncMock()
    mock.fetch_option_quotes_batch = AsyncMock()
    mock.get_quote = AsyncMock()
    mock.disconnect = AsyncMock()
    return mock


@pytest.fixture
def mock_tastytrade_client():
    """Mock TastyTrade API client."""
    mock = AsyncMock()
    mock.authenticate = AsyncMock(return_value=True)
    mock.get_account_info = AsyncMock()
    mock.get_option_chain = AsyncMock()
    mock.place_order = AsyncMock()
    mock.cancel_order = AsyncMock()
    mock.get_positions = AsyncMock()
    return mock


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = MagicMock()
    mock.chat = MagicMock()
    mock.chat.completions = MagicMock()
    mock.chat.completions.create = MagicMock()
    return mock
