"""
Risk Management Tests for 0DTE Trading Bot.
DO NOT MODIFY after approval - this file is part of the immutable test suite.

Tests cover:
- Max 10 concurrent ICs enforced
- Daily risk limit (1-2% account) enforced
- Buying power (50%) enforced
- Stop-loss calculation matches prompt rules
"""

import pytest
from datetime import datetime, date
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestMaxConcurrentTrades:
    """Tests for max 10 concurrent iron condors limit."""

    @pytest.mark.asyncio
    async def test_allows_trade_when_under_limit(self):
        """Test that new trade is allowed when under 10 concurrent ICs."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trade_count = AsyncMock(return_value=5)
        
        risk_manager = RiskManager(db=mock_db, max_concurrent_trades=10)
        
        result = await risk_manager.can_open_new_trade()
        
        assert result["allowed"] is True
        assert result["current_count"] == 5
        assert result["max_allowed"] == 10

    @pytest.mark.asyncio
    async def test_blocks_trade_at_limit(self):
        """Test that new trade is blocked when at 10 concurrent ICs."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trade_count = AsyncMock(return_value=10)
        
        risk_manager = RiskManager(db=mock_db, max_concurrent_trades=10)
        
        result = await risk_manager.can_open_new_trade()
        
        assert result["allowed"] is False
        assert "max" in result["reason"].lower() or "limit" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_blocks_trade_over_limit(self):
        """Test that new trade is blocked when over 10 concurrent ICs."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trade_count = AsyncMock(return_value=12)
        
        risk_manager = RiskManager(db=mock_db, max_concurrent_trades=10)
        
        result = await risk_manager.can_open_new_trade()
        
        assert result["allowed"] is False

    @pytest.mark.asyncio
    async def test_count_only_open_trades(self):
        """Test that only open (not closed) trades are counted."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trade_count = AsyncMock(return_value=3)
        mock_db.get_total_trade_count = AsyncMock(return_value=15)
        
        risk_manager = RiskManager(db=mock_db, max_concurrent_trades=10)
        
        result = await risk_manager.can_open_new_trade()
        
        assert result["allowed"] is True
        assert result["current_count"] == 3


class TestDailyRiskLimit:
    """Tests for daily risk limit (1-2% of account)."""

    @pytest.mark.asyncio
    async def test_allows_trade_within_risk_limit(self, sample_account_info):
        """Test that trade is allowed when within daily risk limit."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[])
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_daily_risk_percent=2.0,
        )
        
        new_trade_risk = 300.0
        
        result = await risk_manager.check_daily_risk_limit(new_trade_risk)
        
        assert result["allowed"] is True
        assert result["current_risk"] == 0
        assert result["max_risk"] == sample_account_info["equity"] * 0.02

    @pytest.mark.asyncio
    async def test_blocks_trade_exceeding_risk_limit(self, sample_account_info):
        """Test that trade is blocked when exceeding daily risk limit."""
        from agentic_ic.services.risk_manager import RiskManager
        
        existing_trades = [
            {"total_premium": 3.00, "status": "open"},
            {"total_premium": 3.00, "status": "open"},
            {"total_premium": 3.00, "status": "open"},
        ]
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=existing_trades)
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_daily_risk_percent=2.0,
        )
        
        new_trade_risk = 1500.0
        
        result = await risk_manager.check_daily_risk_limit(new_trade_risk)
        
        assert result["allowed"] is False
        assert "risk" in result["reason"].lower() or "limit" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_calculates_worst_case_double_stop(self, sample_account_info):
        """Test that worst-case risk assumes double stop-loss (both sides trigger)."""
        from agentic_ic.services.risk_manager import RiskManager
        
        existing_trades = [
            {"total_premium": 3.00, "status": "open"},
        ]
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=existing_trades)
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_daily_risk_percent=2.0,
        )
        
        current_risk = await risk_manager.calculate_current_risk()
        
        assert current_risk == 3.00 * 2 * 100

    @pytest.mark.asyncio
    async def test_halts_new_entries_approaching_limit(self, sample_account_info):
        """Test that new entries are halted when approaching risk limit."""
        from agentic_ic.services.risk_manager import RiskManager
        
        max_risk = sample_account_info["equity"] * 0.02
        existing_risk = max_risk * 0.95
        
        existing_trades = [
            {"total_premium": existing_risk / 200, "status": "open"},
        ]
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=existing_trades)
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_daily_risk_percent=2.0,
        )
        
        new_trade_risk = 300.0
        
        result = await risk_manager.check_daily_risk_limit(new_trade_risk)
        
        assert result["allowed"] is False or result.get("warning") is True


class TestBuyingPowerLimit:
    """Tests for buying power (50%) limit."""

    @pytest.mark.asyncio
    async def test_allows_trade_within_buying_power(self, sample_account_info):
        """Test that trade is allowed when within 50% buying power limit."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[])
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_buying_power_percent=50.0,
        )
        
        required_bp = 20000.0
        
        result = await risk_manager.check_buying_power(required_bp)
        
        assert result["allowed"] is True
        assert result["available"] >= required_bp

    @pytest.mark.asyncio
    async def test_blocks_trade_exceeding_buying_power(self, sample_account_info):
        """Test that trade is blocked when exceeding 50% buying power limit."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[])
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_buying_power_percent=50.0,
        )
        
        required_bp = 60000.0
        
        result = await risk_manager.check_buying_power(required_bp)
        
        assert result["allowed"] is False
        assert "50%" in result["reason"] or "buying power" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_accounts_for_existing_positions(self, sample_account_info):
        """Test that existing positions are accounted for in buying power."""
        from agentic_ic.services.risk_manager import RiskManager
        
        existing_trades = [
            {"buying_power_used": 15000.0, "status": "open"},
            {"buying_power_used": 15000.0, "status": "open"},
        ]
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=existing_trades)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_buying_power_percent=50.0,
        )
        
        required_bp = 25000.0
        
        result = await risk_manager.check_buying_power(required_bp)
        
        assert result["allowed"] is False


class TestStopLossCalculation:
    """Tests for stop-loss calculation matching prompt rules."""

    @pytest.mark.asyncio
    async def test_stop_loss_equals_total_premium(self):
        """Test that stop-loss amount equals total IC premium collected."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        total_premium = 3.00
        
        stop_loss = risk_manager.calculate_stop_loss_amount(total_premium)
        
        assert stop_loss["short_call_stop"] == total_premium * 100
        assert stop_loss["short_put_stop"] == total_premium * 100

    @pytest.mark.asyncio
    async def test_stop_loss_separate_for_each_side(self):
        """Test that stop-loss is calculated separately for each side."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        total_premium = 3.50
        
        stop_loss = risk_manager.calculate_stop_loss_amount(total_premium)
        
        assert "short_call_stop" in stop_loss
        assert "short_put_stop" in stop_loss
        assert stop_loss["short_call_stop"] == stop_loss["short_put_stop"]

    @pytest.mark.asyncio
    async def test_oco_order_structure(self):
        """Test that OCO order structure is correct (stop-limit + stop-market)."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        short_option_price = 1.50
        stop_loss_amount = 3.00
        
        oco_structure = risk_manager.create_oco_stop_structure(
            short_option_price=short_option_price,
            stop_loss_amount=stop_loss_amount,
        )
        
        assert "stop_limit" in oco_structure
        assert "stop_market" in oco_structure
        assert oco_structure["stop_limit"]["trigger_gap"] == 40
        assert oco_structure["stop_market"]["additional_gap"] == 30

    @pytest.mark.asyncio
    async def test_stop_limit_40_point_gap(self):
        """Test that stop-limit has 40-point gap between trigger and limit."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        oco_structure = risk_manager.create_oco_stop_structure(
            short_option_price=1.50,
            stop_loss_amount=3.00,
        )
        
        stop_limit = oco_structure["stop_limit"]
        gap = abs(stop_limit["limit_price"] - stop_limit["trigger_price"])
        
        assert gap == 40 or stop_limit["trigger_gap"] == 40

    @pytest.mark.asyncio
    async def test_stop_market_30_points_further(self):
        """Test that stop-market is 30 points further than stop-limit."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        oco_structure = risk_manager.create_oco_stop_structure(
            short_option_price=1.50,
            stop_loss_amount=3.00,
        )
        
        stop_limit_trigger = oco_structure["stop_limit"]["trigger_price"]
        stop_market_trigger = oco_structure["stop_market"]["trigger_price"]
        
        gap = abs(stop_market_trigger - stop_limit_trigger)
        assert gap == 30 or oco_structure["stop_market"]["additional_gap"] == 30


class TestTrailingStopAdjustment:
    """Tests for trailing stop adjustment as long options decay."""

    @pytest.mark.asyncio
    async def test_tightens_stop_as_long_decays(self):
        """Test that stop is tightened as long option loses value."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        original_long_value = 0.50
        current_long_value = 0.10
        original_stop_loss = 3.00
        
        new_stop = risk_manager.calculate_trailing_stop(
            original_stop_loss=original_stop_loss,
            original_long_value=original_long_value,
            current_long_value=current_long_value,
        )
        
        assert new_stop < original_stop_loss

    @pytest.mark.asyncio
    async def test_ensures_breakeven_if_triggered(self):
        """Test that trailing stop ensures break-even if triggered."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        total_premium_collected = 3.00
        current_long_value = 0.05
        
        new_stop = risk_manager.calculate_trailing_stop_for_breakeven(
            total_premium=total_premium_collected,
            current_long_value=current_long_value,
        )
        
        potential_loss = new_stop - total_premium_collected
        assert potential_loss <= 0 or abs(potential_loss) < 0.10

    @pytest.mark.asyncio
    async def test_no_adjustment_if_long_still_valuable(self):
        """Test that stop is not adjusted if long option still has value."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager()
        
        original_long_value = 0.50
        current_long_value = 0.45
        original_stop_loss = 3.00
        
        new_stop = risk_manager.calculate_trailing_stop(
            original_stop_loss=original_stop_loss,
            original_long_value=original_long_value,
            current_long_value=current_long_value,
        )
        
        assert new_stop == original_stop_loss or abs(new_stop - original_stop_loss) < 0.10


class TestTakeProfitRules:
    """Tests for take-profit rules."""

    @pytest.mark.asyncio
    async def test_close_at_5_cents(self):
        """Test that short option is closed when value decays to $0.05."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager(take_profit_threshold=0.05)
        
        current_price = 0.05
        
        should_close = risk_manager.should_take_profit(current_price)
        
        assert should_close is True

    @pytest.mark.asyncio
    async def test_no_close_above_5_cents(self):
        """Test that short option is not closed when above $0.05."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager(take_profit_threshold=0.05)
        
        current_price = 0.10
        
        should_close = risk_manager.should_take_profit(current_price)
        
        assert should_close is False

    @pytest.mark.asyncio
    async def test_close_below_5_cents(self):
        """Test that short option is closed when below $0.05."""
        from agentic_ic.services.risk_manager import RiskManager
        
        risk_manager = RiskManager(take_profit_threshold=0.05)
        
        current_price = 0.03
        
        should_close = risk_manager.should_take_profit(current_price)
        
        assert should_close is True


class TestRiskValidation:
    """Tests for comprehensive risk validation before trade entry."""

    @pytest.mark.asyncio
    async def test_validates_all_risk_checks(self, sample_account_info):
        """Test that all risk checks are performed before trade entry."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trade_count = AsyncMock(return_value=5)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_concurrent_trades=10,
            max_daily_risk_percent=2.0,
            max_buying_power_percent=50.0,
        )
        
        result = await risk_manager.validate_trade_entry(
            new_trade_risk=300.0,
            required_buying_power=5000.0,
        )
        
        assert "concurrent_trades_ok" in result
        assert "daily_risk_ok" in result
        assert "buying_power_ok" in result
        assert result["allowed"] is True

    @pytest.mark.asyncio
    async def test_fails_if_any_check_fails(self, sample_account_info):
        """Test that trade is blocked if any risk check fails."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trade_count = AsyncMock(return_value=10)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_concurrent_trades=10,
            max_daily_risk_percent=2.0,
            max_buying_power_percent=50.0,
        )
        
        result = await risk_manager.validate_trade_entry(
            new_trade_risk=300.0,
            required_buying_power=5000.0,
        )
        
        assert result["allowed"] is False
        assert "concurrent" in result["reason"].lower() or "max" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_returns_detailed_failure_reason(self, sample_account_info):
        """Test that detailed failure reason is returned."""
        from agentic_ic.services.risk_manager import RiskManager
        
        mock_db = AsyncMock()
        mock_db.get_open_trade_count = AsyncMock(return_value=10)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        
        mock_cache = AsyncMock()
        mock_cache.get_cached_account_info = AsyncMock(return_value=sample_account_info)
        
        risk_manager = RiskManager(
            db=mock_db,
            cache=mock_cache,
            max_concurrent_trades=10,
        )
        
        result = await risk_manager.validate_trade_entry(
            new_trade_risk=300.0,
            required_buying_power=5000.0,
        )
        
        assert "reason" in result
        assert len(result["reason"]) > 0
