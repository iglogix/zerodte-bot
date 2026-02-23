"""
Dual-LLM Validation Tests for 0DTE Trading Bot.
DO NOT MODIFY after approval - this file is part of the immutable test suite.

Tests cover:
- Both LLMs agree on entry decision → execute
- Both LLMs agree on timing → execute
- Both LLMs agree on order sequence (call spread first) → execute
- Both LLMs agree on stop-loss placement → execute
- ANY disagreement (entry, timing, sequence, stops) → block + notify
- Primary LLM fails → block + notify (no fallback for safety)
- Both LLMs fail → block + notify
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestBothLLMsAgreeOnEntry:
    """Tests for when both LLMs agree on entry decision."""

    @pytest.mark.asyncio
    async def test_both_agree_enter_trade_executes(self, sample_llm_decision_enter):
        """Test that trade executes when both LLMs agree to enter."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = sample_llm_decision_enter.copy()
        secondary_decision = sample_llm_decision_enter.copy()
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is True
        assert result["action"] == "enter_trade"
        assert result["execute"] is True

    @pytest.mark.asyncio
    async def test_both_agree_wait_does_not_execute(self, sample_llm_decision_wait):
        """Test that no trade executes when both LLMs agree to wait."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = sample_llm_decision_wait.copy()
        secondary_decision = sample_llm_decision_wait.copy()
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is True
        assert result["action"] == "wait"
        assert result["execute"] is False


class TestBothLLMsAgreeOnTiming:
    """Tests for when both LLMs agree on trade timing."""

    @pytest.mark.asyncio
    async def test_both_agree_timing_now_executes(self, sample_llm_decision_enter):
        """Test that trade executes when both agree on 'now' timing."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "timing": "now"}
        secondary_decision = {**sample_llm_decision_enter, "timing": "now"}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is True
        assert result["timing_agreed"] is True

    @pytest.mark.asyncio
    async def test_timing_disagreement_blocks(self, sample_llm_decision_enter):
        """Test that timing disagreement blocks execution."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "timing": "now"}
        secondary_decision = {**sample_llm_decision_enter, "timing": "wait_5_min"}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is False
        assert result["execute"] is False
        assert "timing" in result["disagreement_reason"].lower()
        mock_notifier.notify_disagreement.assert_called_once()


class TestBothLLMsAgreeOnOrderSequence:
    """Tests for when both LLMs agree on order sequence."""

    @pytest.mark.asyncio
    async def test_both_agree_call_spread_first_executes(self, sample_llm_decision_enter):
        """Test that trade executes when both agree call spread first."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "order_sequence": ["call_spread", "put_spread"]}
        secondary_decision = {**sample_llm_decision_enter, "order_sequence": ["call_spread", "put_spread"]}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is True
        assert result["order_sequence_agreed"] is True
        assert result["order_sequence"] == ["call_spread", "put_spread"]

    @pytest.mark.asyncio
    async def test_order_sequence_disagreement_blocks(self, sample_llm_decision_enter):
        """Test that order sequence disagreement blocks execution."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "order_sequence": ["call_spread", "put_spread"]}
        secondary_decision = {**sample_llm_decision_enter, "order_sequence": ["put_spread", "call_spread"]}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is False
        assert result["execute"] is False
        assert "sequence" in result["disagreement_reason"].lower()
        mock_notifier.notify_disagreement.assert_called_once()


class TestBothLLMsAgreeOnStopLoss:
    """Tests for when both LLMs agree on stop-loss placement."""

    @pytest.mark.asyncio
    async def test_both_agree_stop_loss_executes(self, sample_llm_decision_enter):
        """Test that trade executes when both agree on stop-loss."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "stop_loss_amount": 3.00}
        secondary_decision = {**sample_llm_decision_enter, "stop_loss_amount": 3.00}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is True
        assert result["stop_loss_agreed"] is True

    @pytest.mark.asyncio
    async def test_stop_loss_disagreement_blocks(self, sample_llm_decision_enter):
        """Test that stop-loss disagreement blocks execution."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "stop_loss_amount": 3.00}
        secondary_decision = {**sample_llm_decision_enter, "stop_loss_amount": 4.50}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is False
        assert result["execute"] is False
        assert "stop" in result["disagreement_reason"].lower()
        mock_notifier.notify_disagreement.assert_called_once()


class TestAnyDisagreementBlocks:
    """Tests for any disagreement blocking execution."""

    @pytest.mark.asyncio
    async def test_entry_disagreement_blocks(self, sample_llm_decision_enter, sample_llm_decision_wait):
        """Test that entry decision disagreement blocks execution."""
        from src.agent.dual_validator import DualLLMValidator
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=sample_llm_decision_enter)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=sample_llm_decision_wait)
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is False
        assert result["execute"] is False
        assert "entry" in result["disagreement_reason"].lower() or "action" in result["disagreement_reason"].lower()
        mock_notifier.notify_disagreement.assert_called_once()

    @pytest.mark.asyncio
    async def test_strike_disagreement_blocks(self, sample_llm_decision_enter):
        """Test that strike price disagreement blocks execution."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "short_call_strike": 5890.0}
        secondary_decision = {**sample_llm_decision_enter, "short_call_strike": 5900.0}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["agreed"] is False
        assert result["execute"] is False
        mock_notifier.notify_disagreement.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_sent_on_any_disagreement(self, sample_llm_decision_enter):
        """Test that notification is sent on any disagreement."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_decision = {**sample_llm_decision_enter, "timing": "now"}
        secondary_decision = {**sample_llm_decision_enter, "timing": "wait"}
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=primary_decision)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=secondary_decision)
        
        mock_notifier = AsyncMock()
        mock_notifier.notify_disagreement = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        mock_notifier.notify_disagreement.assert_called_once()
        call_args = mock_notifier.notify_disagreement.call_args
        assert "primary_decision" in call_args.kwargs or len(call_args.args) > 0


class TestPrimaryLLMFailure:
    """Tests for primary LLM failure handling."""

    @pytest.mark.asyncio
    async def test_primary_failure_blocks_no_fallback(self, sample_llm_decision_enter):
        """Test that primary LLM failure blocks execution (no fallback for safety)."""
        from src.agent.dual_validator import DualLLMValidator
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(side_effect=Exception("Primary LLM API Error"))
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=sample_llm_decision_enter)
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["execute"] is False
        assert result["error"] is True
        assert "primary" in result["error_reason"].lower()
        mock_notifier.notify_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_primary_timeout_blocks(self, sample_llm_decision_enter):
        """Test that primary LLM timeout blocks execution."""
        from src.agent.dual_validator import DualLLMValidator
        import asyncio
        
        async def slow_response():
            await asyncio.sleep(100)
            return sample_llm_decision_enter
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = slow_response
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=sample_llm_decision_enter)
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
            timeout_seconds=0.1,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["execute"] is False
        assert result["error"] is True


class TestBothLLMsFailure:
    """Tests for both LLMs failing."""

    @pytest.mark.asyncio
    async def test_both_fail_blocks_and_notifies(self):
        """Test that both LLMs failing blocks execution and notifies."""
        from src.agent.dual_validator import DualLLMValidator
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(side_effect=Exception("Primary API Error"))
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(side_effect=Exception("Secondary API Error"))
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["execute"] is False
        assert result["error"] is True
        assert "both" in result["error_reason"].lower() or "primary" in result["error_reason"].lower()
        mock_notifier.notify_error.assert_called()

    @pytest.mark.asyncio
    async def test_both_return_invalid_response_blocks(self):
        """Test that both LLMs returning invalid responses blocks execution."""
        from src.agent.dual_validator import DualLLMValidator
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value={"invalid": "response"})
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value={"also_invalid": "response"})
        
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            notifier=mock_notifier,
        )
        
        result = await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        assert result["execute"] is False
        assert result["error"] is True or result["agreed"] is False


class TestSecondaryLLMIndependence:
    """Tests for secondary LLM independence (virgin eyes)."""

    @pytest.mark.asyncio
    async def test_secondary_receives_same_data_no_primary_output(self, sample_llm_decision_enter):
        """Test that secondary LLM receives same data but not primary's output."""
        from src.agent.dual_validator import DualLLMValidator
        
        primary_calls = []
        secondary_calls = []
        
        async def capture_primary(market_data, prompt):
            primary_calls.append({"market_data": market_data, "prompt": prompt})
            return sample_llm_decision_enter
        
        async def capture_secondary(market_data, prompt):
            secondary_calls.append({"market_data": market_data, "prompt": prompt})
            return sample_llm_decision_enter
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = capture_primary
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = capture_secondary
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
        )
        
        market_data = {"spx_price": 5850.0, "candles": [1, 2, 3]}
        prompt = "bot_prompt.md content"
        
        await validator.validate_decision(market_data=market_data, prompt=prompt)
        
        assert len(primary_calls) == 1
        assert len(secondary_calls) == 1
        assert primary_calls[0]["market_data"] == secondary_calls[0]["market_data"]
        assert primary_calls[0]["prompt"] == secondary_calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_llms_called_in_parallel(self, sample_llm_decision_enter):
        """Test that both LLMs are called in parallel for efficiency."""
        from src.agent.dual_validator import DualLLMValidator
        import asyncio
        import time
        
        call_times = {"primary": None, "secondary": None}
        
        async def primary_with_delay(market_data, prompt):
            call_times["primary"] = time.time()
            await asyncio.sleep(0.1)
            return sample_llm_decision_enter
        
        async def secondary_with_delay(market_data, prompt):
            call_times["secondary"] = time.time()
            await asyncio.sleep(0.1)
            return sample_llm_decision_enter
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = primary_with_delay
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = secondary_with_delay
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
        )
        
        start = time.time()
        await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        elapsed = time.time() - start
        
        assert elapsed < 0.25
        assert abs(call_times["primary"] - call_times["secondary"]) < 0.05


class TestDecisionLogging:
    """Tests for decision audit logging."""

    @pytest.mark.asyncio
    async def test_agreement_logged(self, sample_llm_decision_enter):
        """Test that agreement decisions are logged."""
        from src.agent.dual_validator import DualLLMValidator
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=sample_llm_decision_enter)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=sample_llm_decision_enter)
        
        mock_logger = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            decision_logger=mock_logger,
        )
        
        await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        mock_logger.log_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_disagreement_logged_with_details(self, sample_llm_decision_enter, sample_llm_decision_wait):
        """Test that disagreement decisions are logged with details."""
        from src.agent.dual_validator import DualLLMValidator
        
        mock_primary = AsyncMock()
        mock_primary.get_decision = AsyncMock(return_value=sample_llm_decision_enter)
        
        mock_secondary = AsyncMock()
        mock_secondary.get_decision = AsyncMock(return_value=sample_llm_decision_wait)
        
        mock_logger = AsyncMock()
        mock_notifier = AsyncMock()
        
        validator = DualLLMValidator(
            primary_llm=mock_primary,
            secondary_llm=mock_secondary,
            decision_logger=mock_logger,
            notifier=mock_notifier,
        )
        
        await validator.validate_decision(
            market_data={"spx_price": 5850.0},
            prompt="bot_prompt.md content",
        )
        
        mock_logger.log_decision.assert_called_once()
        call_args = mock_logger.log_decision.call_args
        logged_data = call_args.kwargs if call_args.kwargs else call_args.args[0]
        assert "primary" in str(logged_data).lower() or "secondary" in str(logged_data).lower()
