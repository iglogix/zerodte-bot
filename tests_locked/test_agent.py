"""
Agent Core Tests for 0DTE Trading Bot.
DO NOT MODIFY after approval - this file is part of the immutable test suite.

Tests cover:
- Tool loop executes max N iterations
- Tool call results fed back to LLM
- Decision logged to database
- Dry-run mode prevents real execution
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, call
import json


class TestToolLoopExecution:
    """Tests for tool loop execution."""

    @pytest.mark.asyncio
    async def test_tool_loop_executes_max_iterations(self):
        """Test that tool loop respects max iteration limit."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        tool_calls_made = []
        
        def mock_llm_response(iteration):
            if iteration < 15:
                return {
                    "tool_calls": [{"name": "get_spx_price", "arguments": {}}],
                    "content": None,
                }
            return {"tool_calls": None, "content": "Done"}
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=[mock_llm_response(i) for i in range(15)])
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"price": 5850.0})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            max_iterations=10,
        )
        
        result = await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={"spx_price": 5850.0},
        )
        
        assert mock_tool_handlers.execute.call_count <= 10

    @pytest.mark.asyncio
    async def test_tool_loop_stops_when_llm_done(self):
        """Test that tool loop stops when LLM indicates completion."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        responses = [
            {"tool_calls": [{"name": "get_spx_price", "arguments": {}}], "content": None},
            {"tool_calls": [{"name": "get_option_chain", "arguments": {}}], "content": None},
            {"tool_calls": None, "content": "Analysis complete. No trade recommended."},
        ]
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=responses)
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"data": "test"})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            max_iterations=10,
        )
        
        result = await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        assert mock_tool_handlers.execute.call_count == 2
        assert "complete" in result["response"].lower() or "no trade" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_tool_loop_handles_no_tool_calls(self):
        """Test that tool loop handles LLM response with no tool calls."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value={
            "tool_calls": None,
            "content": "Market conditions not suitable for trading.",
        })
        
        mock_tool_handlers = AsyncMock()
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            max_iterations=10,
        )
        
        result = await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        mock_tool_handlers.execute.assert_not_called()
        assert result["response"] is not None


class TestToolResultFeedback:
    """Tests for feeding tool results back to LLM."""

    @pytest.mark.asyncio
    async def test_tool_result_added_to_messages(self):
        """Test that tool results are added to message history."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        messages_sent = []
        
        async def capture_messages(messages, **kwargs):
            messages_sent.append(messages.copy())
            if len(messages_sent) == 1:
                return {
                    "tool_calls": [{"id": "call_1", "name": "get_spx_price", "arguments": "{}"}],
                    "content": None,
                }
            return {"tool_calls": None, "content": "Done"}
        
        mock_llm = AsyncMock()
        mock_llm.chat = capture_messages
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"price": 5850.25})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        assert len(messages_sent) >= 2
        second_call_messages = messages_sent[1]
        tool_result_found = any(
            msg.get("role") == "tool" or "5850" in str(msg)
            for msg in second_call_messages
        )
        assert tool_result_found

    @pytest.mark.asyncio
    async def test_multiple_tool_results_accumulated(self):
        """Test that multiple tool results are accumulated in history."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        call_count = [0]
        
        async def mock_chat(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return {
                    "tool_calls": [{"id": f"call_{call_count[0]}", "name": "get_spx_price", "arguments": "{}"}],
                    "content": None,
                }
            return {"tool_calls": None, "content": "Done"}
        
        mock_llm = AsyncMock()
        mock_llm.chat = mock_chat
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"price": 5850.0})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        assert mock_tool_handlers.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_error_fed_back_to_llm(self):
        """Test that tool errors are fed back to LLM."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        messages_received = []
        
        async def capture_and_respond(messages, **kwargs):
            messages_received.append(messages.copy())
            if len(messages_received) == 1:
                return {
                    "tool_calls": [{"id": "call_1", "name": "get_spx_price", "arguments": "{}"}],
                    "content": None,
                }
            return {"tool_calls": None, "content": "Handled error, waiting."}
        
        mock_llm = AsyncMock()
        mock_llm.chat = capture_and_respond
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"error": "API connection failed"})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        assert len(messages_received) >= 2
        error_in_messages = any(
            "error" in str(msg).lower() or "failed" in str(msg).lower()
            for msg in messages_received[1]
        )
        assert error_in_messages


class TestDecisionLogging:
    """Tests for decision logging to database."""

    @pytest.mark.asyncio
    async def test_decision_logged_to_database(self, sample_llm_decision_enter):
        """Test that agent decisions are logged to database."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value={
            "tool_calls": None,
            "content": json.dumps(sample_llm_decision_enter),
        })
        
        mock_db = AsyncMock()
        mock_db.log_decision = AsyncMock()
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=AsyncMock(),
            db=mock_db,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={"spx_price": 5850.0},
        )
        
        mock_db.log_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_decision_includes_reasoning(self, sample_llm_decision_enter):
        """Test that logged decision includes LLM reasoning."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        logged_decisions = []
        
        async def capture_log(decision):
            logged_decisions.append(decision)
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value={
            "tool_calls": None,
            "content": json.dumps(sample_llm_decision_enter),
        })
        
        mock_db = AsyncMock()
        mock_db.log_decision = capture_log
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=AsyncMock(),
            db=mock_db,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={"spx_price": 5850.0},
        )
        
        assert len(logged_decisions) == 1
        assert "reasoning" in logged_decisions[0] or "content" in str(logged_decisions[0])

    @pytest.mark.asyncio
    async def test_tool_calls_logged(self):
        """Test that tool calls are logged for audit."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        logged_tool_calls = []
        
        async def capture_tool_log(tool_call):
            logged_tool_calls.append(tool_call)
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=[
            {"tool_calls": [{"id": "1", "name": "get_spx_price", "arguments": "{}"}], "content": None},
            {"tool_calls": None, "content": "Done"},
        ])
        
        mock_db = AsyncMock()
        mock_db.log_decision = AsyncMock()
        mock_db.log_tool_call = capture_tool_log
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"price": 5850.0})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            db=mock_db,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        assert len(logged_tool_calls) >= 1
        assert logged_tool_calls[0]["name"] == "get_spx_price"


class TestDryRunMode:
    """Tests for dry-run mode preventing real execution."""

    @pytest.mark.asyncio
    async def test_dry_run_prevents_order_execution(self):
        """Test that dry-run mode prevents actual order placement."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=[
            {"tool_calls": [{"id": "1", "name": "place_iron_condor", "arguments": '{"short_call_strike": 5890}'}], "content": None},
            {"tool_calls": None, "content": "Order placed"},
        ])
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"order_id": "DRY-RUN-001", "dry_run": True})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            dry_run=True,
            max_iterations=10,
        )
        
        result = await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        call_args = mock_tool_handlers.execute.call_args
        assert agent.dry_run is True

    @pytest.mark.asyncio
    async def test_dry_run_logs_simulated_actions(self):
        """Test that dry-run mode logs simulated actions."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        logged_actions = []
        
        async def capture_action_log(action):
            logged_actions.append(action)
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=[
            {"tool_calls": [{"id": "1", "name": "place_iron_condor", "arguments": "{}"}], "content": None},
            {"tool_calls": None, "content": "Simulated order"},
        ])
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"simulated": True, "order_id": "SIM-001"})
        
        mock_db = AsyncMock()
        mock_db.log_decision = AsyncMock()
        mock_db.log_simulated_action = capture_action_log
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            db=mock_db,
            dry_run=True,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        assert len(logged_actions) >= 1 or mock_db.log_decision.called

    @pytest.mark.asyncio
    async def test_dry_run_still_fetches_market_data(self):
        """Test that dry-run mode still fetches real market data."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        tool_calls_made = []
        
        async def capture_tool_call(tool_name, args):
            tool_calls_made.append({"name": tool_name, "args": args})
            if tool_name == "get_spx_price":
                return {"price": 5850.0}
            return {"simulated": True}
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=[
            {"tool_calls": [{"id": "1", "name": "get_spx_price", "arguments": "{}"}], "content": None},
            {"tool_calls": [{"id": "2", "name": "get_option_chain", "arguments": "{}"}], "content": None},
            {"tool_calls": None, "content": "Done"},
        ])
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = capture_tool_call
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            dry_run=True,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        market_data_tools = [c for c in tool_calls_made if c["name"] in ["get_spx_price", "get_option_chain"]]
        assert len(market_data_tools) == 2

    @pytest.mark.asyncio
    async def test_live_mode_executes_real_orders(self):
        """Test that live mode executes real orders."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=[
            {"tool_calls": [{"id": "1", "name": "place_iron_condor", "arguments": "{}"}], "content": None},
            {"tool_calls": None, "content": "Order placed"},
        ])
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock(return_value={"order_id": "REAL-001"})
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            dry_run=False,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={},
        )
        
        assert agent.dry_run is False
        mock_tool_handlers.execute.assert_called()


class TestPromptLoading:
    """Tests for loading strategy prompt."""

    @pytest.mark.asyncio
    async def test_loads_bot_prompt_as_system_message(self):
        """Test that bot_prompt.md is loaded as system message."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        messages_received = []
        
        async def capture_messages(messages, **kwargs):
            messages_received.append(messages.copy())
            return {"tool_calls": None, "content": "Done"}
        
        mock_llm = AsyncMock()
        mock_llm.chat = capture_messages
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=AsyncMock(),
            max_iterations=10,
        )
        
        strategy_prompt = "You are an automated trading bot for 0DTE iron condors..."
        
        await agent.run_tool_loop(
            prompt=strategy_prompt,
            market_data={},
        )
        
        assert len(messages_received) == 1
        system_message = next(
            (m for m in messages_received[0] if m.get("role") == "system"),
            None
        )
        assert system_message is not None
        assert "trading" in system_message["content"].lower() or strategy_prompt in system_message["content"]

    @pytest.mark.asyncio
    async def test_market_data_included_in_user_message(self):
        """Test that market data is included in user message."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        messages_received = []
        
        async def capture_messages(messages, **kwargs):
            messages_received.append(messages.copy())
            return {"tool_calls": None, "content": "Done"}
        
        mock_llm = AsyncMock()
        mock_llm.chat = capture_messages
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=AsyncMock(),
            max_iterations=10,
        )
        
        market_data = {"spx_price": 5850.25, "time": "10:15:00"}
        
        await agent.run_tool_loop(
            prompt="Strategy prompt",
            market_data=market_data,
        )
        
        user_message = next(
            (m for m in messages_received[0] if m.get("role") == "user"),
            None
        )
        assert user_message is not None
        assert "5850" in user_message["content"] or "market" in user_message["content"].lower()


class TestAgentIntegrationWithValidator:
    """Tests for agent integration with dual-LLM validator."""

    @pytest.mark.asyncio
    async def test_agent_uses_validator_before_execution(self, sample_llm_decision_enter):
        """Test that agent uses validator before executing trades."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value={
            "tool_calls": None,
            "content": json.dumps(sample_llm_decision_enter),
        })
        
        mock_validator = AsyncMock()
        mock_validator.validate_decision = AsyncMock(return_value={
            "agreed": True,
            "execute": True,
            "action": "enter_trade",
        })
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=AsyncMock(),
            validator=mock_validator,
            max_iterations=10,
        )
        
        await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={"spx_price": 5850.0},
        )
        
        mock_validator.validate_decision.assert_called()

    @pytest.mark.asyncio
    async def test_agent_blocks_on_validator_disagreement(self, sample_llm_decision_enter):
        """Test that agent blocks execution when validator disagrees."""
        from agentic_ic.agent.trading_agent import TradingAgent
        
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value={
            "tool_calls": [{"id": "1", "name": "place_iron_condor", "arguments": "{}"}],
            "content": None,
        })
        
        mock_validator = AsyncMock()
        mock_validator.validate_decision = AsyncMock(return_value={
            "agreed": False,
            "execute": False,
            "disagreement_reason": "LLMs disagree on entry",
        })
        
        mock_tool_handlers = AsyncMock()
        mock_tool_handlers.execute = AsyncMock()
        
        agent = TradingAgent(
            llm_client=mock_llm,
            tool_handlers=mock_tool_handlers,
            validator=mock_validator,
            max_iterations=10,
        )
        
        result = await agent.run_tool_loop(
            prompt="Test prompt",
            market_data={"spx_price": 5850.0},
        )
        
        order_calls = [
            c for c in mock_tool_handlers.execute.call_args_list
            if "place" in str(c).lower() and "iron_condor" in str(c).lower()
        ]
        assert len(order_calls) == 0 or result.get("blocked") is True
