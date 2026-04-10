"""Tests for the ChatAgent class."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from web.chat_agent import ChatAgent


@pytest.fixture
def mock_bot_manager():
    mgr = MagicMock()
    bot = MagicMock(bot_id="abc", status="running",
                     mode="manual", config_path="config/config_iron_fly_spx.yaml", pid=1234)
    bot.name = "iron_fly_spx"
    mgr.list_bots.return_value = [bot]
    mgr.start_bot = AsyncMock(return_value="def456")
    mgr.stop_bot = AsyncMock(return_value=True)
    mgr.restart_bot = AsyncMock(return_value="ghi789")
    return mgr


@pytest.fixture
def agent(mock_bot_manager):
    return ChatAgent(bot_manager=mock_bot_manager)


def test_tool_definitions(agent):
    tools = agent.get_tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert "list_bots" in names
    assert "stop_bot" in names
    assert "start_bot" in names
    assert "get_bot_logs" in names


@pytest.mark.asyncio
async def test_execute_list_bots(agent):
    result = await agent.execute_tool("list_bots", {})
    assert "bots" in result
    assert len(result["bots"]) == 1
    assert result["bots"][0]["name"] == "iron_fly_spx"


@pytest.mark.asyncio
async def test_execute_stop_bot_needs_confirmation(agent):
    result = await agent.execute_tool("stop_bot", {"bot_id": "abc"})
    assert result.get("needs_confirmation") is True


@pytest.mark.asyncio
async def test_execute_stop_bot_confirmed(agent, mock_bot_manager):
    result = await agent.execute_tool("stop_bot", {"bot_id": "abc"}, confirmed=True)
    assert result.get("stopped") is True
    mock_bot_manager.stop_bot.assert_awaited_once_with("abc")


@pytest.mark.asyncio
async def test_execute_unknown_tool(agent):
    result = await agent.execute_tool("nonexistent", {})
    assert "error" in result
