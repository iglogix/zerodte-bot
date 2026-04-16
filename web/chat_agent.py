"""AI chat agent with bot management tools for the web dashboard."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from web.log_streamer import LogFilter, tail_lines

logger = logging.getLogger(__name__)

ACTION_TOOLS = {"start_bot", "stop_bot", "restart_bot", "close_position", "change_mode"}

_SYSTEM_PROMPT = (
    "You are an AI assistant for the 0DTE options trading bot dashboard. "
    "You can list running bots, view logs, check positions, review decisions, "
    "and manage bots (start/stop/restart). Use the provided tools to answer "
    "questions about the system. Be concise and helpful."
)


class ChatAgent:
    """LLM-powered chat agent with tool-calling for bot management."""

    def __init__(
        self,
        bot_manager: Any,
        db: Any = None,
        api_client: Any = None,
        model: str = "openai/gpt-4.1-mini",
    ) -> None:
        self.bot_manager = bot_manager
        self.db = db
        self.api_client = api_client
        self._model = model
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}

    def get_tool_definitions(self) -> List[Dict]:
        """Return tool definitions in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_bots",
                    "description": "List all bots with their status, mode, and config.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bot_logs",
                    "description": "Get recent log lines for a bot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "The bot ID."},
                            "n": {"type": "integer", "description": "Number of lines (default 50)."},
                        },
                        "required": ["bot_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_positions",
                    "description": "Get current open positions/trades.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_decisions",
                    "description": "Get decision log files for a given date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format (default today).",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_account_info",
                    "description": "Get trading account information (balance, buying power).",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "start_bot",
                    "description": "Start a new bot with a given config and mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_path": {"type": "string", "description": "Path to config YAML."},
                            "mode": {
                                "type": "string",
                                "description": "Bot mode: dry_run, manual, sandbox, live.",
                                "enum": ["dry_run", "manual", "sandbox", "live"],
                            },
                            "name": {"type": "string", "description": "Optional bot name."},
                        },
                        "required": ["config_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "stop_bot",
                    "description": "Stop a running bot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "The bot ID to stop."},
                        },
                        "required": ["bot_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restart_bot",
                    "description": "Restart a running bot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "The bot ID to restart."},
                        },
                        "required": ["bot_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "close_position",
                    "description": "Close an open position.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "position_id": {"type": "string", "description": "The position ID to close."},
                        },
                        "required": ["position_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "change_mode",
                    "description": "Change a bot's operating mode by stopping and restarting it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "The bot ID."},
                            "new_mode": {
                                "type": "string",
                                "description": "New mode: dry_run, manual, sandbox, live.",
                                "enum": ["dry_run", "manual", "sandbox", "live"],
                            },
                        },
                        "required": ["bot_id", "new_mode"],
                    },
                },
            },
        ]

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], confirmed: bool = False
    ) -> Dict[str, Any]:
        """Execute a tool, requiring confirmation for action tools."""
        if tool_name in ACTION_TOOLS and not confirmed:
            return {
                "needs_confirmation": True,
                "tool": tool_name,
                "args": args,
                "message": f"Are you sure you want to {tool_name.replace('_', ' ')}?",
            }

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return await handler(**args)
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    #  Read tools
    # ------------------------------------------------------------------ #

    async def _tool_list_bots(self) -> Dict[str, Any]:
        bots = self.bot_manager.list_bots()
        return {
            "bots": [
                {
                    "bot_id": b.bot_id,
                    "name": b.name,
                    "status": b.status,
                    "mode": b.mode,
                    "config_path": b.config_path,
                    "pid": b.pid,
                }
                for b in bots
            ]
        }

    async def _tool_get_bot_logs(self, bot_id: str, n: int = 50) -> Dict[str, Any]:
        bots = self.bot_manager.list_bots()
        bot = next((b for b in bots if b.bot_id == bot_id), None)
        if not bot:
            return {"error": f"Bot {bot_id} not found"}

        log_file = getattr(bot, "log_file", None)
        if not log_file:
            return {"error": "No log file for this bot"}

        lines = tail_lines(log_file, n=n)
        log_filter = LogFilter()
        filtered = [l for l in lines if log_filter.accept(l)]
        return {"bot_id": bot_id, "lines": filtered}

    async def _tool_get_positions(self) -> Dict[str, Any]:
        if not self.db:
            return {"error": "Database not configured"}
        try:
            trades = self.db.get_open_trades()
            return {"positions": [str(t) for t in trades]}
        except Exception as exc:
            return {"error": str(exc)}

    async def _tool_get_decisions(self, date: Optional[str] = None) -> Dict[str, Any]:
        from datetime import date as _date

        date_str = date or _date.today().isoformat()
        decisions_dir = Path("logs/decisions") / date_str
        if not decisions_dir.is_dir():
            return {"date": date_str, "files": [], "message": "No decisions found"}

        files = sorted([f.name for f in decisions_dir.glob("*.json")], reverse=True)
        # Read the most recent decision if available
        details = []
        for fname in files[:5]:  # limit to 5 most recent
            try:
                content = json.loads((decisions_dir / fname).read_text())
                details.append({"file": fname, "summary": _summarize_decision(content)})
            except (json.JSONDecodeError, OSError):
                details.append({"file": fname, "summary": "Failed to read"})

        return {"date": date_str, "count": len(files), "recent": details}

    async def _tool_get_account_info(self) -> Dict[str, Any]:
        if not self.api_client:
            return {"error": "API client not configured"}
        try:
            info = await self.api_client.get_account_info()
            return {"account": info}
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    #  Action tools
    # ------------------------------------------------------------------ #

    async def _tool_start_bot(
        self, config_path: str, mode: str = "dry_run", name: Optional[str] = None
    ) -> Dict[str, Any]:
        bot_id = await self.bot_manager.start_bot(config_path, mode=mode, name=name)
        return {"started": True, "bot_id": bot_id}

    async def _tool_stop_bot(self, bot_id: str) -> Dict[str, Any]:
        result = await self.bot_manager.stop_bot(bot_id)
        return {"stopped": bool(result), "bot_id": bot_id}

    async def _tool_restart_bot(self, bot_id: str) -> Dict[str, Any]:
        new_id = await self.bot_manager.restart_bot(bot_id)
        return {"restarted": True, "new_bot_id": new_id}

    async def _tool_close_position(self, position_id: str) -> Dict[str, Any]:
        return {"error": "Not yet wired — close positions manually from the positions page."}

    async def _tool_change_mode(self, bot_id: str, new_mode: str) -> Dict[str, Any]:
        bots = self.bot_manager.list_bots()
        bot = next((b for b in bots if b.bot_id == bot_id), None)
        if not bot:
            return {"error": f"Bot {bot_id} not found"}

        await self.bot_manager.stop_bot(bot_id)
        new_id = await self.bot_manager.start_bot(
            bot.config_path, mode=new_mode, name=bot.name
        )
        return {"changed": True, "old_bot_id": bot_id, "new_bot_id": new_id, "mode": new_mode}

    # ------------------------------------------------------------------ #
    #  Chat loop
    # ------------------------------------------------------------------ #

    async def chat(self, user_message: str, session_id: str = "default") -> str:
        """Full tool-loop chat: call LLM with tools, execute calls, return text."""
        from src_v2.services.llm_client import LLMClient

        llm = LLMClient(model=self._model, temperature=0.3, max_tokens=2048)

        # Get or create conversation history
        if session_id not in self._conversations:
            self._conversations[session_id] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
            ]

        history = self._conversations[session_id]
        history.append({"role": "user", "content": user_message})

        tools = self.get_tool_definitions()

        for _ in range(5):  # max 5 tool-loop iterations
            response = await llm.chat(messages=history, tools=tools)

            tool_calls = response.get("tool_calls")
            content = response.get("content")

            if not tool_calls:
                # Final text response
                if content:
                    history.append({"role": "assistant", "content": content})
                return content or "I'm not sure how to help with that."

            # Build assistant message with tool calls
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                        if isinstance(tc["arguments"], str)
                        else json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ]
            history.append(assistant_msg)

            # Execute each tool call
            for tc in tool_calls:
                args = tc["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                result = await self.execute_tool(tc["name"], args, confirmed=False)

                # If action needs confirmation, return immediately
                if result.get("needs_confirmation"):
                    confirm_msg = (
                        f"I need your confirmation to **{tc['name'].replace('_', ' ')}**. "
                        f"{result['message']}"
                    )
                    history.append({"role": "assistant", "content": confirm_msg})
                    return confirm_msg

                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, default=str),
                })

        return "I've reached the maximum number of tool calls. Please try a simpler request."


def _summarize_decision(decision: Dict) -> str:
    """Create a one-line summary of a decision JSON."""
    outcome = decision.get("outcome", decision.get("action", "unknown"))
    reason = decision.get("reason", decision.get("reasoning", ""))
    if isinstance(reason, str) and len(reason) > 100:
        reason = reason[:100] + "..."
    return f"{outcome}: {reason}" if reason else str(outcome)
