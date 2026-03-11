"""
Trading Agent core — runs the LLM tool loop.

The agent:
1. Loads the strategy prompt (bot_prompt.md) as system message
2. Injects current market data into the user message
3. Runs a tool loop: LLM calls tools → results fed back → repeat
4. Stops when LLM returns no tool calls or max_iterations reached
5. Logs every decision and tool call to the database
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    AI agentic trading bot core.
    The LLM reads the strategy prompt and controls all logic via tool calls.
    """

    def __init__(
        self,
        llm_client=None,
        tool_handlers=None,
        validator=None,
        db=None,
        dry_run: bool = True,
        max_iterations: int = 10,
        decision_file_logger=None,
        mode: str = "dry_run",
    ):
        self.llm_client = llm_client
        self.tool_handlers = tool_handlers
        self.validator = validator
        self.db = db
        self.dry_run = dry_run
        self.max_iterations = max_iterations
        self.decision_file_logger = decision_file_logger
        self.mode = mode

    async def run_tool_loop(
        self,
        prompt: str,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the LLM tool loop.

        Returns a dict with:
          - response: final LLM text response
          - tool_calls_made: list of tool calls executed
          - blocked: True if validator blocked execution
        """
        from src.agent.tool_definitions import TOOL_DEFINITIONS

        messages = self._build_initial_messages(prompt, market_data)
        tool_calls_made = []
        iteration = 0

        cycle = None
        if self.decision_file_logger:
            cycle = self.decision_file_logger.start_cycle(
                mode=self.mode,
                market_data=market_data,
            )

        while iteration < self.max_iterations:
            iteration += 1
            logger.info("Tool loop iteration %d/%d", iteration, self.max_iterations)

            llm_response = await self.llm_client.chat(
                messages,
                tools=TOOL_DEFINITIONS,
            )

            tool_calls = llm_response.get("tool_calls")
            content = llm_response.get("content")

            if not tool_calls:
                if self.validator and content:
                    decision = _parse_decision_from_content(content)
                    if decision and decision.get("action") == "enter_trade":
                        validation = await self.validator.validate_decision(
                            market_data=market_data,
                            prompt=prompt,
                            messages=messages,
                            proposed_trade=decision,
                        )
                        if cycle:
                            cycle.add_validator_result(
                                proposed_trade=decision,
                                primary_approval=validation.get("primary_decision"),
                                secondary_approval=validation.get("secondary_decision"),
                                outcome="blocked" if not validation.get("execute") else "approved",
                                reason=validation.get("disagreement_reason") or validation.get("error_reason"),
                            )
                        if not validation.get("execute", False):
                            await self._log_decision({"content": content, "blocked": True})
                            if cycle:
                                cycle.finish(outcome="blocked", response=content or "")
                            return {
                                "response": content or "",
                                "tool_calls_made": tool_calls_made,
                                "blocked": True,
                            }
                await self._log_decision({"content": content, "messages": messages})
                if cycle:
                    cycle.finish(outcome="waited", response=content or "")
                return {
                    "response": content or "",
                    "tool_calls_made": tool_calls_made,
                    "blocked": False,
                }

            assistant_msg = {"role": "assistant", "content": content or ""}
            if tool_calls:
                normalized = []
                for tc in tool_calls:
                    normalized.append({
                        "id": tc.get("id", f"call_{iteration}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name") or tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("arguments") or tc.get("function", {}).get("arguments", "{}"),
                        },
                    })
                assistant_msg["tool_calls"] = normalized
            messages.append(assistant_msg)

            for tc in tool_calls:
                tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                raw_args = tc.get("arguments") or tc.get("function", {}).get("arguments", "{}")
                tool_call_id = tc.get("id", f"call_{iteration}")

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}

                if self.validator and _is_execution_tool(tool_name):
                    validation = await self.validator.validate_decision(
                        market_data=market_data,
                        prompt=prompt,
                        messages=messages,
                        proposed_trade=args,
                    )
                    if cycle:
                        cycle.add_validator_result(
                            proposed_trade=args,
                            primary_approval=validation.get("primary_decision"),
                            secondary_approval=validation.get("secondary_decision"),
                            outcome="blocked" if not validation.get("execute") else "approved",
                            reason=validation.get("disagreement_reason") or validation.get("error_reason"),
                        )
                    if not validation.get("execute", False):
                        await self._log_tool_call(tool_name, args, {"blocked": True})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({
                                "blocked": True,
                                "reason": validation.get("disagreement_reason")
                                or validation.get("error_reason")
                                or "Validator blocked execution",
                            }),
                        })
                        tool_calls_made.append({"name": tool_name, "blocked": True})
                        continue

                t0 = time.monotonic()
                result = await self.tool_handlers.execute(tool_name, args)
                duration_ms = int((time.monotonic() - t0) * 1000)

                await self._log_tool_call(tool_name, args, result, duration_ms)
                tool_calls_made.append({"name": tool_name, "args": args, "result": result})
                if cycle:
                    cycle.add_tool_call(tool_name, args, result, duration_ms)

                tool_content = json.dumps(result, default=str)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                })

        logger.warning("Tool loop reached max iterations (%d)", self.max_iterations)
        await self._log_decision({"content": "max_iterations_reached", "messages": messages})
        if cycle:
            cycle.finish(outcome="max_iterations", response="Max iterations reached")
        return {
            "response": "Max iterations reached",
            "tool_calls_made": tool_calls_made,
            "blocked": False,
        }

    def _build_initial_messages(
        self,
        prompt: str,
        market_data: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        market_summary = json.dumps(market_data, indent=2, default=str)
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Current market data:\n{market_summary}\n\n"
                    f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}\n\n"
                    "Please analyze the market and decide whether to enter a trade, "
                    "manage existing positions, or wait. Use the available tools to "
                    "gather any additional information you need."
                ),
            },
        ]

    async def _log_decision(self, data: Dict[str, Any]):
        if self.db:
            try:
                await self.db.log_decision(data)
            except Exception:
                logger.exception("Failed to log decision")

    async def _log_tool_call(
        self,
        name: str,
        args: Dict,
        result: Any,
        duration_ms: Optional[int] = None,
    ):
        if self.db:
            try:
                await self.db.log_tool_call({
                    "name": name,
                    "arguments": args,
                    "result": result,
                    "duration_ms": duration_ms,
                    "call_time": datetime.now(),
                    "session_date": date.today(),
                })
            except Exception:
                logger.exception("Failed to log tool call")


def _parse_decision_from_content(content: str) -> Optional[Dict[str, Any]]:
    """Try to parse a JSON decision from LLM text content."""
    import re
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _is_execution_tool(tool_name: str) -> bool:
    """Return True if the tool places or modifies real orders."""
    execution_tools = {
        "place_iron_condor",
        "place_credit_spread",
        "place_stop_order",
        "close_position",
        "cancel_order",
    }
    return tool_name in execution_tools
