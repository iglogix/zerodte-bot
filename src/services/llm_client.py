"""
LLM client wrapper using LiteLLM for unified API access.
Supports Claude Sonnet 4.6 (primary) and GPT-5.2 (validator).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wraps LiteLLM for unified access to Claude and GPT models.
    Used by both the primary agent and the dual-LLM validator.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return a normalized response dict:
          {
            "tool_calls": [...] | None,
            "content": str | None,
          }
        """
        import litellm

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        response = await litellm.acompletion(**call_kwargs)
        return self._normalize(response)

    def _normalize(self, response: Any) -> Dict[str, Any]:
        """Normalize LiteLLM response to a consistent format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        return {
            "tool_calls": tool_calls,
            "content": message.content,
        }

    async def get_decision(
        self,
        market_data: Dict[str, Any],
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Used by DualLLMValidator — ask the LLM for a structured trading decision.

        When `messages` is provided (the full primary-agent tool-loop conversation),
        the validator sees all tool results (SPX price, option chain, candles, etc.)
        and appends a final user turn asking for a JSON verdict.

        Falls back to building from bare market_data when messages is not provided.

        Returns a dict with action, timing, strikes, stop_loss_amount, etc.
        """
        import litellm

        _DECISION_REQUEST = (
            "\n\nBased on everything above, provide your independent trading decision "
            "as a JSON object with:\n"
            "- action: 'enter_trade' or 'wait'\n"
            "- timing: 'now' or reason to wait\n"
            "- order_sequence: ['call_spread', 'put_spread'] or null\n"
            "- short_call_strike, long_call_strike, short_put_strike, long_put_strike\n"
            "- stop_loss_amount: total premium to use as stop\n"
            "- reasoning: brief explanation\n\n"
            "Respond ONLY with valid JSON."
        )

        if messages:
            # Use the full tool-loop conversation so the validator has the same
            # market data the primary agent gathered via tool calls.
            # - Replace system message with our prompt (virgin eyes).
            # - Keep assistant messages but blank their text content so the validator
            #   cannot see the primary's reasoning; tool_calls must stay so that the
            #   following tool-result messages remain valid (API requires matching IDs).
            validator_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    validator_messages.append({"role": "system", "content": prompt})
                elif msg["role"] == "assistant":
                    sanitised = {"role": "assistant", "content": ""}
                    if msg.get("tool_calls"):
                        sanitised["tool_calls"] = msg["tool_calls"]
                    validator_messages.append(sanitised)
                else:
                    validator_messages.append(msg)
            validator_messages.append({"role": "user", "content": _DECISION_REQUEST})
        else:
            validator_messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Market data:\n{json.dumps(market_data, indent=2, default=str)}"
                        + _DECISION_REQUEST
                    ),
                },
            ]

        response = await litellm.acompletion(
            model=self.model,
            messages=validator_messages,
            temperature=self.temperature,
            max_tokens=1024,
        )

        content = response.choices[0].message.content or ""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        return json.loads(content)
