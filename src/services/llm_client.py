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
    ) -> Dict[str, Any]:
        """
        Used by DualLLMValidator — ask the LLM for a structured trading decision.
        Returns a dict with action, timing, strikes, stop_loss_amount, etc.
        """
        from src.agent.tool_definitions import TOOL_DEFINITIONS

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Market data:\n{json.dumps(market_data, indent=2, default=str)}\n\n"
                    "Based on the strategy, provide your trading decision as a JSON object with:\n"
                    "- action: 'enter_trade' or 'wait'\n"
                    "- timing: 'now' or reason to wait\n"
                    "- order_sequence: ['call_spread', 'put_spread'] or null\n"
                    "- short_call_strike, long_call_strike, short_put_strike, long_put_strike\n"
                    "- stop_loss_amount: total premium to use as stop\n"
                    "- reasoning: brief explanation\n\n"
                    "Respond ONLY with valid JSON."
                ),
            },
        ]

        import litellm

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
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
