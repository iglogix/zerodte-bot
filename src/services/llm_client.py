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

        _new_models = ("gpt-5", "o1", "o3", "o4")
        token_param = "max_completion_tokens" if any(self.model.startswith(f"openai/{p}") or self.model.startswith(p) for p in _new_models) else "max_tokens"
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            token_param: self.max_tokens,
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
            # Flatten the tool-loop into a single user message containing all
            # tool results as plain text. This avoids provider-specific issues
            # (e.g. Anthropic rejecting tool_calls without tool definitions)
            # while still giving the validator full market data context.
            # Build a compact summary of what the primary agent gathered.
            # We only keep the last result per tool to avoid duplication, and
            # cap option_chain results to a short summary to stay within context limits.
            _LARGE_TOOLS = {"get_option_chain", "get_5min_candles"}
            last_results: dict = {}  # tool_name -> compact result string
            last_tool_name = ""
            for msg in messages:
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        last_tool_name = tc.get("function", {}).get("name") or tc.get("name", "")
                elif msg["role"] == "tool":
                    content = msg.get("content", "")
                    if last_tool_name in _LARGE_TOOLS:
                        # Summarise: just count items and show first entry
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, list):
                                first = json.dumps(parsed[0]) if parsed else "{}"
                                content = f"[{len(parsed)} items, first: {first[:300]}]"
                            else:
                                content = content[:300]
                        except Exception:
                            content = content[:300]
                    else:
                        content = content[:500]
                    last_results[last_tool_name] = content

            tool_results_text = [f"{name}: {val}" for name, val in last_results.items()]
            context_block = "\n".join(tool_results_text)
            validator_messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Market data gathered by the primary agent:\n\n{context_block}"
                        + _DECISION_REQUEST
                    ),
                },
            ]
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

        # gpt-5.x and o-series models use max_completion_tokens, not max_tokens
        _new_models = ("gpt-5", "o1", "o3", "o4")
        token_param = "max_completion_tokens" if any(self.model.startswith(f"openai/{p}") or self.model.startswith(p) for p in _new_models) else "max_tokens"
        response = await litellm.acompletion(
            model=self.model,
            messages=validator_messages,
            temperature=self.temperature,
            **{token_param: 1024},
        )

        import re
        content = response.choices[0].message.content or ""
        content = content.strip()

        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strip markdown code block
        if "```" in content:
            parts = content.split("```")
            for part in parts[1::2]:  # odd parts are inside code blocks
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

        # Regex: find first {...} JSON object in the response
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: Claude returned prose — infer action from keywords
        logger.error("Validator could not parse JSON from response: %s", content[:300])
        action = "enter_trade" if any(w in content.lower() for w in ["enter", "approve", "proceed", "execute"]) else "wait"
        return {"action": action, "reasoning": content[:200]}

    async def get_approval(
        self,
        market_data: Dict[str, Any],
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        proposed_trade: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Used by DualLLMValidator — ask the LLM to approve or reject a specific
        proposed trade that the primary agent has already decided on.

        The LLM sees:
          - The compact market data context (from the primary's tool loop)
          - The exact trade parameters proposed (strikes, stop-loss, etc.)
          - A request to approve or reject with reasoning

        Returns {"action": "approve"|"reject", "reasoning": "..."}
        """
        import litellm
        import re

        trade_json = json.dumps(proposed_trade or {}, indent=2)

        # Build compact context from tool-loop messages.
        # The primary agent already truncated option_chain to 30 near-ATM enriched strikes,
        # so we pass that content as-is. Other large tools are capped at 800 chars.
        context_block = ""
        if messages:
            last_results: dict = {}
            last_tool_name = ""
            for msg in messages:
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        last_tool_name = tc.get("function", {}).get("name") or tc.get("name", "")
                elif msg["role"] == "tool":
                    content = msg.get("content", "")
                    # option_chain is already pre-truncated by trading_agent to 30 near-ATM strikes
                    # candles are small; other tools capped at 800 chars
                    if last_tool_name not in ("get_option_chain",) and len(content) > 800:
                        content = content[:800]
                    last_results[last_tool_name] = content
            context_block = "\n".join(f"{k}: {v}" for k, v in last_results.items())
        else:
            context_block = f"Market data: {json.dumps(market_data, default=str)[:500]}"

        from datetime import date as _date
        today_str = _date.today().isoformat()
        approval_request = (
            f"\n\nIMPORTANT: Today's date is {today_str}. Options expiring {today_str} ARE 0DTE (expiring today).\n\n"
            f"The primary trading agent has proposed the following trade:\n"
            f"```json\n{trade_json}\n```\n\n"
            "Review the market data above and decide whether to APPROVE or REJECT this specific trade.\n"
            "Consider: are the strikes appropriate given current SPX price and volatility? "
            "Is the stop-loss reasonable? Is the timing correct?\n\n"
            "Respond ONLY with a JSON object:\n"
            "{\"action\": \"approve\" or \"reject\", \"reasoning\": \"brief explanation\"}"
        )

        validator_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Market data gathered by the primary agent:\n\n{context_block}{approval_request}",
            },
        ]

        _new_models = ("gpt-5", "o1", "o3", "o4")
        token_param = "max_completion_tokens" if any(
            self.model.startswith(f"openai/{p}") or self.model.startswith(p)
            for p in _new_models
        ) else "max_tokens"

        response = await litellm.acompletion(
            model=self.model,
            messages=validator_messages,
            temperature=self.temperature,
            **{token_param: 512},
        )

        content = (response.choices[0].message.content or "").strip()

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strip markdown code block
        if "```" in content:
            for part in content.split("```")[1::2]:
                part = part.strip().lstrip("json").strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

        # Regex fallback
        match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Keyword inference
        logger.error("Approval LLM could not parse JSON: %s", content[:300])
        action = "approve" if any(w in content.lower() for w in ["approve", "yes", "proceed", "looks good"]) else "reject"
        return {"action": action, "reasoning": content[:200]}
