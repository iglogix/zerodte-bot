"""
Dual-LLM Validator for 0DTE Trading Bot.

Both LLMs must independently agree on ALL aspects before any trade executes:
  - Entry decision (enter vs wait)
  - Timing
  - Order sequence
  - Stop-loss amount

Any disagreement → block execution + notify user.
Primary LLM failure → block execution (no fallback for safety).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AGREEMENT_FIELDS = [
    "action",
    "timing",
    "order_sequence",
    "short_call_strike",
    "long_call_strike",
    "short_put_strike",
    "long_put_strike",
    "stop_loss_amount",
]


class DualLLMValidator:
    """
    Validates trading decisions by requiring agreement from two independent LLMs.

    The secondary LLM receives the same market data and prompt as the primary
    (virgin eyes — it never sees the primary's output).
    Both are called in parallel for efficiency.
    """

    def __init__(
        self,
        primary_llm=None,
        secondary_llm=None,
        notifier=None,
        decision_logger=None,
        timeout_seconds: float = 30.0,
    ):
        self.primary_llm = primary_llm
        self.secondary_llm = secondary_llm
        self.notifier = notifier
        self.decision_logger = decision_logger
        self.timeout_seconds = timeout_seconds

    async def validate_decision(
        self,
        market_data: Dict[str, Any],
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Ask both LLMs independently, then compare decisions.
        Returns a result dict with 'execute', 'agreed', and details.
        """
        primary_decision = None
        secondary_decision = None
        primary_error = None
        secondary_error = None

        try:
            primary_task = asyncio.create_task(
                self._call_with_timeout(self.primary_llm, market_data, prompt, messages)
            )
            secondary_task = asyncio.create_task(
                self._call_with_timeout(self.secondary_llm, market_data, prompt, messages)
            )
            results = await asyncio.gather(primary_task, secondary_task, return_exceptions=True)

            if isinstance(results[0], Exception):
                primary_error = str(results[0])
            else:
                primary_decision = results[0]

            if isinstance(results[1], Exception):
                secondary_error = str(results[1])
            else:
                secondary_decision = results[1]

        except Exception as exc:
            primary_error = str(exc)

        if primary_error:
            logger.error("Validator primary LLM failed: %s", primary_error)
            logger.error("Validator secondary error: %s", secondary_error)
            result = {
                "execute": False,
                "agreed": False,
                "error": True,
                "error_reason": f"Primary LLM failed: {primary_error}",
            }
            await self._notify_error(result)
            await self._log(result, primary_decision, secondary_decision)
            return result

        if not self._is_valid_decision(primary_decision):
            result = {
                "execute": False,
                "agreed": False,
                "error": True,
                "error_reason": "Primary LLM returned invalid response",
            }
            await self._notify_error(result)
            await self._log(result, primary_decision, secondary_decision)
            return result

        if secondary_error or not self._is_valid_decision(secondary_decision):
            reason = secondary_error or "Secondary LLM returned invalid response"
            result = {
                "execute": False,
                "agreed": False,
                "error": True,
                "error_reason": f"Secondary LLM failed: {reason}",
            }
            await self._notify_error(result)
            await self._log(result, primary_decision, secondary_decision)
            return result

        disagreement = self._find_disagreement(primary_decision, secondary_decision)

        if disagreement:
            result = {
                "execute": False,
                "agreed": False,
                "error": False,
                "action": primary_decision.get("action"),
                "disagreement_reason": disagreement,
                "primary_decision": primary_decision,
                "secondary_decision": secondary_decision,
                "timing_agreed": primary_decision.get("timing") == secondary_decision.get("timing"),
                "order_sequence_agreed": primary_decision.get("order_sequence") == secondary_decision.get("order_sequence"),
                "stop_loss_agreed": primary_decision.get("stop_loss_amount") == secondary_decision.get("stop_loss_amount"),
            }
            await self._notify_disagreement(result, primary_decision, secondary_decision)
            await self._log(result, primary_decision, secondary_decision)
            return result

        action = primary_decision.get("action", "wait")
        result = {
            "execute": action == "enter_trade",
            "agreed": True,
            "error": False,
            "action": action,
            "decision": primary_decision,
            "timing_agreed": True,
            "order_sequence_agreed": True,
            "stop_loss_agreed": True,
            "order_sequence": primary_decision.get("order_sequence"),
        }
        await self._log(result, primary_decision, secondary_decision)
        return result


    async def _call_with_timeout(
        self,
        llm,
        market_data: Dict,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        return await asyncio.wait_for(
            llm.get_decision(market_data=market_data, prompt=prompt, messages=messages),
            timeout=self.timeout_seconds,
        )

    def _is_valid_decision(self, decision: Any) -> bool:
        if not isinstance(decision, dict):
            return False
        return "action" in decision

    def _find_disagreement(
        self,
        primary: Dict[str, Any],
        secondary: Dict[str, Any],
    ) -> Optional[str]:
        """Return a human-readable disagreement reason, or None if they agree."""
        if primary.get("action") != secondary.get("action"):
            return (
                f"Entry action disagreement: primary={primary.get('action')!r}, "
                f"secondary={secondary.get('action')!r}"
            )

        if primary.get("action") != "enter_trade":
            return None

        if primary.get("timing") != secondary.get("timing"):
            return (
                f"Timing disagreement: primary={primary.get('timing')!r}, "
                f"secondary={secondary.get('timing')!r}"
            )

        if primary.get("order_sequence") != secondary.get("order_sequence"):
            return (
                f"Order sequence disagreement: primary={primary.get('order_sequence')!r}, "
                f"secondary={secondary.get('order_sequence')!r}"
            )

        if primary.get("stop_loss_amount") != secondary.get("stop_loss_amount"):
            return (
                f"Stop-loss disagreement: primary={primary.get('stop_loss_amount')}, "
                f"secondary={secondary.get('stop_loss_amount')}"
            )

        for field in ["short_call_strike", "long_call_strike", "short_put_strike", "long_put_strike"]:
            if primary.get(field) != secondary.get(field):
                return (
                    f"Strike disagreement on {field}: "
                    f"primary={primary.get(field)}, secondary={secondary.get(field)}"
                )

        return None

    async def _notify_disagreement(
        self,
        result: Dict,
        primary_decision: Dict,
        secondary_decision: Dict,
    ):
        if self.notifier:
            try:
                await self.notifier.notify_disagreement(
                    disagreement_reason=result["disagreement_reason"],
                    primary_decision=primary_decision,
                    secondary_decision=secondary_decision,
                )
            except Exception:
                logger.exception("Failed to send disagreement notification")

    async def _notify_error(self, result: Dict):
        if self.notifier:
            try:
                await self.notifier.notify_error(
                    error_reason=result["error_reason"],
                )
            except Exception:
                logger.exception("Failed to send error notification")

    async def _log(
        self,
        result: Dict,
        primary_decision: Optional[Dict],
        secondary_decision: Optional[Dict],
    ):
        if self.decision_logger:
            try:
                await self.decision_logger.log_decision({
                    "timestamp": datetime.now().isoformat(),
                    "result": result,
                    "primary": primary_decision,
                    "secondary": secondary_decision,
                })
            except Exception:
                logger.exception("Failed to log decision")
