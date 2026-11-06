"""
Dual-LLM approval gate for 0DTE Trading Bot.

Flow:
  1. Primary agent (Claude) runs the tool loop and proposes a specific trade
     (strikes, stop-loss, order sequence) via a tool call or JSON block.
  2. Validator calls TWO independent LLMs with the proposed trade details:
       - primary_llm  (same model as trading agent, fresh context)
       - secondary_llm (different model, e.g. GPT-5.2)
  3. Both must return {"action": "approve"} for the trade to execute.
  4. Either returns "reject" or errors → block + notify.

The validator never re-derives strikes — it only approves or rejects the
specific trade the primary agent proposed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)



class DualLLMValidator:
    """
    Approves or rejects the primary agent's proposed trade using two independent LLMs.

    Both validators see the same market context and the exact proposed trade
    (strikes, stop-loss, order sequence). Both must return 'approve' to execute.
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
        proposed_trade: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ask both LLMs to approve or reject the primary agent's proposed trade.

        Both must return {"action": "approve"} for the trade to execute.
        Either returning "reject" or any error → block.
        """
        primary_approval = None
        secondary_approval = None
        primary_error = None
        secondary_error = None

        try:
            primary_task = asyncio.create_task(
                self._call_approval(self.primary_llm, market_data, prompt, messages, proposed_trade)
            )
            secondary_task = asyncio.create_task(
                self._call_approval(self.secondary_llm, market_data, prompt, messages, proposed_trade)
            )
            results = await asyncio.gather(primary_task, secondary_task, return_exceptions=True)

            if isinstance(results[0], Exception):
                primary_error = str(results[0])
            else:
                primary_approval = results[0]

            if isinstance(results[1], Exception):
                secondary_error = str(results[1])
            else:
                secondary_approval = results[1]

        except Exception as exc:
            primary_error = str(exc)

        if primary_error:
            logger.error("Validator primary LLM failed: %s", primary_error)
            result = {
                "execute": False,
                "agreed": False,
                "error": True,
                "error_reason": f"Primary validator failed: {primary_error}",
            }
            await self._notify_error(result)
            await self._log(result, primary_approval, secondary_approval)
            return result

        if secondary_error:
            logger.error("Validator secondary LLM failed: %s", secondary_error)
            result = {
                "execute": False,
                "agreed": False,
                "error": True,
                "error_reason": f"Secondary validator failed: {secondary_error}",
            }
            await self._notify_error(result)
            await self._log(result, primary_approval, secondary_approval)
            return result

        p = primary_approval or {}
        s = secondary_approval or {}

        # Both responses must contain "action" to be valid
        if "action" not in p or "action" not in s:
            result = {
                "execute": False,
                "agreed": False,
                "error": True,
                "error_reason": "One or both LLMs returned invalid response (missing 'action')",
                "primary_decision": primary_approval,
                "secondary_decision": secondary_approval,
            }
            await self._notify_error(result)
            await self._log(result, primary_approval, secondary_approval)
            return result

        primary_action = p["action"]
        secondary_action = s["action"]

        logger.info(
            "Validator: primary=%s reason=%r | secondary=%s reason=%r",
            primary_action, p.get("reasoning", "")[:120],
            secondary_action, s.get("reasoning", "")[:120],
        )

        # Compare key fields for disagreement
        _COMPARE_FIELDS = [
            ("action", "action/entry"),
            ("timing", "timing"),
            ("order_sequence", "sequence"),
            ("stop_loss_amount", "stop-loss"),
            ("short_call_strike", "strike"),
            ("long_call_strike", "strike"),
            ("short_put_strike", "strike"),
            ("long_put_strike", "strike"),
        ]

        disagreements = []
        for field, label in _COMPARE_FIELDS:
            pv, sv = p.get(field), s.get(field)
            if pv is not None and sv is not None and pv != sv:
                disagreements.append((field, label, pv, sv))

        if disagreements:
            _, label, pv, sv = disagreements[0]
            result = {
                "execute": False,
                "agreed": False,
                "error": False,
                "action": primary_action,
                "disagreement_reason": (
                    f"LLMs disagree on {label}: primary={pv}, secondary={sv}"
                ),
                "primary_decision": primary_approval,
                "secondary_decision": secondary_approval,
            }
            await self._notify_disagreement(result, primary_approval, secondary_approval)
            await self._log(result, primary_approval, secondary_approval)
            return result

        # Both agree on all fields
        execute = primary_action == "enter_trade"
        result = {
            "execute": execute,
            "agreed": True,
            "error": False,
            "action": primary_action,
            "timing_agreed": p.get("timing") == s.get("timing") if "timing" in p else True,
            "order_sequence_agreed": p.get("order_sequence") == s.get("order_sequence") if "order_sequence" in p else True,
            "order_sequence": p.get("order_sequence"),
            "stop_loss_agreed": p.get("stop_loss_amount") == s.get("stop_loss_amount") if "stop_loss_amount" in p else True,
            "primary_decision": primary_approval,
            "secondary_decision": secondary_approval,
        }
        await self._log(result, primary_approval, secondary_approval)
        return result


    async def _call_approval(
        self,
        llm,
        market_data: Dict,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        proposed_trade: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        kwargs: Dict[str, Any] = {
            "market_data": market_data,
            "prompt": prompt,
        }
        if messages is not None:
            kwargs["messages"] = messages
        if proposed_trade is not None:
            kwargs["proposed_trade"] = proposed_trade
        return await asyncio.wait_for(
            llm.get_decision(**kwargs),
            timeout=self.timeout_seconds,
        )

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
