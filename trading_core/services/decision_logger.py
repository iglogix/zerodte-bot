"""
Decision File Logger — writes one JSON file per decision cycle.

Files are organized as:
  logs/decisions/YYYY-MM-DD/HH-MM-SS_<cycle_id>.json

Each file contains the full decision context:
  - cycle metadata (time, mode, market snapshot)
  - all tool calls with complete (untruncated) results
  - all validator invocations with full reasoning from both LLMs
  - primary LLM final response
  - outcome (executed / blocked / waited / max_iterations)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DecisionFileLogger:
    """
    Writes one JSON file per decision cycle under logs/decisions/YYYY-MM-DD/.

    Usage:
        dfl = DecisionFileLogger(base_dir="logs/decisions")
        cycle = dfl.start_cycle(mode="sandbox", market_data={...})
        cycle.add_tool_call("get_spx_price", {}, {"price": 5900})
        cycle.add_validator_result(proposed_trade={...}, primary={...}, secondary={...}, outcome="blocked")
        cycle.finish(response="...", outcome="blocked")
    """

    def __init__(self, base_dir: str = "logs/decisions"):
        self.base_dir = Path(base_dir)

    def start_cycle(
        self,
        mode: str = "unknown",
        market_data: Optional[Dict[str, Any]] = None,
    ) -> "DecisionCycle":
        today = date.today().isoformat()
        folder = self.base_dir / today
        folder.mkdir(parents=True, exist_ok=True)
        return DecisionCycle(folder=folder, mode=mode, market_data=market_data or {})


class DecisionCycle:
    """
    Accumulates context for one decision cycle and writes it to disk on finish().
    """

    def __init__(
        self,
        folder: Path,
        mode: str,
        market_data: Dict[str, Any],
    ):
        self._folder = folder
        self._started_at = datetime.now()
        self._cycle_id = self._started_at.strftime("%H-%M-%S") + "_" + uuid.uuid4().hex[:6]
        self._doc: Dict[str, Any] = {
            "cycle_id": self._cycle_id,
            "started_at": self._started_at.isoformat(),
            "finished_at": None,
            "mode": mode,
            "outcome": "unknown",
            "market_snapshot": market_data,
            "tool_calls": [],
            "validator_invocations": [],
            "primary_response": None,
        }

    # ------------------------------------------------------------------
    # Recording methods (called during the cycle)
    # ------------------------------------------------------------------

    def add_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        result: Any,
        duration_ms: Optional[int] = None,
    ):
        """Record a tool call with its full (untruncated) result."""
        self._doc["tool_calls"].append({
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "args": args,
            "result": result,
            "duration_ms": duration_ms,
        })

    @staticmethod
    def _split_reasoning(approval: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return a copy of an approval dict with reasoning split into lines."""
        if not approval:
            return approval
        result = dict(approval)
        if isinstance(result.get("reasoning"), str):
            result["reasoning"] = result["reasoning"].splitlines()
        return result

    def add_validator_result(
        self,
        proposed_trade: Optional[Dict[str, Any]],
        primary_approval: Optional[Dict[str, Any]],
        secondary_approval: Optional[Dict[str, Any]],
        outcome: str,
        reason: Optional[str] = None,
    ):
        """Record one validator invocation (approve/reject) with full reasoning."""
        self._doc["validator_invocations"].append({
            "timestamp": datetime.now().isoformat(),
            "proposed_trade": proposed_trade,
            "primary_approval": self._split_reasoning(primary_approval),
            "secondary_approval": self._split_reasoning(secondary_approval),
            "outcome": outcome,
            "reason": reason.splitlines() if isinstance(reason, str) else reason,
        })

    def set_primary_response(self, response: str):
        self._doc["primary_response"] = response.splitlines() if response else []

    # ------------------------------------------------------------------
    # Finish — write to disk
    # ------------------------------------------------------------------

    def finish(self, outcome: str, response: Optional[str] = None):
        """
        Finalise the cycle and write the JSON file.

        outcome: "executed" | "blocked" | "waited" | "max_iterations" | "error"
        """
        self._doc["finished_at"] = datetime.now().isoformat()
        self._doc["outcome"] = outcome
        if response is not None:
            self._doc["primary_response"] = response.splitlines() if response else []

        file_path = self._folder / f"{self._cycle_id}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(self._doc, f, indent=2, default=str)
            logger.info("Decision file written: %s", file_path)
        except Exception:
            logger.exception("Failed to write decision file %s", file_path)

        return file_path
