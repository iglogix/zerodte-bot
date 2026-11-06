"""
Deterministic Decision Engine — replaces the LLM tool loop.

CycleResult lives here for backward compatibility.
DecisionEngine is a backward-compatible wrapper around ScheduledEngine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    outcome: str          # executed | waited | skip_trade | skip_session | error
    reason: str = ""
    trade: Optional[Dict[str, Any]] = None
    regime: Optional[str] = None
    anomaly: Optional[str] = None
    candidates_found: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str = ""

    def finish(self) -> "CycleResult":
        self.finished_at = datetime.now().isoformat()
        return self


class DecisionEngine:
    """Backward-compatible wrapper. New code should use ScheduledEngine."""

    def __init__(
        self,
        tool_handlers=None,
        risk_manager=None,
        anomaly_detector=None,
        regime_classifier=None,
        decision_file_logger=None,
        strategy=None,
        mode: str = "dry_run",
        stability_fraction: float = 0.15,
        single_candle_fraction: float = 0.10,
        fallback_stability_pts: float = 8.0,
    ):
        from zerodte.agent.base_engine import EngineDeps
        from zerodte.agent.scheduled_engine import ScheduledEngine

        deps = EngineDeps(
            tool_handlers=tool_handlers,
            risk_manager=risk_manager,
            anomaly_detector=anomaly_detector,
            regime_classifier=regime_classifier,
            decision_file_logger=decision_file_logger,
            db=None,
            mode=mode,
            config={"entry": {
                "stability_fraction": stability_fraction,
                "single_candle_fraction": single_candle_fraction,
                "fallback_stability_pts": fallback_stability_pts,
            }},
        )
        self._inner = ScheduledEngine(deps, strategy)
        # Expose attributes for tests that access them directly
        self.tool_handlers = tool_handlers
        self.risk_manager = risk_manager
        self.anomaly_detector = anomaly_detector
        self.regime_classifier = regime_classifier
        self.decision_file_logger = decision_file_logger
        self.strategy = strategy
        self.mode = mode
        self.stability_fraction = stability_fraction
        self.single_candle_fraction = single_candle_fraction
        self.fallback_stability_pts = fallback_stability_pts

    async def run_cycle(self) -> CycleResult:
        return await self._inner.run_cycle()


__all__ = ["CycleResult", "DecisionEngine"]
