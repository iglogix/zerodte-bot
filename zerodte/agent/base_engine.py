"""Base engine interface — all entry engines implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EngineDeps:
    """Shared dependencies injected into every engine."""
    tool_handlers: Any
    risk_manager: Any
    anomaly_detector: Any
    regime_classifier: Any
    decision_file_logger: Any
    db: Any
    mode: str
    config: Dict[str, Any]
    clock: Any = None  # SimClock for backtest, WallClock (default) for live


class BaseEngine(ABC):
    """Protocol for pluggable entry engines.

    Each strategy creates the engine it needs via create_engine().
    The orchestrator only calls run() — it doesn't know which engine type it has.
    """

    def __init__(self, deps: EngineDeps, strategy: Any):
        self.deps = deps
        self.strategy = strategy

    @abstractmethod
    async def run(self):
        """Run the full entry session. Called by BotOrchestrator."""
        ...
