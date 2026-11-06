"""
LiveStrategy ABC — strategy design pattern for the live trading bot.

The bot (DecisionEngine, PositionMonitor) calls only these interface methods.
All strategy-specific logic (strike selection, validation, TP/SL rules)
lives in concrete implementations (IronCondorStrategy, IronFlyStrategy).

To add a new strategy:
  1. Subclass LiveStrategy
  2. Implement all abstract methods
  3. Add strategy name to config routing in main.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zerodte.agent.base_engine import BaseEngine, EngineDeps


@dataclass
class OrderLeg:
    """Single option leg in a trade order."""
    symbol: str
    strike: float
    option_type: str    # "call" | "put"
    side: str           # "buy" | "sell"
    quantity: int


@dataclass
class LiveTradeSetup:
    """Output of strategy.search_candidates() — everything needed to enter a trade.

    strategy_type identifies which strategy produced this setup.
    legs contains the constructed order legs.
    strikes is a strategy-specific strike map (e.g. IC has 4 strikes, IF has 3).
    extra holds any additional strategy-specific metadata.
    """
    strategy_type: str
    legs: List[OrderLeg]
    total_credit: float
    strikes: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitAction:
    """What the strategy wants done on a position check.

    action: "take_profit" | "stop_loss" | "hold" | "trailing_update"
    legs_to_close: list of leg symbols to close (empty for "hold")
    reason: human-readable explanation
    new_stop_price: for trailing stop updates
    leg_quantities: per-symbol contract count to close (defaults to full leg)
    quantity_fraction: whole-trade close fraction (default 1.0 = all)
    """
    action: str
    legs_to_close: List[str] = field(default_factory=list)
    reason: str = ""
    new_stop_price: Optional[float] = None
    leg_quantities: Dict[str, int] = field(default_factory=dict)
    quantity_fraction: float = 1.0


class LiveStrategy(ABC):
    """Protocol for pluggable live trading strategies.

    The bot engine and position monitor call these methods — they never
    import strategy-specific logic directly.
    """

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Read strategy-specific parameters from config dict."""
        ...

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    @abstractmethod
    async def search_candidates(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
    ) -> Dict[str, Any]:
        """Search the option chain for trade candidates.

        Returns dict with:
          candidates_found: int
          top_candidate: Optional[LiveTradeSetup]
          candidates: List[LiveTradeSetup]
        """
        ...

    @abstractmethod
    async def validate(
        self,
        setup: LiveTradeSetup,
        chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Pre-flight validation of a trade setup against live chain data.

        Returns dict with:
          passed: bool
          checks: List[Dict]     — per-check pass/fail
          failures: List[str]    — failed check names
        """
        ...

    @abstractmethod
    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        """Build order payload for the TastyTrade API.

        Returns a dict suitable for api_client.place_order().
        """
        ...

    # ------------------------------------------------------------------
    # Risk / stop-loss
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        """Compute stop-loss levels for a trade.

        Returns strategy-specific stop info (e.g. dollar amounts per side).
        """
        ...

    @abstractmethod
    def build_stop_orders(
        self,
        setup: LiveTradeSetup,
        stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build stop order payloads from stop info.

        Returns list of order dicts for the API.
        """
        ...

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    @abstractmethod
    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        """Given current quotes for all legs, decide what to do.

        Returns ExitAction with action type and legs to close.
        """
        ...

    @abstractmethod
    def compute_trailing_stop(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        """Compute updated trailing stop price, or None if no update needed."""
        ...

    @abstractmethod
    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        """Return leg pair mapping for position monitoring.

        Each tuple: (short_key, long_key, side_name)
        e.g. [("short_call_symbol", "long_call_symbol", "call"),
              ("short_put_symbol", "long_put_symbol", "put")]
        """
        ...

    @abstractmethod
    def create_engine(self, deps: EngineDeps) -> BaseEngine:
        """Return the entry engine this strategy requires.

        Scheduled strategies return ScheduledEngine.
        ORB strategies return OrbEngine.
        """
        ...
