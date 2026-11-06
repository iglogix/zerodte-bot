"""
Strategy protocol for pluggable strike selection and pricing.

All backtest strategies implement this ABC. The engine calls:
  - configure()       at construction
  - select_trade()    at entry time (Step 6)
  - compute_exit_value() on every bar during exit simulation (Step 7)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TradeSetup:
    """Output of strategy.select_trade() — everything needed to enter a trade."""
    spread_type: str           # "put_credit_spread" | "call_credit_spread"
    short_strike: float
    long_strike: float
    spread_width: float
    credit: float              # net credit after slippage
    short_delta: float         # delta at entry (for reporting)
    iv_used: float             # IV used for pricing (for reporting)
    extra: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Protocol for pluggable strike selection + pricing strategies."""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Read strategy-specific params from config dict."""
        ...

    @abstractmethod
    def select_trade(
        self,
        spot: float,
        bias: str,               # "long" | "short"
        entry_time: datetime,
        session_iv: float,
        tte_years: float,
        tte_hours: float,
        engine: Any,             # back-reference for IV helpers
    ) -> Optional[TradeSetup]:
        """Pick strikes and compute credit. Return None to skip."""
        ...

    @abstractmethod
    def compute_exit_value(
        self,
        spot: float,
        short_k: float,
        long_k: float,
        tte_years: float,
        tte_hours: float,
        spread_type: str,
        session_iv: float,
        engine: Any,
    ) -> float:
        """Mark-to-market the spread at a given moment."""
        ...

    @abstractmethod
    def grid_params(self) -> Dict[str, List[Any]]:
        """Return the parameter grid this strategy wants to sweep."""
        ...

    @abstractmethod
    def apply_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizer params to a config dict."""
        ...
