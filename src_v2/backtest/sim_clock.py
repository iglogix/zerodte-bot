"""
Simulated and wall-clock time sources for backtest/live mode.

WallClock: delegates to datetime.now() and asyncio.sleep() — used in live mode.
SimClock: orchestrator-driven time with instant sleep — used in backtesting.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Optional


class WallClock:
    """Real wall-clock time. Default for live trading."""

    def now(self, tz=None) -> datetime:
        return datetime.now(tz)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SimClock:
    """Simulated clock for backtesting.

    Time is advanced by the orchestrator via advance_to().
    sleep() calls the advance callback instead of waiting.
    """

    def __init__(self, start: datetime):
        self._time = start
        self._advance_fn: Optional[Callable] = None

    def now(self, tz=None) -> datetime:
        if tz and self._time.tzinfo != tz:
            return self._time.astimezone(tz)
        return self._time

    def advance_to(self, t: datetime) -> None:
        self._time = t

    def set_advance_fn(self, fn: Callable) -> None:
        """Set callback invoked on each sleep() — typically advances to next bar."""
        self._advance_fn = fn

    async def sleep(self, seconds: float) -> None:
        if self._advance_fn:
            self._advance_fn()
