"""
Risk management service — generic pre-trade risk checks.

Enforces:
- Max concurrent trades
- Daily risk limit (% of account equity)
- Buying power limit (% of equity)

Strategy-specific logic (stop-loss, take-profit, trailing stops) lives in
the strategy implementations (zerodte/strategies/).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates and calculates risk parameters for all trades."""

    def __init__(
        self,
        db=None,
        cache=None,
        max_concurrent_trades: int = 10,
        max_daily_risk_percent: float = 2.0,
        max_buying_power_percent: float = 50.0,
    ):
        self.db = db
        self.cache = cache
        self.max_concurrent_trades = max_concurrent_trades
        self.max_daily_risk_percent = max_daily_risk_percent
        self.max_buying_power_percent = max_buying_power_percent

    # -------------------------------------------------------------------------
    # Concurrent trade limit
    # -------------------------------------------------------------------------

    async def can_open_new_trade(self) -> Dict[str, Any]:
        """Check whether a new trade can be opened (max concurrent limit)."""
        count = await self.db.get_open_trade_count()
        if count >= self.max_concurrent_trades:
            return {
                "allowed": False,
                "current_count": count,
                "max_allowed": self.max_concurrent_trades,
                "reason": (
                    f"Max concurrent trades reached: {count}/{self.max_concurrent_trades}"
                ),
            }
        return {
            "allowed": True,
            "current_count": count,
            "max_allowed": self.max_concurrent_trades,
        }

    # -------------------------------------------------------------------------
    # Daily risk limit
    # -------------------------------------------------------------------------

    async def calculate_current_risk(self) -> float:
        """
        Calculate current worst-case risk in dollars.
        Assumes both sides of every open IC could hit stop-loss simultaneously.
        """
        trades = await self.db.get_open_trades()
        total_risk = 0.0
        for trade in trades:
            premium = trade.get("total_premium", 0.0)
            total_risk += premium * 2 * 100
        return total_risk

    async def check_daily_risk_limit(self, new_trade_risk: float) -> Dict[str, Any]:
        """Check whether adding a new trade stays within the daily risk limit."""
        account = await self.cache.get_cached_account_info(max_age_seconds=60)
        if account is None:
            account = await self.cache.get_account_info_with_fallback()

        equity = account.get("equity", 0.0)
        max_risk = equity * (self.max_daily_risk_percent / 100.0)
        current_risk = await self.calculate_current_risk()
        projected_risk = current_risk + new_trade_risk

        if projected_risk > max_risk:
            return {
                "allowed": False,
                "current_risk": current_risk,
                "new_trade_risk": new_trade_risk,
                "projected_risk": projected_risk,
                "max_risk": max_risk,
                "reason": (
                    f"Projected risk ${projected_risk:,.0f} exceeds daily risk limit "
                    f"${max_risk:,.0f} ({self.max_daily_risk_percent}% of ${equity:,.0f})"
                ),
            }

        remaining = max_risk - projected_risk
        warning = remaining < (max_risk * 0.05)

        return {
            "allowed": True,
            "current_risk": current_risk,
            "new_trade_risk": new_trade_risk,
            "projected_risk": projected_risk,
            "max_risk": max_risk,
            "warning": warning,
        }

    # -------------------------------------------------------------------------
    # Buying power limit
    # -------------------------------------------------------------------------

    async def check_buying_power(self, required_amount: float) -> Dict[str, Any]:
        """Check whether sufficient buying power is available (50% limit)."""
        account = await self.cache.get_cached_account_info(max_age_seconds=60)
        if account is None:
            account = await self.cache.get_account_info_with_fallback()

        equity = account.get("equity", 0.0)
        max_allowed = equity * (self.max_buying_power_percent / 100.0)

        trades = await self.db.get_open_trades()
        used_bp = sum(t.get("buying_power_used", 0.0) for t in trades)
        available = max_allowed - used_bp

        if required_amount > available:
            return {
                "allowed": False,
                "available": available,
                "required": required_amount,
                "max_allowed": max_allowed,
                "used": used_bp,
                "reason": (
                    f"Required ${required_amount:,.0f} exceeds available buying power "
                    f"${available:,.0f} (50% limit: ${max_allowed:,.0f})"
                ),
            }

        return {
            "allowed": True,
            "available": available,
            "required": required_amount,
            "max_allowed": max_allowed,
            "used": used_bp,
        }

    # -------------------------------------------------------------------------
    # Comprehensive pre-trade validation
    # -------------------------------------------------------------------------

    async def validate_trade_entry(
        self,
        new_trade_risk: float,
        required_buying_power: float,
    ) -> Dict[str, Any]:
        """Run all risk checks before allowing a new trade entry."""
        concurrent_result = await self.can_open_new_trade()
        daily_result = await self.check_daily_risk_limit(new_trade_risk)
        bp_result = await self.check_buying_power(required_buying_power)

        concurrent_ok = concurrent_result["allowed"]
        daily_ok = daily_result["allowed"]
        bp_ok = bp_result["allowed"]

        all_ok = concurrent_ok and daily_ok and bp_ok

        if not all_ok:
            reasons = []
            if not concurrent_ok:
                reasons.append(concurrent_result["reason"])
            if not daily_ok:
                reasons.append(daily_result["reason"])
            if not bp_ok:
                reasons.append(bp_result["reason"])

            return {
                "allowed": False,
                "concurrent_trades_ok": concurrent_ok,
                "daily_risk_ok": daily_ok,
                "buying_power_ok": bp_ok,
                "reason": "; ".join(reasons),
                "details": {
                    "concurrent": concurrent_result,
                    "daily_risk": daily_result,
                    "buying_power": bp_result,
                },
            }

        return {
            "allowed": True,
            "concurrent_trades_ok": True,
            "daily_risk_ok": True,
            "buying_power_ok": True,
            "details": {
                "concurrent": concurrent_result,
                "daily_risk": daily_result,
                "buying_power": bp_result,
            },
        }
