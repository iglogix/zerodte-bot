"""
Risk management service for 0DTE Iron Condor Trading Bot.

Enforces:
- Max 10 concurrent iron condors
- Daily risk limit (1-2% of account equity)
- Buying power limit (50% of equity)
- Stop-loss calculation (= total premium collected)
- OCO stop structure (stop-limit 40pt gap + stop-market 30pt further)
- Take-profit at $0.05
- Trailing stop adjustment as long options decay
"""

from __future__ import annotations

import logging
from typing import Any, Dict

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
        take_profit_threshold: float = 0.05,
    ):
        self.db = db
        self.cache = cache
        self.max_concurrent_trades = max_concurrent_trades
        self.max_daily_risk_percent = max_daily_risk_percent
        self.max_buying_power_percent = max_buying_power_percent
        self.take_profit_threshold = take_profit_threshold

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
    # Stop-loss calculation
    # -------------------------------------------------------------------------

    def calculate_stop_loss_amount(self, total_premium: float) -> Dict[str, Any]:
        """
        Stop-loss = total premium collected (per side, in dollars).
        Each short option stop triggers at total_premium * 100.
        """
        stop_dollars = total_premium * 100
        return {
            "short_call_stop": stop_dollars,
            "short_put_stop": stop_dollars,
            "total_premium": total_premium,
            "stop_dollars": stop_dollars,
        }

    def create_oco_stop_structure(
        self,
        short_option_price: float,
        stop_loss_amount: float,
        stop_limit_gap: int = 40,
        stop_market_gap: int = 30,
    ) -> Dict[str, Any]:
        """
        Build OCO stop structure:
          - Stop-limit: trigger at stop_loss_amount, limit = trigger + stop_limit_gap
          - Stop-market: trigger = stop-limit trigger + stop_market_gap (backup)
        """
        trigger_price = short_option_price + stop_loss_amount
        limit_price = trigger_price + stop_limit_gap
        market_trigger = trigger_price + stop_market_gap

        return {
            "stop_limit": {
                "trigger_price": trigger_price,
                "limit_price": limit_price,
                "trigger_gap": stop_limit_gap,
            },
            "stop_market": {
                "trigger_price": market_trigger,
                "additional_gap": stop_market_gap,
            },
        }

    # -------------------------------------------------------------------------
    # Trailing stop
    # -------------------------------------------------------------------------

    def calculate_trailing_stop(
        self,
        original_stop_loss: float,
        original_long_value: float,
        current_long_value: float,
    ) -> float:
        """
        Tighten stop-loss as long option decays.
        New stop = original_stop - (original_long_value - current_long_value)
        """
        decay = original_long_value - current_long_value
        if decay <= 0:
            return original_stop_loss
        new_stop = original_stop_loss - decay
        return max(new_stop, 0.01)

    def calculate_trailing_stop_for_breakeven(
        self,
        total_premium: float,
        current_long_value: float,
    ) -> float:
        """
        Calculate stop that ensures break-even if triggered.
        Stop = total_premium - current_long_value (net cost to close)
        """
        return max(total_premium - current_long_value, 0.01)

    # -------------------------------------------------------------------------
    # Take-profit
    # -------------------------------------------------------------------------

    def should_take_profit(self, current_price: float) -> bool:
        """Return True if short option should be closed (at or below threshold)."""
        return current_price <= self.take_profit_threshold

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
