"""
Database service layer — async wrapper around SQLAlchemy models.
Provides clean async interface for the agent and risk manager.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, time as dt_time
from typing import Any, Dict, List, Optional

from src.models.database import (
    AgentDecision,
    IronCondorTrade,
    MarketDataCache,
    Order,
    ToolCallLog,
    TradingSession,
    get_engine,
    get_session,
    init_db,
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """Async-friendly database service wrapping SQLAlchemy sync operations."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._engine = None

    def initialize(self):
        self._engine = init_db(self.db_path)
        logger.info("Database initialized at %s", self.db_path)

    def _session(self):
        return get_session(self._engine)

    # -------------------------------------------------------------------------
    # Trade operations
    # -------------------------------------------------------------------------

    async def get_open_trade_count(self) -> int:
        with self._session() as session:
            return session.query(IronCondorTrade).filter_by(status="open").count()

    async def get_open_trades(self) -> List[Dict[str, Any]]:
        with self._session() as session:
            trades = session.query(IronCondorTrade).filter_by(status="open").all()
            return [self._trade_to_dict(t) for t in trades]

    async def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        with self._session() as session:
            trade = session.query(IronCondorTrade).filter_by(trade_id=trade_id).first()
            return self._trade_to_dict(trade) if trade else None

    async def create_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._session() as session:
            trade = IronCondorTrade(**trade_data)
            session.add(trade)
            session.flush()
            return self._trade_to_dict(trade)

    async def get_all_trades_for_date(self, session_date: date) -> List[Dict[str, Any]]:
        with self._session() as session:
            start = datetime.combine(session_date, dt_time(0, 0, 0))
            end = datetime.combine(session_date, dt_time(23, 59, 59))
            trades = (
                session.query(IronCondorTrade)
                .filter(IronCondorTrade.entry_time >= start, IronCondorTrade.entry_time <= end)
                .all()
            )
            return [self._trade_to_dict(t) for t in trades]

    async def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        with self._session() as session:
            trade = session.query(IronCondorTrade).filter_by(trade_id=trade_id).first()
            if not trade:
                return False
            for key, value in updates.items():
                setattr(trade, key, value)
            return True

    def _trade_to_dict(self, trade: IronCondorTrade) -> Dict[str, Any]:
        return {
            "id": trade.id,
            "trade_id": trade.trade_id,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "expiration_date": trade.expiration_date,
            "underlying": trade.underlying,
            "underlying_price_at_entry": trade.underlying_price_at_entry,
            "short_call_strike": trade.short_call_strike,
            "long_call_strike": trade.long_call_strike,
            "short_put_strike": trade.short_put_strike,
            "long_put_strike": trade.long_put_strike,
            "call_spread_premium": trade.call_spread_premium,
            "put_spread_premium": trade.put_spread_premium,
            "total_premium": trade.total_premium,
            "call_wing_width": trade.call_wing_width,
            "put_wing_width": trade.put_wing_width,
            "buying_power_used": trade.buying_power_used,
            "stop_loss_amount": trade.stop_loss_amount,
            "realized_pnl": trade.realized_pnl,
            "unrealized_pnl": trade.unrealized_pnl,
            "status": trade.status,
            "close_reason": trade.close_reason,
        }

    # -------------------------------------------------------------------------
    # Decision logging
    # -------------------------------------------------------------------------

    async def log_decision(self, data: Dict[str, Any]):
        with self._session() as session:
            decision = AgentDecision(
                decision_time=data.get("decision_time", datetime.now()),
                primary_llm=data.get("primary_llm", "claude-sonnet-4.6"),
                secondary_llm=data.get("secondary_llm", "gpt-5.2"),
                action=data.get("action", "unknown"),
                decision_data=json.dumps(data, default=str),
                primary_agrees=data.get("primary_agrees", False),
                secondary_agrees=data.get("secondary_agrees", False),
                disagreement_reason=data.get("disagreement_reason"),
                executed=data.get("executed", False),
                execution_result=data.get("execution_result"),
                error=data.get("error", False),
                error_reason=data.get("error_reason"),
            )
            session.add(decision)

    # -------------------------------------------------------------------------
    # Tool call logging
    # -------------------------------------------------------------------------

    async def log_tool_call(self, data: Dict[str, Any]):
        with self._session() as session:
            log = ToolCallLog(
                call_time=data.get("call_time", datetime.now()),
                session_date=data.get("session_date", date.today()),
                name=data["name"],
                arguments=json.dumps(data.get("arguments", {}), default=str),
                result=json.dumps(data.get("result", {}), default=str),
                error=json.dumps(data.get("error")) if data.get("error") else None,
                duration_ms=data.get("duration_ms"),
            )
            session.add(log)

    async def log_simulated_action(self, action: Dict[str, Any]):
        await self.log_tool_call({**action, "simulated": True})

    # -------------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------------

    async def get_or_create_session(self, session_date: date, equity: float) -> Dict[str, Any]:
        with self._session() as session:
            ts = session.query(TradingSession).filter_by(session_date=session_date).first()
            if ts:
                return self._session_to_dict(ts)
            ts = TradingSession(
                session_date=session_date,
                start_time=datetime.now(),
                account_equity_start=equity,
                status="active",
            )
            session.add(ts)
            session.flush()
            return self._session_to_dict(ts)

    def _session_to_dict(self, ts: TradingSession) -> Dict[str, Any]:
        return {
            "id": ts.id,
            "session_date": ts.session_date,
            "start_time": ts.start_time,
            "end_time": ts.end_time,
            "account_equity_start": ts.account_equity_start,
            "account_equity_end": ts.account_equity_end,
            "trades_opened": ts.trades_opened,
            "trades_closed": ts.trades_closed,
            "realized_pnl": ts.realized_pnl,
            "unrealized_pnl": ts.unrealized_pnl,
            "max_concurrent_trades": ts.max_concurrent_trades,
            "status": ts.status,
        }
