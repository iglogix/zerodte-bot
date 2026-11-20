"""
ORB Database service layer — async-friendly wrapper around SQLAlchemy models.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from es_8am.models.database import (
    OrbSpreadTrade,
    OrbToolCallLog,
    OrbTrade,
    get_session,
    init_db,
)

logger = logging.getLogger(__name__)


class OrbDatabaseService:
    """Async-friendly database service for ORB trades."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._engine = None

    def initialize(self):
        self._engine = init_db(self.db_path)
        logger.info("ORB database initialized at %s", self.db_path)

    def _session(self):
        return get_session(self._engine)

    # -------------------------------------------------------------------------
    # Trade operations
    # -------------------------------------------------------------------------

    async def get_trade_for_date(self, session_date: date) -> Optional[Dict[str, Any]]:
        """Return the trade record for today (at most one trade per day)."""
        with self._session() as session:
            trade = (
                session.query(OrbTrade)
                .filter_by(session_date=session_date)
                .first()
            )
            return self._trade_to_dict(trade) if trade else None

    async def traded_today(self, session_date: date) -> bool:
        """Return True if a non-skipped trade exists for today."""
        with self._session() as session:
            count = (
                session.query(OrbTrade)
                .filter(
                    OrbTrade.session_date == session_date,
                    OrbTrade.status.in_(["entered", "closed"]),
                )
                .count()
            )
            return count > 0

    async def create_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._session() as session:
            trade = OrbTrade(**trade_data)
            session.add(trade)
            session.flush()
            return self._trade_to_dict(trade)

    async def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        with self._session() as session:
            trade = session.query(OrbTrade).filter_by(trade_id=trade_id).first()
            if not trade:
                return False
            for key, value in updates.items():
                setattr(trade, key, value)
            trade.updated_at = datetime.now()
            return True

    async def get_closed_trades(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        with self._session() as session:
            q = session.query(OrbTrade).filter_by(status="closed")
            if start_date:
                q = q.filter(OrbTrade.session_date >= start_date)
            if end_date:
                q = q.filter(OrbTrade.session_date <= end_date)
            return [self._trade_to_dict(t) for t in q.order_by(OrbTrade.session_date).all()]

    def _trade_to_dict(self, trade: OrbTrade) -> Dict[str, Any]:
        return {
            "id": trade.id,
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "session_date": trade.session_date,
            "orb_high": trade.orb_high,
            "orb_low": trade.orb_low,
            "orb_midline": trade.orb_midline,
            "orb_range_pts": trade.orb_range_pts,
            "bias": trade.bias,
            "breakout_time": trade.breakout_time,
            "breakout_price": trade.breakout_price,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "tp_price": trade.tp_price,
            "sl_price": trade.sl_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "realized_pnl_pts": trade.realized_pnl_pts,
            "status": trade.status,
            "regime_verdict": trade.regime_verdict,
            "regime_reasoning": trade.regime_reasoning,
        }

    # -------------------------------------------------------------------------
    # Spread trade operations (options variant)
    # -------------------------------------------------------------------------

    async def create_spread_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._session() as session:
            trade = OrbSpreadTrade(**trade_data)
            session.add(trade)
            session.flush()
            return self._spread_to_dict(trade)

    async def update_spread_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        with self._session() as session:
            trade = session.query(OrbSpreadTrade).filter_by(trade_id=trade_id).first()
            if not trade:
                return False
            for key, value in updates.items():
                setattr(trade, key, value)
            trade.updated_at = datetime.now()
            return True

    async def spread_traded_today(self, session_date: date) -> bool:
        with self._session() as session:
            count = (
                session.query(OrbSpreadTrade)
                .filter(
                    OrbSpreadTrade.session_date == session_date,
                    OrbSpreadTrade.status.in_(["entered", "closed"]),
                )
                .count()
            )
            return count > 0

    async def get_closed_spread_trades(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        with self._session() as session:
            q = session.query(OrbSpreadTrade).filter_by(status="closed")
            if start_date:
                q = q.filter(OrbSpreadTrade.session_date >= start_date)
            if end_date:
                q = q.filter(OrbSpreadTrade.session_date <= end_date)
            return [self._spread_to_dict(t) for t in q.order_by(OrbSpreadTrade.session_date).all()]

    def _spread_to_dict(self, trade: OrbSpreadTrade) -> Dict[str, Any]:
        return {
            "id": trade.id,
            "trade_id": trade.trade_id,
            "session_date": trade.session_date,
            "orb_high": trade.orb_high,
            "orb_low": trade.orb_low,
            "orb_midline": trade.orb_midline,
            "orb_range_pts": trade.orb_range_pts,
            "bias": trade.bias,
            "breakout_time": trade.breakout_time,
            "breakout_price": trade.breakout_price,
            "spread_type": trade.spread_type,
            "short_strike": trade.short_strike,
            "long_strike": trade.long_strike,
            "spread_width": trade.spread_width,
            "short_symbol": trade.short_symbol,
            "long_symbol": trade.long_symbol,
            "expiration": trade.expiration,
            "entry_time": trade.entry_time,
            "credit_received": trade.credit_received,
            "max_profit": trade.max_profit,
            "max_loss": trade.max_loss,
            "spx_at_entry": trade.spx_at_entry,
            "exit_time": trade.exit_time,
            "debit_paid": trade.debit_paid,
            "exit_reason": trade.exit_reason,
            "realized_pnl": trade.realized_pnl,
            "status": trade.status,
            "regime_verdict": trade.regime_verdict,
            "regime_reasoning": trade.regime_reasoning,
        }

    # -------------------------------------------------------------------------
    # Tool call logging
    # -------------------------------------------------------------------------

    async def log_tool_call(self, data: Dict[str, Any]):
        with self._session() as session:
            log = OrbToolCallLog(
                call_time=data.get("call_time", datetime.now()),
                session_date=data.get("session_date", date.today()),
                name=data["name"],
                arguments=json.dumps(data.get("arguments", {}), default=str),
                result=json.dumps(data.get("result", {}), default=str),
                error=json.dumps(data.get("error")) if data.get("error") else None,
                duration_ms=data.get("duration_ms"),
            )
            session.add(log)
