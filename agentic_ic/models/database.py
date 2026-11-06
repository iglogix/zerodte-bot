"""
Database models for 0DTE Iron Condor Trading Bot.
SQLAlchemy ORM models with SQLite backend.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, date
from typing import Optional, Generator

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    TypeDecorator,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class Base(DeclarativeBase):
    pass


class IronCondorTrade(Base):
    """Represents a single iron condor trade."""

    __tablename__ = "iron_condor_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    expiration_date = Column(Date, nullable=False)
    underlying = Column(String(10), nullable=False, default="SPX")

    underlying_price_at_entry = Column(Float, nullable=False)
    underlying_price_at_exit = Column(Float, nullable=True)

    short_call_strike = Column(Float, nullable=False)
    long_call_strike = Column(Float, nullable=False)
    short_put_strike = Column(Float, nullable=False)
    long_put_strike = Column(Float, nullable=False)

    call_spread_premium = Column(Float, nullable=False)
    put_spread_premium = Column(Float, nullable=False)
    total_premium = Column(Float, nullable=False)

    call_wing_width = Column(Integer, nullable=False)
    put_wing_width = Column(Integer, nullable=False)

    buying_power_used = Column(Float, nullable=True)

    stop_loss_amount = Column(Float, nullable=True)
    call_stop_order_id = Column(String(50), nullable=True)
    put_stop_order_id = Column(String(50), nullable=True)

    realized_pnl = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)

    status = Column(String(20), nullable=False, default="open", index=True)
    close_reason = Column(String(100), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class Order(Base):
    """Represents a single order (leg of a trade)."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    trade_id = Column(String(50), nullable=False, index=True)

    order_type = Column(String(30), nullable=False)
    side = Column(String(10), nullable=False)
    symbol = Column(String(50), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=True)

    fill_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)

    status = Column(String(20), nullable=False, default="pending", index=True)
    cancel_reason = Column(String(200), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.now)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)


class TradingSession(Base):
    """Represents a single trading day session."""

    __tablename__ = "trading_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_date = Column(Date, nullable=False, unique=True, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)

    account_equity_start = Column(Float, nullable=False)
    account_equity_end = Column(Float, nullable=True)

    trades_opened = Column(Integer, nullable=False, default=0)
    trades_closed = Column(Integer, nullable=False, default=0)
    max_concurrent_trades = Column(Integer, nullable=False, default=0)

    realized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)

    status = Column(String(20), nullable=False, default="active")

    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class AgentDecision(Base):
    """Audit log of all LLM decisions."""

    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_time = Column(DateTime, nullable=False, index=True)

    primary_llm = Column(String(50), nullable=False)
    secondary_llm = Column(String(50), nullable=False)

    action = Column(String(50), nullable=False)
    decision_data = Column(Text, nullable=False)

    primary_agrees = Column(Boolean, nullable=False)
    secondary_agrees = Column(Boolean, nullable=False)

    disagreement_reason = Column(Text, nullable=True)

    executed = Column(Boolean, nullable=False, default=False)
    execution_result = Column(String(100), nullable=True)

    error = Column(Boolean, nullable=False, default=False)
    error_reason = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.now)


class MarketDataCache(Base):
    """Cache for market data (option chains, quotes, candles)."""

    __tablename__ = "market_data_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    data_type = Column(String(30), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    data_json = Column(Text, nullable=False)
    ttl_seconds = Column(Integer, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.now)


class ToolCallLog(Base):
    """Audit log of all tool calls made by the agent."""

    __tablename__ = "tool_call_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_time = Column(DateTime, nullable=False, default=datetime.now, index=True)
    session_date = Column(Date, nullable=True, index=True)

    name = Column(String(100), nullable=False)
    arguments = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)


def get_engine(db_path: str):
    """Create a SQLAlchemy engine for the given SQLite path."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


@contextmanager
def get_session(engine) -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_path: str):
    """Initialize the database, creating all tables."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
