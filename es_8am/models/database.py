"""
Database models for 8AM ORB Break-and-Retest Bot.
SQLAlchemy ORM models with SQLite backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class OrbTrade(Base):
    """Represents a single ORB break-and-retest trade."""

    __tablename__ = "orb_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(10), nullable=False)
    session_date = Column(Date, nullable=False, index=True)

    orb_high = Column(Float, nullable=False)
    orb_low = Column(Float, nullable=False)
    orb_midline = Column(Float, nullable=False)
    orb_range_pts = Column(Float, nullable=False)

    bias = Column(String(10), nullable=True)
    breakout_time = Column(DateTime, nullable=True)
    breakout_price = Column(Float, nullable=True)

    entry_time = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=True)
    tp_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)

    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String(30), nullable=True)

    realized_pnl_pts = Column(Float, nullable=True)

    status = Column(String(20), nullable=False, default="watching", index=True)

    regime_verdict = Column(String(20), nullable=True)
    regime_reasoning = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class OrbToolCallLog(Base):
    """Audit log of tool calls made by the ORB agent."""

    __tablename__ = "orb_tool_call_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_time = Column(DateTime, nullable=False, default=datetime.now, index=True)
    session_date = Column(Date, nullable=True, index=True)

    name = Column(String(100), nullable=False)
    arguments = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)


class OrbSpreadTrade(Base):
    """Represents a single ORB-triggered SPX 0DTE credit spread trade."""

    __tablename__ = "orb_spread_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    session_date = Column(Date, nullable=False, index=True)

    # ORB levels
    orb_high = Column(Float, nullable=False)
    orb_low = Column(Float, nullable=False)
    orb_midline = Column(Float, nullable=False)
    orb_range_pts = Column(Float, nullable=False)

    # Bias + breakout
    bias = Column(String(10), nullable=True)
    breakout_time = Column(DateTime, nullable=True)
    breakout_price = Column(Float, nullable=True)

    # Spread identity
    spread_type = Column(String(20), nullable=True)    # put_credit_spread | call_credit_spread
    short_strike = Column(Float, nullable=True)
    long_strike = Column(Float, nullable=True)
    spread_width = Column(Float, nullable=True)
    short_symbol = Column(String(30), nullable=True)
    long_symbol = Column(String(30), nullable=True)
    expiration = Column(String(10), nullable=True)

    # Fill
    entry_time = Column(DateTime, nullable=True)
    credit_received = Column(Float, nullable=True)     # per-share premium collected
    max_profit = Column(Float, nullable=True)          # credit * 100
    max_loss = Column(Float, nullable=True)            # (width - credit) * 100
    spx_at_entry = Column(Float, nullable=True)

    # Exit
    exit_time = Column(DateTime, nullable=True)
    debit_paid = Column(Float, nullable=True)          # cost to close; 0 if expired worthless
    exit_reason = Column(String(30), nullable=True)    # tp_50pct | sl_200pct | expired_worthless | eod_close

    realized_pnl = Column(Float, nullable=True)        # dollars: (credit - debit) * 100

    status = Column(String(20), nullable=False, default="watching", index=True)

    regime_verdict = Column(String(20), nullable=True)
    regime_reasoning = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


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
