"""
Database Model Tests for 0DTE Trading Bot.
DO NOT MODIFY after approval - this file is part of the immutable test suite.

Tests cover:
- IronCondorTrade CRUD operations
- Order lifecycle (pending → filled → closed)
- TradingSession daily aggregation
- AgentDecision audit logging
- MarketDataCache persistence and retrieval
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class TestIronCondorTrade:
    """Tests for IronCondorTrade model CRUD operations."""

    def test_create_iron_condor_trade(self, temp_db_path, sample_iron_condor_trade):
        """Test creating a new iron condor trade."""
        from src.models.database import Base, IronCondorTrade, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            trade = IronCondorTrade(
                trade_id=sample_iron_condor_trade["trade_id"],
                entry_time=sample_iron_condor_trade["entry_time"],
                expiration_date=sample_iron_condor_trade["expiration_date"],
                underlying=sample_iron_condor_trade["underlying"],
                underlying_price_at_entry=sample_iron_condor_trade["underlying_price_at_entry"],
                short_call_strike=sample_iron_condor_trade["short_call_strike"],
                long_call_strike=sample_iron_condor_trade["long_call_strike"],
                short_put_strike=sample_iron_condor_trade["short_put_strike"],
                long_put_strike=sample_iron_condor_trade["long_put_strike"],
                call_spread_premium=sample_iron_condor_trade["call_spread_premium"],
                put_spread_premium=sample_iron_condor_trade["put_spread_premium"],
                total_premium=sample_iron_condor_trade["total_premium"],
                call_wing_width=sample_iron_condor_trade["call_wing_width"],
                put_wing_width=sample_iron_condor_trade["put_wing_width"],
                status=sample_iron_condor_trade["status"],
            )
            session.add(trade)
            session.commit()
            
            assert trade.id is not None
            assert trade.trade_id == "IC-20240222-001"
            assert trade.status == "open"

    def test_read_iron_condor_trade(self, temp_db_path, sample_iron_condor_trade):
        """Test reading an existing iron condor trade."""
        from src.models.database import Base, IronCondorTrade, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            trade = IronCondorTrade(
                trade_id=sample_iron_condor_trade["trade_id"],
                entry_time=sample_iron_condor_trade["entry_time"],
                expiration_date=sample_iron_condor_trade["expiration_date"],
                underlying=sample_iron_condor_trade["underlying"],
                underlying_price_at_entry=sample_iron_condor_trade["underlying_price_at_entry"],
                short_call_strike=sample_iron_condor_trade["short_call_strike"],
                long_call_strike=sample_iron_condor_trade["long_call_strike"],
                short_put_strike=sample_iron_condor_trade["short_put_strike"],
                long_put_strike=sample_iron_condor_trade["long_put_strike"],
                call_spread_premium=sample_iron_condor_trade["call_spread_premium"],
                put_spread_premium=sample_iron_condor_trade["put_spread_premium"],
                total_premium=sample_iron_condor_trade["total_premium"],
                call_wing_width=sample_iron_condor_trade["call_wing_width"],
                put_wing_width=sample_iron_condor_trade["put_wing_width"],
                status=sample_iron_condor_trade["status"],
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
        
        with get_session(engine) as session:
            retrieved = session.query(IronCondorTrade).filter_by(id=trade_id).first()
            assert retrieved is not None
            assert retrieved.trade_id == "IC-20240222-001"
            assert retrieved.short_call_strike == 5890.0
            assert retrieved.short_put_strike == 5810.0
            assert retrieved.total_premium == 3.00

    def test_update_iron_condor_trade_status(self, temp_db_path, sample_iron_condor_trade):
        """Test updating iron condor trade status."""
        from src.models.database import Base, IronCondorTrade, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            trade = IronCondorTrade(
                trade_id=sample_iron_condor_trade["trade_id"],
                entry_time=sample_iron_condor_trade["entry_time"],
                expiration_date=sample_iron_condor_trade["expiration_date"],
                underlying=sample_iron_condor_trade["underlying"],
                underlying_price_at_entry=sample_iron_condor_trade["underlying_price_at_entry"],
                short_call_strike=sample_iron_condor_trade["short_call_strike"],
                long_call_strike=sample_iron_condor_trade["long_call_strike"],
                short_put_strike=sample_iron_condor_trade["short_put_strike"],
                long_put_strike=sample_iron_condor_trade["long_put_strike"],
                call_spread_premium=sample_iron_condor_trade["call_spread_premium"],
                put_spread_premium=sample_iron_condor_trade["put_spread_premium"],
                total_premium=sample_iron_condor_trade["total_premium"],
                call_wing_width=sample_iron_condor_trade["call_wing_width"],
                put_wing_width=sample_iron_condor_trade["put_wing_width"],
                status="open",
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
        
        with get_session(engine) as session:
            trade = session.query(IronCondorTrade).filter_by(id=trade_id).first()
            trade.status = "closed"
            trade.exit_time = datetime.now()
            trade.realized_pnl = 2.50
            session.commit()
        
        with get_session(engine) as session:
            trade = session.query(IronCondorTrade).filter_by(id=trade_id).first()
            assert trade.status == "closed"
            assert trade.exit_time is not None
            assert trade.realized_pnl == 2.50

    def test_delete_iron_condor_trade(self, temp_db_path, sample_iron_condor_trade):
        """Test deleting an iron condor trade."""
        from src.models.database import Base, IronCondorTrade, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            trade = IronCondorTrade(
                trade_id=sample_iron_condor_trade["trade_id"],
                entry_time=sample_iron_condor_trade["entry_time"],
                expiration_date=sample_iron_condor_trade["expiration_date"],
                underlying=sample_iron_condor_trade["underlying"],
                underlying_price_at_entry=sample_iron_condor_trade["underlying_price_at_entry"],
                short_call_strike=sample_iron_condor_trade["short_call_strike"],
                long_call_strike=sample_iron_condor_trade["long_call_strike"],
                short_put_strike=sample_iron_condor_trade["short_put_strike"],
                long_put_strike=sample_iron_condor_trade["long_put_strike"],
                call_spread_premium=sample_iron_condor_trade["call_spread_premium"],
                put_spread_premium=sample_iron_condor_trade["put_spread_premium"],
                total_premium=sample_iron_condor_trade["total_premium"],
                call_wing_width=sample_iron_condor_trade["call_wing_width"],
                put_wing_width=sample_iron_condor_trade["put_wing_width"],
                status="open",
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
        
        with get_session(engine) as session:
            trade = session.query(IronCondorTrade).filter_by(id=trade_id).first()
            session.delete(trade)
            session.commit()
        
        with get_session(engine) as session:
            trade = session.query(IronCondorTrade).filter_by(id=trade_id).first()
            assert trade is None

    def test_query_open_trades(self, temp_db_path):
        """Test querying only open trades."""
        from src.models.database import Base, IronCondorTrade, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            for i in range(5):
                trade = IronCondorTrade(
                    trade_id=f"IC-20240222-00{i}",
                    entry_time=datetime.now(),
                    expiration_date=date.today(),
                    underlying="SPX",
                    underlying_price_at_entry=5850.0,
                    short_call_strike=5890.0,
                    long_call_strike=5920.0,
                    short_put_strike=5810.0,
                    long_put_strike=5780.0,
                    call_spread_premium=1.50,
                    put_spread_premium=1.50,
                    total_premium=3.00,
                    call_wing_width=30,
                    put_wing_width=30,
                    status="open" if i < 3 else "closed",
                )
                session.add(trade)
            session.commit()
        
        with get_session(engine) as session:
            open_trades = session.query(IronCondorTrade).filter_by(status="open").all()
            assert len(open_trades) == 3


class TestOrder:
    """Tests for Order model lifecycle."""

    def test_create_order_pending(self, temp_db_path):
        """Test creating a pending order."""
        from src.models.database import Base, Order, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            order = Order(
                order_id="ORD-001",
                trade_id="IC-20240222-001",
                order_type="credit_spread",
                side="sell",
                symbol="SPXW240222C5890",
                quantity=1,
                price=1.50,
                status="pending",
                created_at=datetime.now(),
            )
            session.add(order)
            session.commit()
            
            assert order.id is not None
            assert order.status == "pending"

    def test_order_lifecycle_pending_to_filled(self, temp_db_path):
        """Test order status transition from pending to filled."""
        from src.models.database import Base, Order, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            order = Order(
                order_id="ORD-001",
                trade_id="IC-20240222-001",
                order_type="credit_spread",
                side="sell",
                symbol="SPXW240222C5890",
                quantity=1,
                price=1.50,
                status="pending",
                created_at=datetime.now(),
            )
            session.add(order)
            session.commit()
            order_id = order.id
        
        with get_session(engine) as session:
            order = session.query(Order).filter_by(id=order_id).first()
            order.status = "filled"
            order.filled_at = datetime.now()
            order.fill_price = 1.48
            session.commit()
        
        with get_session(engine) as session:
            order = session.query(Order).filter_by(id=order_id).first()
            assert order.status == "filled"
            assert order.filled_at is not None
            assert order.fill_price == 1.48

    def test_order_lifecycle_pending_to_cancelled(self, temp_db_path):
        """Test order status transition from pending to cancelled."""
        from src.models.database import Base, Order, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            order = Order(
                order_id="ORD-001",
                trade_id="IC-20240222-001",
                order_type="stop_loss",
                side="buy",
                symbol="SPXW240222C5890",
                quantity=1,
                price=4.50,
                status="pending",
                created_at=datetime.now(),
            )
            session.add(order)
            session.commit()
            order_id = order.id
        
        with get_session(engine) as session:
            order = session.query(Order).filter_by(id=order_id).first()
            order.status = "cancelled"
            order.cancelled_at = datetime.now()
            order.cancel_reason = "OCO triggered"
            session.commit()
        
        with get_session(engine) as session:
            order = session.query(Order).filter_by(id=order_id).first()
            assert order.status == "cancelled"
            assert order.cancelled_at is not None
            assert order.cancel_reason == "OCO triggered"

    def test_order_lifecycle_filled_to_closed(self, temp_db_path):
        """Test order status transition from filled to closed."""
        from src.models.database import Base, Order, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            order = Order(
                order_id="ORD-001",
                trade_id="IC-20240222-001",
                order_type="credit_spread",
                side="sell",
                symbol="SPXW240222C5890",
                quantity=1,
                price=1.50,
                status="filled",
                created_at=datetime.now(),
                filled_at=datetime.now(),
                fill_price=1.48,
            )
            session.add(order)
            session.commit()
            order_id = order.id
        
        with get_session(engine) as session:
            order = session.query(Order).filter_by(id=order_id).first()
            order.status = "closed"
            order.closed_at = datetime.now()
            order.close_price = 0.05
            order.realized_pnl = 1.43
            session.commit()
        
        with get_session(engine) as session:
            order = session.query(Order).filter_by(id=order_id).first()
            assert order.status == "closed"
            assert order.close_price == 0.05
            assert order.realized_pnl == 1.43


class TestTradingSession:
    """Tests for TradingSession daily aggregation."""

    def test_create_trading_session(self, temp_db_path):
        """Test creating a new trading session."""
        from src.models.database import Base, TradingSession, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            trading_session = TradingSession(
                session_date=date.today(),
                start_time=datetime.now(),
                account_equity_start=100000.0,
                trades_opened=0,
                trades_closed=0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                max_concurrent_trades=0,
                status="active",
            )
            session.add(trading_session)
            session.commit()
            
            assert trading_session.id is not None
            assert trading_session.status == "active"

    def test_update_trading_session_aggregates(self, temp_db_path):
        """Test updating trading session with aggregated data."""
        from src.models.database import Base, TradingSession, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            trading_session = TradingSession(
                session_date=date.today(),
                start_time=datetime.now(),
                account_equity_start=100000.0,
                trades_opened=0,
                trades_closed=0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                max_concurrent_trades=0,
                status="active",
            )
            session.add(trading_session)
            session.commit()
            session_id = trading_session.id
        
        with get_session(engine) as session:
            trading_session = session.query(TradingSession).filter_by(id=session_id).first()
            trading_session.trades_opened = 5
            trading_session.trades_closed = 3
            trading_session.realized_pnl = 450.0
            trading_session.unrealized_pnl = 120.0
            trading_session.max_concurrent_trades = 5
            session.commit()
        
        with get_session(engine) as session:
            trading_session = session.query(TradingSession).filter_by(id=session_id).first()
            assert trading_session.trades_opened == 5
            assert trading_session.trades_closed == 3
            assert trading_session.realized_pnl == 450.0
            assert trading_session.max_concurrent_trades == 5

    def test_close_trading_session(self, temp_db_path):
        """Test closing a trading session at end of day."""
        from src.models.database import Base, TradingSession, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            trading_session = TradingSession(
                session_date=date.today(),
                start_time=datetime.now(),
                account_equity_start=100000.0,
                trades_opened=5,
                trades_closed=5,
                realized_pnl=450.0,
                unrealized_pnl=0.0,
                max_concurrent_trades=5,
                status="active",
            )
            session.add(trading_session)
            session.commit()
            session_id = trading_session.id
        
        with get_session(engine) as session:
            trading_session = session.query(TradingSession).filter_by(id=session_id).first()
            trading_session.status = "closed"
            trading_session.end_time = datetime.now()
            trading_session.account_equity_end = 100450.0
            session.commit()
        
        with get_session(engine) as session:
            trading_session = session.query(TradingSession).filter_by(id=session_id).first()
            assert trading_session.status == "closed"
            assert trading_session.end_time is not None
            assert trading_session.account_equity_end == 100450.0


class TestAgentDecision:
    """Tests for AgentDecision audit logging."""

    def test_log_agent_decision(self, temp_db_path, sample_llm_decision_enter):
        """Test logging an agent decision."""
        from src.models.database import Base, AgentDecision, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            decision = AgentDecision(
                decision_time=datetime.now(),
                primary_llm="claude-sonnet-4.5",
                secondary_llm="gpt-4.1",
                action=sample_llm_decision_enter["action"],
                decision_data=json.dumps(sample_llm_decision_enter),
                primary_agrees=True,
                secondary_agrees=True,
                executed=True,
                execution_result="success",
            )
            session.add(decision)
            session.commit()
            
            assert decision.id is not None
            assert decision.executed is True

    def test_log_disagreement_decision(self, temp_db_path):
        """Test logging a decision where LLMs disagreed."""
        from src.models.database import Base, AgentDecision, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            decision = AgentDecision(
                decision_time=datetime.now(),
                primary_llm="claude-sonnet-4.5",
                secondary_llm="gpt-4.1",
                action="enter_trade",
                decision_data=json.dumps({"primary": "enter", "secondary": "wait"}),
                primary_agrees=True,
                secondary_agrees=False,
                disagreement_reason="Secondary LLM detected market instability",
                executed=False,
                execution_result="blocked_disagreement",
            )
            session.add(decision)
            session.commit()
            
            assert decision.executed is False
            assert decision.disagreement_reason is not None

    def test_query_decisions_by_date(self, temp_db_path):
        """Test querying decisions by date range."""
        from src.models.database import Base, AgentDecision, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        with get_session(engine) as session:
            for i in range(3):
                decision = AgentDecision(
                    decision_time=today - timedelta(hours=i),
                    primary_llm="claude-sonnet-4.5",
                    secondary_llm="gpt-4.1",
                    action="enter_trade",
                    decision_data="{}",
                    primary_agrees=True,
                    secondary_agrees=True,
                    executed=True,
                )
                session.add(decision)
            
            for i in range(2):
                decision = AgentDecision(
                    decision_time=yesterday - timedelta(hours=i),
                    primary_llm="claude-sonnet-4.5",
                    secondary_llm="gpt-4.1",
                    action="wait",
                    decision_data="{}",
                    primary_agrees=True,
                    secondary_agrees=True,
                    executed=False,
                )
                session.add(decision)
            session.commit()
        
        with get_session(engine) as session:
            today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
            today_decisions = session.query(AgentDecision).filter(
                AgentDecision.decision_time >= today_start
            ).all()
            assert len(today_decisions) == 3


class TestMarketDataCache:
    """Tests for MarketDataCache persistence and retrieval."""

    def test_cache_option_chain(self, temp_db_path, sample_option_chain):
        """Test caching option chain data."""
        from src.models.database import Base, MarketDataCache, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        with get_session(engine) as session:
            cache_entry = MarketDataCache(
                symbol="SPX",
                data_type="option_chain",
                timestamp=datetime.now(),
                data_json=json.dumps(sample_option_chain),
                ttl_seconds=60,
            )
            session.add(cache_entry)
            session.commit()
            
            assert cache_entry.id is not None

    def test_retrieve_cached_data(self, temp_db_path, sample_option_chain):
        """Test retrieving cached data."""
        from src.models.database import Base, MarketDataCache, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        cache_time = datetime.now()
        
        with get_session(engine) as session:
            cache_entry = MarketDataCache(
                symbol="SPX",
                data_type="option_chain",
                timestamp=cache_time,
                data_json=json.dumps(sample_option_chain),
                ttl_seconds=60,
            )
            session.add(cache_entry)
            session.commit()
        
        with get_session(engine) as session:
            retrieved = session.query(MarketDataCache).filter_by(
                symbol="SPX",
                data_type="option_chain"
            ).order_by(MarketDataCache.timestamp.desc()).first()
            
            assert retrieved is not None
            data = json.loads(retrieved.data_json)
            assert len(data) == len(sample_option_chain)
            assert data[0]["symbol"] == "SPXW240222C5880"

    def test_cache_expiry_check(self, temp_db_path):
        """Test checking if cached data has expired."""
        from src.models.database import Base, MarketDataCache, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        old_time = datetime.now() - timedelta(seconds=120)
        
        with get_session(engine) as session:
            cache_entry = MarketDataCache(
                symbol="SPX",
                data_type="quote",
                timestamp=old_time,
                data_json='{"price": 5850.0}',
                ttl_seconds=60,
            )
            session.add(cache_entry)
            session.commit()
        
        with get_session(engine) as session:
            retrieved = session.query(MarketDataCache).filter_by(
                symbol="SPX",
                data_type="quote"
            ).order_by(MarketDataCache.timestamp.desc()).first()
            
            is_expired = (datetime.now() - retrieved.timestamp).total_seconds() > retrieved.ttl_seconds
            assert is_expired is True

    def test_cache_5min_candles_permanent(self, temp_db_path, sample_5min_candles):
        """Test that 5-min candles are stored permanently (no TTL expiry)."""
        from src.models.database import Base, MarketDataCache, get_engine, get_session
        
        engine = get_engine(temp_db_path)
        Base.metadata.create_all(engine)
        
        old_time = datetime.now() - timedelta(days=7)
        
        with get_session(engine) as session:
            cache_entry = MarketDataCache(
                symbol="SPX",
                data_type="5min_candle",
                timestamp=old_time,
                data_json=json.dumps(sample_5min_candles[0]),
                ttl_seconds=None,
            )
            session.add(cache_entry)
            session.commit()
        
        with get_session(engine) as session:
            retrieved = session.query(MarketDataCache).filter_by(
                symbol="SPX",
                data_type="5min_candle"
            ).first()
            
            assert retrieved is not None
            assert retrieved.ttl_seconds is None
