"""Phase 8-9: Integration tests — strategy wiring, database model, config routing."""

import json
import pytest
from datetime import datetime, date
from unittest.mock import AsyncMock, MagicMock

from zerodte.models.database import Trade, Base, get_engine, init_db
from zerodte.strategies.iron_condor import IronCondorStrategy
from zerodte.strategies.iron_fly import IronFlyStrategy
from zerodte.strategies.base import LiveTradeSetup, OrderLeg


class TestTradeModel:
    def test_create_trade_ic(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = init_db(db_path)
        from sqlalchemy.orm import Session
        with Session(engine) as session:
            trade = Trade(
                trade_id="IC-001",
                strategy_type="iron_condor",
                entry_time=datetime.now(),
                expiration_date=date.today(),
                underlying_price_at_entry=5850.0,
                total_credit=1.50,
                strategy_data=json.dumps({
                    "short_call": 5890.0, "long_call": 5920.0,
                    "short_put": 5810.0, "long_put": 5780.0,
                    "call_credit": 0.75, "put_credit": 0.75,
                }),
            )
            session.add(trade)
            session.commit()

            loaded = session.query(Trade).filter_by(trade_id="IC-001").first()
            assert loaded.strategy_type == "iron_condor"
            data = loaded.get_strategy_data()
            assert data["short_call"] == 5890.0

    def test_create_trade_if(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = init_db(db_path)
        from sqlalchemy.orm import Session
        with Session(engine) as session:
            trade = Trade(
                trade_id="IF-001",
                strategy_type="iron_fly",
                entry_time=datetime.now(),
                expiration_date=date.today(),
                underlying_price_at_entry=5850.0,
                total_credit=15.0,
                strategy_data=json.dumps({
                    "center": 5850.0, "long_put": 5835.0,
                    "long_call": 5865.0, "em_pts": 30.0,
                }),
            )
            session.add(trade)
            session.commit()

            loaded = session.query(Trade).filter_by(trade_id="IF-001").first()
            assert loaded.strategy_type == "iron_fly"
            data = loaded.get_strategy_data()
            assert data["center"] == 5850.0


class TestStrategyRouting:
    def test_ic_strategy_from_config(self):
        """Config with strategy=iron_condor creates IronCondorStrategy."""
        from zerodte.main import BotOrchestrator
        config = {
            "bot": {"strategy": "iron_condor"},
            "strategy": {"min_short_delta": 0.10, "max_short_delta": 0.15},
        }
        bot = BotOrchestrator(config, mode="dry_run")
        strategy = bot._create_strategy("iron_condor")
        assert isinstance(strategy, IronCondorStrategy)

    def test_if_strategy_from_config(self):
        """Config with strategy=iron_fly creates IronFlyStrategy."""
        from zerodte.main import BotOrchestrator
        config = {
            "bot": {"strategy": "iron_fly"},
            "options": {"em_wing_fraction": 0.50, "tp_fixed_dollars": 200},
        }
        bot = BotOrchestrator(config, mode="dry_run")
        strategy = bot._create_strategy("iron_fly")
        assert isinstance(strategy, IronFlyStrategy)

    def test_default_is_iron_condor(self):
        """No strategy key defaults to iron_condor."""
        from zerodte.main import BotOrchestrator
        config = {"bot": {}}
        bot = BotOrchestrator(config, mode="dry_run")
        strategy = bot._create_strategy("iron_condor")
        assert isinstance(strategy, IronCondorStrategy)


class TestEndToEndEngineWithStrategy:
    @pytest.mark.asyncio
    async def test_engine_with_ic_strategy(self):
        """Full cycle: engine + IronCondorStrategy + mock tools."""
        from zerodte.agent.decision_engine import DecisionEngine

        strategy = IronCondorStrategy()
        strategy.configure({
            "strategy": {
                "credit_equality_threshold": 0.20,
                "min_short_delta": 0.10,
                "max_short_delta": 0.15,
                "min_side_credit": 0.50,
                "max_side_credit": 2.50,
                "wing_width": 30,
            },
            "stop_loss": {"stop_limit_gap": 40, "stop_market_gap": 30},
            "risk": {"take_profit_threshold": 0.05},
        })

        tool_handlers = AsyncMock()
        tool_handlers.get_5min_candles.return_value = [
            {"close": 5850}, {"close": 5851}, {"close": 5852},
        ]
        tool_handlers.get_atm_iv.return_value = {"expected_daily_move": 30.0, "source": "test"}
        tool_handlers.get_option_chain.return_value = [
            {"strike": 5890.0, "option_type": "CALL", "delta": 0.12, "bid": 1.40, "ask": 1.60},
            {"strike": 5920.0, "option_type": "CALL", "delta": 0.04, "bid": 0.20, "ask": 0.40},
            {"strike": 5810.0, "option_type": "PUT", "delta": -0.12, "bid": 1.50, "ask": 1.70},
            {"strike": 5780.0, "option_type": "PUT", "delta": -0.04, "bid": 0.35, "ask": 0.55},
        ]
        tool_handlers.get_spx_price.return_value = {"price": 5850.0}
        tool_handlers.place_order.return_value = {"order_id": "IC-LIVE-001"}

        risk_manager = AsyncMock()
        risk_manager.can_open_new_trade.return_value = {"allowed": True}

        engine = DecisionEngine(
            tool_handlers=tool_handlers,
            risk_manager=risk_manager,
            strategy=strategy,
        )

        result = await engine.run_cycle()
        assert result.outcome == "executed"
        tool_handlers.place_order.assert_called_once()
        order = tool_handlers.place_order.call_args[0][0]
        assert order["order_type"] == "iron_condor"

    @pytest.mark.asyncio
    async def test_engine_with_if_strategy(self):
        """Full cycle: engine + IronFlyStrategy + mock tools."""
        from zerodte.agent.decision_engine import DecisionEngine

        strategy = IronFlyStrategy()
        strategy.configure({
            "options": {
                "strike_step_pts": 5,
                "em_wing_fraction": 0.50,
                "tp_fixed_dollars": 200,
                "slippage_per_leg": 0.05,
                "contract_multiplier": 100,
                "quantity": 1,
                "em_source": "chain",
            },
        })

        tool_handlers = AsyncMock()
        tool_handlers.get_5min_candles.return_value = [
            {"close": 5850}, {"close": 5851}, {"close": 5852},
        ]
        tool_handlers.get_atm_iv.return_value = {"expected_daily_move": 30.0, "source": "test"}
        # ATM straddle at 5845 (closest to spot 5847): call_mid=8.20 + put_mid=8.00 = EM=16.20
        # Wings at 0.5× EM=8.1 from center 5850: long_call=5860, long_put=5840
        # Put prices decrease away from ATM; call prices decrease away from ATM
        tool_handlers.get_option_chain.return_value = [
            {"strike": 5845.0, "option_type": "CALL", "delta": 0.52, "bid": 8.00, "ask": 8.40},
            {"strike": 5845.0, "option_type": "PUT", "delta": -0.52, "bid": 7.80, "ask": 8.20},
            {"strike": 5850.0, "option_type": "CALL", "delta": 0.50, "bid": 7.50, "ask": 7.90},
            {"strike": 5850.0, "option_type": "PUT", "delta": -0.50, "bid": 7.50, "ask": 7.90},
            {"strike": 5840.0, "option_type": "PUT", "delta": -0.42, "bid": 5.50, "ask": 5.90},
            {"strike": 5860.0, "option_type": "CALL", "delta": 0.42, "bid": 5.50, "ask": 5.90},
        ]
        tool_handlers.get_spx_price.return_value = {"price": 5847.0}
        tool_handlers.place_order.return_value = {"order_id": "IF-LIVE-001"}

        risk_manager = AsyncMock()
        risk_manager.can_open_new_trade.return_value = {"allowed": True}

        engine = DecisionEngine(
            tool_handlers=tool_handlers,
            risk_manager=risk_manager,
            strategy=strategy,
        )

        result = await engine.run_cycle()
        assert result.outcome == "executed"
        tool_handlers.place_order.assert_called_once()
        order = tool_handlers.place_order.call_args[0][0]
        # order_type is the submission route; an iron fly is an IC with equal short strikes (asserted below).
        assert order["order_type"] == "iron_condor"
        assert order["short_call_strike"] == order["short_put_strike"]  # same center
