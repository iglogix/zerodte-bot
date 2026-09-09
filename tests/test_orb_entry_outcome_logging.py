"""Non-executed ORB entry outcomes must be observable, and optionally re-armable.

Motivated by 2026-08-21: both live ORB bots died silently ~1s after a retest
trigger. Every early-return in OrbEngine._execute_entry (no_candidates,
validation_failed, chain_fetch_failed, place_order_failed, ...) returned a
CycleResult whose reason was never logged and never written to a decision
file — and run_scheduled() discarded the CycleResult entirely. The process
exited 0 and systemd's Restart=on-failure saw success.

Pins:
  (a) every non-executed _execute_entry outcome is logged AND written as a
      decision file; run_scheduled logs the final CycleResult.
  (b) strategy.rearm_after_failed_entry=True keeps the entry watch alive after
      a retryable failure (pre-order failures only), gated by a cooldown.
      Default (False) preserves the return-immediately behavior.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_v2.agent.base_engine import EngineDeps
from src_v2.agent.decision_engine import CycleResult
from src_v2.agent.orb_engine import OrbEngine
from src_v2.services.decision_logger import DecisionFileLogger

ET = ZoneInfo("America/New_York")


# ----------------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------------

class FakeClock:
    """Wall-clock stand-in whose sleep() advances simulated time."""

    def __init__(self, start: datetime):
        self._t = start

    def now(self, tz=None):
        return self._t.astimezone(tz) if tz else self._t

    async def sleep(self, seconds: float):
        self._t += timedelta(seconds=seconds)


class FakeToolHandlers:
    def __init__(self):
        self.chain_exc: Exception | None = None

    def reset_orb_state(self):
        pass

    async def get_futures_price(self, symbol="ES"):
        return {"price": 7690.0}

    async def get_breakout_status(self, price):
        return {"status": "breakout", "bias": "short"}

    def check_midline_touch(self, price, level, tol):
        return True

    async def get_option_chain(self, expiration="0DTE"):
        if self.chain_exc:
            raise self.chain_exc
        return []

    async def get_orb_spot(self, chain, ref_price, index_symbol=""):
        return {"spot": 7690.0}

    async def place_order(self, payload):
        return {"order_id": "fake-1", "status": "filled", "trade_id": "orbdir-test"}


def _setup():
    return SimpleNamespace(
        strikes={"short_call": 7705, "long_call": 7725},
        total_credit=3.5,
        extra={},
    )


class FakeStrategy:
    """search_candidates pops scripted results; records call times via clock."""

    def __init__(self, script, clock=None):
        self.script = list(script)
        self.clock = clock
        self.search_times: list[datetime] = []
        self.validate_result = {"passed": True}
        self.place_exc: Exception | None = None
        self._orb_bias = None

    async def search_candidates(self, chain, spot):
        now = self.clock.now(ET) if self.clock else datetime.now(ET)
        self.search_times.append(now)
        item = self.script.pop(0) if self.script else self.script_default()
        if isinstance(item, Exception):
            raise item
        return item

    @staticmethod
    def script_default():
        return {"candidates_found": 0}

    async def validate(self, top, chain):
        return self.validate_result

    def build_order(self, top):
        return {"order_type": "call_credit_spread"}


def _mk_engine(tmp_path, strategy, th=None, strat_cfg=None, clock=None, dfl=True):
    config = {
        "bot": {"symbol": "ES", "strategy": "orb_em_directional"},
        "strategy": {"eod_force_close": "15:50", **(strat_cfg or {})},
        "risk": {"max_concurrent_trades": 1},
    }
    deps = EngineDeps(
        tool_handlers=th or FakeToolHandlers(),
        risk_manager=None,
        anomaly_detector=None,
        regime_classifier=None,
        decision_file_logger=DecisionFileLogger(base_dir=str(tmp_path)) if dfl else None,
        db=None,
        mode="alpaca_paper",
        config=config,
        clock=clock or FakeClock(datetime(2026, 8, 21, 10, 0, tzinfo=ET)),
    )
    engine = OrbEngine(deps, strategy)
    engine._orb_levels = {"high": 7696.75, "low": 7693.0, "midline": 7694.88, "range_pts": 3.8}
    return engine


def _decision_docs(tmp_path):
    docs = []
    for f in sorted(tmp_path.rglob("*.json")):
        docs.append(json.loads(f.read_text()))
    return docs


# ----------------------------------------------------------------------------
# (a) Observability: every non-executed outcome is logged + written to disk
# ----------------------------------------------------------------------------

def test_no_candidates_logs_reason_and_writes_decision(tmp_path, caplog):
    strategy = FakeStrategy([{"candidates_found": 0}])
    engine = _mk_engine(tmp_path, strategy)

    with caplog.at_level(logging.INFO, logger="src_v2.agent.orb_engine"):
        result = asyncio.run(engine._execute_entry(7690.0, "short"))

    assert result.outcome == "waited"
    assert result.reason == "no_candidates"
    assert "no_candidates" in caplog.text

    docs = _decision_docs(tmp_path)
    assert len(docs) == 1
    assert docs[0]["outcome"] == "waited"
    assert "no_candidates" in json.dumps(docs[0])


def test_validation_failed_logs_and_writes_decision_with_failures(tmp_path, caplog):
    strategy = FakeStrategy([{"candidates_found": 1, "top_candidate": _setup()}])
    strategy.validate_result = {"passed": False, "failures": ["delta_too_high"]}
    engine = _mk_engine(tmp_path, strategy)

    with caplog.at_level(logging.INFO, logger="src_v2.agent.orb_engine"):
        result = asyncio.run(engine._execute_entry(7690.0, "short"))

    assert result.outcome == "waited"
    assert "validation_failed" in result.reason
    assert "delta_too_high" in caplog.text

    docs = _decision_docs(tmp_path)
    assert len(docs) == 1
    assert docs[0]["outcome"] == "waited"
    assert "delta_too_high" in json.dumps(docs[0])


def test_chain_fetch_failed_logs_error_and_writes_decision(tmp_path, caplog):
    th = FakeToolHandlers()
    th.chain_exc = RuntimeError("dxlink dead")
    engine = _mk_engine(tmp_path, FakeStrategy([]), th=th)

    with caplog.at_level(logging.INFO, logger="src_v2.agent.orb_engine"):
        result = asyncio.run(engine._execute_entry(7690.0, "short"))

    assert result.outcome == "error"
    assert "chain_fetch_failed" in result.reason
    assert "chain_fetch_failed" in caplog.text

    docs = _decision_docs(tmp_path)
    assert len(docs) == 1
    assert docs[0]["outcome"] == "error"
    assert "dxlink dead" in json.dumps(docs[0])


def test_place_order_failed_logs_error_and_writes_decision(tmp_path, caplog):
    strategy = FakeStrategy([{"candidates_found": 1, "top_candidate": _setup()}])
    th = FakeToolHandlers()

    async def boom(payload):
        raise RuntimeError("broker rejected")

    th.place_order = boom
    engine = _mk_engine(tmp_path, strategy, th=th)

    with caplog.at_level(logging.INFO, logger="src_v2.agent.orb_engine"):
        result = asyncio.run(engine._execute_entry(7690.0, "short"))

    assert result.outcome == "error"
    assert "place_order_failed" in result.reason
    assert "place_order_failed" in caplog.text

    docs = _decision_docs(tmp_path)
    assert len(docs) == 1
    assert docs[0]["outcome"] == "error"
    assert "broker rejected" in json.dumps(docs[0])


def test_run_scheduled_logs_final_cycle_result(caplog):
    from src_v2.main import BotOrchestrator

    class FakeSvc:
        async def start(self):
            await asyncio.sleep(3600)

        async def stop(self):
            pass

    class FakeEngine:
        async def run(self):
            return CycleResult(outcome="waited", reason="no_candidates").finish()

    orch = BotOrchestrator.__new__(BotOrchestrator)
    orch._monitor = FakeSvc()
    orch._fill_listener = FakeSvc()
    orch._quote_comparator = FakeSvc()
    orch._engine = FakeEngine()

    with caplog.at_level(logging.INFO, logger="src_v2.main"):
        asyncio.run(orch.run_scheduled())

    assert "outcome=waited" in caplog.text
    assert "no_candidates" in caplog.text


# ----------------------------------------------------------------------------
# (b) Re-arm after failed entry (config-gated, default off)
# ----------------------------------------------------------------------------

def test_rearm_disabled_by_default_returns_after_failed_entry(tmp_path):
    strategy = FakeStrategy([{"candidates_found": 0}])
    engine = _mk_engine(tmp_path, strategy)

    result = asyncio.run(engine._entry_watch())

    assert result.outcome == "waited"
    assert result.reason == "no_candidates"
    assert len(strategy.search_times) == 1  # no second attempt


def test_rearm_enabled_retries_after_cooldown_and_enters(tmp_path):
    clock = FakeClock(datetime(2026, 8, 21, 10, 0, tzinfo=ET))
    strategy = FakeStrategy(
        [{"candidates_found": 0}, {"candidates_found": 1, "top_candidate": _setup()}],
        clock=clock,
    )
    engine = _mk_engine(
        tmp_path, strategy,
        strat_cfg={"rearm_after_failed_entry": True, "rearm_cooldown_minutes": 2},
        clock=clock,
    )

    result = asyncio.run(engine._entry_watch())

    assert result.outcome == "executed"
    assert len(strategy.search_times) == 2
    gap = strategy.search_times[1] - strategy.search_times[0]
    assert gap >= timedelta(minutes=2)


def test_rearm_does_not_retry_place_order_failures(tmp_path):
    """A failed submission may have reached the broker — never auto-retry it."""
    clock = FakeClock(datetime(2026, 8, 21, 10, 0, tzinfo=ET))
    strategy = FakeStrategy(
        [{"candidates_found": 1, "top_candidate": _setup()}], clock=clock,
    )
    th = FakeToolHandlers()

    async def boom(payload):
        raise RuntimeError("timeout after submit")

    th.place_order = boom
    engine = _mk_engine(
        tmp_path, strategy, th=th,
        strat_cfg={"rearm_after_failed_entry": True, "rearm_cooldown_minutes": 2},
        clock=clock,
    )

    result = asyncio.run(engine._entry_watch())

    assert result.outcome == "error"
    assert "place_order_failed" in result.reason
    assert len(strategy.search_times) == 1


def test_rearm_gives_up_at_eod(tmp_path):
    """Flag on + perpetual no_candidates must still terminate at EOD."""
    clock = FakeClock(datetime(2026, 8, 21, 15, 40, tzinfo=ET))
    strategy = FakeStrategy([], clock=clock)  # script empty → always 0 candidates
    engine = _mk_engine(
        tmp_path, strategy,
        strat_cfg={"rearm_after_failed_entry": True, "rearm_cooldown_minutes": 2},
        clock=clock,
    )

    result = asyncio.run(engine._entry_watch())

    assert result.outcome == "waited"
    assert result.reason == "eod_no_entry"
