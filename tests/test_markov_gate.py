"""Tests for the Markov regime entry gate (zerodte/agent/markov_gate.py).

Gate rule (WF-CV validated, results/markov_gate_wf_cv_2026-06-11.txt):
block the session iff the SPX trailing-window state is SIDEWAYS now AND was
SIDEWAYS one window earlier ("stale sideways"). All inputs are prior-session
daily closes — fully causal.
"""
from __future__ import annotations

import pytest

from zerodte.agent.base_engine import EngineDeps
from zerodte.agent.markov_gate import MarkovGate
from zerodte.agent.scheduled_engine import ScheduledEngine

W = 5  # compact test window; live default is 20


def gate(rule="excl_stale_sw", enabled=True, window=W):
    return MarkovGate({"enabled": enabled, "window": window, "rule": rule,
                       "bull_threshold": 0.05, "bear_threshold": -0.05})


# closes are oldest -> newest, last element = prior session close
FLAT = [100.0] * (2 * W + 1)                                   # SW -> SW (stale)
FRESH_FROM_BEAR = [100, 97, 96, 95, 94.5] + [94.0] * (W + 1)   # 100->94 = -6% BEAR, then flat SW
BULL_NOW = [100.0] * (W + 1) + [101, 102, 103, 105, 106]       # SW -> BULL


class TestMarkovGateLogic:
    def test_stale_sideways_blocks(self):
        d = gate().evaluate(FLAT)
        assert d["allow"] is False
        assert d["state_now"] == "SIDEWAYS" and d["state_prev"] == "SIDEWAYS"

    def test_fresh_sideways_from_bear_allows(self):
        d = gate().evaluate(FRESH_FROM_BEAR)
        assert d["allow"] is True
        assert d["state_now"] == "SIDEWAYS" and d["state_prev"] == "BEAR"

    def test_bull_now_allows(self):
        d = gate().evaluate(BULL_NOW)
        assert d["allow"] is True
        assert d["state_now"] == "BULL"

    def test_non_sw_rule_blocks_any_sideways(self):
        assert gate(rule="non_sw").evaluate(FRESH_FROM_BEAR)["allow"] is False
        assert gate(rule="non_sw").evaluate(BULL_NOW)["allow"] is True

    def test_insufficient_data_fails_open(self):
        d = gate().evaluate([100.0] * (2 * W))  # one close short
        assert d["allow"] is True
        assert "insufficient" in d["reason"]

    def test_decision_is_auditable(self):
        d = gate().evaluate(FLAT)
        # FIX 2 spirit: the decision carries the numbers it was made from
        assert d["ret_now"] == pytest.approx(0.0)
        assert d["ret_prev"] == pytest.approx(0.0)
        assert "stale_sideways" in d["reason"]


# ---------------------------------------------------------------- engine wiring

class StubHandlers:
    def __init__(self, closes=None, raise_on_closes=False):
        self._closes = closes or []
        self._raise = raise_on_closes
        self.daily_calls = 0
        self.candle_calls = 0

    async def get_daily_closes(self, symbol="SPX", count=50):
        self.daily_calls += 1
        if self._raise:
            raise RuntimeError("no data source")
        return self._closes[-count:]

    async def get_5min_candles(self, count=10):
        self.candle_calls += 1
        return []  # forces outcome=waited right after the gate


def make_engine(handlers, gate_cfg):
    deps = EngineDeps(
        tool_handlers=handlers, risk_manager=None, anomaly_detector=None,
        regime_classifier=None, decision_file_logger=None, db=None,
        mode="manual",
        config={"entry": {"markov_gate": gate_cfg}},
    )
    return ScheduledEngine(deps, strategy=None)


GATE_CFG = {"enabled": True, "window": W, "rule": "excl_stale_sw",
            "bull_threshold": 0.05, "bear_threshold": -0.05}


async def test_engine_blocks_session_on_stale_sideways():
    h = StubHandlers(closes=FLAT)
    res = await make_engine(h, GATE_CFG).run_cycle()
    assert res.outcome == "skip_session"
    assert res.reason.startswith("markov_gate")
    assert h.candle_calls == 0  # gate fires before any other work


async def test_engine_proceeds_when_gate_allows():
    h = StubHandlers(closes=BULL_NOW)
    res = await make_engine(h, GATE_CFG).run_cycle()
    assert res.outcome == "waited"          # got past the gate to candle fetch
    assert res.reason == "insufficient_candle_data"


async def test_engine_fails_open_on_data_error():
    h = StubHandlers(raise_on_closes=True)
    res = await make_engine(h, GATE_CFG).run_cycle()
    assert res.outcome == "waited"
    assert res.reason == "insufficient_candle_data"


async def test_engine_caches_gate_decision_per_day():
    h = StubHandlers(closes=BULL_NOW)
    eng = make_engine(h, GATE_CFG)
    await eng.run_cycle()
    await eng.run_cycle()
    assert h.daily_calls == 1  # second tranche reuses today's decision


async def test_get_daily_closes_keeps_newest_when_feed_overdelivers():
    """get_candles sorts ascending and truncates at `count` (oldest first);
    get_daily_closes must end up with the MOST RECENT closes regardless."""
    from zerodte.agent.tool_handlers import ToolHandlers

    class FeedStub:
        async def get_candles(self, symbol, interval="1d", from_time=None, count=1):
            # 80 daily candles available in the window; honour the truncation
            all_c = [{"close": 7000.0 + i, "timestamp": f"2026-03-{1:02d}"}
                     for i in range(80)]
            return all_c[:count]

    th = ToolHandlers(api_client=FeedStub(), mode="manual")
    closes = await th.get_daily_closes(count=10)
    assert len(closes) == 10
    assert closes[-1] == 7079.0  # the newest close in the feed, not the oldest slice


async def test_engine_skips_gate_when_disabled():
    h = StubHandlers(closes=FLAT)
    res = await make_engine(h, {**GATE_CFG, "enabled": False}).run_cycle()
    assert res.outcome == "waited"
    assert h.daily_calls == 0
