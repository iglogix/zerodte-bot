# MEIC Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SPX 0DTE Multiple-Entries Iron Condor (MEIC) strategy to the `src_v2` strategy-agnostic bot — pluggable via config, backtestable via the existing engine, built test-first.

**Architecture:** A new `MeicStrategy(LiveStrategy)` reuses `ScheduledEngine` for tranche-cadence entries and `PositionMonitor` for exits. The IC is routed as two independent credit spreads with stop-market orders on the short legs at a % of total credit. Scale-out happens via partial closes carried on a widened `ExitAction`. Backtesting reuses `BacktestEngine` + `CbboStore` with a new MEIC day-processing path.

**Tech Stack:** Python 3, asyncio, pytest, SQLAlchemy, existing `src_v2` framework (`LiveStrategy` ABC, `ScheduledEngine`, `PositionMonitor`, `BacktestEngine`, `CbboStore`).

**Spec:** `docs/superpowers/specs/2026-05-20-meic-strategy-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src_v2/strategies/meic.py` | **New.** `MeicStrategy` — all 10 `LiveStrategy` methods + internal scale-out ledger. |
| `src_v2/strategies/base.py` | **Modify.** Add `leg_quantities` + `quantity_fraction` to `ExitAction`. |
| `src_v2/services/position_monitor.py` | **Modify.** Honour `exit_action.leg_quantities` when closing legs; `_close_leg` takes `quantity`. |
| `src_v2/agent/tool_handlers.py` | **Modify.** Add `place_two_spread_ic` order helper. |
| `src_v2/main.py` | **Modify.** Route `bot.strategy == "meic"`. |
| `src_v2/backtest/engine.py` | **Modify.** MEIC backtest path (`_process_day_meic`, `_simulate_meic_side`). |
| `config/config_meic_spx.yaml` | **New.** Strategy params + tranche schedule. |
| `tests_v2/test_meic_strategy.py` | **New.** Unit tests for `MeicStrategy`. |
| `tests_v2/test_meic_position_monitor.py` | **New.** Partial-close behaviour. |
| `tests_v2/test_meic_wiring.py` | **New.** Config routing + engine wiring. |
| `tests_v2/backtest/test_meic_backtest.py` | **New.** Backtest smoke tests. |
| `tests_v2/conftest.py` | **Modify.** Add `meic_config` + `sample_meic_chain` fixtures. |

**Conventions to follow (from existing code):**
- Strategy classes mirror `src_v2/strategies/iron_condor.py`.
- Tests mirror `tests_v2/test_iron_condor_strategy.py` (class-grouped, `@pytest.mark.asyncio` on async).
- `mid_price` from `src_v2/strategies/utils.py`.
- Option dicts use keys: `strike`, `option_type` (`"CALL"`/`"PUT"`), `delta`, `bid`, `ask`.
- Run tests with the project venv: `python -m pytest`.

---

## Task 1: Widen `ExitAction` for partial closes

**Files:**
- Modify: `src_v2/strategies/base.py:49-61`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write the failing test**

Create `tests_v2/test_meic_strategy.py` with:

```python
"""TDD tests: MeicStrategy."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src_v2.strategies.base import ExitAction

ET = ZoneInfo("America/New_York")


class TestExitActionWidening:
    def test_exit_action_has_partial_fields_with_defaults(self):
        a = ExitAction(action="hold")
        assert a.leg_quantities == {}
        assert a.quantity_fraction == 1.0

    def test_exit_action_accepts_leg_quantities(self):
        a = ExitAction(
            action="take_profit",
            legs_to_close=["SPXW_C"],
            leg_quantities={"SPXW_C": 2},
            quantity_fraction=0.25,
        )
        assert a.leg_quantities == {"SPXW_C": 2}
        assert a.quantity_fraction == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestExitActionWidening -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'leg_quantities'`.

- [ ] **Step 3: Implement the field additions**

In `src_v2/strategies/base.py`, replace the `ExitAction` dataclass body:

```python
@dataclass
class ExitAction:
    """What the strategy wants done on a position check.

    action: "take_profit" | "stop_loss" | "hold" | "trailing_update"
    legs_to_close: list of leg symbols to close (empty for "hold")
    reason: human-readable explanation
    new_stop_price: for trailing stop updates
    leg_quantities: per-symbol contract count to close (defaults to full leg)
    quantity_fraction: whole-trade close fraction (default 1.0 = all)
    """
    action: str
    legs_to_close: List[str] = field(default_factory=list)
    reason: str = ""
    new_stop_price: Optional[float] = None
    leg_quantities: Dict[str, int] = field(default_factory=dict)
    quantity_fraction: float = 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestExitActionWidening -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify no regression in other strategies**

Run: `python -m pytest tests_v2/ -q`
Expected: all existing tests still pass (defaults preserve behaviour).

- [ ] **Step 6: Commit**

```bash
git add src_v2/strategies/base.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): widen ExitAction with leg_quantities and quantity_fraction"
```

---

## Task 2: Test fixtures for MEIC

**Files:**
- Modify: `tests_v2/conftest.py`

- [ ] **Step 1: Add fixtures**

Append to `tests_v2/conftest.py`:

```python
@pytest.fixture
def meic_config() -> Dict[str, Any]:
    """Config dict for MeicStrategy."""
    return {
        "bot": {"strategy": "meic"},
        "entry": {
            "schedule": ["11:00", "11:20", "11:40", "12:00", "12:40",
                         "13:00", "13:20", "14:00", "14:40", "15:00"],
        },
        "strategy": {
            "am_pm_cutoff": "12:30",
            "am_min_credit": 1.25,
            "am_max_credit": 1.65,
            "pm_min_credit": 0.80,
            "pm_max_credit": 1.30,
            "wing_widths": [80, 90, 100],
            "stop_pct_of_total_credit": 0.70,
            "am_scale_out_tiers": [0.55, 0.65, 0.75, 0.85],
            "am_scale_out_fractions": [0.25, 0.25, 0.25, 0.25],
            "pm_scale_out_tiers": [0.65, 0.85],
            "pm_scale_out_fractions": [0.50, 0.50],
            "abandon_long_below": 0.05,
            "credit_equality_threshold": 0.30,
            "default_tranche_qty": 3,
            "tranche_sizing": {"11:00": 3, "11:40": 4, "12:00": 5, "13:00": 6},
        },
    }


@pytest.fixture
def sample_meic_chain() -> List[Dict[str, Any]]:
    """Chain supporting 80-100pt MEIC wings around spot 5850.

    Short call 5950 mid ~1.45; long calls at 6030/6040/6050.
    Short put 5750 mid ~1.45; long puts at 5670/5660/5650.
    Credits land in the AM band [1.25, 1.65].
    """
    return [
        # Short call + its three possible long-call wings (80/90/100)
        _make_opt(5950.0, "call", 0.10, 1.40, 1.50),
        _make_opt(6030.0, "call", 0.03, 0.02, 0.10),   # wing 80 -> credit ~1.39
        _make_opt(6040.0, "call", 0.025, 0.01, 0.09),  # wing 90
        _make_opt(6050.0, "call", 0.02, 0.01, 0.07),   # wing 100
        # Short put + its three possible long-put wings
        _make_opt(5750.0, "put", 0.10, 1.40, 1.50),
        _make_opt(5670.0, "put", 0.03, 0.02, 0.10),    # wing 80 -> credit ~1.39
        _make_opt(5660.0, "put", 0.025, 0.01, 0.09),   # wing 90
        _make_opt(5650.0, "put", 0.02, 0.01, 0.07),    # wing 100
    ]
```

- [ ] **Step 2: Verify fixtures import cleanly**

Run: `python -m pytest tests_v2/test_meic_strategy.py -q`
Expected: still only the Task-1 tests collected/passing (fixtures unused yet, no errors).

- [ ] **Step 3: Commit**

```bash
git add tests_v2/conftest.py
git commit -m "test(meic): add meic_config and sample_meic_chain fixtures"
```

---

## Task 3: `MeicStrategy` scaffolding — configure, simple methods

**Files:**
- Create: `src_v2/strategies/meic.py`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests_v2/test_meic_strategy.py`:

```python
from src_v2.strategies.base import LiveStrategy
from src_v2.strategies.meic import MeicStrategy
from src_v2.agent.scheduled_engine import ScheduledEngine


class TestMeicConfigure:
    def test_is_live_strategy(self):
        assert issubclass(MeicStrategy, LiveStrategy)

    def test_configure_reads_all_keys(self, meic_config):
        s = MeicStrategy()
        s.configure(meic_config)
        assert s._am_min_credit == 1.25
        assert s._am_max_credit == 1.65
        assert s._pm_min_credit == 0.80
        assert s._pm_max_credit == 1.30
        assert s._wing_widths == [80, 90, 100]
        assert s._stop_pct == 0.70
        assert s._am_tiers == [0.55, 0.65, 0.75, 0.85]
        assert s._am_fractions == [0.25, 0.25, 0.25, 0.25]
        assert s._pm_tiers == [0.65, 0.85]
        assert s._pm_fractions == [0.50, 0.50]
        assert s._abandon_long_below == 0.05
        assert s._default_tranche_qty == 3
        assert s._tranche_sizing == {"11:00": 3, "11:40": 4, "12:00": 5, "13:00": 6}
        assert s._am_pm_cutoff.hour == 12 and s._am_pm_cutoff.minute == 30

    def test_configure_rejects_stop_pct_too_low(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.50
        s = MeicStrategy()
        with pytest.raises(ValueError):
            s.configure(meic_config)

    def test_configure_rejects_stop_pct_too_high(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.95
        s = MeicStrategy()
        with pytest.raises(ValueError):
            s.configure(meic_config)


class TestMeicSimpleMethods:
    def test_leg_pairs(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s.leg_pairs() == [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    def test_compute_trailing_stop_returns_none(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s.compute_trailing_stop({}, {}) is None

    def test_create_engine_returns_scheduled_engine(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        engine = s.create_engine(_FakeDeps())
        assert isinstance(engine, ScheduledEngine)


class _FakeDeps:
    """Minimal EngineDeps stand-in for create_engine."""
    tool_handlers = None
    risk_manager = None
    anomaly_detector = None
    regime_classifier = None
    decision_file_logger = None
    db = None
    mode = "dry_run"
    config = {"entry": {}}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicConfigure -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src_v2.strategies.meic'`.

- [ ] **Step 3: Create the module skeleton**

Create `src_v2/strategies/meic.py`:

```python
"""
MeicStrategy — SPX 0DTE Multiple-Entries Iron Condor.

Implements LiveStrategy for tranche-based IC entries with:
  - AM/PM premium-band candidate search across 80-100pt wings
  - per-tranche contract sizing from a config map
  - two-spread routing (independent put/call credit spreads)
  - stop-market on short legs at % of total credit
  - AM/PM scale-out tiers via partial-close ExitActions
  - leftover-long abandonment below a price threshold

Runs on ScheduledEngine (denser schedule than the base IC).
"""

from __future__ import annotations

import math
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src_v2.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.strategies.utils import mid_price as _mid

ET = ZoneInfo("America/New_York")


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


class MeicStrategy(LiveStrategy):

    def __init__(self):
        # (trade_id, side) -> set of fired tier indices
        self._scaleout_state: Dict[Tuple[str, str], set] = {}
        # Injectable clock for testability; tests override this.
        self._now_fn = lambda: datetime.now(ET)

    # ------------------------------------------------------------------
    # Configure
    # ------------------------------------------------------------------

    def configure(self, config: Dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self._am_pm_cutoff = _parse_hhmm(strat.get("am_pm_cutoff", "12:30"))
        self._am_min_credit = strat.get("am_min_credit", 1.25)
        self._am_max_credit = strat.get("am_max_credit", 1.65)
        self._pm_min_credit = strat.get("pm_min_credit", 0.80)
        self._pm_max_credit = strat.get("pm_max_credit", 1.30)
        self._wing_widths = list(strat.get("wing_widths", [80, 90, 100]))
        self._stop_pct = strat.get("stop_pct_of_total_credit", 0.70)
        if not (0.65 <= self._stop_pct <= 0.90):
            raise ValueError(
                f"stop_pct_of_total_credit must be in [0.65, 0.90], got {self._stop_pct}"
            )
        self._am_tiers = list(strat.get("am_scale_out_tiers", [0.55, 0.65, 0.75, 0.85]))
        self._am_fractions = list(strat.get("am_scale_out_fractions", [0.25, 0.25, 0.25, 0.25]))
        self._pm_tiers = list(strat.get("pm_scale_out_tiers", [0.65, 0.85]))
        self._pm_fractions = list(strat.get("pm_scale_out_fractions", [0.50, 0.50]))
        self._abandon_long_below = strat.get("abandon_long_below", 0.05)
        self._credit_equality_threshold = strat.get("credit_equality_threshold", 0.30)
        self._default_tranche_qty = strat.get("default_tranche_qty", 3)
        self._tranche_sizing = dict(strat.get("tranche_sizing", {}))

    # ------------------------------------------------------------------
    # Simple methods
    # ------------------------------------------------------------------

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    def compute_trailing_stop(
        self, trade: Dict[str, Any], quotes: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        return None

    def create_engine(self, deps: "EngineDeps") -> "BaseEngine":
        from src_v2.agent.scheduled_engine import ScheduledEngine
        return ScheduledEngine(deps, self)

    # ------------------------------------------------------------------
    # Stubs filled by later tasks
    # ------------------------------------------------------------------

    async def search_candidates(self, chain, spot):
        raise NotImplementedError

    async def validate(self, setup, chain):
        raise NotImplementedError

    def build_order(self, setup):
        raise NotImplementedError

    def compute_stop_loss(self, setup):
        raise NotImplementedError

    def build_stop_orders(self, setup, stop_info):
        raise NotImplementedError

    def check_exit(self, trade, quotes):
        raise NotImplementedError
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicConfigure tests_v2/test_meic_strategy.py::TestMeicSimpleMethods -v`
Expected: PASS (configure + simple-method tests).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): MeicStrategy scaffolding — configure, leg_pairs, engine"
```

---

## Task 4: `search_candidates` — AM/PM bands, wings, tranche sizing

**Files:**
- Modify: `src_v2/strategies/meic.py`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests_v2/test_meic_strategy.py`:

```python
from src_v2.strategies.base import LiveTradeSetup


def _at(s: MeicStrategy, hhmm: str):
    """Pin the strategy clock to a fixed ET wall time on an arbitrary date."""
    h, m = (int(x) for x in hhmm.split(":"))
    s._now_fn = lambda: datetime(2026, 5, 20, h, m, tzinfo=ET)


class TestMeicSearchCandidates:
    @pytest.mark.asyncio
    async def test_empty_chain_returns_zero(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:30")
        result = await s.search_candidates([], spot=5850.0)
        assert result["candidates_found"] == 0
        assert result["top_candidate"] is None

    @pytest.mark.asyncio
    async def test_am_band_finds_candidate(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:30")
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        assert result["candidates_found"] > 0
        top = result["top_candidate"]
        assert isinstance(top, LiveTradeSetup)
        assert top.strategy_type == "meic"
        # Credits must be in AM band
        assert 1.25 <= top.extra["call_credit"] <= 1.65
        assert 1.25 <= top.extra["put_credit"] <= 1.65

    @pytest.mark.asyncio
    async def test_pm_band_rejects_am_credits(self, meic_config, sample_meic_chain):
        # At 13:00 the PM band is [0.80, 1.30]; the sample chain's ~1.39
        # credits fall OUTSIDE PM band -> no candidates.
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "13:00")
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        assert result["candidates_found"] == 0

    @pytest.mark.asyncio
    async def test_stamps_tranche_qty(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:40")  # tranche_sizing["11:40"] == 4
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        top = result["top_candidate"]
        assert top.extra["qty"] == 4
        assert all(leg.quantity == 4 for leg in top.legs)

    @pytest.mark.asyncio
    async def test_default_qty_when_slot_missing(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config)
        _at(s, "11:20")  # not in tranche_sizing -> default 3
        result = await s.search_candidates(sample_meic_chain, spot=5850.0)
        top = result["top_candidate"]
        assert top.extra["qty"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicSearchCandidates -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `search_candidates`**

Replace the `search_candidates` stub in `src_v2/strategies/meic.py`:

```python
    def _active_band(self) -> Tuple[float, float]:
        now_t = self._now_fn().time()
        if now_t < self._am_pm_cutoff:
            return self._am_min_credit, self._am_max_credit
        return self._pm_min_credit, self._pm_max_credit

    def _tranche_qty(self) -> int:
        slot = self._now_fn().strftime("%H:%M")
        return int(self._tranche_sizing.get(slot, self._default_tranche_qty))

    async def search_candidates(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Dict[str, Any]:
        if not chain:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        min_cr, max_cr = self._active_band()
        qty = self._tranche_qty()

        chain_by_strike: Dict[Tuple[float, str], Dict] = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }
        calls = sorted(
            [o for o in chain if o.get("option_type", "").upper() in ("C", "CALL")],
            key=lambda o: o.get("strike", 0),
        )
        puts = sorted(
            [o for o in chain if o.get("option_type", "").upper() in ("P", "PUT")],
            key=lambda o: o.get("strike", 0), reverse=True,
        )

        candidates: List[LiveTradeSetup] = []
        for sc in calls:
            sc_mid = _mid(sc)
            if sc_mid is None:
                continue
            sc_strike = float(sc.get("strike", 0))
            for sp in puts:
                sp_mid = _mid(sp)
                if sp_mid is None:
                    continue
                sp_strike = float(sp.get("strike", 0))
                for w in self._wing_widths:
                    lc = chain_by_strike.get((sc_strike + w, "call"))
                    lp = chain_by_strike.get((sp_strike - w, "put"))
                    if lc is None or lp is None:
                        continue
                    lc_mid = _mid(lc)
                    lp_mid = _mid(lp)
                    if lc_mid is None or lp_mid is None:
                        continue
                    call_credit = round(sc_mid - lc_mid, 4)
                    put_credit = round(sp_mid - lp_mid, 4)
                    if not (min_cr <= call_credit <= max_cr):
                        continue
                    if not (min_cr <= put_credit <= max_cr):
                        continue
                    credit_diff = abs(call_credit - put_credit)
                    if credit_diff > self._credit_equality_threshold:
                        continue
                    total_credit = round(call_credit + put_credit, 4)
                    candidates.append(LiveTradeSetup(
                        strategy_type="meic",
                        legs=[
                            OrderLeg("", sc_strike, "call", "sell", qty),
                            OrderLeg("", sc_strike + w, "call", "buy", qty),
                            OrderLeg("", sp_strike, "put", "sell", qty),
                            OrderLeg("", sp_strike - w, "put", "buy", qty),
                        ],
                        total_credit=total_credit,
                        strikes={
                            "short_call": sc_strike,
                            "long_call": sc_strike + w,
                            "short_put": sp_strike,
                            "long_put": sp_strike - w,
                        },
                        extra={
                            "wing_width": w,
                            "call_credit": call_credit,
                            "put_credit": put_credit,
                            "credit_diff": round(credit_diff, 4),
                            "total_premium": total_credit,
                            "short_call_mid": sc_mid,
                            "short_put_mid": sp_mid,
                            "qty": qty,
                        },
                    ))

        candidates.sort(key=lambda c: c.extra["credit_diff"])
        candidates = candidates[:5]
        return {
            "candidates_found": len(candidates),
            "top_candidate": candidates[0] if candidates else None,
            "candidates": candidates,
        }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicSearchCandidates -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): search_candidates with AM/PM bands and tranche sizing"
```

---

## Task 5: `validate`

**Files:**
- Modify: `src_v2/strategies/meic.py`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests_v2/test_meic_strategy.py`:

```python
class TestMeicValidate:
    def _setup(self, qty=3, cc=1.40, pc=1.40,
               sc=5950.0, lc=6040.0, sp=5750.0, lp=5660.0):
        from src_v2.strategies.base import LiveTradeSetup, OrderLeg
        return LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", sc, "call", "sell", qty)],
            total_credit=cc + pc,
            strikes={"short_call": sc, "long_call": lc,
                     "short_put": sp, "long_put": lp},
            extra={"call_credit": cc, "put_credit": pc, "qty": qty},
        )

    @pytest.mark.asyncio
    async def test_valid_setup_passes(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup()
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is True
        assert res["failures"] == []

    @pytest.mark.asyncio
    async def test_bad_call_strike_order_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup(lc=5940.0)  # long_call < short_call
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "strike_order_call" in res["failures"]

    @pytest.mark.asyncio
    async def test_asymmetric_wings_fail(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        # call wing 90 (5950->6040), put wing 60 (5750->5690): diff 30 > 5
        setup = self._setup(lp=5690.0)
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "wing_symmetry" in res["failures"]

    @pytest.mark.asyncio
    async def test_wing_out_of_range_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        # both wings 50pt -> below 80 minimum
        setup = self._setup(lc=6000.0, lp=5700.0)
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "wing_range" in res["failures"]

    @pytest.mark.asyncio
    async def test_credit_out_of_band_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup(cc=2.50)  # above AM band max 1.65
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "call_credit_band" in res["failures"]

    @pytest.mark.asyncio
    async def test_zero_qty_fails(self, meic_config, sample_meic_chain):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:30")
        setup = self._setup(qty=0)
        res = await s.validate(setup, sample_meic_chain)
        assert res["passed"] is False
        assert "qty" in res["failures"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicValidate -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `validate`**

Replace the `validate` stub in `src_v2/strategies/meic.py`:

```python
    async def validate(
        self, setup: LiveTradeSetup, chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        failures: List[str] = []

        def record(name: str, ok: bool):
            checks.append({"check": name, "passed": ok})
            if not ok:
                failures.append(name)

        st = setup.strikes
        sc, lc = st["short_call"], st["long_call"]
        sp, lp = st["short_put"], st["long_put"]

        record("strike_order_call", lc > sc)
        record("strike_order_put", sp > lp)

        call_w = lc - sc
        put_w = sp - lp
        record("wing_symmetry", abs(call_w - put_w) <= 5)
        record("wing_range", 80 <= call_w <= 100 and 80 <= put_w <= 100)

        min_cr, max_cr = self._active_band()
        cc = setup.extra.get("call_credit")
        pc = setup.extra.get("put_credit")
        record("call_credit_band", cc is not None and min_cr <= cc <= max_cr)
        record("put_credit_band", pc is not None and min_cr <= pc <= max_cr)

        record("qty", setup.extra.get("qty", 0) >= 1)

        return {"passed": len(failures) == 0, "checks": checks, "failures": failures}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicValidate -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): validate — strike order, wing symmetry/range, credit band, qty"
```

---

## Task 6: `build_order` — two-spread payload

**Files:**
- Modify: `src_v2/strategies/meic.py`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write failing test**

Append to `tests_v2/test_meic_strategy.py`:

```python
class TestMeicBuildOrder:
    def test_two_spread_payload(self, meic_config):
        from src_v2.strategies.base import LiveTradeSetup, OrderLeg
        s = MeicStrategy(); s.configure(meic_config)
        setup = LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", 5950, "call", "sell", 4)],
            total_credit=2.80,
            strikes={"short_call": 5950, "long_call": 6040,
                     "short_put": 5750, "long_put": 5660},
            extra={"call_credit": 1.40, "put_credit": 1.40, "qty": 4},
        )
        order = s.build_order(setup)
        assert order["order_type"] == "meic_two_spreads"
        assert order["quantity"] == 4
        assert order["put_spread"] == {
            "short_strike": 5750, "long_strike": 5660,
            "credit": 1.40, "quantity": 4,
        }
        assert order["call_spread"] == {
            "short_strike": 5950, "long_strike": 6040,
            "credit": 1.40, "quantity": 4,
        }
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicBuildOrder -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `build_order`**

Replace the `build_order` stub in `src_v2/strategies/meic.py`:

```python
    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        qty = int(setup.extra.get("qty", 1))
        st = setup.strikes
        return {
            "order_type": "meic_two_spreads",
            "quantity": qty,
            "put_spread": {
                "short_strike": st["short_put"],
                "long_strike": st["long_put"],
                "credit": setup.extra.get("put_credit"),
                "quantity": qty,
            },
            "call_spread": {
                "short_strike": st["short_call"],
                "long_strike": st["long_call"],
                "credit": setup.extra.get("call_credit"),
                "quantity": qty,
            },
        }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicBuildOrder -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): build_order emits two-spread payload"
```

---

## Task 7: `compute_stop_loss` + `build_stop_orders`

**Files:**
- Modify: `src_v2/strategies/meic.py`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests_v2/test_meic_strategy.py`:

```python
class TestMeicStops:
    def _setup(self, cc=1.40, pc=1.40, sc_mid=1.45, sp_mid=1.45):
        from src_v2.strategies.base import LiveTradeSetup, OrderLeg
        return LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", 5950, "call", "sell", 3)],
            total_credit=cc + pc,
            strikes={"short_call": 5950, "long_call": 6040,
                     "short_put": 5750, "long_put": 5660},
            extra={"call_credit": cc, "put_credit": pc, "qty": 3,
                   "short_call_mid": sc_mid, "short_put_mid": sp_mid},
        )

    def test_compute_stop_loss_math(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)  # stop_pct 0.70
        info = s.compute_stop_loss(self._setup())
        # total credit 2.80 -> stop amount 1.96
        assert info["total_credit"] == 2.80
        assert info["stop_amount"] == pytest.approx(1.96)
        assert info["short_call_stop_price"] == pytest.approx(1.45 + 1.96)
        assert info["short_put_stop_price"] == pytest.approx(1.45 + 1.96)
        assert info["order_type"] == "stop_market"

    def test_build_stop_orders_stop_market_on_shorts_only(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        setup = self._setup()
        info = s.compute_stop_loss(setup)
        orders = s.build_stop_orders(setup, info)
        assert len(orders) == 2
        sides = {o["side"] for o in orders}
        assert sides == {"short_call", "short_put"}
        for o in orders:
            assert "stop_market" in o
            assert "stop_limit" not in o
            assert o["stop_market"]["trigger_price"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicStops -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement both methods**

Replace the `compute_stop_loss` and `build_stop_orders` stubs in `src_v2/strategies/meic.py`:

```python
    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        cc = setup.extra.get("call_credit", 0.0)
        pc = setup.extra.get("put_credit", 0.0)
        total_credit = round(cc + pc, 4)
        stop_amount = round(self._stop_pct * total_credit, 4)
        sc_mid = setup.extra.get("short_call_mid", 0.0)
        sp_mid = setup.extra.get("short_put_mid", 0.0)
        return {
            "total_credit": total_credit,
            "stop_pct": self._stop_pct,
            "stop_amount": stop_amount,
            "short_call_stop_price": round(sc_mid + stop_amount, 4),
            "short_put_stop_price": round(sp_mid + stop_amount, 4),
            "order_type": "stop_market",
        }

    def build_stop_orders(
        self, setup: LiveTradeSetup, stop_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "side": "short_call",
                "stop_market": {"trigger_price": stop_info["short_call_stop_price"]},
            },
            {
                "side": "short_put",
                "stop_market": {"trigger_price": stop_info["short_put_stop_price"]},
            },
        ]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicStops -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): stop-market stops on shorts at pct of total credit"
```

---

## Task 8: `check_exit` — scale-out tiers (AM/PM, no re-fire)

**Files:**
- Modify: `src_v2/strategies/meic.py`
- Test: `tests_v2/test_meic_strategy.py`

Trade dict shape used by `check_exit` (provided by backtest + live):

```python
{
  "trade_id": "2026-05-20_11:40",
  "entry_time_et": datetime(...) ,        # tz-aware ET
  "short_call_symbol": "...", "long_call_symbol": "...",
  "short_put_symbol": "...",  "long_put_symbol": "...",
  "call_credit": 1.40, "put_credit": 1.40,
  "call_open_qty": 4, "put_open_qty": 4,
  "short_call_stop_price": 3.41, "short_put_stop_price": 3.41,
}
# quotes: {symbol: {"bid": x, "ask": y}, ...}
```

- [ ] **Step 1: Write failing tests**

Append to `tests_v2/test_meic_strategy.py`:

```python
class TestMeicScaleOut:
    def _trade(self, entry="11:40", qty=4, cc=1.40, pc=1.40):
        h, m = (int(x) for x in entry.split(":"))
        return {
            "trade_id": f"2026-05-20_{entry}",
            "entry_time_et": datetime(2026, 5, 20, h, m, tzinfo=ET),
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
            "call_credit": cc, "put_credit": pc,
            "call_open_qty": qty, "put_open_qty": qty,
            "short_call_stop_price": 99.0, "short_put_stop_price": 99.0,
        }

    def _quotes(self, sc_ask, sp_ask, lc_bid=0.50, lp_bid=0.50):
        return {
            "SC": {"bid": sc_ask - 0.05, "ask": sc_ask},
            "SP": {"bid": sp_ask - 0.05, "ask": sp_ask},
            "LC": {"bid": lc_bid, "ask": lc_bid + 0.05},
            "LP": {"bid": lp_bid, "ask": lp_bid + 0.05},
        }

    def test_am_tier_fires_at_55pct(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)  # AM
        # call side: credit 1.40, ask 0.60 -> pnl 57% -> tier 0 (55%) fires
        # put side: ask 1.40 -> pnl 0% -> nothing
        action = s.check_exit(trade, self._quotes(sc_ask=0.60, sp_ask=1.40))
        assert action.action == "take_profit"
        # 25% of 4 = 1 contract of short call, plus paired long
        assert action.leg_quantities.get("SC") == 1
        assert action.leg_quantities.get("LC") == 1
        assert "SP" not in action.leg_quantities

    def test_am_tier_does_not_refire(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)
        q = self._quotes(sc_ask=0.60, sp_ask=1.40)
        s.check_exit(trade, q)            # fires tier 0
        action2 = s.check_exit(trade, q)  # same pnl -> no new tier
        assert action2.action == "hold"

    def test_am_multiple_tiers_one_bar(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)
        # call ask 0.20 -> pnl ~86% -> tiers 0,1,2,3 all fire
        action = s.check_exit(trade, self._quotes(sc_ask=0.20, sp_ask=1.40))
        assert action.action == "take_profit"
        # 4 tiers x 25% of 4 = 4 contracts total
        assert action.leg_quantities.get("SC") == 4

    def test_pm_uses_pm_tiers(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="13:00", qty=4)  # PM
        # call ask 0.49 -> pnl 65% -> PM tier 0 (65%) fires; 50% of 4 = 2
        action = s.check_exit(trade, self._quotes(sc_ask=0.49, sp_ask=1.40))
        assert action.action == "take_profit"
        assert action.leg_quantities.get("SC") == 2

    def test_hold_when_below_first_tier(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade(entry="11:40", qty=4)
        # call ask 1.00 -> pnl ~29% -> below 55% tier
        action = s.check_exit(trade, self._quotes(sc_ask=1.00, sp_ask=1.40))
        assert action.action == "hold"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicScaleOut -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `check_exit` (scale-out portion; stop-loss added in Task 9)**

Replace the `check_exit` stub in `src_v2/strategies/meic.py`:

```python
    def _tiers_for(self, trade: Dict[str, Any]):
        entry = trade.get("entry_time_et")
        is_am = entry is not None and entry.time() < self._am_pm_cutoff
        if is_am:
            return self._am_tiers, self._am_fractions
        return self._pm_tiers, self._pm_fractions

    def check_exit(
        self, trade: Dict[str, Any], quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        trade_id = trade.get("trade_id", "")
        tiers, fractions = self._tiers_for(trade)

        legs_to_close: List[str] = []
        leg_qty: Dict[str, int] = {}
        reasons: List[str] = []
        stop_hit = False

        sides = [
            ("call", "short_call_symbol", "long_call_symbol",
             "call_credit", "call_open_qty", "short_call_stop_price"),
            ("put", "short_put_symbol", "long_put_symbol",
             "put_credit", "put_open_qty", "short_put_stop_price"),
        ]

        for side, short_key, long_key, credit_key, qty_key, stop_key in sides:
            short_sym = trade.get(short_key)
            long_sym = trade.get(long_key)
            if not short_sym:
                continue
            sq = quotes.get(short_sym)
            if sq is None:
                continue
            short_ask = sq.get("ask") or sq.get("price") or 0.0
            credit = trade.get(credit_key, 0.0)
            open_qty = int(trade.get(qty_key, 0))
            if credit <= 0 or open_qty <= 0:
                continue

            # --- Stop-loss check (short ask >= stop trigger) ---
            stop_price = trade.get(stop_key)
            if stop_price is not None and short_ask >= stop_price:
                stop_hit = True
                legs_to_close.append(short_sym)
                leg_qty[short_sym] = open_qty
                if long_sym:
                    legs_to_close.append(long_sym)
                    leg_qty[long_sym] = open_qty
                reasons.append(f"{side}_stop@{short_ask:.2f}")
                continue

            # --- Scale-out tiers ---
            pnl_pct = (credit - short_ask) / credit
            fired = self._scaleout_state.setdefault((trade_id, side), set())
            contracts = 0
            for i, tier in enumerate(tiers):
                if i in fired:
                    continue
                if pnl_pct >= tier:
                    fired.add(i)
                    contracts += int(math.floor(open_qty * fractions[i]))
            if contracts <= 0:
                continue
            contracts = min(contracts, open_qty)

            legs_to_close.append(short_sym)
            leg_qty[short_sym] = contracts
            reasons.append(f"{side}_scaleout x{contracts}")

            # Leftover long: close paired long unless bidless
            if long_sym:
                lq = quotes.get(long_sym)
                long_bid = lq.get("bid", 0.0) if lq else 0.0
                if long_bid >= self._abandon_long_below:
                    legs_to_close.append(long_sym)
                    leg_qty[long_sym] = contracts

        if not legs_to_close:
            return ExitAction(action="hold")

        return ExitAction(
            action="stop_loss" if stop_hit else "take_profit",
            legs_to_close=legs_to_close,
            leg_quantities=leg_qty,
            reason="; ".join(reasons),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicScaleOut -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): check_exit scale-out tiers with no-refire ledger"
```

---

## Task 9: `check_exit` — stop-loss + abandon-long

**Files:**
- Modify: `src_v2/strategies/meic.py` (already wired in Task 8; add tests to lock behaviour)
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write failing/locking tests**

Append to `tests_v2/test_meic_strategy.py`:

```python
class TestMeicStopAndAbandon:
    def _trade(self, entry="11:40", qty=4):
        h, m = (int(x) for x in entry.split(":"))
        return {
            "trade_id": f"2026-05-20_{entry}_stop",
            "entry_time_et": datetime(2026, 5, 20, h, m, tzinfo=ET),
            "short_call_symbol": "SC", "long_call_symbol": "LC",
            "short_put_symbol": "SP", "long_put_symbol": "LP",
            "call_credit": 1.40, "put_credit": 1.40,
            "call_open_qty": qty, "put_open_qty": qty,
            "short_call_stop_price": 3.41, "short_put_stop_price": 3.41,
        }

    def test_stop_loss_triggers_on_call_breach(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade()
        quotes = {
            "SC": {"bid": 3.40, "ask": 3.50},   # ask >= 3.41 stop
            "SP": {"bid": 1.35, "ask": 1.40},
            "LC": {"bid": 0.80, "ask": 0.90},
            "LP": {"bid": 0.50, "ask": 0.60},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "stop_loss"
        assert action.leg_quantities.get("SC") == 4   # all contracts
        assert action.leg_quantities.get("LC") == 4   # paired long
        assert "SP" not in action.leg_quantities

    def test_abandon_long_when_bidless(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        trade = self._trade()
        # call side hits 55% tier (ask 0.60); long call bid 0.03 < 0.05 -> abandon
        quotes = {
            "SC": {"bid": 0.55, "ask": 0.60},
            "SP": {"bid": 1.35, "ask": 1.40},
            "LC": {"bid": 0.03, "ask": 0.08},
            "LP": {"bid": 0.50, "ask": 0.60},
        }
        action = s.check_exit(trade, quotes)
        assert action.action == "take_profit"
        assert action.leg_quantities.get("SC") == 1
        assert "LC" not in action.leg_quantities   # long abandoned
```

- [ ] **Step 2: Run to verify (stop-loss path already implemented in Task 8)**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicStopAndAbandon -v`
Expected: PASS (the Task-8 implementation already covers both branches). If any fail, fix `check_exit` until green.

- [ ] **Step 3: Commit**

```bash
git add tests_v2/test_meic_strategy.py
git commit -m "test(meic): lock stop-loss trigger and bidless-long abandonment"
```

---

## Task 10: `PositionMonitor` honours `leg_quantities`

**Files:**
- Modify: `src_v2/services/position_monitor.py:190-233` (`_check_trade`), `:398-418` (`_close_leg`)
- Test: `tests_v2/test_meic_position_monitor.py`

- [ ] **Step 1: Write failing test**

Create `tests_v2/test_meic_position_monitor.py`:

```python
"""PositionMonitor partial-close behaviour for MEIC."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src_v2.services.position_monitor import PositionMonitor
from src_v2.strategies.base import ExitAction


@pytest.mark.asyncio
async def test_monitor_honours_leg_quantities():
    trade = {"trade_id": "T1", "short_call_symbol": "SC", "long_call_symbol": "LC"}

    db = MagicMock()
    db.get_open_trades = AsyncMock(return_value=[trade])
    db.get_trade = AsyncMock(return_value=None)
    db.update_trade = AsyncMock(return_value=True)

    cache = MagicMock()
    cache.get_cached_quote = AsyncMock(return_value={"bid": 0.5, "ask": 0.6})

    tool_handlers = MagicMock()
    tool_handlers.close_position = AsyncMock(return_value={"order_id": "X"})
    tool_handlers.get_spx_price = AsyncMock(return_value={"price": 5850.0})
    tool_handlers.get_open_positions = AsyncMock(return_value=[{"symbol": "SC"}])

    strategy = MagicMock()
    strategy.leg_pairs.return_value = [("short_call_symbol", "long_call_symbol", "call")]
    strategy.check_exit.return_value = ExitAction(
        action="take_profit",
        legs_to_close=["SC"],
        leg_quantities={"SC": 2},
    )
    strategy.compute_trailing_stop.return_value = None

    mon = PositionMonitor(
        db=db, cache=cache, tool_handlers=tool_handlers,
        strategy=strategy, mode="dry_run",
    )
    await mon._check_trade(trade)

    tool_handlers.close_position.assert_awaited()
    call_kwargs = tool_handlers.close_position.await_args.kwargs
    assert call_kwargs["symbol"] == "SC"
    assert call_kwargs["quantity"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_position_monitor.py -v`
Expected: FAIL — `close_position` called with `quantity=1` (current hard-coded default), not 2.

- [ ] **Step 3: Update `_check_trade` take-profit/stop-loss loops**

In `src_v2/services/position_monitor.py`, change the take-profit close loop (currently `:200-201`):

```python
            if exit_action.action == "take_profit":
                logger.info("[TAKE-PROFIT] %s: %s", trade_id, exit_action.reason)
                for sym in exit_action.legs_to_close:
                    qty = exit_action.leg_quantities.get(sym, 1)
                    await self._close_leg(trade_id, sym, "take_profit_leg",
                                          reason="take_profit", quantity=qty)
                return
```

And the stop-loss close loop (currently `:204-207`):

```python
            if exit_action.action == "stop_loss":
                logger.warning("[STOP-LOSS] %s: %s", trade_id, exit_action.reason)
                for sym in exit_action.legs_to_close:
                    qty = exit_action.leg_quantities.get(sym, 1)
                    await self._close_leg(trade_id, sym, "stop_loss_leg",
                                          reason="stop_loss", quantity=qty)
```

- [ ] **Step 4: Update `_close_leg` signature**

Replace the `_close_leg` method body header (`:398-412`):

```python
    async def _close_leg(
        self,
        trade_id: str,
        symbol: str,
        leg_name: str,
        reason: str,
        quantity: int = 1,
    ):
        """Close a single option leg at market."""
        try:
            result = await self.tool_handlers.close_position(
                symbol=symbol,
                order_type="market",
                quantity=quantity,
                reason=reason,
            )
```

(Leave the rest of `_close_leg` — logging + except — unchanged.)

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_position_monitor.py -v`
Expected: PASS.

- [ ] **Step 6: Verify no regression**

Run: `python -m pytest tests_v2/test_position_monitor_v2.py -v`
Expected: all existing monitor tests pass (default `quantity=1` preserved).

- [ ] **Step 7: Commit**

```bash
git add src_v2/services/position_monitor.py tests_v2/test_meic_position_monitor.py
git commit -m "feat(meic): PositionMonitor honours ExitAction.leg_quantities for partial closes"
```

---

## Task 11: `place_two_spread_ic` tool handler

**Files:**
- Modify: `src_v2/agent/tool_handlers.py` (add method near `place_iron_condor`, ~`:336`)
- Test: `tests_v2/test_meic_wiring.py`

- [ ] **Step 1: Write failing test**

Create `tests_v2/test_meic_wiring.py`:

```python
"""MEIC wiring: tool handler + config routing."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src_v2.agent.tool_handlers import ToolHandlers


@pytest.mark.asyncio
async def test_place_two_spread_ic_dry_run_simulates():
    cache = MagicMock()
    api = MagicMock()
    db = MagicMock()
    th = ToolHandlers(cache=cache, api_client=api, db=db, mode="dry_run")

    payload = {
        "order_type": "meic_two_spreads",
        "quantity": 4,
        "put_spread": {"short_strike": 5750, "long_strike": 5660,
                       "credit": 1.40, "quantity": 4},
        "call_spread": {"short_strike": 5950, "long_strike": 6040,
                        "credit": 1.40, "quantity": 4},
    }
    result = await th.place_two_spread_ic(payload)
    assert result["dry_run"] is True
    assert result["quantity"] == 4
    # Both spread legs reported
    assert result["put_spread"]["short_strike"] == 5750
    assert result["call_spread"]["short_strike"] == 5950
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_wiring.py::test_place_two_spread_ic_dry_run_simulates -v`
Expected: FAIL — `AttributeError: 'ToolHandlers' object has no attribute 'place_two_spread_ic'`.

- [ ] **Step 3: Implement the handler**

Add to `src_v2/agent/tool_handlers.py` (after `place_iron_condor`, before `place_credit_spread`):

```python
    async def place_two_spread_ic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Place a MEIC iron condor as two independent credit spreads.

        Submits the put credit spread, then the call credit spread, each at
        the tranche quantity. In dry_run/manual the fills are simulated.
        """
        qty = int(payload.get("quantity", 1))
        put = payload["put_spread"]
        call = payload["call_spread"]

        if self._block_submission():
            if self.dry_run:
                ts = datetime.now().strftime('%Y%m%d%H%M%S')
                logger.info("[DRY RUN] Simulated MEIC two-spread IC: %s", ts)
                return {
                    "order_id": f"DRY-MEIC-{ts}",
                    "dry_run": True, "simulated": True,
                    "quantity": qty,
                    "put_spread": put,
                    "call_spread": call,
                }
            return self._manual_order_result("place_two_spread_ic", payload)

        put_result = await self.place_credit_spread(
            "put", put["short_strike"], put["long_strike"], qty)
        call_result = await self.place_credit_spread(
            "call", call["short_strike"], call["long_strike"], qty)
        return {
            "order_type": "meic_two_spreads",
            "quantity": qty,
            "put_result": put_result,
            "call_result": call_result,
            "put_spread": put,
            "call_spread": call,
        }
```

- [ ] **Step 4: Route `meic_two_spreads` through the generic `place_order`**

In `src_v2/agent/tool_handlers.py`, at the top of `place_order` (`:475`), add a dispatch before the `_block_submission` check:

```python
    async def place_order(self, order_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generic order placement — routes strategy-built payloads to the API."""
        if order_payload.get("order_type") == "meic_two_spreads":
            return await self.place_two_spread_ic(order_payload)
        if self._block_submission():
```

(Leave the rest of `place_order` unchanged.)

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_wiring.py::test_place_two_spread_ic_dry_run_simulates -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src_v2/agent/tool_handlers.py tests_v2/test_meic_wiring.py
git commit -m "feat(meic): place_two_spread_ic tool handler + place_order dispatch"
```

---

## Task 12: Config + main routing

**Files:**
- Create: `config/config_meic_spx.yaml`
- Modify: `src_v2/main.py:227-240` (`_create_strategy`)
- Test: `tests_v2/test_meic_wiring.py`

- [ ] **Step 1: Write failing test**

Append to `tests_v2/test_meic_wiring.py`:

```python
def test_main_routes_meic_strategy():
    import yaml
    from src_v2.main import BotOrchestrator
    from src_v2.strategies.meic import MeicStrategy

    with open("config/config_meic_spx.yaml") as f:
        config = yaml.safe_load(f)

    orch = BotOrchestrator(config=config, mode="dry_run")
    strategy = orch._create_strategy("meic")
    assert isinstance(strategy, MeicStrategy)
    # configured from YAML
    assert strategy._stop_pct == 0.70
    assert strategy._wing_widths == [80, 90, 100]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_wiring.py::test_main_routes_meic_strategy -v`
Expected: FAIL — config file missing / `meic` falls through to IronCondor.

- [ ] **Step 3: Create the config**

Create `config/config_meic_spx.yaml`:

```yaml
bot:
  strategy: meic
  mode: dry_run

entry:
  schedule:
    - "11:00"
    - "11:20"
    - "11:40"
    - "12:00"
    - "12:20"
    - "12:40"
    - "13:00"
    - "13:20"
    - "13:40"
    - "14:00"
    - "14:20"
    - "14:40"
    - "15:00"
  stability_fraction: 0.20
  single_candle_fraction: 0.12
  fallback_stability_pts: 12.0

strategy:
  am_pm_cutoff: "12:30"
  am_min_credit: 1.25
  am_max_credit: 1.65
  pm_min_credit: 0.80
  pm_max_credit: 1.30
  wing_widths: [80, 90, 100]
  stop_pct_of_total_credit: 0.70
  am_scale_out_tiers: [0.55, 0.65, 0.75, 0.85]
  am_scale_out_fractions: [0.25, 0.25, 0.25, 0.25]
  pm_scale_out_tiers: [0.65, 0.85]
  pm_scale_out_fractions: [0.50, 0.50]
  abandon_long_below: 0.05
  credit_equality_threshold: 0.30
  default_tranche_qty: 3
  tranche_sizing:
    "11:00": 3
    "11:20": 4
    "11:40": 4
    "12:00": 5
    "12:20": 5
    "12:40": 6
    "13:00": 6
    "13:20": 5
    "13:40": 5
    "14:00": 4
    "14:20": 4
    "14:40": 3
    "15:00": 3

risk:
  max_concurrent_trades: 14
  max_daily_risk_percent: 3.0
  max_buying_power_percent: 60.0
  take_profit_threshold: 0.05

monitor:
  poll_interval_seconds: 15
  eod_close_time: "15:50"

options:
  contract_multiplier: 100
  quantity: 1
  slippage_per_spread: 0.10
  commission_per_contract_per_leg: 1.0
  strike_step_pts: 5
  option_root: "SPXW"

llm:
  advisor_model: "anthropic/claude-haiku-4-5"
  debrief_model: "anthropic/claude-sonnet-4-5"

cache:
  option_chain_ttl: 60
  quote_ttl: 60
  account_info_ttl: 60

database:
  trading_db: "data/meic_bot.db"
  cache_db: "data/market_cache.db"

logging:
  log_dir: "logs"
  level: "INFO"
```

- [ ] **Step 4: Add routing in `_create_strategy`**

In `src_v2/main.py`, add a branch in `_create_strategy` (before the final `else`):

```python
        elif strategy_name == "meic":
            from src_v2.strategies.meic import MeicStrategy
            strategy = MeicStrategy()
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_wiring.py::test_main_routes_meic_strategy -v`
Expected: PASS.

- [ ] **Step 6: Smoke-test the CLI wiring (dry-run, one cycle)**

Run: `python -m src_v2.main --config config/config_meic_spx.yaml --once`
Expected: process starts, logs `Strategy created: meic`, runs one cycle, exits cleanly (no traceback). A `waited`/`no_candidates` outcome is fine — we're verifying wiring, not a fill.

- [ ] **Step 7: Commit**

```bash
git add config/config_meic_spx.yaml src_v2/main.py tests_v2/test_meic_wiring.py
git commit -m "feat(meic): config_meic_spx.yaml + main.py strategy routing"
```

---

## Task 13: Backtest path — MEIC day processing

**Files:**
- Modify: `src_v2/backtest/engine.py` (`__init__`, `run`, add `_process_day_meic`, `_simulate_meic_side`)
- Test: `tests_v2/backtest/test_meic_backtest.py`

This task reuses `_build_chain`, `_build_quotes`, `_make_symbol`, `CbboStore`, and the `BacktestResult` dataclass already in the file.

- [ ] **Step 1: Write failing test (worthless-expiry path)**

Create `tests_v2/backtest/test_meic_backtest.py`:

```python
"""MEIC backtest smoke tests with a fake CBBO store."""

import pytest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src_v2.strategies.meic import MeicStrategy
from src_v2.backtest.engine import BacktestEngine

ET = ZoneInfo("America/New_York")


class _FakeCbbo:
    """Minimal CbboStore stand-in with optional time-shifted marks.

    Before `shift_minute` (or always, if None) returns `marks`. From
    `shift_minute` onward, any key present in `shift_marks` overrides.
    This lets a trade enter with a valid in-band credit, then have a
    leg's mark move (breach or decay) later in the session.
    """
    def __init__(self, marks, shift_minute=None, shift_marks=None):
        self._marks = marks                       # {(strike, is_call): (bid, ask)}
        self._shift_minute = shift_minute         # "HH:MM" or None
        self._shift_marks = shift_marks or {}

    def load_day(self, d):
        return len(self._marks)

    def derive_spot(self, minute, ref):
        return 5850.0

    def lookup(self, minute, strike, is_call):
        key = (float(strike), is_call)
        if self._shift_minute and minute >= self._shift_minute:
            if key in self._shift_marks:
                return self._shift_marks[key]
        return self._marks.get(key)


class _Day:
    def __init__(self, bars, sd, symbol="SPX"):
        self.bars = bars
        self.session_date = sd
        self.symbol = symbol


def _bars_flat(d, start="11:00", end="15:50", price=5850.0):
    """One bar per minute, flat price (never breaches any short)."""
    out = []
    h0, m0 = (int(x) for x in start.split(":"))
    h1, m1 = (int(x) for x in end.split(":"))
    t = datetime(d.year, d.month, d.day, h0, m0, tzinfo=ET)
    end_t = datetime(d.year, d.month, d.day, h1, m1, tzinfo=ET)
    while t <= end_t:
        out.append({"ts_et": t, "open": price, "high": price,
                    "low": price, "close": price})
        t = t.replace(minute=t.minute + 1) if t.minute < 59 else \
            t.replace(hour=t.hour + 1, minute=0)
    return out


def _meic_marks():
    """Marks giving call & put credits ~1.40 in AM band; longs cheap."""
    return {
        (5950.0, True): (1.40, 1.50),   # short call
        (6040.0, True): (0.01, 0.09),   # long call (wing 90)
        (5750.0, False): (1.40, 1.50),  # short put
        (5660.0, False): (0.01, 0.09),  # long put (wing 90)
    }


def _config():
    return {
        "bot": {"strategy": "meic"},
        "entry": {"schedule": ["11:00"]},
        "strategy": {
            "am_pm_cutoff": "12:30",
            "am_min_credit": 1.25, "am_max_credit": 1.65,
            "pm_min_credit": 0.80, "pm_max_credit": 1.30,
            "wing_widths": [90],
            "stop_pct_of_total_credit": 0.70,
            "am_scale_out_tiers": [0.55, 0.65, 0.75, 0.85],
            "am_scale_out_fractions": [0.25, 0.25, 0.25, 0.25],
            "pm_scale_out_tiers": [0.65, 0.85],
            "pm_scale_out_fractions": [0.50, 0.50],
            "abandon_long_below": 0.05,
            "credit_equality_threshold": 0.30,
            "default_tranche_qty": 3,
            "tranche_sizing": {"11:00": 3},
            "eod_force_close": "15:50",
            "market_open": "09:30",
        },
        "options": {
            "contract_multiplier": 100, "quantity": 1,
            "strike_step_pts": 5, "option_root": "SPXW",
            "tp_pct_of_credit": 0.50,
        },
        "backtest": {"cbbo_cache_dir": ""},
    }


def _make_engine(monkeypatch, fake_cbbo, config=None):
    # Patch CbboStore construction inside BacktestEngine.__init__
    import src_v2.backtest.engine as eng
    monkeypatch.setattr(eng, "CbboStore", lambda *a, **k: fake_cbbo,
                        raising=False)
    return BacktestEngine(config or _config(), MeicStrategy())


def test_meic_backtest_worthless_expiry(monkeypatch):
    d = date(2026, 5, 20)
    engine = _make_engine(monkeypatch, _FakeCbbo(_meic_marks()))
    days = [_Day(_bars_flat(d), d)]
    results = engine.run(days)

    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2          # one per side
    for r in closed:
        # qty 3, credit ~1.40 -> 1.40 * 100 * 3 = 420
        assert r.realized_pnl_usd == pytest.approx(1.40 * 100 * 3, abs=1.0)
        assert r.exit_reason == "expired_worthless"
        assert r.qty == 3
```

> **Note on `CbboStore` patching:** `BacktestEngine.__init__` imports `CbboStore`
> locally (`from src_v2.backtest.cbbo_store import CbboStore`). The
> `monkeypatch.setattr(eng, "CbboStore", ...)` only works if the import is
> hoisted to module level. **In Step 3, move the `CbboStore` import to the top
> of `engine.py`** so the test patch takes effect.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/backtest/test_meic_backtest.py::test_meic_backtest_worthless_expiry -v`
Expected: FAIL — `BacktestEngine` has no MEIC path (runs ORB flow, returns skip).

- [ ] **Step 3: Hoist `CbboStore` import and add MEIC dispatch**

In `src_v2/backtest/engine.py`:

(a) Add at module top (after the existing `from __future__` / dataclass imports):

```python
from src_v2.backtest.cbbo_store import CbboStore
```

(b) In `BacktestEngine.__init__`, remove the local `from src_v2.backtest.cbbo_store import CbboStore` line, record the strategy kind, and keep using `CbboStore(...)`:

```python
        self._strategy_kind = config.get("bot", {}).get("strategy", "orb_em_ic")
        # ... existing config reads ...
        cbbo_dir = config.get("backtest", {}).get("cbbo_cache_dir", "")
        self._cbbo = CbboStore(cbbo_dir, cache_dir=cbbo_dir)
```

(c) In `run`, dispatch to the MEIC path:

```python
    def run(self, days) -> list[BacktestResult]:
        all_results: list[BacktestResult] = []
        for day in days:
            d = day.session_date
            if self._skip_nfp and _is_nfp_friday(d):
                all_results.append(self._skip_result(d, day.symbol, "nfp_friday"))
                continue
            if self._skip_fomc and d in _FOMC_DATES:
                all_results.append(self._skip_result(d, day.symbol, "fomc_day"))
                continue
            if self._skip_cpi and d in _CPI_DATES:
                all_results.append(self._skip_result(d, day.symbol, "cpi_day"))
                continue

            if self._strategy_kind == "meic":
                all_results.extend(self._process_day_meic(day))
            else:
                all_results.extend(self._process_day(day))
        return all_results
```

(d) Add the MEIC day processor. The schedule + EOD come from config:

```python
    def _process_day_meic(self, day) -> list[BacktestResult]:
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")

        d = day.session_date
        symbol = day.symbol
        bars = day.bars

        schedule = self._config.get("entry", {}).get("schedule", [])
        eod = self._parse_time(
            self._config.get("strategy", {}).get("eod_force_close", "15:50"))
        eod_dt = datetime(d.year, d.month, d.day, eod[0], eod[1], tzinfo=ET)

        bars_by_minute = {b["ts_et"].strftime("%H:%M"): b for b in bars}

        if self._cbbo.load_day(d) == 0:
            return [self._skip_result(d, symbol, "no_cbbo_data")]

        results: list[BacktestResult] = []
        for slot in schedule:
            slot_bar = bars_by_minute.get(slot)
            if slot_bar is None:
                continue
            spot = self._cbbo.derive_spot(slot, slot_bar["close"]) or slot_bar["close"]
            chain = self._build_chain(slot, spot, d)
            if not chain:
                continue

            # Pin strategy clock to this slot for AM/PM band + sizing
            h, m = (int(x) for x in slot.split(":"))
            self._strategy._now_fn = lambda h=h, m=m: datetime(
                d.year, d.month, d.day, h, m, tzinfo=ET)

            search = _run_async(self._strategy.search_candidates(chain, spot))
            setup = search.get("top_candidate")
            if setup is None:
                continue
            val = _run_async(self._strategy.validate(setup, chain))
            if not val.get("passed", False):
                continue

            for leg in setup.legs:
                leg.symbol = self._make_symbol(d, leg.strike, leg.option_type)

            stop_info = self._strategy.compute_stop_loss(setup)
            slot_dt = datetime(d.year, d.month, d.day, h, m, tzinfo=ET)
            monitoring_bars = [b for b in bars if b["ts_et"] > slot_dt]

            for side in ("put", "call"):
                results.append(self._simulate_meic_side(
                    side, setup, stop_info, monitoring_bars,
                    d, symbol, slot_dt, spot, eod_dt))
        if not results:
            return [self._skip_result(d, symbol, "no_meic_trades")]
        return results
```

(e) Add the per-side exit simulator:

```python
    def _simulate_meic_side(
        self, side: str, setup, stop_info: dict, monitoring_bars: list,
        d: date, symbol: str, entry_dt: datetime, spot: float,
        eod_dt: datetime,
    ) -> BacktestResult:
        if side == "put":
            credit = setup.extra["put_credit"]
            short_strike = setup.strikes["short_put"]
            long_strike = setup.strikes["long_put"]
            width = short_strike - long_strike
            short_sym = setup.legs[2].symbol
            long_sym = setup.legs[3].symbol
            stop_price = stop_info["short_put_stop_price"]
        else:
            credit = setup.extra["call_credit"]
            short_strike = setup.strikes["short_call"]
            long_strike = setup.strikes["long_call"]
            width = long_strike - short_strike
            short_sym = setup.legs[0].symbol
            long_sym = setup.legs[1].symbol
            stop_price = stop_info["short_call_stop_price"]

        qty = int(setup.extra.get("qty", 1))
        trade = {
            "trade_id": f"{d.isoformat()}_{entry_dt.strftime('%H%M')}_{side}",
            "entry_time_et": entry_dt,
            "short_call_symbol": short_sym if side == "call" else None,
            "long_call_symbol": long_sym if side == "call" else None,
            "short_put_symbol": short_sym if side == "put" else None,
            "long_put_symbol": long_sym if side == "put" else None,
            "call_credit": credit if side == "call" else 0.0,
            "put_credit": credit if side == "put" else 0.0,
            "call_open_qty": qty if side == "call" else 0,
            "put_open_qty": qty if side == "put" else 0,
            "short_call_stop_price": stop_price if side == "call" else None,
            "short_put_stop_price": stop_price if side == "put" else None,
        }
        qty_key = "call_open_qty" if side == "call" else "put_open_qty"

        base = BacktestResult(
            trade_id=trade["trade_id"], session_date=d, symbol=symbol,
            orb_high=0, orb_low=0, orb_midline=0, orb_range_pts=0,
            spread_type=side, short_strike=short_strike,
            long_strike=long_strike, spread_width=width,
            entry_time=entry_dt, underlying_at_entry=spot,
            credit_received=credit, qty=qty, status="closed",
        )

        realized = 0.0   # dollars
        mult = self._multiplier

        for bar in monitoring_bars:
            bt = bar["ts_et"]
            if bt >= eod_dt:
                break
            minute = bt.strftime("%H:%M")

            quotes = self._build_quotes(minute, trade)
            if not quotes:
                continue

            action = self._strategy.check_exit(trade, quotes)

            if action.action == "stop_loss":
                open_q = int(trade[qty_key])
                debit = min(max(stop_info["stop_amount"], 0), width)
                realized += (credit - debit) * mult * open_q
                trade[qty_key] = 0
                base.exit_time = bt
                base.underlying_at_exit = self._cbbo.derive_spot(minute, bar["close"]) or bar["close"]
                base.debit_paid = round(debit, 2)
                base.exit_reason = "short_stopped"
                base.realized_pnl_usd = round(realized, 2)
                return base

            if action.action == "take_profit":
                closed_q = action.leg_quantities.get(short_sym, 0)
                if closed_q > 0:
                    # debit at scale-out = (1 - tp implied) of credit per contract.
                    # Use the short ask as fill (cost to buy back).
                    sq = quotes.get(short_sym, {})
                    debit = sq.get("ask", 0.0)
                    realized += (credit - debit) * mult * closed_q
                    trade[qty_key] = int(trade[qty_key]) - closed_q
                if int(trade[qty_key]) <= 0:
                    base.exit_time = bt
                    base.underlying_at_exit = self._cbbo.derive_spot(minute, bar["close"]) or bar["close"]
                    base.exit_reason = "scaled_out"
                    base.realized_pnl_usd = round(realized, 2)
                    return base

        # EOD / no exit: remaining qty expires worthless (keep full credit)
        remaining = int(trade[qty_key])
        realized += credit * mult * remaining
        base.exit_time = eod_dt
        base.debit_paid = 0.0
        base.exit_reason = "expired_worthless" if remaining == qty else "partial_then_expiry"
        base.realized_pnl_usd = round(realized, 2)
        return base
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/backtest/test_meic_backtest.py::test_meic_backtest_worthless_expiry -v`
Expected: PASS.

- [ ] **Step 5: Add scale-out + stop-loss backtest tests**

Append to `tests_v2/backtest/test_meic_backtest.py`:

```python
def test_meic_backtest_stop_loss(monkeypatch):
    d = date(2026, 5, 20)
    # Enter at 11:00 with valid 1.40 credit; from 11:30 the call short ask
    # jumps to 3.60 (>= stop trigger 1.45 + 0.70*2.80 = 3.41) -> stop fills.
    fake = _FakeCbbo(
        _meic_marks(),
        shift_minute="11:30",
        shift_marks={(5950.0, True): (3.50, 3.60)},
    )
    engine = _make_engine(monkeypatch, fake)
    days = [_Day(_bars_flat(d), d)]
    results = engine.run(days)

    call = next(r for r in results if r.spread_type == "call")
    # stop fills at stop_amount 1.96 -> pnl per contract (1.40-1.96)*100 = -56, x3 = -168
    assert call.exit_reason == "short_stopped"
    assert call.realized_pnl_usd == pytest.approx((1.40 - 1.96) * 100 * 3, abs=1.0)
    # Put side never moved -> expires worthless for full credit.
    put = next(r for r in results if r.spread_type == "put")
    assert put.exit_reason == "expired_worthless"


def _config_qty4():
    """Config with tranche qty 4 so 25% AM tiers close 1 contract each."""
    cfg = _config()
    cfg["strategy"]["tranche_sizing"] = {"11:00": 4}
    cfg["strategy"]["default_tranche_qty"] = 4
    return cfg


def test_meic_backtest_scale_out_full_exit(monkeypatch):
    d = date(2026, 5, 20)
    # qty=4: floor(4*0.25)=1 per AM tier. Enter at 11:00 (credit 1.40);
    # from 11:30 the call short collapses to ask 0.20 -> pnl ~86% -> all
    # four AM tiers fire in one bar -> 4 contracts closed -> full exit.
    fake = _FakeCbbo(
        _meic_marks(),
        shift_minute="11:30",
        shift_marks={(5950.0, True): (0.15, 0.20)},
    )
    engine = _make_engine(monkeypatch, fake, config=_config_qty4())
    days = [_Day(_bars_flat(d), d)]
    results = engine.run(days)

    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "scaled_out"
    assert call.qty == 4
    # Each of 4 contracts bought back at ask 0.20: (1.40-0.20)*100*4 = 480
    assert call.realized_pnl_usd == pytest.approx((1.40 - 0.20) * 100 * 4, abs=1.0)
```

> **Floor-rounding note:** with `qty=3` and 25% fractions, `floor(3*0.25)=0`,
> so AM tiers close nothing until tranche size ≥ 4. The scale-out test uses
> `qty=4` to exercise real partials. The floor behaviour for sizes < 4 is
> expected v1 behaviour (documented in the spec's known gaps).

- [ ] **Step 6: Run all backtest tests**

Run: `python -m pytest tests_v2/backtest/test_meic_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add src_v2/backtest/engine.py tests_v2/backtest/test_meic_backtest.py
git commit -m "feat(meic): backtest path — tranche entries, scale-out, stop-loss simulation"
```

---

## Task 14: Full suite + README update

**Files:**
- Modify: `src_v2/README.md` (strategy comparison table + adding-a-strategy note)
- Test: full `tests_v2` suite

- [ ] **Step 1: Run the entire src_v2 suite**

Run: `python -m pytest tests_v2/ -q`
Expected: all tests pass (existing + new MEIC tests).

- [ ] **Step 2: Run the locked suite (must remain green)**

Run: `python -m pytest tests_locked/ -q`
Expected: all locked tests pass (no `src/` changes were made).

- [ ] **Step 3: Add MEIC row to the README strategy comparison**

In `src_v2/README.md`, add a column/row for MEIC in the "Strategy Comparison"
table:

```markdown
| Aspect | IronCondor | IronFly | OrbEmIC | MEIC |
|--------|-----------|---------|---------|------|
| Engine | Scheduled | Scheduled | OrbEngine | Scheduled |
| Entry signal | Time window + stability | Time window + EM | ORB breakout + retest | Tranche windows 11:00–15:00 |
| Short strikes | Delta-based | ATM | EM-scaled | Credit-band (AM/PM) |
| Wings | 30pt | EM x 0.5 | 5/10pt | 80–100pt |
| TP | Per-leg $0.05 | Fixed $ | 50% credit | AM 55/65/75/85%, PM 65/85% |
| SL | Credit OCO | Wing breach | Breach | Stop-market % of total credit |
| Sizing | 1 | 1 | 1 | Per-tranche (3–6) |
```

And under "Strategy Selection", add `meic` to the list of valid values.

- [ ] **Step 4: Commit**

```bash
git add src_v2/README.md
git commit -m "docs(meic): add MEIC to strategy comparison and selection list"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** Tasks 1–14 cover every spec section — ExitAction widening
  (T1), MeicStrategy methods (T3–T9), PositionMonitor partials (T10),
  two-spread routing (T11), config + routing (T12), backtest (T13), docs (T14).
- **Known v1 gaps (from spec, intentionally not implemented):** scale-out
  ledger is in-memory (re-fires on restart); floor-rounding means tranche
  sizes < 4 close nothing on 25% AM tiers; stop-market backtest fills at the
  exact trigger (optimistic); dynamic stop-pct is a config knob only.
- **Type consistency check:** `_now_fn`, `_active_band`, `_tranche_qty`,
  `_tiers_for`, `_scaleout_state` are defined in `meic.py` and used
  consistently. Trade-dict keys (`entry_time_et`, `*_open_qty`,
  `short_*_stop_price`, `*_credit`, `*_symbol`) match between `check_exit`
  (T8/T9) and the backtest's `_simulate_meic_side` (T13).
- **Reuse:** No new engine, no DB schema change, no duplicated CBBO/report
  logic. `ScheduledEngine`, `PositionMonitor`, `BacktestEngine` helpers,
  `CbboStore`, `mid_price` all reused.
```
