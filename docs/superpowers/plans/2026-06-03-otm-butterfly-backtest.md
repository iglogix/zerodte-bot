# OTM 0DTE Butterfly Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure, unit-tested analytical backtest for the directional OTM 0DTE SPX butterfly so we can measure whether it has positive net-of-cost out-of-sample expectancy.

**Architecture:** Mirror the Jade Lizard pattern — a pure-logic core (`src_v2/backtest/butterfly_core.py`, no I/O, operates on the in-memory quote dict `{(minute, strike, is_call): (bid, ask)}`) plus a thin CLI evaluator (`scripts/analytical_butterfly_eval.py`) that loads the SPXW CBBO cache, bars, and VIX. A wider-window backfill script produces the chain cache. The core reuses JL's chain helpers (`quote_mid`, `atm_from_chain`, `compute_em`, `forward_from_chain`, `settlement_forward`).

**Tech Stack:** Python 3, dataclasses, pytest, Databento (backfill only), existing `src_v2/backtest` modules.

**Spec:** `docs/superpowers/specs/2026-06-03-otm-butterfly-backtest-design.md`

**Run tests with the project venv:** `.venv/bin/python -m pytest`

---

## File structure

| File | Responsibility |
|---|---|
| `src_v2/backtest/butterfly_core.py` | Pure logic: config, strike selection, debit/mark/intrinsic, gates, Hull-MA direction, regime width, exit models, day eval, aggregation, WF folds |
| `tests/test_butterfly_core.py` | Unit tests for all core logic |
| `scripts/backfill_butterfly_cbbo.py` | Wider (±250pt) Databento backfill → `data/cbbo_cache_fly_10min/` |
| `scripts/analytical_butterfly_eval.py` | CLI: load data, resolve per-day side/width, run scenarios + WF-CV sweep, write CSV/JSON |
| `config/config_butterfly_otm_spx.yaml` | Reference defaults + sweep grid |
| `docs/spx_0dte_butterfly_otm_findings.md` | Results write-up (stub now, filled after runs) |

---

## Task 1: Core scaffolding — config, strike selection, intrinsic

**Files:**
- Create: `src_v2/backtest/butterfly_core.py`
- Test: `tests/test_butterfly_core.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_butterfly_core.py
from src_v2.backtest import butterfly_core as bf


def test_round_to_step():
    assert bf.round_to_step(5022.0, 5.0) == 5020.0
    assert bf.round_to_step(5023.0, 5.0) == 5025.0


def test_select_fly_strikes_call_side():
    cfg = bf.ButterflyConfig(side="call", body_em_mult=1.0, width=30.0)
    # forward 5000, em 20 -> body round(5020)=5020; wings +/-30
    assert bf.select_fly_strikes(5000.0, 20.0, cfg) == (4990.0, 5020.0, 5050.0)


def test_select_fly_strikes_put_side():
    cfg = bf.ButterflyConfig(side="put", body_em_mult=1.0, width=30.0)
    # forward 5000, em 20 -> body round(4980)=4980; wings +/-30
    assert bf.select_fly_strikes(5000.0, 20.0, cfg) == (4950.0, 4980.0, 5010.0)


def test_fly_intrinsic_call():
    strikes = (4990.0, 5020.0, 5050.0)
    assert bf.fly_intrinsic(strikes, True, 5020.0) == 30.0   # peak at body = width
    assert bf.fly_intrinsic(strikes, True, 4980.0) == 0.0    # below low wing
    assert bf.fly_intrinsic(strikes, True, 5060.0) == 0.0    # above high wing
    assert bf.fly_intrinsic(strikes, True, 5005.0) == 15.0   # midway low->body


def test_fly_intrinsic_put():
    strikes = (4950.0, 4980.0, 5010.0)
    assert bf.fly_intrinsic(strikes, False, 4980.0) == 30.0  # peak at body
    assert bf.fly_intrinsic(strikes, False, 5020.0) == 0.0   # above high wing
    assert bf.fly_intrinsic(strikes, False, 4940.0) == 0.0   # below low wing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src_v2.backtest.butterfly_core'`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/butterfly_core.py
"""Pure logic for the 0DTE SPX OTM butterfly backtest.

No I/O — operates on the in-memory quote dict
{(minute_et, strike, is_call): (bid, ask)} so it is fully unit-testable.

Reuses JL chain helpers (quote_mid / atm_from_chain / compute_em /
forward_from_chain / settlement_forward). A 1-2-1 long debit butterfly is a
DEBIT structure: profit = exit_value - debit_paid (opposite sign from JL's
credit structures). Direction (call/put side) is set by a 21-period Hull MA
trend; width is set by a VIX regime map. See the design spec for the rule set.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from src_v2.backtest.jl_core import (
    make_anchors, quote_mid, atm_from_chain, compute_em, forward_from_chain,
    settlement_forward, _max_consec_losses, _max_drawdown,
)

Strikes = Tuple[float, float, float]   # (k_low, k_body, k_high)


def round_to_step(x: float, step: float) -> float:
    return round(x / step) * step


@dataclass
class ButterflyConfig:
    entry_anchor: str = "09:30"
    anchors: List[str] = field(default_factory=lambda: make_anchors("09:30", "16:00", 10))
    side: str = "call"            # "call" or "put" (set per-day by the direction filter)
    body_em_mult: float = 1.0     # body placed body_em_mult * EM OTM from the forward
    width: float = 30.0           # symmetric wing width (points)
    step: float = 5.0
    multiplier: int = 100
    # entry gates
    gates_on: bool = True
    max_debit_pct: float = 0.10   # debit <= max_debit_pct * width
    min_r2r: float = 9.0          # (width - debit) / debit >= min_r2r
    # exit model: "hold" | "target" | "trail"
    exit_model: str = "hold"
    target_mult: float = 2.0      # target: exit when profit >= target_mult * debit
    trail_arm_pct: float = 0.50   # trail: arm once profit >= trail_arm_pct * debit
    eod_anchor: str = "15:50"     # no mid-exit trigger fires after this anchor
    # costs (headline = 0)
    slippage_per_leg: float = 0.0
    commission_per_leg: float = 0.0


def select_fly_strikes(forward: float, em: float, cfg: ButterflyConfig) -> Strikes:
    sign = 1.0 if cfg.side == "call" else -1.0
    k_body = round_to_step(forward + sign * cfg.body_em_mult * em, cfg.step)
    return (k_body - cfg.width, k_body, k_body + cfg.width)


def fly_intrinsic(strikes: Strikes, is_call: bool, S: float) -> float:
    k_low, k_body, k_high = strikes
    if is_call:
        return (max(S - k_low, 0.0) - 2 * max(S - k_body, 0.0) + max(S - k_high, 0.0))
    return (max(k_high - S, 0.0) - 2 * max(k_body - S, 0.0) + max(k_low - S, 0.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/butterfly_core.py tests/test_butterfly_core.py
git commit -m "feat(fly): butterfly core config, strike selection, intrinsic"
```

---

## Task 2: Debit, mark, and entry gates

**Files:**
- Modify: `src_v2/backtest/butterfly_core.py`
- Test: `tests/test_butterfly_core.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_butterfly_core.py

def _call_legs_quotes(minute):
    # call mids: low=15.0, body=8.0, high=3.0 -> value = 15 + 3 - 2*8 = 2.0
    return {
        (minute, 4990.0, True): (14.5, 15.5),
        (minute, 5020.0, True): (7.5, 8.5),
        (minute, 5050.0, True): (2.5, 3.5),
    }


def test_fly_value_call():
    q = _call_legs_quotes("09:30")
    assert bf.fly_value(q, "09:30", (4990.0, 5020.0, 5050.0), True) == 2.0


def test_fly_value_none_when_leg_missing():
    q = {("09:30", 4990.0, True): (14.5, 15.5)}
    assert bf.fly_value(q, "09:30", (4990.0, 5020.0, 5050.0), True) is None


def test_fly_mark_clamps_negative_to_zero():
    # noisy mids producing a slightly negative raw value clamp to 0
    q = {
        ("10:00", 4990.0, True): (1.0, 1.0),
        ("10:00", 5020.0, True): (1.0, 1.0),
        ("10:00", 5050.0, True): (0.0, 0.0),   # mid None -> treated as missing
    }
    # high leg missing -> mark None
    assert bf.fly_mark(q, "10:00", (4990.0, 5020.0, 5050.0), True) is None
    q[("10:00", 5050.0, True)] = (0.4, 0.6)    # mid 0.5; raw = 1 + 0.5 - 2 = -0.5 -> clamp 0
    assert bf.fly_mark(q, "10:00", (4990.0, 5020.0, 5050.0), True) == 0.0


def test_passes_gates():
    cfg = bf.ButterflyConfig(width=30.0, max_debit_pct=0.10, min_r2r=9.0)
    assert bf.passes_gates(2.0, cfg) is True      # 2/30 = 6.7%, r2r 14 -> ok
    assert bf.passes_gates(3.0, cfg) is True       # 3/30 = 10%, r2r 9 -> exactly ok
    assert bf.passes_gates(4.0, cfg) is False      # 4/30 = 13.3% -> too rich
    assert bf.passes_gates(0.0, cfg) is False       # non-positive debit
    cfg_off = bf.ButterflyConfig(width=30.0, gates_on=False)
    assert bf.passes_gates(9.9, cfg_off) is True    # gates disabled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'fly_value'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src_v2/backtest/butterfly_core.py

def fly_value(quotes, minute: str, strikes: Strikes, is_call: bool,
              fallback_min: int = 5) -> Optional[float]:
    """Raw combined value of the 1-2-1 fly at `minute` (long low + long high - 2*body)."""
    k_low, k_body, k_high = strikes
    lo = quote_mid(quotes, minute, k_low, is_call, fallback_min)
    bo = quote_mid(quotes, minute, k_body, is_call, fallback_min)
    hi = quote_mid(quotes, minute, k_high, is_call, fallback_min)
    if lo is None or bo is None or hi is None:
        return None
    return lo + hi - 2 * bo


def fly_mark(quotes, minute: str, strikes: Strikes, is_call: bool,
             fallback_min: int = 5) -> Optional[float]:
    """Closeable mark = fly_value clamped to [0, +inf) (a long fly is never worth < 0)."""
    v = fly_value(quotes, minute, strikes, is_call, fallback_min)
    return None if v is None else max(v, 0.0)


def passes_gates(debit: float, cfg: ButterflyConfig) -> bool:
    if not cfg.gates_on:
        return True
    if debit <= 0:
        return False
    r2r = (cfg.width - debit) / debit
    return debit <= cfg.max_debit_pct * cfg.width and r2r >= cfg.min_r2r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/butterfly_core.py tests/test_butterfly_core.py
git commit -m "feat(fly): fly value/mark and entry gates"
```

---

## Task 3: Direction filter (Hull MA), coin-flip, regime width

**Files:**
- Modify: `src_v2/backtest/butterfly_core.py`
- Test: `tests/test_butterfly_core.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_butterfly_core.py
from datetime import date as _date


def test_wma_weights_recent_most():
    # WMA of [1,2,3] with n=3: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
    assert abs(bf.wma([1.0, 2.0, 3.0], 3) - (14.0 / 6.0)) < 1e-9


def test_hull_trend_up_and_down():
    up = [float(x) for x in range(1, 40)]          # strictly rising
    assert bf.hull_trend(up, 21) > 0
    down = [float(x) for x in range(40, 1, -1)]     # strictly falling
    assert bf.hull_trend(down, 21) < 0


def test_hull_trend_insufficient_history_returns_zero():
    assert bf.hull_trend([1.0, 2.0, 3.0], 21) == 0


def test_coin_flip_side_deterministic_by_date():
    # parity of ordinal: even -> call, odd -> put; same date always same side
    d = _date(2025, 1, 2)
    assert bf.coin_flip_side(d) in ("call", "put")
    assert bf.coin_flip_side(d) == bf.coin_flip_side(d)
    assert bf.coin_flip_side(_date(2025, 1, 2)) != bf.coin_flip_side(_date(2025, 1, 3))


def test_resolve_side_modes():
    up = [float(x) for x in range(1, 40)]
    assert bf.resolve_side("call", _date(2025, 1, 2), up) == "call"
    assert bf.resolve_side("put", _date(2025, 1, 2), up) == "put"
    assert bf.resolve_side("hull", _date(2025, 1, 2), up) == "call"
    assert bf.resolve_side("anti", _date(2025, 1, 2), up) == "put"


def test_regime_width():
    assert bf.regime_width(14.0) == 25.0    # Zombieland (<17)
    assert bf.regime_width(22.0) == 40.0    # Goldilocks (17-32)
    assert bf.regime_width(40.0) == 55.0    # Chaos (>=32)
    assert bf.regime_width(None) == 40.0    # missing VIX -> mid default
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'wma'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src_v2/backtest/butterfly_core.py

def wma(values: List[float], n: int) -> Optional[float]:
    """Linearly-weighted moving average of the last n values (weight = recency)."""
    if len(values) < n or n <= 0:
        return None
    window = values[-n:]
    denom = n * (n + 1) / 2
    return sum((i + 1) * window[i] for i in range(n)) / denom


def _hma_series(values: List[float], n: int) -> List[float]:
    """Hull MA series: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half, sq = max(1, n // 2), max(1, int(round(math.sqrt(n))))
    raw: List[float] = []
    for i in range(len(values)):
        w_half = wma(values[: i + 1], half)
        w_full = wma(values[: i + 1], n)
        if w_half is None or w_full is None:
            raw.append(float("nan"))
        else:
            raw.append(2 * w_half - w_full)
    out: List[float] = []
    for i in range(len(raw)):
        valid = [x for x in raw[: i + 1] if not math.isnan(x)]
        h = wma(valid, sq) if len(valid) >= sq else None
        out.append(h if h is not None else float("nan"))
    return out


def hull_trend(daily_closes: List[float], n: int = 21) -> int:
    """Sign of the Hull-MA slope using only the supplied (pre-entry) closes.

    Returns +1 (up), -1 (down), or 0 (flat / insufficient history).
    """
    hma = _hma_series(daily_closes, n)
    valid = [x for x in hma if not math.isnan(x)]
    if len(valid) < 2:
        return 0
    slope = valid[-1] - valid[-2]
    return 1 if slope > 0 else (-1 if slope < 0 else 0)


def coin_flip_side(d: date) -> str:
    """Deterministic ~50/50 baseline: even ordinal -> call, odd -> put."""
    return "call" if d.toordinal() % 2 == 0 else "put"


def resolve_side(mode: str, d: date, closes_before: List[float], n: int = 21) -> str:
    """Map a direction MODE to a side. closes_before excludes the entry day."""
    if mode == "call":
        return "call"
    if mode == "put":
        return "put"
    if mode == "coin":
        return coin_flip_side(d)
    t = hull_trend(closes_before, n)
    if mode == "anti":
        return "put" if t >= 0 else "call"
    return "call" if t >= 0 else "put"   # "hull" (default)


def regime_width(vix: Optional[float]) -> float:
    """Representative wing width per VIX regime (deck bands: <17 / 17-32 / 32+)."""
    if vix is None:
        return 40.0
    if vix < 17:
        return 25.0
    if vix < 32:
        return 40.0
    return 55.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: PASS (15 tests total)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/butterfly_core.py tests/test_butterfly_core.py
git commit -m "feat(fly): Hull-MA direction, coin-flip baseline, regime width"
```

---

## Task 4: Per-day evaluation with three exit models

**Files:**
- Modify: `src_v2/backtest/butterfly_core.py`
- Test: `tests/test_butterfly_core.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_butterfly_core.py

def _atm_chain_5000(minute):
    """Chain so atm=5000, em=20, forward=5000 (call_mid==put_mid==10 at 5000)."""
    return {
        (minute, 4990.0, True): (14.5, 15.5), (minute, 4990.0, False): (4.5, 5.5),
        (minute, 5000.0, True): (9.5, 10.5),  (minute, 5000.0, False): (9.5, 10.5),
        (minute, 5010.0, True): (4.5, 5.5),   (minute, 5010.0, False): (14.5, 15.5),
    }


def _entry_chain_fly():
    """09:30 chain: atm/em/forward = 5000/20/5000, call fly debit = 2.0.

    body_em_mult 1.0, width 30 -> strikes (4990, 5020, 5050).
    call mids low=15 (from atm chain), body=8, high=3 -> debit 15+3-16 = 2.0.
    """
    q = _atm_chain_5000("09:30")
    q.update({
        ("09:30", 5020.0, True): (7.5, 8.5),   # body mid 8.0
        ("09:30", 5050.0, True): (2.5, 3.5),   # high mid 3.0
    })
    return q


def _final_anchor_forward_5020(q):
    """Give the 16:00 anchor an ATM chain at 5020 -> settlement forward 5020."""
    q.update({
        ("16:00", 5015.0, True): (12.0, 13.0), ("16:00", 5015.0, False): (7.0, 8.0),
        ("16:00", 5020.0, True): (9.5, 10.5),  ("16:00", 5020.0, False): (9.5, 10.5),
        ("16:00", 5025.0, True): (7.0, 8.0),   ("16:00", 5025.0, False): (12.0, 13.0),
    })
    return q


def test_evaluate_day_hold_to_expiry_win():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold")
    q = _final_anchor_forward_5020(_entry_chain_fly())
    r = bf.evaluate_day(q, cfg)
    assert r["skip"] is None
    assert r["kind"] == "expiry"
    assert r["debit"] == 2.0
    assert r["exit_mark"] == 30.0           # intrinsic at S=5020 (body) = width
    assert r["pnl"] == 2800.0               # (30 - 2) * 100


def test_evaluate_day_gate_blocks_rich_debit():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold")
    q = _entry_chain_fly()
    q[("09:30", 5020.0, True)] = (6.0, 6.0)   # body mid 6 -> debit 15+3-12 = 6.0 (20% of 30)
    r = bf.evaluate_day(q, cfg)
    assert r["skip"] == "gate"


def test_evaluate_day_target_exit():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "12:00", "16:00"],
                             exit_model="target", target_mult=2.0)
    q = _final_anchor_forward_5020(_entry_chain_fly())
    # 12:00 mark: low=9, body=1, high=0.5 -> value = 9 + 0.5 - 2 = 7.5; profit 5.5 >= 2*2=4
    q.update({
        ("12:00", 4990.0, True): (8.5, 9.5), ("12:00", 5020.0, True): (0.5, 1.5),
        ("12:00", 5050.0, True): (0.0, 1.0),
    })
    r = bf.evaluate_day(q, cfg)
    assert r["kind"] == "target"
    assert r["exit_anchor"] == "12:00"
    assert r["exit_mark"] == 7.5
    assert r["pnl"] == 550.0                # (7.5 - 2) * 100


def test_evaluate_day_trail_exit():
    cfg = bf.ButterflyConfig(side="call",
                             anchors=["09:30", "10:00", "10:30", "16:00"],
                             exit_model="trail", trail_arm_pct=0.50)
    q = _final_anchor_forward_5020(_entry_chain_fly())
    # 10:00 profit peaks; 10:30 retraces past the AM giveback (0.625 of peak) -> exit
    # 10:00 value: low=10, body=1, high=0 -> 10 + 0 - 2 = 8.0 ; profit 6.0 (armed, peak 6)
    q.update({
        ("10:00", 4990.0, True): (9.5, 10.5), ("10:00", 5020.0, True): (0.5, 1.5),
        ("10:00", 5050.0, True): (0.0, 0.0),
    })
    # 10:30 value: low=4, body=1, high=0 -> 4 - 2 = 2.0 ; profit 0.0 <= peak*(1-0.625)=2.25
    q.update({
        ("10:30", 4990.0, True): (3.5, 4.5), ("10:30", 5020.0, True): (0.5, 1.5),
        ("10:30", 5050.0, True): (0.0, 0.0),
    })
    r = bf.evaluate_day(q, cfg)
    assert r["kind"] == "trail"
    assert r["exit_anchor"] == "10:30"
    assert r["exit_mark"] == 2.0
    assert r["pnl"] == 0.0                   # (2.0 - 2.0) * 100


def test_evaluate_day_with_costs():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold",
                             slippage_per_leg=0.05, commission_per_leg=1.0)
    q = _final_anchor_forward_5020(_entry_chain_fly())
    r = bf.evaluate_day(q, cfg)
    # gross 2800; expiry cash-settle: entry slip 0.05*4*100 = 20; no exit slip;
    # commission 4 legs * 1.0 (open only) = 4 -> net 2800 - 24 = 2776
    assert r["pnl"] == 2776.0


def test_evaluate_day_skips_no_atm():
    cfg = bf.ButterflyConfig()
    assert bf.evaluate_day({}, cfg)["skip"] == "no_atm"


def test_trail_giveback_by_time_of_day():
    assert bf.trail_giveback("11:30") == 0.625
    assert bf.trail_giveback("12:30") == 0.35
    assert bf.trail_giveback("14:30") == 0.30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'evaluate_day'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src_v2/backtest/butterfly_core.py

def _to_min(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def trail_giveback(minute: str) -> float:
    """Time-of-day trailing giveback (fraction of peak profit allowed back).

    Published rule: wide AM (50-75% -> 0.625), tighter midday (30-40% -> 0.35),
    tight PM (30% -> 0.30).
    """
    m = _to_min(minute)
    if m < _to_min("12:00"):
        return 0.625
    if m < _to_min("14:00"):
        return 0.35
    return 0.30


def evaluate_day(quotes, cfg: ButterflyConfig,
                 spot_close: Optional[float] = None) -> Dict:
    """Evaluate one trading day. Returns {'skip': reason} or a trade-result dict."""
    ea = cfg.entry_anchor
    atm = atm_from_chain(quotes, ea)
    if atm is None:
        return {"skip": "no_atm"}
    em = compute_em(quotes, ea, atm)
    if em is None or em <= 0:
        return {"skip": "no_em"}
    fwd = forward_from_chain(quotes, ea, atm)
    if fwd is None:
        return {"skip": "no_forward"}

    strikes = select_fly_strikes(fwd, em, cfg)
    is_call = cfg.side == "call"
    debit = fly_value(quotes, ea, strikes, is_call)
    if debit is None or debit <= 0:
        return {"skip": "no_debit"}
    if not passes_gates(debit, cfg):
        return {"skip": "gate"}

    kind: Optional[str] = None
    exit_anchor: Optional[str] = None
    exit_mark: Optional[float] = None
    peak_profit = 0.0
    eod = _to_min(cfg.eod_anchor)

    if cfg.exit_model in ("target", "trail"):
        for a in cfg.anchors:
            if a == ea or _to_min(a) > eod:
                continue
            m = fly_mark(quotes, a, strikes, is_call)
            if m is None:
                continue
            profit = m - debit
            if cfg.exit_model == "target":
                if profit >= cfg.target_mult * debit:
                    kind, exit_anchor, exit_mark = "target", a, m
                    break
            else:  # trail
                peak_profit = max(peak_profit, profit)
                if (peak_profit >= cfg.trail_arm_pct * debit
                        and profit <= peak_profit * (1 - trail_giveback(a))):
                    kind, exit_anchor, exit_mark = "trail", a, m
                    break

    if kind is None:   # hold model, or no mid-exit trigger fired -> settle at expiry
        S = settlement_forward(quotes, cfg.anchors)
        if S is None:
            S = spot_close
        if S is None:
            return {"skip": "no_settle"}
        kind, exit_anchor, exit_mark = "expiry", cfg.anchors[-1], fly_intrinsic(strikes, is_call, S)

    mult = cfg.multiplier
    gross = (exit_mark - debit) * mult
    n_legs = 4   # 1 + 2 + 1 contracts
    entry_slip = cfg.slippage_per_leg * n_legs
    exit_slip = 0.0 if kind == "expiry" else cfg.slippage_per_leg * n_legs
    n_comm = n_legs if kind == "expiry" else n_legs * 2
    cost = (entry_slip + exit_slip) * mult + cfg.commission_per_leg * n_comm
    net = gross - cost

    return {
        "skip": None, "kind": kind, "side": cfg.side, "exit_anchor": exit_anchor,
        "debit": debit, "exit_mark": exit_mark, "width": cfg.width,
        "gross": gross, "pnl": net, "strikes": strikes, "em": em, "forward": fwd,
        "r2r_theo": (cfg.width - debit) / debit if debit > 0 else 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: PASS (23 tests total)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/butterfly_core.py tests/test_butterfly_core.py
git commit -m "feat(fly): per-day eval with hold/target/trail exit models + costs"
```

---

## Task 5: Aggregation, per-regime breakdown, and WF folds

**Files:**
- Modify: `src_v2/backtest/butterfly_core.py`
- Test: `tests/test_butterfly_core.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_butterfly_core.py

def test_make_wf_folds_ratio_5_4folds():
    dates = [_date(2025, 1, 1).fromordinal(_date(2025, 1, 1).toordinal() + i) for i in range(90)]
    folds = bf.make_wf_folds(dates, ratio=5, folds=4)
    assert len(folds) == 4
    # expanding train, 1-step OOS val (step = 90 // 9 = 10)
    train0, val0 = folds[0]
    assert len(train0) == 50 and len(val0) == 10
    train3, val3 = folds[3]
    assert len(train3) == 80 and len(val3) == 10
    # val windows are disjoint and strictly after their train
    assert max(train0) < min(val0)


def test_evaluate_aggregates_pnl_and_winrate():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold")
    win_q = _final_anchor_forward_5020(_entry_chain_fly())          # +2800
    loss_q = _entry_chain_fly()
    # final forward at 4900 -> call fly intrinsic 0 -> full debit loss -200
    loss_q.update({
        ("16:00", 4895.0, True): (12.0, 13.0), ("16:00", 4895.0, False): (7.0, 8.0),
        ("16:00", 4900.0, True): (9.5, 10.5),  ("16:00", 4900.0, False): (9.5, 10.5),
        ("16:00", 4905.0, True): (7.0, 8.0),   ("16:00", 4905.0, False): (12.0, 13.0),
    })
    by_date = {
        _date(2025, 1, 2): {"quotes": win_q, "spot_close": 5020.0, "vix": 14.0},
        _date(2025, 1, 3): {"quotes": loss_q, "spot_close": 4900.0, "vix": 14.0},
    }
    res = bf.evaluate(by_date, cfg)
    assert res["n"] == 2
    assert res["total"] == 2600.0      # 2800 - 200
    assert res["wr"] == 50.0
    assert res["exits"] == {"expiry": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'make_wf_folds'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src_v2/backtest/butterfly_core.py

def make_wf_folds(dates: List[date], ratio: int = 5, folds: int = 4
                  ) -> List[Tuple[List[date], List[date]]]:
    """Ratio-based expanding walk-forward folds: train grows by one step per fold,
    val is the next single step (always out-of-sample)."""
    ds = sorted(dates)
    n = len(ds)
    step = n // (ratio + folds)
    out: List[Tuple[List[date], List[date]]] = []
    if step == 0:
        return out
    for i in range(folds):
        train_end = step * (ratio + i)
        val_end = train_end + step
        train, val = ds[:train_end], ds[train_end:val_end]
        if val:
            out.append((train, val))
    return out


def evaluate(by_date: Dict, cfg: ButterflyConfig) -> Dict:
    """Run evaluate_day over {date: {quotes, spot_close, vix?}}; aggregate metrics."""
    results: List[Dict] = []
    skips: Dict[str, int] = defaultdict(int)
    exits: Dict[str, int] = defaultdict(int)
    for d in sorted(by_date.keys()):
        info = by_date[d]
        r = evaluate_day(info["quotes"], cfg, info.get("spot_close"))
        if r.get("skip"):
            skips[r["skip"]] += 1
            continue
        r["date"] = d
        exits[r["kind"]] += 1
        results.append(r)

    pnls = [r["pnl"] for r in results]
    if not pnls:
        return {"n": 0, "skips": dict(skips), "results": []}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / max(1, len(pnls) - 1)
    std = math.sqrt(var)
    gross_w = sum(wins)
    gross_l = -sum(losses)
    return {
        "n": len(pnls),
        "wr": 100 * len(wins) / len(pnls),
        "total": sum(pnls),
        "mean": mean,
        "expectancy": mean,
        "sharpe_ann": math.sqrt(252) * mean / std if std > 0 else 0.0,
        "profit_factor": gross_w / gross_l if gross_l > 0 else float("inf"),
        "max_consec_L": _max_consec_losses(pnls),
        "max_dd": _max_drawdown(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "realized_r2r": (sum(wins) / len(wins)) / (-sum(losses) / len(losses))
                        if wins and losses else 0.0,
        "worst": min(pnls), "best": max(pnls),
        "exits": dict(exits), "skips": dict(skips),
        "results": results,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py -v`
Expected: PASS (25 tests total)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/butterfly_core.py tests/test_butterfly_core.py
git commit -m "feat(fly): aggregation metrics and walk-forward folds"
```

---

## Task 6: Wider-window backfill script

This is an I/O + Databento script — no unit test (it needs the live API key and spends money). It is adapted from `scripts/backfill_jl_cbbo.py` with a ±250pt window and a new output dir. Verify with the FREE `--dry-run`.

**Files:**
- Create: `scripts/backfill_butterfly_cbbo.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Wider-window 0DTE SPXW CBBO-1m backfill for the OTM butterfly backtest.

Identical to backfill_jl_cbbo.py but with a +/-250pt strike band (default) so
the wide Chaos-regime structures (body ~1sigma OTM + 50-60pt wings) stay priced
all day. Writes per-day pickles {(minute, strike, is_call): (bid, ask)} to
data/cbbo_cache_fly_10min/.

Usage:
  # 1. FREE cost estimate for the full range:
  .venv/bin/python scripts/backfill_butterfly_cbbo.py \\
      --start 2022-05-11 --end 2026-03-30 --dry-run
  # 2. Live fetch with a hard cap:
  .venv/bin/python scripts/backfill_butterfly_cbbo.py \\
      --start 2022-05-11 --end 2026-03-30 --max-cost-usd 20
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import databento as db

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.databento_cost import estimate_request_cost, CostLedger
from src_v2.backtest.meic_backfill import (
    meic_day_symbols, meic_window_utc, quotes_from_df,
)


def fly_center_and_symbols(
    session_date: date, bars: List[dict], half_width_pts: int = 250, step_pts: int = 5,
) -> Tuple[Optional[float], List[str]]:
    """center = 09:30 bar spot; symbols span [center-half, center+half], calls+puts."""
    b0930 = next((b for b in bars if b["ts_et"].time().hour == 9
                  and b["ts_et"].time().minute == 30), None)
    if b0930 is None:
        return None, []
    center = float(b0930["open"])
    symbols = meic_day_symbols(session_date, center, center,
                               buffer_pts=half_width_pts, step_pts=step_pts)
    return center, symbols


def main() -> int:
    p = argparse.ArgumentParser(description="Wider-window 0DTE SPXW butterfly backfill")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--out", default="data/cbbo_cache_fly_10min")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--half-width-pts", type=int, default=250)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--window-start", default="09:30")
    p.add_argument("--window-end", default="16:00")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep-ms", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=20.0)
    args = p.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    loader = BarLoader(args.bars, price_scale=args.price_scale, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"{len(days)} trading days {start}..{end}")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    mode = "dry_run" if args.dry_run else "live"
    n_calls = n_skipped = n_quotes = 0
    run_cost = 0.0

    for day in days:
        d = day.session_date
        out_path = out_dir / f"{d.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue

        center, symbols = fly_center_and_symbols(
            d, day.bars, args.half_width_pts, args.step_pts)
        if center is None or not symbols:
            n_skipped += 1
            continue
        start_iso, end_iso = meic_window_utc(d, args.window_start, args.window_end)

        if args.dry_run:
            try:
                cost = estimate_request_cost(client, symbols, start_iso, end_iso)
            except Exception as e:
                print(f"{d}: cost-estimate ERROR {type(e).__name__}: {str(e)[:120]}")
                continue
            run_cost += cost
            ledger.record(d, "dry_run", cost, len(symbols), 0)
            print(f"{d}: {len(symbols):>4} symbols  est ${cost:.4f}  cum ${run_cost:.2f}")
        else:
            if args.max_cost_usd and run_cost >= args.max_cost_usd:
                print(f"Cost cap ${args.max_cost_usd:.2f} reached (cum ${run_cost:.2f}); stopping.")
                break
            try:
                resp = client.timeseries.get_range(
                    dataset="OPRA.PILLAR", schema="cbbo-1m",
                    symbols=symbols, stype_in="raw_symbol",
                    start=start_iso, end=end_iso,
                )
                df = resp.to_df()
            except Exception as e:
                print(f"{d}: fetch ERROR {type(e).__name__}: {str(e)[:120]}")
                continue
            quotes = quotes_from_df(df)
            with open(out_path, "wb") as f:
                pickle.dump(quotes, f)
            try:
                cost = estimate_request_cost(client, symbols, start_iso, end_iso)
            except Exception:
                cost = 0.0
            run_cost += cost
            n_quotes += len(quotes)
            ledger.record(d, "live", cost, len(symbols), len(quotes))
            print(f"{d}: {len(symbols):>4} symbols -> {len(quotes):>6} quotes  ${cost:.4f}  cum ${run_cost:.2f}")
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)

        n_calls += 1
        if args.limit and n_calls >= args.limit:
            print(f"Limit {args.limit} reached.")
            break

    print(f"\n{mode}: {n_calls} days processed, {n_skipped} skipped, "
          f"{n_quotes} quotes, cumulative cost ${run_cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it parses and the dry-run estimates cost (FREE)**

Run:
```bash
.venv/bin/python scripts/backfill_butterfly_cbbo.py --start 2022-05-11 --end 2022-05-20 --dry-run
```
Expected: prints per-day symbol counts and a cumulative `est $` total (no pickles written). If `DATABENTO_API_KEY` is unset it prints the ERROR and returns 2 — that is acceptable for the parse check; rerun once the key is available.

- [ ] **Step 3: Commit**

```bash
git add scripts/backfill_butterfly_cbbo.py
git commit -m "feat(fly): wider +-250pt SPXW backfill for butterfly cache"
```

---

## Task 7: Analytical evaluator (scenarios + WF-CV sweep)

**Files:**
- Modify: `src_v2/backtest/butterfly_core.py` (add a small per-day config builder)
- Create: `scripts/analytical_butterfly_eval.py`
- Test: `tests/test_butterfly_core.py`

- [ ] **Step 1: Write the failing test for the per-day config builder**

```python
# append to tests/test_butterfly_core.py

def test_build_day_cfg_sets_side_and_width():
    base = bf.ButterflyConfig(exit_model="hold")
    up = [float(x) for x in range(1, 40)]
    cfg = bf.build_day_cfg(base, dir_mode="hull", width_mode="regime",
                           d=_date(2025, 1, 2), closes_before=up, vix=14.0, fixed_width=30.0)
    assert cfg.side == "call"        # rising Hull -> call
    assert cfg.width == 25.0          # VIX 14 -> Zombieland 25
    cfg2 = bf.build_day_cfg(base, dir_mode="put", width_mode="fixed",
                            d=_date(2025, 1, 2), closes_before=up, vix=40.0, fixed_width=35.0)
    assert cfg2.side == "put"
    assert cfg2.width == 35.0          # fixed mode ignores regime
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py::test_build_day_cfg_sets_side_and_width -v`
Expected: FAIL with `AttributeError: ... has no attribute 'build_day_cfg'`

- [ ] **Step 3: Add `build_day_cfg` to the core**

```python
# append to src_v2/backtest/butterfly_core.py
import dataclasses


def build_day_cfg(base: ButterflyConfig, dir_mode: str, width_mode: str,
                  d: date, closes_before: List[float], vix: Optional[float],
                  fixed_width: float) -> ButterflyConfig:
    """Clone `base` with the per-day side (direction mode) and width (width mode)."""
    side = resolve_side(dir_mode, d, closes_before)
    width = regime_width(vix) if width_mode == "regime" else fixed_width
    return dataclasses.replace(base, side=side, width=width)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_butterfly_core.py::test_build_day_cfg_sets_side_and_width -v`
Expected: PASS

- [ ] **Step 5: Write the evaluator script**

```python
# scripts/analytical_butterfly_eval.py
"""Analytical OTM 0DTE butterfly evaluator on a CBBO cache.

Phase 1 (single config, headline + with-costs):
    .venv/bin/python -m scripts.analytical_butterfly_eval \\
        --cache data/cbbo_cache_fly_10min --entry 09:30 \\
        --dir-mode hull --width-mode regime --exit hold \\
        --start 2022-05-11 --end 2026-03-30

Phase 2 (WF-CV sweep over the experimental axes; OOS-scored):
    .venv/bin/python -m scripts.analytical_butterfly_eval \\
        --cache data/cbbo_cache_fly_10min --sweep --start 2022-05-11 --end 2026-03-30
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest import butterfly_core as bf
from src_v2.backtest.vix_loader import load_vix_series, lookup_prior_close


def load_inputs(cache_dir: Path, bars_path: str, start: date, end: date,
                vix_csv: str) -> Tuple[Dict, List[Tuple[date, float]]]:
    """Return ({date: {quotes, spot_close, vix}}, ordered [(date, daily_close)])."""
    loader = BarLoader(bars_path, price_scale=10.0, symbol="SPX")
    vix_series = load_vix_series(vix_csv)
    by_date: Dict = {}
    closes: List[Tuple[date, float]] = []
    for day in loader.iter_days(start_date=start, end_date=end):
        d = day.session_date
        eod = next((b for b in reversed(day.bars)
                    if b["ts_et"].time() >= time(15, 55)), None)
        if eod is not None:
            closes.append((d, eod["close"]))
        pkl = Path(cache_dir) / f"{d.isoformat()}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            quotes = pickle.load(f)
        by_date[d] = {
            "quotes": quotes,
            "spot_close": eod["close"] if eod else None,
            "vix": lookup_prior_close(vix_series, date.fromordinal(d.toordinal() - 1)),
        }
    return by_date, closes


def closes_before(closes: List[Tuple[date, float]], d: date) -> List[float]:
    return [c for (cd, c) in closes if cd < d]


def run_config(by_date: Dict, closes: List[Tuple[date, float]], base: bf.ButterflyConfig,
               dir_mode: str, width_mode: str, fixed_width: float) -> Dict:
    """Build per-day cfgs and aggregate (one trade/day)."""
    results: List[Dict] = []
    skips: Dict[str, int] = {}
    for d in sorted(by_date.keys()):
        info = by_date[d]
        cfg = bf.build_day_cfg(base, dir_mode, width_mode, d,
                               closes_before(closes, d), info.get("vix"), fixed_width)
        r = bf.evaluate_day(info["quotes"], cfg, info.get("spot_close"))
        if r.get("skip"):
            skips[r["skip"]] = skips.get(r["skip"], 0) + 1
            continue
        r["date"] = d
        r["vix"] = info.get("vix")
        results.append(r)
    agg = _aggregate(results)
    agg["skips"] = skips
    return agg


def _aggregate(results: List[Dict]) -> Dict:
    pnls = [r["pnl"] for r in results]
    if not pnls:
        return {"n": 0, "results": []}
    import math
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / max(1, len(pnls) - 1)
    std = math.sqrt(var)
    gl = -sum(losses)
    return {
        "n": len(pnls), "wr": 100 * len(wins) / len(pnls), "total": sum(pnls),
        "mean": mean, "sharpe_ann": math.sqrt(252) * mean / std if std > 0 else 0.0,
        "profit_factor": (sum(wins) / gl) if gl > 0 else float("inf"),
        "max_dd": bf._max_drawdown(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "results": results,
    }


def _print(label: str, res: Dict) -> None:
    if res.get("n", 0) == 0:
        print(f"\n[{label}] NO TRADES  skips={res.get('skips')}")
        return
    print(f"\n[{label}]  n={res['n']}  WR={res['wr']:.1f}%  total=${res['total']:+,.0f}  "
          f"mean=${res['mean']:+.1f}  Sharpe={res['sharpe_ann']:.2f}  "
          f"PF={res['profit_factor']:.2f}  maxDD=${res['max_dd']:+,.0f}  skips={res.get('skips')}")


def _regime_breakdown(res: Dict) -> None:
    buckets = {"Zombieland(<17)": [], "Goldilocks(17-32)": [], "Chaos(>=32)": []}
    for r in res["results"]:
        v = r.get("vix")
        key = ("Chaos(>=32)" if v is not None and v >= 32
               else "Goldilocks(17-32)" if v is not None and v >= 17
               else "Zombieland(<17)")
        buckets[key].append(r["pnl"])
    print("  per-regime:")
    for k, pnls in buckets.items():
        if pnls:
            print(f"    {k:<18} n={len(pnls):>4}  total=${sum(pnls):+,.0f}  "
                  f"mean=${sum(pnls)/len(pnls):+.1f}")
        else:
            print(f"    {k:<18} n=0")


def _write_trades(out_dir: Path, res: Dict, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"trades_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "side", "kind", "exit_anchor", "vix", "em", "forward",
                    "k_low", "k_body", "k_high", "width", "debit", "exit_mark",
                    "r2r_theo", "pnl"])
        for r in res["results"]:
            kl, kb, kh = r["strikes"]
            w.writerow([r["date"], r["side"], r["kind"], r["exit_anchor"],
                        f"{(r.get('vix') or 0):.2f}", f"{r['em']:.2f}", f"{r['forward']:.2f}",
                        kl, kb, kh, r["width"], f"{r['debit']:.2f}", f"{r['exit_mark']:.2f}",
                        f"{r['r2r_theo']:.2f}", f"{r['pnl']:.2f}"])
    summary = {k: v for k, v in res.items() if k != "results"}
    with open(out_dir / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def sweep(by_date: Dict, closes: List[Tuple[date, float]], entry: str, anchors: List[str],
          out_dir: Path) -> None:
    """WF-CV sweep over direction x exit x width; select per fold on train Sharpe,
    score on OOS val. Coin-flip direction is always reported as a baseline."""
    dir_modes = ["hull", "call", "put", "coin", "anti"]
    exit_models = ["hold", "target", "trail"]
    width_specs = [("regime", 0.0), ("fixed", 25.0), ("fixed", 40.0), ("fixed", 55.0)]
    combos = list(itertools.product(dir_modes, exit_models, width_specs))

    dates = sorted(by_date.keys())
    folds = bf.make_wf_folds(dates, ratio=5, folds=4)
    print(f"Sweep: {len(combos)} combos x {len(folds)} folds")

    fold_rows: List[Dict] = []
    for fi, (train, val) in enumerate(folds):
        train_set, val_set = set(train), set(val)
        scored = []
        for (dm, xm, (wmode, fw)) in combos:
            base = bf.ButterflyConfig(entry_anchor=entry, anchors=anchors, exit_model=xm)
            tr = run_config({d: by_date[d] for d in by_date if d in train_set},
                            closes, base, dm, wmode, fw)
            scored.append(((dm, xm, wmode, fw), tr.get("sharpe_ann", 0.0), tr.get("n", 0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        (dm, xm, wmode, fw), tr_sharpe, tr_n = scored[0]
        base = bf.ButterflyConfig(entry_anchor=entry, anchors=anchors, exit_model=xm)
        vr = run_config({d: by_date[d] for d in by_date if d in val_set},
                        closes, base, dm, wmode, fw)
        row = {"fold": fi, "pick": f"{dm}/{xm}/{wmode}:{fw}",
               "train_sharpe": round(tr_sharpe, 2), "train_n": tr_n,
               "val_n": vr.get("n", 0), "val_total": round(vr.get("total", 0.0), 0),
               "val_mean": round(vr.get("mean", 0.0), 1),
               "val_sharpe": round(vr.get("sharpe_ann", 0.0), 2)}
        fold_rows.append(row)
        print(f"  fold {fi}: pick={row['pick']:<28} train_Sharpe={row['train_sharpe']:<6} "
              f"-> OOS n={row['val_n']} mean=${row['val_mean']:+.1f} Sharpe={row['val_sharpe']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wf_folds.json", "w") as f:
        json.dump(fold_rows, f, indent=2)
    oos_mean = sum(r["val_mean"] * r["val_n"] for r in fold_rows) / max(
        1, sum(r["val_n"] for r in fold_rows))
    print(f"\nPooled OOS mean/trade = ${oos_mean:+.1f}  (>0 with positive skew = primary success test)")


def main() -> int:
    p = argparse.ArgumentParser(description="Analytical OTM butterfly evaluator")
    p.add_argument("--cache", default="data/cbbo_cache_fly_10min")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--vix", default="data/vix1d_daily.csv")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-03-30")
    p.add_argument("--entry", default="09:30")
    p.add_argument("--anchors", default="")
    p.add_argument("--dir-mode", default="hull", choices=["hull", "call", "put", "coin", "anti"])
    p.add_argument("--width-mode", default="regime", choices=["regime", "fixed"])
    p.add_argument("--fixed-width", type=float, default=40.0)
    p.add_argument("--exit", default="hold", choices=["hold", "target", "trail"])
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--commission", type=float, default=1.0)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--out", default="results/butterfly_otm")
    args = p.parse_args()

    anchors = ([a.strip() for a in args.anchors.split(",")] if args.anchors
               else bf.make_anchors(args.entry, "16:00", 10))
    by_date, closes = load_inputs(Path(args.cache), args.bars,
                                  date.fromisoformat(args.start), date.fromisoformat(args.end),
                                  args.vix)
    print(f"Days with cache: {len(by_date)}  entry={args.entry}  anchors={len(anchors)}")

    if args.sweep:
        sweep(by_date, closes, args.entry, anchors, Path(args.out))
        return 0

    base = bf.ButterflyConfig(entry_anchor=args.entry, anchors=anchors, exit_model=args.exit)
    headline = run_config(by_date, closes, base, args.dir_mode, args.width_mode, args.fixed_width)
    _print(f"HEADLINE {args.dir_mode}/{args.exit}/{args.width_mode} (mid, zero-cost)", headline)
    if headline.get("n"):
        _regime_breakdown(headline)

    base_c = bf.ButterflyConfig(entry_anchor=args.entry, anchors=anchors, exit_model=args.exit,
                                slippage_per_leg=args.slippage, commission_per_leg=args.commission)
    costs = run_config(by_date, closes, base_c, args.dir_mode, args.width_mode, args.fixed_width)
    _print(f"WITH COSTS (slip=${args.slippage}/leg, comm=${args.commission}/leg)", costs)

    _write_trades(Path(args.out), headline, "headline")
    print(f"\nPer-trade rows -> {args.out}/trades_headline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the full test suite + verify the evaluator parses**

Run:
```bash
.venv/bin/python -m pytest tests/test_butterfly_core.py -v
.venv/bin/python -m scripts.analytical_butterfly_eval --help
```
Expected: all core tests PASS; `--help` prints the usage without error.

- [ ] **Step 7: Commit**

```bash
git add src_v2/backtest/butterfly_core.py scripts/analytical_butterfly_eval.py tests/test_butterfly_core.py
git commit -m "feat(fly): analytical evaluator with scenarios and WF-CV sweep"
```

---

## Task 8: Config reference + findings stub

**Files:**
- Create: `config/config_butterfly_otm_spx.yaml`
- Create: `docs/spx_0dte_butterfly_otm_findings.md`

- [ ] **Step 1: Write the config reference**

```yaml
# config/config_butterfly_otm_spx.yaml
# Reference defaults + sweep grid for the OTM 0DTE butterfly backtest.
# The analytical evaluator takes CLI overrides; this file documents intent.
bot:
  name: "spx-0dte-butterfly-otm"
  symbol: "SPX"

strategy:
  entry_time: "09:30"
  body_em_mult: 1.0           # body 1-sigma (ATM-straddle EM) OTM
  direction: "hull"           # 21-period Hull MA trend
  eod_force_close: "15:50"
  gates:
    max_debit_pct: 0.10
    min_r2r: 9.0

regime_width:                 # VIX prior-close -> representative wing width
  zombieland: {vix_lt: 17, width: 25}
  goldilocks: {vix_lt: 32, width: 40}
  chaos:      {vix_gte: 32, width: 55}

sweep:
  dir_modes: [hull, call, put, coin, anti]
  exit_models: [hold, target, trail]
  entry_times: ["09:30", "10:00", "10:30"]
  widths: {mode: [regime, fixed], fixed: [25, 40, 55]}
  body_em_mult: [1.0, 0.75]

costs:
  slippage_per_leg: 0.05      # sweep 0.025 / 0.05 / 0.10
  commission_per_leg: 1.0

backtest:
  cache_dir: "data/cbbo_cache_fly_10min"
  bars: "data/spy_1m.csv"
  vix: "data/vix1d_daily.csv"
  start: "2022-05-11"
  end: "2026-03-30"
  wf_cv: {ratio: 5, folds: 4}
```

- [ ] **Step 2: Write the findings stub**

```markdown
# SPX 0DTE OTM Butterfly — Backtest Findings

> Status: PENDING DATA RUN. This document is filled after the wider-window
> backfill (Task 6) and the Phase 1/2 runs (Task 9).

## Method
- Spec: `docs/superpowers/specs/2026-06-03-otm-butterfly-backtest-design.md`
- Cache: `data/cbbo_cache_fly_10min` (±250pt, ~940 days 2022-05→2026-03)
- Core: `src_v2/backtest/butterfly_core.py`; eval: `scripts/analytical_butterfly_eval.py`

## Primary question
Does the OTM 0DTE butterfly (body 1σ, ≥9:1) have positive **net-of-cost OOS**
expectancy with positive skew, regardless of side?

## Results
_TBD — headline vs with-costs table, per-regime breakdown (with Chaos N),
direction-edge vs coin-flip, exit-model comparison, WF-CV OOS pooled mean._

## Verdict
_TBD_
```

- [ ] **Step 3: Commit**

```bash
git add config/config_butterfly_otm_spx.yaml docs/spx_0dte_butterfly_otm_findings.md
git commit -m "docs(fly): config reference and findings stub"
```

---

## Task 9: Data run + write up findings (manual / requires Databento)

This task spends Databento budget and produces the actual answer. It is gated on the user approving the backfill spend. Run the phases in order.

- [ ] **Step 1: Phase 0 smoke — validate logic on the EXISTING JL cache (free)**

Run (reuses `data/cbbo_cache_jl_10min`; ±100pt covers Zombieland/Goldilocks):
```bash
.venv/bin/python -m scripts.analytical_butterfly_eval \
    --cache data/cbbo_cache_jl_10min --entry 09:30 \
    --dir-mode hull --width-mode regime --exit hold \
    --start 2022-05-11 --end 2026-03-30
```
Expected: a HEADLINE line with `n` in the hundreds and a per-regime breakdown. Many days may skip as `gate`/`no_debit` — that is informative (the 10% gate is strict). If `n` is near zero, stop and investigate the gate/strike logic before spending on the wider backfill.

- [ ] **Step 2: FREE cost estimate for the wider backfill**

```bash
.venv/bin/python scripts/backfill_butterfly_cbbo.py \
    --start 2022-05-11 --end 2026-03-30 --dry-run
```
Expected: cumulative `est $` total (≈$8–12). Confirm it is under the $20 cap before the live run.

- [ ] **Step 3: Live wider backfill (spends budget — confirm with user first)**

```bash
.venv/bin/python scripts/backfill_butterfly_cbbo.py \
    --start 2022-05-11 --end 2026-03-30 --max-cost-usd 20
```
Expected: per-day `-> N quotes` lines; pickles written to `data/cbbo_cache_fly_10min/`.

- [ ] **Step 4: Phase 1 — full-sample descriptive (headline + costs), all directions**

```bash
for dm in hull coin call put; do
  .venv/bin/python -m scripts.analytical_butterfly_eval \
    --cache data/cbbo_cache_fly_10min --dir-mode $dm --width-mode regime --exit hold \
    --start 2022-05-11 --end 2026-03-30
done
```
Expected: compare `hull` vs `coin` mean/trade and Sharpe — this is the 50/50 direction test. Note net-of-cost mean.

- [ ] **Step 5: Phase 2 — WF-CV sweep (OOS)**

```bash
.venv/bin/python -m scripts.analytical_butterfly_eval \
    --cache data/cbbo_cache_fly_10min --sweep \
    --start 2022-05-11 --end 2026-03-30
```
Expected: per-fold pick + OOS scores and a pooled OOS mean/trade line. Results in `results/butterfly_otm/wf_folds.json`.

- [ ] **Step 6: Fill in `docs/spx_0dte_butterfly_otm_findings.md`**

Record: headline-vs-net table, per-regime breakdown (Chaos N explicit), hull-vs-coin direction edge, best exit model, WF-CV pooled OOS mean, and the verdict against the primary success test (net-of-cost OOS expectancy > 0 with positive skew). Sanity-check realized R2R and win rate against the vendor anchors (~1.8:1, 45–65%).

- [ ] **Step 7: Commit findings**

```bash
git add docs/spx_0dte_butterfly_otm_findings.md results/butterfly_otm/
git commit -m "docs(fly): butterfly backtest findings and WF-CV results"
```

---

## Self-review notes

- **Spec coverage:** §4 locked rules → Tasks 1–4 (strikes, body=1×EM, gates, EOD, costs) + Task 7 (Hull direction, regime width per-day wiring). §5 axes → Task 7 sweep (direction incl. coin-flip, entry time, exit models, gates on/off via `gates_on`, width regime/fixed, body mult). §6 settlement → Task 4 (`fly_intrinsic` + `settlement_forward`). §7 data/wider backfill → Task 6. §8 architecture → file structure. §9 cost model → Task 4. §10 evaluation phases + WF-CV → Tasks 5, 7, 9. §12 outputs → Task 7 `_write_trades` + Task 9.
- **Body 0.75× EM and entry-time {10:00,10:30} and gates-off** are reachable via existing fields (`body_em_mult`, `--entry`, `gates_on`) — add to the sweep grid in Task 7 if desired; documented in the config (Task 8). Not separate tasks.
- **Type consistency:** `ButterflyConfig` fields (`side`, `body_em_mult`, `width`, `exit_model`, `target_mult`, `trail_arm_pct`, `gates_on`, `slippage_per_leg`, `commission_per_leg`) are used identically across Tasks 1–7. `Strikes = (k_low, k_body, k_high)` ordering is consistent in `select_fly_strikes`, `fly_intrinsic`, `fly_value`, `_write_trades`.
- **Known approximation:** within-band width sweep uses representative widths (25/40/55) rather than the full {20,25,30}/{30,40,50}/{50,60} grid; expand `regime_width`/sweep `width_specs` if the coarse grid shows promise.
```
