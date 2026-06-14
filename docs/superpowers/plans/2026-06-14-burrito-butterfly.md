# Burrito Butterfly Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a faithful, Sharpe-optimizing backtest of the discretionary Burrito Butterfly strategy (directional OTM butterfly + linked vertical debit spread on SPX, with a full path-dependent lock/defense/clawback lifecycle), validated by walk-forward CV.

**Architecture:** A single pure-logic engine (`burrito_core.py`) reuses the Iron Duck pricing/loader/Smile primitives. A clock abstraction (`burrito_clock.py`) feeds the lifecycle state machine EOD marks (DTE≥1) or 10-min intraday marks (DTE=0) through one interface. A runner and a WF-CV sweeper drive it; results land in a findings doc.

**Tech Stack:** Python 3.13, scipy (Black-76 / IV solve), pytest. Data: `data/iron_duck_eod/SPXW/*.pkl` (EOD NBBO 2020–2026) + `data/cbbo_cache_fly_10min/` (0DTE intraday). Run all Python via the project venv: `~/Dev/0dte_bot/.venv/bin/python` and `~/Dev/0dte_bot/.venv/bin/pytest`.

**Spec:** `docs/superpowers/specs/2026-06-14-burrito-butterfly-design.md`

---

## Reused primitives (do not reimplement)

From `src_v2/backtest/iron_duck.py` and `src_v2/backtest/hedgehog.py`:
- `DuckStore(cache_dir)` (== `HedgeHogStore`): `.chain(quote_date:int, expiry:int) -> Chain|None`,
  `.expiries_in_dte(date, dte_min, dte_max) -> List[(expiry, dte)]`, `.spot(date) -> float|None`,
  `.quote_dates() -> List[int]`. `Chain = Dict[(strike:float, 'C'|'P'), (bid,ask,flag)]`.
- `Smile(chain, F, T)`: `.ok()`, `.vol(k)`, `.call_delta(k)`, `.put_delta_abs(k)`.
- `_fill(q, side, k)` (mid ± k·half-spread), `_mid(q)`, `black_call`, `black_put`, `black_call_iv`,
  `black_put_iv`, `parity_forward(chain)`, `_to_date(int) -> date`.
- Dates are ints `YYYYMMDD`. Prices are SPX points (per share); P&L uses ×100 multiplier.

Import style (mirror `iron_duck.py`):
```python
from src_v2.backtest.iron_duck import (
    DuckStore, Smile, _fill, _mid, black_call, black_put, black_call_iv,
    black_put_iv, parity_forward, _to_date,
)
```

---

## File structure

| Path | Responsibility |
|---|---|
| `src_v2/backtest/burrito_core.py` | `BurritoConfig`, `build_legs`, `Leg`, `BurritoTrade`, `payoff`, `position_value`, direction signals, lifecycle `BurritoEngine`, `summarize`. Pure logic. |
| `src_v2/backtest/burrito_clock.py` | `Mark` + `marks_for_trade(store, entry_date, expiry, cfg)` yielding EOD or intraday marks. |
| `scripts/backtest_burrito.py` | CLI single-config runner → `results/burrito/trades.csv` + `metrics.json`. |
| `scripts/wf_cv_burrito.py` | Walk-forward CV grid sweep + slippage stress → `results/burrito/wfcv_*.json`. |
| `tests/test_burrito_core.py` | Unit tests for geometry, pricing, signals, every state transition, settlement, summarize. |
| `docs/burrito_butterfly_findings.md` | Final verdict + caveats (written in last task). |

---

## Task 1: Leg model, BurritoConfig, and geometry builder

**Files:**
- Create: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_burrito_core.py
from src_v2.backtest.burrito_core import BurritoConfig, build_legs, Leg


def test_build_legs_bullish_call_outer_link():
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    # forward 5000, body pushed 1.0*EM (EM=20) OTM -> body strike 5020
    legs = build_legs(forward=5000.0, em=20.0, cfg=cfg, side="call")
    # fly: +1 5005? body offset uses round_to_step(forward + body_offset*em)
    by = {(l.strike, l.right): l.qty for l in legs}
    body = 5020.0
    assert by[(body, "C")] == -2                      # short body x2
    assert by[(body - 15.0, "C")] == 1                # inner long (ATM-side wing)
    # outer wing long (body+15) shared with spread long -> net +1 (fly) +1 (spread) = +2
    assert by[(body + 15.0, "C")] == 2
    assert by[(body + 15.0 + 5.0, "C")] == -1         # spread short
    assert all(l.right == "C" for l in legs)


def test_build_legs_bearish_put_mirror():
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    legs = build_legs(forward=5000.0, em=20.0, cfg=cfg, side="put")
    by = {(l.strike, l.right): l.qty for l in legs}
    body = 4980.0                                      # pushed OTM downward
    assert by[(body, "P")] == -2
    assert by[(body + 15.0, "P")] == 1                # inner long (ATM-side)
    assert by[(body - 15.0, "P")] == 2                # outer long shared with spread
    assert by[(body - 15.0 - 5.0, "P")] == -1         # spread short
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src_v2.backtest.burrito_core'`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/burrito_core.py
"""Burrito Butterfly backtest — directional OTM butterfly + linked vertical
debit spread on cash-settled SPX, with a path-dependent lock/defense/clawback
lifecycle. Reuses Iron Duck pricing/loader/Smile primitives. Pure logic; data
I/O lives in burrito_clock.py and scripts/. See spec
docs/superpowers/specs/2026-06-14-burrito-butterfly-design.md.

All prices are SPX points (per share); P&L uses the 100 multiplier.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

Side = Literal["call", "put"]
Right = Literal["C", "P"]


def round_to_step(x: float, step: float) -> float:
    return round(x / step) * step


@dataclass(frozen=True)
class Leg:
    qty: int        # +long / -short, in lots
    strike: float
    right: Right


@dataclass
class BurritoConfig:
    # entry
    entry_dte: int = 3
    entry_anchor: str = "15:55"        # EOD for DTE>=1; intraday bar for DTE=0
    direction_signal: str = "hull"     # hull | momentum | both | markov
    body_offset: float = 1.0           # body pushed body_offset*EM OTM from forward
    # geometry
    fly_wing_width: float = 15.0
    spread_width: float = 5.0
    spread_link: str = "outer"         # "outer" (default read) | "inner"
    step: float = 5.0
    multiplier: int = 100
    # lifecycle (fractions of net debit at risk)
    lock_trigger_pct: float = 0.07     # +X% -> lock opposing spread (FLOATING)
    stop_pct: float = 0.07             # Uncle Point: -X% -> close
    pt_pct: float = 0.10               # profit target on floated structure
    defense_policy: Tuple[str, ...] = ()   # subset of wing_lock|otm_freeze|choo_choo|clawback
    defense_trigger_pct: float = 0.05  # adverse move fraction that arms defenses
    clawback_last_n: int = 1           # final N marks count as the clawback window
    # entry gates
    gates_on: bool = True
    max_debit_pct: float = 0.40        # debit <= max_debit_pct * fly_wing_width
    # costs
    slippage_k: float = 1.0            # _fill half-spread multiplier
    commission_per_leg: float = 0.0


def _merge(legs: List[Leg]) -> List[Leg]:
    agg: Dict[Tuple[float, Right], int] = defaultdict(int)
    for l in legs:
        agg[(l.strike, l.right)] += l.qty
    return [Leg(q, k, r) for (k, r), q in agg.items() if q != 0]


def build_legs(forward: float, em: float, cfg: BurritoConfig, side: Side) -> List[Leg]:
    """Fly (1-2-1, fly_wing_width wings, body pushed OTM) + linked vertical
    debit spread (spread_width). 'outer' link: spread long reuses the fly's
    far/outer wing. Returns net (merged) legs."""
    right: Right = "C" if side == "call" else "P"
    sign = 1.0 if side == "call" else -1.0
    W, S = cfg.fly_wing_width, cfg.spread_width
    body = round_to_step(forward + sign * cfg.body_offset * em, cfg.step)
    inner = body - sign * W            # ATM-side wing
    outer = body + sign * W            # far wing
    fly = [Leg(1, inner, right), Leg(-2, body, right), Leg(1, outer, right)]
    if cfg.spread_link == "outer":
        spread = [Leg(1, outer, right), Leg(-1, outer + sign * S, right)]
    else:  # inner link: spread long reuses the ATM-side wing
        spread = [Leg(1, inner, right), Leg(-1, inner + sign * S, right)]
    return _merge(fly + spread)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/burrito_core.py tests/test_burrito_core.py
git commit -m "feat(burrito): leg model, config, geometry builder"
```

---

## Task 2: Expiration payoff

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_payoff_call_fly_peaks_at_body():
    from src_v2.backtest.burrito_core import build_legs, payoff, BurritoConfig
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    legs = build_legs(5000.0, 20.0, cfg, "call")  # body 5020, inner 5005, outer 5035, spr short 5040
    # payoff is intrinsic value of legs at settlement s (debit handled separately)
    far_below = payoff(legs, 4000.0)
    at_body = payoff(legs, 5020.0)
    at_outer = payoff(legs, 5035.0)
    assert abs(far_below) < 1e-9                       # all OTM -> 0 intrinsic
    assert at_body == 15.0                             # fly peak = wing width
    assert at_outer < at_body                          # past the peak the fly decays
    # spread adds value above its long: at 5040 spread intrinsic = +5
    assert payoff(legs, 5040.0) > payoff(legs, 5035.0) - 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_payoff_call_fly_peaks_at_body -v`
Expected: FAIL — `ImportError: cannot import name 'payoff'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
def _intrinsic(leg: Leg, s: float) -> float:
    if leg.right == "C":
        return max(s - leg.strike, 0.0)
    return max(leg.strike - s, 0.0)


def payoff(legs: List[Leg], s: float) -> float:
    """Total per-share intrinsic value of all legs at settlement price s
    (excludes the entry debit; net trade P&L = payoff - debit)."""
    return sum(leg.qty * _intrinsic(leg, s) for leg in legs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_payoff_call_fly_peaks_at_body -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): expiration payoff of merged legs"
```

---

## Task 3: Mark-to-market position value (chain + model fallback)

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_position_value_from_chain_fill():
    from src_v2.backtest.burrito_core import Leg, position_value
    # chain: (strike,right)->(bid,ask,flag). One long call leg.
    chain = {(5000.0, "C"): (10.0, 12.0, 0)}
    legs = [Leg(1, 5000.0, "C")]
    # closing a +1 long = SELL at mid - k*half_spread; mid=11, hs=1
    val = position_value(legs, chain, smile=None, F=None, slippage_k=1.0)
    assert abs(val - 10.0) < 1e-9     # value of holding = +10 (sell proceeds)


def test_position_value_model_fallback_when_missing():
    from src_v2.backtest.burrito_core import Leg, position_value
    from src_v2.backtest.iron_duck import Smile
    # empty chain -> must fall back to model value via smile
    chain = {(4800.0, "P"): (5.0, 6.0, 0), (5200.0, "C"): (5.0, 6.0, 0)}
    sm = Smile({(4800.0, "P"): (5.0, 6.0, 0), (5200.0, "C"): (5.0, 6.0, 0),
                (4900.0, "P"): (8.0, 9.0, 0), (5100.0, "C"): (8.0, 9.0, 0)},
               F=5000.0, T=5/365)
    legs = [Leg(1, 5050.0, "C")]      # 5050 not in chain
    val = position_value(legs, chain, smile=sm, F=5000.0, slippage_k=0.0)
    assert val is not None and val > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k position_value -v`
Expected: FAIL — `ImportError: cannot import name 'position_value'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
from src_v2.backtest.iron_duck import _fill, black_call, black_put


def _leg_model_value(leg: Leg, smile, F: float) -> float:
    v = smile.vol(leg.strike) or 0.0
    px = black_call(F, leg.strike, smile.T, v) if leg.right == "C" else black_put(F, leg.strike, smile.T, v)
    return leg.qty * px


def position_value(legs, chain, smile=None, F=None, slippage_k: float = 1.0):
    """Current per-share value of HOLDING the position (= proceeds to close it).
    Closing a long leg sells (proceeds +), closing a short buys (cost -). Uses
    chain fills where available; falls back to the Black model via `smile`.
    Returns None if a leg can be priced by neither source."""
    total = 0.0
    for leg in legs:
        q = chain.get((leg.strike, leg.right)) if chain else None
        if q is not None and q[1] > 0:
            # to value holding: a +long is closed by SELL; a -short by BUY
            side = "sell" if leg.qty > 0 else "buy"
            px = _fill(q, side, slippage_k)
            total += abs(leg.qty) * (px if leg.qty > 0 else -px)
        elif smile is not None and F is not None and smile.vol(leg.strike):
            total += _leg_model_value(leg, smile, F)
        else:
            return None
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k position_value -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): MTM position value with model fallback"
```

---

## Task 4: Entry cost (debit) and entry gates

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_entry_debit_and_gate():
    from src_v2.backtest.burrito_core import Leg, entry_debit, passes_gates, BurritoConfig
    # two-leg toy: buy 5000C @ ask side, sell 5005C
    chain = {(5000.0, "C"): (10.0, 11.0, 0), (5005.0, "C"): (7.0, 8.0, 0)}
    legs = [Leg(1, 5000.0, "C"), Leg(-1, 5005.0, "C")]
    # debit: buy 1 @ mid+hs = 10.5+? mid=10.5 hs=0.5 -> 11.0 ; sell 1 @ mid-hs = 7.5-0.5=7.0
    debit = entry_debit(legs, chain, slippage_k=1.0, commission_per_leg=0.0)
    assert abs(debit - (11.0 - 7.0)) < 1e-9
    cfg = BurritoConfig(fly_wing_width=15.0, max_debit_pct=0.40, gates_on=True)
    assert passes_gates(debit=4.0, cfg=cfg) is True       # 4 <= 0.40*15=6
    assert passes_gates(debit=7.0, cfg=cfg) is False      # 7 > 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k "entry_debit or gate" -v`
Expected: FAIL — `ImportError: cannot import name 'entry_debit'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
def entry_debit(legs, chain, slippage_k: float = 1.0, commission_per_leg: float = 0.0):
    """Net debit to OPEN (buy longs at +half-spread, sell shorts at -half-spread).
    Returns None if any leg is unpriced. Positive = pay; negative = credit."""
    total = 0.0
    for leg in legs:
        q = chain.get((leg.strike, leg.right))
        if q is None or q[1] <= 0:
            return None
        side = "buy" if leg.qty > 0 else "sell"
        px = _fill(q, side, slippage_k)
        total += abs(leg.qty) * (px if leg.qty > 0 else -px)
        total += abs(leg.qty) * commission_per_leg
    return total


def passes_gates(debit: float, cfg: BurritoConfig) -> bool:
    if not cfg.gates_on:
        return True
    if debit is None or debit <= 0:
        return False
    return debit <= cfg.max_debit_pct * cfg.fly_wing_width
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k "entry_debit or gate" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): entry debit and entry gates"
```

---

## Task 5: Direction signals

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_direction_signal_hull_and_momentum():
    from src_v2.backtest.burrito_core import direction_from_series
    rising = [100 + i for i in range(40)]
    falling = [100 - i for i in range(40)]
    assert direction_from_series(rising, "hull") == "call"
    assert direction_from_series(falling, "hull") == "put"
    assert direction_from_series(rising, "momentum") == "call"
    assert direction_from_series(falling, "momentum") == "put"
    assert direction_from_series(rising, "both") == "both"
    assert direction_from_series([100.0] * 3, "hull") == "skip"   # insufficient history
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k direction -v`
Expected: FAIL — `ImportError: cannot import name 'direction_from_series'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
import math


def _wma(vals: List[float]) -> float:
    n = len(vals)
    w = list(range(1, n + 1))
    return sum(v * wi for v, wi in zip(vals, w)) / sum(w)


def _hull_ma(prices: List[float], n: int) -> Optional[float]:
    half = max(1, n // 2)
    sq = max(1, int(round(math.sqrt(n))))
    if len(prices) < n + sq:
        return None
    raw = []
    for end in range(len(prices) - sq + 1, len(prices) + 1):
        win_full = prices[end - n:end]
        win_half = prices[end - half:end]
        raw.append(2 * _wma(win_half) - _wma(win_full))
    return _wma(raw)


def direction_from_series(prices: List[float], signal: str, n: int = 21, mom: int = 10) -> str:
    """Map a price series (oldest..newest) to 'call'|'put'|'both'|'skip'.
    'markov' is delegated to direction_markov (simple smoothed-return regime)."""
    if signal == "both":
        return "both"
    if signal == "momentum":
        if len(prices) <= mom:
            return "skip"
        return "call" if prices[-1] >= prices[-1 - mom] else "put"
    if signal == "markov":
        return direction_markov(prices)
    # default: hull
    h = _hull_ma(prices, n)
    if h is None:
        return "skip"
    return "call" if prices[-1] >= h else "put"


def direction_markov(prices: List[float], lookback: int = 20) -> str:
    """Lightweight regime proxy: sign of the mean of recent daily returns.
    Pluggable — replace with the full markov-2-hedge-fund-method skill later."""
    if len(prices) <= lookback:
        return "skip"
    rets = [prices[i] / prices[i - 1] - 1.0 for i in range(len(prices) - lookback, len(prices))]
    m = sum(rets) / len(rets)
    if abs(m) < 1e-5:
        return "skip"
    return "call" if m > 0 else "put"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k direction -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): direction signal menu (hull/momentum/both/markov)"
```

---

## Task 6: Clock — marks for a trade (EOD + intraday)

**Files:**
- Create: `src_v2/backtest/burrito_clock.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_eod_marks_walk_to_expiry():
    from src_v2.backtest.burrito_clock import Mark, eod_marks
    # fake store: chains keyed by (date, expiry)
    class FakeStore:
        def __init__(self, chains, dates):
            self._c = chains; self._d = dates
        def quote_dates(self): return self._d
        def chain(self, d, e): return self._c.get((d, e))
    chain = {(5000.0, "C"): (1.0, 2.0, 0), (5000.0, "P"): (1.0, 2.0, 0),
             (4900.0, "P"): (0.5, 1.0, 0), (5100.0, "C"): (0.5, 1.0, 0)}
    store = FakeStore({(20240101, 20240105): chain, (20240102, 20240105): chain},
                      [20240101, 20240102, 20240105])
    marks = list(eod_marks(store, entry_date=20240101, expiry=20240105))
    assert [m.date for m in marks] == [20240101, 20240102]   # marks up to (not incl) expiry
    assert all(isinstance(m, Mark) for m in marks)
    assert marks[0].forward is not None and marks[0].T > marks[1].T
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_eod_marks_walk_to_expiry -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src_v2.backtest.burrito_clock'`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/burrito_clock.py
"""Yields ordered marks (price snapshots) over a Burrito trade's life. EOD marks
come from the DuckStore (DTE>=1); intraday 0DTE marks from the 10-min fly cache.
A Mark carries everything the lifecycle needs to reprice + decide."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from src_v2.backtest.iron_duck import Smile, parity_forward, _to_date


@dataclass
class Mark:
    date: int
    intraday_min: Optional[str]   # "HH:MM" for 0DTE; None for EOD
    chain: dict
    forward: Optional[float]
    T: float                      # years to expiry at this mark
    is_last: bool                 # final mark before settlement


def _years_to(date: int, expiry: int) -> float:
    days = (_to_date(expiry) - _to_date(date)).days
    return max(days / 365.0, 1e-6)


def eod_marks(store, entry_date: int, expiry: int) -> Iterator[Mark]:
    dates = [d for d in store.quote_dates() if entry_date <= d < expiry]
    for i, d in enumerate(dates):
        ch = store.chain(d, expiry)
        if ch is None:
            continue
        F = parity_forward(ch)
        yield Mark(date=d, intraday_min=None, chain=ch, forward=F,
                   T=_years_to(d, expiry), is_last=(i == len(dates) - 1))
```

(Intraday 0DTE marks are added in Task 6b once the EOD path is proven end-to-end.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_eod_marks_walk_to_expiry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): EOD clock marks for a trade"
```

---

## Task 7: Lifecycle — DIRECTIONAL transitions (lock, Uncle stop, settle)

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

This task adds `BurritoTrade` and the core `step_lifecycle` that, given the current
state + a mark, returns the next state and any realized close.

- [ ] **Step 1: Write the failing test**

```python
def test_lifecycle_lock_then_target_and_uncle_stop():
    from src_v2.backtest.burrito_core import BurritoTrade, lifecycle_decision, BurritoConfig
    cfg = BurritoConfig(lock_trigger_pct=0.07, stop_pct=0.07, pt_pct=0.10)
    tr = BurritoTrade(side="call", legs=[], debit=10.0, entry_date=20240101, expiry=20240105)
    # DIRECTIONAL: pnl +8% of debit -> lock
    d = lifecycle_decision(tr, state="DIRECTIONAL", mtm_value=10.8, cfg=cfg, is_last=False)
    assert d.action == "lock" and d.next_state == "FLOATING"
    # DIRECTIONAL: pnl -8% -> uncle stop close
    d = lifecycle_decision(tr, state="DIRECTIONAL", mtm_value=9.2, cfg=cfg, is_last=False)
    assert d.action == "close" and d.reason == "uncle_stop"
    # FLOATING: pnl +11% -> target close
    d = lifecycle_decision(tr, state="FLOATING", mtm_value=11.1, cfg=cfg, is_last=False)
    assert d.action == "close" and d.reason == "target"
    # any state, is_last -> settle
    d = lifecycle_decision(tr, state="DIRECTIONAL", mtm_value=10.0, cfg=cfg, is_last=True)
    assert d.action == "settle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k lifecycle_lock -v`
Expected: FAIL — `ImportError: cannot import name 'BurritoTrade'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
@dataclass
class BurritoTrade:
    side: Side
    legs: List[Leg]
    debit: float
    entry_date: int
    expiry: int
    state: str = "DIRECTIONAL"
    extra_cost: float = 0.0            # accumulated lock/defense/clawback debits
    exit_date: Optional[int] = None
    exit_reason: Optional[str] = None
    pnl_pts: Optional[float] = None
    pnl_usd: Optional[float] = None
    dte: int = 0
    days_held: int = 0


@dataclass
class Decision:
    action: str            # "hold" | "lock" | "defend" | "clawback" | "close" | "settle"
    next_state: str = ""
    reason: str = ""


def lifecycle_decision(tr, state: str, mtm_value: float, cfg: BurritoConfig,
                       is_last: bool, in_valley: bool = False) -> Decision:
    """Pure decision function for one mark. mtm_value = current per-share value of
    holding (proceeds to close). Capital basis = tr.debit + tr.extra_cost."""
    basis = tr.debit + tr.extra_cost
    pnl_frac = (mtm_value - basis) / basis if basis > 0 else 0.0
    if is_last:
        return Decision("settle", next_state="CLOSED", reason="expiry")
    if state == "DIRECTIONAL":
        if pnl_frac <= -cfg.stop_pct and "wing_lock" not in cfg.defense_policy:
            return Decision("close", next_state="CLOSED", reason="uncle_stop")
        if pnl_frac <= -cfg.defense_trigger_pct and cfg.defense_policy:
            return Decision("defend", next_state="DEFENDED", reason="adverse")
        if pnl_frac >= cfg.lock_trigger_pct:
            return Decision("lock", next_state="FLOATING", reason="lock")
        return Decision("hold", next_state="DIRECTIONAL")
    # FLOATING / DEFENDED
    if pnl_frac >= cfg.pt_pct:
        return Decision("close", next_state="CLOSED", reason="target")
    if in_valley and "clawback" in cfg.defense_policy:
        return Decision("clawback", next_state=state, reason="valley")
    if pnl_frac <= -cfg.stop_pct:
        return Decision("close", next_state="CLOSED", reason="uncle_stop")
    return Decision("hold", next_state=state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k lifecycle_lock -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): lifecycle decision (lock/uncle/target/settle)"
```

---

## Task 8: Adjustment legs — lock, defenses, clawback

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

Each adjustment returns the NEW legs to add and is priced via `entry_debit`.

- [ ] **Step 1: Write the failing test**

```python
def test_adjustment_legs_lock_and_choo_choo():
    from src_v2.backtest.burrito_core import build_legs, lock_legs, choo_choo_legs, BurritoConfig
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    # bullish entry -> lock buys the opposing (put) debit spread
    lock = lock_legs(side="call", forward=5000.0, em=20.0, cfg=cfg)
    assert all(l.right == "P" for l in lock) and any(l.qty > 0 for l in lock)
    # choo-choo adds a second fly into the adverse (down) direction for a call entry
    extra = choo_choo_legs(side="call", forward=4900.0, em=20.0, cfg=cfg)
    assert any(l.qty == -2 for l in extra)   # a second body short pair
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k adjustment_legs -v`
Expected: FAIL — `ImportError: cannot import name 'lock_legs'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
def _opp(side: Side) -> Side:
    return "put" if side == "call" else "call"


def _debit_spread_legs(forward: float, em: float, cfg: BurritoConfig, side: Side) -> List[Leg]:
    right: Right = "C" if side == "call" else "P"
    sign = 1.0 if side == "call" else -1.0
    body = round_to_step(forward + sign * cfg.body_offset * em, cfg.step)
    outer = body + sign * cfg.fly_wing_width
    return [Leg(1, outer, right), Leg(-1, outer + sign * cfg.spread_width, right)]


def lock_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Lock into 'free' status: buy the OPPOSING-side debit spread."""
    return _debit_spread_legs(forward, em, cfg, _opp(side))


def wing_lock_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Preemptive wing lock == buy the opposing debit spread (same as lock)."""
    return _debit_spread_legs(forward, em, cfg, _opp(side))


def otm_freeze_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Buy 1 cheap far-OTM option in the TREND (adverse) direction to flatten."""
    opp = _opp(side)
    right: Right = "C" if opp == "call" else "P"
    sign = 1.0 if opp == "call" else -1.0
    k = round_to_step(forward + sign * 3.0 * em, cfg.step)   # ~far OTM
    return [Leg(1, k, right)]


def choo_choo_legs(side: Side, forward: float, em: float, cfg: BurritoConfig) -> List[Leg]:
    """Add a second fly_wing_width fly into the adverse move (overlapping tent)."""
    return build_legs(forward, em, cfg, side)[:3] if False else _fly_only(forward, em, cfg, side)


def _fly_only(forward: float, em: float, cfg: BurritoConfig, side: Side) -> List[Leg]:
    right: Right = "C" if side == "call" else "P"
    sign = 1.0 if side == "call" else -1.0
    W = cfg.fly_wing_width
    body = round_to_step(forward + sign * cfg.body_offset * em, cfg.step)
    return _merge([Leg(1, body - sign * W, right), Leg(-2, body, right), Leg(1, body + sign * W, right)])


def clawback_legs(legs: List[Leg], forward: float, cfg: BurritoConfig) -> List[Leg]:
    """Hole-injection: a standard fly centered on the nearest sub-zero valley to
    lift it above breakeven. valley_center found by scanning the payoff curve."""
    center = _nearest_valley(legs, forward, cfg)
    if center is None:
        return []
    W = cfg.fly_wing_width
    # use calls above forward, puts below (cheaper OTM wings)
    right: Right = "C" if center >= forward else "P"
    return _merge([Leg(1, center - W, right), Leg(-2, center, right), Leg(1, center + W, right)])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -k adjustment_legs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): lock/defense/clawback adjustment legs"
```

---

## Task 9: Valley detection for clawback

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_nearest_valley_finds_subzero_gap():
    from src_v2.backtest.burrito_core import build_legs, _nearest_valley, BurritoConfig
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    legs = build_legs(5000.0, 20.0, cfg, "call")     # value 0 far from body, peak at 5020
    # with a debit basis, points far below body settle to a loss "valley"
    v = _nearest_valley(legs, forward=4950.0, cfg=cfg, debit=10.0)
    assert v is None or isinstance(v, float)
    # a clearly sub-zero region (deep OTM) should report a valley center
    v2 = _nearest_valley(legs, forward=4700.0, cfg=cfg, debit=10.0)
    assert v2 is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_nearest_valley_finds_subzero_gap -v`
Expected: FAIL — `ImportError: cannot import name '_nearest_valley'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
def _nearest_valley(legs: List[Leg], forward: float, cfg: BurritoConfig,
                    debit: float = 0.0) -> Optional[float]:
    """Scan settlement payoff around the forward; return the strike-step center of
    the nearest region where net payoff (payoff - debit) < 0, else None."""
    lo = forward - 6 * cfg.fly_wing_width
    hi = forward + 6 * cfg.fly_wing_width
    s = lo
    best = None
    best_dist = 1e18
    while s <= hi:
        if payoff(legs, s) - debit < 0:
            d = abs(s - forward)
            if d < best_dist:
                best_dist, best = d, round_to_step(s, cfg.step)
        s += cfg.step
    return best
```

Update `clawback_legs` to pass `debit` (use `tr.debit + tr.extra_cost`); update the call
signature in Task 8 accordingly when wiring the engine in Task 10.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_nearest_valley_finds_subzero_gap -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): payoff-valley detection for clawback"
```

---

## Task 10: Engine — run one trade end to end

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

`run_trade(store, entry_date, expiry, side, em, cfg)` builds legs, walks `eod_marks`,
applies `lifecycle_decision`, executes adjustments (mutating `tr.legs`/`tr.extra_cost`),
and closes with realized P&L. Settlement uses `parity_forward` of the expiry chain.

- [ ] **Step 1: Write the failing test**

```python
def test_run_trade_settles_to_pnl():
    from src_v2.backtest.burrito_core import run_trade, BurritoConfig
    from src_v2.backtest.iron_duck import DuckStore
    store = DuckStore("data/iron_duck_eod/SPXW")
    # pick a real expiry with several lead days
    exp = [e for e in store.expiries if e >= 20240601][0]
    cands = [d for d in store.quote_dates() if d < exp]
    entry = cands[-4]
    tr = run_trade(store, entry_date=entry, expiry=exp, side="call",
                   em=None, cfg=BurritoConfig(direction_signal="both", gates_on=False))
    assert tr.exit_reason in {"target", "uncle_stop", "expiry"}
    assert tr.pnl_usd is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_run_trade_settles_to_pnl -v`
Expected: FAIL — `ImportError: cannot import name 'run_trade'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
from src_v2.backtest.iron_duck import Smile, parity_forward
from src_v2.backtest.burrito_clock import eod_marks


def _em_from_chain(chain, F: float, T: float) -> float:
    """Expected move proxy = ATM straddle mid (per project guidance: ATM straddle,
    not VIX). Falls back to 1% of F if unpriceable."""
    sm = Smile(chain, F, T)
    if sm.ok():
        v = sm.vol(F)
        if v:
            return F * v * (T ** 0.5)
    return 0.01 * F


def run_trade(store, entry_date: int, expiry: int, side: Side,
              em: Optional[float], cfg: BurritoConfig):
    ch0 = store.chain(entry_date, expiry)
    if ch0 is None:
        return None
    F0 = parity_forward(ch0)
    T0 = max((_to_date(expiry) - _to_date(entry_date)).days / 365.0, 1e-6)
    if em is None:
        em = _em_from_chain(ch0, F0, T0)
    legs = build_legs(F0, em, cfg, side)
    debit = entry_debit(legs, ch0, cfg.slippage_k, cfg.commission_per_leg)
    if debit is None or (cfg.gates_on and not passes_gates(debit, cfg)):
        return None
    tr = BurritoTrade(side=side, legs=legs, debit=debit, entry_date=entry_date,
                      expiry=expiry, dte=(_to_date(expiry) - _to_date(entry_date)).days)

    marks = list(eod_marks(store, entry_date, expiry))
    for m in marks:
        sm = Smile(m.chain, m.forward, m.T) if m.forward else None
        val = position_value(tr.legs, m.chain, sm, m.forward, cfg.slippage_k)
        if val is None:
            continue
        in_valley = (m.is_last or marks.index(m) >= len(marks) - cfg.clawback_last_n) and \
            _nearest_valley(tr.legs, m.forward, cfg, tr.debit + tr.extra_cost) is not None
        d = lifecycle_decision(tr, tr.state, val, cfg, m.is_last, in_valley)
        if d.action == "lock":
            add = lock_legs(tr.side, m.forward, em, cfg)
            tr.extra_cost += entry_debit(add, m.chain, cfg.slippage_k, cfg.commission_per_leg) or 0.0
            tr.legs = _merge(tr.legs + add); tr.state = d.next_state
        elif d.action == "defend":
            add = _apply_defense(tr, m, em, cfg)
            tr.extra_cost += entry_debit(add, m.chain, cfg.slippage_k, cfg.commission_per_leg) or 0.0
            tr.legs = _merge(tr.legs + add); tr.state = d.next_state
        elif d.action == "clawback":
            add = clawback_legs(tr.legs, m.forward, cfg, tr.debit + tr.extra_cost)
            tr.extra_cost += entry_debit(add, m.chain, cfg.slippage_k, cfg.commission_per_leg) or 0.0
            tr.legs = _merge(tr.legs + add)
        elif d.action in ("close", "settle"):
            _close(tr, m, val, d.reason, cfg)
            return tr
    # safety net: settle at last mark if loop exits without close
    if marks:
        last = marks[-1]
        val = position_value(tr.legs, last.chain, Smile(last.chain, last.forward, last.T),
                             last.forward, cfg.slippage_k) or (tr.debit + tr.extra_cost)
        _close(tr, last, val, "expiry", cfg)
    return tr


def _apply_defense(tr, m, em, cfg) -> List[Leg]:
    if "wing_lock" in cfg.defense_policy:
        return wing_lock_legs(tr.side, m.forward, em, cfg)
    if "choo_choo" in cfg.defense_policy:
        return choo_choo_legs(tr.side, m.forward, em, cfg)
    if "otm_freeze" in cfg.defense_policy:
        return otm_freeze_legs(tr.side, m.forward, em, cfg)
    return []


def _close(tr, m, val, reason, cfg):
    if reason in ("expiry", "settle"):
        settle = parity_forward(m.chain) or m.forward
        gross = payoff(tr.legs, settle)
    else:
        gross = val
    tr.pnl_pts = gross - (tr.debit + tr.extra_cost)
    tr.pnl_usd = tr.pnl_pts * cfg.multiplier
    tr.exit_date = m.date
    tr.exit_reason = reason if reason != "settle" else "expiry"
    tr.state = "CLOSED"
    tr.days_held = (_to_date(m.date) - _to_date(tr.entry_date)).days
```

Note: update `clawback_legs` signature to `clawback_legs(legs, forward, cfg, debit)` and
`_nearest_valley(legs, forward, cfg, debit)` consistently (done in Task 9).

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_run_trade_settles_to_pnl -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): single-trade engine (entry->lifecycle->settle)"
```

---

## Task 11: summarize (Sharpe / PF / WR / DD) and full-history run loop

**Files:**
- Modify: `src_v2/backtest/burrito_core.py`
- Test: `tests/test_burrito_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_summarize_sharpe_basic():
    from src_v2.backtest.burrito_core import summarize_burrito
    # three closed trades, simple daily equity series
    class T:
        def __init__(self, pnl, d, reason="target"):
            self.pnl_usd = pnl; self.exit_date = d; self.exit_reason = reason; self.debit = 5.0
    trades = [T(100, 20240102), T(-50, 20240103), T(80, 20240104)]
    out = summarize_burrito(trades)
    assert out["trades"] == 3
    assert "sharpe" in out and isinstance(out["sharpe"], float)
    assert out["win_rate"] == round(2 / 3, 4)
    assert out["profit_factor"] > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_summarize_sharpe_basic -v`
Expected: FAIL — `ImportError: cannot import name 'summarize_burrito'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_core.py
def summarize_burrito(trades) -> Dict:
    closed = [t for t in trades if getattr(t, "pnl_usd", None) is not None]
    if not closed:
        return {"trades": 0}
    pnls = [t.pnl_usd for t in closed]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    # daily returns: group P&L by exit_date, normalize by per-trade capital basis
    import statistics
    rets = []
    for t in closed:
        basis = (getattr(t, "debit", 0.0) + getattr(t, "extra_cost", 0.0)) * 100.0
        rets.append(t.pnl_usd / basis if basis else 0.0)
    mean = sum(rets) / len(rets)
    sd = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    sharpe = (mean / sd * (252 ** 0.5)) if sd > 0 else 0.0
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    by_reason: Dict[str, int] = {}
    for t in closed:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
    return {
        "trades": len(closed),
        "net_pnl": round(sum(pnls), 2),
        "win_rate": round(len(wins) / len(closed), 4),
        "avg_pnl": round(sum(pnls) / len(closed), 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else float("inf"),
        "avg_ret": round(mean, 4),
        "exit_reasons": by_reason,
    }


def run_history(store, cfg: BurritoConfig, start: int = 0, end: int = 99999999,
                price_series_fn=None) -> List[BurritoTrade]:
    """Sequential entries: on each eligible quote date pick the expiry nearest
    cfg.entry_dte, derive direction from the spot series, enter, and run to close.
    Re-enter on the next quote date after a close (no overlap)."""
    trades: List[BurritoTrade] = []
    dates = [d for d in store.quote_dates() if start <= d <= end]
    spots = {d: store.spot(d) for d in dates}
    series = [spots[d] for d in dates if spots[d] is not None]
    next_ok_idx = 0
    for i, d in enumerate(dates):
        if i < next_ok_idx:
            continue
        cands = store.expiries_in_dte(d, max(cfg.entry_dte, 0), cfg.entry_dte + 1) \
            if cfg.entry_dte > 0 else store.expiries_in_dte(d, 0, 0)
        if not cands:
            continue
        expiry = min(cands, key=lambda e: (abs(e[1] - cfg.entry_dte), e[1]))[0]
        hist = [spots[x] for x in dates[:i + 1] if spots[x] is not None]
        direction = direction_from_series(hist, cfg.direction_signal)
        if direction == "skip":
            continue
        sides = ["call", "put"] if direction == "both" else [direction]
        for side in sides:
            tr = run_trade(store, d, expiry, side, None, cfg)
            if tr is not None:
                trades.append(tr)
        # advance past this expiry to avoid overlap
        next_ok_idx = next((j for j, x in enumerate(dates) if x > expiry), len(dates))
    return trades
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_summarize_sharpe_basic -v`
Expected: PASS. Then run the whole file: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -v` — all green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): Sharpe summarize + sequential run_history loop"
```

---

## Task 12: CLI runner

**Files:**
- Create: `scripts/backtest_burrito.py`

- [ ] **Step 1: Write the runner (no test — thin I/O wrapper, validated by smoke run)**

```python
# scripts/backtest_burrito.py
"""Run a single Burrito config over the EOD SPXW cache. Writes trades.csv +
metrics.json under results/burrito/. Usage:
    .venv/bin/python scripts/backtest_burrito.py --dte 3 --signal hull \
        --lock 0.07 --stop 0.07 --pt 0.10 --slippage 1.0 --start 20200101
"""
import argparse, csv, json, os
from dataclasses import asdict
from src_v2.backtest.iron_duck import DuckStore
from src_v2.backtest.burrito_core import BurritoConfig, run_history, summarize_burrito

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/iron_duck_eod/SPXW")
    ap.add_argument("--dte", type=int, default=3)
    ap.add_argument("--signal", default="hull")
    ap.add_argument("--lock", type=float, default=0.07)
    ap.add_argument("--stop", type=float, default=0.07)
    ap.add_argument("--pt", type=float, default=0.10)
    ap.add_argument("--defense", default="")  # comma list
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99999999)
    ap.add_argument("--out", default="results/burrito")
    a = ap.parse_args()
    cfg = BurritoConfig(entry_dte=a.dte, direction_signal=a.signal, lock_trigger_pct=a.lock,
                        stop_pct=a.stop, pt_pct=a.pt, slippage_k=a.slippage,
                        defense_policy=tuple(x for x in a.defense.split(",") if x))
    store = DuckStore(a.cache)
    trades = run_history(store, cfg, a.start, a.end)
    metrics = summarize_burrito(trades)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics}, f, indent=2, default=str)
    with open(os.path.join(a.out, "trades.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_date", "expiry", "side", "dte", "debit", "extra_cost",
                    "exit_date", "exit_reason", "pnl_usd"])
        for t in trades:
            w.writerow([t.entry_date, t.expiry, t.side, t.dte, round(t.debit, 2),
                        round(t.extra_cost, 2), t.exit_date, t.exit_reason,
                        None if t.pnl_usd is None else round(t.pnl_usd, 2)])
    print(json.dumps(metrics, indent=2, default=str))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

Run: `~/Dev/0dte_bot/.venv/bin/python scripts/backtest_burrito.py --dte 3 --signal both --start 20240101 --end 20240630`
Expected: prints a metrics JSON with `trades` > 0 and a `sharpe` value; files appear under `results/burrito/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest_burrito.py && git commit -m "feat(burrito): CLI single-config runner"
```

---

## Task 13: Walk-forward CV sweep + slippage stress

**Files:**
- Create: `scripts/wf_cv_burrito.py`

Mirror the methodology in `docs/wf-cv.md` and `tools/theta/wf_cv_regime.py`: rolling
train/validation/test tiers, select params on validation, report OOS (test) metrics, prefer a
plateau over the single best cell, and never select on test.

- [ ] **Step 1: Write the sweeper**

```python
# scripts/wf_cv_burrito.py
"""Walk-forward CV over the Burrito param grid, maximizing OOS Sharpe. Splits the
2020-2026 history into rolling (train+val -> test) windows; selects the param cell
by validation Sharpe with a plateau preference; reports test (OOS) metrics per fold.
Also runs a slippage stress grid on the selected cell.
Usage: .venv/bin/python scripts/wf_cv_burrito.py --folds 5
"""
import argparse, json, itertools, os
from src_v2.backtest.iron_duck import DuckStore
from src_v2.backtest.burrito_core import BurritoConfig, run_history, summarize_burrito

GRID = {
    "entry_dte": [0, 1, 2, 3, 5],
    "direction_signal": ["hull", "momentum", "markov", "both"],
    "lock_trigger_pct": [0.05, 0.07, 0.10],
    "stop_pct": [0.05, 0.07, 0.10],
    "pt_pct": [0.05, 0.10],
    "spread_link": ["outer", "inner"],
    "defense_policy": [(), ("wing_lock",), ("choo_choo",), ("clawback",)],
}

def cells():
    keys = list(GRID)
    for combo in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, combo))

def metric(store, params, start, end, slippage=1.0):
    cfg = BurritoConfig(slippage_k=slippage, gates_on=True, **params)
    trades = run_history(store, cfg, start, end)
    return summarize_burrito(trades)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/iron_duck_eod/SPXW")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results/burrito/wfcv.json")
    a = ap.parse_args()
    store = DuckStore(a.cache)
    qd = store.quote_dates()
    yrs = sorted({d // 10000 for d in qd})
    # rolling: train+val = prior years, test = next year
    results = []
    for k in range(2, len(yrs)):
        train_val = [y for y in yrs[:k]]
        test_y = yrs[k]
        tv_start, tv_end = train_val[0] * 10000 + 101, train_val[-1] * 10000 + 1231
        # split train_val 70/30 by time for validation
        split = train_val[int(len(train_val) * 0.7)] * 10000 + 101
        best, best_sharpe = None, -1e9
        for params in cells():
            v = metric(store, params, split, tv_end)
            s = v.get("sharpe", -1e9) if v.get("trades", 0) >= 10 else -1e9
            if s > best_sharpe:
                best_sharpe, best = s, params
        oos = metric(store, best, test_y * 10000 + 101, test_y * 10000 + 1231)
        results.append({"test_year": test_y, "selected": best,
                        "val_sharpe": round(best_sharpe, 3), "oos": oos})
    # slippage stress on the most recent selected cell
    stress = {}
    if results:
        sel = results[-1]["selected"]
        for sk in (1.0, 1.25, 1.5, 2.0):
            stress[sk] = metric(store, sel, yrs[0] * 10000 + 101, yrs[-1] * 10000 + 1231, sk)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"folds": results, "slippage_stress": stress}, f, indent=2, default=str)
    for r in results:
        print(r["test_year"], "OOS sharpe:", r["oos"].get("sharpe"), "sel:", r["selected"])

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (reduced grid)**

Temporarily shrink `GRID` (e.g. one value per key except `direction_signal`) and run:
`~/Dev/0dte_bot/.venv/bin/python scripts/wf_cv_burrito.py --folds 3`
Expected: prints per-fold OOS Sharpe lines; `results/burrito/wfcv.json` written. Restore the full grid before the real run.

- [ ] **Step 3: Commit**

```bash
git add scripts/wf_cv_burrito.py && git commit -m "feat(burrito): walk-forward CV sweep + slippage stress"
```

---

## Task 14: 0DTE intraday clock (DTE=0 leg)

**Files:**
- Modify: `src_v2/backtest/burrito_clock.py`
- Test: `tests/test_burrito_core.py`

Add `intraday_marks(...)` reading `data/cbbo_cache_fly_10min/` for DTE=0 entries so the
lifecycle fires intraday. Inspect the cache format first:
`~/Dev/0dte_bot/.venv/bin/python -c "import pickle,glob; p=sorted(glob.glob('data/cbbo_cache_fly_10min/*'))[0]; print(p); d=pickle.load(open(p,'rb')); print(type(d)); print(list(d)[:5] if hasattr(d,'__iter__') else d)"`

- [ ] **Step 1: Write the failing test** (adapt keys to the real format discovered above)

```python
def test_intraday_marks_for_0dte():
    from src_v2.backtest.burrito_clock import intraday_marks
    # build a minimal in-memory intraday provider stub and assert ordering + is_last
    class FakeIntraday:
        def bars(self, date):
            return ["09:40", "10:00", "15:50"]
        def chain_at(self, date, minute):
            return {(5000.0, "C"): (1.0, 2.0, 0), (5000.0, "P"): (1.0, 2.0, 0),
                    (4900.0, "P"): (0.5, 1.0, 0), (5100.0, "C"): (0.5, 1.0, 0)}
        def forward_at(self, date, minute):
            return 5000.0
    marks = list(intraday_marks(FakeIntraday(), date=20240705, entry_min="09:40"))
    assert [m.intraday_min for m in marks] == ["09:40", "10:00", "15:50"]
    assert marks[-1].is_last and not marks[0].is_last
    assert marks[0].T > marks[-1].T >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_intraday_marks_for_0dte -v`
Expected: FAIL — `ImportError: cannot import name 'intraday_marks'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to burrito_clock.py
def intraday_marks(provider, date: int, entry_min: str):
    """0DTE marks from a 10-min intraday provider. `provider` exposes bars(date)->
    ['HH:MM'...], chain_at(date,min)->Chain, forward_at(date,min)->float."""
    bars = [b for b in provider.bars(date) if b >= entry_min]
    for i, b in enumerate(bars):
        ch = provider.chain_at(date, b)
        F = provider.forward_at(date, b)
        hh, mm = int(b[:2]), int(b[3:5])
        frac_day_left = max((16 * 60 - (hh * 60 + mm)) / (6.5 * 60), 0.0)
        T = max(frac_day_left / 252.0, 1e-9)
        yield Mark(date=date, intraday_min=b, chain=ch, forward=F, T=T,
                   is_last=(i == len(bars) - 1))
```

Then wire `run_trade` to choose `intraday_marks` when `cfg.entry_dte == 0` (provider built
in the runner from the 10-min cache). Add a `marks` injection parameter to `run_trade`
(`marks: Optional[list] = None`) so it accepts pre-built marks from either clock; default
remains `eod_marks`. Update the Task 10 `run_trade` call sites accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py::test_intraday_marks_for_0dte -v`
Expected: PASS. Full suite still green: `~/Dev/0dte_bot/.venv/bin/pytest tests/test_burrito_core.py -v`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(burrito): 0DTE intraday clock marks"
```

---

## Task 15: Full WF-CV run + findings doc

**Files:**
- Create: `docs/burrito_butterfly_findings.md`

- [ ] **Step 1: Run the full sweep** (restore the full `GRID` first)

Run: `~/Dev/0dte_bot/.venv/bin/python scripts/wf_cv_burrito.py --folds 5`
(Long-running — run in background if needed.) Expected: `results/burrito/wfcv.json` with per-fold
OOS Sharpe and a slippage stress block.

- [ ] **Step 2: Write the findings doc**

Summarize, with numbers pulled from `results/burrito/wfcv.json`:
- OOS Sharpe per fold (and median); does periodic re-optimization hold OOS, or is it curve-fit?
- Which `direction_signal` / DTE / lifecycle params survive walk-forward; structure vs. directional
  edge (the `both` baseline vs. the best signal).
- Slippage sensitivity (Sharpe at `slippage_k` 1.0 → 2.0).
- Honest verdict (validated / marginal / refuted), and the EOD-granularity stop-optimism caveat
  (intraday hit-then-revert invisible for DTE≥1).

- [ ] **Step 3: Commit**

```bash
git add docs/burrito_butterfly_findings.md results/burrito/wfcv.json
git commit -m "docs(burrito): WF-CV results and findings verdict"
```

---

## Self-review notes (addressed)

- **Spec coverage:** geometry+structural link (T1, `spread_link` param), payoff (T2), MTM+model
  fallback (T3), entry+gates (T4), signal menu incl. markov (T5), EOD clock (T6) + intraday 0DTE
  clock (T14), full lifecycle lock/Uncle/target/settle (T7), defenses wing_lock/otm_freeze/choo_choo
  (T8), clawback + valley detection (T8/T9), Sharpe summarize + WF-CV + slippage stress (T11/T13),
  findings verdict (T15). DTE sweep + EM-via-ATM-straddle honored (T10/T13).
- **Type consistency:** `_nearest_valley(legs, forward, cfg, debit)` and
  `clawback_legs(legs, forward, cfg, debit)` use the 4-arg form consistently (T9/T10). `run_trade`
  gains an optional `marks` param in T14; T10 default path uses `eod_marks`.
- **Costs:** every adjustment priced through `entry_debit` with `slippage_k`; results reported net
  and stress-tested (T13).
