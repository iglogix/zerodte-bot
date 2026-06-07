# DC Time Machine Backtest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trustworthy, intraday, multi-expiration backtest of the "DC Time Machine" double-calendar → risk-free iron-condor strategy on SPX, and produce a GO/NO-GO verdict on its execution-robust profitability.

**Architecture:** A self-contained package `src_v2/backtest/dc_time_machine/` with a multi-expiration quote store, pure pricing/IV/fill math, a minute-stepped per-trade lifecycle simulator, a Databento backfill driver, an empirical spread study, a reconstructed "Flux" IV-ratio search, and a dedicated WF-CV runner. It reuses project primitives (`osi_symbol`, the `(minute,strike,is_call)→(bid,ask)` quote shape, the Databento `cbbo-1m` fetch pattern) but touches **zero** existing 0DTE/MEIC code and **nothing** in `tests_locked/`.

**Tech Stack:** Python 3, pytest, numpy/pandas, scipy (`brentq`, `norm`), Databento Historical client (mocked in tests). Run everything with the project venv: `.venv/bin/python` / `.venv/bin/pytest`.

**Design reference:** `docs/superpowers/specs/2026-06-07-dc-time-machine-backtest-design.md`.

**Branch:** `feat/dc-time-machine` (already created; the design spec is committed there).

---

## File Structure

| File | Responsibility |
|---|---|
| `src_v2/backtest/dc_time_machine/__init__.py` | Package marker |
| `src_v2/backtest/dc_time_machine/strategy.py` | `DcParams` — the sweepable parameter dataclass |
| `src_v2/backtest/dc_time_machine/pricing.py` | Pure payoff math: entry debit, transform credit, IC terminal P&L, risk-free check |
| `src_v2/backtest/dc_time_machine/fills.py` | `fill_price` (per-leg), `net_combo_credit` (combo); `mid ± k·half_spread` |
| `src_v2/backtest/dc_time_machine/iv.py` | Black–Scholes price/delta, implied-vol inversion, ATM IV, front/back ratio |
| `src_v2/backtest/dc_time_machine/policy.py` | `entry_allowed` (filters) + `overnight_decision` (seam) |
| `src_v2/backtest/dc_time_machine/multi_expiry_store.py` | `MultiExpiryStore`: `{session:{expiry:{(min,strike,call):(bid,ask)}}}`, lookup, settlement |
| `src_v2/backtest/dc_time_machine/simulator.py` | `simulate_trade` lifecycle + `TradeResult`; strike/expiry resolution |
| `src_v2/backtest/dc_time_machine/dc_backfill.py` | Databento full-day, per-expiry fetch + session backfill writer |
| `src_v2/backtest/dc_time_machine/spreads.py` | Empirical half-spread study (the slippage experiment) |
| `src_v2/backtest/dc_time_machine/flux_scanner.py` | Reconstructed "Flux" scan → candidate entries |
| `src_v2/backtest/dc_time_machine/wf_cv.py` | Dedicated 5:1 / 4-fold WF-CV runner over the DC simulator |
| `src_v2/backtest/dc_time_machine/run_backtest.py` | CLI: `backfill | study-spreads | simulate | scan | wf-cv | crosscheck` |
| `tests/dc_time_machine/test_*.py` | One test module per source module |

Build order respects dependencies: pure math first (Tasks 1–5), then store (6), simulator (7), then data/IO/analysis (8–12), then WF-CV + CLI (13–14).

**Conventions used throughout (defined once, referenced everywhere):**
- A *quote* is `(bid, ask)`; a *quote key* is `(minute_et: str, strike: float, is_call: bool)`.
- A *session panel* is `{expiry_date: {quote_key: (bid, ask)}}`.
- All prices are **per share** ($/share); multiply by `100 * qty` for dollars.
- `side` is the string `"buy"` or `"sell"`.
- `k` (fill aggressiveness) ∈ `[0, 1]`: `0` = mid, `1` = full spread cross.

---

## Task 1: Package scaffold + `DcParams`

**Files:**
- Create: `src_v2/backtest/dc_time_machine/__init__.py`
- Create: `src_v2/backtest/dc_time_machine/strategy.py`
- Test: `tests/dc_time_machine/test_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_strategy.py
from src_v2.backtest.dc_time_machine.strategy import DcParams


def test_defaults_match_video_favorite():
    p = DcParams()
    assert p.entry_minute == "09:40"
    assert p.target_delta == 0.35
    assert p.front_dte == 9
    assert p.gap == 1
    assert p.wing_width == 5.0
    assert p.transform_epsilon == 0.0
    assert p.stop_loss_frac == 0.20
    assert p.fill_k == 1.0
    assert p.entry_filter == "always"
    assert p.overnight_policy == "close"


def test_params_are_frozen_and_overridable():
    p = DcParams(target_delta=0.20, wing_width=10.0, entry_filter="fri_mon")
    assert (p.target_delta, p.wing_width, p.entry_filter) == (0.20, 10.0, "fri_mon")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError: src_v2.backtest.dc_time_machine`.

- [ ] **Step 3: Create the package marker and the params**

```python
# src_v2/backtest/dc_time_machine/__init__.py
"""DC Time Machine — multi-expiration double-calendar backtest harness."""
```

```python
# src_v2/backtest/dc_time_machine/strategy.py
"""DcParams — the single sweepable parameter object for the DC Time Machine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DcParams:
    entry_minute: str = "09:40"        # ET; baseline = shortly after open
    target_delta: float = 0.35         # abs delta for both K_p (put) and K_c (call)
    front_dte: int = 9                 # front expiry = Nth available expiry after session
    gap: int = 1                       # back expiry = front index + gap
    wing_width: float = 5.0            # IC wing width in points (W)
    transform_epsilon: float = 0.0     # extra credit required over (D + W), in $/share
    stop_loss_frac: float = 0.20       # mental stop on the un-transformed DC
    fill_k: float = 1.0                # 0 = mid, 1 = full cross
    entry_filter: str = "always"       # always | backwardation | fri_mon | spike
    overnight_policy: str = "close"    # close | hold
    rate: float = 0.04                 # risk-free rate for BS/IV
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_strategy.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/__init__.py src_v2/backtest/dc_time_machine/strategy.py tests/dc_time_machine/test_strategy.py
git commit -m "feat(dc): package scaffold + DcParams"
```

---

## Task 2: Pricing math (`pricing.py`)

The core payoff identities. `D = backs@ask − fronts@bid`; `C = backs@bid − wings@ask`; the IC terminal P&L and the risk-free invariant.

**Files:**
- Create: `src_v2/backtest/dc_time_machine/pricing.py`
- Test: `tests/dc_time_machine/test_pricing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_pricing.py
import pytest

from src_v2.backtest.dc_time_machine.pricing import (
    entry_debit, transform_credit, ic_terminal_pnl, is_risk_free,
)


def test_entry_debit_is_backs_minus_fronts():
    # buy backs @ ask (5.0 + 5.2), sell fronts @ bid (4.9 + 5.0)
    assert entry_debit(front_put_bid=4.9, front_call_bid=5.0,
                       back_put_ask=5.0, back_call_ask=5.2) == pytest.approx(0.3)


def test_transform_credit_is_backbids_minus_wingasks():
    # sell back bids (5.5 + 5.6), buy wing asks (0.4 + 0.5)
    assert transform_credit(back_put_bid=5.5, back_call_bid=5.6,
                            put_wing_ask=0.4, call_wing_ask=0.5) == pytest.approx(10.2)


def test_his_example_reconciles():
    # Video: D=10.10, C=15.25, W=5 → bank 5.15; inside tent = 5.15/share, floor = 0.15/share
    D, C, W = 10.10, 15.25, 5.0
    bank = C - D
    assert is_risk_free(D, C, W) is True
    assert ic_terminal_pnl(settlement=5000, k_put=4900, k_call=5100, wing=W, bank=bank) == pytest.approx(5.15)
    # settle far below the put long → floor
    assert ic_terminal_pnl(settlement=4000, k_put=4900, k_call=5100, wing=W, bank=bank) == pytest.approx(0.15)


def test_partial_penetration_is_linear():
    bank = 6.0
    # S between short put (100) and long put (95): loss = 100 - 98 = 2
    assert ic_terminal_pnl(settlement=98, k_put=100, k_call=110, wing=5, bank=bank) == pytest.approx(4.0)


def test_risk_free_requires_bank_ge_wing():
    assert is_risk_free(debit=10.0, credit=14.99, wing=5.0) is False
    assert is_risk_free(debit=10.0, credit=15.00, wing=5.0) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_pricing.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/pricing.py
"""Pure payoff math for the DC Time Machine. All values are $/share."""
from __future__ import annotations


def entry_debit(front_put_bid: float, front_call_bid: float,
                back_put_ask: float, back_call_ask: float) -> float:
    """Net debit to open the double calendar: buy backs @ ask, sell fronts @ bid."""
    return (back_put_ask + back_call_ask) - (front_put_bid + front_call_bid)


def transform_credit(back_put_bid: float, back_call_bid: float,
                     put_wing_ask: float, call_wing_ask: float) -> float:
    """Quote-achievable transform credit: sell backs @ bid, buy front wings @ ask."""
    return (back_put_bid + back_call_bid) - (put_wing_ask + call_wing_ask)


def is_risk_free(debit: float, credit: float, wing: float) -> bool:
    """The transform is risk-free iff banked cash (C - D) covers the wing width."""
    return (credit - debit) >= wing


def ic_terminal_pnl(settlement: float, k_put: float, k_call: float,
                    wing: float, bank: float) -> float:
    """Terminal P&L ($/share) of the front-expiry IC held to cash settlement.

    bank = C - D (cash already banked). Only one vertical can be ITM (k_put < k_call).
    """
    put_loss = min(max(k_put - settlement, 0.0), wing)
    call_loss = min(max(settlement - k_call, 0.0), wing)
    return bank - put_loss - call_loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_pricing.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/pricing.py tests/dc_time_machine/test_pricing.py
git commit -m "feat(dc): pricing math (debit, transform credit, IC terminal P&L, risk-free)"
```

---

## Task 3: Fill model (`fills.py`)

`mid ± k·half_spread` per leg, plus the net-credit combo fill used by the transform.

**Files:**
- Create: `src_v2/backtest/dc_time_machine/fills.py`
- Test: `tests/dc_time_machine/test_fills.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_fills.py
import pytest

from src_v2.backtest.dc_time_machine.fills import fill_price, net_combo_credit


def test_buy_fill_walks_from_mid_to_ask():
    assert fill_price(bid=1.0, ask=2.0, side="buy", k=0.0) == pytest.approx(1.5)  # mid
    assert fill_price(bid=1.0, ask=2.0, side="buy", k=1.0) == pytest.approx(2.0)  # ask
    assert fill_price(bid=1.0, ask=2.0, side="buy", k=0.5) == pytest.approx(1.75)


def test_sell_fill_walks_from_mid_to_bid():
    assert fill_price(bid=1.0, ask=2.0, side="sell", k=0.0) == pytest.approx(1.5)  # mid
    assert fill_price(bid=1.0, ask=2.0, side="sell", k=1.0) == pytest.approx(1.0)  # bid


def test_net_combo_credit_brackets_mid_to_cross():
    legs = [
        (5.4, 5.6, "sell"),  # back put
        (5.4, 5.6, "sell"),  # back call
        (0.3, 0.5, "buy"),   # put wing
        (0.4, 0.6, "buy"),   # call wing
    ]
    assert net_combo_credit(legs, k=0.0) == pytest.approx(10.1)  # net mid
    assert net_combo_credit(legs, k=1.0) == pytest.approx(9.7)   # net cross (−4×0.1)


def test_unknown_side_raises():
    with pytest.raises(ValueError):
        fill_price(1.0, 2.0, side="hold", k=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_fills.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/fills.py
"""Parameterized fill model. Every fill is mid ± k·half_spread (k=0 mid, k=1 cross)."""
from __future__ import annotations

from typing import List, Tuple


def fill_price(bid: float, ask: float, side: str, k: float) -> float:
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    if side == "buy":
        return mid + k * half
    if side == "sell":
        return mid - k * half
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def net_combo_credit(legs: List[Tuple[float, float, str]], k: float) -> float:
    """Net credit ($/share) for an atomic combo: +sell mids, −buy mids, less k·Σhalf_spread.

    legs: list of (bid, ask, side). Adverse fill ALWAYS reduces the received credit,
    so a single k·Σhalf term is subtracted regardless of per-leg side.
    """
    net = 0.0
    total_half = 0.0
    for bid, ask, side in legs:
        mid = (bid + ask) / 2.0
        total_half += (ask - bid) / 2.0
        if side == "sell":
            net += mid
        elif side == "buy":
            net -= mid
        else:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return net - k * total_half
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_fills.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/fills.py tests/dc_time_machine/test_fills.py
git commit -m "feat(dc): parameterized fill model (per-leg + net combo, k-sweep)"
```

---

## Task 4: Black–Scholes + implied vol + front/back ratio (`iv.py`)

**Files:**
- Create: `src_v2/backtest/dc_time_machine/iv.py`
- Test: `tests/dc_time_machine/test_iv.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_iv.py
import math
import pytest

from src_v2.backtest.dc_time_machine.iv import (
    bs_price, bs_delta, implied_vol, front_back_ratio,
)


def test_bs_price_round_trips_through_implied_vol():
    S, K, T, r, sigma = 5000.0, 5000.0, 9 / 365, 0.04, 0.18
    price = bs_price(S, K, T, sigma, r, is_call=True)
    iv = implied_vol(price, S, K, T, r, is_call=True)
    assert iv == pytest.approx(0.18, abs=1e-3)


def test_atm_call_delta_near_half():
    d = bs_delta(5000.0, 5000.0, 9 / 365, 0.18, 0.04, is_call=True)
    assert 0.5 < d < 0.6  # slightly above 0.5 due to drift/vega skew of d1


def test_put_delta_is_negative():
    d = bs_delta(5000.0, 5000.0, 9 / 365, 0.18, 0.04, is_call=False)
    assert -0.6 < d < -0.4


def test_implied_vol_nan_when_price_below_intrinsic():
    iv = implied_vol(price=0.01, S=5100.0, K=5000.0, T=9 / 365, r=0.04, is_call=True)
    assert math.isnan(iv)


def test_front_back_ratio():
    assert front_back_ratio(0.22, 0.20) == pytest.approx(1.1)
    assert math.isnan(front_back_ratio(0.22, 0.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_iv.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/iv.py
"""Black–Scholes pricing, implied-vol inversion, and the front/back IV ratio (Flux)."""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def _d1(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_price(S: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = _d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if is_call:
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, T, sigma, r)
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def implied_vol(price: float, S: float, K: float, T: float, r: float, is_call: bool) -> float:
    """Invert BS for sigma via brentq. Returns NaN if price ≤ intrinsic or no root."""
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if T <= 0 or price <= intrinsic + 1e-6:
        return float("nan")
    try:
        return brentq(lambda s: bs_price(S, K, T, s, r, is_call) - price, 1e-4, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return float("nan")


def front_back_ratio(front_iv: float, back_iv: float) -> float:
    if back_iv is None or back_iv <= 0 or front_iv != front_iv:
        return float("nan")
    return front_iv / back_iv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_iv.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/iv.py tests/dc_time_machine/test_iv.py
git commit -m "feat(dc): Black-Scholes, implied-vol inversion, front/back IV ratio"
```

---

## Task 5: Entry filters + overnight seam (`policy.py`)

**Files:**
- Create: `src_v2/backtest/dc_time_machine/policy.py`
- Test: `tests/dc_time_machine/test_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_policy.py
import pytest

from src_v2.backtest.dc_time_machine.policy import entry_allowed, overnight_decision


def test_always_filter_admits_everything():
    assert entry_allowed("always", {}) is True


def test_backwardation_filter_needs_ratio_above_one():
    assert entry_allowed("backwardation", {"ratio": 1.05}) is True
    assert entry_allowed("backwardation", {"ratio": 0.98}) is False


def test_fri_mon_filter_keys_on_flag():
    assert entry_allowed("fri_mon", {"is_fri_mon": True}) is True
    assert entry_allowed("fri_mon", {"is_fri_mon": False}) is False


def test_spike_filter_needs_elevated_and_reverting():
    assert entry_allowed("spike", {"ratio_z": 1.5, "ratio_falling": True}) is True
    assert entry_allowed("spike", {"ratio_z": 1.5, "ratio_falling": False}) is False
    assert entry_allowed("spike", {"ratio_z": 0.2, "ratio_falling": True}) is False


def test_overnight_close_policy_always_closes():
    assert overnight_decision("close", {"dc_pnl_frac": 0.05, "ratio": 1.2}) == "close"


def test_overnight_hold_policy_holds_only_when_shallow_loss_and_backwardated():
    assert overnight_decision("hold", {"dc_pnl_frac": -0.05, "ratio": 1.1}) == "hold"
    assert overnight_decision("hold", {"dc_pnl_frac": -0.15, "ratio": 1.1}) == "close"
    assert overnight_decision("hold", {"dc_pnl_frac": -0.05, "ratio": 0.9}) == "close"


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        entry_allowed("nope", {})
    with pytest.raises(ValueError):
        overnight_decision("nope", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_policy.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/policy.py
"""Entry filters and the overnight-hold decision seam.

`features` keys produced by the simulator: ratio, ratio_z, ratio_falling,
is_backwardation, is_fri_mon. `overnight_decision` is the swappable seam; the
v1 default is "close" (no overnight hold).
"""
from __future__ import annotations

from typing import Dict


def entry_allowed(filter_name: str, features: Dict) -> bool:
    if filter_name == "always":
        return True
    if filter_name == "backwardation":
        return features.get("ratio", 0.0) > 1.0
    if filter_name == "fri_mon":
        return bool(features.get("is_fri_mon", False))
    if filter_name == "spike":
        return features.get("ratio_z", 0.0) >= 1.0 and bool(features.get("ratio_falling", False))
    raise ValueError(f"unknown entry_filter: {filter_name!r}")


def overnight_decision(policy_name: str, state: Dict) -> str:
    """Return 'close' or 'hold' for an un-transformed DC at EOD."""
    if policy_name == "close":
        return "close"
    if policy_name == "hold":
        shallow = state.get("dc_pnl_frac", -1.0) > -0.10
        backwardated = state.get("ratio", 0.0) > 1.0
        return "hold" if (shallow and backwardated) else "close"
    raise ValueError(f"unknown overnight_policy: {policy_name!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_policy.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/policy.py tests/dc_time_machine/test_policy.py
git commit -m "feat(dc): entry filters + overnight decision seam"
```

---

## Task 6: Multi-expiration store (`multi_expiry_store.py`)

Holds `{session:{expiry:{(min,strike,is_call):(bid,ask)}}}`, with stale-tolerant lookup (ported from `CbboStore`), spot derivation, and parity-based settlement. Accepts a `preloaded` dict for tests (no Databento, no disk).

**Files:**
- Create: `src_v2/backtest/dc_time_machine/multi_expiry_store.py`
- Test: `tests/dc_time_machine/test_multi_expiry_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_multi_expiry_store.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore

S = date(2026, 2, 2)        # session
F = date(2026, 2, 11)       # front expiry
B = date(2026, 2, 12)       # back expiry


def _panel():
    # ATM ~5000; provide call & put at 5000 for spot/settlement parity
    return {
        S: {
            F: {
                ("09:40", 5000.0, True): (40.0, 41.0),
                ("09:40", 5000.0, False): (40.0, 41.0),
                ("09:41", 4980.0, False): (30.0, 31.0),
            },
            B: {
                ("09:40", 5000.0, True): (45.0, 46.0),
            },
        },
        # session == front expiry holds the 0DTE settlement panel for F
        F: {
            F: {
                ("15:59", 5000.0, True): (12.0, 12.0),
                ("15:59", 5000.0, False): (8.0, 8.0),
            },
        },
    }


def test_lookup_exact_and_stale():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    assert store.lookup(F, "09:40", 5000.0, True) == (40.0, 41.0)
    # 09:42 not present → walk back to 09:41
    assert store.lookup(F, "09:42", 4980.0, False, max_stale=5) == (30.0, 31.0)
    assert store.lookup(F, "09:40", 9999.0, True) is None


def test_available_expiries_sorted():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    assert store.available_expiries() == [F, B]


def test_derive_spot_from_parity():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    # call_mid == put_mid at 5000 → spot ≈ 5000
    assert store.derive_spot(F, "09:40", 5000.0) == pytest.approx(5000.0, abs=0.6)


def test_settlement_uses_expiry_own_session_panel():
    store = MultiExpiryStore(preloaded=_panel())
    store.load_session(S)
    # at F's last minute: 5000 + call_mid(12) − put_mid(8) = 5004
    assert store.settlement(F) == pytest.approx(5004.0, abs=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_multi_expiry_store.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/multi_expiry_store.py
"""Multi-expiration SPX option quote store for the DC backtest.

One pickle per session: {expiry_date: {(minute,strike,is_call): (bid,ask)}}.
The inner quote-map shape matches the existing CbboStore caches.
"""
from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]
SessionPanel = Dict[date, Dict[QuoteKey, QuoteVal]]


class MultiExpiryStore:
    def __init__(self, cache_dir: Optional[str | Path] = None,
                 preloaded: Optional[Dict[date, SessionPanel]] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: Optional[Dict[date, SessionPanel]] = preloaded
        self._panel: SessionPanel = {}
        self._session: Optional[date] = None

    def _raw(self, session: date) -> SessionPanel:
        if self._mem is not None and session in self._mem:
            return self._mem[session]
        if self.cache_dir:
            p = self.cache_dir / f"{session.isoformat()}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    return pickle.load(f)
        return {}

    def load_session(self, session: date) -> int:
        self._panel = self._raw(session)
        self._session = session
        return sum(len(qm) for qm in self._panel.values())

    def available_expiries(self) -> List[date]:
        return sorted(self._panel.keys())

    def strikes(self, expiry: date) -> set[float]:
        return {k[1] for k in self._panel.get(expiry, {})}

    def lookup(self, expiry: date, minute_et: str, strike: float, is_call: bool,
               max_stale: int = 5) -> Optional[QuoteVal]:
        qm = self._panel.get(expiry)
        if not qm:
            return None
        key = (minute_et, strike, is_call)
        if key in qm:
            return qm[key]
        hh, mm = int(minute_et[:2]), int(minute_et[3:5])
        for off in range(1, max_stale + 1):
            pmm, phh = mm - off, hh
            while pmm < 0:
                pmm += 60
                phh -= 1
            if phh < 0:
                break
            pkey = (f"{phh:02d}:{pmm:02d}", strike, is_call)
            if pkey in qm:
                return qm[pkey]
        return None

    def derive_spot(self, expiry: date, minute_et: str, ref: float,
                    max_stale: int = 5) -> Optional[float]:
        ks = self.strikes(expiry)
        if not ks:
            return None
        atm = min(ks, key=lambda k: abs(k - ref))
        cq = self.lookup(expiry, minute_et, atm, True, max_stale)
        pq = self.lookup(expiry, minute_et, atm, False, max_stale)
        if cq is None or pq is None:
            return None
        c_mid = (cq[0] + cq[1]) / 2
        p_mid = (pq[0] + pq[1]) / 2
        return round(atm + c_mid - p_mid, 2)

    def settlement(self, expiry: date) -> Optional[float]:
        """Cash settlement = parity forward at the expiry's own last loaded minute.

        Reads session == expiry directly (does not disturb the working session).
        """
        panel = self._raw(expiry)
        qm = panel.get(expiry)
        if not qm:
            return None
        last_min = max(k[0] for k in qm)
        best = None
        for strike in {k[1] for k in qm if k[0] == last_min}:
            cq = qm.get((last_min, strike, True))
            pq = qm.get((last_min, strike, False))
            if cq is None or pq is None:
                continue
            c_mid = (cq[0] + cq[1]) / 2
            p_mid = (pq[0] + pq[1]) / 2
            diff = abs(c_mid - p_mid)
            fwd = strike + c_mid - p_mid
            if best is None or diff < best[0]:
                best = (diff, fwd)
        return round(best[1], 2) if best else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_multi_expiry_store.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/multi_expiry_store.py tests/dc_time_machine/test_multi_expiry_store.py
git commit -m "feat(dc): multi-expiration quote store (lookup, spot, parity settlement)"
```

---

## Task 7: Trade simulator (`simulator.py`)

The heart: resolve expiries/strikes, enter the DC, minute-step the transform search with the −20% stop, apply the overnight seam at EOD, and settle the IC. Produces a `TradeResult`. Tested end-to-end against a synthetic store crafted to transform on a specific minute.

**Files:**
- Create: `src_v2/backtest/dc_time_machine/simulator.py`
- Test: `tests/dc_time_machine/test_simulator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_simulator.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import simulate_trade

S = date(2026, 2, 2)
F = date(2026, 2, 11)
B = date(2026, 2, 12)
MIN = "09:40"


def _build_panel(transform_minute="09:42", settle=5000.0):
    """Synthetic panel where the transform credit clears D+W at transform_minute.

    Strikes: K_p=4980 (put, ~35Δ proxy), K_c=5020 (call), wings at 4975 / 5025.
    Two minutes present: entry (09:40) and a later minute where back bids rise.
    """
    # Entry @ 09:40: D = backs@ask − fronts@bid
    entry = {
        # fronts (sold)
        ("09:40", 4980.0, False): (24.9, 25.1),   # front put bid 24.9
        ("09:40", 5020.0, True): (24.9, 25.1),    # front call bid 24.9
        # backs (bought)
        ("09:40", 4980.0, False, ): (24.9, 25.1),
    }
    # Build full quote sets for both expiries at two minutes
    def qm_front(minute, put_bid, call_bid):
        return {
            (minute, 4980.0, False): (put_bid, put_bid + 0.2),     # front put (short)
            (minute, 5020.0, True): (call_bid, call_bid + 0.2),    # front call (short)
            (minute, 4975.0, False): (0.40, 0.60),                 # put wing (buy)
            (minute, 5025.0, True): (0.40, 0.60),                  # call wing (buy)
            (minute, 5000.0, True): (40.0, 41.0),                  # ATM for spot
            (minute, 5000.0, False): (40.0, 41.0),
        }

    def qm_back(minute, put_bid, call_bid):
        return {
            (minute, 4980.0, False): (put_bid, put_bid + 0.2),     # back put (long)
            (minute, 5020.0, True): (call_bid, call_bid + 0.2),    # back call (long)
        }

    panel = {
        S: {
            F: {**qm_front("09:40", 25.0, 25.0), **qm_front("09:42", 25.0, 25.0)},
            B: {**qm_back("09:40", 25.5, 25.5), **qm_back("09:42", 31.0, 31.0)},
        },
        F: {  # settlement panel for F
            F: {
                ("15:59", round(settle), True): (max(settle - 5000.0, 0.0), max(settle - 5000.0, 0.0)),
                ("15:59", round(settle), False): (max(5000.0 - settle, 0.0), max(5000.0 - settle, 0.0)),
                ("15:59", 5000.0, True): (settle - 5000.0 if settle > 5000 else 0.0,) * 2,
                ("15:59", 5000.0, False): (5000.0 - settle if settle < 5000 else 0.0,) * 2,
            }
        },
    }
    return panel


def _params(**kw):
    base = dict(entry_minute=MIN, front_dte=1, gap=1, wing_width=5.0,
                fill_k=0.0, entry_filter="always", overnight_policy="close",
                target_delta=0.35)
    base.update(kw)
    return DcParams(**base)


def test_transform_fires_and_banks_positive():
    store = MultiExpiryStore(preloaded=_build_panel(settle=5000.0))
    res = simulate_trade(store, _params(), session=S, minutes=["09:40", "09:42"],
                         k_put=4980.0, k_call=5020.0, front=F, back=B)
    assert res.outcome == "transformed"
    assert res.transform_minute == "09:42"
    # D = backs@ask(25.7+25.7) − fronts@bid(25.0+25.0) = 1.4 ; C(09:42) net-mid
    #   = back bids(31.0+31.0) − wing asks(0.6+0.6) = 60.8 → bank huge, risk-free
    assert res.bank >= res.params_wing
    assert res.pnl > 0


def test_stop_loss_closes_when_dc_drops_20pct():
    panel = _build_panel()
    # Make the only later minute show a deep DC loss instead of a transform:
    panel[S][B][("09:42", 4980.0, False)] = (5.0, 5.2)   # back put bid collapses
    panel[S][B][("09:42", 5020.0, True)] = (5.0, 5.2)
    store = MultiExpiryStore(preloaded=panel)
    res = simulate_trade(store, _params(), session=S, minutes=["09:40", "09:42"],
                         k_put=4980.0, k_call=5020.0, front=F, back=B)
    assert res.outcome == "stopped"
    assert res.pnl < 0


def test_no_entry_when_filter_blocks():
    store = MultiExpiryStore(preloaded=_build_panel())
    res = simulate_trade(store, _params(entry_filter="fri_mon"), session=S,
                         minutes=["09:40", "09:42"], k_put=4980.0, k_call=5020.0,
                         front=F, back=B, is_fri_mon=False)
    assert res.outcome == "no_entry"
    assert res.pnl == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_simulator.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/simulator.py
"""Per-trade minute-stepped lifecycle for the DC Time Machine.

simulate_trade is deliberately given resolved strikes/expiries and an explicit
minute list so it is fully deterministic and unit-testable without Databento.
Higher layers (run_backtest / wf_cv) resolve those via resolve_expiries /
select_strikes / session_minutes and then call this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from src_v2.backtest.dc_time_machine.fills import fill_price, net_combo_credit
from src_v2.backtest.dc_time_machine.policy import entry_allowed, overnight_decision
from src_v2.backtest.dc_time_machine.pricing import (
    entry_debit, ic_terminal_pnl, is_risk_free,
)
from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore


@dataclass
class TradeResult:
    session: date
    front: date
    back: date
    k_put: float
    k_call: float
    outcome: str                  # transformed | stopped | closed_eod | held_overnight | no_entry
    entry_debit: float = 0.0
    credit: float = 0.0
    bank: float = 0.0
    transform_minute: Optional[str] = None
    settlement: Optional[float] = None
    pnl: float = 0.0              # final $/contract
    params_wing: float = 0.0
    tags: Dict = field(default_factory=dict)


def _q(store, expiry, minute, strike, is_call):
    return store.lookup(expiry, minute, strike, is_call)


def simulate_trade(store: MultiExpiryStore, params: DcParams, *, session: date,
                   minutes: List[str], k_put: float, k_call: float,
                   front: date, back: date,
                   features: Optional[Dict] = None, is_fri_mon: bool = False,
                   ratio: float = float("nan")) -> TradeResult:
    store.load_session(session)
    wing = params.wing_width
    k = params.fill_k
    feats = dict(features or {})
    feats.setdefault("is_fri_mon", is_fri_mon)
    feats.setdefault("ratio", ratio)
    res = TradeResult(session=session, front=front, back=back, k_put=k_put,
                      k_call=k_call, outcome="no_entry", params_wing=wing, tags=feats)

    if not entry_allowed(params.entry_filter, feats):
        return res

    em = params.entry_minute
    fpb = _q(store, front, em, k_put, False)      # front put (sell)
    fcb = _q(store, front, em, k_call, True)      # front call (sell)
    bpa = _q(store, back, em, k_put, False)       # back put (buy)
    bca = _q(store, back, em, k_call, True)       # back call (buy)
    if not all((fpb, fcb, bpa, bca)):
        return res

    # Entry debit with fills: sell fronts @ sell-fill, buy backs @ buy-fill
    D = entry_debit(
        front_put_bid=fill_price(*fpb, "sell", k),
        front_call_bid=fill_price(*fcb, "sell", k),
        back_put_ask=fill_price(*bpa, "buy", k),
        back_call_ask=fill_price(*bca, "buy", k),
    )
    res.entry_debit = D
    threshold = D + wing + params.transform_epsilon

    later = [m for m in minutes if m >= em]
    for m in later:
        # Mark DC to close: sell backs @ sell-fill, buy-to-close fronts @ buy-fill
        bp = _q(store, back, m, k_put, False)
        bc = _q(store, back, m, k_call, True)
        fp = _q(store, front, m, k_put, False)
        fc = _q(store, front, m, k_call, True)
        if not all((bp, bc, fp, fc)):
            continue
        v_close = (fill_price(*bp, "sell", k) + fill_price(*bc, "sell", k)
                   - fill_price(*fp, "buy", k) - fill_price(*fc, "buy", k))
        dc_pnl = v_close - D
        if D > 0 and dc_pnl <= -params.stop_loss_frac * D:
            res.outcome = "stopped"
            res.pnl = round(dc_pnl * 100, 2)
            return res

        # Transform credit (atomic combo): sell backs, buy front wings
        pw = _q(store, front, m, k_put - wing, False)
        cw = _q(store, front, m, k_call + wing, True)
        if pw is None or cw is None:
            continue
        legs = [(*bp, "sell"), (*bc, "sell"), (*pw, "buy"), (*cw, "buy")]
        C = net_combo_credit(legs, k)
        if C >= threshold:
            assert is_risk_free(D, C, wing), "transform fired without risk-free invariant"
            res.outcome = "transformed"
            res.transform_minute = m
            res.credit = C
            res.bank = C - D
            S = store.settlement(front)
            res.settlement = S
            if S is None:
                res.pnl = round(res.bank * 100, 2)   # conservative: no settlement → bank only
            else:
                res.pnl = round(ic_terminal_pnl(S, k_put, k_call, wing, res.bank) * 100, 2)
            return res

    # EOD, not transformed → overnight seam (v1 default: close)
    state = {"dc_pnl_frac": (v_close - D) / D if D else 0.0, "ratio": feats.get("ratio", 0.0)}
    if overnight_decision(params.overnight_policy, state) == "hold":
        res.outcome = "held_overnight"
        res.pnl = round((v_close - D) * 100, 2)   # marked; multi-day hold handled by caller
        return res
    res.outcome = "closed_eod"
    res.pnl = round((v_close - D) * 100, 2)
    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_simulator.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/simulator.py tests/dc_time_machine/test_simulator.py
git commit -m "feat(dc): minute-stepped trade simulator (entry, transform, stop, settle)"
```

---

## Task 8: Strike/expiry resolution + IV features (`simulator.py` additions)

Adds the helpers that turn a `DcParams` + a session into concrete strikes, expiries, the minute grid, and the IV-ratio features. Kept in `simulator.py` (they change together with the lifecycle).

**Files:**
- Modify: `src_v2/backtest/dc_time_machine/simulator.py` (append helpers)
- Test: `tests/dc_time_machine/test_resolution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_resolution.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import (
    resolve_expiries, select_strikes, compute_iv_features,
)

S = date(2026, 2, 2)
EXPS = [date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 11), date(2026, 2, 12)]


def _panel_for_resolution():
    front_strikes = {}
    for kk in range(4900, 5101, 5):
        # symmetric smile-free chain at 18% vol-ish prices is unnecessary; provide
        # monotone put/call mids so delta selection has a unique solution
        front_strikes[("09:40", float(kk), True)] = (max(5000 - kk, 0) + 20.0, max(5000 - kk, 0) + 21.0)
        front_strikes[("09:40", float(kk), False)] = (max(kk - 5000, 0) + 20.0, max(kk - 5000, 0) + 21.0)
    panel = {S: {e: dict(front_strikes) for e in EXPS}}
    return panel


def test_resolve_expiries_picks_front_and_back_by_index():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    # only 4 expiries exist; front_dte=3 → 3rd expiry after S (index 2) = 2026-02-11, gap 1 → 2026-02-12
    front, back = resolve_expiries(store, S, DcParams(front_dte=3, gap=1))
    assert front == date(2026, 2, 11)
    assert back == date(2026, 2, 12)


def test_resolve_expiries_returns_none_when_out_of_range():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    assert resolve_expiries(store, S, DcParams(front_dte=9, gap=1)) == (None, None)


def test_select_strikes_returns_put_below_call_above_spot():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    front = date(2026, 2, 11)
    k_put, k_call = select_strikes(store, S, front, DcParams(target_delta=0.35), spot=5000.0)
    assert k_put is not None and k_call is not None
    assert k_put < 5000.0 < k_call


def test_compute_iv_features_has_ratio_and_flags():
    store = MultiExpiryStore(preloaded=_panel_for_resolution())
    store.load_session(S)
    feats = compute_iv_features(store, S, front=date(2026, 2, 11), back=date(2026, 2, 12),
                                minute="09:40", spot=5000.0, params=DcParams())
    assert "ratio" in feats and "is_fri_mon" in feats
    assert feats["is_fri_mon"] is False   # 2026-02-11 is a Wednesday
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_resolution.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_expiries'`.

- [ ] **Step 3: Append the helpers to `simulator.py`**

```python
# --- append to src_v2/backtest/dc_time_machine/simulator.py ---
from typing import Tuple

from src_v2.backtest.dc_time_machine.iv import bs_delta, implied_vol, front_back_ratio


def resolve_expiries(store: MultiExpiryStore, session: date, params: DcParams
                     ) -> Tuple[Optional[date], Optional[date]]:
    """front = (front_dte)th available expiry strictly after session; back = front idx + gap."""
    exps = [e for e in store.available_expiries() if e > session]
    fi = params.front_dte - 1
    bi = fi + params.gap
    if fi < 0 or bi >= len(exps):
        return None, None
    return exps[fi], exps[bi]


def _T(session: date, expiry: date) -> float:
    return max((expiry - session).days, 1) / 365.0


def select_strikes(store: MultiExpiryStore, session: date, front: date,
                   params: DcParams, spot: float) -> Tuple[Optional[float], Optional[float]]:
    """Pick K_p (put, below spot) and K_c (call, above spot) nearest the target abs delta.

    Delta is computed with each strike's own implied vol at the entry minute; falls
    back to scanning available strikes. Returns (None, None) if no usable strikes.
    """
    em = params.entry_minute
    T = _T(session, front)
    ks = sorted(store.strikes(front))
    best_put = best_call = None
    bp_err = bc_err = 1e9
    for K in ks:
        for is_call in (True, False):
            q = store.lookup(front, em, K, is_call)
            if q is None:
                continue
            mid = (q[0] + q[1]) / 2
            iv = implied_vol(mid, spot, K, T, params.rate, is_call)
            if iv != iv:   # NaN
                continue
            d = bs_delta(spot, K, T, iv, params.rate, is_call)
            err = abs(abs(d) - params.target_delta)
            if is_call and K > spot and err < bc_err:
                bc_err, best_call = err, K
            if (not is_call) and K < spot and err < bp_err:
                bp_err, best_put = err, K
    return best_put, best_call


def compute_iv_features(store: MultiExpiryStore, session: date, front: date,
                        back: date, minute: str, spot: float, params: DcParams) -> Dict:
    """Reconstruct the front/back ATM IV ratio (the 'Flux' signal) + day-combo flags."""
    def atm_iv(expiry: date) -> float:
        ks = store.strikes(expiry)
        if not ks:
            return float("nan")
        atm = min(ks, key=lambda x: abs(x - spot))
        ivs = []
        T = _T(session, expiry)
        for is_call in (True, False):
            q = store.lookup(expiry, minute, atm, is_call)
            if q is None:
                continue
            mid = (q[0] + q[1]) / 2
            iv = implied_vol(mid, spot, atm, T, params.rate, is_call)
            if iv == iv:
                ivs.append(iv)
        return sum(ivs) / len(ivs) if ivs else float("nan")

    fr, bk = atm_iv(front), atm_iv(back)
    ratio = front_back_ratio(fr, bk)
    is_fri_mon = (session.weekday() == 4 or front.weekday() == 4) and back.weekday() == 0
    return {
        "front_iv": fr, "back_iv": bk, "ratio": ratio,
        "is_backwardation": (ratio == ratio and ratio > 1.0),
        "is_fri_mon": bool(is_fri_mon),
        "ratio_z": 0.0, "ratio_falling": False,   # filled by flux_scanner history when available
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_resolution.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/simulator.py tests/dc_time_machine/test_resolution.py
git commit -m "feat(dc): expiry/strike resolution + IV-ratio (Flux) features"
```

---

## Task 9: Databento backfill driver (`dc_backfill.py`)

Generalizes the existing `fetch_day_quotes` to a **specific expiry over a full session**, then a session writer. Mirrors the mock pattern in `tests/test_databento_backfill.py`.

**Files:**
- Create: `src_v2/backtest/dc_time_machine/dc_backfill.py`
- Test: `tests/dc_time_machine/test_dc_backfill.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_dc_backfill.py
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src_v2.backtest.dc_time_machine.dc_backfill import fetch_expiry_session


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    return df.set_index("ts_recv")


def test_fetch_expiry_session_uses_distinct_expiry_and_full_window():
    client = MagicMock()
    client.timeseries.get_range.return_value.to_df.return_value = _df([
        {"ts_recv": "2026-02-02T14:40:00Z", "symbol": "SPXW  260211P05000000", "bid_px_00": 40.0, "ask_px_00": 41.0},
        {"ts_recv": "2026-02-02T14:40:00Z", "symbol": "SPXW  260211C05000000", "bid_px_00": 40.0, "ask_px_00": 41.0},
    ])
    quotes = fetch_expiry_session(
        client, session_date=date(2026, 2, 2), expiry_date=date(2026, 2, 11),
        atm_strike=5000.0, half_width_pts=10, step_pts=5,
        start_et="09:30", end_et="16:00",
    )
    kw = client.timeseries.get_range.call_args.kwargs
    assert kw["dataset"] == "OPRA.PILLAR" and kw["schema"] == "cbbo-1m"
    # expiry (Feb 11) differs from session (Feb 2) → symbols carry 260211
    assert any("260211" in s for s in kw["symbols"])
    # full-day window: 09:30 ET == 14:30 UTC (EST), 16:00 ET == 21:00 UTC
    assert kw["start"] == "2026-02-02T14:30:00"
    assert kw["end"] == "2026-02-02T21:00:00"
    assert quotes[("09:40", 5000.0, False)] == (40.0, 41.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_dc_backfill.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/dc_backfill.py
"""Databento backfill for the DC backtest: full-day cbbo-1m for a SPECIFIC expiry.

Generalizes src_v2/backtest/databento_backfill.fetch_day_quotes by decoupling the
fetched `expiry_date` from the trade `session_date`, and widening the time window
to a full session. Writes one pickle per session: {expiry: {(min,strike,call):(bid,ask)}}.
"""
from __future__ import annotations

import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src_v2.backtest.osi_symbol import build_strike_window, parse_osi

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]


def _window_utc(session_date: date, start_et: str, end_et: str) -> Tuple[str, str]:
    def to_utc(hhmm: str) -> str:
        hh, mm = int(hhmm[:2]), int(hhmm[3:5])
        dt = datetime(session_date.year, session_date.month, session_date.day, hh, mm, tzinfo=ET)
        return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return to_utc(start_et), to_utc(end_et)


def fetch_expiry_session(client: Any, session_date: date, expiry_date: date,
                         atm_strike: float, half_width_pts: int = 80, step_pts: int = 5,
                         start_et: str = "09:30", end_et: str = "16:00") -> Dict[QuoteKey, QuoteVal]:
    """One Databento call → {(minute_et, strike, is_call): (bid, ask)} for `expiry_date`."""
    symbols = build_strike_window(expiry=expiry_date, atm_strike=atm_strike,
                                  half_width_pts=half_width_pts, step_pts=step_pts)
    start_iso, end_iso = _window_utc(session_date, start_et, end_et)
    resp = client.timeseries.get_range(
        dataset="OPRA.PILLAR", schema="cbbo-1m", symbols=symbols,
        stype_in="raw_symbol", start=start_iso, end=end_iso,
    )
    df = resp.to_df()
    out: Dict[QuoteKey, QuoteVal] = {}
    if df is None or df.empty:
        return out
    for ts_utc, row in df.iterrows():
        sym = str(row["symbol"]).rstrip()
        try:
            strike, is_call = parse_osi(f"{sym:<21}")
        except ValueError:
            continue
        bid_raw, ask_raw = row.get("bid_px_00"), row.get("ask_px_00")
        if bid_raw is None or ask_raw is None or pd.isna(bid_raw) or pd.isna(ask_raw):
            continue
        bid, ask = float(bid_raw), float(ask_raw)
        if bid <= 0 and ask <= 0:
            continue
        ts_et = ts_utc.tz_convert(ET)
        minute = f"{ts_et.hour:02d}:{ts_et.minute:02d}"
        out[(minute, strike, is_call)] = (bid, ask)
    return out


def backfill_session(client: Any, session_date: date, expiries: Iterable[date],
                     atm_strike: float, out_dir: str | Path,
                     half_width_pts: int = 80, step_pts: int = 5) -> Path:
    """Fetch each expiry's full-day panel and write {expiry: quotemap} to <out_dir>/<session>.pkl."""
    panel: Dict[date, Dict[QuoteKey, QuoteVal]] = {}
    for exp in expiries:
        panel[exp] = fetch_expiry_session(client, session_date, exp, atm_strike,
                                          half_width_pts=half_width_pts, step_pts=step_pts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session_date.isoformat()}.pkl"
    with open(path, "wb") as f:
        pickle.dump(panel, f)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_dc_backfill.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/dc_backfill.py tests/dc_time_machine/test_dc_backfill.py
git commit -m "feat(dc): Databento full-day per-expiry backfill driver"
```

---

## Task 10: Empirical half-spread study (`spreads.py`)

Measures the actual half-spread distribution from a loaded store, stratified by DTE bucket and time-of-day — the slippage "experiment" from the spec (§5.1).

**Files:**
- Create: `src_v2/backtest/dc_time_machine/spreads.py`
- Test: `tests/dc_time_machine/test_spreads.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_spreads.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.spreads import half_spread_stats

S = date(2026, 2, 2)
F = date(2026, 2, 11)   # 9 DTE


def _panel():
    return {S: {F: {
        ("09:40", 5000.0, True): (10.0, 10.4),   # half-spread 0.20
        ("09:40", 5000.0, False): (10.0, 10.6),  # half-spread 0.30
        ("14:00", 5000.0, True): (10.0, 11.0),   # half-spread 0.50
    }}}


def test_half_spread_stats_returns_percentiles_by_bucket():
    store = MultiExpiryStore(preloaded=_panel())
    stats = half_spread_stats(store, sessions=[S])
    key = (9, "open")   # 9 DTE, morning bucket
    assert key in stats
    assert stats[key]["median"] == pytest.approx(0.25, abs=1e-9)   # median of {0.20, 0.30}
    assert stats[key]["n"] == 2
    assert (9, "midday") in stats   # the 14:00 quote
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_spreads.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/spreads.py
"""Empirical half-spread study — the slippage experiment (spec §5.1).

Stratifies observed half-spreads by DTE bucket and time-of-day bucket so the fill
model's `k` is grounded in observed liquidity rather than assumed.
"""
from __future__ import annotations

import statistics
from datetime import date
from typing import Dict, List, Tuple

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore


def _tod_bucket(minute: str) -> str:
    hh = int(minute[:2])
    if hh < 11:
        return "open"
    if hh < 14:
        return "midday"
    return "close"


def half_spread_stats(store: MultiExpiryStore, sessions: List[date]
                      ) -> Dict[Tuple[int, str], Dict[str, float]]:
    """Return {(dte, tod_bucket): {median, p90, n}} of half-spreads ($/share)."""
    buckets: Dict[Tuple[int, str], List[float]] = {}
    for session in sessions:
        store.load_session(session)
        for expiry in store.available_expiries():
            dte = (expiry - session).days
            qm_keys = store.strikes  # noqa: not used; iterate panel directly below
            for (minute, strike, is_call), (bid, ask) in _iter_quotes(store, expiry):
                half = (ask - bid) / 2.0
                if half < 0:
                    continue
                buckets.setdefault((dte, _tod_bucket(minute)), []).append(half)
    out: Dict[Tuple[int, str], Dict[str, float]] = {}
    for key, vals in buckets.items():
        vals_sorted = sorted(vals)
        p90 = vals_sorted[min(len(vals_sorted) - 1, int(0.9 * len(vals_sorted)))]
        out[key] = {"median": statistics.median(vals_sorted), "p90": p90, "n": len(vals_sorted)}
    return out


def _iter_quotes(store: MultiExpiryStore, expiry: date):
    """Yield ((minute, strike, is_call), (bid, ask)) for one expiry of the loaded session."""
    panel = store._panel.get(expiry, {})   # internal access by design (same package)
    for key, val in panel.items():
        yield key, val
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_spreads.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/spreads.py tests/dc_time_machine/test_spreads.py
git commit -m "feat(dc): empirical half-spread study (DTE × time-of-day buckets)"
```

---

## Task 11: Reconstructed Flux scanner (`flux_scanner.py`)

Computes the front/back IV-ratio per (session, front, back) and ranks candidate entries by ratio change vs a short history — mirroring his scanner.

**Files:**
- Create: `src_v2/backtest/dc_time_machine/flux_scanner.py`
- Test: `tests/dc_time_machine/test_flux_scanner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_flux_scanner.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.flux_scanner import ratio_zscore, rank_candidates


def test_ratio_zscore_positive_when_today_above_history():
    z = ratio_zscore(today=1.20, history=[1.00, 1.02, 0.98, 1.01])
    assert z > 1.0


def test_ratio_zscore_zero_for_flat_history():
    import math
    assert math.isnan(ratio_zscore(1.0, [1.0, 1.0])) or ratio_zscore(1.0, [1.0, 1.0]) == 0.0


def test_rank_candidates_orders_by_ratio_drop_desc():
    cands = [
        {"session": date(2026, 2, 2), "ratio": 1.30, "ratio_prev": 1.40},  # dropped 0.10
        {"session": date(2026, 2, 3), "ratio": 1.10, "ratio_prev": 1.12},  # dropped 0.02
    ]
    ranked = rank_candidates(cands, by="drop")
    assert ranked[0]["session"] == date(2026, 2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_flux_scanner.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/flux_scanner.py
"""Reconstructed 'Flux' scan: rank candidate entries by front/back IV-ratio dynamics."""
from __future__ import annotations

import math
import statistics
from typing import Dict, List


def ratio_zscore(today: float, history: List[float]) -> float:
    """z-score of today's ratio vs its recent history. NaN if history too small."""
    if today != today or len(history) < 2:
        return float("nan")
    mu = statistics.mean(history)
    sd = statistics.pstdev(history)
    if sd == 0:
        return 0.0
    return (today - mu) / sd


def rank_candidates(candidates: List[Dict], by: str = "drop") -> List[Dict]:
    """Rank candidates. by='drop' → largest (ratio_prev − ratio) first (reverting down);
    by='rise' → largest (ratio − ratio_prev) first."""
    def delta(c: Dict) -> float:
        d = c.get("ratio_prev", c.get("ratio", 0.0)) - c.get("ratio", 0.0)
        return d if by == "drop" else -d
    return sorted(candidates, key=delta, reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_flux_scanner.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/flux_scanner.py tests/dc_time_machine/test_flux_scanner.py
git commit -m "feat(dc): reconstructed Flux scanner (ratio z-score + ranking)"
```

---

## Task 12: Session driver — one trade per session (`simulator.py` addition)

Wires resolution + features + lifecycle into `run_session`: given a store, params, and a session, produce a single `TradeResult` (or a `no_entry`).

**Files:**
- Modify: `src_v2/backtest/dc_time_machine/simulator.py` (append `session_minutes`, `run_session`)
- Test: `tests/dc_time_machine/test_run_session.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_run_session.py
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import run_session

S = date(2026, 2, 2)
F = date(2026, 2, 3)
B = date(2026, 2, 4)


def _panel():
    def front(minute, pb, cb):
        return {
            (minute, 4980.0, False): (pb, pb + 0.2),
            (minute, 5020.0, True): (cb, cb + 0.2),
            (minute, 4975.0, False): (0.4, 0.6),
            (minute, 5025.0, True): (0.4, 0.6),
            (minute, 5000.0, True): (40.0, 41.0),
            (minute, 5000.0, False): (40.0, 41.0),
            (minute, 4980.0, True): (45.0, 46.0),   # for delta scan
            (minute, 5020.0, False): (45.0, 46.0),
        }

    def back(minute, pb, cb):
        return {(minute, 4980.0, False): (pb, pb + 0.2), (minute, 5020.0, True): (cb, cb + 0.2)}

    return {
        S: {
            F: {**front("09:40", 25.0, 25.0), **front("09:42", 25.0, 25.0)},
            B: {**back("09:40", 25.5, 25.5), **back("09:42", 31.0, 31.0)},
        },
        F: {F: {("15:59", 5000.0, True): (5.0, 5.0), ("15:59", 5000.0, False): (5.0, 5.0)}},
    }


def test_run_session_produces_a_trade():
    store = MultiExpiryStore(preloaded=_panel())
    params = DcParams(entry_minute="09:40", front_dte=1, gap=1, wing_width=5.0, fill_k=0.0)
    res = run_session(store, params, S)
    assert res is not None
    assert res.outcome in {"transformed", "closed_eod", "stopped", "held_overnight"}
    assert res.front == F and res.back == B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_run_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_session'`.

- [ ] **Step 3: Append to `simulator.py`**

```python
# --- append to src_v2/backtest/dc_time_machine/simulator.py ---

def session_minutes(store: MultiExpiryStore, front: date) -> List[str]:
    """Sorted unique minute strings present for the front expiry of the loaded session."""
    return sorted({k[0] for k in store._panel.get(front, {})})


def run_session(store: MultiExpiryStore, params: DcParams, session: date
                ) -> Optional[TradeResult]:
    """Resolve a single DC trade for `session` and simulate it. None if no expiries."""
    store.load_session(session)
    front, back = resolve_expiries(store, session, params)
    if front is None or back is None:
        return None
    spot = store.derive_spot(front, params.entry_minute, ref=_panel_ref(store, front))
    if spot is None:
        return TradeResult(session=session, front=front, back=back, k_put=0.0,
                           k_call=0.0, outcome="no_entry", params_wing=params.wing_width)
    k_put, k_call = select_strikes(store, session, front, params, spot)
    if k_put is None or k_call is None:
        return TradeResult(session=session, front=front, back=back, k_put=0.0,
                           k_call=0.0, outcome="no_entry", params_wing=params.wing_width)
    feats = compute_iv_features(store, session, front, back, params.entry_minute, spot, params)
    minutes = session_minutes(store, front)
    return simulate_trade(store, params, session=session, minutes=minutes,
                          k_put=k_put, k_call=k_call, front=front, back=back,
                          features=feats, is_fri_mon=feats["is_fri_mon"],
                          ratio=feats["ratio"])


def _panel_ref(store: MultiExpiryStore, front: date) -> float:
    """A rough ATM reference = median available strike for `front` (for spot derivation)."""
    ks = sorted(store.strikes(front))
    return ks[len(ks) // 2] if ks else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_run_session.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/simulator.py tests/dc_time_machine/test_run_session.py
git commit -m "feat(dc): session driver (resolve + features + simulate → one trade)"
```

---

## Task 13: Dedicated WF-CV runner (`wf_cv.py`)

Replicates the project's WF-CV methodology (ratio 5:1, 4 folds, annualized Sharpe) over the DC simulator — decoupled from the 0DTE-bound `WalkForwardOptimizer`. Pure-Python fold split + grid eval; tested on a tiny synthetic day set.

**Files:**
- Create: `src_v2/backtest/dc_time_machine/wf_cv.py`
- Test: `tests/dc_time_machine/test_wf_cv.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_wf_cv.py
from datetime import date, timedelta

import pytest

from src_v2.backtest.dc_time_machine.wf_cv import split_folds, annualized_sharpe


def test_split_folds_ratio_and_count():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    folds = split_folds(days, ratio=5, n_folds=4)
    assert len(folds) == 4
    for train, val in folds:
        # each fold: train ≈ 5× val, train precedes val, no overlap
        assert len(train) >= 4 * len(val)
        assert max(train) < min(val)


def test_annualized_sharpe_positive_for_steady_wins():
    pnls = [100.0, 120.0, 90.0, 110.0, 105.0]
    assert annualized_sharpe(pnls) > 0


def test_annualized_sharpe_zero_for_empty_or_flat():
    assert annualized_sharpe([]) == 0.0
    assert annualized_sharpe([50.0, 50.0]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_wf_cv.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/wf_cv.py
"""Dedicated walk-forward CV for the DC simulator.

Mirrors the project methodology (ratio 5:1, 4 folds, annualized Sharpe) but calls
run_session directly instead of the 0DTE-bound WalkForwardOptimizer. Grid search on
train, evaluate the top combo out-of-sample on val; winner = most val-robust combo.
"""
from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Tuple

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.simulator import run_session

TRADING_DAYS = 252


def split_folds(days: List[date], ratio: int = 5, n_folds: int = 4
                ) -> List[Tuple[List[date], List[date]]]:
    """Contiguous, non-overlapping folds; each train ≈ ratio× its val, train before val."""
    days = sorted(days)
    block = len(days) // n_folds
    folds: List[Tuple[List[date], List[date]]] = []
    for i in range(n_folds):
        seg = days[i * block: (i + 1) * block] if i < n_folds - 1 else days[i * block:]
        if len(seg) < ratio + 1:
            continue
        val_len = max(1, len(seg) // (ratio + 1))
        train, val = seg[:-val_len], seg[-val_len:]
        if train and val:
            folds.append((train, val))
    return folds


def annualized_sharpe(pnls: List[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    sd = statistics.pstdev(pnls)
    if sd == 0:
        return 0.0
    return (statistics.mean(pnls) / sd) * math.sqrt(TRADING_DAYS)


def _eval(store: MultiExpiryStore, params: DcParams, days: List[date]) -> Dict[str, float]:
    pnls: List[float] = []
    for d in days:
        res = run_session(store, params, d)
        if res is not None and res.outcome != "no_entry":
            pnls.append(res.pnl)
    return {"sharpe": annualized_sharpe(pnls), "pnl": sum(pnls), "n": len(pnls)}


def _combos(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


@dataclass
class WfFold:
    train_sharpe: float
    val_sharpe: float
    val_pnl: float
    best: Dict[str, Any]


@dataclass
class WfReport:
    folds: List[WfFold] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    mean_val_sharpe: float = 0.0
    n_profitable: int = 0


def run_wf_cv(store: MultiExpiryStore, days: List[date], grid: Dict[str, List[Any]],
              base: Dict[str, Any] | None = None, ratio: int = 5, n_folds: int = 4) -> WfReport:
    base = base or {}
    combos = _combos(grid)
    folds = split_folds(days, ratio, n_folds)
    rep = WfReport()
    winners: Dict[Tuple, int] = {}
    val_sharpes: List[float] = []
    for train, val in folds:
        scored = [(c, _eval(store, DcParams(**{**base, **c}), train)["sharpe"]) for c in combos]
        best_combo = max(scored, key=lambda x: x[1])[0]
        vm = _eval(store, DcParams(**{**base, **best_combo}), val)
        rep.folds.append(WfFold(train_sharpe=max(s for _, s in scored),
                                val_sharpe=vm["sharpe"], val_pnl=vm["pnl"], best=best_combo))
        winners[tuple(sorted(best_combo.items()))] = winners.get(tuple(sorted(best_combo.items())), 0) + 1
        val_sharpes.append(vm["sharpe"])
        if vm["pnl"] > 0:
            rep.n_profitable += 1
    if rep.folds:
        rep.best_params = dict(max(winners.items(), key=lambda x: x[1])[0])
        rep.mean_val_sharpe = statistics.mean(val_sharpes)
    return rep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_wf_cv.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/dc_time_machine/wf_cv.py tests/dc_time_machine/test_wf_cv.py
git commit -m "feat(dc): dedicated WF-CV runner (5:1 / 4-fold, annualized Sharpe)"
```

---

## Task 14: CLI runner (`run_backtest.py`)

Thin argparse front-end wiring the subcommands. The `simulate` subcommand runs the favorite config over a cache dir and writes `trades.csv`; `crosscheck` filters to Feb–Apr 2026 and prints aggregate stats vs his published numbers. Tested at the arg-parsing + `trades.csv` writer level (no Databento).

**Files:**
- Create: `src_v2/backtest/dc_time_machine/run_backtest.py`
- Test: `tests/dc_time_machine/test_run_backtest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dc_time_machine/test_run_backtest.py
import csv
from datetime import date

from src_v2.backtest.dc_time_machine.simulator import TradeResult
from src_v2.backtest.dc_time_machine.run_backtest import write_trades_csv, summarize


def test_write_trades_csv_roundtrips(tmp_path):
    trades = [
        TradeResult(session=date(2026, 2, 2), front=date(2026, 2, 11), back=date(2026, 2, 12),
                    k_put=4980.0, k_call=5020.0, outcome="transformed", entry_debit=1.4,
                    credit=60.8, bank=59.4, transform_minute="09:42", settlement=5000.0, pnl=515.0),
        TradeResult(session=date(2026, 2, 3), front=date(2026, 2, 12), back=date(2026, 2, 13),
                    k_put=4970.0, k_call=5030.0, outcome="closed_eod", entry_debit=1.5, pnl=-20.0),
    ]
    out = tmp_path / "trades.csv"
    write_trades_csv(trades, out)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert rows[0]["outcome"] == "transformed"
    assert float(rows[0]["pnl"]) == 515.0


def test_summarize_computes_winrate_and_pf():
    trades = [
        TradeResult(session=date(2026, 2, 2), front=date(2026, 2, 11), back=date(2026, 2, 12),
                    k_put=0, k_call=0, outcome="transformed", pnl=515.0),
        TradeResult(session=date(2026, 2, 3), front=date(2026, 2, 12), back=date(2026, 2, 13),
                    k_put=0, k_call=0, outcome="closed_eod", pnl=-20.0),
    ]
    s = summarize(trades)
    assert s["n"] == 2
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] == 515.0 / 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dc_time_machine/test_run_backtest.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/backtest/dc_time_machine/run_backtest.py
"""CLI for the DC Time Machine backtest.

Subcommands:
  backfill       Fetch multi-expiry full-day cbbo-1m into a cache dir (needs Databento).
  study-spreads  Print the empirical half-spread table from a cache dir.
  simulate       Run a config over a cache dir → trades.csv + summary.
  scan           Reconstructed Flux scan over a cache dir.
  wf-cv          Walk-forward CV over a param grid.
  crosscheck     Simulate Feb–Apr 2026 and print aggregate stats vs his published numbers.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import TradeResult, run_session
from src_v2.backtest.dc_time_machine.strategy import DcParams

_CSV_FIELDS = ["session", "front", "back", "k_put", "k_call", "outcome", "entry_debit",
               "credit", "bank", "transform_minute", "settlement", "pnl"]


def write_trades_csv(trades: List[TradeResult], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for t in trades:
            w.writerow({k: getattr(t, k) for k in _CSV_FIELDS})


def summarize(trades: List[TradeResult]) -> Dict[str, float]:
    real = [t for t in trades if t.outcome != "no_entry"]
    wins = [t.pnl for t in real if t.pnl > 0]
    losses = [t.pnl for t in real if t.pnl < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n": len(real),
        "win_rate": (len(wins) / len(real)) if real else 0.0,
        "total_pnl": sum(t.pnl for t in real),
        "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf"),
        "n_transformed": sum(1 for t in real if t.outcome == "transformed"),
    }


def _sessions_in_cache(cache_dir: Path) -> List[date]:
    return sorted(date.fromisoformat(p.stem) for p in cache_dir.glob("*.pkl"))


def _simulate_range(cache_dir: Path, params: DcParams,
                    lo: date | None = None, hi: date | None = None) -> List[TradeResult]:
    store = MultiExpiryStore(cache_dir=cache_dir)
    out: List[TradeResult] = []
    for s in _sessions_in_cache(cache_dir):
        if (lo and s < lo) or (hi and s > hi):
            continue
        res = run_session(store, params, s)
        if res is not None:
            out.append(res)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dc-backtest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate")
    sim.add_argument("--cache", required=True)
    sim.add_argument("--out", default="results/dc_time_machine/trades.csv")
    sim.add_argument("--fill-k", type=float, default=1.0)

    cc = sub.add_parser("crosscheck")
    cc.add_argument("--cache", required=True)
    cc.add_argument("--fill-k", type=float, default=1.0)

    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        params = DcParams(fill_k=args.fill_k)
        trades = _simulate_range(Path(args.cache), params)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_trades_csv(trades, out)
        print(summarize(trades))
        return 0

    if args.cmd == "crosscheck":
        params = DcParams(fill_k=args.fill_k)
        trades = _simulate_range(Path(args.cache), params, date(2026, 2, 1), date(2026, 4, 30))
        s = summarize(trades)
        print("Feb–Apr 2026 cross-check:", s)
        print("His published: ~106 trades, ~63% win, ~25% max-profit-hit")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/dc_time_machine/test_run_backtest.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the whole suite + commit**

Run: `.venv/bin/pytest tests/dc_time_machine/ -v`
Expected: all green.

```bash
git add src_v2/backtest/dc_time_machine/run_backtest.py tests/dc_time_machine/test_run_backtest.py
git commit -m "feat(dc): CLI runner (simulate/crosscheck) + trades.csv + summary"
```

---

## Task 15: Real backfill + the verdict (operational, not unit-tested)

This task spends real Databento budget and produces the actual research output. It is **manual/operational** — guarded behind the CLI, run once, reviewed by a human.

- [ ] **Step 1: Tiered backfill — favorite config, full date range**

Implement the `backfill` subcommand body in `run_backtest.py` (wiring `dc_backfill.backfill_session` over the NYSE calendar, resolving each session's needed expiries via `resolve_expiries`'s expiry logic and ATM from `spy_daily`/parity). Run:

```bash
.venv/bin/python -m src_v2.backtest.dc_time_machine.run_backtest backfill \
  --start 2022-05-01 --end 2026-04-30 --config favorite \
  --out data/dc_cache --half-width 80
```

Expected: `data/dc_cache/*.pkl`, ~2 expiries/session, ~$25 Databento spend (watch `data/databento_cost_ledger.jsonl`).

- [ ] **Step 2: Empirical spread study**

```bash
.venv/bin/python -m src_v2.backtest.dc_time_machine.run_backtest study-spreads --cache data/dc_cache
```

Record the median/p90 half-spreads by DTE × time-of-day into the README. This calibrates the `k` to report as "realistic".

- [ ] **Step 3: His-numbers cross-check**

```bash
.venv/bin/python -m src_v2.backtest.dc_time_machine.run_backtest crosscheck --cache data/dc_cache --fill-k <realistic_k>
```

Compare trade count / win-rate / max-profit-hit to ~106 / ~63% / ~25%. Investigate order-of-magnitude divergence before trusting anything.

- [ ] **Step 4: WF-CV + fill-k sweep**

Run `wf-cv` across the grid (delta, front_dte, gap, wing, entry_filter, overnight_policy) at `fill_k ∈ {0.0, 0.5, 1.0}`. Record OOS Sharpe per fold and the always-enter-vs-filter lift.

- [ ] **Step 5: Write the verdict README + decide GO/NO-GO**

Create `src_v2/backtest/dc_time_machine/README.md` answering: profitable under realistic fills? where is the edge (structural carry vs timing vs neither)? what is load-bearing? Then widen the backfill to the full DTE/gap sweep **only if** the favorite-config verdict is promising.

```bash
git add src_v2/backtest/dc_time_machine/README.md docs/ results/dc_time_machine/
git commit -m "docs(dc): backfill, spread study, cross-check, WF-CV verdict"
```

---

## Self-Review

**1. Spec coverage** (each spec section → task):
- §2 strategy (debit/credit/transform/invariant) → Tasks 2, 7. ✓
- §3 architecture / file layout → File Structure + all tasks. ✓
- §4 overnight seam (default close, counterfactual) → Task 5 (`overnight_decision`), Task 7 (wired), Task 15 step 4 (counterfactual via grid). ✓
- §5 fills (empirical study, k-sweep, combo vs leg) → Tasks 3 (model), 10 (study), 15 (sweep). Leg-by-leg ≈ `fill_k=0` optimistic bound (documented in Task 3 / spec §5.3). ✓
- §5.0 baseline entry time → `DcParams.entry_minute` (Task 1), used in Task 7/8. ✓
- §6 Flux / structural-vs-timing separated, scanner, always-enter baseline → Tasks 5 (filters), 8 (features), 11 (scanner), 15 (lift). ✓
- §7 validation (cross-check, WF-CV, sizing) → Tasks 13, 14, 15. **Gap:** fixed-fractional sizing analysis is described in the spec but has no dedicated task. **Resolution:** sizing is a post-hoc analysis on `trades.csv` reusing the existing MEIC `sizing_montecarlo.py` pattern; it is deferred to Task 15's verdict step and explicitly noted there rather than re-implemented. Acceptable for a GO/NO-GO Phase 1.
- §9 data layer / backfill / cost → Tasks 6, 9, 15. ✓
- §10 Phase 2 → out of scope (preview only). ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"handle edge cases" in code steps; every code step shows complete code. Task 15 is explicitly operational (not TDD) and labeled as such. ✓

**3. Type consistency:** `DcParams` field names (`entry_minute`, `target_delta`, `front_dte`, `gap`, `wing_width`, `fill_k`, `entry_filter`, `overnight_policy`, `transform_epsilon`, `stop_loss_frac`, `rate`) are used identically in Tasks 1, 7, 8, 12, 13, 14. `TradeResult` field names match between Task 7 (definition), Task 12 (construction), Task 14 (CSV writer `_CSV_FIELDS`). `fill_price(bid, ask, side, k)` / `net_combo_credit(legs, k)` signatures consistent across Tasks 3, 7. `lookup(expiry, minute, strike, is_call, max_stale)` consistent across Tasks 6, 7, 8, 10. ✓

**Note on `store._panel` internal access** (Tasks 10, 12): same-package access is intentional (the store and its analyzers are one unit); if a public iterator is preferred, add `MultiExpiryStore.iter_expiry(expiry)` in Task 6 and swap call sites — non-blocking.
