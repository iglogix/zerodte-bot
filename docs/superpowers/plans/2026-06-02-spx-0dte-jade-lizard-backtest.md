# SPX 0DTE Jade Lizard Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest the TastyTrade 9:30am 0DTE SPX Jade Lizard on real SPXW quotes with 10-minute, mid-price evaluation, and report how it performs.

**Architecture:** A pure, unit-tested core (`src_v2/backtest/jl_core.py`) computes EM, strikes, entry credit, the whole-position 50% take-profit across 10-minute anchors, and the exit value (combined bid/ask mid at the exit anchor; spot-intrinsic as a cross-check). A CLI evaluator (`scripts/analytical_jl_eval.py`) loads a CBBO cache and produces the report. A targeted Databento backfill (`scripts/backfill_jl_cbbo.py`, reusing `src_v2/backtest/meic_backfill.py`) builds a purpose-made SPXW cache `data/cbbo_cache_jl_10min/` with a 9:30→16:00 minute window at ±100 strikes, centered on the SPY-derived 9:30 spot. A free dry-run cost estimate gates the spend.

**Tech Stack:** Python 3, `databento`, `pandas`, `pytest`. Reuses `BarLoader`, `meic_backfill` helpers, `databento_cost` (`estimate_request_cost`, `CostLedger`). Pickle cache schema `{(minute_et, strike, is_call): (bid, ask)}` (same as all existing caches).

**Spec:** [docs/superpowers/specs/2026-06-02-spx-0dte-jade-lizard-backtest-design.md](../specs/2026-06-02-spx-0dte-jade-lizard-backtest-design.md)

**Key design decisions (locked):**
- EM = ATM straddle mid at entry. ATM is found from the chain via `argmin|call_mid − put_mid|` (no external spot). Forward = `atm + (call_mid − put_mid)`.
- Short put at `floor((forward − 0.25×EM)/5)×5`, long put −30. Short call at `ceil((forward + 0.25×EM)/5)×5`, long call +10. (Symmetric 0.25×EM placement, asymmetric 30/10 widths.)
- Exit: whole-position. Close at the first 10-min anchor where `combined_mark ≤ 0.50 × total_credit`. No stop-loss; otherwise hold to the final anchor.
- **Exit value = combined bid/ask mid at the exit anchor** (headline). Spot-intrinsic from SPY×10 close is computed as a separate cross-check column.
- Headline P&L: zero execution cost (mid fills, no slippage, no commission). A with-costs sensitivity (slippage 0.10/spread entry+exit, commission $1/contract/leg at open, $0 close) is a second column.
- 1 contract/leg, multiplier 100.

---

## File Structure

| File | Responsibility |
|---|---|
| `src_v2/backtest/jl_core.py` (create) | Pure JL logic: quote mid w/ fallback, ATM/forward/EM from chain, strike selection, entry credit, combined mark, whole-position TP + exit valuation, per-day evaluate, aggregate metrics. No I/O, no Databento, no bars. |
| `tests/test_jl_core.py` (create) | Unit tests for every `jl_core` function using small synthetic quote dicts. |
| `scripts/analytical_jl_eval.py` (create) | CLI: load cache + bars → `by_date` → run `jl_core.evaluate` → print headline / with-costs / intrinsic cross-check / exit breakdown; write per-trade CSV to `results/jade_lizard/`. |
| `scripts/backfill_jl_cbbo.py` (create) | Targeted SPXW backfill: 9:30→16:00 minute window, ±100 strikes centered on SPY-derived 9:30 spot, dry-run cost gate, write `data/cbbo_cache_jl_10min/<date>.pkl`. Reuses `meic_backfill` + `databento_cost`. |
| `docs/spx_0dte_jade_lizard_findings.md` (create, Task 8) | Results writeup. |
| `data/cbbo_cache_jl_10min/` (created by backfill) | New cache dir. Never touches shared caches. |
| `results/jade_lizard/` (created by eval) | Per-trade CSV + summary JSON. |

Tests go in `tests/` (mutable). Never modify `tests_locked/`.

---

## Task 1: jl_core — quote helpers, ATM / forward / EM from chain

**Files:**
- Create: `src_v2/backtest/jl_core.py`
- Test: `tests/test_jl_core.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jl_core.py
from src_v2.backtest import jl_core as jl


def _chain_0930():
    """Synthetic 09:30 chain: ATM at 5000 (call_mid == put_mid == 10.5)."""
    return {
        ("09:30", 4990.0, True): (15.0, 16.0),  ("09:30", 4990.0, False): (5.0, 6.0),
        ("09:30", 4995.0, True): (12.0, 13.0),  ("09:30", 4995.0, False): (7.0, 8.0),
        ("09:30", 5000.0, True): (10.0, 11.0),  ("09:30", 5000.0, False): (10.0, 11.0),
        ("09:30", 5005.0, True): (8.0, 9.0),    ("09:30", 5005.0, False): (12.0, 13.0),
        ("09:30", 5010.0, True): (6.0, 7.0),    ("09:30", 5010.0, False): (15.0, 16.0),
    }


def test_mid_handles_one_sided_and_empty():
    assert jl._mid(10.0, 11.0) == 10.5
    assert jl._mid(0.0, 4.0) == 4.0   # bid missing -> ask
    assert jl._mid(3.0, 0.0) == 3.0   # ask missing -> bid
    assert jl._mid(0.0, 0.0) is None  # both missing


def test_quote_mid_exact_and_fallback():
    q = _chain_0930()
    assert jl.quote_mid(q, "09:30", 5000.0, True) == 10.5
    # 09:33 missing -> falls back to 09:30 within +/-5 min
    assert jl.quote_mid(q, "09:33", 5000.0, True) == 10.5
    # nothing within window
    assert jl.quote_mid(q, "12:00", 5000.0, True) is None


def test_atm_em_forward_from_chain():
    q = _chain_0930()
    assert jl.atm_from_chain(q, "09:30") == 5000.0
    assert jl.compute_em(q, "09:30", 5000.0) == 21.0          # 10.5 + 10.5
    assert jl.forward_from_chain(q, "09:30", 5000.0) == 5000.0  # 5000 + (10.5-10.5)


def test_make_anchors_10min():
    a = jl.make_anchors("09:30", "16:00", 10)
    assert a[0] == "09:30" and a[1] == "09:40" and a[-1] == "16:00"
    assert len(a) == 40   # 09:30..16:00 inclusive, step 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jl_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src_v2.backtest.jl_core'`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/jl_core.py
"""Pure logic for the 0DTE SPX Jade Lizard backtest.

No I/O, no Databento, no bars — everything operates on an in-memory quote dict
{(minute_et, strike, is_call): (bid, ask)} so it is fully unit-testable.

Pricing is chain-derived: ATM = argmin|call_mid - put_mid|, forward = atm +
(call_mid - put_mid), EM = ATM straddle mid. Exit value = combined bid/ask mid
at the exit anchor (TastyTrade-comparable "exercise at mid").
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

QuoteKey = Tuple[str, float, bool]
Quotes = Dict[QuoteKey, Tuple[float, float]]


def make_anchors(start: str = "09:30", end: str = "16:00", step_min: int = 10) -> List[str]:
    t = int(start[:2]) * 60 + int(start[3:5])
    e = int(end[:2]) * 60 + int(end[3:5])
    out: List[str] = []
    while t <= e:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += step_min
    return out


def _mid(bid: float, ask: float) -> Optional[float]:
    if bid <= 0 and ask <= 0:
        return None
    if bid <= 0:
        return ask
    if ask <= 0:
        return bid
    return (bid + ask) / 2


def quote_mid(quotes: Quotes, minute: str, strike: float, is_call: bool,
              fallback_min: int = 5) -> Optional[float]:
    q = quotes.get((minute, strike, is_call))
    if q is None and fallback_min > 0:
        hh, mm = int(minute[:2]), int(minute[3:5])
        offsets: List[int] = []
        for d in range(1, fallback_min + 1):
            offsets += [-d, d]
        for off in offsets:
            nmm, nhh = mm + off, hh
            while nmm < 0:
                nmm += 60
                nhh -= 1
            while nmm >= 60:
                nmm -= 60
                nhh += 1
            if nhh < 0 or nhh > 23:
                continue
            q = quotes.get((f"{nhh:02d}:{nmm:02d}", strike, is_call))
            if q is not None:
                break
    if q is None:
        return None
    return _mid(*q)


def atm_from_chain(quotes: Quotes, minute: str) -> Optional[float]:
    """ATM strike = the one minimizing |call_mid - put_mid| (no external spot)."""
    strikes = sorted({k[1] for k in quotes if k[0] == minute})
    best: Optional[Tuple[float, float]] = None
    for s in strikes:
        c = quote_mid(quotes, minute, s, True, 0)
        p = quote_mid(quotes, minute, s, False, 0)
        if c is None or p is None:
            continue
        diff = abs(c - p)
        if best is None or diff < best[1]:
            best = (s, diff)
    return best[0] if best else None


def compute_em(quotes: Quotes, minute: str, atm: float) -> Optional[float]:
    c = quote_mid(quotes, minute, atm, True)
    p = quote_mid(quotes, minute, atm, False)
    if c is None or p is None:
        return None
    return c + p


def forward_from_chain(quotes: Quotes, minute: str, atm: float) -> Optional[float]:
    c = quote_mid(quotes, minute, atm, True)
    p = quote_mid(quotes, minute, atm, False)
    if c is None or p is None:
        return None
    return atm + (c - p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jl_core.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/jl_core.py tests/test_jl_core.py
git commit -m "feat(jl_core): chain-derived ATM/EM/forward + quote helpers"
```

---

## Task 2: jl_core — config, strike selection, entry credit, combined mark

**Files:**
- Modify: `src_v2/backtest/jl_core.py`
- Test: `tests/test_jl_core.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_jl_core.py

def _legs_for_default_cfg():
    """With forward=5000, em=21, em_mult=0.25 -> offset 5.25:
       sp=floor(4994.75/5)*5=4990, lp=4960, sc=ceil(5005.25/5)*5=5010, lc=5020."""
    return (4990.0, 4960.0, 5010.0, 5020.0)


def test_select_strikes_default():
    cfg = jl.JlConfig()
    assert jl.select_strikes(5000.0, 21.0, cfg) == _legs_for_default_cfg()


def test_entry_credit():
    cfg = jl.JlConfig()
    q = {
        ("09:30", 4990.0, False): (5.0, 6.0),   # short put mid 5.5
        ("09:30", 4960.0, False): (2.0, 3.0),   # long put  mid 2.5 -> put credit 3.0
        ("09:30", 5010.0, True): (6.0, 7.0),    # short call mid 6.5
        ("09:30", 5020.0, True): (3.0, 4.0),    # long call mid 3.5 -> call credit 3.0
    }
    c = jl.entry_credit(q, "09:30", _legs_for_default_cfg(), cfg)
    assert c["put_credit"] == 3.0
    assert c["call_credit"] == 3.0
    assert c["total_credit"] == 6.0


def test_entry_credit_none_when_leg_missing_or_inverted():
    cfg = jl.JlConfig()
    q = {("09:30", 4990.0, False): (5.0, 6.0)}  # only one leg present
    assert jl.entry_credit(q, "09:30", _legs_for_default_cfg(), cfg) is None


def test_combined_mark():
    legs = _legs_for_default_cfg()
    q = {
        ("10:00", 4990.0, False): (1.4, 1.6),   # 1.5
        ("10:00", 4960.0, False): (0.4, 0.6),   # 0.5 -> put mark 1.0
        ("10:00", 5010.0, True): (1.9, 2.1),    # 2.0
        ("10:00", 5020.0, True): (0.4, 0.6),    # 0.5 -> call mark 1.5
    }
    assert jl.combined_mark(q, "10:00", legs) == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jl_core.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'JlConfig'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src_v2/backtest/jl_core.py

@dataclass
class JlConfig:
    entry_anchor: str = "09:30"
    anchors: List[str] = field(default_factory=lambda: make_anchors("09:30", "16:00", 10))
    em_mult_put: float = 0.25
    em_mult_call: float = 0.25
    put_width: float = 30.0
    call_width: float = 10.0
    tp_pct: float = 0.50          # exit when combined mark <= tp_pct * total_credit
    step: float = 5.0
    multiplier: int = 100
    slippage_per_spread: float = 0.0          # headline = 0
    commission_per_contract_per_leg: float = 0.0
    exit_mode: str = "mid"        # "mid" (mark at exit anchor) or "intrinsic"


def select_strikes(forward: float, em: float, cfg: JlConfig
                   ) -> Tuple[float, float, float, float]:
    step = cfg.step
    short_put = math.floor((forward - cfg.em_mult_put * em) / step) * step
    long_put = short_put - cfg.put_width
    short_call = math.ceil((forward + cfg.em_mult_call * em) / step) * step
    long_call = short_call + cfg.call_width
    return short_put, long_put, short_call, long_call


def entry_credit(quotes: Quotes, minute: str,
                 strikes: Tuple[float, float, float, float],
                 cfg: JlConfig) -> Optional[Dict]:
    sp, lp, sc, lc = strikes
    spm = quote_mid(quotes, minute, sp, False)
    lpm = quote_mid(quotes, minute, lp, False)
    scm = quote_mid(quotes, minute, sc, True)
    lcm = quote_mid(quotes, minute, lc, True)
    if None in (spm, lpm, scm, lcm):
        return None
    put_credit = spm - lpm
    call_credit = scm - lcm
    if put_credit <= 0 or call_credit <= 0:
        return None
    return {
        "put_credit": put_credit, "call_credit": call_credit,
        "total_credit": put_credit + call_credit,
        "sp": spm, "lp": lpm, "sc": scm, "lc": lcm,
    }


def combined_mark(quotes: Quotes, minute: str,
                  strikes: Tuple[float, float, float, float]) -> Optional[float]:
    sp, lp, sc, lc = strikes
    spm = quote_mid(quotes, minute, sp, False)
    lpm = quote_mid(quotes, minute, lp, False)
    scm = quote_mid(quotes, minute, sc, True)
    lcm = quote_mid(quotes, minute, lc, True)
    if None in (spm, lpm, scm, lcm):
        return None
    return max(spm - lpm, 0) + max(scm - lcm, 0)


def _intrinsic_debit(strikes: Tuple[float, float, float, float], S: float) -> float:
    sp, lp, sc, lc = strikes
    put_debit = max(sp - S, 0) - max(lp - S, 0)
    call_debit = max(S - sc, 0) - max(S - lc, 0)
    return put_debit + call_debit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jl_core.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/jl_core.py tests/test_jl_core.py
git commit -m "feat(jl_core): config, strike selection, entry credit, combined mark"
```

---

## Task 3: jl_core — per-day evaluation (whole-position TP, exit valuation, costs)

**Files:**
- Modify: `src_v2/backtest/jl_core.py`
- Test: `tests/test_jl_core.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_jl_core.py

def _entry_chain():
    """09:30 chain that yields default legs and total_credit 6.0."""
    q = _chain_0930()                       # ATM/EM/forward = 5000/21/5000
    q.update({
        ("09:30", 4990.0, False): (5.0, 6.0),
        ("09:30", 4960.0, False): (2.0, 3.0),
        ("09:30", 5010.0, True): (6.0, 7.0),
        ("09:30", 5020.0, True): (3.0, 4.0),
    })
    return q


def test_evaluate_day_take_profit_at_mid():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"])
    q = _entry_chain()
    q.update({  # 10:00 combined mark = 1.0 + 1.5 = 2.5 <= 0.5*6 = 3.0 -> TP
        ("10:00", 4990.0, False): (1.4, 1.6), ("10:00", 4960.0, False): (0.4, 0.6),
        ("10:00", 5010.0, True): (1.9, 2.1),  ("10:00", 5020.0, True): (0.4, 0.6),
    })
    r = jl.evaluate_day(q, cfg)
    assert r["skip"] is None
    assert r["kind"] == "tp"
    assert r["exit_anchor"] == "10:00"
    assert r["pnl"] == 350.0    # (6.0 - 2.5) * 100


def test_evaluate_day_expiry_at_mid_loss():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"])
    q = _entry_chain()
    # no anchor reaches TP; final 16:00 combined mid = 8.0 -> loss
    q.update({
        ("10:00", 4990.0, False): (5.0, 5.0), ("10:00", 4960.0, False): (1.0, 1.0),
        ("10:00", 5010.0, True): (5.0, 5.0),  ("10:00", 5020.0, True): (1.0, 1.0),
        ("16:00", 4990.0, False): (9.0, 9.0), ("16:00", 4960.0, False): (1.0, 1.0),
        ("16:00", 5010.0, True): (1.0, 1.0),  ("16:00", 5020.0, True): (1.0, 1.0),
    })
    r = jl.evaluate_day(q, cfg)
    assert r["kind"] == "expiry"
    assert r["exit_anchor"] == "16:00"
    assert r["pnl"] == -200.0   # (6.0 - 8.0) * 100  [put mark 8.0 + call mark 0.0]


def test_evaluate_day_intrinsic_mode():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"], exit_mode="intrinsic")
    q = _entry_chain()
    q.update({  # never TP
        ("10:00", 4990.0, False): (5.0, 5.0), ("10:00", 4960.0, False): (1.0, 1.0),
        ("10:00", 5010.0, True): (5.0, 5.0),  ("10:00", 5020.0, True): (1.0, 1.0),
    })
    # spot_close 4980 -> put_debit = (4990-4980) - 0 = 10; call_debit 0 -> exit 10
    r = jl.evaluate_day(q, cfg, spot_close=4980.0)
    assert r["kind"] == "expiry"
    assert r["pnl"] == -400.0   # (6.0 - 10.0) * 100


def test_evaluate_day_with_costs():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"],
                      slippage_per_spread=0.10, commission_per_contract_per_leg=1.0)
    q = _entry_chain()
    q.update({
        ("10:00", 4990.0, False): (1.4, 1.6), ("10:00", 4960.0, False): (0.4, 0.6),
        ("10:00", 5010.0, True): (1.9, 2.1),  ("10:00", 5020.0, True): (0.4, 0.6),
    })
    r = jl.evaluate_day(q, cfg)
    # gross 350; entry+exit slip = 0.40*100 = 40; commission 4 legs * 1.0 = 4
    assert r["pnl"] == 306.0


def test_evaluate_day_skips():
    cfg = jl.JlConfig()
    assert jl.evaluate_day({}, cfg)["skip"] == "no_atm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jl_core.py -v`
Expected: FAIL with `AttributeError: ... 'evaluate_day'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src_v2/backtest/jl_core.py

def evaluate_day(quotes: Quotes, cfg: JlConfig,
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
    strikes = select_strikes(fwd, em, cfg)
    cred = entry_credit(quotes, ea, strikes, cfg)
    if cred is None:
        return {"skip": "no_credit"}
    total = cred["total_credit"]

    kind: Optional[str] = None
    exit_anchor: Optional[str] = None
    exit_mark: Optional[float] = None
    last_avail: Optional[Tuple[str, float]] = None
    for a in cfg.anchors:
        if a == ea:
            continue
        m = combined_mark(quotes, a, strikes)
        if m is None:
            continue
        last_avail = (a, m)
        if m <= cfg.tp_pct * total:
            kind, exit_anchor, exit_mark = "tp", a, m
            break

    if kind is None:
        if cfg.exit_mode == "intrinsic":
            if spot_close is None:
                return {"skip": "no_settle"}
            exit_anchor, exit_mark, kind = cfg.anchors[-1], _intrinsic_debit(strikes, spot_close), "expiry"
        else:
            if last_avail is None:
                return {"skip": "no_marks"}
            exit_anchor, exit_mark, kind = last_avail[0], last_avail[1], "expiry"

    mult = cfg.multiplier
    gross = (total - exit_mark) * mult
    slip = cfg.slippage_per_spread
    # entry always pays slip on 2 spreads; exit pays slip on 2 spreads UNLESS the
    # losers cash-settle at intrinsic (no closing trade).
    exit_slip = 0.0 if (kind == "expiry" and cfg.exit_mode == "intrinsic") else slip * 2
    cost = (slip * 2 + exit_slip) * mult + cfg.commission_per_contract_per_leg * 4
    net = gross - cost

    return {
        "skip": None, "kind": kind, "exit_anchor": exit_anchor,
        "total_credit": total, "exit_mark": exit_mark,
        "gross": gross, "pnl": net, "strikes": strikes,
        "em": em, "forward": fwd, "put_credit": cred["put_credit"],
        "call_credit": cred["call_credit"],
        "upside_riskfree": total >= cfg.call_width,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jl_core.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/jl_core.py tests/test_jl_core.py
git commit -m "feat(jl_core): per-day evaluation with whole-position TP, mid/intrinsic exit, costs"
```

---

## Task 4: jl_core — aggregate metrics over many days

**Files:**
- Modify: `src_v2/backtest/jl_core.py`
- Test: `tests/test_jl_core.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_jl_core.py
from datetime import date


def test_evaluate_aggregate():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"])
    win_q = _entry_chain()
    win_q.update({
        ("10:00", 4990.0, False): (1.4, 1.6), ("10:00", 4960.0, False): (0.4, 0.6),
        ("10:00", 5010.0, True): (1.9, 2.1),  ("10:00", 5020.0, True): (0.4, 0.6),
    })  # TP +350
    loss_q = _entry_chain()
    loss_q.update({
        ("10:00", 4990.0, False): (5.0, 5.0), ("10:00", 4960.0, False): (1.0, 1.0),
        ("10:00", 5010.0, True): (5.0, 5.0),  ("10:00", 5020.0, True): (1.0, 1.0),
        ("16:00", 4990.0, False): (9.0, 9.0), ("16:00", 4960.0, False): (1.0, 1.0),
        ("16:00", 5010.0, True): (1.0, 1.0),  ("16:00", 5020.0, True): (1.0, 1.0),
    })  # expiry -200
    by_date = {
        date(2025, 1, 2): {"quotes": win_q, "spot_close": 5000.0},
        date(2025, 1, 3): {"quotes": loss_q, "spot_close": 4985.0},
    }
    res = jl.evaluate(by_date, cfg)
    assert res["n"] == 2
    assert res["total"] == 150.0          # 350 - 200
    assert res["wr"] == 50.0
    assert res["exits"] == {"tp": 1, "expiry": 1}
    assert res["skips"] == {}


def test_max_consec_losses_and_drawdown():
    assert jl._max_consec_losses([1.0, -1.0, -1.0, 2.0, -1.0]) == 2
    assert jl._max_drawdown([100.0, -50.0, -80.0, 30.0]) == -130.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jl_core.py -v`
Expected: FAIL with `AttributeError: ... 'evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src_v2/backtest/jl_core.py

def _max_consec_losses(pnls: List[float]) -> int:
    best = cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _max_drawdown(pnls: List[float]) -> float:
    cum = peak = worst = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return worst


def evaluate(by_date: Dict, cfg: JlConfig) -> Dict:
    """Run evaluate_day over a {date: {quotes, spot_close}} map; aggregate metrics."""
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
    credits = [r["total_credit"] for r in results]
    return {
        "n": len(pnls),
        "wr": 100 * len(wins) / len(pnls),
        "total": sum(pnls),
        "mean": mean,
        "sharpe_ann": math.sqrt(252) * mean / std if std > 0 else 0.0,
        "max_consec_L": _max_consec_losses(pnls),
        "max_dd": _max_drawdown(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "worst": min(pnls),
        "best": max(pnls),
        "exits": dict(exits),
        "skips": dict(skips),
        "avg_credit": sum(credits) / len(credits),
        "pct_upside_riskfree": 100 * sum(1 for r in results if r["upside_riskfree"]) / len(results),
        "results": results,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jl_core.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/jl_core.py tests/test_jl_core.py
git commit -m "feat(jl_core): aggregate metrics over a date map"
```

---

## Task 5: CLI evaluator + Phase 0 free smoke test on the 5-anchor cache

**Files:**
- Create: `scripts/analytical_jl_eval.py`
- Test: `tests/test_jl_eval_cli.py`

- [ ] **Step 1: Write the failing test** (loader builds a date map from a cache dir)

```python
# tests/test_jl_eval_cli.py
import pickle
from datetime import date

from scripts.analytical_jl_eval import load_by_date


def test_load_by_date_reads_pickles(tmp_path, monkeypatch):
    import scripts.analytical_jl_eval as mod

    cache = tmp_path / "cache"
    cache.mkdir()
    quotes = {("09:30", 5000.0, True): (10.0, 11.0)}
    with open(cache / "2025-01-02.pkl", "wb") as f:
        pickle.dump(quotes, f)

    class _Day:
        session_date = date(2025, 1, 2)
        bars = [{"ts_et": __import__("datetime").datetime(2025, 1, 2, 15, 59),
                 "close": 5012.0}]

    class _Loader:
        def __init__(self, *a, **k): pass
        def iter_days(self, start_date=None, end_date=None): return [_Day()]

    monkeypatch.setattr(mod, "BarLoader", _Loader)
    by_date = load_by_date(cache, "ignored.csv", date(2025, 1, 1), date(2025, 1, 31))
    assert set(by_date) == {date(2025, 1, 2)}
    assert by_date[date(2025, 1, 2)]["spot_close"] == 5012.0
    assert by_date[date(2025, 1, 2)]["quotes"] == quotes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jl_eval_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.analytical_jl_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/analytical_jl_eval.py
"""Analytical Jade Lizard evaluator on a CBBO cache.

Headline = mid-price fills, zero cost, exit valued at the combined bid/ask mid at
the exit anchor. Also prints a with-costs sensitivity and a spot-intrinsic
cross-check. Per-trade rows are written to results/jade_lizard/.

Phase 0 (free) smoke test on the true-SPX 5-anchor cache:
    .venv/bin/python -m scripts.analytical_jl_eval \\
        --cache data/cbbo_cache_5anchor --entry 09:35 \\
        --anchors 09:35,11:00,13:00,15:30,15:59 \\
        --start 2025-09-01 --end 2026-02-27

Phase 3 (real) on the backfilled 10-min cache:
    .venv/bin/python -m scripts.analytical_jl_eval \\
        --cache data/cbbo_cache_jl_10min --entry 09:30 \\
        --start 2022-05-11 --end 2026-03-30
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest import jl_core as jl


def load_by_date(cache_dir: Path, bars_path: str, start: date, end: date) -> Dict:
    """Build {date: {quotes, spot_close}} for days that have a cache pickle."""
    loader = BarLoader(bars_path, price_scale=10.0, symbol="SPX")
    by_date: Dict = {}
    for day in loader.iter_days(start_date=start, end_date=end):
        d = day.session_date
        pkl = Path(cache_dir) / f"{d.isoformat()}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            quotes = pickle.load(f)
        b_eod = next((b for b in reversed(day.bars)
                      if b["ts_et"].time() >= time(15, 55)), None)
        by_date[d] = {"quotes": quotes,
                      "spot_close": b_eod["close"] if b_eod else None}
    return by_date


def _print_summary(label: str, res: Dict) -> None:
    if res["n"] == 0:
        print(f"\n[{label}] NO TRADES  skips={res.get('skips')}")
        return
    print(f"\n[{label}]  n={res['n']}  WR={res['wr']:.1f}%  total=${res['total']:+,.0f}  "
          f"mean=${res['mean']:+.1f}  Sharpe={res['sharpe_ann']:.2f}")
    print(f"  avg_credit=${res['avg_credit']:.2f}  maxDD=${res['max_dd']:+,.0f}  "
          f"max_consec_L={res['max_consec_L']}  avg_W=${res['avg_win']:+.0f}  "
          f"avg_L=${res['avg_loss']:+.0f}  worst=${res['worst']:+,.0f}  best=${res['best']:+,.0f}")
    print(f"  exits={res['exits']}  pct_upside_riskfree={res['pct_upside_riskfree']:.1f}%  "
          f"skips={res['skips']}")


def _write_trades(out_dir: Path, res: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trades_headline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "kind", "exit_anchor", "em", "forward",
                    "short_put", "long_put", "short_call", "long_call",
                    "total_credit", "exit_mark", "pnl", "upside_riskfree"])
        for r in res["results"]:
            sp, lp, sc, lc = r["strikes"]
            w.writerow([r["date"], r["kind"], r["exit_anchor"], f"{r['em']:.2f}",
                        f"{r['forward']:.2f}", sp, lp, sc, lc,
                        f"{r['total_credit']:.2f}", f"{r['exit_mark']:.2f}",
                        f"{r['pnl']:.2f}", r["upside_riskfree"]])
    summary = {k: v for k, v in res.items() if k != "results"}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main() -> int:
    p = argparse.ArgumentParser(description="Analytical Jade Lizard evaluator")
    p.add_argument("--cache", default="data/cbbo_cache_jl_10min")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-03-30")
    p.add_argument("--entry", default="09:30")
    p.add_argument("--anchors", default="",
                   help="comma list; default = 10-min grid from --entry to 16:00")
    p.add_argument("--em-mult", type=float, default=0.25)
    p.add_argument("--put-width", type=float, default=30.0)
    p.add_argument("--call-width", type=float, default=10.0)
    p.add_argument("--tp-pct", type=float, default=0.50)
    p.add_argument("--slippage", type=float, default=0.10,
                   help="with-costs slippage per spread (headline always 0)")
    p.add_argument("--commission", type=float, default=1.0,
                   help="with-costs commission per contract per leg")
    p.add_argument("--out", default="results/jade_lizard")
    args = p.parse_args()

    anchors = ([a.strip() for a in args.anchors.split(",")] if args.anchors
               else jl.make_anchors(args.entry, "16:00", 10))
    by_date = load_by_date(Path(args.cache), args.bars,
                           date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"Days with cache: {len(by_date)}  entry={args.entry}  "
          f"anchors={len(anchors)} ({anchors[0]}..{anchors[-1]})")

    base = dict(entry_anchor=args.entry, anchors=anchors,
                em_mult_put=args.em_mult, em_mult_call=args.em_mult,
                put_width=args.put_width, call_width=args.call_width,
                tp_pct=args.tp_pct)

    headline = jl.evaluate(by_date, jl.JlConfig(**base))
    _print_summary("HEADLINE (mid, zero-cost)", headline)

    costs = jl.evaluate(by_date, jl.JlConfig(
        **base, slippage_per_spread=args.slippage,
        commission_per_contract_per_leg=args.commission))
    _print_summary(f"WITH COSTS (slip={args.slippage}/spread, comm=${args.commission}/leg)", costs)

    intrinsic = jl.evaluate(by_date, jl.JlConfig(**base, exit_mode="intrinsic"))
    _print_summary("CROSS-CHECK (expiry at spot-intrinsic)", intrinsic)

    _write_trades(Path(args.out), headline)
    print(f"\nPer-trade rows -> {args.out}/trades_headline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jl_eval_cli.py -v`
Expected: PASS

- [ ] **Step 5: Phase 0 — run the free smoke test on the true-SPX 5-anchor cache**

Run:
```bash
.venv/bin/python -m scripts.analytical_jl_eval \
  --cache data/cbbo_cache_5anchor --entry 09:35 \
  --anchors 09:35,11:00,13:00,15:30,15:59 \
  --start 2025-09-01 --end 2026-02-27
```
Expected: prints HEADLINE / WITH COSTS / CROSS-CHECK summaries with `n` in the dozens
and a plausible win rate (whole-position 50% TP usually shows a high WR with a fat
left tail). This validates leg selection, credit, TP, and exit math on real SPX data
at **5 anchors** — a ballpark only; the 10-min/9:30 result comes in Phase 3.
Sanity-check: `avg_credit` should be a few dollars; `pct_upside_riskfree` shows how
often `total_credit ≥ $10`.

- [ ] **Step 6: Commit**

```bash
git add scripts/analytical_jl_eval.py tests/test_jl_eval_cli.py
git commit -m "feat(jl): analytical evaluator CLI + Phase-0 5-anchor smoke test"
```

---

## Task 6: Targeted SPXW backfill script + Phase 1 free dry-run

**Files:**
- Create: `scripts/backfill_jl_cbbo.py`
- Test: `tests/test_jl_backfill.py`

Reuses `src_v2/backtest/meic_backfill.py` (`spot_range_from_bars`, `meic_day_symbols`,
`meic_window_utc`, `quotes_from_df`) and `src_v2/backtest/databento_cost.py`
(`estimate_request_cost`, `CostLedger`). The only JL-specific change vs the MEIC
backfill is the **window (09:30→16:00)** and the **fixed ±100 center on the 9:30 bar**.

- [ ] **Step 1: Write the failing test** (the day-center + symbol-window helper)

```python
# tests/test_jl_backfill.py
from datetime import date, datetime

from scripts.backfill_jl_cbbo import jl_center_and_symbols


def _bars_0930(open_spot):
    return [{"ts_et": datetime(2025, 1, 2, 9, 30), "open": open_spot,
             "high": open_spot + 1, "low": open_spot - 1, "close": open_spot}]


def test_center_and_symbol_window():
    center, symbols = jl_center_and_symbols(
        date(2025, 1, 2), _bars_0930(5000.0), half_width_pts=100, step_pts=5)
    assert center == 5000.0
    # symbols span 4900..5100 (center +/- 100), both call+put at every 5 pts
    assert len(symbols) == ((5100 - 4900) // 5 + 1) * 2
    assert all("SPXW" in s for s in symbols)


def test_center_none_when_no_0930_bar():
    center, symbols = jl_center_and_symbols(
        date(2025, 1, 2), [], half_width_pts=100, step_pts=5)
    assert center is None and symbols == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jl_backfill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.backfill_jl_cbbo'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/backfill_jl_cbbo.py
#!/usr/bin/env python3
"""Targeted 0DTE SPXW CBBO-1m backfill for the Jade Lizard backtest.

For each trading day in [--start, --end], fetches the FULL session window
(09:30-16:00 ET) of cbbo-1m quotes for SPXW same-day strikes in a FIXED band
[center-100, center+100] where center = the 09:30 bar's spot (SPY-derived).
Fixed (non-rolling) centering keeps the entry-set legs covered all day. Writes a
per-day pickle {(minute, strike, is_call): (bid, ask)} to data/cbbo_cache_jl_10min/.

Modes:
  --dry-run         : metadata.get_cost (FREE) per day, print running total, exit
  (default)         : fetch, persist pickles, append spend to the cost ledger
  --max-cost-usd N  : abort the live run once accumulated cost exceeds N (default 15)

Usage:
  # 1. FREE cost estimate for the full range:
  .venv/bin/python scripts/backfill_jl_cbbo.py --start 2022-05-11 --end 2026-03-30 --dry-run
  # 2. Live fetch with a hard cap:
  .venv/bin/python scripts/backfill_jl_cbbo.py --start 2022-05-11 --end 2026-03-30 --max-cost-usd 15
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import databento as db

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.databento_cost import estimate_request_cost, CostLedger
from src_v2.backtest.meic_backfill import (
    meic_day_symbols, meic_window_utc, quotes_from_df,
)


def jl_center_and_symbols(
    session_date: date, bars: List[dict], half_width_pts: int = 100, step_pts: int = 5,
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
    p = argparse.ArgumentParser(description="Targeted 0DTE SPXW Jade Lizard backfill")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--out", default="data/cbbo_cache_jl_10min")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--half-width-pts", type=int, default=100)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--window-start", default="09:30")
    p.add_argument("--window-end", default="16:00")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep-ms", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=15.0)
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

        center, symbols = jl_center_and_symbols(
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jl_backfill.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Phase 1 — free dry-run cost estimate (NO spend)**

Run a small sample first, then the full range:
```bash
.venv/bin/python scripts/backfill_jl_cbbo.py --start 2023-06-01 --end 2023-06-09 --dry-run
.venv/bin/python scripts/backfill_jl_cbbo.py --start 2022-05-11 --end 2026-03-30 --dry-run
```
Expected: per-day `est $X.XXXX` lines and a final `cumulative cost $N.NN`. This call
is FREE (`metadata.get_cost`). **Record the cumulative estimate.** If it exceeds ~$15,
stop and report back before any live run (the spec cap is $15).

- [ ] **Step 6: Commit**

```bash
git add scripts/backfill_jl_cbbo.py tests/test_jl_backfill.py
git commit -m "feat(jl): targeted SPXW 9:30-16:00 backfill + free dry-run cost gate"
```

---

## Task 7: Phase 2 — run the backfill (operational, gated; not TDD)

**Pre-condition:** Task 6 Step 5 dry-run cumulative estimate is known and ≤ $15.
If it is higher, STOP and report the number to the user for a decision.

- [ ] **Step 1: Run the live backfill under the cap**

```bash
.venv/bin/python scripts/backfill_jl_cbbo.py \
  --start 2022-05-11 --end 2026-03-30 --max-cost-usd 15
```
Expected: per-day `... -> N quotes  $X.XXXX  cum $N.NN` lines; the run self-aborts if
cumulative cost hits $15. Re-running is safe and resumable (existing pickles are
skipped unless `--force`).

- [ ] **Step 2: Verify the cache**

```bash
ls data/cbbo_cache_jl_10min/ | wc -l
.venv/bin/python -c "import pickle; q=pickle.load(open('data/cbbo_cache_jl_10min/2024-03-04.pkl','rb')); mins=sorted({k[0] for k in q}); print('minutes', len(mins), mins[0], mins[-1]); print('entries', len(q)); print('strikes@0930', len(sorted({k[1] for k in q if k[0]==\"09:30\"})))"
```
Expected: ~900+ files; a probed day shows minutes spanning ~09:30→16:00 and ~41
strikes at 09:30 (±100 / 5). If a day shows 0 strikes at 09:30, note it — the
evaluator will skip it (`no_atm`) and count it.

- [ ] **Step 3: Confirm ledger spend**

```bash
.venv/bin/python -c "from src_v2.backtest.databento_cost import CostLedger; from pathlib import Path; l=CostLedger(Path('data/databento_cost_ledger.jsonl')); print('live total $', round(l.total_usd('live'),4))"
```
Expected: total live spend printed; the JL increment should be the ~$3–5 just spent.

- [ ] **Step 4: Commit a note (data files are git-ignored; commit only the count/log)**

```bash
git add -A
git commit -m "chore(jl): backfilled SPXW 9:30-16:00 10-min cache (data/cbbo_cache_jl_10min)" --allow-empty
```

---

## Task 8: Phase 3 — full 10-minute evaluation + findings report

**Files:**
- Create: `docs/spx_0dte_jade_lizard_findings.md`

- [ ] **Step 1: Run the full 10-minute evaluation**

```bash
.venv/bin/python -m scripts.analytical_jl_eval \
  --cache data/cbbo_cache_jl_10min --entry 09:30 \
  --start 2022-05-11 --end 2026-03-30
```
Expected: HEADLINE / WITH COSTS / CROSS-CHECK summaries over ~900 days, plus
`results/jade_lizard/trades_headline.csv` and `summary.json`.

- [ ] **Step 2: Sanity-verify the numbers before trusting them**

- HEADLINE `n` ≈ number of backfilled days minus skips; skip reasons should be small
  and explainable (`no_atm`/`no_credit` on thin days).
- HEADLINE vs CROSS-CHECK totals should be close (exit-at-mid ≈ exit-at-intrinsic near
  the close); a large divergence means deep-ITM mid noise — note it in the report.
- Spot-check 2–3 rows in `trades_headline.csv`: legs at 0.25×EM with 30/10 widths, and
  `pnl = (total_credit − exit_mark) × 100`.

- [ ] **Step 3: Write the findings doc**

Create `docs/spx_0dte_jade_lizard_findings.md` with: the strategy recap (link the spec),
the exact run command, the three summary tables (headline / with-costs / intrinsic),
exit-type breakdown, `pct_upside_riskfree`, max drawdown and worst day, scaling to a
$100K account (per-lot max loss ≈ $30×100 on the put side → BP-limited contract count),
and the caveats: single mid-price assumption, deep-ITM mid noise near expiry, SPY×10
only used to center the backfill window, no execution guards.

- [ ] **Step 4: Commit**

```bash
git add docs/spx_0dte_jade_lizard_findings.md results/jade_lizard/summary.json
git commit -m "docs(jl): full 10-min Jade Lizard backtest findings"
```

---

## Self-Review

**Spec coverage:**
- §2 strategy (EM=straddle, 0.25×EM, 30/10 widths, whole-position 50% TP, no SL) → Tasks 1–4 (jl_core), verified by `tests/test_jl_core.py`.
- §3 10-min checks → `make_anchors` (Task 1) + `evaluate_day` anchor loop (Task 3); 09:30 entry default.
- §4 chain-derived spot/forward → `atm_from_chain`/`forward_from_chain` (Task 1). *Refinement vs spec:* exit valued at combined mid at the exit anchor (matches "exercise at mid"); spot-intrinsic kept as a cross-check column (Tasks 3/5).
- §5 zero-cost headline + with-costs sensitivity → `JlConfig` cost fields (Task 2), three-table CLI (Task 5).
- §6 tailored SPXW backfill, fixed ±100 center, 9:30→16:00, new dir, dry-run cap $15 → Task 6/7.
- §7 components → file structure table; high reuse of `meic_backfill`/`databento_cost`/`BarLoader`.
- §8 metrics → `evaluate` (Task 4) + `_print_summary`/`_write_trades` (Task 5); findings doc (Task 8).
- §9 phases → Phase 0 (Task 5 Step 5), Phase 1 (Task 6 Step 5), Phase 2 (Task 7), Phase 3 (Task 8). WF-CV explicitly deferred (out of scope).

**Placeholder scan:** No TBD/TODO; every code step has complete code; every run step has the exact command and expected output. The findings doc (Task 8 Step 3) is content-generation from real results, not a code placeholder.

**Type consistency:** `JlConfig` fields, `select_strikes`/`entry_credit`/`combined_mark`/`evaluate_day`/`evaluate` signatures, the quote-dict schema `{(minute,strike,is_call):(bid,ask)}`, and the result-dict keys (`kind`, `exit_anchor`, `total_credit`, `exit_mark`, `pnl`, `strikes`, `upside_riskfree`) are used identically across `jl_core`, the CLI, and the tests. Backfill reuses `meic_day_symbols(session_date, spot_low, spot_high, buffer_pts, step_pts)` with the real signature from `src_v2/backtest/meic_backfill.py`.
