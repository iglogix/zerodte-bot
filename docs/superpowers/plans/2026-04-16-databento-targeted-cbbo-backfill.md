# Databento Targeted CBBO Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill SPX 0DTE CBBO-1m option quotes for ~600 historical signal days using the Databento API, fetching only the strikes (~ATM ± 50pts) and minutes (signal-relevant windows) actually needed — keeping cost under ~$5 vs. the $181 unfiltered alternative.

**Architecture:** Four independent components: (1) a pure-Python ORB signal detector that replays the strategy logic against existing ES OHLCV-1m bars to enumerate every historical signal day/minute into a JSONL manifest; (2) a Databento client wrapper that reads the manifest, builds OSI symbol lists, and fetches narrow time-window CBBO data via `timeseries.get_range`; (3) a cost-estimator wrapper around `metadata.get_cost` (free to call) that powers a `--dry-run` mode and per-call tracking; (4) a per-day pickle writer that produces files in the exact format the existing [`CbboStore`](src_v2/backtest/cbbo_store.py) already consumes — zero changes needed downstream.

**Persistence model:** Each successful day's quotes are written to `data/cbbo_cache/YYYY-MM-DD.pkl` (matching the existing 125-day cache shape exactly) and a running cost ledger is appended to `data/databento_cost_ledger.jsonl` so any re-run can recover total spend without re-querying. Driver is idempotent — re-running skips dates whose pickle already exists, unless `--force`.

**Cost guardrails:** The driver supports `--dry-run` (uses `metadata.get_cost`, free, exits without fetching) and `--max-cost-usd N` (aborts mid-run if accumulated cost exceeds N). Both rely on calling `metadata.get_cost` with the exact same params as `get_range`, so the dry-run number is the contractual upper bound on what the live run will charge.

**Tech Stack:** Python 3.13, `databento` SDK, `zoneinfo`, existing project's `DatabentoLoader` for ES OHLCV bars, pytest for tests. Output drops into `data/cbbo_cache/YYYY-MM-DD.pkl` matching current pickle format `{(minute_str, strike, is_call): (bid, ask)}`.

---

## File Structure

**New files:**
- `src_v2/backtest/signal_detector.py` — pure function: ES bars + ORB params → list of signal events
- `src_v2/backtest/basis_estimator.py` — pure function: estimates ES→SPX basis from existing CBBO cache
- `src_v2/backtest/osi_symbol.py` — OSI symbol formatting (strike → `SPXW  YYMMDDP00006445000`)
- `src_v2/backtest/databento_backfill.py` — wraps `databento.Historical()`, returns CBBO dict matching existing pickle format
- `src_v2/backtest/databento_cost.py` — wraps `metadata.get_cost`, accumulates a per-call ledger
- `scripts/enumerate_signals.py` — CLI: reads ES CSV → writes signal manifest JSONL
- `scripts/databento_backfill.py` — CLI: reads manifest → (optional dry-run cost estimate) → calls Databento → writes pickles + cost ledger
- `tests/test_signal_detector.py`
- `tests/test_basis_estimator.py`
- `tests/test_osi_symbol.py`
- `tests/test_databento_backfill.py`
- `tests/test_databento_cost.py`

**Modified files:**
- `requirements.txt` — add `databento>=0.40.0`

**Untouched (deliberately):**
- `src_v2/backtest/cbbo_store.py` — already pickle-based, accepts new pickles unchanged
- All strategy / engine / optimizer code

---

## Task 1: Add Databento SDK dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Inspect current requirements.txt**

Run: `grep -n "databento\|zstandard" requirements.txt`
Expected: only `zstandard` line present, no `databento`.

- [ ] **Step 2: Add databento line**

Modify `requirements.txt` — append the line `databento>=0.40.0` after the `zstandard` line.

- [ ] **Step 3: Install into project venv**

Run: `.venv/bin/pip install 'databento>=0.40.0'`
Expected: successful install of `databento` and dependencies (pandas, requests, pyarrow).

- [ ] **Step 4: Verify import works**

Run: `.venv/bin/python -c "import databento; print(databento.__version__)"`
Expected: prints a version `>= 0.40.0`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "deps: add databento SDK for targeted CBBO backfill"
```

---

## Task 2: OSI symbol formatter (smallest leaf)

**Files:**
- Create: `src_v2/backtest/osi_symbol.py`
- Test: `tests/test_osi_symbol.py`

OSI format used by OPRA / Databento `raw_symbol` for SPXW: `SPXW  YYMMDDX00000000` — 21 chars total, where root is left-justified to 6 chars (so `SPXW` becomes `SPXW  ` with two trailing spaces), then YYMMDD expiry, then `C` or `P`, then strike × 1000 zero-padded to 8 digits. Confirmed by [`src_v2/backtest/cbbo_store.py:27`](src_v2/backtest/cbbo_store.py#L27) regex `SPXW\s+(\d{6})([CP])(\d{8})` and example pickle key `(('09:31', 6925.0, False), ...)` paired with raw symbol `"SPXW  250922P06925000"`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_osi_symbol.py`:

```python
from datetime import date
import pytest
from src_v2.backtest.osi_symbol import format_osi, build_strike_window


def test_format_osi_put_integer_strike():
    assert format_osi(date(2025, 9, 22), 6445.0, is_call=False) == "SPXW  250922P06445000"


def test_format_osi_call_integer_strike():
    assert format_osi(date(2024, 3, 20), 5235.0, is_call=True) == "SPXW  240320C05235000"


def test_format_osi_5pt_strike_step():
    assert format_osi(date(2026, 1, 15), 4750.0, is_call=False) == "SPXW  260115P04750000"


def test_format_osi_strike_with_half_point():
    assert format_osi(date(2025, 9, 22), 6442.5, is_call=True) == "SPXW  250922C06442500"


def test_build_strike_window_default():
    syms = build_strike_window(date(2025, 9, 22), atm_strike=6445.0, half_width_pts=50, step_pts=5)
    assert "SPXW  250922P06395000" in syms
    assert "SPXW  250922C06495000" in syms
    assert "SPXW  250922P06445000" in syms
    assert "SPXW  250922C06445000" in syms
    assert len(syms) == 21 * 2  # 21 strikes × call+put


def test_build_strike_window_strikes_are_step_aligned():
    syms = build_strike_window(date(2025, 9, 22), atm_strike=6443.0, half_width_pts=10, step_pts=5)
    parsed_strikes = sorted({int(s[-8:]) // 1000 for s in syms})
    assert parsed_strikes == [6435, 6440, 6445, 6450, 6455]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_osi_symbol.py -v`
Expected: ImportError / module not found.

- [ ] **Step 3: Implement `osi_symbol.py`**

Create `src_v2/backtest/osi_symbol.py`:

```python
"""OSI symbol formatter for SPXW 0DTE options.

OPRA raw_symbol format: 21 chars = 6-char root (left-justified) + YYMMDD + C/P + strike*1000 (8-digit zero-pad).
Example: 'SPXW  250922P06445000' = SPXW Sep 22 2025 6445.000 Put.
"""
from __future__ import annotations

from datetime import date
from typing import List


def format_osi(expiry: date, strike: float, is_call: bool, root: str = "SPXW") -> str:
    """Build a 21-char OSI raw_symbol string for an SPXW option."""
    root_padded = f"{root:<6}"  # 'SPXW  '
    expiry_str = expiry.strftime("%y%m%d")
    cp = "C" if is_call else "P"
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"
    return f"{root_padded}{expiry_str}{cp}{strike_str}"


def build_strike_window(
    expiry: date,
    atm_strike: float,
    half_width_pts: int = 50,
    step_pts: int = 5,
    root: str = "SPXW",
) -> List[str]:
    """Build OSI symbols for ATM ± half_width_pts in step_pts increments, both calls and puts.

    `atm_strike` is snapped to the nearest step before windowing so all
    returned strikes are step-aligned.
    """
    base = int(round(atm_strike / step_pts) * step_pts)
    n_steps = half_width_pts // step_pts
    strikes = [base + i * step_pts for i in range(-n_steps, n_steps + 1)]
    out: List[str] = []
    for k in strikes:
        out.append(format_osi(expiry, float(k), is_call=False, root=root))
        out.append(format_osi(expiry, float(k), is_call=True, root=root))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_osi_symbol.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/osi_symbol.py tests/test_osi_symbol.py
git commit -m "feat(backfill): OSI raw_symbol formatter + strike window builder"
```

---

## Task 3: ES→SPX basis estimator

**Files:**
- Create: `src_v2/backtest/basis_estimator.py`
- Test: `tests/test_basis_estimator.py`

**Why:** To pre-compute the ATM SPX strike before the API call, we need an offline estimate of `SPX_spot ≈ ES_close - basis`. The basis varies (futures premium, dividend drift) but is small and slow-moving. We use the 125 days of CBBO already on disk to fit a per-month median basis, then look it up by month for any historical date.

- [ ] **Step 1: Write failing tests**

Create `tests/test_basis_estimator.py`:

```python
from datetime import date
import pickle
from pathlib import Path

import pytest

from src_v2.backtest.basis_estimator import (
    compute_daily_basis,
    build_monthly_basis_table,
    estimate_spx_from_es,
)


def _fake_cbbo_pickle(tmp_path: Path, day: date, atm_strike: float, call_mid: float, put_mid: float) -> Path:
    """Write a minimal pickle that derive_spot can read at 09:30."""
    quotes = {
        ("09:30", atm_strike, True): (call_mid - 0.5, call_mid + 0.5),
        ("09:30", atm_strike, False): (put_mid - 0.5, put_mid + 0.5),
    }
    p = tmp_path / f"{day.isoformat()}.pkl"
    with open(p, "wb") as f:
        pickle.dump(quotes, f)
    return p


def test_compute_daily_basis_uses_put_call_parity(tmp_path):
    # SPX_spot = K + C - P. With K=5000, C=10, P=4 -> spot=5006. ES=5010 -> basis=4.
    _fake_cbbo_pickle(tmp_path, date(2025, 9, 22), atm_strike=5000.0, call_mid=10.0, put_mid=4.0)
    basis = compute_daily_basis(tmp_path, date(2025, 9, 22), es_open=5010.0, anchor_minute="09:30")
    assert basis == pytest.approx(4.0, abs=0.01)


def test_build_monthly_basis_table_takes_median(tmp_path):
    # Two days in Sep 2025: bases 4.0 and 6.0 -> median 5.0
    _fake_cbbo_pickle(tmp_path, date(2025, 9, 22), 5000.0, 10.0, 4.0)  # spot=5006, basis 4 vs ES 5010
    _fake_cbbo_pickle(tmp_path, date(2025, 9, 23), 5000.0, 10.0, 6.0)  # spot=5004, basis 6 vs ES 5010
    es_opens = {date(2025, 9, 22): 5010.0, date(2025, 9, 23): 5010.0}
    table = build_monthly_basis_table(tmp_path, es_opens)
    assert table[(2025, 9)] == pytest.approx(5.0, abs=0.01)


def test_estimate_spx_from_es_uses_table_or_fallback():
    table = {(2025, 9): 4.0}
    assert estimate_spx_from_es(date(2025, 9, 15), 5010.0, table, default_basis=10.0) == 5006.0
    # Month not in table → fallback
    assert estimate_spx_from_es(date(2024, 1, 15), 5010.0, table, default_basis=10.0) == 5000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_basis_estimator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `basis_estimator.py`**

Create `src_v2/backtest/basis_estimator.py`:

```python
"""ES → SPX basis estimator using existing CBBO cache.

Used offline to estimate SPX_spot from ES futures price for historical dates
where we don't yet have CBBO data. SPX = ES - basis. Basis is dominated by
dividend drift + interest rate carry, so a per-month median is sufficient.
"""
from __future__ import annotations

import pickle
import statistics
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_pickle(cache_dir: Path, day: date) -> Optional[dict]:
    p = cache_dir / f"{day.isoformat()}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def compute_daily_basis(
    cache_dir: Path,
    day: date,
    es_open: float,
    anchor_minute: str = "09:30",
) -> Optional[float]:
    """basis = es_open − spx_spot, where spx_spot is derived from put-call parity.

    Picks the strike closest to es_open with both call and put quotes at anchor_minute.
    Returns None if no usable quotes.
    """
    quotes = _load_pickle(cache_dir, day)
    if not quotes:
        return None

    # All strikes that have BOTH a call and put quote at anchor_minute
    by_strike: Dict[float, Dict[bool, Tuple[float, float]]] = {}
    for (m, strike, is_call), (bid, ask) in quotes.items():
        if m != anchor_minute:
            continue
        by_strike.setdefault(strike, {})[is_call] = (bid, ask)

    paired = [k for k, sides in by_strike.items() if True in sides and False in sides]
    if not paired:
        return None

    atm = min(paired, key=lambda k: abs(k - es_open))
    c_bid, c_ask = by_strike[atm][True]
    p_bid, p_ask = by_strike[atm][False]
    c_mid = (c_bid + c_ask) / 2
    p_mid = (p_bid + p_ask) / 2
    spx = atm + c_mid - p_mid
    return round(es_open - spx, 2)


def build_monthly_basis_table(
    cache_dir: Path,
    es_opens: Dict[date, float],
    anchor_minute: str = "09:30",
) -> Dict[Tuple[int, int], float]:
    """For each (year, month) present in cache_dir, compute median basis across days."""
    by_month: Dict[Tuple[int, int], list] = {}
    for day, es_open in es_opens.items():
        b = compute_daily_basis(cache_dir, day, es_open, anchor_minute)
        if b is None:
            continue
        by_month.setdefault((day.year, day.month), []).append(b)
    return {ym: round(statistics.median(vals), 2) for ym, vals in by_month.items() if vals}


def estimate_spx_from_es(
    day: date,
    es_price: float,
    monthly_basis: Dict[Tuple[int, int], float],
    default_basis: float = 10.0,
) -> float:
    """SPX_spot = ES_price − basis(month). Falls back to default_basis if month missing."""
    basis = monthly_basis.get((day.year, day.month), default_basis)
    return round(es_price - basis, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_basis_estimator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/basis_estimator.py tests/test_basis_estimator.py
git commit -m "feat(backfill): ES→SPX basis estimator using existing CBBO cache"
```

---

## Task 4: ORB signal detector (pure function)

**Files:**
- Create: `src_v2/backtest/signal_detector.py`
- Test: `tests/test_signal_detector.py`

**Why:** We need to enumerate every historical signal minute over 5 years of ES bars. The live `OrbEngine` is async and broker-coupled — too heavy for offline replay. Re-implement the deterministic core: ORB high/low/midline → wait for breakout → wait for retest based on `retest_mode` → emit signal minute.

- [ ] **Step 1: Write failing tests**

Create `tests/test_signal_detector.py`:

```python
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from src_v2.backtest.signal_detector import (
    SignalEvent,
    compute_orb_levels,
    detect_signals_for_day,
)

ET = ZoneInfo("America/New_York")


def _bar(t: str, o: float, h: float, l: float, c: float):
    """Build a one-minute bar dict in the format DatabentoLoader yields."""
    hh, mm = int(t[:2]), int(t[3:5])
    return {
        "ts_et": datetime(2024, 3, 20, hh, mm, tzinfo=ET),
        "open": o, "high": h, "low": l, "close": c, "volume": 100,
    }


def test_compute_orb_levels_takes_window_high_low():
    bars = [
        _bar("08:00", 5000, 5005, 4998, 5003),
        _bar("08:05", 5003, 5010, 5001, 5008),
        _bar("08:14", 5008, 5012, 5007, 5011),
        _bar("08:15", 5011, 5013, 5010, 5012),  # outside window (end exclusive)
    ]
    lo, mid, hi = compute_orb_levels(bars, "08:00", "08:15")
    assert hi == 5012
    assert lo == 4998
    assert mid == pytest.approx(5005.0)


def test_compute_orb_levels_returns_none_if_window_empty():
    bars = [_bar("07:30", 5000, 5005, 4998, 5003)]
    assert compute_orb_levels(bars, "08:00", "08:15") == (None, None, None)


def test_detect_signals_breakout_level_long():
    """Long bias: price breaks ORB high after 09:30, then retests ORB high → signal."""
    bars = (
        # ORB window 08:00-08:14 → high=5012, low=4998
        [_bar(f"08:{m:02d}", 5000, 5012 if m == 7 else 5005, 4998 if m == 3 else 5000, 5005) for m in range(15)]
        # 09:30 onward
        + [_bar("09:30", 5005, 5008, 5004, 5007),
           _bar("09:31", 5007, 5015, 5006, 5014),  # breakout above 5012 → long bias
           _bar("09:32", 5014, 5016, 5012, 5012),  # retest of 5012 → SIGNAL here
           _bar("09:33", 5012, 5018, 5011, 5017)]
    )
    sigs = detect_signals_for_day(
        bars, session_date=date(2024, 3, 20),
        orb_start="08:00", orb_end="08:15", market_open="09:30",
        eod="15:50", retest_mode="breakout_level",
        midline_tol_pts=1.0, min_range_pts=2.0, max_range_pts=30.0,
    )
    assert len(sigs) == 1
    assert sigs[0].bias == "long"
    assert sigs[0].signal_minute_et == "09:32"
    assert sigs[0].es_price_at_signal == 5012.0


def test_detect_signals_skips_when_orb_range_below_min():
    bars = (
        [_bar(f"08:{m:02d}", 5000, 5001, 4999.5, 5000) for m in range(15)]
        + [_bar("09:30", 5000, 5010, 4995, 5008)]  # would-be breakout but range too small
    )
    sigs = detect_signals_for_day(
        bars, date(2024, 3, 20), "08:00", "08:15", "09:30", "15:50",
        retest_mode="breakout_level", midline_tol_pts=1.0,
        min_range_pts=2.0, max_range_pts=30.0,
    )
    assert sigs == []


def test_detect_signals_breakout_level_short():
    bars = (
        [_bar(f"08:{m:02d}", 5005, 5012 if m == 7 else 5008, 4998 if m == 3 else 5005, 5005) for m in range(15)]
        + [_bar("09:30", 5005, 5008, 5004, 5006),
           _bar("09:31", 5006, 5007, 4990, 4995),  # breakout below 4998 → short bias
           _bar("09:32", 4995, 4999, 4994, 4998),  # retest of 4998 → SIGNAL
           _bar("09:33", 4998, 4998, 4985, 4990)]
    )
    sigs = detect_signals_for_day(
        bars, date(2024, 3, 20), "08:00", "08:15", "09:30", "15:50",
        retest_mode="breakout_level", midline_tol_pts=1.0,
        min_range_pts=2.0, max_range_pts=30.0,
    )
    assert len(sigs) == 1
    assert sigs[0].bias == "short"
    assert sigs[0].signal_minute_et == "09:32"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_signal_detector.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `signal_detector.py`**

Create `src_v2/backtest/signal_detector.py`:

```python
"""Pure-function ORB signal detector for offline replay.

Replicates the deterministic core of src_v2/agent/orb_engine.py without async
or broker coupling, so we can enumerate signal events from years of historical
ES bars in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import List, Literal, Optional, Tuple

Bias = Literal["long", "short"]
RetestMode = Literal["midline", "breakout_level", "opposite_level"]


@dataclass(frozen=True)
class SignalEvent:
    session_date: date
    bias: Bias
    signal_minute_et: str  # "HH:MM"
    es_price_at_signal: float
    orb_high: float
    orb_low: float


def _hhmm_to_min(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:5])


def _bar_min(bar) -> int:
    t = bar["ts_et"]
    return t.hour * 60 + t.minute


def compute_orb_levels(
    bars: List[dict], orb_start: str, orb_end: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (low, midline, high) from bars whose minute is in [orb_start, orb_end)."""
    s, e = _hhmm_to_min(orb_start), _hhmm_to_min(orb_end)
    in_window = [b for b in bars if s <= _bar_min(b) < e]
    if not in_window:
        return (None, None, None)
    hi = max(b["high"] for b in in_window)
    lo = min(b["low"] for b in in_window)
    return (lo, (hi + lo) / 2, hi)


def detect_signals_for_day(
    bars: List[dict],
    session_date: date,
    orb_start: str,
    orb_end: str,
    market_open: str,
    eod: str,
    retest_mode: RetestMode,
    midline_tol_pts: float,
    min_range_pts: float,
    max_range_pts: float,
) -> List[SignalEvent]:
    """Detect at most one signal per day. Returns [] if no breakout or filter rejects."""
    lo, mid, hi = compute_orb_levels(bars, orb_start, orb_end)
    if hi is None:
        return []

    rng = hi - lo
    if rng < min_range_pts or rng > max_range_pts:
        return []

    open_min = _hhmm_to_min(market_open)
    eod_min = _hhmm_to_min(eod)
    post_open = [b for b in bars if open_min <= _bar_min(b) <= eod_min]

    # Phase 1: find first breakout
    bias: Optional[Bias] = None
    breakout_idx: Optional[int] = None
    for i, b in enumerate(post_open):
        if b["high"] >= hi:
            bias = "long"
            breakout_idx = i
            break
        if b["low"] <= lo:
            bias = "short"
            breakout_idx = i
            break

    if bias is None:
        return []

    # In opposite_level mode the entry inverts the breakout bias
    entry_bias: Bias = bias
    if retest_mode == "opposite_level":
        entry_bias = "short" if bias == "long" else "long"

    # Phase 2: pick the retest target
    if retest_mode == "midline":
        target = mid
    elif retest_mode == "breakout_level":
        target = hi if bias == "long" else lo
    elif retest_mode == "opposite_level":
        target = lo if bias == "long" else hi
    else:
        raise ValueError(f"unknown retest_mode: {retest_mode}")

    # Phase 3: scan for retest after breakout
    for b in post_open[breakout_idx + 1:]:
        # Touch within tolerance counts as retest
        if b["low"] - midline_tol_pts <= target <= b["high"] + midline_tol_pts:
            t = b["ts_et"]
            return [SignalEvent(
                session_date=session_date,
                bias=entry_bias,
                signal_minute_et=f"{t.hour:02d}:{t.minute:02d}",
                es_price_at_signal=float(target),
                orb_high=float(hi),
                orb_low=float(lo),
            )]

    return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_signal_detector.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/signal_detector.py tests/test_signal_detector.py
git commit -m "feat(backfill): pure-function ORB signal detector for offline replay"
```

---

## Task 5: Signal enumerator CLI

**Files:**
- Create: `scripts/enumerate_signals.py`

This CLI ties together the existing [`DatabentoLoader`](src_8am/backtest/data_loader.py#L58) (for ES bars) with `signal_detector.detect_signals_for_day` and writes a JSONL manifest. No new tests — it's glue code; correctness comes from the underlying tested modules.

- [ ] **Step 1: Create the script**

Create `scripts/enumerate_signals.py`:

```python
#!/usr/bin/env python3
"""Enumerate historical ORB signal events from ES OHLCV bars.

Reads existing GLBX MDP3 ES OHLCV-1m CSVs via DatabentoLoader, replays the ORB
signal detector across all days in the range, and writes one JSON line per
signal event to a manifest file.

Usage:
    python scripts/enumerate_signals.py \
        --es-csv glbx-mdp3-20210227-20260226.ohlcv-1m.csv \
        --config config/config_orb_dir_breakout_spx.yaml \
        --start 2021-03-01 --end 2026-02-26 \
        --out data/signal_manifest_orb_breakout.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_8am.backtest.data_loader import DatabentoLoader
from src_v2.backtest.signal_detector import detect_signals_for_day


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--es-csv", required=True, help="GLBX MDP3 ES OHLCV-1m CSV (.csv or .csv.zst)")
    p.add_argument("--config", required=True, help="Strategy config YAML")
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--out", required=True, help="Output manifest path (.jsonl)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strat = cfg["strategy"]
    orb_start = strat.get("orb_window_start", "08:00")
    orb_end = strat.get("orb_window_end", "08:15")
    market_open = strat.get("market_open", "09:30")
    eod = strat.get("eod_force_close", "15:50")
    retest_mode = strat.get("retest_mode", "midline")
    midline_tol = strat.get("midline_tolerance_pts", 1.0)
    rfilter = strat.get("orb_range_filter", {})
    min_rng = rfilter.get("min_pts", 2.0)
    max_rng = rfilter.get("max_pts", 30.0)

    loader = DatabentoLoader(args.es_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_days = 0
    n_signals = 0
    with open(out_path, "w") as f:
        for day in loader.iter_days(start_date=args.start, end_date=args.end, front_month_only=True):
            n_days += 1
            sigs = detect_signals_for_day(
                bars=day.bars,
                session_date=day.session_date,
                orb_start=orb_start, orb_end=orb_end,
                market_open=market_open, eod=eod,
                retest_mode=retest_mode,
                midline_tol_pts=midline_tol,
                min_range_pts=min_rng, max_range_pts=max_rng,
            )
            for s in sigs:
                f.write(json.dumps({
                    "date": s.session_date.isoformat(),
                    "bias": s.bias,
                    "signal_minute_et": s.signal_minute_et,
                    "es_price_at_signal": s.es_price_at_signal,
                    "orb_high": s.orb_high,
                    "orb_low": s.orb_low,
                }) + "\n")
                n_signals += 1

    print(f"Scanned {n_days} days, emitted {n_signals} signals → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test on the existing 5-year ES CSV**

Run:
```bash
.venv/bin/python scripts/enumerate_signals.py \
  --es-csv glbx-mdp3-20210227-20260226.ohlcv-1m.csv \
  --config config/config_orb_dir_breakout_spx.yaml \
  --start 2025-09-22 --end 2026-02-26 \
  --out data/signal_manifest_smoke.jsonl
```
Expected: prints `Scanned ~110 days, emitted 30-60 signals → data/signal_manifest_smoke.jsonl`. The exact count is informational; what matters is that it runs end-to-end without exceptions and the file has lines that parse as JSON.

- [ ] **Step 3: Spot-check the output**

Run: `head -3 data/signal_manifest_smoke.jsonl && wc -l data/signal_manifest_smoke.jsonl`
Expected: each line has fields `date, bias, signal_minute_et, es_price_at_signal, orb_high, orb_low`. Line count > 0.

- [ ] **Step 4: Cross-check against existing trades.csv**

Run: `wc -l results/orb_dir_breakout_spx/trades.csv`
Expected: `data/signal_manifest_smoke.jsonl` line count is in the same ballpark as the trades.csv row count (within ±20%). Any wider gap means the offline detector has drifted from `OrbEngine`'s logic and Task 4 needs to be revisited before continuing.

- [ ] **Step 5: Commit**

```bash
git add scripts/enumerate_signals.py
git commit -m "feat(backfill): enumerate_signals CLI — replays ORB detector over historical ES bars"
```

---

## Task 6: Databento backfill client wrapper

**Files:**
- Create: `src_v2/backtest/databento_backfill.py`
- Test: `tests/test_databento_backfill.py`

**Why:** Encapsulate the Databento API call so the CLI driver stays simple. The wrapper takes (session_date, signal_minute, atm_strike, time_window_minutes) and returns a quote dict in the existing pickle format `{(minute_str, strike, is_call): (bid, ask)}`. The Databento client itself is injected so tests can mock it.

- [ ] **Step 1: Write failing tests**

Create `tests/test_databento_backfill.py`:

```python
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src_v2.backtest.databento_backfill import (
    fetch_day_quotes,
    build_request_window,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def test_build_request_window_converts_et_to_utc_iso():
    start_iso, end_iso = build_request_window(
        session_date=date(2025, 9, 22),
        anchor_minute_et="09:32",
        before_min=2,
        after_min=10,
    )
    # 09:30 ET == 13:30 UTC (DST), 09:42 ET == 13:42 UTC
    assert start_iso == "2025-09-22T13:30:00"
    assert end_iso == "2025-09-22T13:42:00"


def _fake_dbn_dataframe(rows):
    """Build a DataFrame matching what client.timeseries.get_range(...).to_df() returns
    for an OPRA cbbo-1m request. Index is a DatetimeIndex of ts_recv (UTC)."""
    df = pd.DataFrame(rows)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df = df.set_index("ts_recv")
    return df


def test_fetch_day_quotes_parses_into_pickle_format():
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.to_df.return_value = _fake_dbn_dataframe([
        {"ts_recv": "2025-09-22T13:30:00Z", "symbol": "SPXW  250922P06445000", "bid_px_00": 0.10, "ask_px_00": 0.15},
        {"ts_recv": "2025-09-22T13:30:00Z", "symbol": "SPXW  250922C06445000", "bid_px_00": 12.30, "ask_px_00": 12.50},
        {"ts_recv": "2025-09-22T13:31:00Z", "symbol": "SPXW  250922P06445000", "bid_px_00": 0.11, "ask_px_00": 0.16},
    ])
    fake_client.timeseries.get_range.return_value = fake_response

    quotes = fetch_day_quotes(
        client=fake_client,
        session_date=date(2025, 9, 22),
        anchor_minute_et="09:32",
        atm_strike=6445.0,
        half_width_pts=50,
        step_pts=5,
        before_min=2,
        after_min=10,
    )

    assert quotes[("09:30", 6445.0, False)] == (0.10, 0.15)
    assert quotes[("09:30", 6445.0, True)] == (12.30, 12.50)
    assert quotes[("09:31", 6445.0, False)] == (0.11, 0.16)


def test_fetch_day_quotes_passes_correct_request_params():
    fake_client = MagicMock()
    fake_client.timeseries.get_range.return_value.to_df.return_value = _fake_dbn_dataframe([])

    fetch_day_quotes(
        client=fake_client,
        session_date=date(2025, 9, 22),
        anchor_minute_et="09:32",
        atm_strike=6445.0,
        half_width_pts=10,
        step_pts=5,
        before_min=2,
        after_min=10,
    )

    call_kwargs = fake_client.timeseries.get_range.call_args.kwargs
    assert call_kwargs["dataset"] == "OPRA.PILLAR"
    assert call_kwargs["schema"] == "cbbo-1m"
    assert call_kwargs["stype_in"] == "raw_symbol"
    assert call_kwargs["start"] == "2025-09-22T13:30:00"
    assert call_kwargs["end"] == "2025-09-22T13:42:00"
    # 5 strikes (6435..6455 step 5) × 2 sides
    assert len(call_kwargs["symbols"]) == 10
    assert "SPXW  250922P06445000" in call_kwargs["symbols"]
    assert "SPXW  250922C06445000" in call_kwargs["symbols"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_databento_backfill.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `databento_backfill.py`**

Create `src_v2/backtest/databento_backfill.py`:

```python
"""Targeted Databento CBBO-1m backfill.

Fetches quotes for a narrow strike window around an ATM signal price, in a narrow
time window around the signal minute. Returns a dict in the same shape as the
existing data/cbbo_cache pickles, so the existing CbboStore can read the output
without modification.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from src_v2.backtest.osi_symbol import build_strike_window

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

QuoteKey = Tuple[str, float, bool]
QuoteVal = Tuple[float, float]


def build_request_window(
    session_date: date,
    anchor_minute_et: str,
    before_min: int,
    after_min: int,
) -> Tuple[str, str]:
    """Compute (start_iso_utc, end_iso_utc) for a window centered on anchor_minute_et."""
    hh, mm = int(anchor_minute_et[:2]), int(anchor_minute_et[3:5])
    anchor_et = datetime(session_date.year, session_date.month, session_date.day,
                         hh, mm, tzinfo=ET)
    start_et = anchor_et - timedelta(minutes=before_min)
    end_et = anchor_et + timedelta(minutes=after_min)
    start_utc = start_et.astimezone(UTC)
    end_utc = end_et.astimezone(UTC)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return start_utc.strftime(fmt), end_utc.strftime(fmt)


def _parse_symbol(raw: str) -> Optional[Tuple[date, bool, float]]:
    """'SPXW  250922P06445000' → (date(2025,9,22), False, 6445.0). None if malformed."""
    raw = raw.rstrip()
    if len(raw) < 21 or not raw.startswith("SPXW"):
        return None
    body = raw[6:]  # strip 'SPXW  '
    yy, mm, dd = int(body[0:2]), int(body[2:4]), int(body[4:6])
    cp = body[6]
    strike = int(body[7:15]) / 1000.0
    return date(2000 + yy, mm, dd), cp == "C", strike


def fetch_day_quotes(
    client: Any,
    session_date: date,
    anchor_minute_et: str,
    atm_strike: float,
    half_width_pts: int = 50,
    step_pts: int = 5,
    before_min: int = 2,
    after_min: int = 10,
) -> Dict[QuoteKey, QuoteVal]:
    """One Databento call → {(minute_et, strike, is_call): (bid, ask)} dict.

    `client` should be a databento.Historical() instance (or a mock with the same
    interface).
    """
    symbols = build_strike_window(
        expiry=session_date,
        atm_strike=atm_strike,
        half_width_pts=half_width_pts,
        step_pts=step_pts,
    )
    start_iso, end_iso = build_request_window(
        session_date, anchor_minute_et, before_min, after_min
    )

    response = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        schema="cbbo-1m",
        symbols=symbols,
        stype_in="raw_symbol",
        start=start_iso,
        end=end_iso,
    )
    df = response.to_df()

    quotes: Dict[QuoteKey, QuoteVal] = {}
    if df.empty:
        return quotes

    for ts_utc, row in df.iterrows():
        parsed = _parse_symbol(str(row["symbol"]))
        if parsed is None:
            continue
        _expiry, is_call, strike = parsed
        ts_et = ts_utc.tz_convert(ET)
        minute_str = f"{ts_et.hour:02d}:{ts_et.minute:02d}"
        bid = float(row.get("bid_px_00", 0.0) or 0.0)
        ask = float(row.get("ask_px_00", 0.0) or 0.0)
        if bid <= 0 and ask <= 0:
            continue
        quotes[(minute_str, strike, is_call)] = (bid, ask)

    return quotes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_databento_backfill.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/databento_backfill.py tests/test_databento_backfill.py
git commit -m "feat(backfill): Databento client wrapper for targeted CBBO-1m fetch"
```

---

## Task 7: Cost estimator + ledger

**Files:**
- Create: `src_v2/backtest/databento_cost.py`
- Test: `tests/test_databento_cost.py`

**Why:** `metadata.get_cost` is free to call and returns dollars for an identical-shape `get_range` request. We wrap it so the driver can (a) walk the manifest in dry-run mode and print a total cost estimate before spending anything, and (b) accumulate a per-call ledger during live runs so we always know what we've spent.

- [ ] **Step 1: Write failing tests**

Create `tests/test_databento_cost.py`:

```python
import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src_v2.backtest.databento_cost import (
    estimate_request_cost,
    CostLedger,
)


def test_estimate_request_cost_calls_metadata_get_cost_with_correct_params():
    fake_client = MagicMock()
    fake_client.metadata.get_cost.return_value = 0.0023

    cost = estimate_request_cost(
        client=fake_client,
        symbols=["SPXW  250922P06445000", "SPXW  250922C06445000"],
        start="2025-09-22T13:30:00",
        end="2025-09-22T13:42:00",
    )

    assert cost == pytest.approx(0.0023)
    kwargs = fake_client.metadata.get_cost.call_args.kwargs
    assert kwargs["dataset"] == "OPRA.PILLAR"
    assert kwargs["schema"] == "cbbo-1m"
    assert kwargs["stype_in"] == "raw_symbol"
    assert kwargs["start"] == "2025-09-22T13:30:00"
    assert kwargs["end"] == "2025-09-22T13:42:00"
    assert kwargs["symbols"] == ["SPXW  250922P06445000", "SPXW  250922C06445000"]


def test_cost_ledger_appends_and_totals(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CostLedger(ledger_path)
    ledger.record(date(2025, 9, 22), "live", 0.0023, n_symbols=42, n_quotes=1234)
    ledger.record(date(2025, 9, 23), "live", 0.0019, n_symbols=42, n_quotes=987)

    assert ledger.total_usd() == pytest.approx(0.0042)
    lines = ledger_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["date"] == "2025-09-22"
    assert rec["mode"] == "live"
    assert rec["cost_usd"] == pytest.approx(0.0023)
    assert rec["n_symbols"] == 42
    assert rec["n_quotes"] == 1234


def test_cost_ledger_resumes_total_from_existing_file(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps({"date": "2025-09-22", "mode": "live", "cost_usd": 0.50,
                    "n_symbols": 42, "n_quotes": 100}) + "\n"
    )
    ledger = CostLedger(ledger_path)
    assert ledger.total_usd() == pytest.approx(0.50)

    ledger.record(date(2025, 9, 23), "live", 0.25, n_symbols=42, n_quotes=200)
    assert ledger.total_usd() == pytest.approx(0.75)


def test_cost_ledger_dry_run_records_separately(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CostLedger(ledger_path)
    ledger.record(date(2025, 9, 22), "dry_run", 0.10, n_symbols=42, n_quotes=0)
    # Dry-run records should NOT count toward live total
    assert ledger.total_usd(mode="live") == pytest.approx(0.0)
    assert ledger.total_usd(mode="dry_run") == pytest.approx(0.10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_databento_cost.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `databento_cost.py`**

Create `src_v2/backtest/databento_cost.py`:

```python
"""Cost estimator + spend ledger for Databento backfill.

Wraps metadata.get_cost (free to call) so the driver can dry-run the entire
manifest before fetching anything, and tracks per-call spend during live runs
to a JSONL ledger that survives across restarts.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, List, Optional


def estimate_request_cost(
    client: Any,
    symbols: List[str],
    start: str,
    end: str,
    dataset: str = "OPRA.PILLAR",
    schema: str = "cbbo-1m",
    stype_in: str = "raw_symbol",
) -> float:
    """Returns the expected cost in USD for an identical-shape get_range request.

    Calling metadata.get_cost is free per Databento docs.
    """
    return float(client.metadata.get_cost(
        dataset=dataset,
        schema=schema,
        symbols=symbols,
        start=start,
        end=end,
        stype_in=stype_in,
    ))


class CostLedger:
    """Append-only JSONL log of per-day API costs.

    Records "live" and "dry_run" entries separately so dry-run costs never
    inflate the live total.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._records: List[dict] = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def record(
        self,
        day: date,
        mode: str,
        cost_usd: float,
        n_symbols: int,
        n_quotes: int,
    ) -> None:
        rec = {
            "date": day.isoformat(),
            "mode": mode,
            "cost_usd": float(cost_usd),
            "n_symbols": int(n_symbols),
            "n_quotes": int(n_quotes),
        }
        self._records.append(rec)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def total_usd(self, mode: Optional[str] = "live") -> float:
        if mode is None:
            return sum(r["cost_usd"] for r in self._records)
        return sum(r["cost_usd"] for r in self._records if r["mode"] == mode)

    def n_records(self, mode: Optional[str] = None) -> int:
        if mode is None:
            return len(self._records)
        return sum(1 for r in self._records if r["mode"] == mode)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_databento_cost.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/databento_cost.py tests/test_databento_cost.py
git commit -m "feat(backfill): cost estimator + JSONL spend ledger"
```

---

## Task 8: Backfill driver CLI

**Files:**
- Create: `scripts/databento_backfill.py`

Glue script with three modes:
- `--dry-run`: walks manifest, calls `metadata.get_cost` per day (FREE), prints total and exits without fetching anything
- normal run: estimates cost per day, fetches, accumulates spend in ledger, persists pickle
- `--max-cost-usd N`: aborts mid-run if accumulated live cost exceeds N (safety net)

Idempotent: skips a date when its pickle already exists, unless `--force`.

- [ ] **Step 1: Create the script**

Create `scripts/databento_backfill.py`:

```python
#!/usr/bin/env python3
"""Targeted Databento CBBO-1m backfill driver.

Reads a signal manifest (JSONL produced by enumerate_signals.py), fetches CBBO
quotes for each (date, signal_minute, atm_strike) using the Databento API, and
writes per-day pickles into data/cbbo_cache/ in the format the existing
CbboStore reads.

Modes:
  --dry-run         : call metadata.get_cost (FREE) per day, print total, exit
  (default)         : fetch, persist pickles, append spend to cost ledger
  --max-cost-usd N  : abort mid-run if accumulated live cost exceeds N

Cost control: each request fetches only ~42 OSI symbols across a ~12-minute
window, so the full 5-year backfill should land in the $1-5 range. Always run
--dry-run first against a new manifest to confirm the bill.

Usage:
    # In .env: DATABENTO_API_KEY=db-...
    # 1. Estimate cost (free):
    python scripts/databento_backfill.py \\
        --manifest data/signal_manifest_orb_breakout.jsonl --dry-run
    # 2. Run for real, with safety cap:
    python scripts/databento_backfill.py \\
        --manifest data/signal_manifest_orb_breakout.jsonl \\
        --max-cost-usd 10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import databento as db

from src_v2.backtest.basis_estimator import (
    build_monthly_basis_table,
    estimate_spx_from_es,
)
from src_v2.backtest.databento_backfill import (
    fetch_day_quotes,
    build_request_window,
)
from src_v2.backtest.databento_cost import (
    estimate_request_cost,
    CostLedger,
)
from src_v2.backtest.osi_symbol import build_strike_window


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_manifest(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _es_opens_from_manifest(entries) -> Dict[date, float]:
    """First-seen es_price_at_signal per day — used as anchor for basis estimation.
    Good enough since basis is a monthly median, not per-day."""
    out: Dict[date, float] = {}
    for e in entries:
        d = _parse_date(e["date"])
        out.setdefault(d, e["es_price_at_signal"])
    return out


def _build_request_for_entry(entry, monthly_basis, default_basis, half_width_pts, step_pts,
                             before_min, after_min):
    """Returns (symbols_list, start_iso, end_iso, atm_strike) for one manifest entry."""
    day = _parse_date(entry["date"])
    atm = estimate_spx_from_es(day, entry["es_price_at_signal"], monthly_basis, default_basis)
    atm_snapped = round(atm / step_pts) * step_pts
    symbols = build_strike_window(
        expiry=day, atm_strike=float(atm_snapped),
        half_width_pts=half_width_pts, step_pts=step_pts,
    )
    start_iso, end_iso = build_request_window(
        day, entry["signal_minute_et"], before_min, after_min
    )
    return symbols, start_iso, end_iso, float(atm_snapped)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="data/cbbo_cache")
    p.add_argument("--cbbo-cache", default="data/cbbo_cache",
                   help="Existing CBBO pickles used to seed the basis table")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--before-min", type=int, default=2)
    p.add_argument("--after-min", type=int, default=10)
    p.add_argument("--half-width-pts", type=int, default=50)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--default-basis", type=float, default=10.0)
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if output pickle already exists")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N requests (0 = no limit)")
    p.add_argument("--sleep-ms", type=int, default=0,
                   help="Sleep N ms between requests to stay under rate limits")
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost via metadata.get_cost (FREE) and exit")
    p.add_argument("--max-cost-usd", type=float, default=0.0,
                   help="Abort if accumulated live cost exceeds this (0 = no cap)")
    args = p.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = list(_load_manifest(Path(args.manifest)))
    print(f"Loaded {len(entries)} signal events from manifest")

    es_opens = _es_opens_from_manifest(entries)
    monthly_basis = build_monthly_basis_table(Path(args.cbbo_cache), es_opens)
    print(f"Built basis table for {len(monthly_basis)} months")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    if ledger.total_usd("live") > 0:
        print(f"Existing ledger: ${ledger.total_usd('live'):.4f} live spend, "
              f"{ledger.n_records('live')} records")

    by_day: Dict[date, list] = defaultdict(list)
    for e in entries:
        by_day[_parse_date(e["date"])].append(e)

    mode = "dry_run" if args.dry_run else "live"
    n_calls = 0
    n_skipped = 0
    n_quotes = 0
    run_cost = 0.0

    for day in sorted(by_day.keys()):
        out_path = out_dir / f"{day.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue

        merged: Dict = {}
        day_cost = 0.0
        day_n_symbols = 0
        for entry in by_day[day]:
            symbols, start_iso, end_iso, _atm = _build_request_for_entry(
                entry, monthly_basis, args.default_basis,
                args.half_width_pts, args.step_pts,
                args.before_min, args.after_min,
            )

            try:
                est = estimate_request_cost(
                    client=client, symbols=symbols, start=start_iso, end=end_iso,
                )
            except Exception as exc:
                print(f"  [{day}] cost-estimate FAILED: {exc}", file=sys.stderr)
                continue

            day_cost += est
            day_n_symbols += len(symbols)

            if args.dry_run:
                run_cost += est
                continue

            # Live: enforce cost cap BEFORE making the call
            if args.max_cost_usd > 0 and (run_cost + est) > args.max_cost_usd:
                print(f"\nSTOP: next call would push run cost past --max-cost-usd "
                      f"${args.max_cost_usd:.2f} (current ${run_cost:.4f} + est ${est:.4f})")
                print(f"Run total: ${run_cost:.4f}, ledger total: ${ledger.total_usd('live'):.4f}")
                return 1

            try:
                quotes = fetch_day_quotes(
                    client=client,
                    session_date=day,
                    anchor_minute_et=entry["signal_minute_et"],
                    atm_strike=_atm,
                    half_width_pts=args.half_width_pts,
                    step_pts=args.step_pts,
                    before_min=args.before_min,
                    after_min=args.after_min,
                )
            except Exception as exc:
                print(f"  [{day}] fetch FAILED: {exc}", file=sys.stderr)
                continue

            merged.update(quotes)
            n_quotes += len(quotes)
            run_cost += est
            n_calls += 1
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
            if args.limit and n_calls >= args.limit:
                break

        # Record this day's cost (one ledger entry per day, sums all per-entry costs)
        ledger.record(day, mode, day_cost,
                     n_symbols=day_n_symbols, n_quotes=len(merged))

        if not args.dry_run and merged:
            with open(out_path, "wb") as f:
                pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [{day}] {len(merged):>6} quotes  ${day_cost:.4f}  → {out_path}")
        elif args.dry_run:
            print(f"  [{day}] est ${day_cost:.4f}  ({day_n_symbols} symbols across "
                  f"{len(by_day[day])} signal(s))")

        if args.limit and n_calls >= args.limit:
            print(f"\nReached --limit {args.limit}, stopping")
            break

    if args.dry_run:
        print(f"\nDRY RUN total estimate: ${run_cost:.4f} for {len(by_day)} days "
              f"(saved to {args.ledger} as 'dry_run' records — does NOT count against live total)")
    else:
        print(f"\nDone. {n_calls} API calls, {n_skipped} days skipped (cached), "
              f"{n_quotes} total quotes")
        print(f"Run cost: ${run_cost:.4f} | Ledger live total: "
              f"${ledger.total_usd('live'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the no-key error path**

Run (without DATABENTO_API_KEY in env):
```bash
env -u DATABENTO_API_KEY .venv/bin/python scripts/databento_backfill.py \
  --manifest data/signal_manifest_smoke.jsonl --dry-run
```
Expected: prints `ERROR: DATABENTO_API_KEY env var not set`, exits with code 2.

- [ ] **Step 3: Verify basis-table builds against existing cache**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
import json
from datetime import date
from src_v2.backtest.basis_estimator import build_monthly_basis_table

entries = [json.loads(l) for l in Path('data/signal_manifest_smoke.jsonl').read_text().splitlines()]
es_opens = {}
for e in entries:
    es_opens.setdefault(date.fromisoformat(e['date']), e['es_price_at_signal'])
print('Days in manifest:', len(es_opens))
table = build_monthly_basis_table(Path('data/cbbo_cache'), es_opens)
print('Monthly basis table:', table)
"
```
Expected: non-zero day count and a table covering months in the existing 125-day CBBO range (Sep 2025 – Mar 2026). Each basis value should be in roughly the −10 to +30 pt range; anything outside that is suspicious and worth investigating before any API call.

- [ ] **Step 4: Commit**

```bash
git add scripts/databento_backfill.py
git commit -m "feat(backfill): databento_backfill driver with dry-run cost estimation + ledger"
```

---

## Task 9: End-to-end smoke test + cost report against the live API

**Files:** none new. Operational verification only.

- [ ] **Step 1: Sanity-check Databento credentials (FREE — uses metadata only)**

Run:
```bash
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
import os, databento as db
c = db.Historical(os.environ['DATABENTO_API_KEY'])
# metadata.list_unit_prices is free
prices = c.metadata.list_unit_prices(dataset='OPRA.PILLAR')
for p in prices:
    print(p.get('mode'), p.get('unit_prices'))
"
```
Expected: prints unit prices per schema for OPRA.PILLAR including a `cbbo-1m` rate (likely a small $/GB number). Confirms key works and unit pricing matches expectation.

- [ ] **Step 2: Generate the FULL signal manifest (no API needed)**

Run:
```bash
.venv/bin/python scripts/enumerate_signals.py \
  --es-csv glbx-mdp3-20210227-20260226.ohlcv-1m.csv \
  --config config/config_orb_dir_breakout_spx.yaml \
  --start 2021-03-01 --end 2025-09-21 \
  --out data/signal_manifest_orb_breakout_full.jsonl
```
Expected: prints scanned-day and signal counts, writes JSONL. Signal count should be in the few-hundreds range (~600 expected based on current 30% signal-day rate).

- [ ] **Step 3: DRY RUN — query metadata.get_cost for the entire backfill (FREE)**

Run:
```bash
.venv/bin/python scripts/databento_backfill.py \
  --manifest data/signal_manifest_orb_breakout_full.jsonl \
  --dry-run
```
Expected: per-day cost estimates print, then a final `DRY RUN total estimate: $X.XX for N days` line. **Report this number to the user before proceeding.** If the total exceeds $10 the assumptions in the planning doc are wrong and the user should reconsider before spending.

- [ ] **Step 4: Live fetch ONE signal day as smoke test (sub-1¢ expected)**

Run:
```bash
.venv/bin/python scripts/databento_backfill.py \
  --manifest data/signal_manifest_orb_breakout_full.jsonl \
  --out data/cbbo_cache_test \
  --ledger data/databento_cost_ledger_test.jsonl \
  --limit 1
```
Expected: writes one pickle to `data/cbbo_cache_test/`, ledger reports a tiny dollar amount. Verify with:
```bash
ls -la data/cbbo_cache_test/ && cat data/databento_cost_ledger_test.jsonl
```

- [ ] **Step 5: Validate output format matches existing cache**

Run:
```bash
.venv/bin/python -c "
import pickle, glob
from pathlib import Path
new = pickle.load(open(glob.glob('data/cbbo_cache_test/*.pkl')[0], 'rb'))
existing = pickle.load(open(sorted(Path('data/cbbo_cache').glob('*.pkl'))[0], 'rb'))
new_key, new_val = next(iter(new.items()))
old_key, old_val = next(iter(existing.items()))
print('new :', new_key, '->', new_val)
print('old :', old_key, '->', old_val)
assert type(new_key) is type(old_key) and type(new_val) is type(old_val), 'shape mismatch'
print('OK: shapes match')
"
```
Expected: prints `OK: shapes match`. If not, the SDK response shape differs from what Task 6 expects — fix Task 6's `_parse_symbol`/parsing logic and re-test.

- [ ] **Step 6: Run the full backfill (real money — should match dry-run estimate)**

Run with safety cap based on dry-run total:
```bash
.venv/bin/python scripts/databento_backfill.py \
  --manifest data/signal_manifest_orb_breakout_full.jsonl \
  --max-cost-usd 15 \
  --sleep-ms 100
```
Expected: final line shows total API calls, quotes, and the actual cost — which should be within ±10% of the dry-run estimate from Step 3. Per-day pickles populate `data/cbbo_cache/` covering 2021-2025.

- [ ] **Step 7: Re-run a backtest with the expanded cache**

Update `config/config_orb_dir_breakout_spx.yaml` `start_date` to `"2021-03-01"`, then:
```bash
.venv/bin/python scripts/run_v2_backtest.py --config config/config_orb_dir_breakout_spx.yaml
```
Expected: trade count 3-5× larger than the current 50-row [trades.csv](results/orb_dir_breakout_spx/trades.csv); WF-CV becomes statistically meaningful.

- [ ] **Step 8: Commit config tweaks**

```bash
git add config/config_orb_dir_breakout_spx.yaml
git commit -m "config(orb_breakout): expand backtest range to full 2021-2026 history"
```

---

## Notes for the implementer

- **TDD discipline:** Tasks 2-6 are pure-function logic with full unit-test coverage. Don't skip the failing-test step — it catches typos in the test fixtures themselves.
- **Mocking the Databento client:** Tests in Task 6 use `MagicMock` and a hand-built `pandas.DataFrame` shaped like what `response.to_df()` actually returns for OPRA cbbo-1m. If the real SDK returns a different shape (e.g. different column names than `bid_px_00` / `ask_px_00`), Task 8 Step 1 will surface it — fix Task 6 to match the real shape and re-run its tests.
- **Idempotence:** The driver script in Task 7 skips days whose pickle already exists. This means re-running after a partial failure is safe and cheap. Use `--force` to refetch.
- **Cost guardrail:** Always run with `--limit 1` first when pointing at a new manifest. The unit test in Task 6 confirms each call sends ~42 symbols × ~12 min, but a misconfigured manifest could have thousands of entries.
- **Rate limits:** Databento allows generous burst rates but `--sleep-ms 100` keeps us under any reasonable limit during the full 600-call backfill.
- **Don't touch CbboStore:** The existing reader at [src_v2/backtest/cbbo_store.py](src_v2/backtest/cbbo_store.py) loads pickles whose values are `(bid, ask)` tuples keyed by `(minute_str, strike_float, is_call_bool)`. Task 6's `fetch_day_quotes` produces exactly that shape. If you find yourself wanting to change CbboStore, stop — the contract is in the keys/values, and Task 8 Step 3 verifies it.
