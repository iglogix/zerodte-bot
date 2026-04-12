# src_v2 Backtest Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained backtest engine in `src_v2/backtest/` that runs the actual `LiveStrategy` code against historical SPY bars + SPX CBBO quotes, starting with the `OrbEmIronCondorStrategy`.

**Architecture:** A `BacktestEngine` replays historical 1-min bars day-by-day, detects ORB breakout + midline retest, builds a CBBO-sourced option chain, and feeds it to the real `LiveStrategy.search_candidates()` / `check_exit()` methods. Data loading (SpyLoader, CbboStore) is self-contained in `src_v2/backtest/` with no `src_8am` imports.

**Tech Stack:** Python 3.11+, dataclasses, csv, pickle, zoneinfo. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-11-v2-backtest-engine-design.md`

---

### Task 1: DayBars dataclass + SpyLoader

**Files:**
- Create: `src_v2/backtest/__init__.py`
- Create: `src_v2/backtest/data_loader.py`
- Create: `tests_v2/backtest/__init__.py`
- Create: `tests_v2/backtest/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests_v2/backtest/__init__.py` (empty) and `tests_v2/backtest/test_data_loader.py`:

```python
"""Tests for src_v2.backtest.data_loader."""

import csv
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


@pytest.fixture
def spy_csv(tmp_path):
    """Create a minimal SPY 1-min CSV with 2 days of data."""
    csv_path = tmp_path / "spy_1m.csv"
    rows = [
        # Day 1: 2025-10-01, two bars (UTC timestamps)
        {"ts_event": "2025-10-01T12:00:00.000000000Z", "symbol": "SPY",
         "open": "570.00", "high": "570.50", "low": "569.50", "close": "570.25", "volume": "1000"},
        {"ts_event": "2025-10-01T12:01:00.000000000Z", "symbol": "SPY",
         "open": "570.25", "high": "570.75", "low": "570.00", "close": "570.50", "volume": "1200"},
        # Day 2: 2025-10-02, one bar
        {"ts_event": "2025-10-02T12:00:00.000000000Z", "symbol": "SPY",
         "open": "571.00", "high": "571.50", "low": "570.50", "close": "571.25", "volume": "800"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path)


def test_daybars_repr():
    from src_v2.backtest.data_loader import DayBars
    db = DayBars(date(2025, 10, 1), "SPX", [{"close": 1}])
    assert "2025-10-01" in repr(db)
    assert "1 bars" in repr(db)


def test_spy_loader_iter_days(spy_csv):
    from src_v2.backtest.data_loader import SpyLoader
    loader = SpyLoader(spy_csv, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days())
    assert len(days) == 2

    # Day 1: 2 bars, prices scaled by 10
    d1 = days[0]
    assert d1.session_date == date(2025, 10, 1)
    assert d1.symbol == "SPX"
    assert len(d1.bars) == 2
    assert d1.bars[0]["open"] == pytest.approx(5700.0)
    assert d1.bars[0]["close"] == pytest.approx(5702.5)
    # Timestamps should be ET-aware
    assert d1.bars[0]["ts_et"].tzinfo is not None
    assert d1.bars[0]["ts_et"].hour == 8  # 12:00 UTC = 08:00 ET (Oct, EDT)


def test_spy_loader_date_filter(spy_csv):
    from src_v2.backtest.data_loader import SpyLoader
    loader = SpyLoader(spy_csv, price_scale=10.0)
    days = list(loader.iter_days(start_date=date(2025, 10, 2)))
    assert len(days) == 1
    assert days[0].session_date == date(2025, 10, 2)


def test_spy_loader_missing_file():
    from src_v2.backtest.data_loader import SpyLoader
    with pytest.raises(FileNotFoundError):
        SpyLoader("/nonexistent/path.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src_v2.backtest'`

- [ ] **Step 3: Write implementation**

Create `src_v2/backtest/__init__.py` (empty file).

Create `src_v2/backtest/data_loader.py`:

```python
"""
SPY data loader — reads SPY 1-min CSV and returns per-day DayBars in ET.

Prices are scaled by price_scale (default 10.0: SPY × 10 ≈ SPX).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp to UTC datetime.

    Handles:
      - 2024-01-02T14:30:00.000000000Z  (nanoseconds)
      - 2024-01-02T14:30:00Z             (standard)
    """
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        base, frac = ts_str.split(".", 1)
        ts_str = f"{base}.{frac[:6]}"
    return datetime.fromisoformat(ts_str).replace(tzinfo=UTC)


class DayBars:
    """Container for one trading day's 1-min bars."""

    __slots__ = ("session_date", "symbol", "bars")

    def __init__(self, session_date: date, symbol: str, bars: List[Dict]):
        self.session_date = session_date
        self.symbol = symbol
        self.bars = bars  # list of dicts: ts_et, open, high, low, close, volume

    def __repr__(self) -> str:
        return f"DayBars({self.session_date}, {self.symbol}, {len(self.bars)} bars)"


class SpyLoader:
    """Loads SPY 1-min OHLCV CSV and yields DayBars objects.

    Prices scaled by price_scale (10.0 → SPY×10 ≈ SPX).
    """

    def __init__(self, data_path: str, price_scale: float = 10.0, symbol: str = "SPX"):
        self.data_path = Path(data_path)
        self._price_scale = price_scale
        self._symbol = symbol
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")

    def iter_days(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Iterator[DayBars]:
        buf: Dict[date, List[Dict]] = defaultdict(list)
        prev_date: Optional[date] = None

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_utc = _parse_ts(row["ts_event"])
                except Exception:
                    continue

                ts_et = ts_utc.astimezone(ET)
                bar_date = ts_et.date()

                if start_date and bar_date < start_date:
                    continue
                if end_date and bar_date > end_date:
                    break

                if prev_date is not None and bar_date > prev_date:
                    for completed in sorted(d for d in buf if d < bar_date):
                        if buf[completed]:
                            yield DayBars(
                                completed, self._symbol,
                                sorted(buf[completed], key=lambda b: b["ts_et"]),
                            )
                        del buf[completed]

                prev_date = bar_date

                try:
                    s = self._price_scale
                    buf[bar_date].append({
                        "ts_et": ts_et,
                        "open": float(row["open"]) * s,
                        "high": float(row["high"]) * s,
                        "low": float(row["low"]) * s,
                        "close": float(row["close"]) * s,
                        "volume": int(row.get("volume", 0)),
                    })
                except (ValueError, KeyError):
                    continue

        for remaining in sorted(buf.keys()):
            if buf[remaining]:
                yield DayBars(
                    remaining, self._symbol,
                    sorted(buf[remaining], key=lambda b: b["ts_et"]),
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_data_loader.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/__init__.py src_v2/backtest/data_loader.py \
       tests_v2/backtest/__init__.py tests_v2/backtest/test_data_loader.py
git commit -m "feat(v2-backtest): add DayBars + SpyLoader data loader"
```

---

### Task 2: CbboStore

**Files:**
- Create: `src_v2/backtest/cbbo_store.py`
- Create: `tests_v2/backtest/test_cbbo_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests_v2/backtest/test_cbbo_store.py`:

```python
"""Tests for src_v2.backtest.cbbo_store."""

import pickle
from datetime import date
from pathlib import Path

import pytest


@pytest.fixture
def cbbo_cache(tmp_path):
    """Create a minimal pickle cache with 1 day of CBBO quotes."""
    # Key format: (minute_str, strike, is_call) -> (bid, ask)
    quotes = {
        ("09:45", 5900.0, False): (3.50, 4.00),   # put 5900
        ("09:45", 5895.0, False): (1.50, 2.00),   # put 5895
        ("09:45", 5950.0, True): (3.00, 3.50),    # call 5950
        ("09:45", 5955.0, True): (1.00, 1.50),    # call 5955
        ("09:45", 5925.0, False): (12.00, 13.00), # ATM put
        ("09:45", 5925.0, True): (13.00, 14.00),  # ATM call
        ("10:00", 5900.0, False): (2.00, 2.50),   # later minute
        ("10:00", 5895.0, False): (0.80, 1.20),
        ("10:00", 5950.0, True): (2.00, 2.50),
        ("10:00", 5955.0, True): (0.50, 1.00),
        ("10:00", 5925.0, False): (10.00, 11.00),
        ("10:00", 5925.0, True): (11.00, 12.00),
    }
    pkl_path = tmp_path / "2025-10-01.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(quotes, f)
    return str(tmp_path)


def test_load_day(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    n = store.load_day(date(2025, 10, 1))
    assert n == 12
    assert store.loaded_date == date(2025, 10, 1)


def test_lookup(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))

    bid, ask = store.lookup("09:45", 5900.0, False)
    assert bid == 3.50
    assert ask == 4.00

    assert store.lookup("09:45", 9999.0, False) is None


def test_lookup_stale_fallback(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))

    # "09:46" doesn't exist, should fall back to "09:45"
    result = store.lookup("09:46", 5900.0, False, max_stale_minutes=5)
    assert result == (3.50, 4.00)


def test_atm_straddle(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))

    em = store.atm_straddle("09:45", spot=5925.0)
    # put_mid=12.5, call_mid=13.5
    assert em == pytest.approx(26.0)


def test_derive_spot(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))

    # S = K + C_mid - P_mid = 5925 + 13.5 - 12.5 = 5926.0
    spot = store.derive_spot("09:45", bar_price=5925.0)
    assert spot == pytest.approx(5926.0)


def test_available_strikes(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    store.load_day(date(2025, 10, 1))
    assert 5900.0 in store.available_strikes
    assert 5925.0 in store.available_strikes


def test_preload_cache(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    mem = CbboStore.preload_cache(cbbo_cache)
    assert date(2025, 10, 1) in mem
    store = CbboStore(cbbo_cache, preloaded=mem)
    n = store.load_day(date(2025, 10, 1))
    assert n == 12


def test_load_missing_day(cbbo_cache):
    from src_v2.backtest.cbbo_store import CbboStore
    store = CbboStore(cbbo_cache, cache_dir=cbbo_cache)
    n = store.load_day(date(2099, 1, 1))
    assert n == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_cbbo_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `src_v2/backtest/cbbo_store.py`:

```python
"""
CBBO Price Store — loads SPX option CBBO quotes from pickle cache for backtesting.

Provides real bid/ask quotes per (minute, strike, is_call) for 0DTE options.
Loads one day at a time to keep memory bounded.

Usage:
    store = CbboStore("data/cbbo_cache", cache_dir="data/cbbo_cache")
    store.load_day(date(2025, 9, 22))
    bid, ask = store.lookup("09:45", 5900.0, False)
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

# Key: (minute_str "HH:MM", strike, is_call) -> (bid, ask)
QuoteMap = Dict[Tuple[str, float, bool], Tuple[float, float]]


class CbboStore:
    """Loads and serves 0DTE SPX option quotes from pickle cache files."""

    def __init__(self, opra_dir: str | Path, cache_dir: str | Path = "",
                 preloaded: Optional[Dict] = None):
        self.opra_dir = Path(opra_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._quotes: QuoteMap = {}
        self._loaded_date: Optional[date] = None
        self._strikes: set[float] = set()
        self._mem_cache: Optional[Dict] = preloaded

    @staticmethod
    def preload_cache(cache_dir: str | Path) -> Dict:
        """Load all pickle files into memory. Returns {date: QuoteMap}."""
        cache_path = Path(cache_dir)
        mem: Dict = {}
        for pkl in sorted(cache_path.glob("*.pkl")):
            d = date.fromisoformat(pkl.stem)
            with open(pkl, "rb") as f:
                mem[d] = pickle.load(f)
        return mem

    @property
    def loaded_date(self) -> Optional[date]:
        return self._loaded_date

    @property
    def available_strikes(self) -> set[float]:
        return self._strikes

    def load_day(self, trading_date: date) -> int:
        """Load 0DTE quotes for a single trading day. Returns record count."""
        self._quotes = {}
        self._strikes = set()
        self._loaded_date = trading_date

        if self._mem_cache is not None and trading_date in self._mem_cache:
            self._quotes = self._mem_cache[trading_date]
            self._strikes = {k[1] for k in self._quotes}
            return len(self._quotes)

        if self.cache_dir:
            pkl_path = self.cache_dir / f"{trading_date.isoformat()}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    self._quotes = pickle.load(f)
                self._strikes = {k[1] for k in self._quotes}
                return len(self._quotes)

        return 0

    def lookup(
        self,
        minute_et: str,
        strike: float,
        is_call: bool,
        max_stale_minutes: int = 5,
    ) -> Optional[Tuple[float, float]]:
        """Look up (bid, ask) for given minute/strike/type.

        Searches back up to max_stale_minutes if exact minute not found.
        """
        key = (minute_et, strike, is_call)
        if key in self._quotes:
            return self._quotes[key]

        hh, mm = int(minute_et[:2]), int(minute_et[3:5])
        for offset in range(1, max_stale_minutes + 1):
            prev_mm = mm - offset
            prev_hh = hh
            while prev_mm < 0:
                prev_mm += 60
                prev_hh -= 1
            if prev_hh < 0:
                break
            prev_key = (f"{prev_hh:02d}:{prev_mm:02d}", strike, is_call)
            if prev_key in self._quotes:
                return self._quotes[prev_key]

        return None

    def atm_straddle(
        self,
        minute_et: str,
        spot: float,
        max_stale_minutes: int = 5,
    ) -> Optional[float]:
        """ATM straddle price (put_mid + call_mid at nearest strike to spot)."""
        if not self._strikes:
            return None
        atm_k = min(self._strikes, key=lambda k: abs(k - spot))
        put_q = self.lookup(minute_et, atm_k, False, max_stale_minutes)
        call_q = self.lookup(minute_et, atm_k, True, max_stale_minutes)
        if put_q is None or call_q is None:
            return None
        return (put_q[0] + put_q[1]) / 2 + (call_q[0] + call_q[1]) / 2

    def derive_spot(
        self,
        minute_et: str,
        bar_price: float,
        max_stale_minutes: int = 5,
    ) -> Optional[float]:
        """Derive index spot from put-call parity: S = K + C_mid - P_mid."""
        if not self._strikes:
            return None
        atm_k = min(self._strikes, key=lambda k: abs(k - bar_price))
        put_q = self.lookup(minute_et, atm_k, False, max_stale_minutes)
        call_q = self.lookup(minute_et, atm_k, True, max_stale_minutes)
        if put_q is None or call_q is None:
            return None
        c_mid = (call_q[0] + call_q[1]) / 2
        p_mid = (put_q[0] + put_q[1]) / 2
        return round(atm_k + c_mid - p_mid, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_cbbo_store.py -v`
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/cbbo_store.py tests_v2/backtest/test_cbbo_store.py
git commit -m "feat(v2-backtest): add CbboStore for CBBO quote lookups"
```

---

### Task 3: Report generator

**Files:**
- Create: `src_v2/backtest/report.py`
- Create: `tests_v2/backtest/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests_v2/backtest/test_report.py`:

```python
"""Tests for src_v2.backtest.report."""

from datetime import date
import pytest


def _make_result(**overrides):
    from src_v2.backtest.engine import BacktestResult
    defaults = dict(
        trade_id="T1", session_date=date(2025, 10, 1), symbol="SPX",
        orb_high=5930, orb_low=5920, orb_midline=5925, orb_range_pts=10,
        status="closed",
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def test_empty_report():
    from src_v2.backtest.report import generate_report
    report = generate_report([])
    assert report["trades_placed"] == 0
    assert report["win_rate"] == 0.0


def test_basic_stats():
    from src_v2.backtest.report import generate_report

    results = [
        _make_result(trade_id="T1", realized_pnl_usd=100.0, exit_reason="tp_50pct",
                     credit_received=1.0, short_delta_at_entry=0.10, bias="long"),
        _make_result(trade_id="T2", realized_pnl_usd=80.0, exit_reason="expired_worthless",
                     credit_received=0.80, short_delta_at_entry=0.12, bias="long"),
        _make_result(trade_id="T3", realized_pnl_usd=-200.0, exit_reason="short_breached",
                     credit_received=1.20, short_delta_at_entry=0.15, bias="short",
                     session_date=date(2025, 10, 2)),
    ]

    report = generate_report(results)
    assert report["trades_placed"] == 3
    assert report["wins"] == 2
    assert report["losses"] == 1
    assert report["win_rate"] == pytest.approx(0.6667, abs=0.001)
    assert report["total_pnl_usd"] == pytest.approx(-20.0)
    assert report["profit_factor"] == pytest.approx(0.9)
    assert report["max_drawdown_usd"] == pytest.approx(200.0)
    assert report["by_exit_reason"]["tp_50pct"]["count"] == 1


def test_sharpe_calculation():
    from src_v2.backtest.report import generate_report

    # Two days, same day trades produce daily PnL
    results = [
        _make_result(trade_id="T1", session_date=date(2025, 10, 1),
                     realized_pnl_usd=50.0, exit_reason="tp", bias="long"),
        _make_result(trade_id="T2", session_date=date(2025, 10, 1),
                     realized_pnl_usd=50.0, exit_reason="tp", bias="long"),
        _make_result(trade_id="T3", session_date=date(2025, 10, 2),
                     realized_pnl_usd=100.0, exit_reason="tp", bias="short"),
    ]
    report = generate_report(results)
    assert report["sharpe_annualised"] > 0


def test_skipped_results():
    from src_v2.backtest.report import generate_report

    results = [
        _make_result(trade_id="S1", status="skipped", skip_reason="no_breakout"),
        _make_result(trade_id="S2", status="skipped", skip_reason="no_breakout"),
        _make_result(trade_id="S3", status="skipped", skip_reason="range_too_narrow"),
    ]
    report = generate_report(results)
    assert report["trades_placed"] == 0
    assert report["skipped_days"] == 3
    assert report["skip_reasons"]["no_breakout"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Note: `BacktestResult` will be defined in `engine.py` (Task 4), but we need it here for tests. Create the dataclass in `engine.py` first (just the dataclass, not the engine class):

Create `src_v2/backtest/engine.py` with only the dataclass for now:

```python
"""
src_v2 Backtest Engine — runs LiveStrategy code against historical data.

Full engine implementation in Task 4. This file starts with the result dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class BacktestResult:
    trade_id: str
    session_date: date
    symbol: str

    # ORB
    orb_high: float
    orb_low: float
    orb_midline: float
    orb_range_pts: float

    bias: Optional[str] = None
    breakout_time: Optional[datetime] = None
    breakout_price: Optional[float] = None

    # Spread
    spread_type: Optional[str] = None
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None
    spread_width: float = 0.0

    # Entry
    entry_time: Optional[datetime] = None
    underlying_at_entry: Optional[float] = None
    credit_received: Optional[float] = None
    short_delta_at_entry: Optional[float] = None

    # Exit
    exit_time: Optional[datetime] = None
    underlying_at_exit: Optional[float] = None
    debit_paid: Optional[float] = None
    exit_reason: Optional[str] = None

    realized_pnl_usd: Optional[float] = None

    status: str = "skipped"
    skip_reason: Optional[str] = None

    qty: int = 1
```

Now create `src_v2/backtest/report.py`:

```python
"""
Backtest report — P&L summary for src_v2 strategy backtests.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from src_v2.backtest.engine import BacktestResult


def generate_report(results: List[BacktestResult]) -> Dict[str, Any]:
    closed = [r for r in results if r.status == "closed"]
    skipped = [r for r in results if r.status == "skipped"]

    if not closed:
        return {
            "total_days": len(results),
            "trades_placed": 0,
            "skipped_days": len(skipped),
            "win_rate": 0.0,
            "total_pnl_usd": 0.0,
            "skip_reasons": _count_reasons(skipped),
        }

    wins = [r for r in closed if (r.realized_pnl_usd or 0) > 0]
    losses = [r for r in closed if (r.realized_pnl_usd or 0) <= 0]

    total_pnl = sum(r.realized_pnl_usd or 0 for r in closed)
    win_rate = len(wins) / len(closed) if closed else 0.0

    avg_win = sum(r.realized_pnl_usd or 0 for r in wins) / len(wins) if wins else 0.0
    avg_loss = sum(r.realized_pnl_usd or 0 for r in losses) / len(losses) if losses else 0.0

    gross_profit = sum(r.realized_pnl_usd or 0 for r in wins)
    gross_loss = abs(sum(r.realized_pnl_usd or 0 for r in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in closed:
        equity += r.realized_pnl_usd or 0
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    # Sharpe (annualised, daily returns, risk-free=0)
    daily_pnl: Dict[str, float] = defaultdict(float)
    for r in closed:
        daily_pnl[str(r.session_date)] += r.realized_pnl_usd or 0
    daily_returns = list(daily_pnl.values())
    sharpe = 0.0
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0

    # By exit reason
    by_reason: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for r in closed:
        by_reason[r.exit_reason or "unknown"]["count"] += 1
        by_reason[r.exit_reason or "unknown"]["pnl"] += r.realized_pnl_usd or 0

    # By bias
    by_bias: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
    for r in closed:
        b = r.bias or "unknown"
        by_bias[b]["count"] += 1
        if (r.realized_pnl_usd or 0) > 0:
            by_bias[b]["wins"] += 1
        by_bias[b]["pnl"] += r.realized_pnl_usd or 0

    credits = [r.credit_received for r in closed if r.credit_received]
    avg_credit = sum(credits) / len(credits) if credits else 0.0

    deltas = [r.short_delta_at_entry for r in closed if r.short_delta_at_entry]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

    return {
        "total_days": len(results),
        "trades_placed": len(closed),
        "skipped_days": len(skipped),
        "win_rate": round(win_rate, 4),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl_usd": round(total_pnl, 2),
        "avg_pnl_per_trade_usd": round(total_pnl / len(closed), 2),
        "avg_winner_usd": round(avg_win, 2),
        "avg_loser_usd": round(avg_loss, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "max_drawdown_usd": round(max_dd, 2),
        "sharpe_annualised": round(sharpe, 2),
        "avg_credit_received": round(avg_credit, 3),
        "avg_short_delta": round(avg_delta, 4),
        "by_exit_reason": {k: {"count": v["count"], "pnl_usd": round(v["pnl"], 2)}
                           for k, v in sorted(by_reason.items())},
        "by_bias": {k: {
            "count": v["count"],
            "win_rate": round(v["wins"] / v["count"], 3) if v["count"] else 0,
            "pnl_usd": round(v["pnl"], 2),
        } for k, v in sorted(by_bias.items())},
        "skip_reasons": _count_reasons(skipped),
    }


def print_report(report: Dict[str, Any], title: str = "src_v2 Backtest") -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Total days:        {report['total_days']}")
    print(f"  Trades placed:     {report['trades_placed']}")
    print(f"  Skipped days:      {report['skipped_days']}")
    if report["trades_placed"] == 0:
        print(sep)
        return
    print(f"  Win rate:          {report['win_rate']*100:.1f}%  "
          f"({report.get('wins',0)}W / {report.get('losses',0)}L)")
    print(f"  Total P&L:         ${report['total_pnl_usd']:,.2f}")
    print(f"  Avg P&L/trade:     ${report['avg_pnl_per_trade_usd']:,.2f}")
    print(f"  Avg winner:        ${report['avg_winner_usd']:,.2f}")
    print(f"  Avg loser:         ${report['avg_loser_usd']:,.2f}")
    print(f"  Profit factor:     {report['profit_factor']}")
    print(f"  Max drawdown:      ${report['max_drawdown_usd']:,.2f}")
    print(f"  Sharpe (ann.):     {report['sharpe_annualised']:.2f}")
    print(f"  Avg credit:        ${report['avg_credit_received']:.3f}")
    print(f"  Avg short delta:   {report['avg_short_delta']:.4f}")

    if report.get("by_exit_reason"):
        print()
        print("  By exit reason:")
        for reason, v in report["by_exit_reason"].items():
            print(f"    {reason:<28} {v['count']:>4} trades  ${v['pnl_usd']:>9,.2f}")

    if report.get("by_bias"):
        print()
        print("  By bias:")
        for bias, v in report["by_bias"].items():
            print(f"    {bias:<8} {v['count']:>4} trades  WR={v['win_rate']*100:.0f}%  ${v['pnl_usd']:>9,.2f}")

    if report.get("skip_reasons"):
        print()
        print("  Skip reasons:")
        for reason, count in sorted(report["skip_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<40} {count:>4}")

    print(sep)


def _count_reasons(skipped: List[BacktestResult]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in skipped:
        counts[r.skip_reason or "unknown"] += 1
    return dict(counts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_report.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/engine.py src_v2/backtest/report.py tests_v2/backtest/test_report.py
git commit -m "feat(v2-backtest): add BacktestResult dataclass and report generator"
```

---

### Task 4: BacktestEngine — ORB detection + entry via LiveStrategy

**Files:**
- Modify: `src_v2/backtest/engine.py`
- Create: `tests_v2/backtest/test_engine.py`

This is the core task. The engine detects ORB, finds breakout + midline retest, builds chain from CBBO, and calls `strategy.search_candidates()` / `strategy.check_exit()`.

- [ ] **Step 1: Write the failing test**

Create `tests_v2/backtest/test_engine.py`:

```python
"""Tests for src_v2.backtest.engine.BacktestEngine."""

import asyncio
import pickle
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def _make_bar(hour, minute, open_, high, low, close, volume=100, d=date(2025, 10, 1)):
    return {
        "ts_et": datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }


def _make_orb_bars(d=date(2025, 10, 1)):
    """ORB candle: 08:00-08:14, range = 5920-5930, midline = 5925."""
    bars = []
    for m in range(0, 15):
        bars.append(_make_bar(8, m, 5924, 5930, 5920, 5925, d=d))
    return bars


def _make_breakout_bars(bias="long", d=date(2025, 10, 1)):
    """Post-open bars: breakout + midline retest + fill + monitoring bars."""
    bars = []
    if bias == "long":
        # 09:30: open inside range
        bars.append(_make_bar(9, 30, 5925, 5928, 5922, 5926, d=d))
        # 09:31: breakout above 5930
        bars.append(_make_bar(9, 31, 5926, 5935, 5926, 5932, d=d))
        # 09:32: midline retest (touches 5925)
        bars.append(_make_bar(9, 32, 5932, 5932, 5924, 5926, d=d))
        # 09:33: fill bar (entry at open)
        bars.append(_make_bar(9, 33, 5927, 5935, 5926, 5933, d=d))
        # 09:34-09:40: monitoring bars (price stays safe)
        for m in range(34, 41):
            bars.append(_make_bar(9, m, 5933, 5938, 5930, 5935, d=d))
    return bars


def _make_eod_bar(d=date(2025, 10, 1)):
    """15:50 bar triggers EOD."""
    return _make_bar(15, 50, 5935, 5936, 5934, 5935, d=d)


@pytest.fixture
def cbbo_cache(tmp_path):
    """CBBO cache with quotes needed for orb_em_ic strategy."""
    quotes = {}
    # Need quotes at "09:33" (fill minute) and monitoring minutes
    for minute in ["09:33", "09:34", "09:35", "09:36", "09:37", "09:38", "09:39", "09:40", "15:50"]:
        # ATM straddle at 5925 (for EM computation)
        quotes[(minute, 5925.0, False)] = (12.0, 13.0)  # ATM put
        quotes[(minute, 5925.0, True)] = (13.0, 14.0)   # ATM call
        # Put spread strikes (short=5910, long=5905 with 0.5×EM~12.5)
        quotes[(minute, 5910.0, False)] = (4.00, 4.50)
        quotes[(minute, 5905.0, False)] = (2.50, 3.00)
        # Call spread strikes (short=5945, long=5955 with 0.75×EM~18.75)
        quotes[(minute, 5945.0, True)] = (3.50, 4.00)
        quotes[(minute, 5955.0, True)] = (1.50, 2.00)
        # Wide range of strikes for chain building
        for k in range(5800, 6100, 5):
            if (minute, float(k), False) not in quotes:
                quotes[(minute, float(k), False)] = (0.05, 0.15)
            if (minute, float(k), True) not in quotes:
                quotes[(minute, float(k), True)] = (0.05, 0.15)

    pkl_path = tmp_path / "2025-10-01.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(quotes, f)
    return str(tmp_path)


@pytest.fixture
def base_config(cbbo_cache):
    return {
        "bot": {"symbol": "ES", "strategy": "orb_em_ic"},
        "strategy": {
            "orb_window_start": "08:00",
            "orb_window_end": "08:15",
            "market_open": "09:30",
            "eod_force_close": "15:50",
            "midline_tolerance_pts": 1.0,
            "orb_range_filter": {"min_pts": 2.0, "max_pts": 30.0},
            "skip_nfp_fridays": False,
            "skip_fomc_days": False,
            "skip_cpi_days": False,
        },
        "options": {
            "option_root": "SPXW",
            "strike_step_pts": 5,
            "min_credit_per_spread": 0.05,
            "tp_pct_of_credit": 0.50,
            "sl_mult_of_credit": 0,
            "hold_put_breach": True,
            "em_with_trend": 0.50,
            "em_against_trend": 0.75,
            "width_with_trend": 5,
            "width_against_trend": 10,
            "slippage_per_spread": 0.10,
            "contract_multiplier": 100,
            "quantity": 1,
        },
        "market": {
            "trading_days_per_year": 252,
            "trading_hours_per_day": 6.5,
            "market_close": "16:00",
        },
        "risk": {"max_loss_per_trade_usd": 50000},
        "backtest": {"cbbo_cache_dir": cbbo_cache},
    }


def test_skip_no_orb_data(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)
    # Day with no bars in ORB window
    day = DayBars(date(2025, 10, 1), "SPX", [_make_bar(10, 0, 5925, 5930, 5920, 5925)])
    results = engine.run([day])
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].skip_reason == "no_orb_data"


def test_skip_range_too_narrow(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)
    # ORB range = 1 pt (below min_pts=2)
    bars = [_make_bar(8, m, 5925, 5925.5, 5924.5, 5925) for m in range(15)]
    bars.append(_make_bar(9, 30, 5925, 5926, 5924, 5925))
    day = DayBars(date(2025, 10, 1), "SPX", bars)
    results = engine.run([day])
    assert results[0].status == "skipped"
    assert "range_too_narrow" in results[0].skip_reason


def test_skip_no_breakout(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)
    orb = _make_orb_bars()
    # Post-open bars stay inside range
    post = [_make_bar(9, 30+m, 5925, 5928, 5922, 5925) for m in range(10)]
    post.append(_make_eod_bar())
    day = DayBars(date(2025, 10, 1), "SPX", orb + post)
    results = engine.run([day])
    assert results[0].status == "skipped"
    assert results[0].skip_reason == "no_breakout"


def test_full_trade_expired_worthless(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)

    orb = _make_orb_bars()
    post = _make_breakout_bars("long")
    post.append(_make_eod_bar())
    day = DayBars(date(2025, 10, 1), "SPX", orb + post)

    results = engine.run([day])
    # IC produces 2 results (put side + call side)
    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2

    for r in closed:
        assert r.bias == "long"
        assert r.credit_received is not None
        assert r.credit_received > 0
        assert r.exit_reason in ("expired_worthless", "tp_50pct")
        if r.exit_reason == "expired_worthless":
            assert r.debit_paid == 0.0
            assert r.realized_pnl_usd > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_engine.py -v`
Expected: FAIL — `BacktestEngine` not defined (only `BacktestResult` exists)

- [ ] **Step 3: Write implementation**

Add the full `BacktestEngine` class to `src_v2/backtest/engine.py` (append after `BacktestResult`):

```python
# --- Add these imports at top of file ---
import asyncio
import uuid
from datetime import time
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from src_v2.backtest.cbbo_store import CbboStore
from src_v2.strategies.base import LiveStrategy, LiveTradeSetup, ExitAction

ET = ZoneInfo("America/New_York")


# --- FOMC / CPI / NFP event dates ---

_FOMC_DATES = {
    date(2021,1,27), date(2021,3,17), date(2021,4,28), date(2021,6,16),
    date(2021,7,28), date(2021,9,22), date(2021,11,3), date(2021,12,15),
    date(2022,1,26), date(2022,3,16), date(2022,5,4), date(2022,6,15),
    date(2022,7,27), date(2022,9,21), date(2022,11,2), date(2022,12,14),
    date(2023,2,1), date(2023,3,22), date(2023,5,3), date(2023,6,14),
    date(2023,7,26), date(2023,9,20), date(2023,11,1), date(2023,12,13),
    date(2024,1,31), date(2024,3,20), date(2024,5,1), date(2024,6,12),
    date(2024,7,31), date(2024,9,18), date(2024,11,7), date(2024,12,18),
    date(2025,1,29), date(2025,3,19), date(2025,5,7), date(2025,6,18),
    date(2025,7,30), date(2025,9,17), date(2025,10,29), date(2025,12,10),
    date(2026,1,28), date(2026,3,18), date(2026,4,29), date(2026,6,17),
    date(2026,7,29), date(2026,9,16), date(2026,10,28), date(2026,12,9),
}

_CPI_DATES = {
    date(2021,1,13), date(2021,2,10), date(2021,3,10), date(2021,4,13),
    date(2021,5,12), date(2021,6,10), date(2021,7,13), date(2021,8,11),
    date(2021,9,14), date(2021,10,13), date(2021,11,10), date(2021,12,10),
    date(2022,1,12), date(2022,2,10), date(2022,3,10), date(2022,4,12),
    date(2022,5,11), date(2022,6,10), date(2022,7,13), date(2022,8,10),
    date(2022,9,13), date(2022,10,13), date(2022,11,10), date(2022,12,13),
    date(2023,1,12), date(2023,2,14), date(2023,3,14), date(2023,4,12),
    date(2023,5,10), date(2023,6,13), date(2023,7,12), date(2023,8,10),
    date(2023,9,13), date(2023,10,12), date(2023,11,14), date(2023,12,12),
    date(2024,1,11), date(2024,2,13), date(2024,3,12), date(2024,4,10),
    date(2024,5,15), date(2024,6,12), date(2024,7,11), date(2024,8,14),
    date(2024,9,11), date(2024,10,10), date(2024,11,13), date(2024,12,11),
    date(2025,1,15), date(2025,2,12), date(2025,3,12), date(2025,4,10),
    date(2025,5,13), date(2025,6,11), date(2025,7,15), date(2025,8,12),
    date(2025,9,11), date(2025,10,24), date(2025,12,18),
    date(2026,1,13), date(2026,2,13), date(2026,3,11), date(2026,4,10),
    date(2026,5,12), date(2026,6,10), date(2026,7,14), date(2026,8,12),
    date(2026,9,11), date(2026,10,14), date(2026,11,10), date(2026,12,10),
}


class BacktestEngine:
    """Runs LiveStrategy against historical SPY bars + CBBO quotes.

    Currently supports ORB-based entry only (breakout → retest → fill).
    """

    def __init__(self, config: Dict[str, Any], strategy: LiveStrategy):
        self._strategy = strategy
        self._strategy.configure(config)

        strat = config.get("strategy", {})
        self._orb_start = _parse_time(strat.get("orb_window_start", "08:00"))
        self._orb_end = _parse_time(strat.get("orb_window_end", "08:15"))
        self._market_open = _parse_time(strat.get("market_open", "09:30"))
        self._eod = _parse_time(strat.get("eod_force_close", "15:50"))

        range_cfg = strat.get("orb_range_filter", {})
        self._min_range = range_cfg.get("min_pts", 2.0)
        self._max_range = range_cfg.get("max_pts", 30.0)
        self._midline_tol = strat.get("midline_tolerance_pts", 0.50)

        self._skip_nfp = strat.get("skip_nfp_fridays", False)
        self._skip_fomc = strat.get("skip_fomc_days", False)
        self._skip_cpi = strat.get("skip_cpi_days", False)

        opt = config.get("options", {})
        self._multiplier = opt.get("contract_multiplier", 100)
        self._qty = opt.get("quantity", 1)

        market = config.get("market", {})
        self._market_close = _parse_time(market.get("market_close", "16:00"))

        bt = config.get("backtest", {})
        cbbo_dir = bt.get("cbbo_cache_dir", "")
        self._cbbo = CbboStore(cbbo_dir, cache_dir=cbbo_dir) if cbbo_dir else None

    def run(self, days: List) -> List[BacktestResult]:
        """Run backtest on list of DayBars. Returns all results (closed + skipped)."""
        results = []
        for day in sorted(days, key=lambda d: d.session_date):
            sd = day.session_date
            if self._skip_nfp and self._is_nfp_friday(sd):
                continue
            if self._skip_fomc and sd in _FOMC_DATES:
                continue
            if self._skip_cpi and sd in _CPI_DATES:
                continue

            day_results = self._process_day(day)
            if isinstance(day_results, list):
                results.extend(r for r in day_results if r is not None)
            elif day_results is not None:
                results.append(day_results)
        return results

    def _skip_result(self, sd: date, symbol: str, reason: str,
                     orb_high=0, orb_low=0, orb_midline=0, orb_range=0,
                     bias=None, breakout_bar=None) -> BacktestResult:
        return BacktestResult(
            trade_id=f"V2-{sd}-SKIP",
            session_date=sd, symbol=symbol,
            orb_high=orb_high, orb_low=orb_low,
            orb_midline=orb_midline, orb_range_pts=orb_range,
            bias=bias,
            breakout_time=breakout_bar["ts_et"] if breakout_bar else None,
            breakout_price=breakout_bar["close"] if breakout_bar else None,
            status="skipped", skip_reason=reason,
        )

    def _process_day(self, day_bars) -> List[BacktestResult]:
        bars = day_bars.bars
        sd = day_bars.session_date
        sym = day_bars.symbol

        # Load CBBO
        if self._cbbo is not None:
            n = self._cbbo.load_day(sd)
            if n == 0:
                return [self._skip_result(sd, sym, "no_cbbo_data")]

        # Step 1: ORB candle
        orb_bars = [b for b in bars if self._orb_start <= b["ts_et"].time() < self._orb_end]
        if not orb_bars:
            return [self._skip_result(sd, sym, "no_orb_data")]

        orb_high = max(b["high"] for b in orb_bars)
        orb_low = min(b["low"] for b in orb_bars)
        orb_midline = (orb_high + orb_low) / 2
        orb_range = round(orb_high - orb_low, 4)

        if orb_range < self._min_range:
            return [self._skip_result(sd, sym, f"range_too_narrow_{orb_range}pts",
                                      orb_high, orb_low, orb_midline, orb_range)]
        if orb_range > self._max_range:
            return [self._skip_result(sd, sym, f"range_too_wide_{orb_range}pts",
                                      orb_high, orb_low, orb_midline, orb_range)]

        # Step 2: Post-open bars
        post_open = [b for b in bars if b["ts_et"].time() >= self._market_open]
        if not post_open:
            return [self._skip_result(sd, sym, "no_post_open_bars",
                                      orb_high, orb_low, orb_midline, orb_range)]

        # Step 3: Breakout
        breakout_bar = None
        bias = None
        for bar in post_open:
            if bar["ts_et"].time() >= self._eod:
                break
            if bar["close"] > orb_high:
                bias = "long"
                breakout_bar = bar
                break
            if bar["close"] < orb_low:
                bias = "short"
                breakout_bar = bar
                break

        if breakout_bar is None:
            return [self._skip_result(sd, sym, "no_breakout",
                                      orb_high, orb_low, orb_midline, orb_range)]

        post_breakout = post_open[post_open.index(breakout_bar):]

        # Step 4: Midline retest
        retest_bar = None
        for bar in post_breakout:
            if bar["ts_et"].time() >= self._eod:
                break
            if (bar["low"] <= orb_midline + self._midline_tol and
                    bar["high"] >= orb_midline - self._midline_tol):
                retest_bar = bar
                break

        if retest_bar is None:
            return [self._skip_result(sd, sym, "no_retest",
                                      orb_high, orb_low, orb_midline, orb_range,
                                      bias, breakout_bar)]

        # Step 5: Fill bar
        retest_idx = post_breakout.index(retest_bar)
        post_retest = post_breakout[retest_idx + 1:]
        if not post_retest:
            return [self._skip_result(sd, sym, "no_bar_after_retest",
                                      orb_high, orb_low, orb_midline, orb_range,
                                      bias, breakout_bar)]

        fill_bar = post_retest[0]
        entry_price = fill_bar["open"]
        entry_time = fill_bar["ts_et"]
        minute_et = entry_time.strftime("%H:%M")

        # Step 6: Derive spot from CBBO
        spot = entry_price
        if self._cbbo is not None:
            derived = self._cbbo.derive_spot(minute_et, entry_price)
            if derived is not None:
                spot = derived

        # Step 7: Build chain from CBBO
        chain = self._build_chain(minute_et, spot, sd)
        if not chain:
            return [self._skip_result(sd, sym, "no_chain_data",
                                      orb_high, orb_low, orb_midline, orb_range,
                                      bias, breakout_bar)]

        # Step 8: Call strategy
        self._strategy._orb_bias = bias
        candidates = asyncio.get_event_loop().run_until_complete(
            self._strategy.search_candidates(chain, spot)
        ) if asyncio.get_event_loop().is_running() else _run_async(
            self._strategy.search_candidates(chain, spot)
        )

        setup: LiveTradeSetup = candidates.get("top_candidate")
        if setup is None:
            return [self._skip_result(sd, sym, "strategy_rejected",
                                      orb_high, orb_low, orb_midline, orb_range,
                                      bias, breakout_bar)]

        # Step 9: Validate
        validation = _run_async(self._strategy.validate(setup, chain))
        if not validation.get("passed"):
            return [self._skip_result(sd, sym, f"validation_failed:{validation.get('failures')}",
                                      orb_high, orb_low, orb_midline, orb_range,
                                      bias, breakout_bar)]

        # Step 10: Simulate exit for each side of the IC
        monitoring_bars = post_retest[1:]  # bars after fill
        put_credit = setup.extra.get("put_credit", 0)
        call_credit = setup.extra.get("call_credit", 0)

        # Build trade dict for check_exit()
        trade = {
            "short_put_symbol": setup.legs[0].symbol,
            "long_put_symbol": setup.legs[1].symbol,
            "short_call_symbol": setup.legs[2].symbol,
            "long_call_symbol": setup.legs[3].symbol,
            "short_put_strike": setup.strikes["short_put"],
            "short_call_strike": setup.strikes["short_call"],
            "put_credit": put_credit,
            "call_credit": call_credit,
        }

        put_result = self._simulate_side(
            trade, setup, monitoring_bars, fill_bar, "put",
            setup.legs[0], setup.legs[1],
            put_credit, sd, sym, orb_high, orb_low, orb_midline, orb_range,
            bias, breakout_bar, entry_time, spot,
        )

        call_result = self._simulate_side(
            trade, setup, monitoring_bars, fill_bar, "call",
            setup.legs[2], setup.legs[3],
            call_credit, sd, sym, orb_high, orb_low, orb_midline, orb_range,
            bias, breakout_bar, entry_time, spot,
        )

        return [put_result, call_result]

    def _simulate_side(
        self,
        trade: Dict,
        setup: LiveTradeSetup,
        monitoring_bars: List[Dict],
        fill_bar: Dict,
        side: str,
        short_leg,
        long_leg,
        credit: float,
        sd, sym, orb_high, orb_low, orb_midline, orb_range,
        bias, breakout_bar, entry_time, spot,
    ) -> BacktestResult:
        """Simulate one side (put or call) of the IC through exit."""
        spread_type = "put_credit_spread" if side == "put" else "call_credit_spread"
        width = abs(short_leg.strike - long_leg.strike)

        # Walk bars for exit
        check_bars = [fill_bar] + list(monitoring_bars)
        exit_time = None
        exit_price = None
        exit_reason = None
        debit = 0.0

        for bar in check_bars:
            if bar["ts_et"].time() >= self._eod:
                exit_time = bar["ts_et"]
                exit_price = bar["close"]
                exit_reason = "expired_worthless"
                debit = 0.0
                break

            minute = bar["ts_et"].strftime("%H:%M")

            # Derive bar spot from CBBO if available
            bar_spot = bar["close"]
            if self._cbbo is not None:
                derived = self._cbbo.derive_spot(minute, bar["close"])
                if derived is not None:
                    bar_spot = derived

            # Build quotes for check_exit
            quotes = self._build_quotes(minute, setup.legs, bar_spot)
            action = self._strategy.check_exit(trade, quotes)

            if action.action == "hold":
                continue

            # Check if this action targets our side
            our_symbols = {short_leg.symbol, long_leg.symbol}
            action_symbols = set(action.legs_to_close)
            if not our_symbols.intersection(action_symbols):
                continue

            exit_time = bar["ts_et"]
            exit_price = bar_spot

            if action.action == "take_profit":
                exit_reason = f"tp_{int(self._strategy._tp_pct * 100)}pct"
                debit = round(credit * (1 - self._strategy._tp_pct), 2)
            elif action.action == "stop_loss":
                exit_reason = "short_breached"
                # Compute debit from CBBO mark
                short_q = self._cbbo.lookup(minute, short_leg.strike,
                                            short_leg.option_type == "call") if self._cbbo else None
                long_q = self._cbbo.lookup(minute, long_leg.strike,
                                           long_leg.option_type == "call") if self._cbbo else None
                if short_q and long_q:
                    short_mid = (short_q[0] + short_q[1]) / 2
                    long_mid = (long_q[0] + long_q[1]) / 2
                    debit = round(min(short_mid - long_mid, width), 2)
                else:
                    debit = width  # worst case
            break

        if exit_reason is None:
            # No exit found — expire worthless at last bar
            last = check_bars[-1] if check_bars else fill_bar
            exit_time = last["ts_et"]
            exit_price = last["close"]
            exit_reason = "expired_worthless"
            debit = 0.0

        pnl = round((credit - debit) * self._multiplier * self._qty, 2)

        return BacktestResult(
            trade_id=f"V2-{sd}-{uuid.uuid4().hex[:6].upper()}-{side[0].upper()}",
            session_date=sd, symbol=sym,
            orb_high=orb_high, orb_low=orb_low,
            orb_midline=orb_midline, orb_range_pts=orb_range,
            bias=bias,
            breakout_time=breakout_bar["ts_et"],
            breakout_price=breakout_bar["close"],
            spread_type=spread_type,
            short_strike=short_leg.strike,
            long_strike=long_leg.strike,
            spread_width=width,
            entry_time=entry_time,
            underlying_at_entry=spot,
            credit_received=credit,
            short_delta_at_entry=round(setup.extra.get(f"{side}_delta", 0), 4),
            exit_time=exit_time,
            underlying_at_exit=exit_price,
            debit_paid=debit,
            exit_reason=exit_reason,
            realized_pnl_usd=pnl,
            status="closed",
            qty=self._qty,
        )

    def _build_chain(self, minute: str, spot: float, session_date: date) -> List[Dict]:
        """Build option chain from CBBO store at given minute."""
        if self._cbbo is None:
            return []
        chain = []
        for strike in sorted(self._cbbo.available_strikes):
            if abs(strike - spot) > 200:
                continue
            for is_call in [False, True]:
                q = self._cbbo.lookup(minute, strike, is_call)
                if q is None:
                    continue
                bid, ask = q
                if bid <= 0 and ask <= 0:
                    continue
                sym = self._make_symbol(session_date, strike, is_call)
                chain.append({
                    "strike": strike,
                    "option_type": "call" if is_call else "put",
                    "bid": bid,
                    "ask": ask,
                    "symbol": sym,
                    "delta": 0,
                })
        return chain

    def _build_quotes(self, minute: str, legs, underlying_price: float) -> Dict:
        """Build quotes dict for strategy.check_exit()."""
        quotes: Dict[str, Any] = {}
        for leg in legs:
            is_call = leg.option_type == "call"
            q = self._cbbo.lookup(minute, leg.strike, is_call) if self._cbbo else None
            if q:
                quotes[leg.symbol] = {"bid": q[0], "ask": q[1]}
            else:
                quotes[leg.symbol] = {"bid": 0.01, "ask": 0.01}
        quotes["_underlying"] = {"price": underlying_price}
        return quotes

    @staticmethod
    def _make_symbol(session_date: date, strike: float, is_call: bool) -> str:
        """Build OCC-style symbol: SPXW YYMMDDCSSSSSSSS"""
        cp = "C" if is_call else "P"
        yy = session_date.strftime("%y%m%d")
        strike_int = int(strike * 1000)
        return f"SPXW  {yy}{cp}{strike_int:08d}"

    @staticmethod
    def _is_nfp_friday(sd: date) -> bool:
        return sd.weekday() == 4 and sd.day <= 7


def _parse_time(s: str) -> time:
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


def _run_async(coro):
    """Run an async coroutine synchronously (for backtest context)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests_v2/backtest/test_engine.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/engine.py tests_v2/backtest/test_engine.py
git commit -m "feat(v2-backtest): add BacktestEngine with ORB entry + exit simulation"
```

---

### Task 5: CLI runner script

**Files:**
- Create: `scripts/run_v2_backtest.py`

- [ ] **Step 1: Write the script**

Create `scripts/run_v2_backtest.py`:

```python
"""
Run src_v2 backtest with current config and print report.

Usage:
  python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml
  python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml --start 2025-10-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import SpyLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.engine import BacktestEngine
from src_v2.backtest.report import generate_report, print_report


def _load_strategy(config):
    """Instantiate the strategy from config."""
    name = config.get("bot", {}).get("strategy", "orb_em_ic")
    if name == "orb_em_ic":
        from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy
        return OrbEmIronCondorStrategy()
    else:
        raise ValueError(f"Unsupported strategy for backtest: {name}")


def main():
    parser = argparse.ArgumentParser(description="src_v2 Backtest Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bt = config.get("backtest", {})
    data_file = bt.get("data_file", "data/spy_1m.csv")
    price_scale = bt.get("price_scale", 10.0)
    start = date.fromisoformat(args.start) if args.start else date.fromisoformat(bt.get("start_date", "2025-09-22"))
    end = date.fromisoformat(args.end) if args.end else date.fromisoformat(bt.get("end_date", "2026-03-20"))

    symbol = config.get("bot", {}).get("symbol", "SPX")

    # Preload CBBO cache
    cbbo_dir = bt.get("cbbo_cache_dir", "")
    if cbbo_dir:
        print(f"Preloading CBBO cache from {cbbo_dir}...")
        mem = CbboStore.preload_cache(cbbo_dir)
        print(f"  {len(mem)} days loaded into memory")
        # Inject preloaded cache into config for engine
        config.setdefault("_cbbo_preloaded", mem)

    # Load bars
    print(f"Loading bars from {data_file} (scale={price_scale})...")
    loader = SpyLoader(data_file, price_scale=price_scale, symbol=symbol)
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} trading days")

    # Create strategy + engine
    strategy = _load_strategy(config)
    engine = BacktestEngine(config, strategy)

    # If we preloaded CBBO, inject into engine's store
    if cbbo_dir and "_cbbo_preloaded" in config:
        engine._cbbo = CbboStore(cbbo_dir, preloaded=config["_cbbo_preloaded"])

    # Run
    print(f"Running backtest {start} -> {end}...")
    results = engine.run(days)

    # Report
    report = generate_report(results)
    print_report(report, title=f"src_v2 Backtest: {Path(args.config).stem}")

    return report


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add backtest section to config**

Add the following to the end of `config/config_orb_em_ic_spx.yaml`:

```yaml

backtest:
  data_file: "data/spy_1m.csv"
  data_source: "spy"
  price_scale: 10.0
  cbbo_cache_dir: "data/cbbo_cache"
  start_date: "2025-09-22"
  end_date: "2026-03-20"
```

- [ ] **Step 3: Run the backtest end-to-end**

Run: `.venv/bin/python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml`

Expected output: Report with ~72 trades, ~93% WR, ~14 Sharpe (should closely match src_8am results). Any divergence reveals bugs to investigate.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v2_backtest.py config/config_orb_em_ic_spx.yaml
git commit -m "feat(v2-backtest): add CLI runner and config backtest section"
```

---

### Task 6: Validation — compare src_v2 vs src_8am results

This task is a manual verification step, not code.

- [ ] **Step 1: Run src_8am backtest**

Run: `.venv/bin/python -m scripts.run_backtest_chart --config config/config_8am_spx_options_em.yaml`

Record: trades, WR, Sharpe, PF, exit breakdown.

- [ ] **Step 2: Run src_v2 backtest**

Run: `.venv/bin/python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml`

Record: trades, WR, Sharpe, PF, exit breakdown.

- [ ] **Step 3: Compare and investigate divergence**

If numbers match closely (within ~5% due to minor implementation differences like spot derivation timing), the v2 backtest is validated.

If they diverge significantly, check:
1. Are the same days being skipped? (Compare skip reasons)
2. Are entry prices the same? (spot derivation differences)
3. Are exits triggering at the same bars? (check_exit logic differences)
4. Are credits matching? (chain building / slippage differences)

Fix any bugs found and re-run until results converge.

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests_v2/backtest/ -v`
Expected: All tests PASS

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat(v2-backtest): complete v2 backtest engine — validated against src_8am"
```
