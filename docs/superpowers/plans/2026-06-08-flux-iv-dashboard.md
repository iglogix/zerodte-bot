# Flux IV Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small web dashboard that shows SPX front-IV / back-IV / their ratio (intraday + 5-day + 20-day) plus an expiry-pair scanner, reconstructing Steve Bernich's "Flux" tool, sourced from marketdata.app.

**Architecture:** A pure-Python, fully unit-tested data/compute layer (`models → sources → engine → cache → history → scanner → live`) behind a thin Flask JSON shell, with a single plotly.js page. Data from marketdata.app (IV + delta returned directly — no Black–Scholes). Historical-first (EOD `date=` snapshots, offline-testable) with a live-ready `FluxSource` seam.

**Tech Stack:** Python 3, pytest, Flask (NEW dependency — installed in Task 1), urllib, SQLite, plotly.js (CDN). Run everything with the project venv: `.venv/bin/python` / `.venv/bin/pytest`.

**Design reference:** `docs/superpowers/specs/2026-06-08-flux-iv-dashboard-design.md`.
**Branch:** `feat/flux` (already created; spec committed there).

**marketdata.app facts (verified) the engineer must know:**
- Endpoint `GET https://api.marketdata.app/v1/options/chain/SPX/?expiration=YYYY-MM-DD[&date=YYYY-MM-DD]&strikeLimit=25&columns=optionSymbol,side,strike,bid,ask,iv,delta,dte,updated`, header `Authorization: Bearer <MARKET_DATA_TOKEN>`.
- Response is **columnar**: `{"side":["call",...],"strike":[...],"bid":[...],"ask":[...],"iv":[...],"delta":[...],"updated":[<unix>,...], ...}`. `iv` is decimal (0.176 = 17.6%), `delta` signed (calls +, puts −), `updated` is a unix timestamp, `side` is `"call"`/`"put"`.
- When `columns=` is set the `s` status key is **absent**; detect no-data by an **empty/missing `strike` list**. HTTP **200 and 203** are both valid (203 = delayed feed).
- `&date=D` → EOD snapshot for D (`updated` = 16:00 ET). No `date` → live/delayed.
- Token lives in `MARKET_DATA_TOKEN` env, else `../options_api/.env` (existing repo convention).
- Bills per option returned → always pass `strikeLimit=25`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src_v2/flux/__init__.py` | Package marker |
| `src_v2/flux/models.py` | `OptionRow`, `ChainSnapshot`, `RatioPoint`, `ScanRow` dataclasses |
| `src_v2/flux/sources/__init__.py` | sub-package marker |
| `src_v2/flux/sources/base.py` | `FluxSource` ABC (`chain` abstract, `expirations` concrete via calendar) |
| `src_v2/flux/sources/marketdata.py` | `MarketDataSource` + token loader + columnar parser |
| `src_v2/flux/engine.py` | `atm_iv(chain, ref_delta)`, `ratio_point(front, back, ref_delta)` |
| `src_v2/flux/cache.py` | `ChainCache` (SQLite) for EOD snapshots |
| `src_v2/flux/history.py` | `daily_ratio_series(...)` |
| `src_v2/flux/scanner.py` | `scan(...)` → ranked `ScanRow`s |
| `src_v2/flux/live.py` | `LivePoller` intraday buffer (market-hours only) |
| `src_v2/flux/app.py` | `create_app(source)` Flask JSON endpoints + `/` |
| `src_v2/flux/static/flux.html` | plotly.js dashboard |
| `src_v2/flux/cli.py` + `__main__.py` | `python -m src_v2.flux serve|scan` |
| `src_v2/flux/README.md` | usage / token / cost notes |
| `tests/flux/test_*.py` | one test module per source module |

**Conventions (defined once):** IV is decimal; `delta` signed; `ref_delta` is a positive magnitude (we match calls to `+ref_delta`, puts to `−ref_delta`). A *ratio* > 1 = backwardation. Run with the project venv.

**Reuse:** `from src_v2.backtest.dc_time_machine.iv import front_back_ratio`; `from src_v2.backtest.dc_time_machine.flux_scanner import ratio_zscore, rank_candidates`; `from src_v2.backtest.dc_time_machine.trading_calendar import sessions_between, nth_trading_day_after`.

---

## Task 1: Package scaffold + Flask dependency + models

**Files:**
- Create: `src_v2/flux/__init__.py`, `src_v2/flux/sources/__init__.py`, `src_v2/flux/models.py`
- Test: `tests/flux/test_models.py`

- [ ] **Step 1: Install Flask into the venv**

Run: `.venv/bin/pip install flask`
Expected: `Successfully installed flask-...` (and add `flask` to `pyproject.toml` `[project].dependencies` if present).

- [ ] **Step 2: Write the failing test**

```python
# tests/flux/test_models.py
from datetime import date, datetime, timezone

from src_v2.flux.models import OptionRow, ChainSnapshot, RatioPoint, ScanRow


def test_optionrow_and_chain():
    r = OptionRow(strike=7000.0, is_call=True, bid=40.0, ask=41.0, iv=0.18, delta=0.52)
    c = ChainSnapshot(underlying="SPX", expiry=date(2026, 6, 12),
                      as_of=datetime(2026, 6, 8, 20, 0, tzinfo=timezone.utc), rows=(r,))
    assert c.rows[0].iv == 0.18 and c.expiry == date(2026, 6, 12)


def test_ratiopoint_and_scanrow_fields():
    rp = RatioPoint(as_of=datetime(2026, 6, 8, tzinfo=timezone.utc),
                    front_expiry=date(2026, 6, 12), back_expiry=date(2026, 6, 15),
                    ref_delta=0.35, front_iv=0.22, back_iv=0.20, ratio=1.1)
    assert rp.ratio == 1.1
    sr = ScanRow(front_expiry=date(2026, 6, 12), back_expiry=date(2026, 6, 15), gap_td=1,
                 ratio=1.1, ratio_prev=1.15, drop=0.05, zscore=1.3, flag=True)
    assert sr.drop == 0.05 and sr.flag is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: src_v2.flux`.

- [ ] **Step 4: Create the package + models**

```python
# src_v2/flux/__init__.py
"""Flux — SPX IV term-structure dashboard."""
```
```python
# src_v2/flux/sources/__init__.py
"""Flux data sources."""
```
```python
# src_v2/flux/models.py
"""Immutable data types for Flux."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Tuple


@dataclass(frozen=True)
class OptionRow:
    strike: float
    is_call: bool
    bid: float
    ask: float
    iv: float       # decimal (0.18 = 18%)
    delta: float    # signed: calls +, puts −


@dataclass(frozen=True)
class ChainSnapshot:
    underlying: str
    expiry: date
    as_of: datetime
    rows: Tuple[OptionRow, ...]


@dataclass(frozen=True)
class RatioPoint:
    as_of: datetime
    front_expiry: date
    back_expiry: date
    ref_delta: float
    front_iv: float
    back_iv: float
    ratio: float    # front_iv / back_iv


@dataclass(frozen=True)
class ScanRow:
    front_expiry: date
    back_expiry: date
    gap_td: int
    ratio: float
    ratio_prev: float
    drop: float     # ratio_prev − ratio (positive = reverting down)
    zscore: float
    flag: bool      # elevated (>1) AND reverting (drop>0)
```

- [ ] **Step 5: Run test + commit**

Run: `.venv/bin/pytest tests/flux/test_models.py -v` → PASS (2 passed).
```bash
git add src_v2/flux/__init__.py src_v2/flux/sources/__init__.py src_v2/flux/models.py tests/flux/test_models.py pyproject.toml
git commit -m "feat(flux): package scaffold + dataclasses (+ flask dep)"
```

---

## Task 2: marketdata.app parser + token loader

**Files:**
- Create: `src_v2/flux/sources/marketdata.py`
- Test: `tests/flux/test_marketdata.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_marketdata.py
from datetime import date

from src_v2.flux.sources.marketdata import MarketDataSource

# Columnar response like marketdata.app returns (columns= set, so NO "s" key)
CANNED = {
    "side": ["call", "put", "call"],
    "strike": [7000.0, 7000.0, 7010.0],
    "bid": [40.0, 38.0, 35.0],
    "ask": [41.0, 39.0, 36.0],
    "iv": [0.180, 0.190, None],          # last row has null iv → dropped
    "delta": [0.52, -0.48, 0.47],
    "updated": [1780940400, 1780940400, 1780940400],
}


def test_parse_columnar_into_chainsnapshot():
    src = MarketDataSource(token="x", fetch_json=lambda url: CANNED)
    snap = src.chain(date(2026, 6, 12))
    assert snap.underlying == "SPX" and snap.expiry == date(2026, 6, 12)
    assert len(snap.rows) == 2                       # null-iv row dropped
    call = [r for r in snap.rows if r.is_call][0]
    assert call.strike == 7000.0 and call.iv == 0.180 and call.delta == 0.52
    put = [r for r in snap.rows if not r.is_call][0]
    assert put.delta == -0.48


def test_url_includes_strikelimit_columns_and_date():
    seen = {}
    def fake(url):
        seen["url"] = url
        return CANNED
    src = MarketDataSource(token="tok", fetch_json=fake)
    src.chain(date(2026, 6, 12), on_date=date(2026, 4, 9))
    u = seen["url"]
    assert "/options/chain/SPX/" in u
    assert "expiration=2026-06-12" in u
    assert "date=2026-04-09" in u
    assert "strikeLimit=25" in u
    assert "columns=" in u


def test_empty_response_is_empty_chain_not_error():
    src = MarketDataSource(token="x", fetch_json=lambda url: {})
    snap = src.chain(date(2026, 6, 12))
    assert snap.rows == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_marketdata.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/sources/marketdata.py
"""marketdata.app adapter — returns IV/delta directly (no Black–Scholes)."""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from src_v2.flux.models import ChainSnapshot, OptionRow
from src_v2.flux.sources.base import FluxSource

BASE = "https://api.marketdata.app/v1"
COLUMNS = "optionSymbol,side,strike,bid,ask,iv,delta,dte,updated"


def load_token() -> Optional[str]:
    """MARKET_DATA_TOKEN from env, else ../options_api/.env (existing repo convention)."""
    t = os.getenv("MARKET_DATA_TOKEN")
    if t:
        return t
    for p in (Path("../options_api/.env"), Path(".env")):
        try:
            for line in p.read_text().splitlines():
                if line.startswith("MARKET_DATA_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            continue
    return None


def _default_fetch(token: str) -> Callable[[str], dict]:
    def fetch(url: str) -> dict:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            return json.loads(body) if body else {}
    return fetch


class MarketDataSource(FluxSource):
    def __init__(self, token: Optional[str] = None,
                 fetch_json: Optional[Callable[[str], dict]] = None,
                 strike_limit: int = 25, underlying: str = "SPX"):
        self.token = token or load_token()
        self.strike_limit = strike_limit
        self.underlying = underlying
        self._fetch = fetch_json or _default_fetch(self.token or "")

    def chain(self, expiry: date, on_date: Optional[date] = None) -> ChainSnapshot:
        url = (f"{BASE}/options/chain/{self.underlying}/?expiration={expiry.isoformat()}"
               f"&strikeLimit={self.strike_limit}&columns={COLUMNS}")
        if on_date is not None:
            url += f"&date={on_date.isoformat()}"
        data = self._fetch(url) or {}
        strikes = data.get("strike") or []
        rows: List[OptionRow] = []
        as_of = datetime.now(timezone.utc)
        for i in range(len(strikes)):
            iv = data["iv"][i] if data.get("iv") else None
            dl = data["delta"][i] if data.get("delta") else None
            if iv is None or dl is None:
                continue
            rows.append(OptionRow(
                strike=float(strikes[i]),
                is_call=(data["side"][i] == "call"),
                bid=float(data["bid"][i]) if data["bid"][i] is not None else 0.0,
                ask=float(data["ask"][i]) if data["ask"][i] is not None else 0.0,
                iv=float(iv), delta=float(dl)))
            if data.get("updated") and data["updated"][i]:
                as_of = datetime.fromtimestamp(int(data["updated"][i]), tz=timezone.utc)
        return ChainSnapshot(self.underlying, expiry, as_of, tuple(rows))
```

- [ ] **Step 4: Run test + commit**

Run: `.venv/bin/pytest tests/flux/test_marketdata.py -v` → expect 3 passed (after Task 3 base exists; if `FluxSource` import fails, do Task 3 first then re-run — they are paired).

> Note: `marketdata.py` imports `FluxSource` from Task 3. Implement Task 3's `base.py` in the same commit if the import fails.

```bash
git add src_v2/flux/sources/marketdata.py tests/flux/test_marketdata.py
git commit -m "feat(flux): marketdata.app columnar parser + token loader"
```

---

## Task 3: FluxSource base (chain abstract, expirations via calendar)

**Files:**
- Create: `src_v2/flux/sources/base.py`
- Test: `tests/flux/test_source_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_source_base.py
from datetime import date

from src_v2.flux.models import ChainSnapshot
from src_v2.flux.sources.base import FluxSource


class _Fake(FluxSource):
    def chain(self, expiry, on_date=None):
        return ChainSnapshot("SPX", expiry, None, ())


def test_expirations_are_calendar_days_after_ref():
    # ref Mon 2026-06-08; 1..3 trading days out = 06-09,06-10,06-11
    exps = _Fake().expirations(date(2026, 6, 8), min_dte=1, max_dte=3)
    assert exps == [date(2026, 6, 9), date(2026, 6, 10), date(2026, 6, 11)]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_source_base.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/sources/base.py
"""FluxSource interface. chain() is provider-specific; expirations() is derived from the
NYSE calendar (SPX has near-daily expiries; a chain() that returns empty == not an expiry)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

from src_v2.flux.models import ChainSnapshot
from src_v2.backtest.dc_time_machine.trading_calendar import nth_trading_day_after


class FluxSource(ABC):
    @abstractmethod
    def chain(self, expiry: date, on_date: Optional[date] = None) -> ChainSnapshot:
        ...

    def expirations(self, ref: date, min_dte: int, max_dte: int) -> List[date]:
        out: List[date] = []
        for n in range(max(0, min_dte), max_dte + 1):
            d = ref if n == 0 else nth_trading_day_after(ref, n)
            if d is not None:
                out.append(d)
        return out
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_source_base.py tests/flux/test_marketdata.py -v` → all pass.
```bash
git add src_v2/flux/sources/base.py tests/flux/test_source_base.py
git commit -m "feat(flux): FluxSource base (calendar-derived expirations)"
```

---

## Task 4: Engine — atm_iv + ratio_point

**Files:**
- Create: `src_v2/flux/engine.py`
- Test: `tests/flux/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_engine.py
from datetime import date, datetime, timezone

import pytest

from src_v2.flux.models import OptionRow, ChainSnapshot
from src_v2.flux.engine import atm_iv, ratio_point


def _chain(expiry, rows):
    return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), tuple(rows))


def _rows():
    # calls at delta .60/.50/.35 ; puts at -.60/-.50/-.35
    return [
        OptionRow(6990, True, 1, 1, 0.205, 0.60), OptionRow(7000, True, 1, 1, 0.200, 0.50),
        OptionRow(7030, True, 1, 1, 0.230, 0.35),
        OptionRow(7010, False, 1, 1, 0.215, -0.60), OptionRow(7000, False, 1, 1, 0.210, -0.50),
        OptionRow(6970, False, 1, 1, 0.250, -0.35),
    ]


def test_atm_iv_at_50_delta_averages_call_and_put():
    c = _chain(date(2026, 6, 12), _rows())
    # nearest |delta|→0.50: call .200, put .210 → avg .205
    assert atm_iv(c, ref_delta=0.50) == pytest.approx(0.205)


def test_atm_iv_at_35_delta_picks_wing_strikes():
    c = _chain(date(2026, 6, 12), _rows())
    # nearest to 0.35: call .230, put .250 → avg .240
    assert atm_iv(c, ref_delta=0.35) == pytest.approx(0.240)


def test_atm_iv_none_when_empty():
    assert atm_iv(_chain(date(2026, 6, 12), []), 0.35) is None


def test_ratio_point():
    front = _chain(date(2026, 6, 12), _rows())
    back = _chain(date(2026, 6, 15), [OptionRow(7000, True, 1, 1, 0.18, 0.50),
                                      OptionRow(7000, False, 1, 1, 0.18, -0.50)])
    rp = ratio_point(front, back, ref_delta=0.50)
    assert rp.front_iv == pytest.approx(0.205) and rp.back_iv == pytest.approx(0.18)
    assert rp.ratio == pytest.approx(0.205 / 0.18)
    assert rp.front_expiry == date(2026, 6, 12) and rp.back_expiry == date(2026, 6, 15)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_engine.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/engine.py
"""Read 'the' front/back IV at a reference delta off each expiration's skew, and the ratio."""
from __future__ import annotations

from typing import Optional

from src_v2.flux.models import ChainSnapshot, RatioPoint
from src_v2.backtest.dc_time_machine.iv import front_back_ratio


def atm_iv(chain: ChainSnapshot, ref_delta: float = 0.35) -> Optional[float]:
    """IV at ref_delta: nearest call to +ref_delta and nearest put to −ref_delta, averaged."""
    calls = [r for r in chain.rows if r.is_call]
    puts = [r for r in chain.rows if not r.is_call]
    ivs = []
    if calls:
        ivs.append(min(calls, key=lambda r: abs(r.delta - ref_delta)).iv)
    if puts:
        ivs.append(min(puts, key=lambda r: abs(r.delta + ref_delta)).iv)
    return sum(ivs) / len(ivs) if ivs else None


def ratio_point(front: ChainSnapshot, back: ChainSnapshot,
                ref_delta: float) -> Optional[RatioPoint]:
    f, b = atm_iv(front, ref_delta), atm_iv(back, ref_delta)
    if f is None or b is None:
        return None
    ratio = front_back_ratio(f, b)
    if ratio != ratio:   # NaN
        return None
    return RatioPoint(as_of=front.as_of, front_expiry=front.expiry, back_expiry=back.expiry,
                      ref_delta=ref_delta, front_iv=f, back_iv=b, ratio=ratio)
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_engine.py -v` → 4 passed.
```bash
git add src_v2/flux/engine.py tests/flux/test_engine.py
git commit -m "feat(flux): engine — atm_iv at ref-delta + front/back ratio"
```

---

## Task 5: ChainCache (SQLite)

**Files:**
- Create: `src_v2/flux/cache.py`
- Test: `tests/flux/test_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_cache.py
from datetime import date, datetime, timezone

from src_v2.flux.models import OptionRow, ChainSnapshot
from src_v2.flux.cache import ChainCache


def _snap():
    return ChainSnapshot("SPX", date(2026, 6, 12),
                         datetime(2026, 4, 9, 20, tzinfo=timezone.utc),
                         (OptionRow(7000, True, 40, 41, 0.18, 0.5),))


def test_put_then_get_roundtrip(tmp_path):
    cache = ChainCache(tmp_path / "f.db")
    assert cache.get("SPX", date(2026, 6, 12), date(2026, 4, 9)) is None
    cache.put(_snap(), on_date=date(2026, 4, 9))
    got = cache.get("SPX", date(2026, 6, 12), date(2026, 4, 9))
    assert got is not None and got.rows[0].iv == 0.18 and got.expiry == date(2026, 6, 12)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_cache.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/cache.py
"""SQLite cache for EOD chain snapshots (bounds marketdata.app credit spend)."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from src_v2.flux.models import ChainSnapshot, OptionRow


class ChainCache:
    def __init__(self, db_path: str | Path = "data/flux_cache.db"):
        self.db = sqlite3.connect(str(db_path))
        self.db.execute("CREATE TABLE IF NOT EXISTS chains "
                        "(underlying TEXT, expiry TEXT, on_date TEXT, payload TEXT, "
                        "PRIMARY KEY(underlying, expiry, on_date))")

    def get(self, underlying: str, expiry: date, on_date: date) -> Optional[ChainSnapshot]:
        row = self.db.execute(
            "SELECT payload FROM chains WHERE underlying=? AND expiry=? AND on_date=?",
            (underlying, expiry.isoformat(), on_date.isoformat())).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        rows = tuple(OptionRow(**r) for r in d["rows"])
        return ChainSnapshot(d["underlying"], date.fromisoformat(d["expiry"]),
                             datetime.fromisoformat(d["as_of"]), rows)

    def put(self, snap: ChainSnapshot, on_date: date) -> None:
        payload = json.dumps({
            "underlying": snap.underlying, "expiry": snap.expiry.isoformat(),
            "as_of": snap.as_of.astimezone(timezone.utc).isoformat(),
            "rows": [r.__dict__ for r in snap.rows]})
        self.db.execute("INSERT OR REPLACE INTO chains VALUES (?,?,?,?)",
                        (snap.underlying, snap.expiry.isoformat(), on_date.isoformat(), payload))
        self.db.commit()
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_cache.py -v` → 1 passed.
```bash
git add src_v2/flux/cache.py tests/flux/test_cache.py
git commit -m "feat(flux): SQLite chain cache"
```

---

## Task 6: History — daily ratio series

**Files:**
- Create: `src_v2/flux/history.py`
- Test: `tests/flux/test_history.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_history.py
from datetime import date, datetime, timezone

from src_v2.flux.models import OptionRow, ChainSnapshot
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.history import daily_ratio_series


class FakeSource(FluxSource):
    """front IV 0.22, back IV 0.20 on every session → ratio 1.1."""
    def chain(self, expiry, on_date=None):
        # front expiries are sooner (smaller date) → give them higher IV
        is_front = expiry.day % 2 == 1
        iv = 0.22 if is_front else 0.20
        rows = (OptionRow(7000, True, 1, 1, iv, 0.5), OptionRow(7000, False, 1, 1, iv, -0.5))
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), rows)


def test_daily_series_has_one_point_per_session():
    # 3 sessions ending Wed 2026-06-10 → 06-08,06-09,06-10
    pts = daily_ratio_series(FakeSource(), front_dte=9, gap=1, n_days=3,
                             ref_delta=0.5, end_date=date(2026, 6, 10))
    assert len(pts) == 3
    assert all(abs(p.ratio - 1.1) < 1e-9 for p in pts)
    assert pts[0].as_of.date() < pts[-1].as_of.date()  # chronological
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_history.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/history.py
"""Daily EOD front/back ratio series over the last N NYSE sessions."""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import List, Optional

from src_v2.flux.engine import ratio_point
from src_v2.flux.models import RatioPoint
from src_v2.flux.sources.base import FluxSource
from src_v2.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after, sessions_between,
)


def _resolve(session: date, front_dte: int, gap: int):
    return nth_trading_day_after(session, front_dte), nth_trading_day_after(session, front_dte + gap)


def daily_ratio_series(source: FluxSource, front_dte: int, gap: int, n_days: int,
                       ref_delta: float, end_date: date) -> List[RatioPoint]:
    from datetime import timedelta
    sessions = sessions_between(end_date - timedelta(days=n_days * 2 + 10), end_date)[-n_days:]
    out: List[RatioPoint] = []
    for s in sessions:
        front, back = _resolve(s, front_dte, gap)
        if front is None or back is None:
            continue
        fc = source.chain(front, on_date=s)
        bc = source.chain(back, on_date=s)
        rp = ratio_point(fc, bc, ref_delta)
        if rp is None:
            continue
        # stamp the point at the session's EOD (16:00 ET ≈ 20:00 UTC) for a clean x-axis
        stamped = RatioPoint(datetime.combine(s, time(20, 0, tzinfo=timezone.utc)),
                             rp.front_expiry, rp.back_expiry, ref_delta,
                             rp.front_iv, rp.back_iv, rp.ratio)
        out.append(stamped)
    return out
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_history.py -v` → 1 passed.
```bash
git add src_v2/flux/history.py tests/flux/test_history.py
git commit -m "feat(flux): daily EOD ratio series (5/20-day views)"
```

---

## Task 7: Scanner

**Files:**
- Create: `src_v2/flux/scanner.py`
- Test: `tests/flux/test_scanner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_scanner.py
from datetime import date, datetime, timezone

from src_v2.flux.models import OptionRow, ChainSnapshot
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.scanner import scan


class FakeSource(FluxSource):
    """Ratio falls from 1.20 (prior sessions) to 1.05 today for every pair → reverting down."""
    def chain(self, expiry, on_date=None):
        front_iv, back_iv = (0.21, 0.20) if on_date == date(2026, 6, 10) else (0.24, 0.20)
        is_front = expiry.day % 2 == 1
        iv = front_iv if is_front else back_iv
        rows = (OptionRow(7000, True, 1, 1, iv, 0.5), OptionRow(7000, False, 1, 1, iv, -0.5))
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), rows)


def test_scan_ranks_by_drop_and_flags_reverting_backwardation():
    rows = scan(FakeSource(), on_date=date(2026, 6, 10), min_dte=8, max_dte=10,
                gaps=[1], ref_delta=0.5, by="drop")
    assert rows, "expected candidates"
    top = rows[0]
    assert top.ratio < top.ratio_prev      # reverting down
    assert top.drop > 0
    assert top.flag is True                 # ratio>1 (backwardation) AND drop>0
    # sorted by drop descending
    assert all(rows[i].drop >= rows[i + 1].drop for i in range(len(rows) - 1))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_scanner.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/scanner.py
"""Rank expiry-pairs by how much the front/back ratio dropped (or rose) vs the prior session."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

from src_v2.flux.engine import ratio_point
from src_v2.flux.models import ScanRow
from src_v2.flux.sources.base import FluxSource
from src_v2.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after, sessions_between,
)
from src_v2.backtest.dc_time_machine.flux_scanner import ratio_zscore, rank_candidates


def _prev_session(d: date) -> date:
    s = sessions_between(d - timedelta(days=10), d)
    return s[-2] if len(s) >= 2 and s[-1] == d else s[-1]


def _ratio_on(source, session, front, back, ref_delta):
    rp = ratio_point(source.chain(front, on_date=session),
                     source.chain(back, on_date=session), ref_delta)
    return rp.ratio if rp else None


def scan(source: FluxSource, on_date: date, min_dte: int, max_dte: int,
         gaps: List[int], ref_delta: float, by: str = "drop") -> List[ScanRow]:
    prev = _prev_session(on_date)
    raw = []
    for dte in range(min_dte, max_dte + 1):
        front = nth_trading_day_after(on_date, dte)
        if front is None:
            continue
        for gap in gaps:
            back = nth_trading_day_after(on_date, dte + gap)
            if back is None:
                continue
            today = _ratio_on(source, on_date, front, back, ref_delta)
            yest = _ratio_on(source, prev, front, back, ref_delta)
            if today is None or yest is None:
                continue
            raw.append({"front": front, "back": back, "gap": gap,
                        "ratio": today, "ratio_prev": yest})
    ranked = rank_candidates(raw, by=by)
    out: List[ScanRow] = []
    for c in ranked:
        drop = c["ratio_prev"] - c["ratio"]
        z = ratio_zscore(c["ratio"], [c["ratio_prev"], c["ratio"]])
        out.append(ScanRow(front_expiry=c["front"], back_expiry=c["back"], gap_td=c["gap"],
                           ratio=c["ratio"], ratio_prev=c["ratio_prev"], drop=drop,
                           zscore=0.0 if z != z else z,
                           flag=(c["ratio"] > 1.0 and drop > 0)))
    return out
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_scanner.py -v` → 1 passed.
```bash
git add src_v2/flux/scanner.py tests/flux/test_scanner.py
git commit -m "feat(flux): expiry-pair scanner (rank by ratio drop/rise + flag)"
```

---

## Task 8: Live intraday poller

**Files:**
- Create: `src_v2/flux/live.py`
- Test: `tests/flux/test_live.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_live.py
from datetime import date, datetime, timezone

from src_v2.flux.models import OptionRow, ChainSnapshot
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.live import LivePoller

ET_OPEN = datetime(2026, 6, 8, 14, 35, tzinfo=timezone.utc)   # 10:35 ET (open)
ET_CLOSED = datetime(2026, 6, 8, 2, 0, tzinfo=timezone.utc)   # overnight (closed)


class FakeSource(FluxSource):
    def chain(self, expiry, on_date=None):
        is_front = expiry.day % 2 == 1
        iv = 0.22 if is_front else 0.20
        return ChainSnapshot("SPX", expiry, datetime.now(timezone.utc),
                             (OptionRow(7000, True, 1, 1, iv, 0.5),
                              OptionRow(7000, False, 1, 1, iv, -0.5)))


def test_poll_appends_when_open():
    p = LivePoller(FakeSource(), front_dte=9, gap=1, ref_delta=0.5, now_utc=lambda: ET_OPEN)
    p.poll_once()
    assert len(p.buffer) == 1 and abs(p.buffer[0].ratio - 1.1) < 1e-9


def test_poll_noop_when_closed():
    p = LivePoller(FakeSource(), front_dte=9, gap=1, ref_delta=0.5, now_utc=lambda: ET_CLOSED)
    p.poll_once()
    assert p.buffer == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_live.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/live.py
"""Intraday RatioPoint buffer — polls the live chain only during the NYSE session."""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

from src_v2.flux.engine import ratio_point
from src_v2.flux.models import RatioPoint
from src_v2.flux.sources.base import FluxSource
from src_v2.backtest.dc_time_machine.trading_calendar import sessions_between

ET = ZoneInfo("America/New_York")


def _market_open(now_utc: datetime) -> bool:
    et = now_utc.astimezone(ET)
    d = et.date()
    if not sessions_between(d, d):          # not a trading day
        return False
    return time(9, 30) <= et.time() <= time(16, 0)


class LivePoller:
    def __init__(self, source: FluxSource, front_dte: int, gap: int, ref_delta: float,
                 poll_seconds: int = 60, now_utc: Optional[Callable[[], datetime]] = None):
        self.source = source
        self.front_dte, self.gap, self.ref_delta = front_dte, gap, ref_delta
        self.poll_seconds = poll_seconds
        self._now = now_utc or (lambda: datetime.now(timezone.utc))
        self.buffer: List[RatioPoint] = []

    def poll_once(self) -> Optional[RatioPoint]:
        now = self._now()
        if not _market_open(now):
            return None
        from src_v2.backtest.dc_time_machine.trading_calendar import nth_trading_day_after
        today = now.astimezone(ET).date()
        front = nth_trading_day_after(today, self.front_dte)
        back = nth_trading_day_after(today, self.front_dte + self.gap)
        if front is None or back is None:
            return None
        rp = ratio_point(self.source.chain(front), self.source.chain(back), self.ref_delta)
        if rp is not None:
            self.buffer.append(rp)
        return rp
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_live.py -v` → 2 passed.
```bash
git add src_v2/flux/live.py tests/flux/test_live.py
git commit -m "feat(flux): live intraday poller (market-hours only, injectable clock)"
```

---

## Task 9: Flask app + JSON endpoints

**Files:**
- Create: `src_v2/flux/app.py`
- Test: `tests/flux/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/flux/test_app.py
from datetime import date, datetime, timezone

from src_v2.flux.models import OptionRow, ChainSnapshot
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.app import create_app


class FakeSource(FluxSource):
    def chain(self, expiry, on_date=None):
        is_front = expiry.day % 2 == 1
        iv = 0.22 if is_front else 0.20
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc),
                             (OptionRow(7000, True, 1, 1, iv, 0.5),
                              OptionRow(7000, False, 1, 1, iv, -0.5)))


def _client():
    return create_app(FakeSource()).test_client()


def test_ratio_endpoint_returns_series():
    r = _client().get("/api/ratio?front_dte=9&gap=1&ref_delta=0.5&view=5day&end_date=2026-06-10")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["points"]) >= 1
    assert "front_iv" in body["points"][0] and "ratio" in body["points"][0]


def test_scan_endpoint_returns_rows():
    r = _client().get("/api/scan?on_date=2026-06-10&min_dte=8&max_dte=10&gaps=1&ref_delta=0.5")
    assert r.status_code == 200
    assert "rows" in r.get_json()


def test_expirations_endpoint():
    r = _client().get("/api/expirations?ref=2026-06-08&min_dte=1&max_dte=3")
    assert r.status_code == 200
    assert r.get_json()["expirations"] == ["2026-06-09", "2026-06-10", "2026-06-11"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_app.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

```python
# src_v2/flux/app.py
"""Flask shell: JSON endpoints over the Flux compute layer + the dashboard page."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src_v2.flux.history import daily_ratio_series
from src_v2.flux.scanner import scan
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.sources.marketdata import MarketDataSource

_STATIC = Path(__file__).parent / "static"


def _rp_json(p):
    return {"as_of": p.as_of.isoformat(), "front_iv": p.front_iv,
            "back_iv": p.back_iv, "ratio": p.ratio,
            "front_expiry": p.front_expiry.isoformat(), "back_expiry": p.back_expiry.isoformat()}


def _sr_json(s):
    return {"front_expiry": s.front_expiry.isoformat(), "back_expiry": s.back_expiry.isoformat(),
            "gap_td": s.gap_td, "ratio": round(s.ratio, 4), "ratio_prev": round(s.ratio_prev, 4),
            "drop": round(s.drop, 4), "zscore": round(s.zscore, 2), "flag": s.flag}


def create_app(source: FluxSource | None = None) -> Flask:
    src = source or MarketDataSource()
    app = Flask(__name__)

    @app.get("/api/ratio")
    def ratio():
        fd = int(request.args.get("front_dte", 9))
        gap = int(request.args.get("gap", 1))
        rd = float(request.args.get("ref_delta", 0.35))
        view = request.args.get("view", "5day")
        end = date.fromisoformat(request.args.get("end_date", date.today().isoformat()))
        n = {"5day": 5, "20day": 20}.get(view, 5)
        try:
            pts = daily_ratio_series(src, fd, gap, n, rd, end)
        except Exception as e:
            return jsonify({"error": str(e), "points": []}), 200
        meta = {"front_expiry": pts[-1].front_expiry.isoformat() if pts else None,
                "back_expiry": pts[-1].back_expiry.isoformat() if pts else None, "view": view}
        return jsonify({"points": [_rp_json(p) for p in pts], "meta": meta})

    @app.get("/api/scan")
    def do_scan():
        on = date.fromisoformat(request.args.get("on_date", date.today().isoformat()))
        rd = float(request.args.get("ref_delta", 0.35))
        gaps = [int(x) for x in request.args.get("gaps", "1").split(",")]
        try:
            rows = scan(src, on, int(request.args.get("min_dte", 0)),
                        int(request.args.get("max_dte", 30)), gaps, rd,
                        by=request.args.get("by", "drop"))
        except Exception as e:
            return jsonify({"error": str(e), "rows": []}), 200
        return jsonify({"rows": [_sr_json(s) for s in rows]})

    @app.get("/api/expirations")
    def expirations():
        ref = date.fromisoformat(request.args.get("ref", date.today().isoformat()))
        exps = src.expirations(ref, int(request.args.get("min_dte", 0)),
                               int(request.args.get("max_dte", 30)))
        return jsonify({"expirations": [e.isoformat() for e in exps]})

    @app.get("/")
    def index():
        return send_from_directory(_STATIC, "flux.html")

    return app
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_app.py -v` → 3 passed.
```bash
git add src_v2/flux/app.py tests/flux/test_app.py
git commit -m "feat(flux): Flask JSON endpoints (ratio/scan/expirations)"
```

---

## Task 10: Dashboard page + CLI

**Files:**
- Create: `src_v2/flux/static/flux.html`, `src_v2/flux/cli.py`, `src_v2/flux/__main__.py`
- Test: `tests/flux/test_cli.py`

- [ ] **Step 1: Write the failing test (CLI arg wiring only — HTML is manual-tested)**

```python
# tests/flux/test_cli.py
from src_v2.flux.cli import build_parser


def test_cli_has_serve_and_scan():
    p = build_parser()
    ns = p.parse_args(["serve", "--port", "8787"])
    assert ns.cmd == "serve" and ns.port == 8787
    ns2 = p.parse_args(["scan", "--on-date", "2026-06-10", "--by", "drop"])
    assert ns2.cmd == "scan" and ns2.on_date == "2026-06-10" and ns2.by == "drop"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/flux/test_cli.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the CLI + `__main__` + HTML**

```python
# src_v2/flux/cli.py
"""`python -m src_v2.flux serve|scan`."""
from __future__ import annotations

import argparse
from datetime import date


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="flux")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--host", default="127.0.0.1")
    sc = sub.add_parser("scan")
    sc.add_argument("--on-date", default=date.today().isoformat())
    sc.add_argument("--min-dte", type=int, default=0)
    sc.add_argument("--max-dte", type=int, default=30)
    sc.add_argument("--gaps", default="1")
    sc.add_argument("--ref-delta", type=float, default=0.35)
    sc.add_argument("--by", choices=["drop", "rise"], default="drop")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "serve":
        from src_v2.flux.app import create_app
        create_app().run(host=args.host, port=args.port, debug=False)
        return 0
    if args.cmd == "scan":
        from src_v2.flux.scanner import scan
        from src_v2.flux.sources.marketdata import MarketDataSource
        rows = scan(MarketDataSource(), date.fromisoformat(args.on_date), args.min_dte,
                    args.max_dte, [int(x) for x in args.gaps.split(",")], args.ref_delta, args.by)
        for r in rows[:25]:
            star = "★" if r.flag else " "
            print(f"{star} {r.front_expiry}->{r.back_expiry} gap{r.gap_td} "
                  f"ratio={r.ratio:.3f} drop={r.drop:+.3f} z={r.zscore:+.2f}")
        return 0
    return 1
```
```python
# src_v2/flux/__main__.py
from src_v2.flux.cli import main
raise SystemExit(main())
```
```html
<!-- src_v2/flux/static/flux.html -->
<!doctype html><html><head><meta charset="utf-8"><title>Flux — SPX IV term structure</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font:14px system-ui;margin:0;background:#0e1117;color:#e6e6e6}
 header{padding:10px 16px;background:#161b22;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 label{font-size:12px;color:#9aa4b2} select,button{background:#21262d;color:#e6e6e6;border:1px solid #30363d;border-radius:6px;padding:4px 8px}
 button.active{background:#1f6feb;border-color:#1f6feb}
 #banner{padding:6px 16px;color:#f0b72f;font-size:12px;min-height:18px}
 #chart{height:55vh} table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{padding:6px 10px;border-bottom:1px solid #21262d;text-align:right} th:first-child,td:first-child{text-align:left}
 tr.flag td{color:#3fb950;font-weight:600} h3{margin:12px 16px 4px}
</style></head><body>
<header>
 <strong>Flux — SPX</strong>
 <label>front DTE</label><select id="fd">9</select>
 <label>gap</label><select id="gap"></select>
 <label>ref Δ</label><select id="rd"><option value="0.35">0.35 (traded)</option><option value="0.5">0.50 (ATM)</option></select>
 <span><button class="view active" data-v="intraday">1-min</button>
 <button class="view" data-v="5day">5-day</button>
 <button class="view" data-v="20day">20-day</button></span>
</header>
<div id="banner"></div>
<div id="chart"></div>
<h3>Scanner — expiry pairs by ratio drop</h3>
<table id="scan"><thead><tr><th>front→back</th><th>gap</th><th>ratio</th><th>prev</th><th>drop</th><th>z</th></tr></thead><tbody></tbody></table>
<script>
const $=s=>document.querySelector(s);
for(const e of [3,5,7,9,12,15]){let o=document.createElement('option');o.value=e;o.text=e;if(e==9)o.selected=true;$('#fd').add(o);}
for(const g of [1,2,3,4]){let o=document.createElement('option');o.value=g;o.text=g;$('#gap').add(o);}
let view='intraday';
document.querySelectorAll('.view').forEach(b=>b.onclick=()=>{document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));b.classList.add('active');view=b.dataset.v;refresh();});
['#fd','#gap','#rd'].forEach(s=>$(s).onchange=refresh);
async function refresh(){
 const fd=$('#fd').value,gap=$('#gap').value,rd=$('#rd').value;
 const v=view==='intraday'?'5day':view; // intraday falls back to daily until live poller is wired in serve
 const r=await fetch(`/api/ratio?front_dte=${fd}&gap=${gap}&ref_delta=${rd}&view=${v}`).then(r=>r.json());
 $('#banner').textContent=r.error?('error: '+r.error):(r.meta&&r.meta.front_expiry?`${r.meta.front_expiry} / ${r.meta.back_expiry}  (${view})`:'no data');
 const x=r.points.map(p=>p.as_of);
 Plotly.react('chart',[
  {x,y:r.points.map(p=>p.front_iv),name:'front IV',line:{color:'#58a6ff'}},
  {x,y:r.points.map(p=>p.back_iv),name:'back IV',line:{color:'#3fb950'}},
  {x,y:r.points.map(p=>p.ratio),name:'ratio',yaxis:'y2',line:{color:'#f0883e',width:3}},
 ],{paper_bgcolor:'#0e1117',plot_bgcolor:'#0e1117',font:{color:'#e6e6e6'},margin:{t:20},
    legend:{orientation:'h'},yaxis:{title:'IV'},yaxis2:{title:'ratio',overlaying:'y',side:'right'}});
 const sc=await fetch(`/api/scan?ref_delta=${rd}&min_dte=0&max_dte=30&gaps=1,2,3&by=drop`).then(r=>r.json());
 $('#scan tbody').innerHTML=(sc.rows||[]).slice(0,15).map(s=>`<tr class="${s.flag?'flag':''}"><td>${s.front_expiry}→${s.back_expiry}</td><td>${s.gap_td}</td><td>${s.ratio}</td><td>${s.ratio_prev}</td><td>${s.drop}</td><td>${s.zscore}</td></tr>`).join('');
}
refresh(); setInterval(()=>{if(view==='intraday')refresh();},60000);
</script></body></html>
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/pytest tests/flux/test_cli.py -v` → 1 passed.
```bash
git add src_v2/flux/cli.py src_v2/flux/__main__.py src_v2/flux/static/flux.html tests/flux/test_cli.py
git commit -m "feat(flux): plotly.js dashboard + CLI (serve/scan)"
```

---

## Task 11: Wire MarketDataSource through the cache + live smoke (operational)

**Files:**
- Modify: `src_v2/flux/sources/marketdata.py` (optional cache hook), `src_v2/flux/app.py`
- Create: `src_v2/flux/README.md`

- [ ] **Step 1: Add an optional cache to `MarketDataSource`**

Make EOD fetches consult `ChainCache` first. Edit `MarketDataSource.__init__` to accept `cache: Optional[ChainCache] = None` and, in `chain()`, when `on_date is not None and self.cache`: return `cache.get(...)` if present, else fetch, `cache.put(...)`, return. Add a test `tests/flux/test_marketdata_cache.py` asserting the second EOD `chain()` call does not invoke `fetch_json` (use a counter). Commit.

```python
# test sketch — full code to write:
def test_eod_fetch_is_cached(tmp_path):
    calls = {"n": 0}
    def fetch(url):
        calls["n"] += 1
        return CANNED
    from src_v2.flux.cache import ChainCache
    src = MarketDataSource(token="x", fetch_json=fetch, cache=ChainCache(tmp_path / "c.db"))
    src.chain(date(2026, 6, 12), on_date=date(2026, 4, 9))
    src.chain(date(2026, 6, 12), on_date=date(2026, 4, 9))
    assert calls["n"] == 1
```

- [ ] **Step 2: Live smoke against marketdata.app (real token, manual)**

Run (Theta not needed; this hits marketdata.app):
```bash
.venv/bin/python -m src_v2.flux scan --on-date <a recent trading day> --min-dte 5 --max-dte 12 --gaps 1,2
```
Expected: a ranked table prints with real SPX ratios (~1.0–1.2). Then:
```bash
.venv/bin/python -m src_v2.flux serve --port 8787
```
Open `http://127.0.0.1:8787` → confirm the three lines render and the 5-day/20-day toggles load. Record marketdata.app credit usage from the dashboard/account and note it in the README.

- [ ] **Step 3: Write `src_v2/flux/README.md`**

Cover: what Flux shows (front/back IV + ratio, scanner), the `MARKET_DATA_TOKEN` setup (`../options_api/.env`), `serve`/`scan` usage, the ref-delta meaning (0.35 traded vs 0.50 ATM), the **EOD-historical / delayed-live** limitation, and the **per-option credit cost** (mitigated by `strikeLimit=25` + cache). Commit `README.md` + the full suite green:

```bash
.venv/bin/pytest tests/flux/ -v
git add src_v2/flux/README.md src_v2/flux/sources/marketdata.py tests/flux/test_marketdata_cache.py
git commit -m "feat(flux): cache-backed source + README + live smoke notes"
```

---

## Self-Review

**1. Spec coverage:**
- §2 marketdata facts (columnar, 203, EOD `date=`, strikeLimit, token) → Tasks 2, 11. ✓
- §3 architecture/file layout → File Structure + all tasks. ✓
- §4 models → Task 1. ✓
- §5 FluxSource + MarketDataSource (+ calendar-derived expirations) → Tasks 2, 3. ✓
- §6 engine (atm_iv ref-delta 0.35 default, ratio) → Task 4. ✓
- §7 cache → Tasks 5, 11. ✓
- §8 history (5/20-day) → Task 6. ✓
- §9 scanner → Task 7. ✓
- §10 live poller (market-hours, injectable clock) → Task 8. ✓
- §11 web app (3 endpoints + dashboard, error banners) → Tasks 9, 10. ✓
- §12 error handling (token/API/no-data) → Tasks 2 (empty→empty chain), 9 (try/except→error JSON). ✓
- §13 testing → every task is TDD; frontend manual (Task 11 smoke). ✓
- §14 deliverables (package, README) → Tasks 1–11. ✓
- **Known gap (intentional):** the intraday *live view* in the dashboard falls back to the 5-day series until the `LivePoller` is mounted into `serve` (the poller exists + is tested in Task 8, but wiring a background thread + buffer endpoint into the Flask app is deferred to keep v1 lean — noted in the HTML comment and acceptable since marketdata.app live is delayed anyway). If the live intraday view is required for v1, add a Task 12: a `/api/intraday` endpoint backed by a `LivePoller` started in `create_app`.

**2. Placeholder scan:** No "TBD"/"add error handling"-style placeholders; Task 11 Step 1 gives a test sketch + the exact edit description (acceptable — it's a small modify, full test code shown). Every code step shows complete code.

**3. Type consistency:** `OptionRow(strike,is_call,bid,ask,iv,delta)`, `ChainSnapshot(underlying,expiry,as_of,rows)`, `RatioPoint(as_of,front_expiry,back_expiry,ref_delta,front_iv,back_iv,ratio)`, `ScanRow(front_expiry,back_expiry,gap_td,ratio,ratio_prev,drop,zscore,flag)` used identically across Tasks 1,4,6,7,8,9. `chain(expiry,on_date)` / `expirations(ref,min_dte,max_dte)` consistent (Tasks 2,3,6,7,8,9). `atm_iv(chain,ref_delta)` / `ratio_point(front,back,ref_delta)` consistent (Tasks 4,6,7,8). `daily_ratio_series(source,front_dte,gap,n_days,ref_delta,end_date)` and `scan(source,on_date,min_dte,max_dte,gaps,ref_delta,by)` consistent (Tasks 6,7,9,10). ✓
