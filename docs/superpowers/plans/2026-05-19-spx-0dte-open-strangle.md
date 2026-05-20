# SPX 0DTE Open Strangle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a WF-CV-validated SPX 0DTE open strangle (short put + short call, no wings) and compare it head-to-head against the existing open IC under a per-trade matched $-SL risk-normalization constraint.

**Architecture:** New `OpenStrangleStrategy` forked from `OpenIcStrategy` — 2 legs instead of 4, gross credit, `sl_dollars` primitive replacing `sl_mult_of_credit`, and a per-day SL$ lookup from a pre-extracted IC artifact (`min(IC_per_day, IC_p75)`). The WF-CV runner is generalized to load the strategy class via a `strategy.class` config key so the strangle reuses the existing optimizer end-to-end.

**Tech Stack:** Python 3.13 · existing `src_v2` backtest pipeline (`WindowEntryEngine`, `WalkForwardOptimizer`, `CbboStore`) · pandas/numpy for the SL artifact · yaml configs · pytest for tests.

**Spec:** [docs/superpowers/specs/2026-05-19-spx-0dte-open-strangle-design.md](../specs/2026-05-19-spx-0dte-open-strangle-design.md)

---

## File map

| Path | Action | Purpose |
|---|---|---|
| [scripts/run_v2_wf_cv_open_ic.py](../../../scripts/run_v2_wf_cv_open_ic.py) | Modify | Generalize: resolve `strategy.class` from config |
| [src_v2/strategies/open_strangle.py](../../../src_v2/strategies/open_strangle.py) | Create | `OpenStrangleStrategy` |
| [config/config_spx_0dte_open_strangle.yaml](../../../config/config_spx_0dte_open_strangle.yaml) | Create | Strangle config |
| [scripts/extract_ic_sl_dollars.py](../../../scripts/extract_ic_sl_dollars.py) | Create | Emit IC SL-by-day artifact |
| [data/ic_sl_dollars_by_day.csv](../../../data/ic_sl_dollars_by_day.csv) | Create (generated) | Per-day IC $-SL + p75 ceiling |
| [scripts/run_v2_wf_cv_open_strangle.py](../../../scripts/run_v2_wf_cv_open_strangle.py) | Create | Strangle WF-CV entry script (thin wrapper) |
| [scripts/analyze_open_strangle_trades.py](../../../scripts/analyze_open_strangle_trades.py) | Create | Strangle trade-log analyzer |
| [scripts/compare_strangle_vs_ic.py](../../../scripts/compare_strangle_vs_ic.py) | Create | Side-by-side IC vs. strangle reporter |
| [tests/test_open_strangle_strategy.py](../../../tests/test_open_strangle_strategy.py) | Create | Unit tests for `OpenStrangleStrategy` |
| [tests/test_wf_runner_strategy_class.py](../../../tests/test_wf_runner_strategy_class.py) | Create | Verify runner backcompat + dynamic class resolution |
| [docs/spx_0dte_open_strangle_findings.md](../../../docs/spx_0dte_open_strangle_findings.md) | Create (post-sweep) | Findings doc |

Run all commands from repo root: `~/Dev/0dte_bot`. Use the project's `.venv` Python: `.venv/bin/python` (or activate first). Use `pip install -e ".[dev]"` if dependencies are missing.

---

## Task 1: Generalize WF-CV runner to load strategy class from config

**Files:**
- Modify: `scripts/run_v2_wf_cv_open_ic.py` (lines 22-28, 60-118)
- Test: `tests/test_wf_runner_strategy_class.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wf_runner_strategy_class.py`:

```python
"""Verify the WF-CV runner can resolve strategy class from config."""
import importlib
import pytest

from src_v2.strategies.open_ic import OpenIcStrategy


def _resolve_strategy_class(dotted_path: str | None):
    """Mirror the runner's resolution logic for direct testing."""
    from scripts.run_v2_wf_cv_open_ic import resolve_strategy_class
    return resolve_strategy_class(dotted_path)


def test_default_resolves_to_open_ic():
    cls = _resolve_strategy_class(None)
    assert cls is OpenIcStrategy


def test_explicit_open_ic_path():
    cls = _resolve_strategy_class("src_v2.strategies.open_ic.OpenIcStrategy")
    assert cls is OpenIcStrategy


def test_unknown_module_raises():
    with pytest.raises((ImportError, ModuleNotFoundError)):
        _resolve_strategy_class("src_v2.strategies.no_such.NoSuchStrategy")


def test_unknown_attr_raises():
    with pytest.raises(AttributeError):
        _resolve_strategy_class("src_v2.strategies.open_ic.NoSuchClass")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_wf_runner_strategy_class.py -v
```

Expected: FAIL with `ImportError: cannot import name 'resolve_strategy_class'`.

- [ ] **Step 3: Implement resolution helper**

Edit `scripts/run_v2_wf_cv_open_ic.py`. Replace lines 22-28 (the imports block):

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib

from src_v2.backtest.data_loader import SpyLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.wf_optimizer import WalkForwardOptimizer
from src_v2.strategies.open_ic import OpenIcStrategy


def resolve_strategy_class(dotted_path: str | None):
    """Resolve a strategy class from a dotted import path.

    Defaults to OpenIcStrategy when path is None or empty (backcompat).
    """
    if not dotted_path:
        return OpenIcStrategy
    mod_path, _, cls_name = dotted_path.rpartition(".")
    if not mod_path:
        raise ValueError(f"strategy.class must be a dotted path; got {dotted_path!r}")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)
```

- [ ] **Step 4: Wire the resolver into `main()`**

In `scripts/run_v2_wf_cv_open_ic.py`, replace line 106 (the `strategy_cls=OpenIcStrategy,` argument):

```python
        strategy_cls=resolve_strategy_class(
            config.get("strategy", {}).get("class")
        ),
```

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_wf_runner_strategy_class.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Smoke-check existing IC config still parses**

```bash
.venv/bin/python -c "
import yaml
from scripts.run_v2_wf_cv_open_ic import resolve_strategy_class
cfg = yaml.safe_load(open('config/config_spx_0dte_open_ic.yaml'))
cls = resolve_strategy_class(cfg.get('strategy', {}).get('class'))
print(f'Resolved: {cls.__name__}')
assert cls.__name__ == 'OpenIcStrategy', cls
"
```

Expected output: `Resolved: OpenIcStrategy`.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_v2_wf_cv_open_ic.py tests/test_wf_runner_strategy_class.py
git commit -m "feat(wf-cv): resolve strategy class from config (backcompat OpenIcStrategy)"
```

---

## Task 2: Extract IC SL-by-day artifact

**Files:**
- Create: `scripts/extract_ic_sl_dollars.py`
- Create (generated, gitignored): `data/ic_sl_dollars_by_day.csv`

This task produces the data artifact §3 of the spec depends on. It re-runs the winning IC config's WF-CV with `--top-k 1`, walks the trade log, and emits per-day `IC_SL$ = sl_mult × IC_total_credit × 100`.

- [ ] **Step 1: Verify the existing IC trade-log shape**

```bash
.venv/bin/python -c "
from src_v2.backtest.wf_optimizer import WalkForwardOptimizer
import inspect
src = inspect.getsource(WalkForwardOptimizer)
# Look for trade-result fields the optimizer surfaces
for line in src.splitlines():
    if 'total_credit' in line or 'put_credit' in line or 'trade_date' in line:
        print(line)
"
```

Expected: at least one line referencing `total_credit` or `put_credit` and a date-like field — confirms the artifact extractor has data to work with. If no match, instead inspect a recent IC trade log under `results/spx_0dte_open_ic/`.

- [ ] **Step 2: Locate an existing IC results dir**

```bash
ls -1d results/spx_0dte_open_ic* 2>/dev/null | head -3
ls results/$(ls -1 results/ | grep -i open_ic | head -1)/ 2>/dev/null | head
```

Expected: at least one results dir with a `trades.csv` or `trades.parquet`. Record the path you find; the extractor will read from it. If no results dir exists, the extractor must run its own WF-CV — fall through to Step 3.

- [ ] **Step 3: Write the extractor**

Create `scripts/extract_ic_sl_dollars.py`:

```python
"""Extract per-day effective $-SL from the winning open-IC WF-CV trade log.

Outputs data/ic_sl_dollars_by_day.csv with columns:
    trade_date (YYYY-MM-DD), ic_total_credit, sl_mult, ic_sl_dollars

Also emits a one-line companion file data/ic_sl_dollars_by_day.meta.json with:
    {"p75": <float>, "median": <float>, "mean": <float>, "n_days": <int>}

Usage:
    python scripts/extract_ic_sl_dollars.py \\
        --trades results/spx_0dte_open_ic/<run>/trades.csv \\
        --sl-mult 2.0 \\
        --out data/ic_sl_dollars_by_day.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True,
                    help="Path to IC trades.csv (or .parquet) from the WF-CV run")
    ap.add_argument("--sl-mult", type=float, default=2.0,
                    help="The IC's sl_mult_of_credit (default 2.0, matches v5 winner)")
    ap.add_argument("--out", default="data/ic_sl_dollars_by_day.csv")
    ap.add_argument("--multiplier", type=int, default=100,
                    help="Contract multiplier (SPX = 100)")
    args = ap.parse_args()

    p = Path(args.trades)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Locate the credit column (prefer total_credit, fall back to put+call).
    if "total_credit" in df.columns:
        credit = df["total_credit"].astype(float)
    elif {"put_credit", "call_credit"}.issubset(df.columns):
        credit = (df["put_credit"].astype(float)
                  + df["call_credit"].astype(float))
    else:
        raise SystemExit(
            f"Trade log {p} has no total_credit or put_credit/call_credit columns. "
            f"Columns present: {list(df.columns)}"
        )

    # Locate the date column.
    date_col = None
    for c in ("trade_date", "entry_date", "date", "day"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise SystemExit(
            f"Trade log {p} has no recognizable date column. "
            f"Columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "trade_date": pd.to_datetime(df[date_col]).dt.date.astype(str),
        "ic_total_credit": credit.round(2),
        "sl_mult": args.sl_mult,
        "ic_sl_dollars": (credit * args.sl_mult * args.multiplier).round(2),
    })
    out = out.drop_duplicates(subset="trade_date").sort_values("trade_date")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    arr = out["ic_sl_dollars"].to_numpy()
    meta = {
        "p75": float(np.percentile(arr, 75)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_days": int(len(arr)),
        "source_trades": str(p),
        "sl_mult": args.sl_mult,
    }
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(out)} days to {args.out}")
    print(f"Wrote meta to {meta_path}")
    print(f"  p75=${meta['p75']:.2f}  median=${meta['median']:.2f}  "
          f"mean=${meta['mean']:.2f}  range=[${meta['min']:.2f},${meta['max']:.2f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the extractor against the existing IC trade log**

Substitute the actual path from Step 2:

```bash
.venv/bin/python scripts/extract_ic_sl_dollars.py \
    --trades results/spx_0dte_open_ic/<RUN_DIR>/trades.csv \
    --sl-mult 2.0 \
    --out data/ic_sl_dollars_by_day.csv
```

Expected: writes a CSV (~170 rows for the 172 gated trade-days in the IC findings) and a `.meta.json`. Inspect the meta:

```bash
cat data/ic_sl_dollars_by_day.meta.json
```

Expected: `p75` somewhere in $200-$500 range, `n_days` ≈ 170.

- [ ] **Step 5: Add the artifact path to `.gitignore`**

Append to `.gitignore`:

```
# IC SL-by-day artifact (regenerated from trade log)
data/ic_sl_dollars_by_day.csv
data/ic_sl_dollars_by_day.meta.json
```

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_ic_sl_dollars.py .gitignore
git commit -m "feat(open-strangle): IC SL-by-day extractor + gitignored artifact"
```

---

## Task 3: `OpenStrangleStrategy` — configure() + strike selection (no exits yet)

**Files:**
- Create: `src_v2/strategies/open_strangle.py`
- Create: `tests/test_open_strangle_strategy.py`

- [ ] **Step 1: Write the failing test for strike selection**

Create `tests/test_open_strangle_strategy.py`:

```python
"""Unit tests for OpenStrangleStrategy."""
import asyncio
import math
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from src_v2.strategies.open_strangle import OpenStrangleStrategy


ET = ZoneInfo("America/New_York")


def _make_chain(spot: float, atm_credit: float = 10.0):
    """Build a synthetic chain centered on spot.

    ATM straddle = atm_credit (so EM = atm_credit).
    Symmetric prices decreasing linearly with strike distance.
    """
    strikes = [spot + i * 5 for i in range(-30, 31)]
    chain = []
    for k in strikes:
        d = abs(k - spot)
        # Linear taper, floor at $0.05
        c_mid = max(0.05, atm_credit / 2 - d * 0.10)
        p_mid = max(0.05, atm_credit / 2 - d * 0.10)
        chain.append({
            "strike": k, "option_type": "call",
            "bid": c_mid - 0.02, "ask": c_mid + 0.02,
        })
        chain.append({
            "strike": k, "option_type": "put",
            "bid": p_mid - 0.02, "ask": p_mid + 0.02,
        })
    return chain


def _make_strategy(**opt_overrides):
    s = OpenStrangleStrategy()
    cfg = {
        "options": {
            "strike_step_pts": 5.0,
            "strike_mode": "em_mult",
            "em_mult": 0.30,
            "min_credit_per_spread": 0.05,
            "slippage_per_spread": 0.05,
            "contract_multiplier": 100,
            "quantity": 1,
            "tp_pct_of_credit": 0.50,
            "sl_dollars_csv": None,
            "sl_dollars_fallback": 300.0,
            **opt_overrides,
        },
        "market": {"market_close": "16:00"},
        "strategy": {"eod_force_close": "15:30"},
    }
    s.configure(cfg)
    s._now_dt = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    return s


def test_search_candidates_returns_two_legs():
    s = _make_strategy()
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    assert out["candidates_found"] == 1
    setup = out["top_candidate"]
    assert len(setup.legs) == 2, f"expected 2 legs, got {len(setup.legs)}"
    sides = sorted(leg.option_type for leg in setup.legs)
    assert sides == ["call", "put"]
    assert all(leg.side == "sell" for leg in setup.legs)


def test_gross_credit_no_wing_subtraction():
    s = _make_strategy()
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    setup = out["top_candidate"]
    # Strangle credit = put_mid + call_mid - 2 * slippage; no wings.
    assert setup.total_credit > 0
    assert "put_credit" in setup.extra and "call_credit" in setup.extra
    # Total = put_credit + call_credit (post-slippage)
    expected = round(setup.extra["put_credit"] + setup.extra["call_credit"], 2)
    assert setup.total_credit == expected


def test_em_mult_strike_placement():
    """At em_mult=0.30 with EM=10pt and 5pt strike step:
       0.30 × 10 = 3.0 pt offset.
       Put: floor((5000 - 3) / 5) * 5 = 4995.
       Call: ceil((5000 + 3) / 5) * 5 = 5005."""
    s = _make_strategy(em_mult=0.30)
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    setup = out["top_candidate"]
    assert setup.strikes["short_put"] == 4995.0
    assert setup.strikes["short_call"] == 5005.0
    assert "long_put" not in setup.strikes
    assert "long_call" not in setup.strikes
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_open_strangle_strategy.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src_v2.strategies.open_strangle'`.

- [ ] **Step 3: Implement the strategy skeleton**

Create `src_v2/strategies/open_strangle.py`:

```python
"""
OpenStrangleStrategy — 0DTE short strangle entered at a fixed time near the open.

Forked from OpenIcStrategy. Differences:
  - 2 legs (short put + short call); no protective wings
  - total_credit = put_mid + call_mid - 2*slippage (gross premium)
  - sl_dollars (absolute $-SL) replaces sl_mult_of_credit. Set per-trade from
    a pre-extracted IC SL-by-day artifact: sl_dollars[d] = min(IC_d, IC_p75).
  - No breach handling (no defined wings to breach)
  - Designed for eod_force_close earlier than 15:59 to avoid assignment risk
"""

from __future__ import annotations

import csv
import math
from datetime import date as date_cls, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src_v2.strategies.base import (
    LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction,
)
from src_v2.agent.base_engine import BaseEngine, EngineDeps
from src_v2.strategies.utils import mid_price as _mid


class OpenStrangleStrategy(LiveStrategy):

    def configure(self, config: Dict[str, Any]) -> None:
        opt = config.get("options", {})
        self._strike_step = opt.get("strike_step_pts", 5.0)
        # "delta" mode is supported for parity with OpenIcStrategy but is
        # unreliable at 0DTE. Default is em_mult.
        self._strike_mode = opt.get("strike_mode", "em_mult")
        self._target_short_delta = opt.get("target_short_delta", 0.15)
        self._em_mult = opt.get("em_mult", 0.30)
        self._em_mult_put = opt.get("em_mult_put", None)
        self._em_mult_call = opt.get("em_mult_call", None)
        self._min_credit = opt.get("min_credit_per_spread", 0.05)
        self._max_credit_imbalance = opt.get("max_credit_imbalance", 1.00)
        self._tp_pct = opt.get("tp_pct_of_credit", 0.50)
        self._tp_pct_end = opt.get("tp_pct_of_credit_end", None)
        # tp_mode: "total" closes both sides when (put+call mark) <= tp_pct*total;
        #          "per_side" closes one side when its mark <= tp_pct*side_credit.
        self._tp_mode = opt.get("tp_mode", "total")
        self._slippage = opt.get("slippage_per_spread", 0.10)
        self._multiplier = opt.get("contract_multiplier", 100)
        self._quantity = opt.get("quantity", 1)

        # SL artifact: per-day $-SL from the matched IC. Set None to disable.
        self._sl_csv_path = opt.get("sl_dollars_csv", None)
        # Fallback $-SL when a date isn't in the artifact (also serves as the
        # p75 ceiling for matched-day values).
        self._sl_fallback = float(opt.get("sl_dollars_fallback", 0.0))
        self._sl_by_date: Dict[date_cls, float] = {}
        self._sl_p75: float = self._sl_fallback
        if self._sl_csv_path:
            self._load_sl_csv(self._sl_csv_path)

        market_close = config.get("market", {}).get("market_close", "16:00")
        h, m = [int(x) for x in market_close.split(":")]
        self._market_close = time(h, m)

        eod = config.get("strategy", {}).get("eod_force_close", "15:30")
        eh, em_ = [int(x) for x in eod.split(":")]
        self._eod_time = time(eh, em_)

        self._now_dt: Optional[datetime] = None
        self._last_em_pts: Optional[float] = None

    # ------------------------------------------------------------------
    # SL artifact loader
    # ------------------------------------------------------------------

    def _load_sl_csv(self, path: str) -> None:
        """Load IC per-day $-SL from a CSV produced by
        scripts/extract_ic_sl_dollars.py. Expected columns:
            trade_date (YYYY-MM-DD), ic_sl_dollars.
        Also reads <path>.meta.json for the p75 ceiling.
        """
        import json
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"sl_dollars_csv not found: {p}")
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = date_cls.fromisoformat(row["trade_date"])
                self._sl_by_date[d] = float(row["ic_sl_dollars"])
        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._sl_p75 = float(meta.get("p75", self._sl_fallback))
        elif self._sl_by_date:
            # Compute p75 from data if meta missing.
            import numpy as np
            self._sl_p75 = float(np.percentile(
                list(self._sl_by_date.values()), 75
            ))

    def _sl_dollars_for(self, when: Optional[datetime]) -> float:
        """Return min(IC_per_day, IC_p75) with fallback to _sl_fallback.

        If no artifact is loaded (_sl_by_date empty), returns _sl_fallback.
        """
        if not self._sl_by_date:
            return self._sl_fallback
        if when is None:
            return self._sl_p75
        d = when.date() if isinstance(when, datetime) else when
        per_day = self._sl_by_date.get(d)
        if per_day is None:
            return self._sl_p75
        return min(per_day, self._sl_p75)

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    def create_engine(self, deps: EngineDeps) -> BaseEngine:
        from src_v2.agent.window_entry_engine import WindowEntryEngine
        from src_v2.backtest.sim_clock import WallClock
        self._clock_ref = deps.clock or WallClock()
        return WindowEntryEngine(deps, self)

    # ------------------------------------------------------------------
    # Entry: candidate search
    # ------------------------------------------------------------------

    async def search_candidates(
        self,
        chain: List[Dict[str, Any]],
        spot: float,
    ) -> Dict[str, Any]:
        if not chain or spot <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        chain_by = self._chain_map(chain)
        em = self._compute_em_from_straddle(chain_by, spot)
        if em is None or em <= 0:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}
        self._last_em_pts = em

        if self._strike_mode == "em_mult":
            put_mult = (self._em_mult_put if self._em_mult_put is not None
                        else self._em_mult)
            call_mult = (self._em_mult_call if self._em_mult_call is not None
                         else self._em_mult)
            short_call_k = math.ceil(
                (spot + call_mult * em) / self._strike_step
            ) * self._strike_step
            short_put_k = math.floor(
                (spot - put_mult * em) / self._strike_step
            ) * self._strike_step
        else:
            # Delta mode (BS inversion — kept for parity; not recommended at 0DTE)
            from src_8am.backtest.pricing import (
                bs_call, bs_put, bs_call_delta, bs_put_delta,
            )
            t = self._time_to_expiration_years()
            if t <= 0:
                return {"candidates_found": 0, "top_candidate": None, "candidates": []}
            iv = self._derive_iv_from_atm(chain_by, spot, t)
            if iv is None:
                return {"candidates_found": 0, "top_candidate": None, "candidates": []}
            short_call_k = self._find_strike_for_delta(
                chain_by, spot, self._target_short_delta, "call", t, iv,
            )
            short_put_k = self._find_strike_for_delta(
                chain_by, spot, self._target_short_delta, "put", t, iv,
            )

        if short_call_k is None or short_put_k is None:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        sc = chain_by.get((short_call_k, "call"))
        sp = chain_by.get((short_put_k, "put"))
        if not sc or not sp:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        sc_mid = _mid(sc)
        sp_mid = _mid(sp)
        if sc_mid is None or sp_mid is None:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        call_credit = round(sc_mid - self._slippage, 2)
        put_credit = round(sp_mid - self._slippage, 2)

        if call_credit < self._min_credit or put_credit < self._min_credit:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}
        if abs(call_credit - put_credit) > self._max_credit_imbalance:
            return {"candidates_found": 0, "top_candidate": None, "candidates": []}

        total_credit = round(call_credit + put_credit, 2)

        # Resolve today's sl_dollars from artifact (used by check_exit).
        sl_dollars = self._sl_dollars_for(self._now_dt)

        setup = LiveTradeSetup(
            strategy_type="open_strangle",
            legs=[
                OrderLeg("", short_put_k, "put", "sell", self._quantity),
                OrderLeg("", short_call_k, "call", "sell", self._quantity),
            ],
            total_credit=total_credit,
            strikes={
                "short_put": short_put_k,
                "short_call": short_call_k,
            },
            extra={
                "put_credit": put_credit,
                "call_credit": call_credit,
                "em_pts": round(em, 2),
                "strike_mode": self._strike_mode,
                "em_mult_used": self._em_mult,
                "sl_dollars": sl_dollars,
            },
        )

        return {"candidates_found": 1, "top_candidate": setup, "candidates": [setup]}

    # ------------------------------------------------------------------
    # Entry: validation
    # ------------------------------------------------------------------

    async def validate(
        self,
        setup: LiveTradeSetup,
        chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        checks, failures = [], []
        for side in ("put_credit", "call_credit"):
            if setup.extra.get(side, 0) < self._min_credit:
                failures.append(f"{side}_min")
        return {"passed": len(failures) == 0, "checks": checks, "failures": failures}

    def build_order(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        s = setup.strikes
        return {
            "order_type": "strangle",
            "short_put_strike": s["short_put"],
            "short_call_strike": s["short_call"],
            "quantity": self._quantity,
        }

    # ------------------------------------------------------------------
    # Stop-loss (placeholder — real SL fires in check_exit)
    # ------------------------------------------------------------------

    def compute_stop_loss(self, setup: LiveTradeSetup) -> Dict[str, Any]:
        return {
            "mode": "sl_dollars",
            "sl_dollars": setup.extra.get("sl_dollars", self._sl_fallback),
        }

    def build_stop_orders(self, setup, stop_info) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    # Exit — implemented in Task 5
    # ------------------------------------------------------------------

    def check_exit(self, trade, quotes) -> ExitAction:
        # Implemented in Task 5; placeholder so the ABC is satisfied.
        return ExitAction(action="hold")

    def compute_trailing_stop(self, trade, quotes) -> Optional[float]:
        return None

    def leg_pairs(self) -> List[Tuple[str, str, str]]:
        # Strangle has no long legs; PositionMonitor skips when long is None.
        return [
            ("short_call_symbol", "long_call_symbol", "call"),
            ("short_put_symbol", "long_put_symbol", "put"),
        ]

    # ------------------------------------------------------------------
    # Internal helpers (copied from OpenIcStrategy for self-containment)
    # ------------------------------------------------------------------

    def _time_to_expiration_years(self) -> float:
        now = self._now_dt
        if now is None:
            return 0.0
        close_dt = datetime(
            now.year, now.month, now.day,
            self._market_close.hour, self._market_close.minute,
            tzinfo=now.tzinfo,
        )
        seconds = max(0.0, (close_dt - now).total_seconds())
        return seconds / (365.0 * 24.0 * 3600.0)

    def _compute_em_from_straddle(
        self, chain_by: Dict, spot: float,
    ) -> Optional[float]:
        step = self._strike_step
        atm_k = round(spot / step) * step
        call = chain_by.get((atm_k, "call"))
        put = chain_by.get((atm_k, "put"))
        if not call or not put:
            strikes = sorted({k for (k, _) in chain_by.keys()})
            if not strikes:
                return None
            atm_k = min(strikes, key=lambda k: abs(k - spot))
            call = chain_by.get((atm_k, "call"))
            put = chain_by.get((atm_k, "put"))
        if not call or not put:
            return None
        c_mid = _mid(call); p_mid = _mid(put)
        if c_mid is None or p_mid is None:
            return None
        return c_mid + p_mid

    def _derive_iv_from_atm(self, chain_by, spot, t):
        from src_8am.backtest.pricing import bs_call, bs_put
        atm_k = round(spot / self._strike_step) * self._strike_step
        call = chain_by.get((atm_k, "call"))
        put = chain_by.get((atm_k, "put"))
        if not call or not put:
            return None
        c_mid = _mid(call); p_mid = _mid(put)
        if c_mid is None or p_mid is None:
            return None
        target = c_mid + p_mid
        if target <= 0 or t <= 0:
            return None
        lo, hi = 0.02, 4.0
        for _ in range(50):
            mid = (lo + hi) / 2
            s = bs_call(spot, atm_k, t, mid) + bs_put(spot, atm_k, t, mid)
            if s < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _find_strike_for_delta(self, chain_by, spot, target_delta, side, t, iv):
        from src_8am.backtest.pricing import bs_call_delta, bs_put_delta
        step = self._strike_step
        best_k, best_diff = None, 1.0
        if side == "call":
            k = math.ceil(spot / step) * step
            limit = spot + 250
            while k < limit:
                if (k, "call") in chain_by:
                    d = bs_call_delta(spot, k, t, iv)
                    diff = abs(d - target_delta)
                    if diff < best_diff:
                        best_diff = diff
                        best_k = k
                    if d < target_delta * 0.3:
                        break
                k += step
        else:
            k = math.floor(spot / step) * step
            limit = spot - 250
            while k > limit:
                if (k, "put") in chain_by:
                    d = abs(bs_put_delta(spot, k, t, iv))
                    diff = abs(d - target_delta)
                    if diff < best_diff:
                        best_diff = diff
                        best_k = k
                    if d < target_delta * 0.3:
                        break
                k -= step
        return best_k

    @staticmethod
    def _chain_map(chain: List[Dict[str, Any]]) -> Dict[Tuple[float, str], Dict]:
        return {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_open_strangle_strategy.py -v
```

Expected: 3 passed (`test_search_candidates_returns_two_legs`, `test_gross_credit_no_wing_subtraction`, `test_em_mult_strike_placement`).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/open_strangle.py tests/test_open_strangle_strategy.py
git commit -m "feat(open-strangle): strategy skeleton + strike selection"
```

---

## Task 4: `OpenStrangleStrategy` — per-trade SL$ lookup from artifact

**Files:**
- Modify: `tests/test_open_strangle_strategy.py` (append new tests)

The strategy already implements `_sl_dollars_for()` and the CSV loader (added in Task 3). This task verifies it works against a real-shaped artifact.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_open_strangle_strategy.py`:

```python
def _write_sl_artifact(tmp_path, rows, p75):
    """Write a temporary IC SL-by-day artifact + meta.json."""
    import json
    csv_path = tmp_path / "sl.csv"
    with open(csv_path, "w") as f:
        f.write("trade_date,ic_total_credit,sl_mult,ic_sl_dollars\n")
        for d, sl in rows:
            f.write(f"{d},{sl/2/100:.2f},2.0,{sl:.2f}\n")
    meta_path = tmp_path / "sl.meta.json"
    meta_path.write_text(json.dumps({"p75": p75, "median": 250, "mean": 280}))
    return str(csv_path)


def test_sl_lookup_uses_per_day_value(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 240.0), ("2025-06-03", 280.0)],
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    when = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    assert s._sl_dollars_for(when) == 240.0  # per-day < p75 → per-day


def test_sl_lookup_caps_at_p75(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 900.0)],  # per-day > p75
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    when = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    assert s._sl_dollars_for(when) == 350.0  # capped at p75


def test_sl_lookup_fallback_when_date_missing(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 240.0)],
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    when = datetime(2025, 7, 15, 9, 35, tzinfo=ET)  # not in artifact
    assert s._sl_dollars_for(when) == 350.0  # falls back to p75


def test_sl_lookup_no_artifact_uses_fallback():
    s = _make_strategy(sl_dollars_csv=None, sl_dollars_fallback=300.0)
    when = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    assert s._sl_dollars_for(when) == 300.0


def test_setup_records_sl_dollars(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 275.0)],
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    setup = out["top_candidate"]
    assert setup.extra["sl_dollars"] == 275.0
```

- [ ] **Step 2: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_open_strangle_strategy.py -v -k "sl_lookup or sl_dollars"
```

Expected: 5 passed (all four `sl_lookup_*` plus `setup_records_sl_dollars`). If anything fails, re-check that Task 3 included the loader (`_load_sl_csv`) and the `sl_dollars` field on `setup.extra`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_open_strangle_strategy.py
git commit -m "test(open-strangle): per-trade SL\$ lookup with p75 cap"
```

---

## Task 5: `OpenStrangleStrategy` — TP/SL/EOD exit logic

**Files:**
- Modify: `src_v2/strategies/open_strangle.py` (replace `check_exit` placeholder)
- Modify: `tests/test_open_strangle_strategy.py` (append exit tests)

- [ ] **Step 1: Write failing tests for exit logic**

Append to `tests/test_open_strangle_strategy.py`:

```python
# ---------------- exit logic ----------------

def _entered_trade(setup, sl_dollars=300.0, entry_dt=None):
    """Build a trade dict shaped the way PositionMonitor/backtest passes it."""
    return {
        "trade_id": "t1",
        "short_put_symbol": "SPX_P_4995",
        "short_call_symbol": "SPX_C_5005",
        "short_put_strike": setup.strikes["short_put"],
        "short_call_strike": setup.strikes["short_call"],
        "put_credit": setup.extra["put_credit"],
        "call_credit": setup.extra["call_credit"],
        "total_credit": setup.total_credit,
        "sl_dollars": setup.extra.get("sl_dollars", sl_dollars),
        "entry_time": entry_dt or datetime(2025, 6, 2, 9, 35, tzinfo=ET),
    }


def _quotes(put_mid: float, call_mid: float):
    return {
        "SPX_P_4995": {"bid": put_mid - 0.02, "ask": put_mid + 0.02},
        "SPX_C_5005": {"bid": call_mid - 0.02, "ask": call_mid + 0.02},
    }


def test_check_exit_hold_when_quiet():
    s = _make_strategy(tp_pct_of_credit=0.50)
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    trade = _entered_trade(setup)
    # Marks at entry credit → no TP or SL trigger.
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] + s._slippage,
        call_mid=setup.extra["call_credit"] + s._slippage,
    )
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    assert action.action == "hold"


def test_check_exit_total_tp_fires():
    s = _make_strategy(tp_pct_of_credit=0.50, tp_mode="total")
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    trade = _entered_trade(setup)
    # Marks at half the credit → total_mark = 0.5*total_credit → TP fires.
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] * 0.5,
        call_mid=setup.extra["call_credit"] * 0.5,
    )
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    assert action.action == "take_profit"
    assert "SPX_P_4995" in action.legs_to_close
    assert "SPX_C_5005" in action.legs_to_close


def test_check_exit_sl_dollars_fires():
    s = _make_strategy(tp_pct_of_credit=0.25, tp_mode="total")
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    # Manually set per-trade SL$ to $50 for a clean trigger.
    setup.extra["sl_dollars"] = 50.0
    trade = _entered_trade(setup)
    trade["sl_dollars"] = 50.0
    # SL fires when (total_mark - total_credit) * 100 >= sl_dollars.
    # total_credit ≈ put_credit + call_credit (per-contract dollars).
    # Choose marks so total_mark = total_credit + 0.51 → loss = $51 > $50 SL.
    bump = (50.0 / 100) + 0.01  # 0.51 in per-contract terms
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] + bump / 2,
        call_mid=setup.extra["call_credit"] + bump / 2,
    )
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    assert action.action == "stop_loss"
    assert "sl_dollars" in action.reason or "SL" in action.reason


def test_check_exit_no_breach_action():
    """Strangle does not exit on spot-breach — relies on $-SL only."""
    s = _make_strategy(tp_pct_of_credit=0.25, tp_mode="total")
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    trade = _entered_trade(setup)
    trade["sl_dollars"] = 1_000_000.0  # effectively unlimited
    # Spot crosses short call but SL$ huge → still hold.
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] * 0.5,
        call_mid=setup.extra["call_credit"] * 0.5,
    )
    quotes["_underlying"] = {"price": setup.strikes["short_call"] + 50}
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    # tp_pct=0.25 means TP fires when mark <= 0.25*credit. At 0.5*credit no TP.
    # No breach action ever fires for strangle → hold.
    assert action.action in ("hold", "take_profit")  # tp_pct may trigger
    # Critical: never stop_loss from breach (no breach logic).
    assert action.action != "stop_loss"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_open_strangle_strategy.py -v -k "check_exit"
```

Expected: 3-4 failures with `action.action == 'hold'` (placeholder returns hold for everything).

- [ ] **Step 3: Implement `check_exit`**

In `src_v2/strategies/open_strangle.py`, replace the placeholder `check_exit` method:

```python
    def check_exit(
        self,
        trade: Dict[str, Any],
        quotes: Dict[str, Dict[str, Any]],
    ) -> ExitAction:
        tp_pct = self._current_tp_pct(trade)

        put_sym = trade.get("short_put_symbol")
        call_sym = trade.get("short_call_symbol")

        put_mark = _mid(quotes.get(put_sym)) if put_sym else None
        call_mark = _mid(quotes.get(call_sym)) if call_sym else None

        legs_all = [s for s in (put_sym, call_sym) if s]
        total_credit = (trade.get("put_credit", 0)
                        + trade.get("call_credit", 0))

        # Total-mode TP/SL.
        if self._tp_mode == "total" and put_mark is not None and call_mark is not None:
            total_mark = put_mark + call_mark
            if total_credit > 0:
                tp_debit = total_credit * tp_pct
                if total_mark <= tp_debit:
                    return ExitAction(
                        action="take_profit",
                        legs_to_close=legs_all,
                        reason=f"total TP: mark={total_mark:.2f} <= tp={tp_debit:.2f}",
                    )
                sl_dollars = float(trade.get("sl_dollars", self._sl_fallback))
                if sl_dollars > 0:
                    # Per-contract loss in dollars: (mark - credit) × multiplier.
                    loss_dollars = (total_mark - total_credit) * self._multiplier
                    if loss_dollars >= sl_dollars:
                        return ExitAction(
                            action="stop_loss",
                            legs_to_close=legs_all,
                            reason=(f"sl_dollars: loss=${loss_dollars:.0f} "
                                    f">= cap=${sl_dollars:.0f}"),
                        )

        # Per-side TP (does not fire in tp_mode=total).
        if self._tp_mode == "per_side":
            for side, sym, credit_key in (
                ("put", put_sym, "put_credit"),
                ("call", call_sym, "call_credit"),
            ):
                if not sym:
                    continue
                m = _mid(quotes.get(sym))
                if m is None:
                    continue
                credit = trade.get(credit_key, 0)
                if credit > 0 and m <= credit * tp_pct:
                    return ExitAction(
                        action="take_profit",
                        legs_to_close=[sym],
                        reason=f"{side} TP: mark={m:.2f} <= tp={credit * tp_pct:.2f}",
                    )

        return ExitAction(action="hold")

    def _current_tp_pct(self, trade: Dict[str, Any]) -> float:
        """Time-ramp tp_pct from initial → end over the trade's life."""
        if self._tp_pct_end is None or self._tp_pct_end == self._tp_pct:
            return self._tp_pct
        clock = getattr(self, "_clock_ref", None)
        entry_time = trade.get("entry_time")
        if clock is None or entry_time is None:
            return self._tp_pct
        from zoneinfo import ZoneInfo as _ZI
        ET_ = _ZI("America/New_York")
        now = clock.now(ET_)
        eod_dt = datetime(now.year, now.month, now.day,
                          self._eod_time.hour, self._eod_time.minute,
                          tzinfo=now.tzinfo)
        if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_dt = entry_time.replace(tzinfo=now.tzinfo)
        else:
            entry_dt = entry_time
        total = (eod_dt - entry_dt).total_seconds()
        elapsed = (now - entry_dt).total_seconds()
        if total <= 0:
            return self._tp_pct_end
        f = max(0.0, min(1.0, elapsed / total))
        return self._tp_pct + (self._tp_pct_end - self._tp_pct) * f
```

- [ ] **Step 4: Run all strangle tests**

```bash
.venv/bin/python -m pytest tests/test_open_strangle_strategy.py -v
```

Expected: all tests pass (12 tests across Tasks 3+4+5).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/open_strangle.py tests/test_open_strangle_strategy.py
git commit -m "feat(open-strangle): TP/SL\$/EOD exit logic"
```

---

## Task 6: Strangle config yaml

**Files:**
- Create: `config/config_spx_0dte_open_strangle.yaml`

- [ ] **Step 1: Read the IC config to fork from**

```bash
.venv/bin/python -c "
import yaml
print(yaml.dump(yaml.safe_load(open('config/config_spx_0dte_open_ic.yaml')),
                default_flow_style=False, sort_keys=False))
" | head -80
```

Note the structure — `bot`, `backtest`, `strategy`, `options`, `entry`, `market` sections.

- [ ] **Step 2: Write the strangle config**

Create `config/config_spx_0dte_open_strangle.yaml`. Start from the IC config and apply these diffs only — keep every other section byte-identical (read the IC yaml and copy the rest verbatim):

```yaml
# config/config_spx_0dte_open_strangle.yaml
#
# SPX 0DTE open strangle. Risk-normalized against the winning open IC via
# per-trade matched $-SL (loaded from data/ic_sl_dollars_by_day.csv).
# See docs/superpowers/specs/2026-05-19-spx-0dte-open-strangle-design.md.

bot:
  symbol: SPX

backtest:
  data_file: data/spy_1m.csv
  price_scale: 10.0
  start_date: "2022-04-18"
  end_date:   "2026-02-28"
  cbbo_cache_dir: data/cbbo_cache
  results_dir: results/spx_0dte_open_strangle

strategy:
  class: src_v2.strategies.open_strangle.OpenStrangleStrategy
  market_open:     "09:30"
  eod_force_close: "15:30"           # before assignment-risk window
  skip_nfp_fridays:    true
  skip_fomc_days:      true
  skip_cpi_days:       true
  skip_fomc_next_day:  true
  skip_cpi_next_day:   false
  skip_weekdays:       ["Fri"]

entry:
  entry_time: "09:35"
  stability_check: true
  stability_fraction: 0.20
  single_candle_fraction: 0.15
  fallback_stability_pts: 12.0
  opening_range_filter:
    enabled: true
    window_start: "09:30"
    window_end:   "09:35"
    max_open_range_em_ratio: 0.5

options:
  strike_step_pts:        5.0
  strike_mode:            em_mult   # WF-CV will sweep delta and em_mult
  em_mult:                0.30
  target_short_delta:     0.20
  min_credit_per_spread:  0.05      # lower than IC (no wing debit)
  max_credit_imbalance:   1.50
  tp_mode:                total
  tp_pct_of_credit:       0.25
  tp_pct_of_credit_end:   null
  slippage_per_spread:    0.10
  contract_multiplier:    100
  quantity:               1
  sl_dollars_csv:         data/ic_sl_dollars_by_day.csv
  sl_dollars_fallback:    300.0     # placeholder; runner overrides from meta.json p75

market:
  market_close: "16:00"
```

- [ ] **Step 3: Verify config loads and resolves the strategy class**

```bash
.venv/bin/python -c "
import yaml
from scripts.run_v2_wf_cv_open_ic import resolve_strategy_class
cfg = yaml.safe_load(open('config/config_spx_0dte_open_strangle.yaml'))
cls = resolve_strategy_class(cfg['strategy']['class'])
print(f'OK: {cls.__module__}.{cls.__name__}')
"
```

Expected: `OK: src_v2.strategies.open_strangle.OpenStrangleStrategy`.

- [ ] **Step 4: Verify strategy can configure() and produce a candidate from the yaml**

```bash
.venv/bin/python -c "
import asyncio, yaml
from src_v2.strategies.open_strangle import OpenStrangleStrategy
from datetime import datetime
from zoneinfo import ZoneInfo

cfg = yaml.safe_load(open('config/config_spx_0dte_open_strangle.yaml'))
# Drop the artifact requirement for this smoke check — the file may not exist yet.
cfg['options']['sl_dollars_csv'] = None
s = OpenStrangleStrategy()
s.configure(cfg)
s._now_dt = datetime(2025, 6, 2, 9, 35, tzinfo=ZoneInfo('America/New_York'))
# Synthetic chain
chain = []
spot = 5000.0
for i in range(-30, 31):
    k = spot + i * 5
    d = abs(k - spot)
    m = max(0.05, 5.0 - d * 0.10)
    chain.append({'strike': k, 'option_type': 'call', 'bid': m - 0.02, 'ask': m + 0.02})
    chain.append({'strike': k, 'option_type': 'put',  'bid': m - 0.02, 'ask': m + 0.02})
out = asyncio.run(s.search_candidates(chain, spot))
assert out['candidates_found'] == 1, out
print('OK:', out['top_candidate'].strikes, out['top_candidate'].total_credit)
"
```

Expected output like `OK: {'short_put': 4995.0, 'short_call': 5005.0} 3.7` (numbers will vary).

- [ ] **Step 5: Commit**

```bash
git add config/config_spx_0dte_open_strangle.yaml
git commit -m "feat(open-strangle): config yaml (em_mult=0.30, tp=0.25, EOD 15:30)"
```

---

## Task 7: Strangle WF-CV runner entry script

**Files:**
- Create: `scripts/run_v2_wf_cv_open_strangle.py`

- [ ] **Step 1: Write the runner**

This is a thin wrapper that reuses the generalized runner's `main()` after swapping `PARAM_GRID`. Create `scripts/run_v2_wf_cv_open_strangle.py`:

```python
"""Walk-forward cross-validation for the SPX 0DTE open strangle.

Sweep: strike_mode × (target_short_delta | em_mult) × tp_pct × tp_pct_end.
SL is per-trade-matched from data/ic_sl_dollars_by_day.csv (not swept).

Usage:
  python scripts/run_v2_wf_cv_open_strangle.py \\
      --config config/config_spx_0dte_open_strangle.yaml \\
      --ratio 5 --folds 4 --top-k 5
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the IC runner's main() but swap in the strangle PARAM_GRID first.
import scripts.run_v2_wf_cv_open_ic as base_runner


EM_MULT_GRID = {
    "options.strike_mode":            ["em_mult"],
    "options.em_mult":                [0.20, 0.30, 0.40, 0.50, 0.75, 1.00],
    "options.tp_pct_of_credit":       [0.25, 0.50, 0.75],
    "options.tp_pct_of_credit_end":   [None, 0.75],
    "strategy.skip_fomc_next_day":    [True],
    "strategy.skip_weekdays":         [["Fri"]],
}
# 6 em_mults × 3 tp × 2 tp_end = 36 combos.

DELTA_GRID = {
    "options.strike_mode":            ["delta"],
    "options.target_short_delta":     [0.10, 0.15, 0.20, 0.25, 0.30],
    "options.tp_pct_of_credit":       [0.25, 0.50, 0.75],
    "options.tp_pct_of_credit_end":   [None, 0.75],
    "strategy.skip_fomc_next_day":    [True],
    "strategy.skip_weekdays":         [["Fri"]],
}
# 5 deltas × 3 tp × 2 tp_end = 30 combos.

# Total across both modes = 66, matching spec §4.


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["em_mult", "delta"], default="em_mult",
                        help="Strike-selection mode to sweep (run once per mode).")
    # Parse only --mode here; let base_runner.main() handle the rest.
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    base_runner.PARAM_GRID = EM_MULT_GRID if args.mode == "em_mult" else DELTA_GRID
    print(f"Strangle WF-CV: sweeping {args.mode} mode "
          f"({len(base_runner.PARAM_GRID['options.' + ('em_mult' if args.mode == 'em_mult' else 'target_short_delta')])} strike configs)")
    base_runner.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with a dry-run on a tiny window**

```bash
.venv/bin/python scripts/run_v2_wf_cv_open_strangle.py \
    --mode em_mult \
    --config config/config_spx_0dte_open_strangle.yaml \
    --start 2025-06-01 --end 2025-06-15 \
    --ratio 5 --folds 2 --top-k 1 \
    --workers 2 2>&1 | tail -30
```

Expected: the runner loads, prints `Strangle WF-CV: sweeping em_mult mode (6 strike configs)`, then `Grid: 36 combos`, runs a tiny WF-CV, and prints WINNING PARAMS. If the artifact CSV doesn't exist yet, this will load with the fallback SL — that's OK for a smoke test, just confirms wiring.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_v2_wf_cv_open_strangle.py
git commit -m "feat(open-strangle): WF-CV runner entry script"
```

---

## Task 8: Strangle trade-log analyzer

**Files:**
- Create: `scripts/analyze_open_strangle_trades.py`

This is a fork of [scripts/analyze_open_ic_trades.py](../../../scripts/analyze_open_ic_trades.py) adjusted for 2 legs.

- [ ] **Step 1: Inspect the reference**

```bash
head -60 scripts/analyze_open_ic_trades.py
wc -l scripts/analyze_open_ic_trades.py
```

Note the columns it reads (look for `short_put`, `long_put`, `wing_width`, etc.) and the summary stats it prints. Anything keyed off the long-leg fields needs to be removed for the strangle version.

- [ ] **Step 2: Copy + adapt**

```bash
cp scripts/analyze_open_ic_trades.py scripts/analyze_open_strangle_trades.py
```

Then edit `scripts/analyze_open_strangle_trades.py`:

1. At the top docstring, change the title to "Open strangle trade-log analyzer".
2. Remove every reference to `long_put`, `long_call`, `wing_width` — these don't exist on strangle trades. Where a section computes "spread mark = short - long," replace with `mark = short_mid` (no long leg).
3. Wherever max-loss is computed from wing width (`width × 100 − credit`), replace with reading `sl_dollars` directly off the trade row.
4. Update the "section header" prints to read "STRANGLE" instead of "IC".
5. Keep the WR / PnL / Sharpe / worst-day / per-fold breakdown logic — those are not IC-specific.

If the IC analyzer has no long-leg fields (just uses `short_*` and `*_credit`), the diff is minimal: change titles and remove `wing_width` references.

- [ ] **Step 3: Smoke-run against any existing strangle trade log**

After Task 7 produces a small results dir, run:

```bash
.venv/bin/python scripts/analyze_open_strangle_trades.py \
    --trades results/spx_0dte_open_strangle/<RUN_DIR>/trades.csv 2>&1 | tail -20
```

Expected: prints a summary block without `KeyError: 'long_put'` or similar.

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_open_strangle_trades.py
git commit -m "feat(open-strangle): trade-log analyzer (forked from IC)"
```

---

## Task 9: Side-by-side strangle vs. IC comparison reporter

**Files:**
- Create: `scripts/compare_strangle_vs_ic.py`

- [ ] **Step 1: Write the comparator**

Create `scripts/compare_strangle_vs_ic.py`:

```python
"""Compare WF-CV winners: open strangle vs. open IC.

Loads each strategy's WINNER trade log and emits a markdown table covering
the spec's acceptance criteria + a per-fold breakdown.

Usage:
  python scripts/compare_strangle_vs_ic.py \\
      --ic results/spx_0dte_open_ic/<RUN_DIR>/trades.csv \\
      --strangle results/spx_0dte_open_strangle/<RUN_DIR>/trades.csv \\
      --ic-meta data/ic_sl_dollars_by_day.meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# Match the IC findings doc (docs/spx_0dte_open_ic_findings.md).
IC_BASELINE: Dict[str, float] = {
    "trades": 172,
    "win_rate": 0.453,
    "total_pnl": 15014.0,
    "sharpe": 4.53,
    "worst_day": -826.0,
    "best_day": 793.0,
    "mean_win": 340.0,
    "mean_loss": -123.0,
}


def _load_trades(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _per_day_pnl(df: pd.DataFrame) -> pd.Series:
    date_col = next(c for c in ("trade_date", "entry_date", "date") if c in df.columns)
    return df.groupby(date_col)["pnl"].sum()


def _metrics(df: pd.DataFrame, label: str) -> Dict[str, float]:
    daily = _per_day_pnl(df)
    daily_arr = daily.to_numpy(dtype=float)
    total = float(daily_arr.sum())
    wins = daily_arr[daily_arr > 0]
    losses = daily_arr[daily_arr < 0]
    wr = float(len(wins) / len(daily_arr)) if len(daily_arr) else 0.0
    sharpe = (
        float(daily_arr.mean() / daily_arr.std() * np.sqrt(252))
        if daily_arr.std() > 0 else 0.0
    )
    return {
        "label": label,
        "trades": int(len(df)),
        "trade_days": int(len(daily_arr)),
        "win_rate": wr,
        "total_pnl": total,
        "sharpe": sharpe,
        "worst_day": float(daily_arr.min()) if len(daily_arr) else 0.0,
        "best_day": float(daily_arr.max()) if len(daily_arr) else 0.0,
        "mean_win": float(wins.mean()) if len(wins) else 0.0,
        "mean_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", required=False,
                    help="IC trade log (optional — defaults to baseline numbers)")
    ap.add_argument("--strangle", required=True)
    ap.add_argument("--ic-meta", required=False)
    args = ap.parse_args()

    if args.ic:
        ic_df = _load_trades(args.ic)
        ic = _metrics(ic_df, "Open IC")
    else:
        ic = {"label": "Open IC (findings baseline)", **IC_BASELINE,
              "trade_days": IC_BASELINE["trades"]}

    str_df = _load_trades(args.strangle)
    st = _metrics(str_df, "Open Strangle")

    # Acceptance gate (per spec §1)
    passes = (
        st["sharpe"] >= IC_BASELINE["sharpe"]
        and st["worst_day"] >= IC_BASELINE["worst_day"]
        and st["total_pnl"] > IC_BASELINE["total_pnl"]
    )

    print()
    print("| Metric | IC | Strangle | Δ | Strangle better? |")
    print("|---|---|---|---|---|")
    for k, label, sign, fmt in [
        ("trade_days",  "Trade days", +1, "{:.0f}"),
        ("win_rate",    "Win rate",   +1, "{:.1%}"),
        ("total_pnl",   "Total PnL",  +1, "${:+,.0f}"),
        ("sharpe",      "Sharpe",     +1, "{:.2f}"),
        ("worst_day",   "Worst day",  +1, "${:+,.0f}"),
        ("best_day",    "Best day",   +1, "${:+,.0f}"),
        ("mean_win",    "Mean win",   +1, "${:+,.0f}"),
        ("mean_loss",   "Mean loss",  +1, "${:+,.0f}"),
    ]:
        ic_v = ic.get(k, 0)
        st_v = st.get(k, 0)
        diff = st_v - ic_v
        better = ("✓" if (sign * diff) > 0 else "✗") if k != "trade_days" else "—"
        print(f"| {label} | {fmt.format(ic_v)} | {fmt.format(st_v)} | "
              f"{fmt.format(diff)} | {better} |")

    print()
    print(f"**Acceptance (must hit all three vs. IC baseline):**")
    print(f"- Sharpe ≥ {IC_BASELINE['sharpe']}: "
          f"{'PASS' if st['sharpe'] >= IC_BASELINE['sharpe'] else 'FAIL'}")
    print(f"- Worst day ≥ ${IC_BASELINE['worst_day']:+.0f}: "
          f"{'PASS' if st['worst_day'] >= IC_BASELINE['worst_day'] else 'FAIL'}")
    print(f"- Total PnL > ${IC_BASELINE['total_pnl']:+.0f}: "
          f"{'PASS' if st['total_pnl'] > IC_BASELINE['total_pnl'] else 'FAIL'}")
    print()
    print(f"**OVERALL: {'STRANGLE WINS' if passes else 'IC HOLDS'}**")

    if args.ic_meta:
        meta = json.loads(Path(args.ic_meta).read_text())
        print()
        print(f"IC SL$ distribution used for risk-normalization: "
              f"p75=${meta['p75']:.0f}, median=${meta['median']:.0f}, "
              f"n_days={meta['n_days']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/compare_strangle_vs_ic.py
git commit -m "feat(open-strangle): side-by-side comparison reporter"
```

---

## Task 10: Extract artifact, run the WF-CV sweep, write findings

This is the operational task that produces the result. It uses everything built above.

- [ ] **Step 1: Identify the winning IC trade log**

```bash
ls -1d results/spx_0dte_open_ic*/ 2>/dev/null
# Pick the most recent run from the v5/v7 series referenced in
# docs/spx_0dte_open_ic_findings.md. If multiple, prefer the one
# whose params match the findings doc (sl_mult=2.0, em_mult=0.30, etc.).
ls -1 results/spx_0dte_open_ic/<MOST_RECENT_DIR>/
```

Expected: a `trades.csv` (or `.parquet`) exists. Note its full path — used in Step 2.

- [ ] **Step 2: Extract the SL-by-day artifact**

```bash
.venv/bin/python scripts/extract_ic_sl_dollars.py \
    --trades results/spx_0dte_open_ic/<MOST_RECENT_DIR>/trades.csv \
    --sl-mult 2.0 \
    --out data/ic_sl_dollars_by_day.csv
cat data/ic_sl_dollars_by_day.meta.json
```

Expected: `n_days` around 170, `p75` somewhere in $200-$500. Record `p75` — you'll use it to update the fallback in the config in Step 3.

- [ ] **Step 3: Update strangle config's fallback to match measured p75**

Edit `config/config_spx_0dte_open_strangle.yaml`, set `options.sl_dollars_fallback` to the measured p75 from Step 2. This only affects days missing from the artifact (defensive).

- [ ] **Step 4: Run the full WF-CV sweep — once per strike mode**

```bash
mkdir -p results/spx_0dte_open_strangle

# em_mult mode (36 combos)
.venv/bin/python scripts/run_v2_wf_cv_open_strangle.py \
    --mode em_mult \
    --config config/config_spx_0dte_open_strangle.yaml \
    --ratio 5 --folds 4 --top-k 5 \
    --objective sharpe_annualised \
    --min-trades 10 \
    2>&1 | tee results/spx_0dte_open_strangle/wf_cv_em_mult.log

# delta mode (30 combos)
.venv/bin/python scripts/run_v2_wf_cv_open_strangle.py \
    --mode delta \
    --config config/config_spx_0dte_open_strangle.yaml \
    --ratio 5 --folds 4 --top-k 5 \
    --objective sharpe_annualised \
    --min-trades 10 \
    2>&1 | tee results/spx_0dte_open_strangle/wf_cv_delta.log
```

Expected: each run completes in roughly the same wall-time as the IC v5 run (no extra legs vs. IC). The "WINNING PARAMS" block + per-fold table lands at the end of each log. Inspect briefly. Pick the higher-Sharpe winner across the two modes for downstream comparison.

- [ ] **Step 5: Compare against IC baseline**

```bash
.venv/bin/python scripts/compare_strangle_vs_ic.py \
    --strangle results/spx_0dte_open_strangle/<RUN_DIR>/trades.csv \
    --ic-meta data/ic_sl_dollars_by_day.meta.json \
    > docs/_strangle_compare_table.md
cat docs/_strangle_compare_table.md
```

Expected: markdown table + final verdict line ("STRANGLE WINS" or "IC HOLDS").

- [ ] **Step 6: Write the findings doc**

Create `docs/spx_0dte_open_strangle_findings.md`. Use the IC findings doc ([docs/spx_0dte_open_ic_findings.md](../../../docs/spx_0dte_open_ic_findings.md)) as structural template. Required sections:

```markdown
# SPX 0DTE Open Strangle — Findings

*Last updated: YYYY-MM-DD*

Walk-forward-validated SPX 0DTE short strangle entered at 09:35 ET, risk-normalized against the [open IC](spx_0dte_open_ic_findings.md) via per-trade matched $-SL (capped at the IC SL distribution's 75th percentile).

## Final configuration

[config/config_spx_0dte_open_strangle.yaml](../config/config_spx_0dte_open_strangle.yaml)

```yaml
# paste the WF-CV WINNING PARAMS block
```

## OOS performance (full 4-year backtest)

| Metric | Open IC | Open Strangle | Δ |
|---|---|---|---|
# paste output of compare_strangle_vs_ic.py

## Per-fold WF-CV

# paste the per-fold table from the runner output

## Acceptance result

# PASS / FAIL on each of the three gates from spec §1

## Risk caveats observed

# Address each item in spec §5: pin/near-pin diagnostic, slippage at high em_mult,
# BP per-trade, per-trade SL coupling, no gate re-derivation.

## Reproduce

# Commands used: extract → run → compare
```

- [ ] **Step 7: Commit findings + the regenerated config**

```bash
git add config/config_spx_0dte_open_strangle.yaml docs/spx_0dte_open_strangle_findings.md
git rm -f docs/_strangle_compare_table.md  # scratch file, not committed
git commit -m "feat(open-strangle): WF-CV findings doc + p75-tuned config"
```

---

## Self-review checklist (for the implementer)

After all tasks complete, before declaring done:

- [ ] `pytest tests/test_open_strangle_strategy.py tests/test_wf_runner_strategy_class.py -v` — all pass
- [ ] `pytest tests_locked/ -v` — the 128 immutable tests still pass (no regressions)
- [ ] `ruff check src_v2/strategies/open_strangle.py scripts/` — clean
- [ ] `black --check src_v2/strategies/open_strangle.py scripts/run_v2_wf_cv_open_strangle.py scripts/extract_ic_sl_dollars.py scripts/compare_strangle_vs_ic.py` — clean
- [ ] Findings doc exists and references the config + WF-CV results dir
- [ ] Acceptance verdict (STRANGLE WINS / IC HOLDS) is stated plainly in §1 of findings
