# GEX Regime Feasibility (Rungs 1 + 1.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tested tooling to test whether the *reliably-observable* dealer-gamma regime (standing all-expiry GEX, and the 1DTE-as-0DTE proxy) predicts 0DTE IC outcomes once volatility is controlled — and decide, via a day-over-day OI-evolution stability test, whether the expensive Rung 2 (intraday trade reconstruction) is even necessary.

**Architecture:** A new pure-logic module `src_v2/backtest/gex_regime.py` builds multi-expiry GEX features on top of the existing `gex.py` primitives. A new `scripts/backfill_oi_multi.py` extends the existing OI backfill to multiple expiries (cheap `statistics` schema). Analysis scripts join per-day regime features to a vol-normalized realized-move dependent variable and run a pre-registered single-hypothesis test across sequential OOS folds. No sweeps (the butterfly's 120-combo sweep overfit).

**Tech Stack:** Python 3.12 (project `.venv`), numpy + scipy (statsmodels NOT available — OLS hand-rolled via `np.linalg.lstsq` + `scipy.stats`), Databento (`statistics` schema), existing `BarLoader` / `CbboStore` / `gex.py`.

**Spec:** `docs/superpowers/specs/2026-06-16-gex-regime-feasibility-0dte-ic-design.md`

---

## File structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src_v2/backtest/gex_regime.py` | Create | Multi-expiry OI slicing, T helpers, standing + proxy regime features, OI-evolution stability |
| `src_v2/backtest/study_expiries.py` | Create | Enumerate study expiries (next dailies + monthlies/EOM ≤120 DTE) from a trading calendar |
| `src_v2/backtest/meic_backfill.py` | Modify | Add `multi_expiry_symbols`, `parse_osi_multi` (retain expiry) |
| `scripts/backfill_oi_multi.py` | Create | Multi-expiry OI backfill → `data/oi_cache_multi/{date}.pkl` ( `{(expiry,strike,is_call):oi}` ) |
| `src_v2/backtest/gex_dependent.py` | Create | Vol-normalized realized-move dependent variable from spot bars |
| `src_v2/backtest/gex_stats.py` | Create | OLS-with-SE, tercile vol-bucketing, sequential-OOS partial-effect test |
| `scripts/gex_rung1.py` | Create | Rung 1 analysis: standing regime → Y, de-confounded, across OOS folds |
| `scripts/gex_rung15.py` | Create | Rung 1.5 analysis: 1DTE proxy predictive test + OI-evolution stability → gate decision |
| `src_v2/backtest/canonical_ic.py` | Create | Single-entry fixed-EM IC priced from CbboStore, hold-to-expiry (confirmation layer) |
| `scripts/gex_ic_confirm.py` | Create | Run the IC confirmation layer split by regime (only if Rung 1 primary survives) |
| `docs/spx_0dte_gex_regime_findings.md` | Create | Findings writeup + scoreboard verdict line |
| `tests/test_gex_regime.py` … | Create | Unit tests (mutable suite — `tests_locked/` is immutable) |

All commands assume `cd ~/Dev/0dte_bot` and the project venv (`.venv/bin/python`, `.venv/bin/pytest`).

---

## Task 1: Multi-expiry slicing + time-to-expiry helpers

**Files:**
- Create: `src_v2/backtest/gex_regime.py`
- Test: `tests/test_gex_regime.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gex_regime.py
from datetime import date
from src_v2.backtest import gex_regime as gr

def test_slice_and_expiries():
    moi = {
        (date(2022, 10, 5), 5000.0, True): 100,
        (date(2022, 10, 5), 5000.0, False): 80,
        (date(2022, 10, 6), 5000.0, True): 40,
    }
    assert gr.expiries(moi) == [date(2022, 10, 5), date(2022, 10, 6)]
    assert gr.slice_expiry(moi, date(2022, 10, 5)) == {(5000.0, True): 100, (5000.0, False): 80}

def test_years_to_expiry_0dte_matches_butterfly():
    # entry 09:30 → 6.5 trading hours to 16:00 → 1.0 trading day → 1/252 yr
    assert abs(gr.years_to_expiry("09:30", 0) - (1.0 / 252.0)) < 1e-12

def test_years_to_expiry_1dte_adds_one_session():
    assert abs(gr.years_to_expiry("09:30", 1) - (2.0 / 252.0)) < 1e-12

def test_intraday_fraction_floor():
    # after close the fraction floors at 0.25h/6.5h, never zero/negative
    assert gr.years_to_expiry("16:30", 0) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gex_regime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src_v2.backtest.gex_regime'`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/gex_regime.py
"""Multi-expiry GEX regime features built on src_v2/backtest/gex.py primitives.

A multi-expiry open-interest map is keyed by (expiry, strike, is_call):
    MultiOi = Dict[Tuple[date, float, bool], int]
Each expiry is evaluated with its own time-to-expiry (kernel widens with T) but a
single, fixed implied vol derived from the 0DTE ATM-straddle EM (flat-vol modeling
choice: the regime SIGN and FLIP location are dominated by the OI distribution, not
the absolute vol level; vol sensitivity is checked in the convergence test).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from src_v2.backtest.gex import bs_gamma, total_gex, gamma_flip, magnet_strike

MultiOi = Dict[Tuple[date, float, bool], int]
Oi = Dict[Tuple[float, bool], int]

TRADING_DAYS_PER_YEAR = 252.0
TRADING_HOURS = 6.5


def expiries(moi: MultiOi) -> List[date]:
    return sorted({e for (e, _, _) in moi})


def slice_expiry(moi: MultiOi, expiry: date) -> Oi:
    return {(k, c): v for (e, k, c), v in moi.items() if e == expiry}


def intraday_trading_fraction(entry_anchor: str) -> float:
    """Remaining fraction of the trading day from entry to 16:00 ET, in trading days.
    Mirrors scripts/analytical_butterfly_eval.py: hours = max(.., 0.25); frac = hours/6.5."""
    mins_to_close = 16 * 60 - (int(entry_anchor[:2]) * 60 + int(entry_anchor[3:5]))
    hours = max(mins_to_close / 60.0, 0.25)
    return hours / TRADING_HOURS


def years_to_expiry(entry_anchor: str, sessions_after_today: int) -> float:
    """sessions_after_today = trading sessions strictly after the entry day up to and
    including the expiry session (0 for 0DTE, 1 for the next-session expiry)."""
    return (intraday_trading_fraction(entry_anchor) + sessions_after_today) / TRADING_DAYS_PER_YEAR
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gex_regime.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/gex_regime.py tests/test_gex_regime.py
git commit -m "feat(gex): multi-expiry OI slicing + time-to-expiry helpers"
```

---

## Task 2: Standing + proxy regime features

**Files:**
- Modify: `src_v2/backtest/gex_regime.py`
- Test: `tests/test_gex_regime.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_gex_regime.py
def test_standing_regime_sign_and_distflip():
    # put-heavy below spot → net short dealer gamma at spot (negative), flip below spot
    moi = {}
    for k in range(4800, 5201, 25):
        moi[(date(2022,10,5), float(k), True)] = 500 if k >= 5000 else 100
        moi[(date(2022,10,5), float(k), False)] = 500 if k <= 5000 else 100
    t = {date(2022,10,5): gr.years_to_expiry("09:30", 0)}
    feat = gr.standing_regime(moi, S=5000.0, em=40.0, sigma=0.20, t_by_expiry=t)
    assert feat.sign in (-1, 0, 1)
    assert feat.total_gex == gr.total_gex_multi(moi, 5000.0, 0.20, t)
    if feat.flip is not None:
        assert abs(feat.dist_flip_em - (5000.0 - feat.flip) / 40.0) < 1e-9

def test_proxy_uses_0dte_kernel_on_1dte_oi():
    # proxy = run single-expiry gex on the 1DTE OI but at the 0DTE time-to-expiry
    moi = {
        (date(2022,10,6), 5000.0, True): 1000,   # 1DTE expiry
        (date(2022,10,6), 5000.0, False): 200,
        (date(2022,10,5), 5000.0, True): 9,      # 0DTE expiry (ignored by proxy)
    }
    feat = gr.proxy_regime(moi, expiry_1dte=date(2022,10,6), S=5000.0, em=40.0,
                           sigma=0.20, entry_anchor="09:30")
    # call-heavy 1DTE → positive dealer gamma
    assert feat.sign == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gex_regime.py -k "standing or proxy" -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'standing_regime'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src_v2/backtest/gex_regime.py

@dataclass
class RegimeFeatures:
    total_gex: float
    sign: int
    flip: Optional[float]
    dist_flip_em: Optional[float]      # (S - flip) / EM  (primary regime variable)
    magnet: Optional[float]
    dist_magnet_em: Optional[float]    # (S - magnet) / EM


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def total_gex_multi(moi: MultiOi, S: float, sigma: float,
                    t_by_expiry: Dict[date, float], mult: int = 100) -> float:
    return sum(total_gex(slice_expiry(moi, e), S, sigma, t_by_expiry[e], mult)
               for e in expiries(moi))


def gamma_flip_multi(moi: MultiOi, S: float, sigma: float,
                     t_by_expiry: Dict[date, float], mult: int = 100) -> Optional[float]:
    strikes = sorted({k for (_, k, _) in moi})
    if not strikes:
        return None
    prev = total_gex_multi(moi, strikes[0], sigma, t_by_expiry, mult)
    for lvl in strikes[1:]:
        cur = total_gex_multi(moi, lvl, sigma, t_by_expiry, mult)
        if (prev < 0 <= cur) or (prev > 0 >= cur):
            return lvl
        prev = cur
    return None


def magnet_strike_multi(moi: MultiOi, S: float, sigma: float,
                        t_by_expiry: Dict[date, float]) -> Optional[float]:
    best: Optional[float] = None
    best_val = -1.0
    for K in sorted({k for (_, k, _) in moi}):
        val = 0.0
        for e in expiries(moi):
            oc = moi.get((e, K, True), 0)
            op = moi.get((e, K, False), 0)
            val += bs_gamma(S, K, t_by_expiry[e], sigma) * (oc + op)
        if val > best_val:
            best_val, best = val, K
    return best


def _features(tg: float, flip: Optional[float], magnet: Optional[float],
              S: float, em: float) -> RegimeFeatures:
    d_flip = (S - flip) / em if (flip is not None and em > 0) else None
    d_mag = (S - magnet) / em if (magnet is not None and em > 0) else None
    return RegimeFeatures(tg, _sign(tg), flip, d_flip, magnet, d_mag)


def standing_regime(moi: MultiOi, S: float, em: float, sigma: float,
                    t_by_expiry: Dict[date, float], mult: int = 100) -> RegimeFeatures:
    tg = total_gex_multi(moi, S, sigma, t_by_expiry, mult)
    flip = gamma_flip_multi(moi, S, sigma, t_by_expiry, mult)
    magnet = magnet_strike_multi(moi, S, sigma, t_by_expiry)
    return _features(tg, flip, magnet, S, em)


def proxy_regime(moi: MultiOi, expiry_1dte: date, S: float, em: float, sigma: float,
                 entry_anchor: str, mult: int = 100) -> RegimeFeatures:
    """1DTE OI distribution evaluated at the 0DTE time-to-expiry kernel."""
    oi = slice_expiry(moi, expiry_1dte)
    t0 = years_to_expiry(entry_anchor, 0)
    tg = total_gex(oi, S, sigma, t0, mult)
    flip = gamma_flip(oi, S, sigma, t0, mult)
    magnet = magnet_strike(oi, S, sigma, t0)
    return _features(tg, flip, magnet, S, em)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gex_regime.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/gex_regime.py tests/test_gex_regime.py
git commit -m "feat(gex): standing + 1DTE-proxy regime features"
```

---

## Task 3: OI-evolution stability (the gate metric)

**Files:**
- Modify: `src_v2/backtest/gex_regime.py`
- Test: `tests/test_gex_regime.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_gex_regime.py
def test_evolution_stability_stable_when_sign_and_flip_constant():
    # same expiry observed on 3 obs-days; OI grows but sign/flip barely move
    oi_a = {(5000.0, True): 600, (5000.0, False): 100, (4950.0, False): 300}
    oi_b = {(5000.0, True): 900, (5000.0, False): 150, (4950.0, False): 320}
    snaps = [(date(2022,10,3), oi_a), (date(2022,10,4), oi_b)]
    res = gr.evolution_stability(snaps, S=5000.0, em=40.0, sigma=0.20, T_eval=gr.years_to_expiry("09:30",0))
    assert res["signs"][0] == res["signs"][1]      # sign stable
    assert res["sign_stable"] is True
    assert res["max_flip_drift_em"] is not None

def test_evolution_stability_unstable_on_sign_flip():
    call_heavy = {(5000.0, True): 1000, (5000.0, False): 100}
    put_heavy = {(5000.0, True): 100, (5000.0, False): 1000}
    snaps = [(date(2022,10,3), call_heavy), (date(2022,10,4), put_heavy)]
    res = gr.evolution_stability(snaps, S=5000.0, em=40.0, sigma=0.20, T_eval=gr.years_to_expiry("09:30",0))
    assert res["sign_stable"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gex_regime.py -k evolution -v`
Expected: FAIL with `AttributeError: ... 'evolution_stability'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src_v2/backtest/gex_regime.py

def evolution_stability(snapshots: List[Tuple[date, Oi]], S: float, em: float,
                        sigma: float, T_eval: float, mult: int = 100) -> dict:
    """Track one expiry's regime across consecutive observation days (e.g. its 2DTE,
    1DTE, 0DTE-prior-close snapshots), all evaluated at a common T_eval kernel.

    Returns sign sequence, flip sequence, max flip drift in EM units, and sign_stable.
    sign_stable=True iff the dealer-gamma sign is identical across every snapshot.
    """
    signs: List[int] = []
    flips: List[Optional[float]] = []
    for _, oi in snapshots:
        signs.append(_sign(total_gex(oi, S, sigma, T_eval, mult)))
        flips.append(gamma_flip(oi, S, sigma, T_eval, mult))
    known = [f for f in flips if f is not None]
    drift = (max(known) - min(known)) / em if (len(known) >= 2 and em > 0) else None
    return {
        "obs_days": [d for d, _ in snapshots],
        "signs": signs,
        "flips": flips,
        "max_flip_drift_em": drift,
        "sign_stable": len(set(signs)) == 1,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gex_regime.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/gex_regime.py tests/test_gex_regime.py
git commit -m "feat(gex): OI-evolution stability metric (Rung 2 gate)"
```

---

## Task 4: Study-expiry enumeration

**Files:**
- Create: `src_v2/backtest/study_expiries.py`
- Test: `tests/test_study_expiries.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_study_expiries.py
from datetime import date, timedelta
from src_v2.backtest.study_expiries import study_expiries, third_fridays, month_end_sessions

def _weekday_calendar(start: date, n: int):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

def test_includes_next_n_sessions():
    cal = _weekday_calendar(date(2024, 1, 2), 200)
    exps = study_expiries(date(2024, 1, 2), cal, n_dailies=5, max_dte_days=120)
    # the entry day + next 4 sessions are present
    assert cal[0] in exps and cal[4] in exps
    # all expiries are within 120 calendar days and on the calendar
    assert all((e - date(2024, 1, 2)).days <= 120 for e in exps)
    assert all(e in set(cal) for e in exps)

def test_includes_a_third_friday():
    cal = _weekday_calendar(date(2024, 1, 2), 200)
    tf = third_fridays(date(2024, 1, 2), date(2024, 1, 2) + timedelta(days=120))
    assert date(2024, 1, 19) in tf   # Jan 2024 third Friday
    exps = study_expiries(date(2024, 1, 2), cal, n_dailies=5, max_dte_days=120)
    assert date(2024, 1, 19) in exps

def test_sorted_and_deduped():
    cal = _weekday_calendar(date(2024, 1, 2), 200)
    exps = study_expiries(date(2024, 1, 2), cal, n_dailies=5, max_dte_days=120)
    assert exps == sorted(set(exps))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_study_expiries.py -v`
Expected: FAIL with `ModuleNotFoundError: ... study_expiries`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/study_expiries.py
"""Select the study's GEX expiries: the near-dated dailies (which carry the 0DTE-like
day-trader gamma, incl. 1DTE/2DTE for the evolution test) plus the standard monthly
third-Fridays and month-end sessions out to max_dte_days. Bounds symbol count/cost
while keeping the dealer-gamma-relevant OI."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Sequence


def third_fridays(start: date, end: date) -> List[date]:
    out: List[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        fridays = [d for d in (date(y, m, day) for day in range(15, 22))
                   if d.weekday() == 4]
        tf = fridays[0]
        if start <= tf <= end:
            out.append(tf)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def month_end_sessions(calendar: Sequence[date], start: date, end: date) -> List[date]:
    by_month: dict = {}
    for d in calendar:
        if start <= d <= end:
            by_month[(d.year, d.month)] = d  # calendar is sorted → last wins
    return sorted(by_month.values())


def study_expiries(session_date: date, calendar: Sequence[date],
                   n_dailies: int = 5, max_dte_days: int = 120) -> List[date]:
    cal = [d for d in sorted(calendar) if d >= session_date]
    end = session_date + timedelta(days=max_dte_days)
    near = cal[:n_dailies]
    cal_set = set(cal)
    monthlies = [d for d in third_fridays(session_date, end) if d in cal_set]
    eom = [d for d in month_end_sessions(cal, session_date, end)]
    picked = {d for d in (near + monthlies + eom) if session_date <= d <= end}
    return sorted(picked)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_study_expiries.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/study_expiries.py tests/test_study_expiries.py
git commit -m "feat(gex): study-expiry enumeration (near dailies + monthlies/EOM)"
```

---

## Task 5: Multi-expiry OSI symbols + parse (retain expiry)

**Files:**
- Modify: `src_v2/backtest/meic_backfill.py` (add two functions; do not change existing ones)
- Test: `tests/test_multi_expiry_symbols.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_expiry_symbols.py
from datetime import date
from src_v2.backtest.meic_backfill import multi_expiry_symbols, parse_osi_multi

def test_multi_expiry_symbols_cover_each_expiry():
    exps = [date(2024, 1, 5), date(2024, 1, 19)]
    syms = multi_expiry_symbols(exps, 5000.0, 5000.0, buffer_pts=10, step_pts=5)
    # 5 strikes (4990..5010) × 2 cp × 2 expiries = 20 symbols
    assert len(syms) == 20
    assert any("240105" in s and s.rstrip().endswith("C05000000") for s in syms)
    assert any("240119" in s for s in syms)

def test_multi_expiry_symbols_asymmetric_down():
    # wider downside so the put-driven gamma flip is never truncated
    syms = multi_expiry_symbols([date(2024, 1, 5)], 5000.0, 5000.0,
                                buffer_down_pts=400, buffer_up_pts=200, step_pts=5)
    strikes = sorted({int(s[-8:]) / 1000.0 for s in syms})
    assert min(strikes) == 4600.0 and max(strikes) == 5200.0   # -400 / +200

def test_parse_osi_multi_roundtrip():
    assert parse_osi_multi("SPXW  240119P04700000") == (date(2024, 1, 19), 4700.0, False)
    assert parse_osi_multi("garbage") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multi_expiry_symbols.py -v`
Expected: FAIL with `ImportError: cannot import name 'multi_expiry_symbols'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src_v2/backtest/meic_backfill.py (near meic_day_symbols)
from datetime import datetime as _dt  # if not already imported

def multi_expiry_symbols(expiries, spot_low, spot_high, buffer_pts=150,
                         buffer_down_pts=None, buffer_up_pts=None, step_pts=5,
                         root="SPXW"):
    """OSI symbols for a list of expiries over [spot_low-down, spot_high+up].

    Defaults to a symmetric buffer_pts; pass buffer_down_pts/buffer_up_pts for an
    asymmetric (wider-downside) window so the put-driven gamma flip — which sits well
    below spot — is never truncated. Truncation must not be vol-correlated, so over-fetch
    generously rather than tightly fit (statistics schema is cheap)."""
    down = buffer_down_pts if buffer_down_pts is not None else buffer_pts
    up = buffer_up_pts if buffer_up_pts is not None else buffer_pts
    base_lo = int((spot_low - down) // step_pts) * step_pts
    base_hi = int(-(-(spot_high + up) // step_pts)) * step_pts
    out = []
    for expiry in expiries:
        k = base_lo
        while k <= base_hi:
            out.append(format_osi(expiry, float(k), is_call=False, root=root))
            out.append(format_osi(expiry, float(k), is_call=True, root=root))
            k += step_pts
    return out


def parse_osi_multi(symbol):
    """'SPXW  240119P04700000' -> (date(2024,1,19), 4700.0, False); None on bad input."""
    s = symbol.strip()
    try:
        exp = _dt.strptime(s[-15:-9], "%y%m%d").date()
        is_call = s[-9] == "C"
        strike = int(s[-8:]) / 1000.0
        return exp, strike, is_call
    except (ValueError, IndexError):
        return None
```

Note: confirm `format_osi` is already imported in `meic_backfill.py` (it is used by `meic_day_symbols`). The `-15:-9` slice extracts `YYMMDD` from the 21-char OSI (6-char root + 6 date + 1 cp + 8 strike).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_multi_expiry_symbols.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/meic_backfill.py tests/test_multi_expiry_symbols.py
git commit -m "feat(gex): multi-expiry OSI symbol builder + parser"
```

---

## Task 6: Multi-expiry OI backfill script

**Files:**
- Create: `scripts/backfill_oi_multi.py`

This mirrors `scripts/backfill_oi.py`'s Databento plumbing but pulls multiple expiries per day and writes a `{(expiry, strike, is_call): oi}` pkl to `data/oi_cache_multi/`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Multi-expiry open-interest backfill for the GEX regime study.

For each trading day: pick study expiries (next dailies + monthlies/EOM ≤120 DTE) from
the bar calendar, pull the OPRA `statistics` OPEN_INTEREST stat for SPXW strikes around
the 09:30 spot across those expiries, and write a per-day pickle
{(expiry, strike, is_call): open_interest} to data/oi_cache_multi/.

OI is published pre-market and reflects the prior session's close — exactly the standing
positioning carried into the session. Statistics is a tiny schema; run --dry-run first.

Usage:
  .venv/bin/python scripts/backfill_oi_multi.py --start 2022-05-11 --end 2026-06-13 --dry-run
  .venv/bin/python scripts/backfill_oi_multi.py --start 2022-05-11 --end 2026-06-13 --max-cost-usd 60
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import dotenv_values
import databento as db
from databento import StatType

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.databento_cost import CostLedger
from src_v2.backtest.meic_backfill import (multi_expiry_symbols, parse_osi_multi,
                                           meic_window_utc)
from src_v2.backtest.study_expiries import study_expiries

OI_STAT = int(StatType.OPEN_INTEREST)


def oi_multi_from_df(df) -> Dict[Tuple[date, float, bool], int]:
    out: Dict[Tuple[date, float, bool], int] = {}
    sub = df[df["stat_type"] == OI_STAT]
    if "symbol" not in sub.columns:
        return out
    for sym, q in zip(sub["symbol"], sub["quantity"]):
        pk = parse_osi_multi(str(sym))
        if pk is None or q is None:
            continue
        try:
            out[pk] = int(q)
        except (ValueError, TypeError):
            continue
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-expiry SPXW OI backfill (GEX regime study)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--out", default="data/oi_cache_multi")
    p.add_argument("--ledger", default="data/databento_cost_ledger.jsonl")
    p.add_argument("--buffer-down-pts", type=int, default=400)  # wider downside: flip sits below spot
    p.add_argument("--buffer-up-pts", type=int, default=200)
    p.add_argument("--step-pts", type=int, default=5)
    p.add_argument("--n-dailies", type=int, default=5)
    p.add_argument("--max-dte-days", type=int, default=120)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=60.0)
    args = p.parse_args()

    env = dotenv_values(str(Path(__file__).resolve().parent.parent / ".env"))
    api_key = env.get("DATABENTO_API_KEY") or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    loader = BarLoader(args.bars, price_scale=args.price_scale, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    calendar = [d.session_date for d in days]
    print(f"{len(days)} trading days {start}..{end}")

    client = db.Historical(api_key)
    ledger = CostLedger(Path(args.ledger))
    run_cost = 0.0
    n_calls = n_skipped = 0

    for day in days:
        d = day.session_date
        out_path = out_dir / f"{d.isoformat()}.pkl"
        if not args.dry_run and out_path.exists() and not args.force:
            n_skipped += 1
            continue
        b0930 = next((b for b in day.bars if b["ts_et"].time().hour == 9
                      and b["ts_et"].time().minute == 30), None)
        if b0930 is None:
            n_skipped += 1
            continue
        center = float(b0930["open"])
        exps = study_expiries(d, calendar, n_dailies=args.n_dailies,
                              max_dte_days=args.max_dte_days)
        symbols = multi_expiry_symbols(exps, center, center,
                                       buffer_down_pts=args.buffer_down_pts,
                                       buffer_up_pts=args.buffer_up_pts,
                                       step_pts=args.step_pts)
        if not symbols:
            n_skipped += 1
            continue
        start_iso, end_iso = meic_window_utc(d, "04:00", "12:00")
        try:
            cost = client.metadata.get_cost(
                dataset="OPRA.PILLAR", schema="statistics", symbols=symbols,
                stype_in="raw_symbol", start=start_iso, end=end_iso)
        except Exception as e:
            print(f"{d}: cost ERROR {type(e).__name__}: {str(e)[:80]}")
            cost = 0.0

        if args.dry_run:
            run_cost += cost
            ledger.record(d, "dry_run", cost, len(symbols), 0)
            print(f"{d}: {len(exps):>2} exps {len(symbols):>5} syms  est ${cost:.5f}  cum ${run_cost:.3f}")
            continue

        if args.max_cost_usd and run_cost >= args.max_cost_usd:
            print(f"STOP: cum cost ${run_cost:.2f} >= cap ${args.max_cost_usd}")
            break
        try:
            df = client.timeseries.get_range(
                dataset="OPRA.PILLAR", schema="statistics", symbols=symbols,
                stype_in="raw_symbol", start=start_iso, end=end_iso).to_df()
        except Exception as e:
            print(f"{d}: fetch ERROR {type(e).__name__}: {str(e)[:80]}")
            continue
        oi = oi_multi_from_df(df) if len(df) else {}
        with open(out_path, "wb") as f:
            pickle.dump(oi, f)
        run_cost += cost
        n_calls += 1
        ledger.record(d, "fetch", cost, len(symbols), len(oi))
        print(f"{d}: {len(exps):>2} exps {len(oi):>5} OI rows  ${cost:.5f}  cum ${run_cost:.3f}")

    print(f"done: {n_calls} fetched, {n_skipped} skipped, cum ${run_cost:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-check it imports and parses args**

Run: `.venv/bin/python scripts/backfill_oi_multi.py --start 2024-01-02 --end 2024-01-02 --dry-run --bars data/spy_1m.csv`
Expected: prints one line `2024-01-02: NN exps NNNN syms est $0.00xxx cum $...` (or a clean skip if no 09:30 bar / no key). No traceback.

- [ ] **Step 3: Commit**

```bash
git add scripts/backfill_oi_multi.py
git commit -m "feat(gex): multi-expiry OI backfill script"
```

---

## Task 7: Cost dry-run, then backfill the multi-expiry OI

**Files:** none (data generation)

- [ ] **Step 1: Full-range cost dry-run**

Run: `.venv/bin/python scripts/backfill_oi_multi.py --start 2022-05-11 --end 2026-06-13 --dry-run --bars data/spy_1m.csv | tail -5`
Expected: a final `cum $XX.xxx` total. **Decision gate:** if the cum estimate is > ~$80, reduce `--max-dte-days` (e.g. 90) or `--n-dailies` and re-run the dry-run before spending. Record the estimate in the findings doc.

- [ ] **Step 2: Run the real backfill**

Run: `.venv/bin/python scripts/backfill_oi_multi.py --start 2022-05-11 --end 2026-06-13 --max-cost-usd <dry_run_estimate_plus_10pct> --bars data/spy_1m.csv`
Expected: per-day fetch lines; `data/oi_cache_multi/*.pkl` populated (~900+ files).

- [ ] **Step 3: Verify cache integrity**

Run:
```bash
.venv/bin/python - <<'PY'
import pickle, glob
from datetime import date
fs = sorted(glob.glob('data/oi_cache_multi/*.pkl'))
print('files:', len(fs))
d = pickle.load(open(fs[len(fs)//2], 'rb'))
exps = sorted({e for (e, _, _) in d})
print('sample file expiries:', len(exps), exps[:3], '...', exps[-1])
print('rows:', len(d))
PY
```
Expected: hundreds of files; each file spans multiple expiries (the 0DTE through ~120-DTE set).

- [ ] **Step 4: Convergence / no-truncation check (the load-bearing guardrail)**

The gamma flip must sit *interior* to the fetched strike window on ~every day, else the window truncated it and the regime variable is biased (the butterfly confound). Run:
```bash
.venv/bin/python - <<'PY'
import pickle, glob
from datetime import date
from src_v2.backtest import gex_regime as gr
from src_v2.backtest.gex import gamma_flip
boundary_hits = total = 0
for f in sorted(glob.glob('data/oi_cache_multi/*.pkl')):
    moi = pickle.load(open(f, 'rb'))
    if not moi: continue
    d = date.fromisoformat(f.split('/')[-1][:-4])
    oi0 = gr.slice_expiry(moi, d)        # 0DTE expiry
    if not oi0: continue
    strikes = sorted({k for (k, _) in oi0})
    S = strikes[len(strikes)//2]
    flip = gamma_flip(oi0, S, 0.20, gr.years_to_expiry("09:45", 0))
    total += 1
    if flip is not None and (flip <= strikes[1] or flip >= strikes[-2]):
        boundary_hits += 1
print(f"flip at window boundary on {boundary_hits}/{total} days "
      f"({boundary_hits/max(total,1):.1%})")
PY
```
Expected: boundary-hit rate **< ~5%**. If higher, the downside window is too narrow — increase `--buffer-down-pts` (e.g. 600) and re-run Task 7 Step 2 with `--force`. Record the final rate in the findings doc as the convergence evidence.

- [ ] **Step 5: Commit a note (data is gitignored; commit the cost record)**

```bash
git add data/databento_cost_ledger.jsonl
git commit -m "chore(gex): record multi-expiry OI backfill cost"
```
(If `data/` is fully gitignored, skip the add and just note the cost in the findings doc instead.)

---

## Task 8: Vol-normalized realized-move dependent variable

**Files:**
- Create: `src_v2/backtest/gex_dependent.py`
- Test: `tests/test_gex_dependent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gex_dependent.py
from src_v2.backtest.gex_dependent import realized_move

def _bar(h, m, o, hi, lo, c):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return {"ts_et": datetime(2024,1,2,h,m,tzinfo=ZoneInfo("America/New_York")),
            "open": o, "high": hi, "low": lo, "close": c}

def test_realized_move_terminal_and_excursion():
    bars = [
        _bar(9,30, 5000, 5000, 5000, 5000),
        _bar(12,0, 5000, 5040, 4950, 5010),   # excursion 50 down / 40 up
        _bar(15,55, 5000, 5005, 4995, 5030),  # close 5030 → terminal +30
    ]
    r = realized_move(bars, entry_anchor="09:30", em=40.0)
    assert r["entry_spot"] == 5000
    assert abs(r["terminal_abs_em"] - 30/40) < 1e-9
    assert abs(r["max_excursion_em"] - 50/40) < 1e-9   # max(|5040-5000|,|4950-5000|)=50

def test_realized_move_none_when_no_entry_bar():
    assert realized_move([_bar(9,31,1,1,1,1)], "09:30", 40.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gex_dependent.py -v`
Expected: FAIL with `ModuleNotFoundError: ... gex_dependent`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/gex_dependent.py
"""Vol-normalized realized-move dependent variable (the clean mechanism ICs bet on).

terminal_abs_em   = |close_eod - entry_spot| / EM     (settlement-relevant)
max_excursion_em  = max(|high - entry_spot|, |low - entry_spot|) over [entry, eod] / EM
                    (touch/stop-relevant — the measure weighted most)
"""
from __future__ import annotations

from datetime import time
from typing import Dict, List, Optional


def _to_min(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def realized_move(bars: List[Dict], entry_anchor: str, em: float,
                  eod_after: time = time(15, 55)) -> Optional[Dict]:
    if em is None or em <= 0:
        return None
    anchor_min = _to_min(entry_anchor)
    entry_bar = next((b for b in bars
                      if b["ts_et"].hour * 60 + b["ts_et"].minute == anchor_min), None)
    if entry_bar is None:
        return None
    entry_spot = float(entry_bar["open"])
    window = [b for b in bars if b["ts_et"].hour * 60 + b["ts_et"].minute >= anchor_min]
    if not window:
        return None
    hi = max(float(b["high"]) for b in window)
    lo = min(float(b["low"]) for b in window)
    eod = next((b for b in reversed(window) if b["ts_et"].time() >= eod_after), window[-1])
    close = float(eod["close"])
    return {
        "entry_spot": entry_spot,
        "terminal_abs_em": abs(close - entry_spot) / em,
        "max_excursion_em": max(abs(hi - entry_spot), abs(lo - entry_spot)) / em,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gex_dependent.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/gex_dependent.py tests/test_gex_dependent.py
git commit -m "feat(gex): vol-normalized realized-move dependent variable"
```

---

## Task 9: De-confound statistics (OLS + tercile buckets)

**Files:**
- Create: `src_v2/backtest/gex_stats.py`
- Test: `tests/test_gex_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gex_stats.py
import numpy as np
from src_v2.backtest.gex_stats import ols, tercile_buckets, partial_effect

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.1, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    beta, se, t, p = ols(X, y)
    assert abs(beta[0] - 1.0) < 0.05
    assert abs(beta[1] - 2.0) < 0.05
    assert abs(beta[2] + 0.5) < 0.05
    assert p[1] < 1e-6   # x1 highly significant

def test_tercile_buckets_split_three_ways():
    em = np.arange(30.0)
    b = tercile_buckets(em)
    assert set(b) == {0, 1, 2}
    assert (b == 0).sum() == (b == 2).sum() == 10

def test_partial_effect_zero_when_regime_is_pure_vol_proxy():
    # regime perfectly collinear with EM → partial effect ~0 after controlling EM
    rng = np.random.default_rng(1)
    n = 1500
    em = rng.uniform(20, 60, n)
    regime = (em - em.mean()) / em.std()          # regime IS vol, nothing more
    y = 0.3 * (em / em.mean()) + rng.normal(scale=0.05, size=n)
    res = partial_effect(y, regime, em)
    assert abs(res["regime_t"]) < 3.0             # not robustly significant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gex_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: ... gex_stats`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/gex_stats.py
"""De-confound statistics for the GEX regime study (statsmodels not available)."""
from __future__ import annotations

from typing import Dict
import numpy as np
from scipy import stats


def ols(X: np.ndarray, y: np.ndarray):
    """Ordinary least squares with classical standard errors.
    X includes its own intercept column. Returns (beta, se, t, p)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, p


def tercile_buckets(em: np.ndarray) -> np.ndarray:
    """0/1/2 by EM tercile (rank-based, balanced)."""
    order = np.argsort(np.argsort(em))
    n = len(em)
    return np.minimum((order * 3) // n, 2)


def partial_effect(y: np.ndarray, regime: np.ndarray, em: np.ndarray) -> Dict:
    """Regress y ~ 1 + regime + em + regime*em; report the partial regime effect
    (coefficient, t-stat, p) holding EM fixed."""
    y = np.asarray(y, float); regime = np.asarray(regime, float); em = np.asarray(em, float)
    X = np.column_stack([np.ones_like(y), regime, em, regime * em])
    beta, se, t, p = ols(X, y)
    return {"regime_coef": float(beta[1]), "regime_se": float(se[1]),
            "regime_t": float(t[1]), "regime_p": float(p[1]),
            "interaction_coef": float(beta[3]), "interaction_t": float(t[3]),
            "n": int(len(y))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gex_stats.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/gex_stats.py tests/test_gex_stats.py
git commit -m "feat(gex): de-confound OLS + tercile bucket stats"
```

---

## Task 10: Sequential-OOS fold harness for the partial-effect test

**Files:**
- Modify: `src_v2/backtest/gex_stats.py`
- Test: `tests/test_gex_stats.py`

Reuse the contiguous sequential-OOS split pattern from `dc_time_machine/wf_cv.py:24` (pure, no deps).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_gex_stats.py
from src_v2.backtest.gex_stats import oos_partial_effect

def test_oos_partial_effect_reports_per_fold():
    rng = np.random.default_rng(2)
    n = 1000
    regime = rng.normal(size=n)
    em = rng.uniform(20, 60, n)
    y = 0.4 * regime + 0.2 * (em / em.mean()) + rng.normal(scale=0.05, size=n)  # real regime effect
    res = oos_partial_effect(y, regime, em, n_folds=4)
    assert len(res["folds"]) == 4
    # a genuine positive effect should appear in a majority of OOS folds
    assert sum(1 for f in res["folds"] if f["regime_coef"] > 0 and f["regime_p"] < 0.1) >= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gex_stats.py -k oos -v`
Expected: FAIL with `ImportError: cannot import name 'oos_partial_effect'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src_v2/backtest/gex_stats.py

def _sequential_folds(n: int, n_folds: int):
    """Contiguous, non-overlapping OOS blocks in time order (rows assumed date-sorted)."""
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n_folds)]


def oos_partial_effect(y, regime, em, n_folds: int = 4) -> Dict:
    """Run partial_effect within each sequential OOS block. Rows MUST be date-sorted."""
    y = np.asarray(y, float); regime = np.asarray(regime, float); em = np.asarray(em, float)
    folds = []
    for k, (lo, hi) in enumerate(_sequential_folds(len(y), n_folds)):
        sl = slice(lo, hi)
        if hi - lo < 20:
            folds.append({"fold": k, "n": hi - lo, "regime_coef": float("nan"),
                          "regime_t": float("nan"), "regime_p": float("nan"), "thin": True})
            continue
        pe = partial_effect(y[sl], regime[sl], em[sl])
        pe["fold"] = k
        folds.append(pe)
    valid = [f for f in folds if not f.get("thin")]
    n_pos_sig = sum(1 for f in valid if f["regime_coef"] > 0 and f["regime_p"] < 0.1)
    n_neg_sig = sum(1 for f in valid if f["regime_coef"] < 0 and f["regime_p"] < 0.1)
    return {"folds": folds, "n_valid": len(valid),
            "n_pos_sig": n_pos_sig, "n_neg_sig": n_neg_sig,
            "consistent_majority": max(n_pos_sig, n_neg_sig) > len(valid) / 2}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gex_stats.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src_v2/backtest/gex_stats.py tests/test_gex_stats.py
git commit -m "feat(gex): sequential-OOS partial-effect harness"
```

---

## Task 11: Rung 1 analysis script (standing GEX, de-confounded)

**Files:**
- Create: `scripts/gex_rung1.py`

Assembles per-day rows: standing regime feature `(S-flip)/EM`, dependent `max_excursion_em` / `terminal_abs_em`, and EM; runs the de-confound (full-sample partial effect, per-vol-bucket effect, and sequential OOS). Reuses the EM/spot extraction shown by the scouts (`compute_em` via the cbbo cache; sigma via `iv_from_em`).

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Rung 1: does the standing all-expiry GEX regime predict vol-normalized realized
move, after controlling for EM? Pre-registered single hypothesis, sequential OOS."""
from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest import gex_regime as gr
from src_v2.backtest.gex import iv_from_em
from src_v2.backtest.gex_dependent import realized_move
from src_v2.backtest.gex_stats import partial_effect, tercile_buckets, oos_partial_effect

ENTRY = "09:45"   # canonical single entry; matches the bot's first scheduled window


def sessions_after(calendar, d, expiry) -> int:
    after = [x for x in calendar if d < x <= expiry]
    return len(after)


def build_rows(oi_dir, bars, cbbo_dir, start, end):
    loader = BarLoader(bars, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))
    calendar = [x.session_date for x in days]
    store = CbboStore(cbbo_dir)
    rows = []
    for day in days:
        d = day.session_date
        oi_pkl = Path(oi_dir) / f"{d.isoformat()}.pkl"
        if not oi_pkl.exists():
            continue
        with open(oi_pkl, "rb") as f:
            moi = pickle.load(f)
        if not moi:
            continue
        try:
            store.load_day(d)
        except Exception:
            continue
        b = next((x for x in day.bars if x["ts_et"].time().hour == 9
                  and x["ts_et"].time().minute == 45), None)
        if b is None:
            continue
        spot = store.derive_spot(ENTRY, float(b["open"])) or float(b["open"])
        # EM via ATM straddle from cbbo
        atm = round(spot / 5) * 5
        cq = store.lookup(ENTRY, float(atm), True); pq = store.lookup(ENTRY, float(atm), False)
        if not cq or not pq:
            continue
        em = (cq[0] + cq[1]) / 2 + (pq[0] + pq[1]) / 2
        if em <= 0:
            continue
        t0 = gr.years_to_expiry(ENTRY, 0)
        sigma = iv_from_em(spot, em, t0)
        if sigma <= 0:
            continue
        t_by = {e: gr.years_to_expiry(ENTRY, sessions_after(calendar, d, e))
                for e in gr.expiries(moi)}
        feat = gr.standing_regime(moi, spot, em, sigma, t_by)
        if feat.dist_flip_em is None:
            continue
        rm = realized_move(day.bars, ENTRY, em)
        if rm is None:
            continue
        rows.append({"date": d, "dist_flip_em": feat.dist_flip_em, "sign": feat.sign,
                     "em": em, "excursion": rm["max_excursion_em"],
                     "terminal": rm["terminal_abs_em"]})
    rows.sort(key=lambda r: r["date"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oi-dir", default="data/oi_cache_multi")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--cbbo", default="data/cbbo_cache_meic")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-06-13")
    p.add_argument("--y", choices=["excursion", "terminal"], default="excursion")
    args = p.parse_args()

    rows = build_rows(args.oi_dir, args.bars, args.cbbo,
                      date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"rows: {len(rows)}")
    if len(rows) < 100:
        print("INSUFFICIENT DATA"); return
    y = np.array([r[args.y] for r in rows])
    regime = np.array([r["dist_flip_em"] for r in rows])
    em = np.array([r["em"] for r in rows])

    full = partial_effect(y, regime, em)
    print(f"\n[FULL SAMPLE] regime coef={full['regime_coef']:+.4f} "
          f"t={full['regime_t']:+.2f} p={full['regime_p']:.4f} "
          f"interaction t={full['interaction_t']:+.2f}")

    b = tercile_buckets(em)
    print("\n[BY EM TERCILE]")
    for tb in (0, 1, 2):
        m = b == tb
        if m.sum() < 30:
            continue
        pe = partial_effect(y[m], regime[m], em[m])
        print(f"  tercile {tb} (n={m.sum()}, EM {em[m].min():.0f}-{em[m].max():.0f}): "
              f"coef={pe['regime_coef']:+.4f} t={pe['regime_t']:+.2f} p={pe['regime_p']:.4f}")

    oos = oos_partial_effect(y, regime, em, n_folds=4)
    print("\n[SEQUENTIAL OOS]")
    for f in oos["folds"]:
        if f.get("thin"):
            print(f"  fold {f['fold']}: thin (n={f['n']})"); continue
        print(f"  fold {f['fold']} (n={f['n']}): coef={f['regime_coef']:+.4f} "
              f"t={f['regime_t']:+.2f} p={f['regime_p']:.4f}")
    print(f"\nVERDICT: consistent_majority={oos['consistent_majority']} "
          f"(pos_sig={oos['n_pos_sig']}, neg_sig={oos['n_neg_sig']}, valid={oos['n_valid']})")
    print("PASS Rung 1 if consistent_majority AND >=2/3 terciles same-sign significant.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/gex_rung1.py --y excursion`
Expected: prints row count, full-sample/tercile/OOS coefficients, and a VERDICT line. Also run `--y terminal` for the secondary measure. Capture both outputs for the findings doc.

- [ ] **Step 3: Commit**

```bash
git add scripts/gex_rung1.py
git commit -m "feat(gex): Rung 1 standing-GEX de-confounded analysis"
```

---

## Task 12: Rung 1.5 analysis (1DTE proxy + OI-evolution stability gate)

**Files:**
- Create: `scripts/gex_rung15.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Rung 1.5: (a) does the 1DTE-as-0DTE proxy predict realized move and add value over
the standing signal; (b) OI-evolution stability across each expiry's 2DTE→1DTE→0DTE
prior-close snapshots → the gate deciding whether Rung 2 (trades) is needed."""
from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest import gex_regime as gr
from src_v2.backtest.gex import iv_from_em
from src_v2.backtest.gex_dependent import realized_move
from src_v2.backtest.gex_stats import partial_effect, oos_partial_effect

ENTRY = "09:45"


def load_moi(oi_dir, d):
    pk = Path(oi_dir) / f"{d.isoformat()}.pkl"
    if not pk.exists():
        return None
    with open(pk, "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oi-dir", default="data/oi_cache_multi")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--cbbo", default="data/cbbo_cache_meic")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-06-13")
    args = p.parse_args()

    loader = BarLoader(args.bars, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=date.fromisoformat(args.start),
                                 end_date=date.fromisoformat(args.end)))
    calendar = [x.session_date for x in days]
    store = CbboStore(args.cbbo)

    proxy_rows, stab = [], []
    for i, day in enumerate(days):
        d = day.session_date
        moi = load_moi(args.oi_dir, d)
        if not moi:
            continue
        exps = gr.expiries(moi)
        next_exps = [e for e in exps if e > d]
        if not next_exps:
            continue
        e1 = next_exps[0]   # 1DTE expiry
        try:
            store.load_day(d)
        except Exception:
            continue
        b = next((x for x in day.bars if x["ts_et"].time().hour == 9
                  and x["ts_et"].time().minute == 45), None)
        if b is None:
            continue
        spot = store.derive_spot(ENTRY, float(b["open"])) or float(b["open"])
        atm = round(spot / 5) * 5
        cq = store.lookup(ENTRY, float(atm), True); pq = store.lookup(ENTRY, float(atm), False)
        if not cq or not pq:
            continue
        em = (cq[0] + cq[1]) / 2 + (pq[0] + pq[1]) / 2
        if em <= 0:
            continue
        sigma = iv_from_em(spot, em, gr.years_to_expiry(ENTRY, 0))
        if sigma <= 0:
            continue
        feat = gr.proxy_regime(moi, e1, spot, em, sigma, ENTRY)
        rm = realized_move(day.bars, ENTRY, em)
        if feat.dist_flip_em is None or rm is None:
            continue
        proxy_rows.append({"date": d, "dist_flip_em": feat.dist_flip_em, "em": em,
                           "excursion": rm["max_excursion_em"]})

        # OI-evolution: track THIS expiry e (=d, the 0DTE today) across its prior snapshots
        # i.e. its appearances in the caches of d-2, d-1 (when it was 2DTE, 1DTE) and d (0DTE prior-close)
        snaps = []
        for back in (2, 1, 0):
            j = i - back
            if j < 0:
                continue
            obs_day = days[j].session_date
            m2 = load_moi(args.oi_dir, obs_day)
            if not m2:
                continue
            oi_e = gr.slice_expiry(m2, d)   # the expiry that is 0DTE *today*
            if oi_e:
                snaps.append((obs_day, oi_e))
        if len(snaps) >= 2:
            stab.append(gr.evolution_stability(snaps, spot, em, sigma,
                                               gr.years_to_expiry(ENTRY, 0)))

    print(f"proxy rows: {len(proxy_rows)}  stability samples: {len(stab)}")
    if len(proxy_rows) >= 100:
        y = np.array([r["excursion"] for r in proxy_rows])
        regime = np.array([r["dist_flip_em"] for r in proxy_rows])
        em = np.array([r["em"] for r in proxy_rows])
        full = partial_effect(y, regime, em)
        oos = oos_partial_effect(y, regime, em, n_folds=4)
        print(f"[PROXY] full coef={full['regime_coef']:+.4f} t={full['regime_t']:+.2f} "
              f"p={full['regime_p']:.4f}  OOS consistent={oos['consistent_majority']}")

    if stab:
        sign_stable = sum(1 for s in stab if s["sign_stable"]) / len(stab)
        drifts = [s["max_flip_drift_em"] for s in stab if s["max_flip_drift_em"] is not None]
        med_drift = float(np.median(drifts)) if drifts else float("nan")
        print(f"\n[OI-EVOLUTION STABILITY]")
        print(f"  sign stable across final days: {sign_stable:.1%} of samples")
        print(f"  median flip drift: {med_drift:.2f} EM")
        gate = "STICKY → Rung 2 likely UNNECESSARY" if (sign_stable >= 0.7 and med_drift < 0.5) \
            else "FLOW-DRIVEN → Rung 2 REQUIRED"
        print(f"  GATE: {gate}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/gex_rung15.py`
Expected: proxy predictive coefficients + the `[OI-EVOLUTION STABILITY]` block with a `GATE:` line. Capture for the findings doc.

- [ ] **Step 3: Commit**

```bash
git add scripts/gex_rung15.py
git commit -m "feat(gex): Rung 1.5 proxy + OI-evolution stability gate"
```

---

## Task 13: Canonical IC confirmation layer (run only if Rung 1 primary survives)

**Files:**
- Create: `src_v2/backtest/canonical_ic.py`
- Create: `scripts/gex_ic_confirm.py`
- Test: `tests/test_canonical_ic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_ic.py
from src_v2.backtest.canonical_ic import price_ic

def test_price_ic_credit_and_settlement():
    # synthetic chain: {(strike, is_call): (bid, ask)}
    chain = {
        (5120.0, True): (4.0, 4.4), (5180.0, True): (1.0, 1.2),    # short/long call
        (4880.0, False): (4.0, 4.4), (4820.0, False): (1.0, 1.2),  # short/long put
    }
    r = price_ic(chain, spot=5000.0, em=120.0, em_mult=1.0, wing=60,
                 step=5, slippage=0.10, settle_spot=5000.0)
    # credit = (4.2-1.1-0.10)*2 sides = 3.0*2 = 6.0 ; settle ATM → no loss
    assert abs(r["credit"] - 6.0) < 1e-6
    assert abs(r["pnl"] - 6.0 * 100) < 1e-6

def test_price_ic_loss_capped_at_wing():
    chain = {
        (5120.0, True): (4.0, 4.4), (5180.0, True): (1.0, 1.2),
        (4880.0, False): (4.0, 4.4), (4820.0, False): (1.0, 1.2),
    }
    r = price_ic(chain, spot=5000.0, em=120.0, em_mult=1.0, wing=60,
                 step=5, slippage=0.10, settle_spot=5400.0)  # blows through call wing
    assert abs(r["call_loss"] - 60) < 1e-6   # capped at wing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_canonical_ic.py -v`
Expected: FAIL with `ModuleNotFoundError: ... canonical_ic`

- [ ] **Step 3: Write minimal implementation**

```python
# src_v2/backtest/canonical_ic.py
"""Canonical single-entry 0DTE IC: EM-relative short strikes, fixed wing, hold-to-expiry,
mid-minus-slippage fills. Mirrors src_v2/strategies/meic.py + engine.py settlement."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def _mid(q: Optional[Tuple[float, float]]) -> Optional[float]:
    if not q or q[0] <= 0 or q[1] <= 0:
        return None
    return (q[0] + q[1]) / 2


def price_ic(chain: Dict[Tuple[float, bool], Tuple[float, float]], spot: float, em: float,
             em_mult: float, wing: int, step: int, slippage: float,
             settle_spot: Optional[float], mult: int = 100,
             commission_per_leg: float = 1.0) -> Optional[Dict]:
    short_call = math.ceil((spot + em_mult * em) / step) * step
    short_put = math.floor((spot - em_mult * em) / step) * step
    long_call, long_put = short_call + wing, short_put - wing
    legs = {"sc": _mid(chain.get((float(short_call), True))),
            "lc": _mid(chain.get((float(long_call), True))),
            "sp": _mid(chain.get((float(short_put), False))),
            "lp": _mid(chain.get((float(long_put), False)))}
    if any(v is None for v in legs.values()):
        return None
    call_credit = legs["sc"] - legs["lc"] - slippage
    put_credit = legs["sp"] - legs["lp"] - slippage
    credit = call_credit + put_credit
    if settle_spot is None:
        return {"credit": credit, "pnl": None}
    call_loss = min(max(0.0, settle_spot - short_call), wing)
    put_loss = min(max(0.0, short_put - settle_spot), wing)
    commission = 4 * commission_per_leg  # 4 legs
    pnl = (credit - call_loss - put_loss) * mult - commission
    return {"short_call": short_call, "short_put": short_put, "credit": credit,
            "call_loss": call_loss, "put_loss": put_loss, "pnl": pnl}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_canonical_ic.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the confirmation runner**

```python
# scripts/gex_ic_confirm.py
#!/usr/bin/env python3
"""Confirmation layer: price the canonical IC each day, split P&L by regime
(suppression vs amplification via dist_flip_em sign), report the spread net of fills.
Run ONLY if Rung 1 primary (realized-move) survived."""
from __future__ import annotations

import argparse, pickle, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest import gex_regime as gr
from src_v2.backtest.gex import iv_from_em
from src_v2.backtest.canonical_ic import price_ic

ENTRY = "09:45"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oi-dir", default="data/oi_cache_multi")
    ap.add_argument("--bars", default="data/spy_1m.csv")
    ap.add_argument("--cbbo", default="data/cbbo_cache_meic")
    ap.add_argument("--start", default="2022-05-11"); ap.add_argument("--end", default="2026-06-13")
    ap.add_argument("--em-mult", type=float, default=1.0); ap.add_argument("--wing", type=int, default=60)
    ap.add_argument("--slippage", type=float, default=0.10)
    a = ap.parse_args()

    loader = BarLoader(a.bars, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=date.fromisoformat(a.start), end_date=date.fromisoformat(a.end)))
    cal = [x.session_date for x in days]
    store = CbboStore(a.cbbo)
    supp, ampl = [], []
    for day in days:
        d = day.session_date
        pk = Path(a.oi_dir) / f"{d.isoformat()}.pkl"
        if not pk.exists():
            continue
        moi = pickle.load(open(pk, "rb"))
        if not moi:
            continue
        try:
            store.load_day(d)
        except Exception:
            continue
        b = next((x for x in day.bars if x["ts_et"].time().hour == 9 and x["ts_et"].time().minute == 45), None)
        if b is None:
            continue
        spot = store.derive_spot(ENTRY, float(b["open"])) or float(b["open"])
        atm = round(spot / 5) * 5
        cq = store.lookup(ENTRY, float(atm), True); pq = store.lookup(ENTRY, float(atm), False)
        if not cq or not pq:
            continue
        em = (cq[0] + cq[1]) / 2 + (pq[0] + pq[1]) / 2
        if em <= 0:
            continue
        sigma = iv_from_em(spot, em, gr.years_to_expiry(ENTRY, 0))
        if sigma <= 0:
            continue
        t_by = {e: gr.years_to_expiry(ENTRY, len([x for x in cal if d < x <= e])) for e in gr.expiries(moi)}
        feat = gr.standing_regime(moi, spot, em, sigma, t_by)
        if feat.dist_flip_em is None:
            continue
        chain = {}
        for k in range(int(spot - 400), int(spot + 401), 5):
            for ic in (True, False):
                q = store.lookup(ENTRY, float(k), ic)
                if q:
                    chain[(float(k), ic)] = q
        settle = store.settlement_forward()
        r = price_ic(chain, spot, em, a.em_mult, a.wing, 5, a.slippage, settle)
        if r is None or r["pnl"] is None:
            continue
        (supp if feat.dist_flip_em > 0 else ampl).append(r["pnl"])

    def line(name, xs):
        if not xs:
            print(f"  {name}: n=0"); return
        arr = np.array(xs)
        print(f"  {name}: n={len(arr)} mean=${arr.mean():+.1f} median=${np.median(arr):+.1f} "
              f"win={ (arr>0).mean():.1%} total=${arr.sum():+.0f}")
    print("[CANONICAL IC P&L BY REGIME — net of slippage+commission]")
    line("suppression (S>flip)", supp)
    line("amplification (S<flip)", ampl)
    if supp and ampl:
        print(f"  spread (supp-ampl) mean: ${np.mean(supp)-np.mean(ampl):+.1f}/trade")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it**

Run: `.venv/bin/python scripts/gex_ic_confirm.py`
Expected: per-regime P&L stats and the suppression−amplification spread. Capture for findings.

- [ ] **Step 7: Commit**

```bash
git add src_v2/backtest/canonical_ic.py scripts/gex_ic_confirm.py tests/test_canonical_ic.py
git commit -m "feat(gex): canonical IC confirmation layer"
```

---

## Task 14: Findings doc + scoreboard verdict

**Files:**
- Create: `docs/spx_0dte_gex_regime_findings.md`
- Modify: memory `project_strategy_scoreboard.md` (a one-line verdict; do this via the memory workflow, not a code commit)

- [ ] **Step 1: Write the findings doc**

Capture, with the actual numbers from Tasks 7/11/12/13:
- Backfill cost (Task 7 dry-run) and row counts.
- Rung 1: full-sample / per-tercile / OOS partial-effect coefficients for both `excursion` and `terminal`; PASS/FAIL vs the pre-registered bar (consistent OOS majority AND ≥2/3 terciles same-sign significant).
- Rung 1.5: proxy predictive coefficients; **OI-evolution stability** (sign-stable %, median flip drift) and the **GATE** decision (sticky → Rung 2 unnecessary; flow-driven → Rung 2 required).
- IC confirmation (if run): suppression vs amplification P&L spread net of fills.
- A bottom-line verdict sentence in the butterfly/burrito style: *strategy element + verdict + headline metric + load-bearing caveat.*

- [ ] **Step 2: Commit the findings doc**

```bash
git add docs/spx_0dte_gex_regime_findings.md
git commit -m "docs(gex): Rung 1 + 1.5 findings and Rung-2 gate decision"
```

- [ ] **Step 3: Update the strategy scoreboard memory**

Add one line to `~/.claude/projects/-home-ig-Dev-0dte-bot/memory/project_strategy_scoreboard.md` and a pointer/update in `MEMORY.md`, summarizing the verdict and the gate decision (whether Rung 2 is warranted). Follow the memory format in the repo guidance.

---

## Rung 2 (deferred)

Rung 2 (intraday 0DTE trade reconstruction from `tbbo`, internal reliability metrics, value-add) is **only planned if Task 12's gate returns "FLOW-DRIVEN → Rung 2 REQUIRED."** At that point, write a separate plan `docs/superpowers/plans/<date>-gex-rung2-trade-reconstruction.md` covering: the cherry-picked `tbbo` backfill (0DTE × ±2×EM ATM band × open→entry window), tick/quote/combo-aware signing rules, sign-perturbation + rule-agreement + stabilization reliability metrics, and the incremental-value test over Rungs 1/1.5. See spec §7.

---

## Self-review notes (completed)

- **Spec coverage:** §3 Rung 1 → Tasks 6–11, 13; §6 Rung 1.5 → Tasks 4, 12 (proxy + evolution gate); §4 variables → Tasks 2 (X), 8 (Y); §5 de-confound → Tasks 9–10; §8 guardrails (EM-scaled, OOS, no sweep, .venv) → honored throughout; §10 deliverable → Task 14; §7 Rung 2 → explicitly deferred behind the Task 12 gate.
- **Type consistency:** `MultiOi` key `(expiry, strike, is_call)` is consistent across `gex_regime.py`, `parse_osi_multi`, and `backfill_oi_multi.py`; `RegimeFeatures.dist_flip_em` is the regime variable used in Tasks 11–13; `realized_move` returns `max_excursion_em`/`terminal_abs_em` used in Tasks 11–12.
- **No placeholders:** every code step has complete, runnable code; analysis "run" steps give exact commands and the decision bar.
- **Strike window (spec §3/§5 load-bearing rule):** asymmetric-down, generously over-fetched (`--buffer-down-pts 400 / --buffer-up-pts 200`) so the put-driven flip is never truncated; Task 7 Step 4 verifies the flip is interior on ≥95% of days (the convergence/no-vol-correlated-truncation evidence). EM-scaling is achieved at the feature layer (the regime variable is `(S-flip)/EM`) plus over-fetch, rather than per-day EM-sized fetches (which would couple the backfill to the cbbo cache).
- **Known modeling choice (documented, not a gap):** a single 0DTE-implied vol is applied across all expiries' gamma kernels (each at its own T) — the regime sign/flip are dominated by the OI distribution, not the absolute vol level; vol sensitivity is implicitly covered by the boundary check.
