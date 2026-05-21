# EM-Relative MEIC Strike Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MeicStrategy's absolute credit-band + fixed-wing entry with expected-move-relative strike selection so MEIC trades sensibly across vol/index regimes.

**Architecture:** EM = ATM straddle (delta-free, from the chain). Shorts at `short_em_mult×EM` OTM, wings at `wing_em_frac×EM`, snapped to the strike step. A regime-neutral credit-to-width ratio gate replaces the absolute bands. Only entry changes (`configure`/`search_candidates`/`validate`); exit/stop/scale-out/order/monitor logic is untouched.

**Tech Stack:** Python 3, pytest, existing `src_v2` framework (`MeicStrategy`, `BacktestEngine`, `CbboStore`, `mid_price`).

**Spec:** `docs/superpowers/specs/2026-05-21-meic-em-relative-design.md`

---

## File Structure

| Path | Change |
|---|---|
| `src_v2/strategies/meic.py` | Rewrite `configure` params, `search_candidates`, `validate`; add `_compute_em`; remove `_active_band`. Exit methods untouched. |
| `tests_v2/conftest.py` | Update `meic_config` fixture to EM-relative keys. |
| `tests_v2/test_meic_strategy.py` | Replace `TestMeicConfigure` / `TestMeicSearchCandidates` / `TestMeicValidate`; add regime + gate tests. Exit tests unchanged. |
| `src_v2/backtest/engine.py` | Widen `_build_chain` half-span (configurable). |
| `tests_v2/backtest/test_meic_backtest.py` | Swap `_FakeCbbo` for a linear-premium fake; update `_config`; recompute P&L expectations. |
| `config/config_meic_spx.yaml` | Swap credit/wing keys for EM-relative keys. |
| `scripts/run_v2_wf_cv.py` | New `MEIC_PARAM_GRID` (EM-relative). |

**Conventions:** run tests from repo root with explicit paths, e.g. `python -m pytest tests_v2/test_meic_strategy.py -v`. `math` and `_mid` (`mid_price`) are already imported in `meic.py`. Two unrelated pre-existing iron_fly failures in `tests_v2/` are NOT your concern.

---

## Task 1: `_compute_em` helper

**Files:**
- Modify: `src_v2/strategies/meic.py`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Write the failing test**

Append to `tests_v2/test_meic_strategy.py`:

```python
class TestMeicComputeEm:
    def _chain(self, spot, em, step=5):
        # ATM straddle (call_mid + put_mid at the strike nearest spot) == em.
        atm = round(spot / step) * step
        half = em / 2.0
        bid, ask = round(half - 0.05, 2), round(half + 0.05, 2)
        return [
            {"strike": float(atm), "option_type": "call", "bid": bid, "ask": ask},
            {"strike": float(atm), "option_type": "put", "bid": bid, "ask": ask},
        ]

    def test_em_is_atm_straddle(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s._compute_em(self._chain(6700, 120.0), 6700.0) == pytest.approx(120.0)

    def test_em_uses_nearest_strike_to_spot(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        # spot 6702 -> nearest listed strike 6700; straddle still 120
        assert s._compute_em(self._chain(6700, 120.0), 6702.0) == pytest.approx(120.0)

    def test_em_none_when_atm_legs_missing(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        calls_only = [{"strike": 6700.0, "option_type": "call", "bid": 59.95, "ask": 60.05}]
        assert s._compute_em(calls_only, 6700.0) is None

    def test_em_none_for_empty_chain(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s._compute_em([], 6700.0) is None
```

NOTE: this task adds `self._strike_step` to `configure` (used by `_compute_em`). The `meic_config` fixture is updated in Task 2; for Task 1 the default `strat.get("strike_step_pts", 5)` applies, so these tests pass against the current fixture too.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicComputeEm -v`
Expected: FAIL — `AttributeError: 'MeicStrategy' object has no attribute '_compute_em'` (and `_strike_step`).

- [ ] **Step 3: Add `_strike_step` to configure and the `_compute_em` method**

In `src_v2/strategies/meic.py` `configure`, add this line (anywhere in the body, e.g. after the `_am_pm_cutoff` line):

```python
        self._strike_step = strat.get("strike_step_pts", 5)
```

Add this method to `MeicStrategy` (place it just above `search_candidates`):

```python
    def _compute_em(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Optional[float]:
        """Expected move = ATM straddle (call_mid + put_mid at nearest strike)."""
        if not chain:
            return None
        step = self._strike_step
        by = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }
        strikes = [float(o.get("strike", 0)) for o in chain]
        atm_k = round(spot / step) * step
        cp = by.get((float(atm_k), "call"))
        pp = by.get((float(atm_k), "put"))
        if not cp or not pp:
            atm_k = min(set(strikes), key=lambda k: abs(k - spot))
            cp = by.get((float(atm_k), "call"))
            pp = by.get((float(atm_k), "put"))
        if not cp or not pp:
            return None
        cm = _mid(cp)
        pm = _mid(pp)
        if cm is None or pm is None:
            return None
        return round(cm + pm, 4)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicComputeEm -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): add _compute_em (ATM straddle) helper"
```

---

## Task 2: EM-relative entry (configure / search_candidates / validate)

This is the core change. `configure`, `search_candidates`, `validate`, the `meic_config` fixture, and the three entry test classes all change together so the suite stays green at the task boundary.

**Files:**
- Modify: `src_v2/strategies/meic.py`
- Modify: `tests_v2/conftest.py`
- Test: `tests_v2/test_meic_strategy.py`

- [ ] **Step 1: Update the `meic_config` fixture**

In `tests_v2/conftest.py`, in `meic_config`, replace the credit-band + `wing_widths` keys with EM-relative keys. The `strategy` dict becomes:

```python
        "strategy": {
            "am_pm_cutoff": "12:30",
            "short_em_mult": 1.0,
            "wing_em_frac": 0.5,
            "min_credit_width_ratio": 0.08,
            "strike_step_pts": 5,
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
```

(Leave `bot` and `entry` keys in the fixture unchanged.)

- [ ] **Step 2: Replace the entry test classes (write failing tests)**

In `tests_v2/test_meic_strategy.py`, REPLACE the three classes `TestMeicConfigure`, `TestMeicSearchCandidates`, and `TestMeicValidate` (keep the module-level `_at` helper — do not remove it) with the following. Keep `TestMeicSimpleMethods`, `TestMeicComputeEm`, `TestMeicBuildOrder`, `TestMeicStops`, `TestMeicScaleOut`, `TestMeicStopAndAbandon`, and `TestExitActionWidening` as-is.

FIRST ensure the test file's imports include `LiveTradeSetup` and `OrderLeg` (the new `TestMeicValidate._setup` constructs them at module level). At the top of the file, make sure this import exists (add/extend it):

```python
from src_v2.strategies.base import LiveStrategy, LiveTradeSetup, OrderLeg, ExitAction
```

```python
def _em_chain(spot, em, step=5, span=400, rate=0.10, floor=0.05):
    """Synthetic chain: ATM straddle == em; premium decays linearly at `rate`
    per point from a peak of em/2, floored at `floor`. spot must be a step
    multiple so the ATM straddle equals em exactly. Far strikes hit the floor
    (bid 0 -> treated as bidless by mid_price)."""
    peak = em / 2.0
    atm = round(spot / step) * step
    chain = []
    k = atm - span
    while k <= atm + span:
        dist = abs(k - spot)
        mid = max(floor, peak - rate * dist)
        bid = round(mid - 0.05, 2)
        ask = round(mid + 0.05, 2)
        for otype in ("put", "call"):
            chain.append({"strike": float(k), "option_type": otype,
                          "bid": bid, "ask": ask})
        k += step
    return chain


class TestMeicConfigure:
    def test_is_live_strategy(self):
        assert issubclass(MeicStrategy, LiveStrategy)

    def test_configure_reads_em_relative_keys(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        assert s._short_em_mult == 1.0
        assert s._wing_em_frac == 0.5
        assert s._min_credit_width_ratio == 0.08
        assert s._strike_step == 5
        assert s._stop_pct == 0.70
        assert s._am_pm_cutoff.hour == 12 and s._am_pm_cutoff.minute == 30
        assert s._tranche_sizing == {"11:00": 3, "11:40": 4, "12:00": 5, "13:00": 6}

    def test_configure_rejects_stop_pct_too_low(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.50
        with pytest.raises(ValueError):
            MeicStrategy().configure(meic_config)

    def test_configure_rejects_stop_pct_too_high(self, meic_config):
        meic_config["strategy"]["stop_pct_of_total_credit"] = 0.95
        with pytest.raises(ValueError):
            MeicStrategy().configure(meic_config)


class TestMeicSearchCandidates:
    @pytest.mark.asyncio
    async def test_empty_chain_returns_zero(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates([], spot=6700.0)
        assert result["candidates_found"] == 0
        assert result["top_candidate"] is None

    @pytest.mark.asyncio
    async def test_high_vol_regime_candidate(self, meic_config):
        # spot 6700, EM 120 -> short_call 6820, short_put 6580, wing 60, credit 6.0
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates(_em_chain(6700, 120.0), spot=6700.0)
        assert result["candidates_found"] == 1
        top = result["top_candidate"]
        assert top.strategy_type == "meic"
        assert top.strikes["short_call"] == 6820.0
        assert top.strikes["short_put"] == 6580.0
        assert top.extra["wing_pts"] == 60
        assert top.extra["em_pts"] == pytest.approx(120.0)
        assert top.extra["call_credit"] == pytest.approx(6.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_low_vol_regime_candidate_scales_down(self, meic_config):
        # SAME config; spot 4285, EM 24 -> short_call 4310, short_put 4260, wing 10
        # This is the regime that the absolute-band version produced 0 trades for.
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates(_em_chain(4285, 24.0), spot=4285.0)
        assert result["candidates_found"] == 1
        top = result["top_candidate"]
        assert top.strikes["short_call"] == 4310.0
        assert top.strikes["short_put"] == 4260.0
        assert top.extra["wing_pts"] == 10
        assert top.extra["call_credit"] == pytest.approx(1.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_stamps_tranche_qty(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:40")  # qty 4
        top = (await s.search_candidates(_em_chain(6700, 120.0), 6700.0))["top_candidate"]
        assert top.extra["qty"] == 4
        assert all(leg.quantity == 4 for leg in top.legs)

    @pytest.mark.asyncio
    async def test_credit_to_width_gate_rejects_thin_credit(self, meic_config):
        # rate 0.05 -> credit/width = 0.05 < 0.08 gate -> no candidate
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        result = await s.search_candidates(_em_chain(6700, 120.0, rate=0.05), 6700.0)
        assert result["candidates_found"] == 0

    @pytest.mark.asyncio
    async def test_skips_when_wing_leg_bidless(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config); _at(s, "11:00")
        chain = [o for o in _em_chain(6700, 120.0)
                 if not (o["strike"] == 6880.0 and o["option_type"] == "call")]
        result = await s.search_candidates(chain, 6700.0)
        assert result["candidates_found"] == 0


class TestMeicValidate:
    def _setup(self, sc=6820.0, lc=6880.0, sp=6580.0, lp=6520.0,
               ccw=0.10, pcw=0.10, qty=3):
        return LiveTradeSetup(
            strategy_type="meic",
            legs=[OrderLeg("", sc, "call", "sell", qty)],
            total_credit=12.0,
            strikes={"short_call": sc, "long_call": lc,
                     "short_put": sp, "long_put": lp},
            extra={"call_credit": 6.0, "put_credit": 6.0, "qty": qty,
                   "call_credit_width": ccw, "put_credit_width": pcw,
                   "wing_pts": int(lc - sc)},
        )

    @pytest.mark.asyncio
    async def test_valid_setup_passes(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(), [])
        assert res["passed"] is True and res["failures"] == []

    @pytest.mark.asyncio
    async def test_bad_call_strike_order_fails(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(lc=6810.0), [])  # long_call < short_call
        assert "strike_order_call" in res["failures"]

    @pytest.mark.asyncio
    async def test_asymmetric_wings_fail(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        # call wing 60 (6820->6880), put wing 40 (6580->6540): diff 20 > step 5
        res = await s.validate(self._setup(lp=6540.0), [])
        assert "wing_symmetry" in res["failures"]

    @pytest.mark.asyncio
    async def test_credit_to_width_below_gate_fails(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(ccw=0.05), [])  # below 0.08
        assert "credit_to_width_call" in res["failures"]

    @pytest.mark.asyncio
    async def test_zero_qty_fails(self, meic_config):
        s = MeicStrategy(); s.configure(meic_config)
        res = await s.validate(self._setup(qty=0), [])
        assert "qty" in res["failures"]
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicConfigure tests_v2/test_meic_strategy.py::TestMeicSearchCandidates tests_v2/test_meic_strategy.py::TestMeicValidate -v`
Expected: FAIL — `configure` still sets old attrs; `search_candidates` still uses credit bands; assertions about `short_call==6820` etc. fail.

- [ ] **Step 4: Rewrite `configure`**

In `src_v2/strategies/meic.py`, replace the entire `configure` method body with:

```python
    def configure(self, config: Dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self._am_pm_cutoff = _parse_hhmm(strat.get("am_pm_cutoff", "12:30"))
        self._short_em_mult = strat.get("short_em_mult", 1.0)
        self._wing_em_frac = strat.get("wing_em_frac", 0.5)
        self._min_credit_width_ratio = strat.get("min_credit_width_ratio", 0.08)
        self._strike_step = strat.get("strike_step_pts", 5)
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
```

- [ ] **Step 5: Remove `_active_band` and rewrite `search_candidates`**

In `src_v2/strategies/meic.py`, DELETE the `_active_band` method entirely. Keep `_tranche_qty`. Replace the entire `search_candidates` method with:

```python
    async def search_candidates(
        self, chain: List[Dict[str, Any]], spot: float,
    ) -> Dict[str, Any]:
        empty = {"candidates_found": 0, "top_candidate": None, "candidates": []}
        if not chain:
            return empty
        em = self._compute_em(chain, spot)
        if em is None or em <= 0:
            return empty

        step = self._strike_step
        qty = self._tranche_qty()
        by = {
            (float(o.get("strike", 0)), o.get("option_type", "").lower()): o
            for o in chain
        }

        short_call = math.ceil((spot + self._short_em_mult * em) / step) * step
        short_put = math.floor((spot - self._short_em_mult * em) / step) * step
        wing_pts = max(step, int(round(self._wing_em_frac * em / step)) * step)
        long_call = short_call + wing_pts
        long_put = short_put - wing_pts

        sc_mid = _mid(by.get((float(short_call), "call")))
        lc_mid = _mid(by.get((float(long_call), "call")))
        sp_mid = _mid(by.get((float(short_put), "put")))
        lp_mid = _mid(by.get((float(long_put), "put")))
        if sc_mid is None or lc_mid is None or sp_mid is None or lp_mid is None:
            return empty

        call_credit = round(sc_mid - lc_mid, 4)
        put_credit = round(sp_mid - lp_mid, 4)
        if call_credit <= 0 or put_credit <= 0:
            return empty

        call_cw = call_credit / wing_pts
        put_cw = put_credit / wing_pts
        if call_cw < self._min_credit_width_ratio or put_cw < self._min_credit_width_ratio:
            return empty

        total_credit = round(call_credit + put_credit, 4)
        setup = LiveTradeSetup(
            strategy_type="meic",
            legs=[
                OrderLeg("", float(short_call), "call", "sell", qty),
                OrderLeg("", float(long_call), "call", "buy", qty),
                OrderLeg("", float(short_put), "put", "sell", qty),
                OrderLeg("", float(long_put), "put", "buy", qty),
            ],
            total_credit=total_credit,
            strikes={
                "short_call": float(short_call),
                "long_call": float(long_call),
                "short_put": float(short_put),
                "long_put": float(long_put),
            },
            extra={
                "em_pts": round(em, 2),
                "wing_pts": wing_pts,
                "call_credit": call_credit,
                "put_credit": put_credit,
                "credit_diff": round(abs(call_credit - put_credit), 4),
                "total_premium": total_credit,
                "short_call_mid": sc_mid,
                "short_put_mid": sp_mid,
                "call_credit_width": round(call_cw, 4),
                "put_credit_width": round(put_cw, 4),
                "qty": qty,
            },
        )
        return {"candidates_found": 1, "top_candidate": setup, "candidates": [setup]}
```

- [ ] **Step 6: Rewrite `validate`**

In `src_v2/strategies/meic.py`, replace the entire `validate` method with:

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
        record("wing_symmetry", abs((lc - sc) - (sp - lp)) <= self._strike_step)

        ccw = setup.extra.get("call_credit_width")
        pcw = setup.extra.get("put_credit_width")
        record("credit_to_width_call",
               ccw is not None and ccw >= self._min_credit_width_ratio)
        record("credit_to_width_put",
               pcw is not None and pcw >= self._min_credit_width_ratio)
        record("qty", setup.extra.get("qty", 0) >= 1)

        return {"passed": len(failures) == 0, "checks": checks, "failures": failures}
```

- [ ] **Step 7: Run the entry tests**

Run: `python -m pytest tests_v2/test_meic_strategy.py::TestMeicConfigure tests_v2/test_meic_strategy.py::TestMeicSearchCandidates tests_v2/test_meic_strategy.py::TestMeicValidate -v`
Expected: PASS (configure 4, search 6, validate 5).

- [ ] **Step 8: Run the whole MEIC strategy test file (exit tests must still pass)**

Run: `python -m pytest tests_v2/test_meic_strategy.py -v`
Expected: PASS. `TestMeicBuildOrder`, `TestMeicStops`, `TestMeicScaleOut`, `TestMeicStopAndAbandon` are unaffected (they construct setups/trades directly and don't depend on the entry rewrite).

- [ ] **Step 9: Commit**

```bash
git add src_v2/strategies/meic.py tests_v2/conftest.py tests_v2/test_meic_strategy.py
git commit -m "feat(meic): EM-relative entry — shorts at k*EM, wings at f*EM, credit/width gate"
```

---

## Task 3: Widen backtest chain + rework backtest tests

**Files:**
- Modify: `src_v2/backtest/engine.py`
- Test: `tests_v2/backtest/test_meic_backtest.py`

- [ ] **Step 1: Widen `_build_chain` (write failing test for coverage)**

The current `_build_chain` spans `spot ± 200pt`. EM-relative strikes in high vol can exceed that. Add a test to `tests_v2/backtest/test_meic_backtest.py` (at the end of the file):

```python
def test_build_chain_covers_wide_span():
    import yaml
    cfg = yaml.safe_load(open("config/config_meic_spx.yaml"))
    # fake cbbo that prices any strike within +-400 of spot
    import src_v2.backtest.engine as eng
    eng_obj = None

    class _AllStrikes:
        def load_day(self, d): return 1
        def derive_spot(self, m, ref): return 6700.0
        def lookup(self, m, k, is_call): return (1.0, 1.1)

    e = BacktestEngine(cfg, MeicStrategy())
    e._cbbo = _AllStrikes()
    chain = e._build_chain("11:00", 6700.0, date(2026, 1, 5))
    strikes = [o["strike"] for o in chain]
    # must reach at least +-350pt to cover high-vol EM strikes + wings
    assert max(strikes) >= 7050
    assert min(strikes) <= 6350
```

Run: `python -m pytest tests_v2/backtest/test_meic_backtest.py::test_build_chain_covers_wide_span -v`
Expected: FAIL — current span is ±200 (max 6900, min 6500).

- [ ] **Step 2: Widen the span**

In `src_v2/backtest/engine.py` `_build_chain`, change the span. Find:

```python
        low_k = int((spot - 200) / step) * step
        high_k = int((spot + 200) / step) * step
```

Replace with (configurable half-span, default 400; covers EM-relative reach in high vol):

```python
        half = getattr(self, "_chain_half_span_pts", 400)
        low_k = int((spot - half) / step) * step
        high_k = int((spot + half) / step) * step
```

And in `BacktestEngine.__init__`, add (next to the other `opt_cfg` reads):

```python
        self._chain_half_span_pts = opt_cfg.get("chain_half_span_pts", 400)
```

Run: `python -m pytest tests_v2/backtest/test_meic_backtest.py::test_build_chain_covers_wide_span -v`
Expected: PASS.

- [ ] **Step 3: Replace `_FakeCbbo` + `_config` + `_meic_marks` with a linear-premium fake**

In `tests_v2/backtest/test_meic_backtest.py`, REPLACE the `_FakeCbbo` class, the `_meic_marks` function, and the `_config`/`_config_qty4` functions with the following. Keep `_Day`, `_bars_flat`, `_make_engine`, and `ET` imports. The strategy now derives strikes from EM, so tests compute expected strikes from (spot, em).

```python
class _LinearCbbo:
    """Synthetic CBBO: ATM straddle == em; premium decays linearly at `rate`/pt
    from peak em/2, floored at `floor`. Optional time-shifted per-(strike,is_call)
    overrides from `shift_minute` onward (to drive stops / scale-outs)."""
    def __init__(self, spot, em, rate=0.10, floor=0.05, step=5, span=400,
                 shift_minute=None, shift_overrides=None):
        self._spot = spot
        self._peak = em / 2.0
        self._rate = rate
        self._floor = floor
        self._span = span
        self._shift_minute = shift_minute
        self._shift = shift_overrides or {}

    def load_day(self, d):
        return 1

    def derive_spot(self, minute, ref):
        return self._spot

    def lookup(self, minute, strike, is_call):
        key = (float(strike), is_call)
        if self._shift_minute and minute >= self._shift_minute and key in self._shift:
            return self._shift[key]
        dist = abs(strike - self._spot)
        if dist > self._span:
            return None
        mid = max(self._floor, self._peak - self._rate * dist)
        return (round(mid - 0.05, 2), round(mid + 0.05, 2))


def _config(tranche=None):
    return {
        "bot": {"strategy": "meic"},
        "entry": {"schedule": ["11:00"]},
        "strategy": {
            "am_pm_cutoff": "12:30",
            "short_em_mult": 1.0,
            "wing_em_frac": 0.5,
            "min_credit_width_ratio": 0.08,
            "strike_step_pts": 5,
            "stop_pct_of_total_credit": 0.70,
            "am_scale_out_tiers": [0.55, 0.65, 0.75, 0.85],
            "am_scale_out_fractions": [0.25, 0.25, 0.25, 0.25],
            "pm_scale_out_tiers": [0.65, 0.85],
            "pm_scale_out_fractions": [0.50, 0.50],
            "abandon_long_below": 0.05,
            "credit_equality_threshold": 0.30,
            "default_tranche_qty": (tranche or 3),
            "tranche_sizing": {"11:00": (tranche or 3)},
            "eod_force_close": "15:50",
            "market_open": "09:30",
        },
        "options": {
            "contract_multiplier": 100, "quantity": 1,
            "strike_step_pts": 5, "option_root": "SPXW",
            "tp_pct_of_credit": 0.50,
            "slippage_per_spread": 0.0,
            "commission_per_contract_per_leg": 0.0,
            "stop_slippage_per_spread": 0.0,
            "chain_half_span_pts": 400,
        },
        "backtest": {"cbbo_cache_dir": ""},
    }
```

Reference geometry for spot 6700, em 120, rate 0.10 (used by the tests below):
short_call 6820, long_call 6880, short_put 6580, long_put 6520, wing 60;
short mid = 60 − 0.10×120 = 48; long mid = 60 − 0.10×180 = 42; **each side credit = 6.0**;
total credit 12.0; stop_amount = 0.70×12 = 8.4; short stop price = 48 + 8.4 = 56.4.

- [ ] **Step 4: Rewrite the backtest P&L tests**

In `tests_v2/backtest/test_meic_backtest.py`, REPLACE the existing tests
`test_meic_backtest_worthless_expiry`, `test_meic_backtest_stop_loss`,
`test_meic_backtest_scale_out_full_exit`, `test_meic_backtest_partial_then_expiry`,
`test_meic_backtest_applies_costs_on_worthless_expiry`, and
`test_meic_backtest_applies_stop_slippage` with:

```python
def test_meic_backtest_worthless_expiry(monkeypatch):
    d = date(2026, 1, 5)
    engine = _make_engine(monkeypatch, _LinearCbbo(6700, 120.0))
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2  # one per side
    for r in closed:
        # qty 3, side credit 6.0 -> 6.0 * 100 * 3 = 1800
        assert r.realized_pnl_usd == pytest.approx(6.0 * 100 * 3, abs=1.0)
        assert r.exit_reason == "expired_worthless"
        assert r.qty == 3


def test_meic_backtest_stop_loss(monkeypatch):
    d = date(2026, 1, 5)
    # From 11:30 the call short ask jumps to 57 (>= stop 56.4) -> stop fills.
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (56.5, 57.0)})
    engine = _make_engine(monkeypatch, fake)
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    # debit = min(stop_amount 8.4, width 60) = 8.4 -> (6.0-8.4)*100*3 = -720
    assert call.exit_reason == "short_stopped"
    assert call.realized_pnl_usd == pytest.approx((6.0 - 8.4) * 100 * 3, abs=1.0)
    put = next(r for r in results if r.spread_type == "put")
    assert put.exit_reason == "expired_worthless"


def test_meic_backtest_scale_out_full_exit(monkeypatch):
    d = date(2026, 1, 5)
    # qty 4; from 11:30 call short ask collapses to 0.90 -> pnl 85% -> all 4 AM
    # tiers fire -> 4 contracts closed -> full exit. long call (6880) bid ~41.95
    # (>= abandon 0.05) so it is also bought back at its mid.
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (0.85, 0.90)})
    engine = _make_engine(monkeypatch, fake, config=_config(tranche=4))
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "scaled_out"
    assert call.qty == 4
    # short bought back at ask 0.90 each, 4 contracts: (6.0-0.90)*100*4 = 2040
    assert call.realized_pnl_usd == pytest.approx((6.0 - 0.90) * 100 * 4, abs=1.0)


def test_meic_backtest_partial_then_expiry(monkeypatch):
    d = date(2026, 1, 5)
    # qty 4; from 11:30 call short ask -> 2.70 -> pnl 55% -> only AM tier 0 fires
    # (floor(4*0.25)=1 contract); remaining 3 ride to EOD worthless.
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (2.65, 2.70)})
    engine = _make_engine(monkeypatch, fake, config=_config(tranche=4))
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "partial_then_expiry"
    # tier0: (6.0-2.70)*100*1 = 330 ; remaining 3 worthless: 6.0*100*3 = 1800 -> 2130
    assert call.realized_pnl_usd == pytest.approx(330 + 1800, abs=1.0)


def _config_costs(tranche=3):
    cfg = _config(tranche=tranche)
    cfg["options"]["slippage_per_spread"] = 0.10
    cfg["options"]["commission_per_contract_per_leg"] = 1.0
    cfg["options"]["stop_slippage_per_spread"] = 0.25
    return cfg


def test_meic_backtest_applies_costs_on_worthless_expiry(monkeypatch):
    d = date(2026, 1, 5)
    engine = _make_engine(monkeypatch, _LinearCbbo(6700, 120.0), config=_config_costs())
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2
    for r in closed:
        # credit_eff = 6.0-0.10 = 5.90 ; commission 2*3*1=6 ; 5.90*100*3 - 6 = 1764
        assert r.realized_pnl_usd == pytest.approx(5.90 * 100 * 3 - 6.0, abs=0.5)


def test_meic_backtest_applies_stop_slippage(monkeypatch):
    d = date(2026, 1, 5)
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (56.5, 57.0)})
    engine = _make_engine(monkeypatch, fake, config=_config_costs())
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    # debit = min(8.4+0.25, 60) = 8.65 ; credit_eff 5.90 ; comm 6
    # (5.90-8.65)*100*3 - 6 = -831
    assert call.exit_reason == "short_stopped"
    assert call.realized_pnl_usd == pytest.approx((5.90 - 8.65) * 100 * 3 - 6.0, abs=0.5)
```

NOTE: `_bars_flat` is called with `price=6700.0`. Confirm the existing `_bars_flat(d, start="11:00", end="15:50", price=5850.0)` signature accepts `price=`; it does. Also confirm `_make_engine(monkeypatch, fake, config=None)` patches `eng.CbboStore` and returns `BacktestEngine(config or _config(), MeicStrategy())` — it does (unchanged).

- [ ] **Step 5: Run the backtest tests**

Run: `python -m pytest tests_v2/backtest/test_meic_backtest.py -v`
Expected: PASS (worthless, stop, scale-out, partial-then-expiry, 2 cost tests, chain-span test = 7).

- [ ] **Step 6: Commit**

```bash
git add src_v2/backtest/engine.py tests_v2/backtest/test_meic_backtest.py
git commit -m "feat(meic): widen backtest chain span; EM-relative backtest tests"
```

---

## Task 4: Config + WF-CV grid

**Files:**
- Modify: `config/config_meic_spx.yaml`
- Modify: `scripts/run_v2_wf_cv.py`

- [ ] **Step 1: Update the strategy section of `config/config_meic_spx.yaml`**

Replace the credit-band + `wing_widths` keys in the `strategy:` block with EM-relative keys. The `strategy:` block becomes:

```yaml
strategy:
  am_pm_cutoff: "12:30"
  short_em_mult: 1.0
  wing_em_frac: 0.5
  min_credit_width_ratio: 0.08
  strike_step_pts: 5
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
```

Also add `chain_half_span_pts: 400` to the `options:` block of that config (so the backtest covers high-vol strikes):

```yaml
  chain_half_span_pts: 400
```

- [ ] **Step 2: Update `MEIC_PARAM_GRID` in `scripts/run_v2_wf_cv.py`**

Replace the `MEIC_PARAM_GRID` definition with:

```python
# MEIC grid — EM-relative knobs. Combos = 3*3*3 = 27
MEIC_PARAM_GRID = {
    "strategy.short_em_mult":            [0.8, 1.0, 1.2],
    "strategy.wing_em_frac":             [0.4, 0.5, 0.6],
    "strategy.stop_pct_of_total_credit": [0.65, 0.70, 0.75],
}
```

- [ ] **Step 3: Verify config loads + routes**

Run: `python -m pytest tests_v2/test_meic_wiring.py -v`
Expected: PASS — `test_main_routes_meic_strategy` reads the config and builds a `MeicStrategy`. NOTE: that test asserts `strategy._wing_widths == [80,90,100]` and `strategy._stop_pct == 0.70`. The `_wing_widths` assertion is now stale — UPDATE that test: replace `assert strategy._wing_widths == [80, 90, 100]` with `assert strategy._wing_em_frac == 0.5` and keep `assert strategy._stop_pct == 0.70`. (Add this edit to `tests_v2/test_meic_wiring.py`, then re-run.)

- [ ] **Step 4: Commit**

```bash
git add config/config_meic_spx.yaml scripts/run_v2_wf_cv.py tests_v2/test_meic_wiring.py
git commit -m "feat(meic): EM-relative config + WF-CV grid"
```

---

## Task 5: Full-suite check + 2023 regime validation gate

**Files:** none (verification only, unless a fix is needed)

- [ ] **Step 1: Run all MEIC + backtest + wiring tests**

Run: `python -m pytest tests_v2/test_meic_strategy.py tests_v2/test_meic_backtest.py tests_v2/test_meic_wiring.py tests_v2/test_meic_position_monitor.py tests_v2/backtest/test_meic_backtest.py tests/test_meic_backfill.py -v`
Expected: all pass.

- [ ] **Step 2: Run the broader suite for regressions**

Run: `python -m pytest tests_v2/ -q`
Expected: only the 2 known pre-existing iron_fly failures.

- [ ] **Step 3: The regime validation gate — re-run the 2023 sanity week**

The 2023 CBBO sanity data is already fetched at `data/cbbo_cache_meic` (2023-06-01..09). Confirm MEIC now PRODUCES trades there (the absolute version produced 0):

```bash
python -c "
from datetime import date
from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest.engine import BacktestEngine
from src_v2.strategies.meic import MeicStrategy
from src_v2.backtest.report import generate_report
import yaml
cb = CbboStore('data/cbbo_cache_meic', cache_dir='data/cbbo_cache_meic')
cfg = yaml.safe_load(open('config/config_meic_spx.yaml'))
days = list(BarLoader('data/spy_1m.csv', price_scale=10.0, symbol='SPX').iter_days(date(2023,6,1), date(2023,6,9)))
eng = BacktestEngine(cfg, MeicStrategy()); eng._cbbo = cb
rep = generate_report(eng.run(days))
print('2023 sanity:', rep['trades_placed'], 'trades, PnL', rep['total_pnl_usd'])
assert rep['trades_placed'] > 0, 'REGIME GATE FAILED: still 0 trades in 2023'
print('REGIME GATE PASSED')
"
```
Expected: `trades_placed > 0` and `REGIME GATE PASSED`. If it prints 0 trades, the EM-relative selection or `min_credit_width_ratio` is still too strict for low vol — investigate `_compute_em`/strike snapping/gate on a 2023 day before declaring done (do NOT lower the gate blindly; first confirm EM is computed and the legs are priced).

- [ ] **Step 4: Commit (only if a fix was needed in Step 3)**

If Step 3 required a code fix, commit it:
```bash
git add -A
git commit -m "fix(meic): EM-relative tuning for low-vol regime"
```

---

## Post-plan (NOT part of this plan — separate follow-ups)
1. Run the full ~$7 CBBO backfill (`scripts/backfill_meic_cbbo.py --start 2022-05-11 --end 2025-09-19 --out data/cbbo_cache_meic --max-cost-usd 15`).
2. Extended backtest + WF-CV through the 2022 bear, with the daily-equity drawdown report.

## Self-review notes (for the implementer)
- **Spec coverage:** EM=straddle (Task 1), EM-relative shorts+wings+gate (Task 2), chain widen (Task 3), config+grid (Task 4), regime gate (Task 5). All spec sections covered.
- **Type/name consistency:** `_compute_em`, `_short_em_mult`, `_wing_em_frac`, `_min_credit_width_ratio`, `_strike_step`, and `extra` keys (`em_pts`, `wing_pts`, `call_credit_width`, `put_credit_width`, `short_call_mid`, `short_put_mid`, `call_credit`, `put_credit`) are defined in Task 1/2 and consumed consistently in validate, the backtest, and `compute_stop_loss` (which reads `call_credit`/`put_credit`/`short_*_mid` — preserved).
- **Unchanged contracts:** `build_order`, `compute_stop_loss`, `build_stop_orders`, `check_exit`, `leg_pairs`, `create_engine`, `_tranche_qty`, `_now_fn` clock injection — all untouched.
