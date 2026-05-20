# SPX 0DTE Open Strangle — Risk-Equivalent Comparison vs. Open IC

*Design spec — 2026-05-19*

## 1. Goal & scope

Run a single-phase WF-CV native optimization of an SPX 0DTE **open strangle** (short put + short call, no protective wings) and compare head-to-head with the existing winning **open iron condor** ([docs/spx_0dte_open_ic_findings.md](../../spx_0dte_open_ic_findings.md)) over the same 4-year CBBO window (2022-04-18 → 2026-02-28).

The sweep is free to choose delta / em_mult / TP, but candidate configs are **risk-normalized** against the IC: each strangle trade's `sl_dollars` is set to that day's IC effective $-SL (capped at the 75th percentile of the IC SL distribution). Without this constraint a strangle "wins" trivially by accepting fat tails — the gate enforces apples-to-apples downside.

**Acceptance criteria** (strangle must beat IC on all three to be reported as superior):

| Metric | IC baseline | Strangle must achieve |
|---|---|---|
| OOS Sharpe (annualized) | 4.53 | ≥ 4.53 |
| Worst day | −$826 | ≥ −$826 |
| Total OOS PnL | +$15,014 | > +$15,014 |

## 2. Architecture

### New module — `src_v2/strategies/open_strangle.py`

`OpenStrangleStrategy`, forked from [src_v2/strategies/open_ic.py](../../../src_v2/strategies/open_ic.py). Differences:

- `setup_trade()` returns 2 `OrderLeg`s (short put + short call) instead of 4
- `total_credit = put_mid + call_mid` (gross premium, no debit subtraction)
- New config primitive `sl_dollars: float` replaces `sl_mult_of_credit`. Set per-trade by the runner (see §3), not from static config
- Per-side TP at `tp_pct_of_credit × side_credit` (mirrors IC's per-side TP semantics)
- Total-mark SL exit when `(put_mark + call_mark) − total_credit ≥ sl_dollars`
- No breach handling (no defined wings to breach)
- `eod_force_close: "15:30"` — close before assignment-risk window

### Generalized WF-CV runner — `scripts/run_v2_wf_cv_open_ic.py`

Add `strategy.class` config key (dotted import path, default `src_v2.strategies.open_ic.OpenIcStrategy`). Resolve to class at runtime via `importlib.import_module`. Behavior unchanged for existing IC configs. New runner script `scripts/run_v2_wf_cv_open_strangle.py` is a thin entry point that defaults the config to the strangle yaml — most logic lives in the generalized runner.

### New config — `config/config_spx_0dte_open_strangle.yaml`

Starts from [config/config_spx_0dte_open_ic.yaml](../../../config/config_spx_0dte_open_ic.yaml). Inherits all gates (ORB filter 0.5, NFP/FOMC/CPI/FOMC+1 skip, Fri skip) and stability checks. Diffs:

- `strategy.class: src_v2.strategies.open_strangle.OpenStrangleStrategy`
- `strategy.eod_force_close: "15:30"`
- `options.wing_width` removed
- `options.sl_mult_of_credit` removed
- `options.sl_dollars` — placeholder, overridden per-trade by runner

### Analysis scripts

- `scripts/analyze_open_strangle_trades.py` — fork of [scripts/analyze_open_ic_trades.py](../../../scripts/analyze_open_ic_trades.py), adjusted for 2-leg strangle
- `scripts/compare_strangle_vs_ic.py` — side-by-side WF-CV winners with metric delta table and per-fold breakdown

### Reused unchanged

- [src_v2/agent/window_entry_engine.py](../../../src_v2/agent/window_entry_engine.py) — entry scheduler, gates, stability checks
- [src_v2/backtest/wf_optimizer.py](../../../src_v2/backtest/wf_optimizer.py) — WF-CV mechanics
- CBBO data loader, decision logger, position sim

## 3. SL calibration (per-trade-matched)

**Step 1 — extract IC's per-trade $-SL.** Load the winning IC config's trade log (or re-run its WF-CV once). For each trade-day `d`, compute:

```
IC_SL$[d] = sl_mult_of_credit (2.0) × IC_total_credit[d] × 100
```

Expected range ~$200–$500/day; persisted to `data/ic_sl_dollars_by_day.parquet` (or csv) keyed by trade date.

**Step 2 — compute 75th-percentile ceiling.**

```
IC_SL$_p75 = numpy.percentile(IC_SL$.values(), 75)
```

**Step 3 — at strangle backtest entry-time on day `d`:**

```
strangle_sl_dollars[d] = min(IC_SL$[d], IC_SL$_p75)
```

If `d` is not in the IC trade-day set (gates differ — should not happen since gates are inherited, but defensive), fall back to `IC_SL$_p75`.

**Step 4 — reporting.** Record per-trade effective SL alongside trade. Findings doc reports realized worst-day and loss-stdev for each strangle config next to the IC's so we catch nominal vs. realized SL divergence (gap-throughs, EOD pin).

### SL-fill convention

When `total_mark − total_credit ≥ sl_dollars` triggers intrabar, exit at the **next bar's CBBO mark** — matches IC's existing convention for fairness, even though both are mildly optimistic.

## 4. WF-CV sweep design

**Methodology:** ratio 5:1, 4 folds, window 2022-04-18 → 2026-02-28, objective `sharpe_annualised`, `min-trades = 10` per fold, `require-all-folds-profitable = false` (matches IC v5).

**Grid:**

| Param | Values |
|---|---|
| `strike_mode` | `delta`, `em_mult` |
| `target_short_delta` *(when delta mode)* | 0.10, 0.15, 0.20, 0.25, 0.30 |
| `em_mult` *(when em_mult mode, symmetric)* | 0.20, 0.30, 0.40, 0.50, 0.75, 1.00 |
| `tp_pct_of_credit` | 0.25, 0.50, 0.75 |
| `tp_pct_of_credit_end` | `null`, 0.75 |

**Sweep size:** 11 strike configs × 3 TP × 2 TP-end = **66 configs**. Tractable for the existing parallelized WF runner.

**Inherited & held constant (not swept):**

- All entry gates: ORB filter 0.5, NFP/FOMC/CPI/FOMC+1 skip, Fri skip
- Stability checks: `stability_fraction=0.20`, `single_candle_fraction=0.15`, `fallback_stability_pts=12.0`
- Entry time: 09:35 ET
- `sl_dollars`: per-trade-matched per §3 (not a sweep dim)

**Reporting:**

- Top-5 configs by `sharpe_annualised` with full per-fold table
- Side-by-side IC-winner vs. strangle-winner metric table
- Per-trade buying-power-used (BP) reported but not constraining

## 5. Risks & open issues

### Modeling risks (must surface in findings doc)

1. **Pin/assignment risk.** Mitigated structurally by `eod_force_close: 15:30` (30 min before bell). Findings doc still reports a "near-pin at 15:30" diagnostic (count of days where any short strike within ±2 pts of spot at close-time) so the reader can see residual exposure.

2. **CBBO mark optimism at high em_mult.** SPX 0DTE strikes at em_mult ≥ 0.75 trade $0.05–$0.30 with wide percent-spreads — mid fills are upper-bound. Same assumption as the IC backtest, so within-study comparison is fair, but absolute strangle PnL at `em_mult ∈ {0.75, 1.00}` is upper-bound; flagged in findings.

3. **SPAN/Reg-T margin not enforced.** Strangles deploy 5–10× IC buying power. This study matches dollar-loss, not BP. BP-per-trade reported for reader to assess capital-efficiency trade-off.

4. **Per-trade SL coupling to IC log.** Strangle's daily SL is sourced from IC's same-day realized credit. Document that the result is "strangle-conditional-on-IC-risk-budget," not "strangle-standalone-optimal."

5. **No gate re-derivation.** Inherits IC's gates under the assumption of gate-strategy independence. Out of scope for this study — captured as follow-up.

### Open issues / follow-ups (deferred)

- **Sweep `eod_force_close` ∈ {15:30, 15:45, 15:55}** — 15:30 chosen as conservative default for this study. The optimal theta-vs-pin trade-off is its own study.
- **Gate re-derivation for strangle** — re-run filter-impact analysis on strangle returns to see if NFP / Fri / FOMC+1 skip ports cleanly.
- **Asymmetric strangle (different em_mult put/call)** — IC findings showed call-side dominance in losses (65% of losses). A skewed strangle (further-OTM call) may be the natural extension.

## 6. Deliverables

1. New strategy module, generalized runner, new config, analysis scripts (§2)
2. Persisted IC SL-by-day artifact (§3)
3. WF-CV results dir under `results/spx_0dte_open_strangle/`
4. Findings doc at `docs/spx_0dte_open_strangle_findings.md` mirroring the IC findings doc's structure
