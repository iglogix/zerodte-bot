# SPX 0DTE Open Strangle — Findings

*Last updated: 2026-05-20*

> **Status: pipeline implemented, full WF-CV sweep blocked on a shared-orchestrator extension. See "Known issue — backtest orchestrator" below.**

A walk-forward-validated SPX 0DTE short strangle entered at 09:35 ET, designed for risk-equivalent comparison against the [open IC](spx_0dte_open_ic_findings.md) via per-trade matched $-SL (capped at the IC SL distribution's 75th percentile).

## Design & spec

- Design spec: [docs/superpowers/specs/2026-05-19-spx-0dte-open-strangle-design.md](superpowers/specs/2026-05-19-spx-0dte-open-strangle-design.md)
- Implementation plan: [docs/superpowers/plans/2026-05-19-spx-0dte-open-strangle.md](superpowers/plans/2026-05-19-spx-0dte-open-strangle.md)

## What ships

| Component | File | Status |
|---|---|---|
| Strategy | [src_v2/strategies/open_strangle.py](../src_v2/strategies/open_strangle.py) | ✓ implemented, 12 unit tests passing |
| Config | [config/config_spx_0dte_open_strangle.yaml](../config/config_spx_0dte_open_strangle.yaml) | ✓ tuned to measured IC p75 ($2,397.50) |
| SL artifact extractor | [scripts/extract_ic_sl_dollars.py](../scripts/extract_ic_sl_dollars.py) | ✓ 270 days extracted from `results/spx_0dte_open_ic/trades.csv` |
| WF-CV runner (generalized) | [scripts/run_v2_wf_cv_open_ic.py](../scripts/run_v2_wf_cv_open_ic.py) | ✓ now resolves `strategy.class` from config |
| Strangle WF-CV entry | [scripts/run_v2_wf_cv_open_strangle.py](../scripts/run_v2_wf_cv_open_strangle.py) | ✓ `--mode em_mult` (36 combos) / `--mode delta` (30 combos) |
| Trade-log analyzer | [scripts/analyze_open_strangle_trades.py](../scripts/analyze_open_strangle_trades.py) | ✓ forked from IC analyzer |
| Comparator | [scripts/compare_strangle_vs_ic.py](../scripts/compare_strangle_vs_ic.py) | ✓ verified against IC baseline |

### IC SL artifact (measured)

```json
{
  "p75": 2397.5,
  "median": 2130.0,
  "mean": 2151.61,
  "min": 1250.0,
  "max": 3060.0,
  "n_days": 270,
  "sl_mult": 2.0
}
```

Strangle config now uses `sl_dollars_fallback: 2400.0` (≈ measured p75). Per-day SL$ is resolved at trade-entry from `data/ic_sl_dollars_by_day.csv` (regenerated artifact, gitignored).

## Acceptance gate (per spec §1)

Strangle winner must beat IC v5 on all three:

| Metric | IC baseline | Strangle target |
|---|---|---|
| OOS Sharpe (annualized) | 4.53 | ≥ 4.53 |
| Worst day | −$826 | ≥ −$826 |
| Total OOS PnL | +$15,014 | > +$15,014 |

**Not yet evaluated** — see blocker below.

## Known issue — backtest orchestrator assumes 4-leg trades

`src_v2/backtest/orchestrator.py:_convert_trades_to_results()` iterates over `("put", "call")` and requires both `short_{side}_strike` AND `long_{side}_strike` on each trade. Strangles only have shorts, so the loop's `if short_k is None or long_k is None: continue` skips every strangle trade — producing 0 `BacktestResult` rows even though the engine correctly opens trades and force-closes them at 15:30 ET (visible in the run log).

Symptom: `WalkForwardOptimizer` reports `Fold N: no valid combos in train (min_trades=N)` because the metrics function sees 0 trades per combo.

This is an orchestrator-side gap, not a strategy-side defect. Required fix (single function, ~30 lines):

1. When `long_k is None`, treat the side as a naked short:
   - Skip the long-leg exit-log lookup
   - `debit = max(0, short_exit_mid)` (no long-leg offset)
   - `spread_width = 0`, `long_strike = 0` (schema requires non-null)
   - `spread_type = f"{side}_short"` (or `naked_{side}`)
2. PnL formula `(credit - debit) × multiplier × qty - commission` is unchanged — works for both ICs and strangles because `credit` is gross premium and `debit` is whatever it costs to close.

This patch should land in its own branch with its own design review — touching shared backtest infrastructure affects every strategy. Out of scope for this delivery; flagged here so the next session has full context.

## Reproduce (after orchestrator fix)

```bash
# 1. Extract IC SL artifact (already done)
.venv/bin/python scripts/extract_ic_sl_dollars.py \
    --trades results/spx_0dte_open_ic/trades.csv \
    --sl-mult 2.0 \
    --out data/ic_sl_dollars_by_day.csv

# 2. Run em_mult sweep (36 combos × 4 folds)
.venv/bin/python scripts/run_v2_wf_cv_open_strangle.py \
    --mode em_mult \
    --config config/config_spx_0dte_open_strangle.yaml \
    --ratio 5 --folds 4 --top-k 5 \
    --objective sharpe_annualised \
    --min-trades 10 \
    2>&1 | tee results/spx_0dte_open_strangle/wf_cv_em_mult.log

# 3. Run delta sweep (30 combos × 4 folds)
.venv/bin/python scripts/run_v2_wf_cv_open_strangle.py \
    --mode delta \
    --config config/config_spx_0dte_open_strangle.yaml \
    --ratio 5 --folds 4 --top-k 5 \
    --objective sharpe_annualised \
    --min-trades 10 \
    2>&1 | tee results/spx_0dte_open_strangle/wf_cv_delta.log

# 4. Compare winner vs. IC baseline
.venv/bin/python scripts/compare_strangle_vs_ic.py \
    --strangle results/spx_0dte_open_strangle/<RUN_DIR>/trades.csv \
    --ic-meta data/ic_sl_dollars_by_day.meta.json
```

## Caveats (apply once results land)

Per spec §5 — must be addressed in the post-sweep update of this doc:

1. **Pin/assignment risk** — mitigated structurally by `eod_force_close: 15:30`. Report a "near-pin at 15:30" diagnostic (count of days where any short strike was within ±2 pts of spot at close-time).
2. **CBBO mark optimism at high em_mult** — SPX 0DTE strikes at em_mult ≥ 0.75 trade $0.05–$0.30 with wide percent-spreads. Same assumption as the IC backtest, fair within-study but absolute PnL at high em_mult is upper-bound.
3. **SPAN/Reg-T margin not enforced** — strangles deploy 5–10× IC buying power. Report BP-per-trade.
4. **Per-trade SL coupling to IC log** — strangle's daily SL is sourced from IC's same-day realized credit. Result is "strangle-conditional-on-IC-risk-budget."
5. **No gate re-derivation** — inherits IC's gates assuming gate-strategy independence.

## Follow-ups (deferred)

- Sweep `eod_force_close ∈ {15:30, 15:45, 15:55}` to find the theta-vs-pin sweet spot.
- Re-derive entry gates specifically for strangle returns.
- Asymmetric strangle (different em_mult put/call) — call-side dominance in IC losses suggests a skewed strangle may be the natural extension.
