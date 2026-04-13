# ORB EM IC — Walk-Forward CV Findings

**Date:** 2026-04-13
**Strategy:** `orb_em_ic` (SPX 0DTE bias-adaptive iron condor)
**Data:** SPY 1-min bars + SPX CBBO 1-min quotes, 2025-09-22 → 2026-03-20 (125 days)
**Framework:** `src_v2/backtest/` orchestrator running real OrbEngine + PositionMonitor + LiveStrategy

## Baseline: Original config params fail under accurate backtest

**Params (pre-optimization):** em_with=0.5, em_against=0.75, width_with=5, width_against=10, tp_pct=0.5

| | src_8am | src_v2 (accurate) |
|---|---|---|
| Trades | 72 | 72 |
| Win Rate | 93.1% | 70.8% |
| Total PnL | +$4,518 | **−$1,420** |
| Sharpe | 14.44 | −3.12 |
| Profit Factor | 4.35 | 0.73 |

**Why src_8am was wrong:**
1. Used SPY×10 bar highs for breach detection (~50pt below true SPX spot, no dividend adjustment)
2. Marked all EOD exits as "expired_worthless" $0 debit, even when ITM

## In-sample grid search (576 combos, ratio-filtered)

**Best Sharpe (≥50 trades):**
```
em_with_trend=1.0, em_against_trend=2.0, width_with_trend=20, width_against_trend=10, tp_pct_of_credit=0.65
```

| Metric | Value |
|---|---|
| Trades | 56 |
| Win Rate | 96.4% |
| Total PnL | +$2,371 |
| Sharpe | 36.90 |
| Profit Factor | 23.58 |
| Max Drawdown | $95 |

Key insight: strikes must be **much wider** than original (2× ATM straddle vs 0.75×) to avoid breaches under realistic SPX spot detection.

## Walk-Forward CV (ratio 5:1, 4 folds)

**Grid:** 288 combos (focused around IS best zone)

**Fold structure** (each val window = 13 trading days):

| Fold | Train | Val |
|---|---|---|
| 1 | 2025-09-22 → 2025-12-15 (60d) | 2025-12-16 → 2026-01-05 |
| 2 | 2025-09-22 → 2026-01-02 (72d) | 2026-01-05 → 2026-01-22 |
| 3 | 2025-09-22 → 2026-01-21 (84d) | 2026-01-22 → 2026-02-09 |
| 4 | 2025-09-22 → 2026-02-06 (96d) | 2026-02-09 → 2026-02-26 |

### Winner (appeared in top-5 of all 4 folds)

```yaml
options:
  em_with_trend: 1.25
  em_against_trend: 2.25
  width_with_trend: 10
  width_against_trend: 10
  tp_pct_of_credit: 0.75
strategy:
  midline_tolerance_pts: 1.0
  orb_range_filter:
    min_pts: 2.0
```

### Per-fold OOS performance

| Fold | Val window | Trades | WR | PnL | Sharpe |
|---|---|---|---|---|---|
| 1 | Dec 16 – Jan 5 | 6 | 100% | +$81 | 123.7 |
| 2 | Jan 5 – Jan 22 | 6 | 83% | +$72 | 18.0 |
| 3 | Jan 22 – Feb 9 | 2 | 100% | +$20 | 0 (1 day) |
| 4 | Feb 9 – Feb 26 | 2 | 100% | +$57 | 0 (1 day) |

### Aggregate OOS (52 days combined)

| Metric | Value |
|---|---|
| Total trades | 16 |
| Win Rate | 93.75% |
| Total PnL | +$230 |
| Max Drawdown | $25 |
| Mean fold Sharpe | 35.43 |
| **Annualized ARR on $100K** | **~1.1%** |

### Put/Call breakdown (combined OOS)

| Side | Trades | Wins | Losses | WR | Total PnL | Avg PnL | Max Loss |
|---|---|---|---|---|---|---|---|
| **Put** | 8 | 7 | 1 | 87.5% | +$98 | +$12.2 | −$25 |
| **Call** | 8 | 8 | 0 | 100% | +$132 | +$16.5 | +$3 |

**Exit reasons:**
- Put: 7 × TP_50pct, 1 × EOD forced close (−$25)
- Call: 8 × TP_50pct

**Zero breaches OOS.** The wider strikes (2.25× ATM straddle) successfully avoided all short-strike breaches. Only loss was a put expiring slightly ITM at EOD.

## Reality check

**Good signs:**
- Same param combo wins consistently across all 4 folds (not fold-specific overfitting)
- Positive OOS PnL and high WR
- Zero breaches OOS — wider strikes work
- Symmetric put/call performance

**Concerning:**
- Trade frequency drops hard in later folds (only 2 trades per 13-day window in folds 3-4)
- Only 16 OOS trades total — thin statistical basis
- 1.1% ARR is far below 70% min ARR target
- 125 days of CBBO data is insufficient for robust conclusions

## Recommendations

1. **Collect more CBBO history** — need 2+ years for meaningful WF-CV (500+ OOS trades)
2. **EM floor** — consider `em = max(atm_straddle, 20pt)` so strikes aren't placed too close on calm days (current skips on low-EM days likely drive the trade frequency drop)
3. **Don't deploy live at current scale** — $14/trade is too thin for live execution costs and slippage
4. **Scale quantity cautiously** if params validate with more data

## Commands to reproduce

```bash
# Single backtest with WF-CV winner params (edit config first)
python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml

# Full WF-CV rerun
python -m scripts.run_v2_wf_cv --config config/config_orb_em_ic_spx.yaml

# Grid search (edit PARAM_GRID in script)
python -m scripts.run_v2_optimizer --config config/config_orb_em_ic_spx.yaml --top 20 --min-trades 50
```
