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

## In-sample grid search (648 combos with expanded range + tolerance)

**Best by total_pnl (min 50 trades, Sharpe ≥ 3):**
```
em_with_trend=0.5, em_against_trend=2.0, width_with_trend=15, width_against_trend=10, tp_pct_of_credit=0.75
```

| Metric | Value |
|---|---|
| Trades | 56 |
| Win Rate | 91.1% |
| Total PnL | +$1,517 |
| Sharpe | 6.82 |
| Profit Factor | 2.52 |
| Max Drawdown | $371 |

### Key findings from grid search
- **em_against_trend=2.0 is critical** — all top-10 combos have this param. Wide against-trend strikes avoid breaches.
- **em_with_trend=0.5 matches 1.0 on PnL, with lower Sharpe** — tighter with-trend side trades more frequently
- **Wider midline_tolerance (5.0, 10.0) didn't make top 10** — the current 1.0 works fine
- **tp_pct=0.75 dominates** — early profit-taking reduces breach exposure

## Walk-Forward CV (ratio 5:1, 4 folds, total_pnl objective with Sharpe≥3 floor)

**Grid:** 96 combos (focused around tighter with-trend zone)

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
  em_with_trend: 0.5
  em_against_trend: 2.0
  width_with_trend: 15
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
| 1 | Dec 16 – Jan 5 | 10 | 90.0% | +$424 | 79.36 |
| 2 | Jan 5 – Jan 22 | 8 | 87.5% | +$270 | 13.30 |
| 3 | Jan 22 – Feb 9 | 2 | 50.0% | **−$326** | 0 (1 loss) |
| 4 | Feb 9 – Feb 26 | 4 | 100% | +$230 | 64.54 |

### Aggregate OOS (52 days combined)

| Metric | Value |
|---|---|
| Total trades | 24 |
| Win Rate | 87.5% |
| Total PnL | +$598 |
| Max Drawdown | $338 |
| Mean fold Sharpe | 39.3 |
| **Annualized ARR on $100K (1 contract)** | **~2.9%** |

## Comparison: objective choice matters

| | Sharpe objective | total_pnl objective (Sharpe≥3) |
|---|---|---|
| Winner em_with | 1.25 | **0.5** |
| Winner em_against | 2.25 | 2.0 |
| OOS trades | 16 | **24** |
| OOS PnL | +$230 | **+$598** (2.6×) |
| OOS WR | 93.75% | 87.5% |
| Max fold DD | $25 | $338 |

**Key insight:** Tighter with-trend strikes (0.5× straddle) generate ~50% more trades and ~2.6× more dollars, at the cost of slightly lower WR and higher drawdowns. The trend protects the with-trend side well enough that tighter is OK.

## Quantity scaling

At 1 contract, $598 over 52 OOS days = ~$2,898/year = **2.9% ARR**.

Scaling to hit 70% ARR target:

| Contracts | BP/trade | Projected ARR | Max DD | BP util on $100K |
|-----------|----------|---------------|--------|-------------------|
| 1 | ~$500 | 2.9% | $338 | 0.5% |
| 10 | ~$5,000 | 29% | $3,380 | 5% |
| **25** | **~$12,500** | **72%** | **$8,450** | **12.5%** |
| 50 | ~$25,000 | 144% | $16,900 | 25% |

**25 contracts reaches the 70% ARR target** while staying well under the 50% BP cap. Max DD of $8.4K = 8.4% of account, manageable.

## Put/Call breakdown (combined OOS, 16-trade sample from Sharpe objective winner)

| Side | Trades | Wins | Losses | WR | Total PnL | Avg PnL | Max Loss |
|---|---|---|---|---|---|---|---|
| **Put** | 8 | 7 | 1 | 87.5% | +$98 | +$12.2 | −$25 |
| **Call** | 8 | 8 | 0 | 100% | +$132 | +$16.5 | +$3 |

Zero breaches OOS for the Sharpe-optimized combo; the single loss was a put EOD close (slightly ITM).

## Reality check

**Good signs:**
- Same param combo wins consistently across all 4 folds (not fold-specific overfitting)
- Positive OOS PnL under both Sharpe and total_pnl objectives
- 25 contracts gets to target ARR with reasonable risk metrics

**Concerning:**
- Only 24 OOS trades total — thin statistical basis
- Fold 3 showed a real −$326 loss on 2 trades (one big put breach)
- 125 days of CBBO data is insufficient for robust conclusions
- Kelly sizing would likely recommend fewer than 25 contracts given small sample

## Recommendations

1. **Collect more CBBO history** — need 2+ years for 500+ OOS trades before real deployment
2. **Don't scale to 25 contracts yet** — start smaller (5-10) until more data validates
3. **Consider min-EM floor** — on calm days EM is small; strikes placed close even at 2× multiplier (may drive some fold 3 losses)
4. **Monitor live deployment carefully** — one fold having 2 trades means market regimes matter. Be ready to pause if WR drops.

## Commands to reproduce

```bash
# Single backtest with WF-CV winner params
python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml

# Full WF-CV with total_pnl objective (best result)
python -m scripts.run_v2_wf_cv --config config/config_orb_em_ic_spx.yaml \
    --objective total_pnl --min-trades 20 --min-sharpe 3

# Grid search with constraints
python -m scripts.run_v2_optimizer --config config/config_orb_em_ic_spx.yaml \
    --objective total_pnl --min-trades 50 --min-sharpe 3 --top 25
```
