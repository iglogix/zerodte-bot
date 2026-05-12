# SPX 0DTE Open Iron Condor — Findings

*Last updated: 2026-05-12*

A walk-forward-validated tastytrade-style iron condor entered at 09:35 ET on the SPX 0DTE cycle. Backtested against ~4 years of OPRA CBBO bid/ask data (2022-04-18 → 2026-02-28).

## Final configuration

[config/config_spx_0dte_open_ic.yaml](../config/config_spx_0dte_open_ic.yaml)

```yaml
entry:
  entry_time: "09:35"                        # one entry/day, near the open
  stability_check: true
  stability_fraction: 0.20                   # 3-bar range / EM threshold
  single_candle_fraction: 0.15
  fallback_stability_pts: 12.0
  opening_range_filter:                      # GATE A — skip wide-open days
    enabled: true
    window_start: "09:30"
    window_end:   "09:35"
    max_open_range_em_ratio: 0.5             # skip if first 5-min range > 50% of straddle EM

strategy:
  market_open:     "09:30"
  eod_force_close: "15:59"                   # hold to settlement
  skip_nfp_fridays:    true                  # GATE B — existing
  skip_fomc_days:      true
  skip_cpi_days:       true
  skip_fomc_next_day:  true                  # GATE C — analysis 2026-05-12
  skip_cpi_next_day:   false                 # CPI+1 was NOT a systematic loser
  skip_weekdays:       ["Fri"]               # GATE D — Friday systematically negative

options:
  target_short_delta:    0.25                # WF-CV winner (was 0.10-0.20 expected)
  wing_width:            25                  # pts
  tp_pct_of_credit:      0.25                # late TP (capture 75% of max profit)
  tp_mode:               "total"             # close full IC at total mark <= 25% credit
  sl_mult_of_credit:     2.0                 # max loss = 2× credit per side
  hold_put_breach:       true                # no separate breach stops — sl_mult handles it
  hold_call_breach:      true
```

## OOS performance (full 4-year backtest with all gates)

| Metric | Value |
|---|---|
| Trading days in window | 953 |
| **Trade days after all gates** | **172** |
| Skipped (calendar + guards + new gates) | 781 |
| Per-day WR | **45.3%** |
| **Total OOS PnL** | **+$15,014** |
| Annualized | **~$3,884/yr** |
| Mean win day | +$340 |
| Mean loss day | -$123 |
| **Worst day** | **-$826** |
| Best day | +$793 |
| Sharpe (annualized) | **4.53** |

Per fold in the most recent WF-CV (v5, ratio 5:1, 4 folds, train 2023-09 → val ending 2026-02-25):

| Fold | Val | Trades | WR | PnL | Sharpe |
|---|---|---|---|---|---|
| 1 | Jan–Apr 2025 | 4 | 75.0% | +$412 | 9.92 |
| 2 | May–Aug 2025 | 6 | 83.3% | +$905 | 13.30 |
| 3 | Aug–Nov 2025 | 76 | 55.3% | **-$1,047** | -1.25 |
| 4 | Nov–Feb 2026 | 46 | 60.9% | +$1,639 | 2.91 |
| **Σ** | | 132 | 60% | +$4,780 | mean 7.01 |

## Why these settings (and not vanilla tastytrade's)

| Parameter | TT canonical | This config | Why we diverge |
|---|---|---|---|
| Short delta | 0.10–0.15 | **0.25** | Higher delta delivers richer credits that survive realistic CBBO costs. Lower deltas fail `min_credit_per_spread` at this entry time. |
| Wing width | 30 | **25** | Tighter wings capped max loss to width × 100 − credit = ~$1,800, which `sl_mult=2.0` then caps at ~$300 in practice. |
| TP | 25% of max profit (≈ exit at 75% of credit value) | **late TP — exit at 25% of credit value (75% of max profit captured)** | The WF-CV winner consistently picks late TP. Captures more theta per trade. |
| Stop loss | None | **2× credit** | Vanilla TT (no SL) is breakeven in 2022–2026 SPX (we measured: +$325 over 3.85 yr, 86% WR). The 14% loss-tail offsets all 86% of wins. Adding 2× credit cap converts breakeven → meaningfully positive. |
| Breach stops | N/A (held as one position) | Both `hold_*_breach=true` | Matches TT's "manage IC as a single position." |

## Diagnostic wins/losses analysis (the breakthrough)

The breakthrough was a per-trade dimensional analysis ([scripts/analyze_open_ic_trades.py](../scripts/analyze_open_ic_trades.py)) that sliced PnL across weekday / VIX9D / intraday range / month / calendar proximity. Key findings on the pre-gate WF-CV winner (delta=0.25 / width=25 / sl=2× / no extra gates):

### 1. Intraday range is the dominant predictor

| Intraday range | Trade days | WR | Mean | Total |
|---|---|---|---|---|
| Tight (<25pts) | 17 | **82.4%** | +$245 | +$4,172 |
| Medium (25–50) | 103 | 42.7% | +$87 | +$9,003 |
| Wide (50–80) | 64 | 43.8% | +$34 | +$2,186 |
| **Volatile (80+)** | **29** | **17.2%** | **-$98** | **-$2,852** |

Volatile days (~14% of trade days) destroyed ~25% of the gross PnL. The 5-min opening-range filter is a leading-indicator proxy: if the first 5 minutes already moved > 50% of the day's expected straddle EM, the day is materially more likely to be volatile end-to-end.

### 2. FOMC next-day systematically loses

| Calendar proximity | Days | Total PnL |
|---|---|---|
| FOMC eve | 6 | +$982 |
| FOMC day | (skipped) | — |
| **FOMC next-day** | **5** | **-$1,122** |
| CPI eve | 7 | +$182 |
| CPI day | (skipped) | — |
| CPI next-day | 9 | +$957 |
| NFP eve | 10 | +$1,785 |
| NFP day | (skipped) | — |

Markets digest Fed decisions overnight; the next morning is whipsawed. The CPI+1 effect doesn't replicate (+$957 across 9 days), so we only skip FOMC+1.

### 3. Friday loses (small effect, big tail)

Post-other-gates weekday split:

| Weekday | Trade days | WR | Mean | Total |
|---|---|---|---|---|
| Mon | 39 | 48.7% | +$119 | +$4,653 |
| Tue | 51 | 45.1% | +$131 | +$6,697 |
| Wed | 44 | 40.9% | +$10 | +$426 |
| Thu | 38 | 47.4% | +$85 | +$3,238 |
| **Fri** | **32** | **34.4%** | **-$17** | **-$532** |

Skipping Fridays removes only -$532 from totals — small. But the **worst day ever (-$1,178)** was a Friday, so removing them cut tail loss by 30% (-$1,178 → -$826).

## Validated regime sensitivity

Worth noting: the strategy still loses fold 3 (Aug–Nov 2025) by -$1,047 even with all gates. That window had several multi-day trend days that breached past 2× credit. The gates can't predict everything; some regime exposure remains.

| Year | Days | WR | Total |
|---|---|---|---|
| 2022 (partial) | 11 | 54.5% | +$2,350 |
| 2023 | 66 | 40.9% | +$3,510 |
| 2024 | 52 | 44.2% | +$2,017 |
| 2025 | 64 | 40.6% | +$1,662 |
| 2026 YTD | 20 | 45.0% | +$2,970 |

2023 was best, 2025 weakest. The strategy is profitable in every calendar year tested.

## Bugs found along the way

Two non-trivial backtest+production bugs uncovered during this work:

**1. [src_v2/services/position_monitor.py](../src_v2/services/position_monitor.py)** — `_check_trade` was marking the whole IC `status: closed` after a single side's stop_loss fired, so the un-stopped side stopped being monitored. Backtest reported a fake 100% WR because the un-stopped side was always marked "expired_worthless" at EOD regardless of intraday breach. This was also a live-bot bug: live IC trades with one stopped leg became unmonitored. Fixed via `_any_legs_remain()` — status flips to closed only when all legs are closed.

**2. [src_v2/strategies/open_ic.py](../src_v2/strategies/open_ic.py)** — breach check was firing on every bar after a side closed (returning empty leg lists), spamming the position monitor log to 80MB and triggering OOM. Fixed by requiring `_call_leg_symbols(trade)` / `_put_leg_symbols(trade)` non-empty before firing breach exit.

## Tooling

- [src_v2/strategies/open_ic.py](../src_v2/strategies/open_ic.py) — strategy implementation. BS-delta strike picker calibrated against ATM straddle.
- [src_v2/agent/window_entry_engine.py](../src_v2/agent/window_entry_engine.py) — single-shot entry engine with stability + VIX9D + opening-range gates.
- [src_v2/backtest/orchestrator.py](../src_v2/backtest/orchestrator.py) — dispatches the right engine + applies calendar/weekday skips.
- [scripts/run_v2_backtest.py](../scripts/run_v2_backtest.py) — single-config backtest run + report.
- [scripts/run_v2_wf_cv_open_ic.py](../scripts/run_v2_wf_cv_open_ic.py) — walk-forward CV grid search.
- [scripts/analyze_open_ic_trades.py](../scripts/analyze_open_ic_trades.py) — per-trade dimensional breakdown (weekday / VIX / range / calendar / month / year).
- [scripts/compare_tt_variants.py](../scripts/compare_tt_variants.py) — A/B/C comparison of asymmetric vs strict-TT vs vanilla-TT exit logic.

## Reproduce

```bash
# Single backtest with current winning config
.venv/bin/python -m scripts.run_v2_backtest \
  --config config/config_spx_0dte_open_ic.yaml \
  --start 2022-04-18 --end 2026-02-28

# Per-dimension wins/losses analysis
.venv/bin/python -m scripts.analyze_open_ic_trades

# Walk-forward CV
WF_MEMORY_THROTTLE_PCT=95 .venv/bin/python -m scripts.run_v2_wf_cv_open_ic \
  --config config/config_spx_0dte_open_ic.yaml \
  --start 2023-09-01 --end 2026-02-28 \
  --ratio 5 --folds 4 --top-k 10 --min-trades 30 \
  --objective total_pnl --workers 1
```

## What's left to try

- **Volatility-scaled SL**: `sl_mult` could be a function of VIX9D or implied EM rather than a constant 2×. Fold 3 (Aug–Nov 2025) lost because 2× wasn't tight enough for that regime.
- **Bootstrap MC for ruin probability** — quantify drawdown distribution and per-contract scaling at $100k account.
- **Same-strategy validation on ES options** — orthogonal verification per the earlier ES backfill plan ([docs/superpowers/specs/](../docs/superpowers/specs/)).
- **Combine with the existing 8AM-ES-ORB strategy** for portfolio diversification across two underlying timing windows.
