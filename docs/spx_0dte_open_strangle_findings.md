# SPX 0DTE Open Strangle — Findings

*Last updated: 2026-05-20*

> **Result: IC HOLDS.** Under per-trade matched $-SL (the strangle's daily stop is set to that day's IC $-SL, capped at the IC SL distribution's 75th percentile = $2,397.50), the open IC beats the open strangle on every acceptance metric. Wings are NOT vestigial — the implicit insurance the long legs provide is worth more than the credit forgone to the long debit.

A walk-forward-validated SPX 0DTE short strangle entered at 09:35 ET, compared head-to-head against the [open IC](spx_0dte_open_ic_findings.md) over the same 4-year CBBO window (2022-05-11 → 2026-02-28).

## Acceptance gate vs. IC v5 baseline

| Metric | IC v5 baseline | Strangle (em_mult winner) | Strangle (delta winner) | Strangle wins? |
|---|---|---|---|---|
| **OOS Sharpe (mean per fold)** | 4.53 | 1.51 | 1.03 | ❌ |
| **Total OOS PnL** | +$15,014 | +$5,246 | +$2,676 | ❌ |
| **Profitable folds** | 3 of 4 | 2 of 4 | 3 of 4 | ❌ (em_mult) / tie (delta) |
| **Worst fold PnL** | — *(IC reports worst day -$826)* | -$6,305 | -$5,826 | ❌ |
| Aggregate win rate | 45.3% per day | 72.4% per trade | 81.0% per trade | ✓ but not relevant |
| Total OOS trades | 172 day-trades | 304 trades | 116 trades | — |

The strangle wins **more often** (high WR) but with **smaller wins and larger tails** — the canonical naked-premium-seller payout profile. Under matched dollar-risk, the IC's tail truncation from its long legs is worth more than the strangle's higher gross credit.

## Winning configurations

### em_mult mode winner (preferred)

[config/config_spx_0dte_open_strangle.yaml](../config/config_spx_0dte_open_strangle.yaml) with sweep-tuned overrides:

```yaml
options:
  strike_mode:          em_mult
  em_mult:              0.20      # tightest in the sweep — closer to spot for richer credit
  tp_pct_of_credit:     0.75      # canonical "25% of max profit" TP
  tp_pct_of_credit_end: null      # no time-ramp
  sl_dollars_csv:       data/ic_sl_dollars_by_day.csv
  sl_dollars_fallback:  2400.0    # = IC SL p75 ($2,397.50)
strategy:
  eod_force_close:      "15:30"
  skip_fomc_next_day:   true
  skip_weekdays:        ["Fri"]
```

**Per-fold OOS** (em_mult winner, ratio 5:1, 4 folds):

| Fold | Validation window | Trades | WR | PnL | Sharpe |
|---|---|---|---|---|---|
| 1 | 2024-06 → 2024-11 | 64 | 78.1% | **+$3,343** | 1.70 |
| 2 | 2024-11 → 2025-04 | 72 | 70.8% | **-$6,305** | -2.06 |
| 3 | 2025-04 → 2025-09 | 88 | 72.7% | **+$13,779** | 8.21 |
| 4 | 2025-09 → 2026-02 | 80 | 68.8% | **-$5,571** | -1.81 |
| **Σ** | | **304** | **72.4%** | **+$5,246** | mean 1.51 |

### delta mode winner

```yaml
options:
  strike_mode:          delta
  target_short_delta:   0.30      # widest in the sweep
  tp_pct_of_credit:     0.25      # late TP (capture 75% of max)
  tp_pct_of_credit_end: 0.75      # ramp loose by EOD
```

**Per-fold OOS** (delta winner):

| Fold | Trades | WR | PnL | Sharpe |
|---|---|---|---|---|
| 1 | 22 | 86.4% | +$609 | 0.76 |
| 2 | 34 | 79.4% | +$389 | 0.30 |
| 3 | 38 | 81.6% | **+$7,504** | 9.74 |
| 4 | 22 | 77.3% | **-$5,826** | -6.68 |
| **Σ** | **116** | **81.0%** | **+$2,676** | mean 1.03 |

Delta-mode strangles trade much less (116 vs 304) because the 0.30-delta short typically prices at $5–$15 — fewer days pass `min_credit_per_spread` after applying slippage. WR is higher (81%) but smaller sample and same regime-dependent tails.

## Interpretation

### Why does the strangle lose?

**Risk-equivalent comparison** (matched $-SL) holds the *downside dollar exposure* of each strategy equal — the strangle's daily SL = IC's same-day $-SL, capped at $2,398 (p75). But:

1. **The IC pays for protection it actually uses.** IC max-loss-per-side ≈ 2× credit (sl_mult=2.0). The IC's long wing limits per-side losses to (width − credit) × 100. At wing_width=25 and call_credit ~$0.75, that's $1,825 — well above sl_mult cap of $150. So in practice the IC's SL fires *before* the wing matters; the wing is "wasted" most of the time. BUT — on tail days, when the SL is breached intrabar at a worse mark than expected, the wing caps slippage. **The strangle has no such cap.**

2. **Strangle credit at em_mult=0.20 is ~$8 (vs IC's ~$1.50 net), 5× richer.** That's the gross premium advantage. But the strangle's max single-trade loss is also uncapped (truncated at $2,400 SL, not at wing). On a fast tail move, the strangle gets stopped harder than the IC.

3. **Fold 2 and Fold 4 are losing folds for both modes.** Both correspond to choppy/trending periods in late-2024 to early-2025 and again Q4 2025 — regimes where small directional moves break the SL without giving the position time to decay. The strangle's higher gamma (no protective long) amplifies this.

### The empirical answer to "wings are vestigial?"

**No. Wings provide tail-event insurance that pays out under the matched-SL regime.** The IC's 4-year aggregate Sharpe of 4.53 vs strangle's 1.51 (best mode) is a 3× advantage in risk-adjusted terms. Even though the IC pays a debit to buy wings, the realized tail protection — visible in IC's worst day -$826 vs strangle's worst fold -$6,305 — more than offsets the credit loss.

### Caveats (as predicted in spec §5)

1. **Pin/assignment risk** — mitigated by `eod_force_close: 15:30`. Not yet quantified with a near-pin diagnostic. Real-world assignment cost would make the strangle worse, not better — this finding is conservative.
2. **CBBO mark optimism at high em_mult / low delta** — applies to both ICs and strangles symmetrically in this study, so within-comparison the verdict is fair. Absolute strangle PnL at em_mult=0.20 is *less* speculative than at em_mult=1.00 (closer-to-spot strikes have tighter pct-spreads).
3. **SPAN/Reg-T margin not enforced** — strangles deploy 5–10× IC buying power. This study matched $-loss, not BP, so the comparison disadvantages strangles even more on capital-efficiency grounds.
4. **Per-trade SL coupling to IC log** — strangle SL is sourced from IC's same-day credit. Result is "strangle-conditional-on-IC-risk-budget." A standalone strangle with its own SL ceiling could differ.
5. **No gate re-derivation** — inherited IC's gates. A strangle-native gate set might claw back some alpha, but unlikely to close a 3× Sharpe gap.

## Implementation that ships

| Component | File | Status |
|---|---|---|
| Strategy | [zerodte/strategies/open_strangle.py](../zerodte/strategies/open_strangle.py) | ✓ 12 unit tests passing |
| Config | [config/config_spx_0dte_open_strangle.yaml](../config/config_spx_0dte_open_strangle.yaml) | ✓ tuned to IC p75 |
| Backtest support (2-leg) | [zerodte/backtest/orchestrator.py](../zerodte/backtest/orchestrator.py), [zerodte/backtest/bt_tool_handlers.py](../zerodte/backtest/bt_tool_handlers.py) | ✓ naked-short branches |
| SL artifact extractor | [scripts/extract_ic_sl_dollars.py](../scripts/extract_ic_sl_dollars.py) | ✓ 270-day artifact, p75=$2,397.50 |
| WF-CV runner | [scripts/run_v2_wf_cv_open_ic.py](../scripts/run_v2_wf_cv_open_ic.py), [scripts/run_v2_wf_cv_open_strangle.py](../scripts/run_v2_wf_cv_open_strangle.py) | ✓ `--mode em_mult` / `delta` |
| Trade-log analyzer | [scripts/analyze_open_strangle_trades.py](../scripts/analyze_open_strangle_trades.py) | ✓ forked from IC analyzer |
| Comparator | [scripts/compare_strangle_vs_ic.py](../scripts/compare_strangle_vs_ic.py) | ✓ |

## Reproduce

```bash
# 1. Extract IC SL artifact (only once; output is gitignored)
.venv/bin/python scripts/extract_ic_sl_dollars.py \
    --trades results/spx_0dte_open_ic/trades.csv \
    --sl-mult 2.0 \
    --out data/ic_sl_dollars_by_day.csv

# 2. em_mult sweep (~3 min on workers=1, 36 combos × 4 folds)
.venv/bin/python scripts/run_v2_wf_cv_open_strangle.py \
    --mode em_mult \
    --config config/config_spx_0dte_open_strangle.yaml \
    --ratio 5 --folds 4 --top-k 5 \
    --objective sharpe_annualised \
    --min-trades 10 \
    --workers 1 \
    > results/spx_0dte_open_strangle/wf_cv_em_mult.log 2>&1

# 3. delta sweep (~2 min, 30 combos × 4 folds)
.venv/bin/python scripts/run_v2_wf_cv_open_strangle.py \
    --mode delta \
    --config config/config_spx_0dte_open_strangle.yaml \
    --ratio 5 --folds 4 --top-k 5 \
    --objective sharpe_annualised \
    --min-trades 10 \
    --workers 1 \
    > results/spx_0dte_open_strangle/wf_cv_delta.log 2>&1
```

**Note on `--workers 1`:** parallel workers each duplicate the in-memory CBBO cache (~9 GB at full window), so workers=2 pushes a 16-GB-free system into swap and stalls. Sequential is fine — the full sweep is fast (~3 min).

## Follow-ups (deferred, no longer urgent)

- Sweep `eod_force_close ∈ {15:30, 15:45, 15:55}` — would only matter if strangle were a contender, which it isn't.
- Re-derive entry gates specifically for strangle returns — same.
- **Asymmetric strangle (further-OTM call than put)** — the IC findings showed 65% of losses were call-side breaches in 2022-2023. A skewed strangle might do better on call-side tails, but a 3× Sharpe gap is unlikely to close.
- The orchestrator's naked-short support shipped in [zerodte/backtest/orchestrator.py](../zerodte/backtest/orchestrator.py) and [zerodte/backtest/bt_tool_handlers.py](../zerodte/backtest/bt_tool_handlers.py) — usable by any future 2-leg strategy (covered call, naked put, etc.) without further changes.
