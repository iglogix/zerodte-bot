# SPX 0DTE Jade Lizard — Backtest Findings

**Date:** 2026-06-02
**Spec:** [docs/superpowers/specs/2026-06-02-spx-0dte-jade-lizard-backtest-design.md](superpowers/specs/2026-06-02-spx-0dte-jade-lizard-backtest-design.md)
**Plan:** [docs/superpowers/plans/2026-06-02-spx-0dte-jade-lizard-backtest.md](superpowers/plans/2026-06-02-spx-0dte-jade-lizard-backtest.md)
**Source strategy:** [docs/spx_0dte_jl.md](spx_0dte_jl.md) (TastyTrade 9:30am Jade Lizard)

## Bottom line

**As specified, the strategy has no edge.** Across 2022-05-11 → 2026-02-27 (940 trades,
full multi-regime incl. the 2022 bear), faithfully managed at 10-minute checks:

| View (1 contract, ×100) | Total | Mean/trade | WR | Sharpe (ann.) | Max DD |
|---|---|---|---|---|---|
| **Mid-exit, zero-cost** (TT-idealized headline) | **−$19,500** | −$20.7 | 67.7% | −0.40 | −$23,887 |
| With costs (slip 0.10/spread, comm $1/leg) | −$60,860 | −$64.7 | 65.2% | −1.25 | −$62,282 |
| Chain-intrinsic settlement (honest) | −$30,828 | −$32.8 | 67.3% | −0.61 | −$32,505 |

A 67.7% win rate masks a **losing expectancy**: `avg_win $444 ≪ avg_loss $995`. This is
the classic negative-skew profile of premium selling — frequent small wins, occasional
large losses (worst day −$2,160) that more than cancel them. Even the perfectly idealized
"enter and exit at mid, pay nothing" number is negative; realistic costs roughly triple
the loss.

## Strategy (as backtested)

- **Entry** at the 09:30 opening interval (the first OPRA `cbbo-1m` bar, labeled `09:31`).
- **EM** = ATM straddle mid at entry. ATM and the SPX forward are derived from the option
  chain itself (`argmin|call_mid − put_mid|`; forward via put-call parity) — never SPY×10.
- **Short put** at `floor((fwd − 0.25×EM)/5)×5`, **long put** −30 → $30-wide put spread.
- **Short call** at `ceil((fwd + 0.25×EM)/5)×5`, **long call** +10 → $10-wide call spread.
  (Symmetric 0.25×EM placement, asymmetric widths — a *defined-risk asymmetric IC*. It only
  has the true jade-lizard "no upside risk" property when `total_credit ≥ $10`, which held
  on **55.5%** of days; avg credit was **$10.30**.)
- **Exit:** whole-position. Close all legs at the first 10-min anchor where
  `combined_mark ≤ 0.50 × total_credit`. **No stop-loss**; otherwise hold to the close.
- **Fills at bid/ask mid.** Headline is zero-cost; a with-costs sensitivity is shown.

## Data & method

- **Tailored SPXW backfill** (`scripts/backfill_jl_cbbo.py`): OPRA.PILLAR `cbbo-1m`, the full
  09:30→16:00 session, ±100-pt fixed strike band centered on the 09:30 spot, written to
  `data/cbbo_cache_jl_10min/`. **942 days, ~30.1M quotes, $4.51** (free dry-run gated; $15 cap).
- **`cbbo_cache_es` was rejected** — it is options on **ES futures** (`collect_es_options.py`
  uses `NestedFutureOptionChain("ES")`; its ATM forward sits ~20 pts above the true-SPX
  5-anchor cache = the ES basis). Using it would misplace the put side by ~the basis.
- **Note on timestamps:** `cbbo-1m` bars are stamped at interval END, so the session reads
  `09:31…15:59`. Entry uses `09:31` (the 09:30:00–09:31:00 opening interval); the 10-min grid
  is `09:31, 09:40, 09:50, …, 15:50, 15:59` (40 anchors), verified present with zero gaps.
- **Missing days:** 11 of ~953 trading days failed to fetch (1 symbology error on 2022-05-12,
  10 transient Databento 504 timeouts) and are absent — <1.2%, scattered, immaterial to the
  result. Re-running the backfill (resumable) would fill them for ~$0.05.

**Reproduce:**
```bash
.venv/bin/python -m scripts.analytical_jl_eval \
  --cache data/cbbo_cache_jl_10min --entry 09:31 \
  --anchors 09:31,09:40,09:50,10:00,10:10,10:20,10:30,10:40,10:50,11:00,11:10,11:20,11:30,11:40,11:50,12:00,12:10,12:20,12:30,12:40,12:50,13:00,13:10,13:20,13:30,13:40,13:50,14:00,14:10,14:20,14:30,14:40,14:50,15:00,15:10,15:20,15:30,15:40,15:50,15:59 \
  --start 2022-05-11 --end 2026-02-27
```
Per-trade rows: `results/jade_lizard/trades_headline.csv`. Logic is unit-tested
(`tests/test_jl_core.py`) and every per-trade row was checked for the width / 0.25×EM /
`pnl = (credit − exit_mark)×100` invariants (0 structural violations).

## 10-minute vs 5-anchor — a methodology note

The same logic on the existing true-SPX **5-anchor** cache (entry 09:35; checks at
11:00/13:00/15:30/15:59) gave a *positive* mid-exit headline (+$9,137). The faithful
**10-minute** version is *negative* (−$19,500). The difference is not noise — it is the
take-profit rule applied honestly:

- Exit-type mix shifts from `{tp:316, expiry:631}` (5-anchor) to `{tp:441, expiry:499}` (10-min):
  10-min closes 125 more positions at the 50% target.
- Coarse checking lets smooth winners **run past** 50% to fuller decay (a luckier, larger
  win) because it never samples the 50% touch. Faithful 10-min checking caps each winner at
  exactly +50% of credit (avg win $488 → $444).
- That capping costs more than the gain from catching "dip-to-50%-then-revert" days as wins.

**Implication:** "close winners at 50%" *as written* is the wrong knob here — letting winners
run (a deeper TP threshold or hold-to-expiry) would have done better. The 5-anchor "+$9k"
was an artifact of under-applying the TP rule, not a real edge.

## Caveats

- **Idealized mid is optimistic.** Held-to-expiry losers are marked at the 15:59 combined mid;
  honest chain-intrinsic settlement is worse (−$30.8k). Real fills (with costs) are worse still
  (−$60.9k). None are positive.
- **Symmetric 0.25×EM ≠ classic jade lizard.** The call strike is not moved to guarantee
  credit ≥ width, so 44.5% of days carried genuine upside risk.
- **No stop-loss, no entry filters** (TT-vanilla). Adding a stop or skipping high-range/event
  days was not tested here.
- Scaling to a $100K account amplifies the *negative* expectancy (defined-risk BP ≈ $1,970/lot
  → ~50 lots → ~50× the per-lot loss); there is no positive expectancy to scale.

## Suggested next steps (optional, not run)

1. **WF-CV sweep** (`feedback_cv_methodology`: 5:1, 4 folds) over `tp_pct` (incl. "let winners
   run" / hold-to-expiry), `em_mult`, and put/call widths — to test whether *any* parameterization
   has an edge, given the TP-capping finding above.
2. **True jade-lizard placement** (move the short call so credit ≥ $10 width) to remove upside risk.
3. **Add a stop / event filters** (the open-IC work found a 2× stop + opening-range/FOMC gates
   turned a breakeven IC positive; the same levers are untested for this structure).
