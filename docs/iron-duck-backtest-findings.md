# Iron Duck (Broken-Skewed Iron Condor) — SPX Backtest Findings

**Strategy source:** `docs/The-Iron-Duck-Options-Spread/The-Iron-Duck-Options-Spread-Strategy.md`
(NavigationTrading "Iron Duck" course; instructor's own Option Omega results on deck p.31-38).

**Status:** COMPLETE (2026-06-12).

## TL;DR — verdict

1. **As taught, the Iron Duck is REFUTED on SPX 2022→2026.** The instructor's
   headline (~296% ROC, 7 DTE, 100%-of-credit stop) is nowhere: at frictionless
   *mid* fills the best as-taught cell earns ~8-31%/yr on margin and 7 DTE ≈ 0;
   under an honest fill model (entry mid+25% of half-spread, stop exits at the
   natural) **every as-taught cell loses money** (−$6k to −$24k over 4.45y).
   Walk-forward, the as-taught constants lose −$15k (Sharpe −0.53) on the
   untouched test tier.
2. **The taught 100%-credit stop is the killer.** On a defined-risk structure the
   stop adds no protection (max loss is capped at entry) but pays the spread on
   every triggered exit and locks in vol-spike marks that usually recover by
   expiry. Across the whole 60-combo grid, *stop-100 is negative nearly
   everywhere; no-stop dominates at every DTE* — the exact MEIC lesson again.
3. **The only robust survivor is a structural plateau, not a tuned cell:
   14 DTE + as-taught strikes (65Δ/+1 call, 15Δ/5Δ puts) + NO stop + hold to
   cash settlement.** All ten 14-DTE no-stop cells are positive (mean +$21k);
   the family is flat across call-delta (0.60/0.65) and put-short-delta
   (0.10–0.20). The as-taught-deltas member earns **+$29k over 4.45y at entry
   fill k=0.25 (≈18%/yr on $36k margin, 91% win)**, survives entry k=0.5
   (+$19k), makes money through the COVID stress window (+$7.8k, Jan–Aug 2020)
   — but carries true tail risk: worst trade −$15k, maxDD −$25k (~68% of
   margin), and 2022 as a whole lost −$19k.
4. **Periodic re-optimization adds nothing reliable.** WF method validation
   (60 combos, 130/65d folds): 8 distinct winners in 10 folds (churn =
   curve-fitting per docs/wf-cv.md), stitched method P&L +$5.3k / Sharpe 0.20
   (noise). At 260/130d: +$12.3k / Sharpe 0.72 over only 3 folds, and the
   WFE gate *hurt*. Freeze the structural variant or skip the strategy; don't
   chase parameters.
5. Bottom line: the Iron Duck's true edge (if any) is **modest premium harvest
   with rare large left-tail losses** — roughly the opposite of the marketed
   "no-risk beak, 296% ROC, loss maybe once a year" pitch. The no-stop 14-DTE
   variant is tradeable but unremarkable: ~18%/yr on committed margin with
   crash-year drawdowns comparable to a full year's profit.

## What was built

| Piece | Path |
|---|---|
| EOD chain backfill (ThetaData VALUE, SPXW 2020-01 → present) | `scripts/backfill_iron_duck_eod.py` → `data/iron_duck_eod/SPXW/` |
| Backtest engine | `src_v2/backtest/iron_duck.py` |
| Runner / as-taught sweep | `scripts/backtest_iron_duck.py` → `results/iron_duck_spx/` |
| WF combo-log generator | `scripts/gen_duck_combo_logs.py` → `results/iron_duck_wf/` |
| Tests (synthetic Black-76 chains) | `tests/test_iron_duck_backtest.py` |

## Methodology

- **Data**: ThetaData EOD NBBO (16:00 close) for every SPXW expiration 2020-01-02 →
  present, 35-day lookback per expiry. The instructor's window (Jul 2014 – Jul 2019)
  is *below the VALUE-tier history floor*, so his exact period cannot be reproduced.
- **⚠ SPXW NBBO gap (Sep 2020 → Dec 2021)**: ThetaData has *no CBOE index-option
  quote history* in that span on this tier (bid/ask all zero; volume present). SPY
  (OPRA) is unaffected — it is an index-options feed limitation, not a backfill bug.
  Evaluation therefore uses two clean windows: **2022-01 → present (primary)** and
  **2020-01 → 2020-08 (COVID stress sample)**; the gap is excluded.
- **Greeks**: implied forward per (date, expiry) via put-call parity (validated ±1 pt
  against SPX closes, including the 2020-03-23 trough); Black-76 IV per strike implied
  from the OTM side only (ITM EOD mids are unreliable); deltas from the smile.
- **Construction** (as-taught): SELL ~65Δ call, BUY next strike up (cheat sheet) or
  ~62Δ (OO config); SELL ~15Δ put, BUY ~5Δ (OO) / ~10Δ (cheat sheet) put.
- **Lifecycle**: sequential — one position; re-establish next trading day after close.
  Entry/marks/fills at EOD NBBO (the stop is the instructor's *manual alert* stop, so
  a close-evaluated stop is faithful; intraday touches are invisible by design).
- **Exits**: stop at 100% of credit (loss ≥ credit on mid mark, filled per fill model),
  else cash settlement at expiry-day parity forward (PM settlement). Optional beak
  early-exit (P(return to head) < 10%).
- **Fills**: `fill_k` = 0 (mid) → 1 (natural: cross the full half-spread).
  Costs: $1.00 commission + $0.15 fees per contract per side.
- **POP / constraint gates** (cheat sheet, optional): POP ≥ 85% (lognormal at the BE
  strike's vol), credit ≥ call width (no upside risk), theoretical T+0 loss at the
  downside BE < credit.

## Instructor's claims to test (his OO runs, SPX 2014-2019, 100% stop)

| DTE | ROC | % win | Max DD |
|---|---|---|---|
| 21 | 165% | 88% | −2.68% |
| 14 | 293% | 90% | −2.01% |
| 7 ⭐ | 296% | 80% | −1.82% |

Claimed floor economics: ~50-60% of trades expire in the beak; full downside breach
"maybe once a year".

## Early observations (partial data, Jan–Apr 2020)

1. **The "no upside risk" condition is fragile under realistic fills.** With natural
   fills and the 10Δ cheat-sheet long put, the net credit frequently drops below the
   5-pt call width → the beak goes *negative* → POP collapses (≈33% avg in the
   sample). The OO config (5Δ long put) collects more credit and mostly keeps the
   beak ≥ 0. Confirms spec ambiguity #3: credit ≥ call width must be a hard gate.
2. **The cheat-sheet's third gate (T+0 loss at BE < credit) fails routinely in
   low IV** (Jan-2020 entries: BE-loss ≈ 7.7 pts vs credit ≈ 5.8) — "adjust until
   true" would skip many days or push strikes well off the taught deltas.
3. COVID window (Jan–Apr 2020) is heavily negative across the whole as-taught grid,
   as expected for a short-premium structure with a big short-put wing.

## Results

### As-taught grid (sequential, 100% stop), primary span 2022-01 → 2026-06 (4.45y)

Mid fills (`fill_k=0`, the Option-Omega-style fantasy):

| DTE | put long | n | net | ROC/yr | win% | exit mix |
|---|---|---|---|---|---|---|
| 5 | 5Δ | 260 | −$2.6k | −1% | 82% | 39 stops |
| 7 | 5Δ | 211 | +$5.3k | +3% | 79% | 39 stops |
| 10 | 5Δ | 155 | +$17.8k | +8% | 76% | 32 stops |
| 14 | 5Δ | 123 | +$19.1k | +8% | 75% | 29 stops |
| 21 | 5Δ | 86 | −$4.6k | −2% | 72% | 24 stops |
| 10 | 10Δ | 149 | +$19.9k | +31% | 82% | 21 stops |

Natural fills: **every cell −$37k … −$76k.** Honest fills (entry k=0.25, stop
exits k=1.0): best cells −$6k … −$24k. Fill-sensitivity breakeven sits between
k=0.25 and k=0.5 — i.e. the entire mid-fill "edge" is smaller than the spread.

Win rates (75-86%) and the many-small-wins / few-large-losses shape match the
course's claims; the *economics* don't. Expected ~50-60% of trades expiring in
the beak is roughly observed; "full breach maybe once a year" is not (2022 alone
had several).

### Stop ablation (60-combo grid, honest fills, full span)

| DTE | no stop (mean net) | stop 100% (mean net) |
|---|---|---|
| 7 | −$8.6k | −$13.7k |
| 10 | +$0.6k | −$8.4k |
| 14 | **+$21.1k** | −$9.3k |

Top-10 full-span combos are *all* no-stop; 7 of 10 are 14 DTE. The stop turns
the best family from +$21k mean to −$9k mean: it pays the spread to realize
vol-spike marks that mostly mean-revert by settlement.

### Recommended variant (structural, not optimized)

`14 DTE, SELL 65Δ call / BUY +1 strike, SELL 15Δ put / BUY 5Δ put, NO stop,
hold to cash settlement, sequential 1-lot` (the as-taught construction at the
instructor's own "best bang for the buck" DTE, minus the stop):

| metric | value |
|---|---|
| Net (2022-01→2026-06, entry k=0.25) | +$29.1k |
| ROC/yr on $36k max margin | ~18% |
| Win rate | 91.4% (105 trades) |
| Worst trade / maxDD | −$15.1k / −$24.7k |
| By year | 2022 −$18.8k · 2023 +$15.2k · 2024 +$6.8k · 2025 +$10.8k · 2026 YTD +$15.0k |
| Entry k=0.5 robustness | +$18.9k (11.7%/yr) |
| COVID stress (2020-01→08, k=0.25) | +$7.8k, 94% win, worst −$8.2k |

The neighboring cells (put short 0.10–0.20, call 0.60–0.65, all no-stop 14 DTE)
range +$14.6k…+$43.9k — a plateau, so the as-taught-deltas member is reported
rather than the grid max (per docs/wf-cv.md plateau-over-best-cell).

### Walk-forward METHOD validation (docs/wf-cv.md; 60 combos; honest fills)

Stitched untouched test tier, baseline = as-taught `d7_c0.65_ps0.15_pl0.05_st1`:

| policy | 130/65d (10 folds) | 260/130d (3 folds) |
|---|---|---|
| deploy fold winner | +$5.3k, Sharpe 0.20 | +$12.3k, Sharpe 0.72 |
| WFE-gated (fallback as-taught) | −$0.7k, Sharpe −0.03 | −$13.0k, Sharpe −0.71 |
| as-taught constants | **−$15.2k, Sharpe −0.53** | **−$13.1k, Sharpe −0.65** |

Winner churn: 8 distinct winners / 10 folds (doc: heavy fluctuation =
curve-fitting). The no-stop 14-DTE family won half the folds but re-optimization
delivered only noise-level OOS gains. **Method verdict: as-taught constants are
refuted; quarterly re-optimization is NOT validated for this strategy — the only
defensible deployment is the frozen structural variant above, or nothing.**

### Caveats

- EOD-only marks: entries at the 16:00 NBBO; intraday stop touches are
  invisible (faithful to the taught *manual alert* stop, but a resting-order
  implementation would differ).
- 2020-09→2021-12 unobservable (SPXW quote-history gap on this data tier).
- The instructor's 2014-2019 regime (low vol, pre-daily-expirations) is not
  testable here; his numbers may have been real *then* — they do not transfer.
- Sequential 1-lot, no laddering/portfolio effects; margin = max wing − credit.
- No-stop variant's risk is genuinely fat-tailed: sizing must assume the
  −$15k single-trade loss and a −$19k year are repeatable.
