# EM-Relative MEIC Strike Selection — Design Spec

**Date:** 2026-05-21
**Supersedes entry logic of:** `docs/superpowers/specs/2026-05-20-meic-strategy-design.md`

## Problem

`MeicStrategy` selects strikes by **absolute** dollar targets calibrated to the
2025 SPX regime (~6700 index): fixed credit bands (AM $1.25–1.65, PM $0.80–1.30)
and fixed 80–100pt wings. These do not transfer across index levels / vol
regimes. A 2023 sanity backtest (spot ~4284, low vol) produced **0 trades**
(`no_meic_trades`): far-OTM 0DTE legs are bidless in low vol, so 80–100pt-wing
spreads can't be priced, and computed credits ($25–$100, near-money) never land
in the AM band. Net: MEIC can only trade its native 2025 regime, so it cannot be
backtested through the 2022 bear even after a CBBO backfill.

## Goal

Make MEIC's strike selection **expected-move-relative** so it trades sensibly
across regimes (low/high vol, any index level), works identically in live and
backtest with **no Black-Scholes** (the backtest CBBO chain has no delta), and
reuses the proven delta-free EM pattern from `OrbEmIronCondorStrategy`.

Out of scope: MEIC's exit logic (scale-out tiers, stops, abandon-long),
order routing, position monitoring, and the CBBO backfill tooling — all unchanged.

## Core design

EM = the ATM straddle price (`call_mid + put_mid` at the nearest-to-spot listed
strike), computed from the option chain — delta-free, identical in live/backtest.
This mirrors `OrbEmIronCondorStrategy._compute_em_from_chain` and matches the
project's validated finding that the ATM straddle (not VIX) is the right 0DTE EM.

`step` = the listed strike increment (`strike_step_pts`, 5 for SPX). All computed
prices snap to a multiple of `step` so they land on real listed strikes.

Strike selection (one direct construction, not a credit-band sweep):

```
EM          = atm_call_mid + atm_put_mid          # from chain
short_call  = ceil((spot + short_em_mult*EM)/step)*step
short_put   = floor((spot - short_em_mult*EM)/step)*step
wing_pts    = max(step, round(wing_em_frac*EM/step)*step)
long_call   = short_call + wing_pts
long_put    = short_put - wing_pts
```

Look up mids for all four legs. If any leg is missing or bidless
(`mid_price` is None), skip the slot (no candidate). Otherwise:

```
call_credit = sc_mid - lc_mid
put_credit  = sp_mid - lp_mid
```

Quality gate (regime-neutral, replaces the absolute credit bands):

```
each side: credit / wing_pts >= min_credit_width_ratio
```

If the gate passes, emit one `LiveTradeSetup` (strategy_type `"meic"`), with the
per-tranche `qty` from `tranche_sizing` stamped on every leg (unchanged).

This is regime-relative by construction: low vol → small EM → tighter strikes
and narrower wings (long leg stays priced); high vol → everything widens
proportionally.

## Changes by file

### `src_v2/strategies/meic.py` (entry only)

**`configure`** — remove absolute keys, add EM-relative ones:

| Removed | Added (default) |
|---|---|
| `am_min_credit`, `am_max_credit` | `short_em_mult` (1.0) |
| `pm_min_credit`, `pm_max_credit` | `wing_em_frac` (0.5) |
| `wing_widths` | `min_credit_width_ratio` (0.08) |

Kept unchanged: `am_pm_cutoff` (still used by scale-out exits), `stop_pct_of_total_credit`,
`am_scale_out_tiers`/`fractions`, `pm_scale_out_tiers`/`fractions`,
`abandon_long_below`, `credit_equality_threshold`, `default_tranche_qty`,
`tranche_sizing`. `_now_fn` clock injection unchanged.

**`search_candidates`** — rewritten to the direct EM construction above. Adds a
helper `_compute_em(chain, spot)` (port of `_compute_em_from_chain`). Emits a
single candidate (or zero). `extra` carries: `em_pts`, `wing_pts`, `call_credit`,
`put_credit`, `credit_diff`, `total_premium`, `short_call_mid`, `short_put_mid`,
`qty`, `call_credit_width`, `put_credit_width`. (`compute_stop_loss` already reads
`call_credit`/`put_credit`/`short_*_mid`, so its contract is preserved.)

**`validate`** — checks become:
- `strike_order_call` (`long_call > short_call`), `strike_order_put` (`short_put > long_put`)
- `wing_symmetry` (call wing == put wing — auto-satisfied, kept as a guard)
- `credit_to_width_call` / `credit_to_width_put` (`>= min_credit_width_ratio`)
- `qty` (`>= 1`)

Removed: `wing_range [80,100]`, `call_credit_band`/`put_credit_band`.

### Unchanged in `meic.py`
`build_order`, `compute_stop_loss`, `build_stop_orders`, `check_exit`,
`compute_trailing_stop`, `leg_pairs`, `create_engine`.

### `src_v2/backtest/engine.py`
`_build_chain` currently spans **spot ± 200pt** — too narrow for high-vol EM
strikes. Widen to cover the EM-relative reach: span = `spot ± (short_em_mult*EM +
wing + buffer)`, or simply a larger fixed half-span (e.g. ±400pt). Since
`_build_chain` runs before EM is known, use a generous fixed half-span
(configurable, default 400pt) sufficient for the largest expected EM reach.
`_process_day_meic` and `_simulate_meic_side` are otherwise unchanged.

### `scripts/run_v2_wf_cv.py`
`MEIC_PARAM_GRID` changes from `stop_pct × wing_widths` to:

```python
MEIC_PARAM_GRID = {
    "strategy.short_em_mult":            [0.8, 1.0, 1.2],
    "strategy.wing_em_frac":             [0.4, 0.5, 0.6],
    "strategy.stop_pct_of_total_credit": [0.65, 0.70, 0.75],
}   # 27 combos
```

### `config/config_meic_spx.yaml`
Replace the credit-band + `wing_widths` keys with `short_em_mult`,
`wing_em_frac`, `min_credit_width_ratio`. Keep all other sections.

## Defaults (starting points; to be WF-CV-tuned post-backfill)
`short_em_mult = 1.0`, `wing_em_frac = 0.5`, `min_credit_width_ratio = 0.08`.

## Testing (TDD)

Unit (`tests_v2/test_meic_strategy.py` — update the absolute-band tests):
1. `_compute_em` returns `atm_call_mid + atm_put_mid` from a chain; None if ATM legs missing.
2. `search_candidates` low-EM chain (small EM, spot ~4284) → valid candidate, priced legs.
3. `search_candidates` high-EM chain (large EM, spot ~6700) → valid candidate; strikes scale up vs the low-EM case. **(Regime test — the case the original failed.)**
4. `search_candidates` stamps tranche qty on legs + `extra["qty"]` (unchanged behavior).
5. `search_candidates` skips when a wing leg is bidless (no candidate).
6. credit-to-width gate: a setup below `min_credit_width_ratio` is rejected.
7. `validate`: strike-order, wing-symmetry, credit-to-width, qty cases.
8. Existing exit/stop/scale-out tests unchanged and still pass.

Backtest (`tests_v2/backtest/test_meic_backtest.py`):
9. A synthetic **low-EM** day now produces trades (regression against the 2023 zero-trade failure).
10. Existing worthless/stop/scale-out/cost tests still pass (configs updated to EM-relative params).

## Validation gate (definition of done)
Re-run the 2023 sanity week (`data/cbbo_cache_meic`, already fetched): MEIC must
produce a non-zero, sensible number of trades. That confirms regime portability
before spending the ~$7 on the full 2022–2025 backfill.

## Risks / notes
- `min_credit_width_ratio` default (0.08) is a guess; if it's too strict it
  reproduces the zero-trade failure in some regimes — the regime unit test (3)
  and the 2023 validation gate guard against this.
- Widening `_build_chain` increases per-slot work slightly; negligible.
- EM-relative wings can snap to a single step in very low vol; `wing_pts >= step`
  prevents zero-width spreads. Very narrow wings have higher per-contract risk —
  acceptable and regime-appropriate; the credit-to-width gate bounds risk/reward.
- This replaces the absolute entry in place; the 2025-validated absolute version
  remains recoverable via git history.
