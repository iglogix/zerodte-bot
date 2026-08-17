# Plan — SPX 0DTE expected-move fade backtest (0dte_bot)

Spec: `docs/spx_0dte_em_fade_spec.md`.

**Supersedes** an options_api-based plan that would have rebuilt this engine from
scratch and spent ~$2.75 on Databento. All of it already exists here.

## Global constraints

- **No new market-data spend.** `data/cbbo_cache_fly_10min/` covers
  2022-05-11 → 2026-06-11 at minute × 5-pt-strike resolution, both sides. If a
  needed (day, minute, strike) is missing, that is a SKIP recorded with a
  reason, never a synthetic fill and never a fetch-on-miss inside the loop.
- **Reuse, do not rebuild.** `CbboStore` (`lookup`, `atm_straddle`,
  `derive_spot`, `settlement_forward`), `BacktestOrchestrator`, `LiveStrategy`
  base (`search_candidates` / `validate` / `check_exit` / `compute_stop_loss` /
  `compute_trailing_stop`), `src_v2/flux/iv_compute.compute_iv_rows` for
  IV+delta. `src_v2/strategies/orb_em_directional.py` is the structural template.
- **`tests_locked/` is immutable** (enforced by the pre-commit hook). New tests
  go in `tests/`.
- Real CBBO bid/ask only. Mid is a statistic, not a price anyone trades at.
- TDD: test first, watch it fail, then implement.

## Task 1 — `src_v2/strategies/em_fade.py`

New `LiveStrategy` subclass, modelled on `OrbEmDirectionalStrategy`.

Config (`config/config_em_fade_spx.yaml`), all tunable:
`em_multiplier` (band width, default 1.0), `em_source` (atm_straddle),
`band_anchor` (open), `entry_cutoff_et` (13:00), `entry_start_et` (09:35),
`target_delta` (0.20), `delta_min`/`delta_max` (0.10/0.25), `width_pts` (10),
`tp_pct_of_credit` (0.50), `sl_mult_of_credit` (1.00), `trail_enabled` (false),
`trail_arm_pct` (0.25), `trail_stop_pct` (0.10), `eod_flat_et` (15:50),
`max_trades_per_day` (1), `slippage_per_spread`, `commission_per_spread` (2.60).

Behaviour:
- EM = ATM straddle mid at `entry_start_et` via `CbboStore.atm_straddle`.
- Bands = `open ± em_multiplier × EM`, open = parity spot at 09:31.
- Scan minutes forward; **first** band touch inside the window wins. Upper touch
  → sell CALL spread (fade); lower touch → sell PUT spread (fade).
- Short strike: among listed strikes on the traded side, pick the one whose
  |delta| is nearest `target_delta`, where delta comes from `compute_iv_rows`
  (per-strike IV from that strike's own mid). Reject if outside
  `[delta_min, delta_max]`. Wing = short ∓ `width_pts`; reject if unlisted.
- Exits, evaluated minute by minute on real quotes: TP limit at
  `0.5 × credit` (fills when the **ask** reaches it), stop-market at
  `2.0 × credit` (triggers on the package price, fills by **crossing**),
  optional trail (once unrealised ≥ `trail_arm_pct`, move the stop to
  `+trail_stop_pct` of credit), EOD flat at `eod_flat_et`.
- pnl confined to `[−max_loss, +credit]`; out-of-band marks skipped as glitches.

**Done when:** `tests/test_em_fade.py` covers — no-touch day, both-band day
(first touch wins), cutoff enforced, delta targeting against a constructed
chain, missing-wing rejection, each of the four exit paths firing, the stop
filling worse than it triggers, and a glitch mark being skipped.

## Task 2 — sanity gate (blocking; no result is read before it passes)

- parity spot (`derive_spot`) vs SPY-derived spot agree within tolerance on ≥30
  sampled days; log the distribution rather than asserting a single number
- strike ladder is centred enough that the traded strike is interior, not at the
  cache edge, on ≥99% of trade days — an edge hit means the ladder truncated the
  choice and biases strike selection
- count and log every skip reason (no touch, no quote, delta out of range,
  wing unlisted) — silent skips are how a sample gets curated
- band-shift sensitivity ±0.05% (Ruling: touch detection is a threshold crossing)
- smoke-run one month end to end before the full span

## Task 3 — measure

Full span 2022-05-11 → 2026-06-11. Depth sweep `f ∈ {0.5, 0.6, 0.7, 0.85, 1.0}`
× `trail ∈ {off, on}`. Report per-year, put-vs-call split, VIX-regime split,
touch-time bucket, hold-time distribution. Claim-by-claim table against C1–C9.

## Task 4 — controls

With-trend mirror (same entries, inverted side), unconditional 10:00 entry,
seeded random entry minute, buy-and-hold SPX. The mirror is the key one.

## Task 5 — walk-forward

`scripts/run_wf_method_strategy.py` ladder as used by the ORB EM studies
(train/test 100/50, 130/65, 160/80), frozen-vs-reoptimised, winner-stability
check. Grid-corner and tiny-max-loss winners are leverage artifacts, not alpha.

## Task 6 — verdict

Seven structural moves, batched audit pass, council at elevated effort. Must
answer: **bound or measurement?** Must reconcile with the existing
`orb_dir_fade_compare` result — agreeing or disagreeing, explicitly. Write
`docs/spx_0dte_em_fade_findings.md` and save a memory entry.
