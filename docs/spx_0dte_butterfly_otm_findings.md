# SPX 0DTE OTM Butterfly — Backtest Findings

> Status: PENDING DATA RUN. This document is filled after the wider-window
> backfill (Task 6) and the Phase 1/2 runs (Task 9).

## Method
- Spec: `docs/superpowers/specs/2026-06-03-otm-butterfly-backtest-design.md`
- Cache: `data/cbbo_cache_fly_10min` (±250pt, ~940 days 2022-05→2026-03)
- Core: `src_v2/backtest/butterfly_core.py`; eval: `scripts/analytical_butterfly_eval.py`

## Primary question
Does the OTM 0DTE butterfly (body 1σ, ≥9:1) have positive **net-of-cost OOS**
expectancy with positive skew, regardless of side?

## Results
_TBD — headline vs with-costs table, per-regime breakdown (with Chaos N),
direction-edge vs coin-flip, exit-model comparison, WF-CV OOS pooled mean._

## Verdict
_TBD_
