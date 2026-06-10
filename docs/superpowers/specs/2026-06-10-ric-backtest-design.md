# 0DTE Reverse Iron Condor (RIC) — Backtest + Live Strategy Design

**Date:** 2026-06-10
**Spec source:** `docs/0DTE-Reverse-Iron-Condor-Strategy.md` (§15 is the encode-ready spec)
**Pattern source:** the Power Hour build (`src_v2/backtest/power_hour.py`,
`scripts/backtest_power_hour.py`, `src_v2/strategies/power_hour.py`) — same
instructor, inverse structure, same data layer.

## Approaches considered

1. **Dedicated engine + runner mirroring Power Hour** *(chosen)* — `RicEngine`
   over `CbboStore` pickles with SPY×10 bars for the timeline. The RIC is a
   debit/no-stop strategy; the doc explicitly warns (§14.9) against letting
   short-premium stop/credit logic leak in, so a clean module is safest.
2. Extend `PowerHourEngine` with a "RIC version" — rejected: opposite
   mechanics (debit vs credit, no stop, gap gate, morning entry) would tangle
   an engine the doc says must stay separate.
3. Reuse `WindowEntryEngine`/`engine.py` orchestrator — rejected: built for
   credit condors with stop management; more adaptation than a fresh ~400-line
   engine.

## Data

- **Quotes:** `data/cbbo_cache_fly_10min` — 952 days, 2022-05-11 → 2026-02-27,
  1-min 09:31–15:59, ~100 strikes spanning open±250pts. Covers 50Δ call,
  35Δ put, +20/−35 wings on all days. Optional overlay `cbbo_cache_meic`
  (extra strikes 11:00–15:49) via the `MergedCbboStore` pattern.
- **Timeline:** `data/spy_1m.csv` ×10 via `BarLoader` (as Power Hour).
- **VIX:** `data/vix_daily.csv` for regime buckets.
- **FOMC:** `_FOMC_DATES` in `src_v2/backtest/engine.py` (2021→2026).

## Key modeling decisions (resolving §14 of the spec)

| Open item | Decision |
|---|---|
| Instrument/scale | SPX, 20/35pt wings (matches instructor's backtest numbers) |
| Wings: points vs width | Fixed point offsets from the delta-selected longs, snapped to nearest cached strike strictly beyond; record realized widths |
| Gap reference | **Parity-based, pure SPX:** prior day's `settlement_forward()` (15:59) vs `derive_spot()` at the gap-check minute (09:31). Fallback to SPY×10 bar open vs SPY×10 prior close when prior settle is unavailable (scale-consistent within the fallback; never mix sources). Record `gap_source`. |
| Gap threshold | 0.25% flat default; per-weekday override map in config (test Mon 0.1%) |
| "Sets up" filter | Directly computable, no IV proxy: upside max profit = `call_width − debit`; require `≥ setup_min_upside_pct × debit` (default 0.25) |
| Entry time | Config `entry_time`, default 09:33; sweep 09:32/09:35/09:40/09:45 |
| Strike selection | 50Δ call / 35Δ put via BS delta implied from quote mids (reuse the Power Hour `_bs_delta` bisection, t = minutes to 16:00) |
| TP | 25% of debit default (sweep 30%); resting limit modeled as fill at the limit level when combo mid crosses it (`tp_fill: limit`), variant `mid` |
| Scale-out variant | ½ @ +20%, ¼ @ +30%, ¼ runner to expiry — config-gated |
| No stop | Combo held; max loss = debit + costs. Combo value floored at 0. |
| EOD | `final_exit_mode: expire` (settlement intrinsic at parity forward) default — matches "let it expire"; `close` variant exits at 15:55 mids |
| Skips | gap_down, gap_below_min, fomc_day, setup_filter, no_quotes / dislocated legs (same dislocation guard as MEIC/PH) |
| Costs | $1/contract/leg open + close (closed legs only; expired legs free), slippage $0.10/spread/side ×2 spreads, like Power Hour |
| Reporting | Win rate, PF, Sharpe (daily), by year, by VIX bucket, by weekday, by gap size, exit reasons, **loss distribution as % of debit** (the §6 edge test: partial vs full-debit losers) |

## Components

1. **`src_v2/backtest/reverse_ic.py`** — `RicResult` dataclass + `RicEngine`
   (config-driven, `set_cbbo_store`, `run(days_data)` walking days in order so
   the prior-day settle is available for the gap). Minute-walk per day:
   compute combo value `(LC−SC)+(LP−SP)` mids; TP / scale tiers / EOD.
2. **`config/config_ric_spx.yaml`** — strategy + options costs + backtest paths
   (naming consistent with `config_power_hour_spx.yaml`).
3. **`scripts/backtest_reverse_ic.py`** — loader, CLI overrides
   (`--entry-time --tp --gap-min --scale-out --final-exit-mode --start --end`),
   report, trades CSV to `results/ric_spx/trades.csv`. Reuses
   `MergedCbboStore` (lifted into `src_v2/backtest/cbbo_store.py` so both
   scripts share it).
4. **`src_v2/strategies/reverse_ic.py`** — `ReverseIcStrategy(LiveStrategy)`
   on `ScheduledEngine`, one entry window. Gap + setup gates in
   `search_candidates` (prior close via deps' market data captured at
   `create_engine`); `check_exit` = TP only (no stop); EOD handled by
   PositionMonitor config (or expire). Registered in `src_v2/main.py`;
   `config/config_ric_live.yaml` (dry_run default).
5. **Tests** — `tests/test_ric_backtest.py` (synthetic chains: gap gates incl.
   weekday override, FOMC, delta selection, wing snapping, setup filter, TP
   path, pin-day full loss at expiry, close-mode exit, scale-out, costs) and
   `tests/test_ric_live_strategy.py` (mirrors PH live tests).

## Success criteria

- Locked + new test suites pass; backtest runs over the full 952-day window.
- Report answers the doc's §6/§14.6 question: is avg winner > avg loser given
  the loss distribution, and how does the edge bucket by VIX (his claim: dies
  sub-15)?
