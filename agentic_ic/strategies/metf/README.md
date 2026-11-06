# METF — Multiple Entry Trend Following (imported)

Origin: `options_api/app/strategies/metf_strategy.py` and `options_api/app/backtesting/metf_backtester.py`, copied verbatim on 2026-05-20.

## Files

- `metf_strategy.py` — pluggable strategy class (`METFStrategy`, `METFConfig`, signal/position dataclasses, journal exporter). Self-contained, depends only on stdlib + `pandas` + `numpy`.
- `metf_backtester.py` — historical 1-minute bar backtester (`METFBacktester`, `METFResultsReporter`). Also includes a CLI: `python -m agentic_ic.strategies.metf.metf_backtester --help`. Self-contained, no `app.*` imports.

## State on arrival

**Not production-ready.** The backtester uses two synthetic shortcuts:

1. `get_credit_for_em_distance()` returns `random.uniform(1.25, 2.50)` — credit is not derived from real option quotes.
2. `simulate_spread_outcome()` uses a heuristic time-decay model — no real bid/ask consulted for stop-loss checks.

These were the reason for the move: `0dte_bot/data/cbbo_cache/` has the per-minute SPX bid/ask data needed to replace both, and `options_api`'s option-price DB does not.

## Why it was moved here

See [`docs/metf_signal_first_look_2026-05-20.md`](../../../docs/metf_signal_first_look_2026-05-20.md) for the full evaluation, including:

- Data inventory comparison (options_api vs 0dte_bot)
- cbbo_cache strike-band feasibility by year (default METF params are fully covered on ~125 days in 2025-2026; ~700 days become usable with tighter params)
- First-look backtest results on synthetic-credit run: **late entries dominate** (14:30 = 74% win rate, 14:45 = 79%); **bearish-side outperforms** (CCS 70% vs PCS 62%).

## Open work

1. Replace synthetic credit with real `cbbo_cache` quote lookups in `METFBacktester`.
2. Replace heuristic stop-loss model with real spread-mark path from `cbbo_cache`.
3. Validate the late-entry edge on real-price data over the 2025-2026 sample.
4. Test parameter-relaxed variants (0.5% EM short, 10-pt width) on 2022-2024 days.
5. Check `cbbo_cache_es` band width — strategy may be more cleanly testable on ES/MES.
6. Wire `METFStrategy` into `BotOrchestrator.run_scheduled` / `DecisionEngine` for live execution alongside the existing iron-condor pipeline.

## Known bug (pre-existing)

11 trades in the first-look run were labeled `EXPIRATION_LOSS` with positive avg P&L (+$60). When the spread expires ITM by less than the credit received, the trade is actually a partial win — the classifier in `simulate_spread_outcome` should label it `EXPIRATION_WIN`.
