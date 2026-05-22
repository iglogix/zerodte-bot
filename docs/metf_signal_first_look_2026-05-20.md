# METF Strategy — Signal First-Look & Transfer Decision

**Date:** 2026-05-20
**Status:** Recommendation: move strategy to `../0dte_bot/` for proper backtesting against real option quotes.

## Strategy summary

METF (Multiple Entry Trend Following) is a mechanical 0DTE SPX credit-spread strategy.

- **Signal:** 20/40 EMA crossover on 1-min underlying (0.1% min separation gates noise)
- **Direction:** Bullish trend → sell put credit spread; Bearish → sell call credit spread
- **Entries:** Six scheduled times ET — 12:30, 13:00, 13:30, 14:00, 14:30, 14:45
- **Structure:** Short strike at ~Expected Move OTM; long strike 20 pts further OTM
- **Risk:** 1× stop loss (exit when spread doubles); hold to expiration otherwise
- **Filters:** Skip FOMC rate-decision and minutes-release days

Code: [app/strategies/metf_strategy.py](../app/strategies/metf_strategy.py)
Backtester: [app/backtesting/metf_backtester.py](../app/backtesting/metf_backtester.py)

## Why move it to 0dte_bot

The current backtester in `options_api` has two synthetic shortcuts that make its dollar P&L untrustworthy:

1. `get_credit_for_em_distance()` returns `random.uniform(1.25, 2.50)` — credit is not derived from any real option quote.
2. `simulate_spread_outcome()` uses a heuristic time-decay model — no real bid/ask is consulted for the stop-loss check.

`options_api`'s `backtest_data/option_prices.db` is EOD-only (DTE-0 has 384 rows across 8 years) — there's no path to fix the synthetic shortcuts here.

`0dte_bot` has `data/cbbo_cache/` — 837 day-files of intraday SPX option bid/ask quotes at 1-min resolution, 2022-04-18 → 2026-03-20. That is exactly the data shape METF needs.

## Data inventory comparison

| Data needed for METF | options_api | 0dte_bot |
|---|---|---|
| SPX 1-min underlying bars | 4 days | Same as options_api (SPY proxy); plus `data/spy_1m.csv` (786K rows) |
| SPY 1-min bars | 757 days (2023-01 → 2025-12) | ~3.5 years (2022-05 → recent) |
| **SPX intraday option bid/ask** | **None** | **837 day-files of minute-bar bid/ask quotes** |
| ES intraday option bid/ask | None | `data/cbbo_cache_es/` — 832 day-files, same window |
| GLBX 1-min futures | None | 295MB Databento CSV, 5 years |

## cbbo_cache strike-band feasibility

The cache schema is `dict[(HH:MM_str, strike, is_call_bool)] → (bid, ask)`. Coverage is **dense in time** but **narrow in strike** for most years:

| Year | Cached days | Avg band span | Avg strikes | Full-session % |
|---|---|---|---|---|
| 2022 | 143 | $100 | 21 | 43% |
| 2023 | 208 | $100 | 21 | 100% |
| 2024 | 211 | $100 | 21 | 100% |
| 2025 | 221 | $2,209 | 116 | 95% |
| 2026 | 54 | $6,451 | 248 | 100% |

For METF's default parameters (1% EM short strike, 20-pt-wide spread), the long leg falls outside the narrow ±$50 band on 2022-2024 days:

| Year | Days | Both sides fit | One side fits | Neither |
|---|---|---|---|---|
| 2024 | 211 | 0 | 0 | 211 |
| 2025 | 221 | 71 | 0 | 150 |
| 2026 | 54 | 54 | 0 | 0 |

**Conclusion:** at default METF params, ~125 days of real-price backtesting are immediately available. With tighter parameters (0.5% EM short, 10-pt-wide spread), the 2022-2024 days mostly become feasible, giving ~700 days.

837 cached days vs. ~1,000+ trading days in the span = **~20% of days are missing entirely** from the cache.

## First-look backtest (synthetic-credit run on SPY underlying)

To get a directional signal-quality read before investing in real-pricing integration, ran the existing backtester on options_api's cached SPY 1-min bars.

**Command:**
```
python -m app.backtesting.metf_backtester \
  --db backtest_data/underlying_bars.db --symbol SPY \
  --start 2023-01-03 --end 2025-12-31 \
  --output backtest_data/metf_spy_2023_2025.csv
```

**Results (synthetic credit, so $ amounts not trustworthy):**

- 755 trading days → 550 trades (~73% of days produced ≥1 entry)
- Win rate: 66% (363W / 187L)
- Profit factor 1.98, Sharpe 3.15, Max DD $3,776

**By entry time:**

| Entry | W | L | Win % | P&L |
|---|---|---|---|---|
| 12:30 | 50 | 38 | 57% | $1,918 |
| 13:00 | 55 | 34 | 62% | $4,330 |
| 13:30 | 60 | 33 | 65% | $5,094 |
| 14:00 | 49 | 36 | 58% | $2,040 |
| 14:30 | 68 | 24 | 74% | $8,300 |
| 14:45 | 81 | 22 | 79% | $11,826 |

**By trend direction:**

| Trend | W | L | Win % | P&L |
|---|---|---|---|---|
| BEARISH (CCS) | 197 | 84 | 70% | $22,246 |
| BULLISH (PCS) | 166 | 103 | 62% | $11,262 |

**Exit reasons:** 354 EXPIRATION_WIN, 185 STOP_LOSS, 11 EXPIRATION_LOSS

### What this tells us

**Trustworthy patterns (independent of the random credit):**
1. **Late entries dominate.** 14:30 wins 74%, 14:45 wins 79%, vs 57-62% for 12:30/13:00. Pure mechanical effect — less time to expire = less time to breach the EM-OTM short. This is the cleanest signal in the run.
2. **Bearish-side outperforms.** Over a SPY uptrend regime, CCS entries (70%) beat PCS entries (62%). The EMA filter seems to catch reversals more reliably than continuations. Worth confirming on real prices.

**Untrustworthy (artifacts of synthetic credit):**
- All absolute dollar figures, profit factor, and Sharpe — pinned by `random.uniform(1.25, 2.50)`, not by where the underlying actually ended up.
- Stop-loss frequency (34%) — the heuristic spread-value model likely fires stops at wrong moments.

**Bug noted:** 11 trades labeled `EXPIRATION_LOSS` have positive avg P&L (+$60). When spread expires ITM by less than the credit received, the trade is actually a partial win — classifier should flag it accordingly.

## Recommended next steps in 0dte_bot

1. **Re-implement the entry/exit simulator** to read real bid/ask from `data/cbbo_cache/`. Replace the random-credit and heuristic-decay paths.
2. **Validate the late-entry edge** on real prices over the ~125 day 2025-2026 sample where the cache fully covers METF's default spread.
3. **Test parameter-relaxed variants** (0.5% EM, 10-pt width) on 2022-2024 to extend the sample to ~700 days.
4. **Check `cbbo_cache_es` strike band** — if ES chains are wider, the strategy may be more cleanly testable on ES/MES (right-sized notional too).
5. **Wire into existing 0dte_bot pipeline**: METF's scheduled-entry shape fits `BotOrchestrator.run_scheduled` and `DecisionEngine.run_cycle`. Live-execution path through `RiskManager` and `ToolHandlers` is already in place.
