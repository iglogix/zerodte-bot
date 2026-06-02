# SPX 0DTE Jade Lizard Backtest — Design Spec

**Date:** 2026-06-02
**Status:** Approved design, pending implementation plan
**Source strategy:** [docs/spx_0dte_jl.md](../../spx_0dte_jl.md) (TastyTrade 9:30am Jade Lizard)

## 1. Goal

Backtest the TastyTrade "9:30am Jade Lizard" 0DTE SPX strategy on real SPX option
quotes and report how it performs, in a way that is **comparable with TastyTrade's
published results**: mid-price fills, no transaction costs in the headline number,
and position status evaluated **every 10 minutes** (not per-minute, not at sparse
hourly anchors).

This is a research/evaluation deliverable. Live-trading integration is out of scope.

## 2. Strategy definition (locked)

Entry once per day at the **09:30 ET** open. One position per day.

| Element | Rule |
|---|---|
| Expected move (EM) | `EM = ATM straddle mid at 09:30` = (ATM call mid + ATM put mid), ATM strike = nearest 5-pt strike to the entry spot. (Project convention; never VIX — see `feedback_0dte_em`.) |
| Short put | nearest 5-pt strike ≤ `spot − 0.25×EM` (`floor((spot − 0.25×EM)/5)×5`) |
| Long put | `short_put − 30` → **$30-wide** put credit spread |
| Short call | nearest 5-pt strike ≥ `spot + 0.25×EM` (`ceil((spot + 0.25×EM)/5)×5`) |
| Long call | `short_call + 10` → **$10-wide** call credit spread |
| Entry credit | `total_credit = (short_put_mid − long_put_mid) + (short_call_mid − long_call_mid)`, all at 09:30 mids |
| Exit (take profit) | **Whole-position.** Close all four legs the first 10-min anchor at which `combined_mark ≤ 0.50 × total_credit` (captured ≥50% of credit) |
| Exit (no TP) | **No stop-loss.** If 50% is never reached, hold all legs to expiry ("losers stay till expiration") |
| Contracts | 1 contract per leg, multiplier 100, for the per-lot headline; results also scaled to a $100K account |

**Symmetric-placement note (intentional):** both shorts sit at 0.25×EM from spot, with
asymmetric widths (put 30, call 10). This is a **defined-risk asymmetric iron condor**.
It only has the canonical jade-lizard "no upside risk" property on days where
`total_credit ≥ $10` (the call-spread width). We do **not** enforce that property by
moving the call strike; instead the report records the fraction of days on which it holds
and the realized upside-risk exposure. (User explicitly chose symmetric 0.25×EM over the
credit≥width placement.)

### Trade eligibility / skip rules

A day is **skipped** (and the reason counted) when:
- No 09:30 chain available, or ATM straddle legs missing → cannot compute EM.
- Any of the 4 selected legs has no quote at 09:30.
- `put_credit ≤ 0` or `call_credit ≤ 0` (un-priceable spread).

## 3. Status checks every 10 minutes

- **Anchors:** entry at `09:30`, then TP checks at `09:40, 09:50, … , 15:50, 16:00`
  (10-minute grid). The final anchor (`16:00`, falling back to the last available anchor
  ≥ 15:55) is also used for settlement.
- At each post-entry anchor `t`:
  `combined_mark(t) = (short_put_mid(t) − long_put_mid(t)) + (short_call_mid(t) − long_call_mid(t))`.
  The first `t` with `combined_mark(t) ≤ 0.50 × total_credit` closes the trade at that mark:
  `pnl = (total_credit − combined_mark(t)) × 100`.
- If no anchor triggers, **expiry intrinsic** at settlement spot `S`:
  - `put_debit = max(short_put − S, 0) − max(long_put − S, 0)`
  - `call_debit = max(S − short_call, 0) − max(S − long_call, 0)`
  - `pnl = (total_credit − (put_debit + call_debit)) × 100`
- A missing leg quote at an intermediate anchor uses a ±N-minute fallback (same as the
  existing evaluator's `_quote_mid`); if still missing, that anchor's TP check is skipped
  and evaluation continues to the next anchor. Expiry is always computable from `S`.

## 4. Spot / forward reference

The SPX reference price is derived from the **0DTE option chain itself**, not from
SPY×10 OHLC bars (directly addresses "don't infer the price from 1-min OHLC"):

- `forward(t) = K_atm + (call_mid(K_atm, t) − put_mid(K_atm, t))` (put-call parity at the
  ATM strike). For 0DTE the carry term is negligible, so `forward ≈ SPX spot`.
- **Entry strike placement** uses `forward(09:30)`.
- **Settlement** uses `forward(final anchor)`; if the final-anchor ATM legs are missing,
  fall back to `SPY×10` close from `data/spy_1m.csv`.

Rationale on "trades data": OPRA `trades` prints for OTM 0DTE legs are frequently absent
within a given minute, so a quote-mid is the sound, gap-free price source — which also
matches the "all fills at mid" instruction. We therefore price from `cbbo-1m` mids and do
not pull a separate trades feed.

## 5. Execution-cost model

Two separate notions of "cost":

1. **Execution cost (in the P&L):**
   - **Headline = zero-cost mid.** Fills at bid/ask midpoint; no slippage, no commissions.
     This is the TastyTrade-comparable number.
   - **Secondary sensitivity table** subtracts realistic costs (configurable, defaults:
     `slippage_per_spread = 0.10` on entry and on exit per side; `commission_per_contract_per_leg = 1.0`
     at open, `0.0` at close per TT SPX). Honors `feedback_slippage_liquidity`.
2. **Data cost (Databento):** ~$3–5 one-time to pull the quotes (see §6). Unrelated to the
   P&L; spent regardless.

## 6. Data plan — tailored SPXW backfill

`cbbo_cache_es` is **rejected**: it is options on **ES futures**, confirmed two ways —
`scripts/collect_es_options.py` pulls `NestedFutureOptionChain("ES")`, and its ATM forward
sits ~20 pts above the true-SPX 5-anchor cache on the same minute (the ES-vs-SPX basis).
A 20-pt forward error misplaces the $30 put side (which sits ~15 pts OTM at 0.25×EM).

The existing SPX caches are insufficient for 10-minute checks (`cbbo_cache_5anchor` has
only 5 time points/day; `cbbo_cache` is single-anchor/sparse). So we backfill a purpose-built
SPXW cache:

- **Tool:** adapt the SPX-direct MEIC backfill (`src_v2/backtest/meic_backfill.py` +
  `scripts/backfill_meic_cbbo.py`), which centers its strike window on a **SPY-bar-derived
  spot** and pulls `OPRA.PILLAR` `cbbo-1m`. We deliberately **avoid**
  `scripts/databento_backfill.py` because its ES-basis centering is circular for months with
  no existing CBBO (documented in `project_meic_backtest`).
- **Window:** strikes in `[center − 100, center + 100]` (5-pt grid) where `center = SPY×10`
  open at 09:30 — a **fixed** window for the whole day. Fixed (not rolling) centering is the
  key fix vs `cbbo_cache_es`: the entry-set legs stay covered all day even on trending days.
  ±100 pts comfortably covers `long_put ≈ spot−55` and `long_call ≈ spot+35` on high-EM days.
- **Anchors per day:** `09:30, 09:40, …, 15:50, 16:00`.
- **Date range:** full available history, ~950 trading days (2022-05-11 → 2026-03-30,
  bounded by `data/spy_1m.csv` coverage).
- **Output dir:** `data/cbbo_cache_jl_10min/` — a new directory; never writes to the shared
  caches. Same pickle schema as the other caches: `{(minute_et, strike, is_call): (bid, ask)}`.
- **Cost control:** run Databento's free `metadata.get_cost` **dry-run first**; abort if the
  estimate exceeds `--max-cost-usd` (default **$15**). Expected ~$3–5. Log spend to the
  existing `data/databento_cost_ledger.jsonl`.

## 7. Architecture / components

High reuse — the existing `scripts/analytical_ic_eval.py` already prices and marks with
**four independent strikes** (`_ic_credit`, `_ic_mark`), so asymmetric widths already work;
it already supports whole-position TP (`manage="total"`) and a configurable anchor list.

| Component | Path | Responsibility |
|---|---|---|
| Backfill manifest + runner | `scripts/backfill_jl_cbbo.py` (adapts `meic_backfill`) | Per day: SPY-derived center → ±100 window → pull 09:30 + 10-min anchors → `data/cbbo_cache_jl_10min/`. Dry-run cost gate. |
| Evaluator | `scripts/analytical_jl_eval.py` | Per day: compute EM, place 4 strikes (30/10 widths), entry credit, 10-min whole-position TP, expiry settlement. Emits metrics, per-trade log, exit breakdown. Imports shared helpers from `analytical_ic_eval` (`_quote_mid`, `_atm_strike`, stat helpers) — refactor those into a small shared module if cleaner. |
| Shared core (if extracted) | `src_v2/backtest/analytical_core.py` | `_quote_mid`, `_atm_strike`, `_compute_em`, `_max_consec_losses`, `_max_drawdown`. Extract only if it keeps both scripts focused; otherwise import directly. |
| Report | `docs/spx_0dte_jade_lizard_findings.md` + `results/jade_lizard/` (per-trade CSV/JSON) | Headline (zero-cost) + with-costs tables, exit-type breakdown, equity curve, skip-reason counts, scaling to $100K. |
| Tests | `tests/test_jade_lizard_eval.py` | Unit tests for strike placement, credit, whole-position TP trigger, expiry intrinsic on each side, the credit≥width flag, and skip rules. (New tests go in `tests/`, never `tests_locked/`.) |

### Data flow

```
data/spy_1m.csv ──► backfill_jl_cbbo.py ──► data/cbbo_cache_jl_10min/<date>.pkl
   (09:30 center)        (Databento, dry-run gated)        (39 anchors, ±100 strikes)
                                                                    │
                                                                    ▼
                                          analytical_jl_eval.py ──► metrics + per-trade log
                                          (EM, strikes, credit,        + results/jade_lizard/
                                           10-min TP, expiry)          + findings doc
```

## 8. Metrics / reporting

Reported for both **zero-cost** and **with-costs** runs:

- Trade count; skip count broken down by reason.
- Per-day win rate; total P&L; mean P&L/trade; annualized Sharpe.
- Max consecutive losses; max drawdown (trade-ordered **and** daily-equity).
- Avg win, avg loss, worst day, best day.
- Exit-type breakdown: TP / expiry-win / expiry-loss (count and P&L each).
- Avg entry credit; fraction of days with `total_credit ≥ $10` (true jade-lizard property);
  realized upside breaches and downside breaches.
- Scaling: per-1-lot P&L and annualized ROI on a $100K account (defined-risk sizing:
  per-lot max loss ≈ `$30×100` put side, so buying-power-limited contract count).

## 9. Phases (de-risks the spend)

0. **Free smoke test** on `cbbo_cache_5anchor` (true SPX, 5 checks not 10-min, entry 09:35):
   validate JL leg selection / credit / whole-position TP / expiry math and get a ballpark.
   Cost $0. Gate before any Databento spend.
1. **Backfill manifest + dry-run** cost estimate; confirm with user.
2. **Run backfill** into `data/cbbo_cache_jl_10min/` (after confirmation, under the $15 cap).
3. **Full 10-min evaluation** + findings report.
4. *(Optional, separate spec later)* WF-CV sweep over `em_mult`, put/call widths, TP% per the
   5:1 / 4-fold methodology (`feedback_cv_methodology`).

## 10. Orchestration

Implementation is run as a **Workflow** (multi-agent): parallel agents author the evaluator,
the backfill runner, and the report writer against this spec, followed by an
adversarial-verification pass on the P&L arithmetic (entry credit, 10-min TP trigger, and
expiry intrinsic on synthetic cases) before any numbers are trusted.

## 11. Out of scope (YAGNI)

- Live/paper trading, order routing, LLM advisors, scheduler integration.
- Per-side exit management, stop-losses, trailing stops (user chose whole-position, no SL).
- Regime/event filters (TT-vanilla headline; filters can be a later sweep).
- Re-centering the call strike to enforce the jade-lizard credit≥width property.
- Modifying anything under `tests_locked/`.

## 12. Open items (carried into the plan, not blockers)

- Exact final-anchor handling for SPXW PM settlement (use 16:00 chain forward; document the
  approximation vs the official SET/close).
- Whether to extract `analytical_core.py` or import helpers directly (decide during impl by
  what keeps both scripts focused).
