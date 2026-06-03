# OTM 0DTE Butterfly — Backtest Design Spec

- **Date:** 2026-06-03
- **Status:** Approved for planning
- **Source strategy:** Ernie Varitimos / Fat Tail "Asymmetric 0DTE Butterfly" (deck `docs/E048_Adjust_Butterfly_Width_R2R_4.pptx`, plus 0-dte.com / fattail.ai published rules)
- **Closest existing analog:** Jade Lizard backtest (`src_v2/backtest/jl_core.py` + `scripts/analytical_jl_eval.py`)

---

## 1. Background & motivation

The strategy is a **directional, out-of-the-money 1‑2‑1 long debit butterfly on 0DTE SPX**, sized so the debit is ≤10% of the wing width (equivalently, a ≥9:1 theoretical reward-to-risk). Width is set by a VIX regime map; the body is placed ~1σ OTM; direction is chosen by a daily trend filter. The marketed thesis is *convexity / positive skew* — "solve for risk-adjusted return, not win rate."

**Why we are backtesting it independently:**

- Every public performance number (the deck's Sharpe 4.02 / 123.57% over 1,422 days; a separate "114% / Sharpe 5.87 over 18 months" claim) is **vendor self-reported**. No independent backtest *with realistic fills* exists anywhere we could find.
- The headline **9:1 R2R is theoretical max** (price pinning the body at expiry). The vendor ecosystem's own figures imply realized outcomes of **~1.8:1** (avg return ≈180% of risk) at a **45–65% win rate** — an order of magnitude below the advertised asymmetry.
- The edge lives inside a tight 10%-debit tolerance on a **4-leg OTM 0DTE structure** with wide bid/ask. Slippage of $0.25–0.50 flips "viable 10%" into "dead 15%." Execution realism is the whole question.
- A conceptual tension we want to resolve empirically: the deck's philosophy is **direction-agnostic (50/50)**, yet it places a *directional* fly. If direction is truly 50/50, the side is irrelevant and only the structure's expectancy matters.

## 2. Research questions

**Primary:** Does an OTM 0DTE SPX butterfly, body at 1σ, sized for ≥9:1, have **positive net-of-cost expectancy out-of-sample**, with positive skew — *regardless of side*?

**Secondary:**
1. Does direction selection (21-day Hull MA trend) beat a **coin-flip baseline**? (Tests the 50/50 hypothesis directly.)
2. Which **exit model** maximizes realized edge: hold-to-expiry, the documented dynamic trailing stop, or a flat profit target?
3. Do the **10%-debit / 9:1 gates** add edge, or merely shrink the sample?
4. Does the **VIX→width** regime map help vs a regime-free fixed width?
5. How does behavior differ **by regime** (Zombieland / Goldilocks / Chaos), reported with explicit N?

## 3. Goals / non-goals

**Goals:** a pure, testable analytical backtest reusing existing JL infrastructure; honest OOS evaluation via WF-CV; headline *and* net-of-cost results for every figure; per-regime breakdown including Chaos.

**Non-goals:** live trading / `LiveStrategy` ABC integration; reconstruction of proprietary discretionary signals ("Levels / Wells / Crevasse / divergence tool" — replaced by the Hull-MA trend filter); ES port (deferred unless SPXW shows edge); intraday re-entry / multiple trades per day.

## 4. Locked rules (fixed inputs)

| # | Rule | Value |
|---|---|---|
| 1 | Instrument | SPXW 0DTE, 4-leg 1‑2‑1 long debit butterfly, 3 strikes, multiplier 100 |
| 2 | Body placement | **1σ OTM** = 1.0 × ATM-straddle EM from the entry-chain forward, rounded to nearest 5-pt strike |
| 3 | Direction | **21-period Hull MA** on SPX daily close; uptrend → call-side fly (body above forward), downtrend → put-side fly (body below forward). Lookahead-free (slope from closes strictly before the entry day) |
| 4 | Wing width by VIX regime (prior close) | **VIX <17 → 20–30 · 17–32 → 30–50 · ≥32 → 50+** (symmetric wings). VIX <13 treated as Zombieland. Width within band is a swept parameter |
| 5 | Entry gates | debit ≤ **10% of width** AND R2R = (width − debit)/debit ≥ **9:1**; else skip the day |
| 6 | Entry timing | **one trade per day**, fixed morning entry time (swept across runs) |
| 7 | Hard exit | EOD force-close **15:50**; loser left to expire at full debit (= max loss) |
| 8 | Sizing | baseline **1 contract**; risk-normalized variant sizes to fixed $ risk for ARR on a $100K account |
| 9 | Costs | **headline** (all legs at mid) **+ with-costs** (slippage/leg + commission/leg over 4 legs); both reported for every figure |

### 4.1 Strike construction

Let `F` = chain-implied forward at the entry anchor (put-call parity: `F = atm + (call_mid − put_mid)`), `EM` = ATM-straddle mid at the entry anchor, `W` = wing width.

- **Call-side (uptrend):** `K_body = round5(F + EM)`, `K_low = K_body − W`, `K_high = K_body + W`, all **calls**. Position: +1 `K_low`, −2 `K_body`, +1 `K_high`.
- **Put-side (downtrend):** `K_body = round5(F − EM)`, `K_low = K_body − W`, `K_high = K_body + W`, all **puts**. Position: +1 `K_high`, −2 `K_body`, +1 `K_low`.

Entry **debit** = combined mark of the structure at entry (must be > 0). Mark uses `quote_mid` with the existing ±5-minute fallback. If any of the 3 strikes is missing after fallback, the day is `skip: no_quote`.

### 4.2 Direction filter (Hull MA)

`HMA(n) = WMA( 2·WMA(price, n/2) − WMA(price, n), round(√n) )`, `n = 21`, on SPX daily close (derive from `data/spy_1m.csv` ×10 or a daily series). Trend at entry day `t` = `sign(HMA[t−1] − HMA[t−2])` (uses only closes through `t−1`). `>0` → call-side, `<0` → put-side, `0` → carry previous. Requires ≥23 daily-close warmup.

### 4.3 VIX regime → width

Regime from **prior-close `vix1d`** (fallback `vix9d`, then `vix`) via `src_v2/backtest/vix_loader.py`. Width swept within band: Zombieland {20, 25, 30}, Goldilocks {30, 40, 50}, Chaos {50, 60}. A **regime-free** control sweeps a single fixed width across all days.

## 5. Experimental axes (what the backtest decides)

| Axis | Settings |
|---|---|
| **Direction** | Hull-MA trend · call-only · put-only · **coin-flip baseline** (deterministic: parity of date ordinal) · anti-trend |
| **Entry time** | 09:30 · 10:00 · 10:30 (one trade/day each) |
| **Exit model** | (a) hold-to-expiry intrinsic *reference* · (b) documented **time-of-day dynamic trailing stop** · (c) flat profit target {1.5×, 2×, 3× debit} |
| **Gates** | gates-on vs gates-off |
| **Width** | regime-tied vs regime-free |
| **Body (secondary)** | 1.0× EM vs 0.75× EM |

### 5.1 Exit model (b) — documented dynamic trailing stop

From the published rule: *"trailing stop kicks in at 50% profit, starting wide (50–75%) in the morning, shrinking to 30–40% by noon, locking tight at 30% in the afternoon."* Modeled deterministically (this interpretation is an explicit assumption, params sweepable):

- Track unrealized profit `P = (mark − debit)` and its running peak `P_peak` since entry, evaluated at each cache anchor.
- **Arm** the trail once `P ≥ 0.50 × debit`.
- Once armed, **exit** when `P ≤ P_peak × (1 − g)` where the giveback tolerance `g` is time-of-day dependent: `g = 0.625` before 12:00 ET, `g = 0.35` for 12:00–14:00, `g = 0.30` after 14:00 (mid-points of the published 50–75% / 30–40% / 30% bands).
- If never exited, settle at EOD per §6. No hard stop loss; max loss = full debit.

## 6. Settlement

Held-to-expiry / EOD legs settle at **intrinsic vs the chain-implied forward at the final anchor** (per JL; never SPY×10 close, which runs ~18pt below the SPX chain forward and manufactures phantom losses). Butterfly intrinsic at forward `S`:

- Call fly: `max(0, S−K_low) − 2·max(0, S−K_body) + max(0, S−K_high)`
- Put fly: `max(0, K_high−S) − 2·max(0, K_body−S) + max(0, K_low−S)`

## 7. Data & backfill

Reuse the 10-min SPXW CBBO cache pattern, but **re-backfill with a wider strike window** so all three regimes (including Chaos) are fully priced:

- **New cache:** `data/cbbo_cache_fly_10min/{date}.pkl`, schema `{(minute_et, strike, is_call): (bid, ask)}`.
- **Window:** **±250 pt** around the 09:30 SPY-derived center, 5-pt grid (vs JL's ±100). Covers body(1σ EM, large at high VIX) + wings(50–60) on Chaos days.
- **Coverage:** ~940 trading days, 2022-05-11 → 2026-03-30; 1-min anchors on the 10-min grid (09:31–15:59).
- **Cost:** ~$8–10 Databento (dry-run estimate first; log to `data/databento_cost_ledger.jsonl`; cap $20). Script: `scripts/backfill_butterfly_cbbo.py` (adapt `backfill_jl_cbbo.py`, reuse `meic_backfill.quotes_from_df`).
- **Note:** ES-basis caches (`cbbo_cache_es`) are rejected — ES forward sits ~20pt off SPX, breaking OTM placement.

**Sample-size caveat:** VIX closed ≥32 on only a handful of clusters in-window (late-2022, Aug-2024 spike, Apr-2025 selloff), ~15–40 days. Chaos results are **indicative, not conclusive**; N reported separately, never blended into the headline.

## 8. Architecture

| File | Role |
|---|---|
| `src_v2/backtest/butterfly_core.py` | Pure logic, no I/O: `select_fly_strikes()`, `entry_debit()`, `fly_mark()`, `fly_intrinsic()`, `hull_ma_trend()`, `evaluate_day()`, `evaluate()`, `ButterflyConfig` dataclass |
| `scripts/analytical_butterfly_eval.py` | CLI runner: load cache + bars + VIX, run scenarios/axes, write CSV + JSON. Mirrors `analytical_jl_eval.py` |
| `scripts/backfill_butterfly_cbbo.py` | Wider-window Databento backfill |
| `tests/test_butterfly_core.py` | Unit tests (strike selection, debit, trailing-stop logic, intrinsic settlement, Hull MA) |
| `config/config_butterfly_otm_spx.yaml` | Defaults + sweep grid (reference; CLI overrides) |
| `docs/spx_0dte_butterfly_otm_findings.md` | Written-up results |

**Reused as-is:** `data_loader.BarLoader`, `jl_core` helpers (`quote_mid`, `atm_from_chain`, `forward_from_chain`, `compute_em`), `vix_loader`, the cost model, `wf_optimizer` (Phase 2).

## 9. Cost model

Reuse the JL model adapted to 4 legs:
- **Entry:** each leg filled at mid ± `slippage_per_leg`.
- **Exit (trail/TP, mid-exit):** each leg at mid ± `slippage_per_leg`.
- **Exit (expiry, intrinsic):** no closing trade, no exit slippage.
- **Commission:** `commission_per_leg` × 4 at open (+ ×4 at close for mid-exits).
- **Defaults:** `slippage_per_leg = $0.05` (sweep 0.025 / 0.05 / 0.10), `commission_per_leg = $1.00` (TastyTrade SPX). Headline scenario sets both to 0.

## 10. Evaluation methodology

**Staged:**
- **Phase 0 — smoke:** logic on a handful of days vs hand-computed values (unit tests).
- **Phase 1 — full-sample descriptive:** every axis, headline + with-costs, full ~940 days. Gate to Phase 2 only if *any* configuration shows net-of-cost expectancy > 0.
- **Phase 2 — WF-CV:** ratio **5:1, 4 folds** (`feedback_cv_methodology`); params chosen on train, scored OOS.

**Metrics:** net expectancy/trade, realized R2R (sanity-anchor ≈1.8:1), win rate (anchor 45–65%), Sharpe, profit factor, max drawdown, full return distribution + the deck's 4-zone shape, **per-regime breakdown with N**.

**Primary success criterion:** **net-of-cost OOS expectancy > 0 with positive skew.** ARR is reported **descriptively** on a $100K account (1-contract and risk-normalized) — **not** held to the standing 70% ARR gate, since a capped-risk single-contract 0DTE fly cannot realistically reach it (the deck itself is ~15% CAGR). Sanity anchors: if results diverge wildly from the vendor's own ~1.8:1 / 45–65% figures, suspect a fill fantasy or bug before celebrating.

## 11. Key assumptions & risks

1. **Trailing-stop interpretation** (§5.1) is inferred from a fuzzy published description; params are swept and the hold-to-expiry reference bounds it.
2. **Hull-MA trend definition** (slope sign) is one reasonable reading of "check the 21-day Hull MA, pick a side."
3. **Body = 1.0× ATM-straddle EM** equates "1σ" with the straddle EM; the deck's own example is ~0.6× VIX-daily, so 0.75× EM is carried as a secondary sweep.
4. **Mid-fill optimism** is the dominant risk; the with-costs scenario and slippage sweep are the guard. Realized 0DTE 4-leg fills may be worse than $0.10/leg.
5. **Chaos sample is small** — treat as indicative.
6. **Quote sparsity** on far OTM 0DTE legs; ±5-min fallback mirrors JL, missing-strike days skipped and counted.

## 12. Outputs

- `results/butterfly_otm/trades_*.csv` (per-trade, per scenario), `summary_*.json`, WF-CV fold reports under `results/butterfly_otm_wf/`.
- `docs/spx_0dte_butterfly_otm_findings.md` — verdict on the primary/secondary questions, with the headline-vs-net and per-regime tables.

## 13. Sources

- Deck: `docs/E048_Adjust_Butterfly_Width_R2R_4.pptx`
- https://0-dte.com/0dte-asymmetric-strategy/classic-otm-butterfly/ (Hull-MA direction, VIX→width, trailing-stop rule)
- https://0-dte.com/review-of-the-0dte-otm-butterfly-trading-strategy/
- https://ippei.com/ernie-varitimos/ (win rate 45–65%, avg return ~180% of risk, "far from foolproof")
- https://www.ninjaspread.com/spx-0-dte-butterflies/ ("best setups in the first 30–60 minutes")
