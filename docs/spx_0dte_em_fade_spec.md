# SPX 0DTE Expected-Move Fade — extracted spec

Source: YouTube `jvlGGTVE234` — "Setup Profits" interview with **Jamal Gi**
("credit spread mean reversion strategy"). Transcript pulled 2026-08-16.

## Structure

Single-side **0DTE vertical credit spread on SPX**, sold *against* an
expected-move touch (fade / mean reversion).

## Rules as stated

| Element | Rule (his words) |
|---|---|
| Universe | ~90% SPX (rest "blue chips"); SPX chosen for liquidity + §1256 |
| Expected move (EM) | derived from the options chain (straddle); "66% chance price stays inside" → 1 SD |
| Bands | spot ± EM, marked at the start of the day |
| Entry trigger | price **touches** a band → sell the credit spread on **that** side (call spread at upper band, put spread at lower) |
| Confirmation | **none** — "we will enter the trade no matter what" |
| No-trade case | market chops inside the band all day → no trade |
| Time cutoff | "if it doesn't hit before **1 pm ET** I won't take the trade at all"; open-flush touches are best |
| Short strike | **10–25 delta**, placed just beyond a recent support/resistance |
| Width | 5–20 points, "usually 10" |
| Take profit | **50% of credit** |
| Stop loss | **100% of credit** (spread price = 2× credit), stop-**market**, bracket order |
| Trail | at ~+25% unrealized, move stop to break-even or +10% — discretionary |
| Hold | "20–30 minutes on average" |
| Sizing | risking 1–2% of account per trade / per day |

## Claims to test

| # | Claim | Stated value |
|---|---|---|
| C1 | Win rate | **85%** |
| C2 | Reward:risk | 1:2 (avg loss ≈ 2× avg win, "slightly higher") |
| C3 | Annualized return | **~35%** at 1–2% risk per trade |
| C4 | Average hold | 20–30 min |
| C5 | Put side beats call side | put win rate higher |
| C6 | Best in high-vol regimes | wider bands = more room |
| C7 | Earlier touch = better | open flush strongest |
| C8 | Risk rating | 3–4 / 10 |
| C9 | Worst case with stop | −100% of credit |

Live-traded 1.5 years + "a few backtests"; no equity curve, no trade log,
no Sharpe/Calmar figures given (he names Calmar as a favourite metric but
does not quote one). **No cited example trades** → no reproduction anchor
beyond the rule set itself.

## ASSUMPTIONs (unstated → canonical default), with direction of bias

| # | Assumption | Default chosen | Biases result |
|---|---|---|---|
| A1 | EM definition | ATM straddle mid at 09:32 ET (canonical `otm_skew_calculator.calculate_expected_move`); 0.8× straddle as sensitivity | neutral; 0.8× → more touches |
| A2 | Band anchor | 09:30 ET SPX open | neutral |
| A3 | Trades per day | max 1 (matches "1–2% per day" ≈ "1–2% per trade") | conservative (fewer shots) |
| A4 | Touch window | 09:35–13:00 ET | as stated |
| A5 | Short delta | 20Δ default; 10/15/25 in sweep | neutral |
| A6 | Width | 10 pts default; 5/20 in sweep | neutral |
| A7 | Trail rule | mechanised: at +25% unrealized, stop → +10% of credit. Tested as an **arm**, both on and off | **favours** the claim (his stated win-rate driver) |
| A8 | End-of-day | flatten 15:50 ET if neither bracket hit | conservative vs letting OTM expire worthless |
| A9 | Support/resistance strike nudge | **omitted** — delta target only | favours the claim (removes a discretionary filter that can only be curve-fit after the fact) |
| A10 | Commissions | $0.65/contract/side → $2.60 round trip per spread | conservative |
| A11 | Instrument | SPX (SPXW) — as claimed, not a SPY proxy | none |

## Data plan — ZERO new spend, the cache already exists

| Need | Source | Cost |
|---|---|---|
| SPX 0DTE option NBBO, 1-min, both sides | **`data/cbbo_cache_fly_10min/`** — 1022 day-files, 2022-05-11 → 2026-06-11, keyed `(minute, strike, is_put) → (bid, ask)`; median **389 minutes** (09:31–15:59) × **99 strikes** at 5-pt spacing spanning **505 pts** | already paid |
| Intraday SPX spot | `CbboStore.derive_spot()` — put-call parity off the same cache, minute by minute | free |
| Signal bars | `BarLoader` (SPY 1-min) — used only where the engine already uses it; parity spot is authoritative for band logic | free |
| IV / delta | `src_v2/flux/iv_compute.compute_iv_rows` — per-strike IV inverted from that strike's OWN mid, then `bs_delta` | free |

**Coverage verified** (86-day sample across the span): 1 day short of 380
minutes, none short on strikes. Span ±~250 pts ≈ ±3.8% at SPX 6700 — comfortably
wider than 1 EM (~0.6%) plus the ±90-pt strike ladder.

**Window limitation:** the cache starts **2022-05-11**, so Jan–Apr 2022 (the
first leg of the bear) is NOT covered. June/Sept/Oct 2022 lows ARE. Flag in the
verdict; do not claim full-cycle coverage.

## Fill model — non-negotiable

Scored at **real CBBO bid/ask** from the start, never mid. The store returns
`(bid, ask)` per leg per minute; the package is built from them:

- `open_bid  = short_bid − long_ask` (credit if you sell now)
- `close_ask = short_ask − long_bid` (debit if you buy back now)
- depth knob `f`: 0.5 = mid, 1.0 = far side. Sweep `{0.5, 0.6, 0.7, 0.85, 1.0}`.
- entry: sell-limit → credit at depth `f`
- TP: buy-limit at 0.5 × credit → fills only when the **ask** reaches it
- SL: stop-**market** at 2 × credit → triggers on the package price, fills by
  **crossing** (`f = 1.0`), because that is what a stop-market does
- pnl clamped to the structural band `[−max_loss, +credit]`; marks outside are
  crossed/stale quotes and are skipped, not traded on

## Controls

1. **With-trend mirror** — identical trigger, spread sold on the opposite side.
   This is the single most informative control: the repo's existing
   `orb_em_directional` is that arm on a *different* trigger and its fade
   variant is already documented dead
   (`docs/orb_dir_debit_excursion_diag_2026-06-18.md`).
2. Unconditional entry (same delta/width, fixed 10:00 ET, no touch required)
3. Random entry minute inside the window (seeded)
4. Buy-and-hold SPX

## Prior evidence (must be reconciled, not ignored)

`docs/orb_dir_debit_excursion_diag_2026-06-18.md` already tested a **fade**
variant on real CBBO over 2022-04→2026-06 and found **no fade edge**:
fade_breakout +$39/trade, 74% win, Sharpe/trade 0.06, and negative across
2023–2025 (positive only on a partial 2026). That is a **different trigger**
(ORB breakout + retest, not an EM-band touch from the open) and **different
management** (tp 0.75, no stop, hold breach — vs 50% TP / 100% stop-market /
trail), so it does not settle this claim. It is a strong negative prior and the
verdict must say where this study agrees or disagrees with it.

## Pre-registered arithmetic (L1 invariant)

With TP = +50% of credit and SL = −100% of credit, per-trade EV
= C(1.5w − 1) → **break-even win rate = 66.7%**. Claimed 85% ⇒ EV = +0.275 C.
At 1–2% of account risked per trade and ~40% of days producing a touch
(~100 trades/yr), that implies **~23–47%/yr** — the 35% claim is internally
consistent, not self-refuting. This study is therefore a real test, and the
binding question is whether the 85% survives real fills.
