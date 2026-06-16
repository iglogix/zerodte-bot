# GEX Regime Feasibility Study for 0DTE Iron Condors — Design

**Date:** 2026-06-16
**Status:** Design approved, pending spec review → writing-plans
**Branch:** dedicated study branch off `master` (e.g. `feat/gex-regime-study`); do **not** co-mingle with the current `feat/flux` branch
**Related:** [[project_butterfly_backtest]] (prior-close GEX refuted), `src_v2/backtest/gex.py`, `scripts/backfill_oi.py`, `docs/wf-cv.md`

---

## 1. Motivation & central question

We want to know whether a **dealer gamma-exposure (GEX) regime filter** improves 0DTE iron-condor (IC) outcomes — and, prior to that, **how reliably the regime can even be determined.** ICs are the cleanest possible test bed: their P&L is almost a pure bet on *suppression* (range-bound, dealer long gamma) vs *amplification* (trending, dealer short gamma), so a real regime signal should surface here if it surfaces anywhere.

The reliability question splits along an **observability line** discovered during design:

| Regime signal | Observable at entry? | Reliably determinable? | Edge on IC? |
|---|---|---|---|
| Standing all-expiry GEX (prior-close OI, all expiries near spot) | Yes, at open, stable intraday | **Yes** (cheap, statistics schema) | **Unknown** — butterfly hint was a vol confound, but used 0DTE-only OI |
| 1DTE-as-0DTE proxy (next-expiry OI, evaluated at 0DTE kernel) | Yes (published OI) | **Yes**, if carried-in regime is sticky | Unknown — captures carried-in, not same-day flow |
| 0DTE intraday gamma (today's build) | **No** — must reconstruct from signed trades | **Hard/noisy** (sign-classification floor) | Plausibly the real driver, untested |

**Two testable questions:**
1. Does the regime we *can* measure reliably (standing GEX) predict 0DTE IC outcomes once volatility is controlled?
2. Can we determine the regime that *matters* (intraday 0DTE gamma) reliably enough to act on — or does an observable proxy (1DTE) make that moot?

## 2. Background & prior work (what's already established)

- **GEX engine exists** — `src_v2/backtest/gex.py`. Pure logic over an in-memory OI dict `{(strike, is_call): oi}` plus spot, ATM-straddle EM, and time-to-expiry. Dealer convention: long gamma from calls, short from puts. Exposes `bs_gamma`, `iv_from_em` (Brenner-Subrahmanyam ATM-straddle→σ), `gex_by_strike`, `total_gex` (sign = regime), `magnet_strike` (pin), `gamma_flip` (zero-crossing level). **Crucially, OI is decoupled from the `(sigma, T)` gamma kernel**, so any OI distribution can be evaluated at any time-to-expiry — this is what makes the 1DTE-at-0DTE-kernel proxy a one-liner.
- **OI cache exists but is 0DTE-only, prior-close** — `data/oi_cache/` (~952 daily `.pkl`, 2022-05 → present), built by `scripts/backfill_oi.py` from the OPRA `statistics` schema (OPEN_INTEREST). Per its docstring, the morning-published OI "reflects the prior session's close." `meic_day_symbols` restricts to **same-day SPXW expiry**.
- **Prior-close GEX was REFUTED for the butterfly** (`project_butterfly_backtest`): the +GEX vs −GEX performance gap was a **vol/cheapness confound** (−GEX ≡ higher VIX → bigger EM → cheaper structure → higher R2R), identical ~30% win rates, identical morning drift. Prior-close GEX had **no relation to realized intraday regime.**
- **No-0DTE-oracle finding (design-time):** a 0DTE contract expires before OCC publishes another OI snapshot, so today's intraday 0DTE flow *never* lands in published OI. There is **no perfect-hindsight 0DTE OI**; the intraday 0DTE regime is only recoverable from intraday signed trades.
- **No trade cache exists** — only quote (CBBO) caches: `data/cbbo_cache_meic`, etc. Rung 2 requires a new `tbbo` pull.
- **EM convention** — ATM-straddle expected move, not VIX ([[feedback_0dte_em]]); already how `gex.py`/`iv_from_em` work.

## 3. Signal hierarchy (the rungs)

| Rung | Signal | Data | Cost | Role |
|---|---|---|---|---|
| **1** | Standing all-expiry GEX | OI (statistics), all expiries ≤~120 DTE | cheap | Broad dealer regime, de-confounded; full history |
| **1.5** | 1DTE levels as 0DTE proxy + day-over-day OI-evolution stability | OI (statistics), 1DTE & 2DTE expiries | cheap | Best *observable* 0DTE-specific proxy; **gate that decides whether Rung 2 is needed** |
| **2** | 0DTE intraday trade reconstruction | `tbbo`, 0DTE, ATM band, morning | expensive | **Only runs if Rung 1.5 shows the regime is flow-driven** |

The study is therefore *mostly cheap OI work*, with the expensive trades pull gated behind a result that proves it's necessary.

## 4. Variables

### 4.1 Regime variable (X)
- **Primary:** signed distance from spot to the gamma-flip, in EM units — `(S − flip)/EM`. Continuous; encodes sign *and* magnitude (above flip & far → strong suppression; below flip → amplification).
- **Robustness cross-check:** binary net-GEX sign at spot (`total_gex(S) ≷ 0`).
- **Exploratory secondary:** magnet/pin distance `(S − magnet)/EM` (mean-reversion-toward-pin).

All three come directly from `gex.py`; only the OI *input* changes per rung.

### 4.2 Dependent variable (Y)
- **Primary (mechanism, used for the kill-test):** vol-normalized realized move from entry —
  - terminal `|S_close − S_entry| / EM` (settlement-relevant), and
  - **max intraday excursion / EM** (touch/stop-relevant) — **weighted most**, as amplification/trend days are *defined* by excursion and that's what stops out a real IC.
- **Confirmation layer (only if primary survives):** canonical single-entry 0DTE IC P&L — fixed short-delta, fixed wing width, fixed entry time, hold-to-expiry, priced from the existing cbbo cache, with fill realism applied ([[feedback_slippage_liquidity]]).

Rationale: the kill-test runs on the clean realized-move variable so a null result means "regime has no signal," not "our execution model is noisy." The IC P&L layer only verifies the mechanism translates to the structure net of fills.

## 5. Rung 1 — Standing all-expiry GEX (cheap, full history)

**Data / backfill (new):** extend the OI backfill to **all expiries ≤ ~120 DTE** near spot. Strike window **EM-scaled** (not fixed points) and **asymmetric to the downside** (the gamma flip is typically −5% to −10%, set by put-heavy OI; a symmetric window would mislocate it). Window width fixed by a **convergence check**: widen until `total_gex` sign and `gamma_flip` stop moving on a handful of probe days, then lock that EM-scaled width. Statistics schema; run a `get_cost` dry-run first (est. ~$10–50 for full history). Period: full oi range (~2022-05 → present, ~950 days).

**De-confound (the core methodological fix over the butterfly):**
- Y is already EM-normalized, killing the mechanical "high vol → big move" scaling.
- On top of that: (a) regress `excursion/EM ~ regime + EM_level + regime×EM`, report the **partial regime effect holding EM fixed**; (b) bucket days by EM tercile and test the regime effect **within** each bucket.
- The butterfly confound was −GEX ≡ high-vol. If regime adds nothing beyond EM, the partial effect is ~0.

**Success bar (pre-registered, OOS):** regime's partial effect on `excursion/EM` is consistent-sign and economically meaningful across a **majority of sequential OOS folds** *and* within ≥2 of 3 vol buckets; confirmation layer shows a positive suppression-vs-amplification IC P&L spread net of fills in a majority of folds. **Single pre-specified hypothesis — no combo sweep** (the butterfly's 120-combo WF-CV sweep overfit; [[feedback_cv_methodology]]).

**Kill gate:** if the clean, de-confounded standing signal has no partial effect, the broad-regime thesis is weakened (but not fully dead — 0DTE-specific could still matter, → Rung 1.5).

## 6. Rung 1.5 — 1DTE-as-0DTE proxy + OI-evolution stability (cheap; the Rung-2 gate)

**Idea:** use the **observable 1DTE expiry's OI** (published prior close), evaluated at the **0DTE time-to-expiry kernel**, as the regime read. The 1DTE expiry is the most 0DTE-like population (same day-trader crowd, one session early) — a tighter proxy than the all-expiry blend, and it carries **no signing noise** (it's published OI).

**Data:** extend backfill to the **1DTE and 2DTE expiry symbols** near spot (same cheap statistics schema; comes essentially for free inside the Rung-1 all-expiry pull).

**Two analyses:**
1. **Proxy quality** — does `(S − flip_1DTE)/EM` predict the same Y as Rung 1, and does it add incremental predictive value over the standing signal?
2. **OI-evolution stability (the natural experiment, zero trades):** today's 1DTE contract *becomes* tomorrow's 0DTE; tomorrow morning we observe its matured prior-close OI. So track each expiry's gamma profile across **2DTE → 1DTE → 0DTE-prior-close**:
   - If **sign & flip are stable** across these snapshots → carried-in regime is sticky, the 1DTE proxy is reliable, same-day flow is a minor perturbation → **Rung 2 is unnecessary.**
   - If they **swing day-to-day** → regime is flow-driven, the proxy inherits the prior-close refutation → **Rung 2 is required.**

**Gate decision:** Rung 1.5's stability result determines whether the expensive Rung 2 trades pull runs at all. This is the key cost-control mechanism of the study.

## 7. Rung 2 — 0DTE intraday reconstruction (gated, sample-first)

**Runs only if Rung 1.5 shows the regime is flow-driven.**

**Data (aggressive lossless cherry-pick):**
- **Expiry:** 0DTE only. **Strike:** tight ±~2×EM ATM band (0DTE gamma is hyper-peaked at T≈0; far strikes contribute ~0 regardless of OI). **Time:** open → last scored entry window (+ short tail for the stabilization curve). **Schema:** `tbbo` (trade + prevailing BBO) — carries the quote-rule signing inputs, far lighter than book reconstruction.
- **The cut we may NOT make:** thinning prints *within* a kept contract — dealer gamma is the cumulative *signed* volume (an integral); subsampling biases it. Prune the universe/window hard; take **every print** inside what survives.
- **Sample first:** 60–90 days spanning one high-vol and one calm stretch; `get_cost` dry-run before committing; size cost + noise before any full-history pull.

**Reliability — measured *internally* (no 0DTE oracle exists):**
1. **Sign-rule sensitivity** — perturb buy/sell classification (flip a fraction of ambiguous mid-prints / combo legs); does the regime call hold?
2. **Rule agreement** — tick-rule vs quote-rule vs a combo-aware classifier; how often do they agree on the regime?
3. **Intraday stabilization** — how early does the reconstructed regime stop flipping (the "how early can we know" curve) relative to a tradeable entry time.

**Value-add:** does the reconstructed 0DTE regime improve prediction of `excursion/EM` **beyond** Rung 1 + Rung 1.5 (incremental fit, OOS)?

**Caveat (acknowledged up front):** SPX 0DTE flow is dominated by multi-leg combos (ICs, verticals) that print as combos and are exactly where per-leg signing mangles direction. Sign-classification accuracy — not the data window — is the irreducible reliability floor.

**Success bar:** regime call robust to sign-error perturbation **and** ≥ an agreement floor across rules **and** stabilizes before a tradeable entry time — **and** shows incremental predictive value. Miss any → "not reliably determinable as specified," documented.

## 8. Methodology guardrails

- Sequential OOS folds per `docs/wf-cv.md`; pre-registered single hypothesis; no val-leaked selection; plateau over best-cell ([[feedback_cv_methodology]]).
- EM-scaled windows everywhere — fixed-point windows truncate more on high-vol days and re-introduce the vol confound (the JL-cache lesson: fixed ±100pt truncation was non-random and side-dependent).
- Fill realism on the IC confirmation layer ([[feedback_slippage_liquidity]]).
- Use the project `.venv` Python ([[feedback_venv_python]]).

## 9. Decision flow

```
Rung 1 (standing GEX, de-confounded, full history)
  └─ partial regime effect ≈ 0 in all buckets/folds? ── broad-regime thesis weakened
Rung 1.5 (1DTE proxy + 2DTE→1DTE→0DTE OI-evolution stability)
  ├─ sign/flip STABLE across final days ──► proxy reliable; Rung 2 UNNECESSARY → conclude
  └─ sign/flip SWINGS day-to-day ─────────► regime flow-driven; Rung 2 REQUIRED
Rung 2 (0DTE trade reconstruction, sample-first)  [gated]
  └─ reliability bar + incremental value ► verdict on whether the *actionable* regime is determinable
```

## 10. Deliverable & out of scope

**Deliverable:** findings doc `docs/spx_0dte_gex_regime_findings.md` + a one-line verdict added to the strategy scoreboard, mirroring the butterfly/burrito writeups (verdict + headline metrics + load-bearing caveat).

**Out of scope (deferred to follow-ups):** Rung 3 live gating of the IC configs; MEIC multi-entry regime (each entry has its own regime read); any live/real-time GEX product; non-IC strategy families.

## 11. Open risks

- All-expiry strike-interest differs from 0DTE (round-number hedging walls vs day-trader strikes) — the 1DTE proxy partly addresses this, but Rung 1's standing signal may be diluted by far-expiry structure.
- Rung 2 `tbbo` cost on liquid 0DTE strikes is the largest unknown; the sample-first gate bounds it.
- Combo-print signing may be irreducibly noisy enough that Rung 2 returns "not determinable" — which is itself a valid, publishable conclusion.
