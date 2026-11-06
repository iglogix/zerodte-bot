# Burrito Butterfly Strategy — Backtest Design Spec

**Date:** 2026-06-14
**Source:** the course PDF "The Burrito Butterfly Strategy" (not published) (thetaprofits.com / "Dan")
**Objective:** Extract the strategy precisely, build a faithful backtest, and **maximize Sharpe**.
**Status:** Design — awaiting user review before implementation plan.

---

## 1. Strategy extraction (from the deck)

The Burrito Butterfly is a **hybrid options-income** strategy on cash-settled index options
(SPX, optionally XSP for micro adjustments). It is explicitly **discretionary** — the deck gives
structure and intent but no exact execution rules. We therefore treat the discretionary choices as
**parameters to sweep / optimize**, validated by walk-forward CV rather than curve-fit.

### 1.1 The structure (per the "Anatomy of the Setup" slide)
Two components, both placed in the direction of a **directional bias**:

- **Core butterfly (1-2-1):** Buy 1 ATM, sell 2 OTM, buy 1 far-OTM. **15-point wings.** The short
  (body) strikes sit OTM in the bias direction, so the expiration **peak ("ATM Target") is OTM**.
  - Bullish call example, body offset `O` OTM: `+1 K0`, `-2 K0+W`, `+1 K0+2W` with `W = 15`.
- **OTM vertical debit spread:** 5-point wide, in the bias direction. **Structural link:** the spread's
  long leg reuses the butterfly's **outer (far) wing** strike, extending reach further OTM.
  - Default read: `+1 K0+2W`, `-1 K0+2W+S` with `S = 5`.

> **Geometry caveat.** The deck text is internally contradictory ("buy 1 **near**-OTM" in the spread
> bullet vs. "uses the same long strike as the **outer wing**" in the structural-link bullet). The
> slide screenshot is generic platform filler (an unrelated **BITA stock** 60P/65P vertical), so it
> does **not** pin the geometry. Resolution: **default to the outer-wing read** but make spread
> placement (`inner` vs `outer` link, and `S`) **swept parameters** so the data decides.

### 1.2 The edge claim & lifecycle
1. **Enter directional & cheap.** Net debit = capital at risk. Loss accrues slowly ("Delayed Loss
   Engine"); profit pulls toward the OTM peak.
2. **Lock into "free" status.** Once the directional thesis prints **+5–10% on capital**, buy the
   **opposing-side** debit spread cheaply. Both expiration tails then float **above zero** ("Magic
   Money Rainbow" / floating double-winged butterfly) — a locked 5–10% regardless of direction.
3. **Theta bow.** Over the hold, theta decay bows the T+0 line upward, lifting all price ranges above
   breakeven; max yield if the index drifts back into the central tent.

### 1.3 Risk management & defenses
- **Uncle Point (1:1):** if the trade dips **−5–10%** (symmetric to the +5–10% target), terminate.
- **Preemptive Wing Lock:** on adverse test of the downside threshold, buy the opposing debit spread
  immediately — small defined loss, freezes the risk profile.
- **OTM Option Freeze:** buy a cheap (~$1) far-OTM option in the trend direction to halt structural
  bleeding and flatten the P&L curve.
- **Choo-Choo Train:** add a second overlapping 15-pt butterfly into the adverse move to widen the
  delta-neutral tent (instead of a hard stop).
- **Clawback spreads (valley-of-death defense):** the structural **valleys** between the fly body and
  the outer debit wings create expiration gaps that only bite in the **final 2–3 hours of expiry**
  ("Friday Droop"). When settlement is at risk of landing in a valley, inject a corrective spread/fly
  at those exact structural strikes ("Hole Injection") to **lift underwater wings** and flatten the
  profit landscape. XSP micro overlays are the deck's preferred clawback vehicle for fine sizing.

### 1.4 Instrument rules
- Cash-settled indexes only (SPX primary; XSP as micro adjustment) — no early assignment.
- Backtest models **SPX/SPXW**; XSP overlays approximated as 1/10-scale SPX legs (no separate XSP cache).

---

## 2. Decisions locked in the interview

| Topic | Decision |
|---|---|
| **Horizon** | Sweep **entry DTE 0 → ~5** as an optimizable parameter. |
| **Direction signal** | **Optimizable menu**: Hull-MA trend, Markov regime gate, momentum/ORB, and a both-sides/neutral baseline. |
| **Lifecycle** | **Full path-dependent** state machine (lock + all defenses + clawback). |
| **Granularity** | **EOD checkpoints for DTE≥1**; **10-min intraday** (`cbbo_cache_fly_10min`) for the **DTE=0** leg. |
| **Objective** | Maximize **Sharpe** (daily equity returns), validated by **WF-CV**. |

---

## 3. Architecture

**Approach:** one lifecycle engine with a pluggable **clock**, reusing Iron Duck primitives.

### 3.1 Modules
| Path | Purpose | Depends on |
|---|---|---|
| `src_v2/backtest/burrito_core.py` | Pure logic: structure builder, pricing wrappers, lifecycle state machine, `summarize`. No I/O. | `iron_duck` (Smile, `_fill`, `black_*`, `parity_forward`), `jl_core` helpers |
| `src_v2/backtest/burrito_clock.py` | Yields ordered price-steps for a live trade: EOD steps from `DuckStore` (DTE≥1) or 10-min steps from the 0DTE cache (DTE=0). One interface, two sources. | `DuckStore`, 0DTE intraday loader |
| `scripts/backtest_burrito.py` | Single-config runner → `trades.csv` + metrics JSON. | `burrito_core`, `burrito_clock` |
| `scripts/wf_cv_burrito.py` | Walk-forward CV sweep over the param grid; reports OOS metrics per fold. | runner + `wf_cv_regime` patterns |
| `tests/test_burrito_core.py` | Unit tests: geometry, pricing, each state transition, settlement. | engine |

### 3.2 Reused primitives (from `src_v2/backtest/iron_duck.py`)
- `DuckStore` (== `HedgeHogStore`) — EOD NBBO chains `data/iron_duck_eod/SPXW/<expiry>.pkl`,
  structure `{expiry, days{trade_date:{(strike,'C'/'P'):(bid,ask,flag)}}}`, 2020-01-03 → 2026-06-12.
- `Smile` — Black-model IV-by-strike with linear interpolation + `call_delta`/`put_delta_abs`
  (for delta-targeted strike selection and repricing strikes missing from a snapshot).
- `_fill(q, side, k)` — mid ± `k`·half-spread fill model (slippage knob).
- `black_call/put`, `*_iv`, `parity_forward` — pricing & PM-settlement forward.
- `summarize(trades, daily_equity)` — Sharpe / PF / WR / max-DD / max-consec-loss.

---

## 4. Trade structure builder (`burrito_core`)

`BurritoConfig` (sweepable fields in **bold**):
- Entry: **`entry_dte`** (0–5), **`entry_anchor`** (time-of-day), **`direction_signal`** (menu).
- Geometry: **`fly_wing_width W`** (default 15), **`spread_width S`** (default 5),
  **`body_offset`** (EM-multiple OTM, default ~1.0×EM), **`spread_link`** (`outer`|`inner`),
  `step` (5), `multiplier` (100).
- Lifecycle: **`lock_trigger_pct`** (+5–10% of debit), **`stop_pct`** (Uncle, −5–10%),
  **`pt_pct`** (profit-target on floated structure), **`defense_policy`** (subset of
  {wing_lock, otm_freeze, choo_choo, clawback}), **`defense_trigger_pct`** (adverse threshold),
  `clawback_window` (final-day / final-N-hours).
- Costs: `slippage_k` (half-spread multiplier), `commission_per_leg`.

`build_legs(forward, em, cfg, side)` → list of `(qty, strike, right)` legs for the fly + linked
debit spread. Put side mirrors the call side. Net debit computed via `_fill(..., 'buy'/'sell')`.

---

## 5. Lifecycle state machine (`burrito_core`)

Evaluated **once per clock step** (EOD or 10-min). Position value via chain mids, falling back to
`Smile` model value for strikes absent from a snapshot.

```
DIRECTIONAL
  ├─ unrealized PnL ≥ lock_trigger_pct·debit ──▶ buy opposing debit spread ──▶ FLOATING
  ├─ unrealized PnL ≤ −stop_pct·debit (Uncle) ──▶ close ──▶ CLOSED(stop)
  ├─ adverse move ≥ defense_trigger_pct ──▶ apply defense_policy (wing_lock / otm_freeze /
  │                                          choo_choo) ──▶ DEFENDED (re-priced structure)
  └─ reach expiry ──▶ settle

FLOATING / DEFENDED
  ├─ unrealized PnL ≥ pt_pct·debit ──▶ close ──▶ CLOSED(target)
  ├─ in clawback_window AND projected settlement ∈ valley ──▶ inject clawback fly/spread
  └─ reach expiry ──▶ cash-settle at parity_forward ──▶ CLOSED(expiry)
```

- **Lock cost** and every defense/clawback adjustment are filled via `_fill` and added to cost basis.
- **Valley detection:** compute the expiration payoff curve from current legs; a "valley" is a local
  region where payoff < 0 between the body and outer wings (or beyond). Clawback injects the standard
  fly/spread at those strikes to raise the floor (capped by `defense_policy` inclusion).
- **0DTE specialization:** with the 10-min clock, the lock/defense/clawback triggers can fire
  intraday; `clawback_window` maps to the final N 10-min bars. For DTE≥1, triggers fire at EOD only —
  a known **optimistic bias on stops** (intraday hit-then-revert is invisible), flagged and
  stress-tested (see §7).

---

## 6. Direction signal menu

Each returns `call` / `put` / `skip` per candidate entry day:
- **`hull`** — 21-period Hull MA trend (reuse `butterfly_core` logic).
- **`markov`** — Markov regime gate (bull/bear/sideways) from the existing skill/infra.
- **`momentum`** — simple intraday momentum / ORB sign.
- **`both`** — enter both sides (or neutral body) as a structural baseline to isolate structure vs.
  directional edge.

Signal choice is a swept dimension; `both` quantifies how much edge is structural vs. directional.

---

## 7. Costs, validation & objective

- **Costs:** half-spread fills (`slippage_k≥1.0` headline), per-leg commission, reported **net**.
  Per standing guidance, **never trust frictionless results**; every headline metric is re-run at
  elevated slippage (`slippage_k` stress grid) and the degradation reported.
- **Objective:** **Sharpe** on daily equity returns (capital base = net debit at risk; equity marked
  per closed trade, optionally per-step for MTM Sharpe).
- **Validation — WF-CV (per project methodology, `docs/wf-cv.md`):** three tiers, **no
  val-leaked selection**, **no OOS-less runs**, prefer a **plateau over the single best cell**.
  Report OOS Sharpe / PF / WR / max-DD per fold; the deliverable validates the *method* (does periodic
  re-optimization of Burrito params hold OOS), not one frozen parameter set.

---

## 8. Deliverables

1. `burrito_core.py` + `burrito_clock.py` + `scripts/backtest_burrito.py` + `scripts/wf_cv_burrito.py`.
2. `tests/test_burrito_core.py` (geometry, pricing, every state transition, settlement, costs).
3. WF-CV results table (in-sample + OOS per fold) and a slippage stress grid.
4. `docs/burrito_butterfly_findings.md` — honest verdict: does the lock/defense lifecycle produce a
   real, OOS-robust Sharpe edge, or is it (like several prior strategies in this repo) refuted once
   execution costs and walk-forward selection are applied? Includes the EOD-granularity stop caveat.

---

## 9. Out of scope (YAGNI)

- Live trading / order routing for Burrito (backtest + research only).
- A separate XSP options cache (XSP overlays approximated as 1/10 SPX).
- Paid intraday SPXW backfill for DTE≥1 (deferred unless EOD results look promising).
- A generic strategy-agnostic simulator framework (build the one engine the strategy needs).
