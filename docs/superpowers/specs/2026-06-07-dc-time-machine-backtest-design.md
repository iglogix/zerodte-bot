# DC Time Machine — Backtest Harness Design (Phase 1)

**Date:** 2026-06-07
**Source:** "Double Calendar Time Machine" — Steve Bernich (Navigation Trading), interviewed by John on Theta Profits (YouTube `JtGW1wNFNIY`).
**Status:** Approved design, pending spec review → implementation plan.

---

## 1. Purpose & scope

Build a **trustworthy backtest** of the "DC Time Machine" strategy on SPX to determine
whether it has a real, execution-robust edge. This is **Phase 1**. A live bot (Phase 2,
an AI-advised strategy plugin) is built **only if** Phase 1 clears the bar.

The decision was made (see brainstorming session) to:
- Phase the work: **backtest first, bot conditional on a GO verdict.**
- Use the **full date range 2022-05 → 2026-04** up front. (This extends past the existing
  0DTE caches' 2026-02 end so the Feb–Apr 2026 his-numbers cross-check window is covered;
  all DC data is fetched fresh from Databento regardless.)
- Build a **dedicated multi-expiration simulator** (Approach A), isolated from the
  battle-tested 0DTE engine, rather than retrofitting the existing `src_v2` engine.

### Non-goals (Phase 1)
- No live order routing, no `LiveStrategy` implementation, no `PositionMonitor` changes.
- No early-scale-out exit logic in v1 (hold the risk-free IC to cash settlement).
- No modification to `tests_locked/` or the existing 0DTE/MEIC code paths.

---

## 2. The strategy (made precise)

### 2.1 Entry — double calendar
On SPXW (cash-settled, European; what the video uses and what the data supports):

- **Put calendar:** sell front put @ `K_p`, buy back put @ `K_p`.
- **Call calendar:** sell front call @ `K_c`, buy back call @ `K_c`.
- Same strike within each calendar; two expirations.
- **Strikes:** `K_p` ≈ 30–40Δ put, `K_c` ≈ 30–40Δ call. Default 35Δ; sweep 15–40Δ.
  (Tighter/closer-to-ATM ⇒ faster transform but narrower final tent; wider ⇒ slower
  transform but wider risk-free IC.)
- **Expirations:** front `F`, back `B = F + gap`.
  - Front DTE sweep: **1–11 trading days** (covers his ~9-DTE "following week" favorite
    *and* short Friday→Monday fronts).
  - `gap ∈ {1, 2, 3, 4}` trading days (video favorite: **1**; the Fri→Mon combo is gap-1
    across a weekend).
- **Entry debit** (buy backs at ask, sell fronts at bid; subject to the fill model in §5):
  ```
  D = (bp_ask + bc_ask) − (fp_bid + fc_bid)
  ```
  where `bp/bc` = back put/call, `fp/fc` = front put/call. ≈ $10/share in his example.

### 2.2 Transform — to a risk-free front-expiry iron condor
Each evaluated minute, compute the **quote-achievable** transform credit:
```
C = (bp_bid + bc_bid) − (put_wing_ask[K_p − W] + call_wing_ask[K_c + W])
```
- Fire the transform the first minute `C ≥ D + W`, where `W` = wing width in points
  (video uses 5; sweep `{5, 10}`).
- **Transformer order (one atomic combo):** SELL back put @ `K_p`, SELL back call @ `K_c`
  (close the longs), BUY front put wing @ `K_p − W`, BUY front call wing @ `K_c + W`.
- **Resulting position:** a pure **front-expiry iron condor** — short put `K_p` / long put
  `K_p − W`, short call `K_c` / long call `K_c + W`. The back-dated legs are gone.

### 2.3 The "risk-free" invariant (verified, not assumed)
Total cash banked over the round-trip = `C − D`. Holding the front IC to cash settlement:
- **Inside the tent** (SPX settles in `[K_p, K_c]`): both verticals expire worthless →
  terminal P&L = `(C − D)` × 100 × qty. **This is the max.**
- **Outside the tent:** the breached vertical loses at most `W` → terminal P&L floor =
  `(C − D − W)` × 100 × qty.

Because the trigger guarantees `C ≥ D + W`, the **floor is ≥ 0**. The simulator asserts
this invariant on every transformed trade — if a transform is recorded with `C − D < W`,
that is a bug, not a trade.

### 2.4 Management & exit
- **−20% mental stop on the un-transformed DC:** if the DC mark-to-market ≤ `−0.20 · D`,
  close the DC. No hard/exchange stop and **no spike-through modeling** — close at the
  next evaluated minute's mid − slippage.
- **EOD if not transformed:** **default = close the DC at EOD** (book the small win/loss).
  **No overnight hold by default.** (See §4 — overnight is a swappable decision seam.)
- **After transform:** **hold the IC to cash settlement** (matches his stated "ideal").
  Early scale-out is a later sensitivity layer, not v1.

---

## 3. Architecture (Approach A — dedicated multi-expiry simulator)

New, self-contained package: `src_v2/backtest/dc_time_machine/`. It reuses project
primitives (Databento fetch, OSI symbol builder, the `(minute, strike, is_call)` quote-map
shape, the WF-CV optimizer) but introduces a separate, multi-expiration data layer and a
trade-lifecycle simulator. **Zero changes to the existing 0DTE/MEIC engine or
`tests_locked/`.**

```
src_v2/backtest/dc_time_machine/
  __init__.py
  multi_expiry_store.py   # session → {expiry → {(minute,strike,is_call):(bid,ask)}}
  dc_backfill.py          # Databento backfill driver (generalized expiry)
  iv.py                   # Black–Scholes ATM-IV inversion + front/back ratio ("Flux")
  spreads.py              # empirical half-spread study (the slippage "experiment")
  fills.py                # parameterized fill model (combo / leg-by-leg, k-sweep)
  policy.py               # OvernightPolicy seam (default CloseAtEOD) + entry filters
  simulator.py            # per-trade minute-stepped lifecycle → trade record
  strategy.py             # DcTimeMachineParams (the sweepable param object)
  flux_scanner.py         # reconstructed "Flux" scan → candidate entries (also a Phase-2 screen)
  run_backtest.py         # CLI: backfill | study-spreads | simulate | wf-cv | crosscheck
```

### Lifecycle (per candidate entry)
```
enter DC (debit D, fill model) at entry minute t0
for each evaluated minute t ≥ t0 within the entry session:
    mark DC P&L
    if DC P&L ≤ −0.20·D:           close DC (loss); STOP
    compute achievable transform credit C(t)
    if C(t) ≥ D + W:               fire transform (fill model); GOTO settle
at EOD (still a DC):
    OvernightPolicy.decide(...)     # default: CLOSE → book DC P&L; STOP
                                    # (alt policies measured as counterfactual; §4)
settle:
    hold front IC to cash settlement at expiry F (parity-derived settlement)
    terminal P&L = bank(C−D) − breach_loss(settlement, K_p, K_c, W)
    assert floor (C−D−W) ≥ 0
```

`Minute evaluation` cadence: 1-minute (the transform *timing* is the edge; 10-min
downsampling — as used by the fly/jl caches — is a later cost lever, not the default).

---

## 4. Overnight hold = a swappable decision seam

The EOD-not-transformed branch calls `OvernightPolicy.decide(trade_state) → {CLOSE, HOLD}`.

- **Default:** `CloseAtEOD` (always CLOSE) — no overnight risk in the v1 baseline.
- **Pluggable alternatives** (deterministic rules, e.g. "hold if DC P&L > −x% **and**
  front/back IV ratio still favorable", capped at N extra days), and in Phase 2 the
  **AI agent**.
- The backtest **measures the counterfactual**: run policy-OFF vs a few candidate
  ON-policies and report the incremental P&L, transform-rate lift, and the added overnight
  tail. This tells us **whether the overnight decision module is worth building at all**,
  and if so, what signal it should key on — rather than assuming it.

---

## 5. Fill model (the load-bearing part — empirical, not assumed)

Lesson carried from the MEIC reconciliation work: **the fill assumption is load-bearing**;
a strategy can look great at mid and lose under realistic fills. So:

### 5.0 Baseline entry time
Default entry is **shortly after the open (09:35–09:45 ET)** — matching his "waited a few
minutes after the open" example — and is itself a **sweepable parameter** (entry-time and the
timing filter in §6 interact). The minute-stepped transform search runs from the entry minute
to EOD of the entry session (extended only by an overnight policy, §4).

### 5.1 Experimental spread study (prerequisite, reported before any P&L)
`spreads.py` measures the **actual half-spread distribution** from the CBBO cache for
exactly the legs we trade — back longs (~35Δ), front wings, front shorts — **stratified by
DTE and time-of-day**. Output: median / p25 / p75 / p90 half-spread in **both $ and ticks**.
Rationale: SPX is very liquid and these legs are near-ATM and **not 0DTE**, so gamma (and
thus spread) is lower than the 0DTE strategies — but a short-DTE Fri→Mon front is closer to
0DTE and must not inherit the low-gamma assumption. The study makes the slippage **observed,
not guessed.**

### 5.2 Fill function
Every fill is `mid ± k · half_spread` with the **empirical** half-spread plugged in and
**`k` swept (0 = mid, 1 = full cross)**. All headline results are reported **as a function
of `k`**, so the edge's fill-sensitivity is explicit rather than hidden in one number.

### 5.3 Combo vs leg-by-leg (transform)
Confirmed in code: the TT client submits a **single 4-leg complex order** at a net limit
(`_place_iron_condor`, `tastytrade_client.py:1011`) — so a headless bot is **not** forced
to leg in; atomic net-credit combo fills work without the GUI.

- **Base model:** the transform is an **atomic net-credit combo**, filled at
  `net_mid ± k · net_half_spread`, bracketed **net-mid → net-cross**. (Matches the GUI
  "one transformer order" workflow and the bot's complex-order capability; no legging risk.)
- **Sensitivity:** **leg-by-leg at per-leg mid** — to see whether working each leg
  independently beats the combo (legging/partial-fill risk flagged, not free).
- **Never** fill the transform at an unachievable price: `C` is always computed from real
  back-bids and front-wing-asks at that minute.

### 5.4 Settlement & scale
Front IC settles to **exact SPX cash intrinsic** at expiry (no fill). CBBO is **top-of-book**;
the report carries an explicit **size/liquidity caveat** per the standing "account for
slippage at scale" rule — quoted size vs intended contract count.

---

## 6. Entry conditions & the reconstructed "Flux" IV-ratio search

The video blends **two distinct edges** under one "ratio" heuristic; we separate and
**measure** them rather than hand-tuning a rule. (Detailed reasoning in the brainstorming
session; summary below.)

- **(a) Structural backwardation (carry):** front IV > back IV (ratio > 1), driven e.g. by
  the weekend effect on Fri→Mon (Monday options mark down non-trading weekend vol). Persistent
  until resolved; deterministic; also yields a **smaller entry debit `D`** ⇒ less profit
  needed to transform ⇒ **faster transform**.
- **(b) Transient ratio spike (timing):** an intraday/event front-IV pop expected to
  mean-revert **down**; a reversion bet that can go wrong (he concedes it's "not a prediction
  tool … provides context").

Both want the **same thing** — front IV falling relative to back IV — which is what profits a
short-front/long-back calendar (differential theta + net-long-vega + ratio-fall). They are not
contradictory; they are different *sources* of the same directional edge.

### Implementation
1. **`iv.py`** — Black–Scholes ATM-IV inversion per expiry from quote mids ⇒ front IV, back
   IV, ratio `= front/back`, plus 5-/20-day history and a short-horizon z-score / "spike &
   reverting" state. This is the reconstructed **Flux** signal.
2. **Separated entry filters (sweepable), with `always-enter` as the baseline** so we can
   measure each one's **lift**:
   - **Structural:** backwardation level + day-of-week (the **Fri→Mon** subset isolated).
   - **Timing:** ratio spike + early downward reversion.
   - Both, and neither.
3. **`flux_scanner.py`** — mirrors his scanner ("0–30 DTE, any gap, rank by min ratio
   drop/rise") emitting candidate entries. Doubles as the Phase-2 live entry screen.

Every simulated trade is **tagged** with ratio level, backwardation flag, day-combo, and
time-of-day, so post-hoc we can see which signal actually pays.

---

## 7. Validation & acceptance ("prove profitability")

1. **His-numbers cross-check (reality anchor).** Simulate his favorite config over
   **Feb–Apr 2026** and compare against his published stats: ~106 trades, ~63% win rate,
   the per-month profit-factor profile, and the ~25% max-profit-hit rate. His are *real*
   fills with discretion, so exact match isn't expected — but order-of-magnitude divergence
   would flag a model/data error.
2. **Walk-forward CV (the overfitting gate).** Reuse `src_v2/backtest/wf_optimizer.py`:
   ratio **5:1, 4 folds, annualized Sharpe**, over the param grid (delta, front-DTE, gap,
   wing width, entry filter, transform threshold ε, fill-`k`). The honest forward guide is
   the **OOS** result, not in-sample.
3. **Sizing.** Fixed-fractional **defined-risk** analysis (mirroring the MEIC sizing toolkit):
   post-transform the IC is defined-risk; **pre-transform the DC max-loss governs sizing** and
   the black-swan-before-transform tail (his stated #1 risk). Report max-DD vs CAGR per `f`.
4. **Fill brackets.** All headline metrics reported across the `k`-sweep and combo-vs-leg
   variants; the **verdict is taken from the realistic end**, not from mid.

### Acceptance bar (default — adjustable)
**GO** if: OOS-positive across folds under realistic fills (`k` toward full-cross), with a
sane CAGR/Sharpe/PF/max-DD under defined-risk fixed-fractional sizing, **and** the his-numbers
cross-check is not wildly inconsistent. Caveats reported honestly: in-sample vs WF-CV gap,
the irreducible un-transformed-DC gap tail, top-of-book size limits.

---

## 8. Deliverables (Phase 1)

- The `src_v2/backtest/dc_time_machine/` package (§3).
- A **2022→2026 multi-expiry cache** (tiered backfill — §9).
- `trades.csv` + a reconciliation/accounting report (per-trade `D`, `C`, transform minute,
  outcome, P&L) in the style of the MEIC toolkit.
- The **empirical spread study** report (§5.1).
- The **WF-CV** report and the **his-numbers cross-check** report.
- A **README verdict**: is it profitable under realistic fills? where is the edge
  (structural carry vs timing vs neither)? what is load-bearing?
- Tests in **`tests/`** (not `tests_locked/`): transform-credit math, risk-free-invariant
  assertion, fill function, IV inversion, store round-trip, a golden end-to-end trade.
- A clear **GO / NO-GO gate** for Phase 2.

---

## 9. Data layer & backfill plan

- **`multi_expiry_store.py`**: pickle keyed `session → {expiry → {(minute,strike,is_call):
  (bid,ask)}}` — the only change from the existing store is one extra expiry dimension; the
  inner quote-map shape and parity/settlement helpers are reused.
- **`dc_backfill.py`**: generalizes `fetch_day_quotes(expiry=…)` (`src_v2/backtest/
  databento_backfill.py`) to fetch **full-day 1-min CBBO** for the needed expiries per
  session, OPRA.PILLAR / `cbbo-1m`, strike band ±~80 pts around ATM (covers 15–40Δ + wings),
  NYSE calendar over 2022-05 → 2026-04 (through the Feb–Apr 2026 cross-check window).
- **Settlement** via the parity-based `settlement_forward()` at the front expiry's own 0DTE
  panel — avoids the SPYx10 basis flagged in prior work.
- **Cost (calibrated from `data/databento_cost_ledger.jsonl`):** ≈ **$0.011 / session /
  expiry**. Tiered:
  1. **Favorite config first** (front ≈ 9 DTE + gap-1 ⇒ ~2 expiries/session) over the full
     date range ≈ **~$25**, ~couple GB → runs the v1 verdict + his-numbers cross-check.
  2. **Widen to the full DTE/gap sweep** (~6–10 expiries/session, ~**$70–110**, ~10–30 GB)
     **only if** the v1 verdict is promising.
  Full *date range* up front (as decided); expiry *breadth* expands on evidence.

---

## 10. Phase 2 preview (only if GO)

Promote the validated logic into a `DcTimeMachineStrategy(LiveStrategy)` plus a
multi-expiry-aware entry engine and position monitor (the unification deferred from
Approach B, now justified by a proven edge). `OrderLeg` gains an `expiration` field.

The **AI advisor** (mirroring the existing haiku `AnomalyDetector` / `RegimeClassifier`
pattern) handles **only** the statistically-uncertain junctures the backtest shows have
value:
- **(a)** the front/back IV-ratio "context" entry call (the Flux judgment) — *iff* the
  timing signal beat always-enter in Phase 1;
- **(b)** hold-overnight-vs-close on an un-transformed DC at EOD — *iff* an overnight policy
  beat CloseAtEOD;
- **(c)** scale-out-vs-hold when price threatens the tent on expiration day.

The deterministic engine owns entry / transform / settlement; the AI advises only at those
points, **logged and overridable**. If Phase 1 finds the structural edge suffices and the
timing signal doesn't pay, the live strategy stays deterministic ("simplicity wins").

---

## 11. Risks & caveats

- **Un-transformed-DC gap tail.** The only real risk window is between DC entry and
  transform; a black-swan gap there forces the DC max loss. Position sizing (defined-risk
  fixed-fractional) bounds it; nothing removes it. This is his own stated #1 risk.
- **Top-of-book data.** CBBO is best-bid/offer, not depth — multi-contract size may not fill
  at the quoted spread. Reported as a caveat, not silently assumed away.
- **Transform-fill optimism.** The single most optimism-prone step; mitigated by the
  `k`-sweep, the combo-vs-leg brackets, and the rule that `C` must be quote-achievable.
- **In-sample vs OOS.** Headline in-sample CAGR is a ceiling; the WF-CV OOS is the forward
  guide.
- **IV reconstruction.** ATM-IV inversion needs a rate and a parity-clean spot; we use
  parity-derived forwards (not SPYx10) and validate the reconstructed ratio against the
  observed Fri→Mon backwardation as a sanity check.
```
