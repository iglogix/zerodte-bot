# Hedge Hog — Implemented Trading Logic

Precise description of what `zerodte/backtest/hedgehog.py` actually does,
decision by decision. This is the implementation-level companion to
the course's Hedge-Hog strategy spec (not published)
(the course spec) and
[hedgehog_backtest_findings.md](hedgehog_backtest_findings.md) (results).
Everything below runs on EOD data: one decision point per trading day, at
the 16:00 ET close, using that close's NBBO chain snapshot.

---

## 1. Daily loop

For each trading day in the cache, in order:

1. **Manage** every open position: evaluate the exit triggers (§4) against
   today's closing marks; close at today's quotes if any trigger fires.
2. **Enter** if there is capacity: `open_positions < max_concurrent`
   (default 1) and at least `ladder_every_days` (default 1) since the last
   entry. A position closed in step 1 frees capacity immediately, so the
   base config closes and re-opens **the same day** ("rinse & repeat").
3. **Mark to market** the whole book (realized + open at mid) for the
   daily equity curve.

There is no other entry timing — no time-of-day, no IV trigger, no price
condition. This encodes the course's "no precise entry trigger, recycle
continuously".

## 2. Derived market state (no external inputs)

All greeks come from the option chain itself:

- **Forward / spot**: for a given (day, expiry), take the strikes where
  both call and put have a live bid, find the 3 strikes with the smallest
  |call_mid − put_mid| (≈ ATM), and average `K + C_mid − P_mid`
  (put-call parity, discount ≈ 1). "Spot" for trigger logic is the
  forward of the nearest expiry 1–30 days out.
- **IV per strike**: Black-76 inversion of the put mid against that
  expiry's forward (Brent root-find on σ ∈ [1e-4, 5]).
- **Delta per strike**: Black-76 put delta from that IV.
- **ATM IV**: IV of the nearest-to-forward put with a live bid (OTM side
  preferred, walking up to 6 neighbors if inversion fails).

Strikes with no bid are excluded from selection. A previously traded leg
whose strike has dropped off the tape while spot is above it is marked
nearly worthless (bid 0 / ask 0.05); if spot is at/below it, the day's
marks are treated as missing and the exit retries next day.

## 3. Entry construction (one attempt per day with capacity)

All four steps must succeed or the day is skipped with a recorded reason.

1. **Pick expirations.**
   - Hedge: the listed expiry with DTE in `[25, 35]` closest to 30.
   - Hog: the listed expiry with DTE in `[75, 115]` closest to 100.
2. **Build the hedge (long put vertical), long ~30Δ / short ~20Δ.**
   - Long strike = put with |delta| closest to 0.30.
   - Short strike = put with |delta| closest to 0.20 (must have bid ≥ 0.05),
     **or**, when `hedge_rr_target` is set, the strike whose vertical
     reward:risk `(width − debit)/debit` is closest to the target (the
     course's ~3:1 guide).
   - `debit = fill(long, buy) − fill(short, sell)` (fill model §5).
   - Reject if `debit ≤ 0` or `debit ≥ width`; reward:risk recorded on
     every trade (`rr_ratio`).
   - Note: on SPY 2020–2026 the natural 30Δ/20Δ construction prices at a
     median 4.4:1 (only 8% of entries inside the course's 2.5–3.5 band).
     Enforcing 3:1 (`hedge_rr_target=3.0`) lifts the win rate to ~70% but
     the taught structure still loses (−$688, worse tail −$2,182) and it
     *hurts* the single-cycle variant (+$3,542 → −$960) — see findings.
3. **Build the hog (far short put), ~7Δ stepped until net credit.**
   - Candidates: puts with |delta| in `[0.04, 0.12]` and bid ≥ 0.05,
     sorted by delta ascending.
   - Starting from ~5Δ (anything below `hog_delta − 0.02` skipped), take
     the **first** strike whose sale credit exceeds the vertical's debit:
     `credit = fill(hog, sell) > debit`. This is the spec's "sell ~7Δ; if
     the credit is too small, step the strike closer" rule — it always
     books the *minimum* qualifying credit.
   - If no strike ≤ 12Δ clears the debit → skip day (`credit_le_debit`).
4. **Quality gate (PoP at the downside loss point).**
   - Model the package P&L at the hedge expiry T₁ as a function of S:
     vertical at intrinsic minus debit, hog valued by Black-76 with its
     entry IV (sticky-strike) and the remaining T₁→hog-expiry time.
   - Scan S upward from 0.5×spot: the first S where P&L ≥ 0 is the **deep
     downside breakeven** (the small "droop" near flat spot is tolerated
     by design).
   - PoP = P(S_T₁ > breakeven) under a zero-drift lognormal at the hedge
     expiry's ATM IV. Require PoP ≥ `pop_min` (0.65). (Empirically this
     never binds on SPY — the deep breakeven sits 10–15% below spot.)

Position recorded with: strikes, debit, credit, net credit, hog delta,
PoP, ATM IV, and Reg-T margin
`max(20%·spot − OTM, 10%·strike)·100 + credit·100 + debit·100`.

## 4. Exit triggers (checked daily at the close, first match wins)

| # | Trigger | Condition (as implemented) | Action |
|---|---------|---------------------------|--------|
| 4 | **fast_breach** | spot < vertical short strike AND days held < 10 | close package |
| 1 | **hog_50pct** | hog mark ≤ 50% of entry credit (hog banked ≥ 50% max profit) | close package |
| 2 | **vertical_80pct** | vertical mark − debit ≥ 80% of (width − debit) | close package |
| 3 | **vertical_lt7dte** | hedge expiry < 7 days away | close package |
| — | hard fallback | hedge expiry ≤ 1 day and nothing else fired | close package |

Every trigger's action is **close the whole 3-leg package and recycle**
(the instructor's "~90% of the time" default). Rolling the hog up /
vertical down (his occasional adjustment path) is **not** implemented.
With a 30-DTE entry, trigger 3 alone caps holding at ~23 days; realized
average is ~16–20 days depending on variant.

## 5. Fills, costs, sizing

- **Fill model**: `mid ± fill_k × half-spread` per leg, against us on both
  entry and exit. Base `fill_k = 0.5`; sensitivity runs at 0.0 (mid) and
  1.0 (cross the NBBO).
- **Costs**: $1.00 commission per contract per leg at open, $0 at close
  (TT-style), plus $0.15 regulatory/exchange fees per leg each way.
  3 legs ⇒ ~$4.80 per round trip at 1-lot.
- **Size**: fixed `qty` contracts (default 1). No compounding, no
  per-trade stop — risk is bounded only by the structure itself plus the
  triggers (the hog is naked below its strike).

## 6. Named variants (sweep)

Same engine, parameter overrides only:

- `fill_mid` / `fill_nbbo` — fill_k 0 / 1.
- `hog_5d`, `hog_10d`, `hog_dte75` — hog placement.
- `single_cycle_50` — **both** legs in the same 45–60 DTE expiry
  (the "1-1-1"); construction and triggers unchanged.
- `no_hog_tp`, `no_vert_tp`, `no_fast_breach`, `hold_to_3dte` — trigger
  ablations.
- `no_pop_gate`, `pop_78` — gate ablations.
- `hog_only` (`trade_hedge=False`) — identical expiry/strike selection,
  identical triggers (vertical marks still computed as *virtual* signals),
  but only the naked put is actually filled: 1 leg of costs, margin
  without the debit. Isolates the hedge's contribution.
- `ladder_4x3d` — up to 4 concurrent positions, one new every ≥3 days.

## 7. What this logic does NOT do

- No intraday checks: a breach of the vertical short strike is seen at
  that day's close, not at the touch.
- No rolls, no re-hedging, no delta management between triggers.
- No earnings/dividend calendar (SPY index ETF; ex-div assignment risk on
  short ITM puts is unmodeled).
- No early-assignment modeling on the naked put.
- No multi-symbol portfolio, no portfolio margin, no futures sleeve.
- No position-size scaling by volatility or account equity.

## 8. Outcome (one line)

As specified (two-cycle, hedged): 24.6% WR, −$2,443 / 1-lot over 6.5y —
refuted. The profitable degenerations are the same-cycle "1-1-1"
(+$3,542) and especially its naked-put-only form (+$9,457, Sharpe 1.0),
i.e., a classic 45–60 DTE ~10Δ short put with 50% TP and a ~3-week time
stop. See the findings doc for full tables.
