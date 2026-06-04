# SPX 0DTE OTM Butterfly — Backtest Findings

> **Status: COMPLETE.** Caches: ±100pt JL (`cbbo_cache_jl_10min`) and the clean
> ±250pt (`cbbo_cache_fly_10min`, $10.49) + OI (`oi_cache`, $0.08). The doc reads
> chronologically (§1–8 on the ±100pt cache; §9 + FINAL VERDICT on the clean ±250pt
> cache). **Headline: the author's specific recipes show no edge, but a real intraday
> mean-reversion signal — fade the morning move (`anti`), the inverse of the deck —
> is net-positive across all 4 OOS folds, of modest strength. See FINAL VERDICT.**

## Method
- Spec: `docs/superpowers/specs/2026-06-03-otm-butterfly-backtest-design.md`
- Core: `src_v2/backtest/butterfly_core.py`; eval: `scripts/analytical_butterfly_eval.py`
- Cache: `data/cbbo_cache_jl_10min` (±100pt, 942 days 2022-05-11→2026-03-30)
- Entry **09:31** (cbbo-1m interval-end stamping), body **1σ** (ATM-straddle EM),
  direction **21-period Hull MA** vs **coin-flip** baseline, width by **VIX9D regime**
  (vix1d only starts 2023-04). Settlement at the final grid anchor (15:51) via the
  chain forward.

## Primary question
Does the OTM 0DTE butterfly (body 1σ, ≥9:1) have positive **net-of-cost** expectancy
with positive skew, regardless of side?

## Results (preliminary, ±100pt cache)

### 1. The 9:1 entry gate is almost never fillable
With the gate ON (debit ≤ 10% of width, R2R ≥ 9:1), only **2 of 942 days** traded
(`gate: 905`, `no_debit: 35`). At real 0DTE prices a 1σ-body fly almost never offers
9:1 — the deck's headline structure is largely **theoretical**, not available.

### 2. Trade-every-day (gates-off) — no net edge, costs dominate
Full sample, one trade/day, n≈907 (35 days truncated by the ±100pt window):

| Config | WR | Mean/trade (mid) | Sharpe (mid) | Mean/trade (w/ costs) | Sharpe (costs) |
|---|---|---|---|---|---|
| hull / hold-to-expiry | 37.3% | **−$4.4** | −0.07 | **−$28.4** | −0.43 |
| hull / trail | 56.8% | **+$2.3** | +0.04 | **−$32.8** | −0.65 |
| coin / hold (baseline) | 37.3% | **−$1.3** | −0.02 | **−$25.3** | −0.38 |

- **At mid the structure is ~break-even** (Sharpe ≈ 0) — an efficiently-priced 0DTE
  fly is ~zero-EV held to expiry, exactly as theory predicts. No free lunch.
- **Realistic costs ($0.05/leg slip + $1/leg comm on 4 legs) turn it decisively
  negative** (−$25 to −$33/trade). The 4-leg transaction cost swamps any edge — the
  central execution-realism concern, confirmed.
- **Direction is noise:** hull (−$4.4) ≈ coin-flip (−$1.3). The 21-day Hull MA adds
  no measurable edge over a coin flip — consistent with the deck's own 50/50 premise.

### 3. Regime breakdown (hold, mid) — mixed, small, likely noise
| Regime | n | Mean/trade (mid) |
|---|---|---|
| Zombieland (<17) | 532 | −$32.3 |
| Goldilocks (17–32) | 365 | +$29.6 |
| Chaos (≥32) | 10 | **+$237.0** |

Hold makes money in Goldilocks/Chaos but loses in Zombieland; the **trail** exit
*flips* this (Zombieland +$17, Goldilocks −$20). Opposite signs across exit models on
small per-regime samples ⇒ treat as noise, not a robust regime effect. **Chaos n=10 is
tiny and biased** (only the Chaos days whose wide structure happened to fit ±100pt).

### 4. Free WF-CV sweep (120 combos × 4 folds, select on NET train Sharpe)
Pooled **OOS net mean = +$70/trade** (mid +$94) — but fragile and confounded:
- 3/4 folds picked `anti/hold/fixed:40/gOff`; folds 1–2 genuinely positive OOS (net
  Sharpe 1.9, 1.0), fold 0 flat (−$3.9), **fold 3 blew up** (picked a degenerate
  gates-on combo, n=12, net Sharpe −4.5).
- Full-sample `anti/hold/fixed40/gOff` = **+$76/trade net, Sharpe 0.91**, positive in
  all three regimes — vs `hull` −$25 and `coin` −$21.

### 5. ⚠️ The ±100pt cache contaminates these results
`call`-only (+$28) **and** `put`-only (+$22) are *both* net-positive, yet their random
mixture `coin` is **−$21**. That is impossible without a sampling artifact: the
`no_debit` skips (strikes outside the ±100pt window) are **non-random** — they drop the
big-move / tail days — and differ by side (call drops 69, put drops 9, coin 44). Each
variant is silently scored on a different, truncation-selected subset of days. So the
apparent `anti` edge cannot be trusted on this cache.

### 6. The author's actual guide differs from the deck — and I had tested the wrong setup
Per fattail.ai's full SPX guide (updated Apr 2026), the real trade is **centered, not
1σ-OTM**: body **at-the-money / at the GEX magnet** ("where price settles"), **narrow
10/15/20/25pt** wings (4 VIX regimes), entry **9:45–11:00** (it explicitly calls "a
butterfly at 9:31 gambling"), exit by **profit target (75–350% of risk)**, and — the
core — a **GEX regime filter**: trade centered flies only on **positive-GEX
(mean-reverting)** days; go directional or sit out on **negative-GEX (trending)** days.

Re-tested faithfully on the existing cache (centered ATM body, narrow widths, 09:45,
**no truncation** — no_debit ≈ 3–8):

| Setup | Mean/trade (mid) | Mean/trade (net) | Net Sharpe |
|---|---|---|---|
| centered/10w/hold | +$2.8 | −$21.2 | −1.28 |
| centered/15w/hold | +$2.9 | −$21.1 | −0.74 |
| centered/20w/hold | +$3.3 | −$20.7 | −0.51 |
| centered/15w/**target 0.75×** | −$24.3 | **−$59.3** | −3.36 |
| centered/15w/target 2.0× | −$3.7 | −$30.0 | −1.12 |
| centered/15w/hold/**gates-on** (n=65) | +$32.4 | +$8.4 | +0.41 (t≈0.2, n.s.) |

Two clean conclusions: (a) the centered fly is **~zero-EV at mid, negative net** — same
as every other variant; (b) **profit-targeting makes it strictly worse** — exiting
winners early clips the convex right tail (the whole edge) while losers still run to max
loss, inverting positive skew into negative. This contradicts the guide's own
"take-profit-at-75%" management.

### 7. Calm-morning (+GEX proxy) filter — the regime signal is REAL but execution-bound
A free, lookahead-free pre-trade proxy for the +GEX/range-bound regime: trade the centered
fly only when the 09:31→09:45 drift ≤ X·EM (calm morning). Centered/15w/hold:

| Calm filter | n | Mean (mid) | Net @ $0.05/leg | Net Sharpe |
|---|---|---|---|---|
| none | 937 | +$2.9 | −$21.1 | −0.74 |
| ≤0.3 EM | 615 | +$13.0 | −$11.0 | −0.38 |
| ≤0.2 EM | 444 | **+$20.7** (Sharpe 0.71) | −$3.3 | −0.11 |

The mid edge rises **monotonically** as mornings get calmer (+$2.9 → +$20.7) — a
dose-response that supports the author's GEX thesis: range-bound (+GEX) days favor the
centered butterfly. **Cost sensitivity at calm≤0.2 (mid edge +$20.7):**

| slip/leg, comm/leg | Net mean/trade | Net Sharpe |
|---|---|---|
| $0.10 / $1.00 (realistic retail) | −$23.3 | −0.80 |
| $0.05 / $1.00 | −$3.3 | −0.11 |
| $0.025 / $0.65 (good fills) | +$8.1 | +0.28 |
| $0.00 / $0.65 (mid fills) | +$18.1 | +0.62 |

**Break-even ≈ $0.04/leg slippage.** The edge is real but sits entirely inside the
bid/ask spread of a 4-leg 0DTE structure. Caveat: the calm-filtered mid edge is only
**t≈0.9** (n=444) — suggestive, not statistically robust; the *monotonicity* is the
stronger evidence.

## Verdict
- The deck's literal strategy (1σ-OTM body, 9:1 gate, follow-trend) has **no edge**: gate
  unfillable, Hull ≈ coin ≈ negative.
- The guide's centered/narrow fly is **efficiently priced (~zero-EV at mid)**; profit
  targets make it worse (they clip the convex tail).
- The author's **regime thesis is directionally CORRECT**: filtering to calm/+GEX
  mornings monotonically improves the centered fly (+$2.9 → +$20.7/trade at mid).
- **But the edge is execution-bound** — net-positive only below ~$0.04/leg slippage. For
  realistic retail 0DTE fills ($0.10/leg ≈ half a wing spread) it loses; for a desk that
  reliably gets near-mid fills it's marginally positive (Sharpe ~0.3–0.6). It is an
  *execution-quality* play, not a structural free lunch — and weak (t≈0.9) on this sample.

### 8. Real GEX test (prior-close OI) — the author's core claim does NOT validate
Built a GEX engine (`src_v2/backtest/gex.py`): BS gamma per strike × (call−put OI) →
dealer GEX sign, magnet strike, gamma flip. OI pulled from OPRA statistics (949 days,
$0.08). Sign convention is standard (**+GEX = 60% of days**, a healthy majority). Tested
the author's actual recipe — centered fly **only on +GEX days, body at the magnet**:

| Variant (centered/15w/hold) | n | Mean (mid) | Mean (net @$0.05) |
|---|---|---|---|
| all days | 937 | +$2.9 | −$21.1 |
| **+GEX days only** | 559 | **−$21.3** | −$45.3 |
| +GEX + magnet body | 557 | −$9.8 | −$33.8 |
| −GEX days (control) | 377 | **+$38.9** | — |

The filter **inverts** the claim: the centered fly does *worse* on +GEX days and *better*
on −GEX days. Diagnosis (identical ~30% win rates both groups) shows it's a **vol
confound, not a regime effect**: −GEX days have higher VIX9D (19.4 vs 16.6) → larger EM
(32.5 vs 25.7) → a fixed 15-wide fly is **cheaper** ($2.58 vs $3.10) → higher R2R (5.9 vs
4.7) → bigger convex payoff at the same hit rate. Prior-close GEX is just a noisy proxy
for vol level; it has **no predictive relation to the realized intraday regime** (morning
drift identical, 0.256 vs 0.255). The real driver is fly *cheapness* (width vs EM) — the
deck's original 9:1 framing, come full circle.

**Caveat:** the guide uses a *real-time intraday* GEX overlay (a live product). 0DTE OI is
largely built intraday, so prior-close OI may genuinely miss it. That intraday signal is
unprovable with historical data — but the *pre-open* GEX the guide says to "check before
the open" does not validate here.

### 9. ⚠️ CLEAN ±250 cache OVERTURNS the no-edge verdict for the directional `anti` fly
The earlier "no edge" rested on the truncated ±100 cache, where the directional 1σ-OTM
fly was confounded. On the **clean ±250 cache** (`no_debit` skips = 2, truncation gone),
the **`anti` (fade-the-morning-trend) 1σ-OTM, 40-wide, hold-to-expiry** fly is robustly
net-positive — and the `anti`−`hull` spread (~$132/trade) is a large, consistent intraday
**mean-reversion** signal:

| dir (hold/fixed40/1σ) | MID mean | NET @$0.05 | NET Sharpe |
|---|---|---|---|
| **anti** | +$90.1 | **+$66.1** | 0.77 |
| call-only | +$62.9 | +$38.9 | 0.45 (likely bull-drift) |
| hull (follow) | −$42.3 | −$66.3 | −0.82 |
| coin | −$40.5 | −$64.5 | −0.78 |

**Cost sensitivity (`anti`):** +$46 @ $0.10/leg (Sharpe 0.54) · +$66 @ $0.05 · +$77 @ $0.025
— net-positive even at pessimistic retail fills (break-even ≫ $0.10/leg, unlike the
centered fly's $0.04).

**4-fold sequential OOS (pre-specified `anti/hold/fixed40`, net @ $0.05; `hull` control):**

| Period | `anti` (fade) | `hull` (follow) |
|---|---|---|
| 2022-05→2023-05 | +$8.9 (Sh 0.11) | −$52.0 |
| 2023-05→2024-05 | +$26.2 (Sh 0.31) | −$35.3 |
| 2024-05→2025-05 | +$11.6 (Sh 0.14) | −$0.4 |
| 2025-05→2026-03 | **+$179.2 (Sh 2.10)** | −$147.0 |

`anti` is net-positive in **all 4 folds**, `hull` net-negative in **all 4** — the
direction effect (fade beats follow) is **consistent across every sub-period**.

**The 120-combo WF-CV sweep, by contrast, is unreliable** (pooled OOS +$60.9 but it never
selects `anti`; it picks bull-drift `call/target/55` and degenerate gates-on combos at
n=8–9). Selecting the single best train-Sharpe among 120 combos overfits — the
pre-specified single-hypothesis test above is the honest one.

## FINAL VERDICT
**The author's *specific* recipes do NOT work; but the investigation surfaced one real,
robust effect — the *opposite* of what the deck teaches.**

- **Deck (1σ-OTM, 9:1 gate, follow-trend):** no edge — gate unfillable; following the
  trend (`hull`) **loses in every period**.
- **Guide (centered/narrow at the GEX magnet, profit-target):** ~zero-EV at mid, loses
  net; profit-targets clip the convex tail and make it worse.
- **GEX regime filter (prior-close OI):** does **not** validate — a vol/cheapness confound,
  not mean-reversion. (Real-time intraday GEX remains untested — needs live data.)

- **✅ The real finding — intraday mean-reversion:** *fade* the morning move with a cheap
  1σ-OTM 40-wide fly, hold to expiry (`anti`). Net-positive in **all 4 OOS folds**,
  survives costs to ≫$0.10/leg, and the `anti`−`hull` spread (~$132/trade) is large and
  sign-consistent. **Caveats:** profitability is **marginal in 3 of 4 folds** (+$9–26/trade,
  Sharpe 0.1–0.3) and **concentrated in the recent high-vol fold** (+$179); full-sample net
  t≈1.5 (suggestive, not >2σ). So: a **genuine, economically-sensible signal of modest
  strength**, worth forward/paper validation — not yet a proven money machine. It is also
  the inverse of the deck's "trend-continuation" thesis: the edge is in **fading**, not
  following, the morning move.

## Optional follow-ups
- **Forward/paper-trade `anti`** with realistic SPXW 0DTE fills to confirm the modest edge
  out-of-sample in live conditions.
- **Tune `anti`** (width by regime, body multiple, entry time, partial profit-taking) —
  but guard hard against overfitting given the modest t-stat.
- **Real-time intraday GEX** — the only untested author signal; needs live OI/positioning.
