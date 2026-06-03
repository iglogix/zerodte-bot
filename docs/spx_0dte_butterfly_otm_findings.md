# SPX 0DTE OTM Butterfly — Backtest Findings

> **Status: PRELIMINARY (free Phase-0).** Results below are from the EXISTING
> ±100pt JL cache (`data/cbbo_cache_jl_10min`), which covers Zombieland +
> Goldilocks well but **truncates wide / high-EM days** (Chaos is under-sampled).
> The wider ±250pt backfill (Phase 1/2) is **not yet run** — see "Decision" below.

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

## Verdict
Across **both** of the author's framings — the deck's 1σ-OTM/9:1 directional fly **and**
the guide's centered/narrow/profit-target fly — there is **no net-of-cost edge** when you
trade every day. At mid the structure is efficiently priced (~zero-EV); 4-leg costs make
it a reliable loser. The deck's 9:1 gate is unfillable at a 1σ body; direction (Hull) ≈
coin flip; profit targets destroy the convexity. The only flickers (`anti`/mean-reversion
+$76 confounded by truncation; narrow-ATM gates-on +$8 at n=65) are weak or contaminated.

**The one untested ingredient is the author's actual edge claim: the GEX regime filter**
(centered fly + body at the GEX magnet, *only on positive-gamma days*). My runs trade
every day, so −GEX trending days (price blows through the narrow body → max loss) drown
any +GEX mean-reversion edge. The mean-reversion flicker I *did* see is consistent with
this mechanism.

## Decision: to test the real claim, we need GEX (open-interest) data
GEX = Σ(OI × gamma × …) per strike — it needs **per-strike open interest / dealer
gamma**, which the cbbo quote cache lacks. Testing the author's actual strategy faithfully
requires pulling OI (a separate Databento schema) and computing the daily GEX regime +
magnet strike, then: trade the centered fly only on +GEX days, body at the magnet.
- The ±250pt backfill (in progress) cleanly enables the **−GEX directional/wide** variants.
- A GEX/OI pull would enable the **+GEX centered** variant — the author's core edge.
**Pending user direction on whether to pursue the GEX data.**
