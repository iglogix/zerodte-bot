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

## Verdict (preliminary)
- The advertised structure (1σ body, 9:1 gate, follow-the-trend) shows **no edge**: the
  gate is unfillable, `hull` ≈ `coin` ≈ negative net.
- The sweep surfaced a **plausible but unproven** counter-thesis: **`anti` (fade the
  morning trend) / hold-to-expiry** is net-positive — consistent with intraday 0DTE
  mean-reversion. It is economically sensible but **confounded by ±100pt truncation**
  and weak (t≈1.7), so it can be neither confirmed nor dismissed on the current data.

## Decision: the ±250pt backfill is now worth running
Earlier I expected the wider backfill to be a formality. The free sweep changed that: it
produced an **ambiguous, truncation-contaminated positive**, not a clean negative. The
±250pt window (`no_debit` skips → ~0) removes the side-dependent selection bias and
includes the tail days, giving a **clean, trustworthy** WF-CV answer on whether the
`anti`/mean-reversion edge is real. Cost ≈ **$11 Databento**, ~1 cycle. Recommended.
**Pending user go/no-go.**
