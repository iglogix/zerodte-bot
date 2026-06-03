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

## Preliminary verdict
As specified, **no robust net-of-cost edge.** At mid the fly is ~zero-EV; realistic
4-leg costs make it a reliable loser (Sharpe −0.4 to −0.65). Direction selection ≈
coin flip. The advertised 9:1 gate is essentially unfillable. The single flicker of
interest is **Chaos / hold (+$237/trade at mid)** — but it is under-sampled and
under-covered by the ±100pt cache, and even a real positive Chaos mean is unlikely to
overcome the per-trade cost drag given how rare Chaos days are.

## Decision: is the ±250pt backfill worth it?
The wider backfill's *only* incremental value is to **de-bias the Chaos regime** (add
the truncated high-vol/high-EM days and full far-OTM strikes) — the one place with a
hint of positive mean and the home of the "fat tail" thesis. Cost ≈ **$11 Databento**.
Given the free evidence already shows no net edge in the dominant regimes and a brutal
cost drag, the wider run is a *thoroughness* step to nail the Chaos claim, not a likely
verdict-changer. **Pending user go/no-go on the spend.**
