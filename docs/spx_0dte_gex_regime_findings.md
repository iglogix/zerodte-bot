# SPX 0DTE GEX Regime Filter — Feasibility Findings (Rungs 1 + 1.5)

**Date:** 2026-06-17
**Branch:** `feat/gex-regime-study`
**Spec:** `docs/superpowers/specs/2026-06-16-gex-regime-feasibility-0dte-ic-design.md`
**Plan:** `docs/superpowers/plans/2026-06-16-gex-regime-feasibility-0dte-ic.md`
**Verdict:** **REFUTED** — the dealer-gamma (GEX) regime carries no de-confounded predictive edge for 0DTE IC outcomes, and the 0DTE-specific regime that *might* matter is not reliably observable from published OI.

---

## Question

Does special treatment of **negative-GEX vs positive-GEX** regimes improve 0DTE iron-condor (IC) outcomes — and, prior to that, **how reliably can the regime be determined?** Split along an observability line into two testable halves:

1. Does the regime we *can* measure reliably (standing all-expiry GEX; and a 1DTE-as-0DTE proxy) predict 0DTE IC outcomes once volatility (EM) is controlled? → **Rung 1 / Rung 1.5 proxy**
2. Can we determine the regime that *matters* (intraday 0DTE gamma) reliably enough to act on? → **Rung 1.5 OI-evolution gate**

## Data & method

- **OI:** new multi-expiry cache `data/oi_cache_multi/` — OPRA `statistics` OPEN_INTEREST, all expiries ≤120 DTE near spot, **asymmetric −400/+200 pt** window, 5-pt step. **1,017 days** (2022-05-11 → 2026-06-11), **$1.81** total Databento spend. Symbol requests chunked at ≤1,900 (Databento per-request cap).
- **Convergence/no-truncation guardrail:** gamma flip is interior to the fetched window on **100.0%** of days (0/1,017 boundary hits) → the asymmetric window is wide enough; no vol-correlated truncation (the bias that confounded the prior butterfly study).
- **Entry:** **11:00 ET** (first window with full-chain CBBO coverage in `data/cbbo_cache_meic`; also a natural MEIC-style entry). Spot via put-call parity, EM via ATM straddle, σ via Brenner-Subrahmanyam.
- **Regime variable X:** `(S − gamma_flip)/EM` (signed distance to the zero-gamma level, in EM units; >0 suppression, <0 amplification).
- **Dependent Y:** vol-normalized realized move from entry — **max intraday excursion / EM** (primary) and terminal `|Δspot|/EM` (secondary).
- **De-confound:** partial-effect regression `Y ~ 1 + regime + EM + regime×EM` (the regime effect holding EM fixed), plus within-EM-tercile tests and 4-fold **sequential OOS**. Single pre-registered hypothesis — no sweeps.
- **Analyzable sample:** ~943 days (Rung 1) / 994 (proxy) after requiring OI + full-chain quotes at 11:00.

## Rung 1 — standing all-expiry GEX (the reliably-measurable regime)

| Y | Full-sample regime coef (t, p) | EM terciles | Sequential OOS | Raw means (supp / ampl) |
|---|---|---|---|---|
| **max excursion / EM** | −0.031 (**t=−1.04, p=0.30**) | none sig; signs −0.19/+0.14/−0.10 | 0/4 folds sig; `consistent_majority=False` | 1.640 / **1.768** |
| **terminal \|move\| / EM** | +0.003 (**t=+0.10, p=0.92**) | none sig; p 0.42–0.75 | 0/4 folds sig; `consistent_majority=False` | 0.999 / **1.129** |

- The raw means are **directionally correct** — amplification (S<flip) days move more than suppression (S>flip) days — but the gap (~0.13 EM) **evaporates once EM is controlled**. That is the classic **vol confound**: −GEX days coincide with higher EM, and EM already explains the larger move.
- No EM-tercile shows a significant, consistent-sign effect; the lone near-miss (lowest-EM tercile, excursion, p=0.089) is a single in-sample bucket — noise by the pre-registered (no-cherry-pick) bar.
- **Rung 1 FAILS** its success bar (consistent OOS majority AND ≥2/3 terciles same-sign significant).

## Rung 1.5 — 1DTE-as-0DTE proxy + OI-evolution gate

- **Proxy predictive test** (1DTE OI evaluated at the 0DTE kernel, X=(S−flip)/EM vs excursion/EM, n=994): full-sample coef −0.008, **t=−0.33, p=0.74**; OOS 0/4 folds significant. **No edge.**
- **OI-evolution stability** (each expiry's 2DTE→1DTE→0DTE-prior-close snapshots, n=1,012; 3 snapshots on 1,009):
  - dealer-gamma **sign stable across the final days: 75.2%**
  - **median flip-level drift: 2.07 EM**
  - **GATE → FLOW-DRIVEN → Rung 2 REQUIRED.** The *sign* is mostly sticky, but the *flip level* swings ~2 EM across the final days as 0DTE/1DTE OI builds — so the carried-in (observable) snapshot is **not** a reliable stand-in for the realized 0DTE regime. Capturing it would require intraday signed-trade reconstruction (Rung 2).

## IC-dollar confirmation (descriptive)

Canonical single-entry IC: 1.0×EM short strikes, 60-pt wings, hold-to-expiry, mid − $0.10/spread slippage − $4 commission, split by standing-GEX regime sign.

**n=81 priceable days** (strike-grid mismatch — see caveat):

| regime | n | mean P&L | median | win% | total |
|---|---|---|---|---|---|
| suppression (S>flip) | 39 | +$288 | +$521 | 87.2% | +$11,244 |
| amplification (S<flip) | 42 | +$106 | +$609 | 78.6% | +$4,470 |

Spread (supp − ampl): **+$182/trade, Welch t=+0.72, p=0.47 — not significant.** Directionally consistent with "ICs prefer suppression," but on a small, non-random subsample (only days where all four IC legs landed on the cache's strike grid) and statistically indistinguishable from zero. Not relied upon — the 943-day realized-move excursion test (the mechanism that *drives* IC P&L) is the powered evidence, and it is null. (Both regimes are net-profitable here, which is about the canonical IC itself, not the GEX split.)

## Answer to the original question

- **"Does NEG vs POS GEX special treatment affect IC results?"** — Not in any way you can trade. The observable regime (standing all-expiry **and** 1DTE proxy) has **no de-confounded predictive power** on IC-relevant realized moves (Rung 1 t=−1.04 / +0.10; proxy t=−0.33; all OOS-null). The apparent NEG-GEX-bigger-move effect is **what EM already tells you**, not an independent gamma signal.
- **"How reliably can we determine the regime?"** — The observable regime is reliably *measurable* (0% flip truncation; sign 75% stable day-to-day) but **not predictive**. The 0DTE-specific regime that could matter is **flow-driven** (flip drifts ~2 EM), so it is **not reliably determinable from published OI** — only from intraday signed trades (Rung 2), which carries an irreducible trade-signing noise floor.

## Relationship to prior work

Strengthens [[project_butterfly_backtest]]'s prior-close GEX refutation: that study used 0DTE-only OI and an incomplete de-confound. This one uses the **correct all-expiry standing signal**, an **EM-scaled non-truncating window** (0% boundary hits), the **proper de-confound** (partial effect + tercile + OOS), and the **observable 1DTE proxy** — and the GEX regime still shows no usable edge for 0DTE ICs.

## Caveats

- **SPXW daily-expiry rollout:** early-2022 Tuesday/Thursday sessions genuinely lack a same-day 0DTE contract (Tue/Thu SPXW weeklys launched through 2022); those days are skipped. Effective 0DTE coverage is complete from ~late 2022.
- **Entry = 11:00** (not the bot's 09:45) because only the MEIC full-chain CBBO cache (11:00–15:49) spans the period; the morning cache is a narrow delta band unusable for ATM EM/settlement. A 09:45 test would need a new morning full-chain backfill.
- **Realized-move is fill-free** (the clean mechanism test); the IC-dollar layer applies fills but is descriptive (gated off by the null primary).
- **IC-dollar layer is underpowered (n=81):** the MEIC CBBO cache's strike grid is sparser than the 5-pt strike targets, so days where any of the four IC legs (shorts at 1×EM, wings 60 pt beyond) missed an available strike were dropped — a non-random subsample. Snapping target strikes to the cache's available grid would recover most days, but the 943-day realized-move test already settles the verdict, so it was not pursued.
- **Rung 2 not run.** The gate says it *would* be needed to chase the flow-driven 0DTE regime, but with both observable proxies showing zero predictive signal, Rung 2's expected value is low relative to its cost (tbbo trade data + irreducible signing noise). **Recommendation: do not pursue Rung 2 unless a specific intraday-flow hypothesis motivates it.**

## Reproduce

```bash
# data ($1.81): .venv/bin/python scripts/backfill_oi_multi.py --start 2022-05-11 --end 2026-06-13 --max-cost-usd 10
.venv/bin/python scripts/gex_rung1.py --y excursion     # standing GEX, primary
.venv/bin/python scripts/gex_rung1.py --y terminal      # standing GEX, secondary
.venv/bin/python scripts/gex_rung15.py                  # 1DTE proxy + OI-evolution gate
.venv/bin/python scripts/gex_ic_confirm.py              # IC P&L by regime (descriptive)
```
