# SPX 0DTE MEIC — Strategy Specification (authoritative)

**Status:** current · validated by walk-forward CV (2026-06)
**Supersedes** the risk-management/exit logic in
[`spx_0dte_meic_early_mgmt.md`](spx_0dte_meic_early_mgmt.md) and
[`superpowers/specs/2026-05-20-meic-strategy-design.md`](superpowers/specs/2026-05-20-meic-strategy-design.md).
Entry logic from [`superpowers/specs/2026-05-21-meic-em-relative-design.md`](superpowers/specs/2026-05-21-meic-em-relative-design.md) stands.
Evidence & reproduction: [`../tools/theta/README.md`](../tools/theta/README.md).

> ✅ **Implemented.** Set `exit_mode: hold_to_expiry` in `config_meic_spx.yaml` to run the
> validated variant — `MeicStrategy.check_exit` then never stops/scales, and the backtest
> engine settles remaining qty at expiry intrinsic (robust parity forward via
> `CbboStore.settlement_forward`). Default `exit_mode: managed` preserves the legacy
> stop + scale-out path unchanged. The engine in hold-to-expiry mode reproduces the
> independent re-simulation to ~1.5% (identical 87.5% win rate).

---

## 1. Thesis (what changed and why)

The canonical MEIC recipe — *"rely on early profits, very tight stop-losses on the shorts"* —
is **value-destroying**. These are **defined-risk** iron condors (the long wings cap each
spread's loss at its width), so the stop *adds* risk rather than removing it, and the early
scale-outs forfeit credit on the ~87% of spreads that expire worthless. Independent
re-simulation and walk-forward CV (see §5) show the validated strategy is the **opposite**
of the recipe:

> **Sell the EM-relative iron condor in tranches, then hold every spread to cash settlement
> — no stop, no scale-out — and control risk purely through position size.**

---

## 2. Instrument & schedule

- **Underlying:** SPX / **SPXW** 0DTE options — European, **cash-settled at the PM close**
  (no assignment, no exit trade required).
- **Trading window / tranches:** 13 entries every 20 min, **11:00 → 15:00 ET**
  (11:00, 11:20, … 15:00).
- **Per-tranche routing:** two **independent** credit spreads (1 put spread + 1 call spread).

---

## 3. Entry logic *(validated — unchanged)*

For each tranche, compute the **expected move** and place EM-relative strikes:

- **EM = ATM straddle** = `call_mid + put_mid` at the strike nearest spot (delta-free).
- `short_call = ceil((spot + short_em_mult·EM) / step) · step`
- `short_put  = floor((spot − short_em_mult·EM) / step) · step`
- `wing = max(step, round(wing_em_frac·EM / step) · step)`;
  `long_call = short_call + wing`, `long_put = short_put − wing`
- **Credit** per side = `short_mid − long_mid`.

**Gates (skip the tranche if any fail):**
- **Credit/width** ≥ `min_credit_width_ratio` on *both* sides.
- **Dislocation guard** *(added 2026-06)* — reject if any of the 4 legs has a quote whose
  spread is `> dislocation_min_spread_pts` **and** `> dislocation_spread_mult × mid`
  (e.g. a stale `0.05/45.00` tick — its mid is a fantasy fill). See the guard in
  `MeicStrategy.search_candidates`.

**Parameters** (`config/config_meic_spx.yaml → strategy/options`):

| param | value |
|---|---|
| `short_em_mult` | 1.0 |
| `wing_em_frac` | 0.5 |
| `strike_step_pts` | 5 |
| `min_credit_width_ratio` | 0.08 |
| `dislocation_min_spread_pts` / `dislocation_spread_mult` | 1.0 / 1.5 |
| `option_root` | SPXW |

**Fills (realistic, validated):** work the **liquid short legs to mid**; buy the cheap,
far-OTM **wings at the ask**. This "M2" model nets ~$2.05M gross in-sample — *better* than a
flat mid−0.10 haircut — and the edge survives even with zero price improvement (§5).

---

## 4. Exit logic *(CORRECTED — supersedes the old recipe)*

**Hold both spreads to cash settlement at the PM close. No stop. No scale-out. No leftover-long
management** (everything cash-settles).

Per-spread outcome at expiry:
- short **OTM** → expires worthless → **keep the full credit**.
- short **ITM** → loss = `min(intrinsic, wing) − credit` (capped at the wing width).

**Why the old exits are removed (evidence in §5):**
- **Stops:** under realistic fills the stop-market-on-shorts (the recipe's "CRITICAL" rule)
  turns +$507k into **−$2.9M** (close-spread-at-market) to **−$9.4M** (short-only, the actual
  spec). On a 0DTE breach the short fills far through the trigger while the held long is worthless.
- **Scale-outs:** holding *without* scale-outs makes **+$1.45M** vs **+$80k** with them —
  buying winners back early forfeits credit you'd keep when most spreads expire worthless.

**Optional entry overlay (survivability, not edge):** an intraday "move-so-far" filter — skip a
tranche if the underlying has already moved `> k·EM` from the day's open — cuts the worst day
(~−$146k → −$95k at k≈0.75) at a small edge cost. It does **not** raise Sharpe; use it only for
tail control. **Do NOT add VIX-regime sizing** — it looked great in-sample but was **rejected by
walk-forward CV** (overfit; see §5).

---

## 5. Validation status

| Check | Result |
|---|---|
| Engine accounting | Independent re-sim reproduces all 22,166 spreads **to the penny** (no look-ahead). |
| Data integrity | ThetaData NBBO == the Databento cache to the penny (both OPRA). |
| **Walk-forward CV** (5:1, 4 folds, ann. Sharpe) | **Plain hold-to-expiry survives OOS: Sharpe 2.71, profitable 4/4 folds, +$967k OOS.** Regime/sizing overlays *overfit* (OOS 2.62, 3/4) and lose to flat. |
| Execution robustness | Realistic fills (M2 $2.05M) beat the conservative baseline; missed fills (>4pt 1-min move) cost ~−3% (≈neutral). |

**Caveat:** in-sample returns are vol-regime-rich (2022/2025 carried it; calm 2023 was the
weak year). The WF-CV — *positive every OOS fold but materially smaller than the in-sample
headline* — is the honest forward anchor. The fat short-gamma tail is structural and
irreducible (e.g. the 2025-04-09 gap moved *after* entry).

---

## 6. Sizing policy *(the binding constraint — not edge or fills)*

> ✅ **Implemented.** `sizing_mode: fixed_fractional` + `sizing_fraction: <f>`. Per tranche,
> `lots = floor(f × equity × weight/total_weight / IC_margin_per_contract)`. Equity is the
> running account value: the **backtest** compounds from `backtest.starting_equity`; **live**
> must inject account net-liq into `MeicStrategy._equity_fn` (the orchestrator wiring is the
> one remaining live step — until then fixed-fractional **fails safe = skips**, never
> oversizes). `sizing_mode: static` keeps the legacy fixed-qty map.
>
> **Granularity floor (important):** integer lots mean a too-small account can't afford a
> contract on the expensive high-vol tranches and **skips exactly the profitable days** →
> near-breakeven. Verified: at **f=0.03** the strategy needs **~$1M** to deploy cleanly
> (\$250k near-breakevens; \$1M→+75%, \$2M→+88%, \$5M→+95% over 3.8y compounded). For
> smaller accounts use a **higher f** (more drawdown) or accept fewer tranches.

The raw backtest uses **fixed** contracts (`tranche_sizing` map, qty 3–6 by time-of-day), so
its headline P&L is a non-compounded sum and "% return on $100k" is meaningless. Trade it
instead with **fixed-fractional defined-risk sizing**:

> Each day deploy total buying power **`BP = f × current_equity`**. Because every IC is
> defined-risk, that BP *is* the day's maximum possible loss → the **worst conceivable day is
> ≈ −1.06·f of equity (ruin is impossible by construction).**

**Daily rule:**
```
BP_budget = f × equity                                   # recompute from equity each day
per tranche: lots = floor( (BP_budget / n_tranches) / IC_margin )
IC_margin  = (max(put_wing, call_wing) − total_credit) × 100   # ≈ $725/contract
```
Size grows with equity after wins, shrinks after losses (which is *why* the drawdown
self-limits). On high-vol days margins are larger, so you naturally trade fewer lots — correct.

**Choosing f (block-bootstrap, preserves bad-day clustering; ~20% drawdown target):**

| f (% equity as BP) | worst day | median DD | P95 DD | in-sample CAGR* |
|---|---|---|---|---|
| **3%** (conservative, P95≈20%) | −3% | −14% | −22% | ~45% |
| **5%** (median≈20%) | −5% | −23% | −35% | ~83% |

\*In-sample ceiling — discount to the WF-CV forward expectation.

**Capital floor:** the full 13-tranche structure needs **~$7.7k BP/day at 1 lot**, so at
f≈3–5% the minimum account is **~$150–250k**. Smaller accounts are granularity-constrained to a
**subset of tranches** (e.g. ~$100k → ~8/13), which is lumpier and lower-return.

---

## 7. Parameter reference (live config)

`config/config_meic_spx.yaml` — note these still include the **deprecated** exit params, which the
validated variant ignores:

- **Used:** entry schedule, `short_em_mult`, `wing_em_frac`, `min_credit_width_ratio`,
  `strike_step_pts`, `dislocation_*`, `option_root`, costs (`commission_per_contract_per_leg`,
  entry slippage), `eod_close_time` (15:50).
- **Deprecated (do not rely on):** `stop_pct_of_total_credit`, `stop_slippage_per_spread`,
  `am/pm_scale_out_tiers`/`_fractions`, `abandon_long_below`, `tranche_sizing` (replaced by
  fixed-fractional sizing; the map is only a relative template).

---

## 8. References

- Research record & reproducible scripts: [`../tools/theta/README.md`](../tools/theta/README.md)
- Original recipe (superseded exits): [`spx_0dte_meic_early_mgmt.md`](spx_0dte_meic_early_mgmt.md)
- Implementation design: [`superpowers/specs/2026-05-20-meic-strategy-design.md`](superpowers/specs/2026-05-20-meic-strategy-design.md)
- EM-relative strikes (current): [`superpowers/specs/2026-05-21-meic-em-relative-design.md`](superpowers/specs/2026-05-21-meic-em-relative-design.md)
