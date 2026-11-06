# SPX 0DTE Expected-Move Fade — findings (2026-08-16)

## Verdict, in two parts

**1. The performance claims are REFUTED.** Claimed 85% win rate and ~35%/yr.
The 85% reproduces almost exactly (85.1–86.9%) — but **only with the stop
deleted**, and that variant loses $16–25/trade. With the stop as specified the
win rate is 67.5% against a realised break-even of 71.4%. The **most generous
assumption stack in the whole study** — the more permissive of two touch
definitions, mid fills on entry, mid-triggered take-profits, the trail on, an
*unattainable* stop that fills exactly at its trigger, and zero commission —
yields **+1.83%/yr**. That is ~19× short of the claim, and every step toward
realism reduces it.

**2. Whether the structure is net-negative is NOT established by this study.**
Per-trade P&L ranges from **−$9.26** (realistic fills, 1-minute stop model) to
**+$8.78** (mid fills, idealised continuous stop) depending on one modelling
choice: **how a stop-market fills**. My data is 1-minute resolution and
demonstrably inflates stop slippage (§4). The honest reading is
**approximately break-even, sign unresolved** — not "loses money".

**3. The trigger is not anti-informative.** An earlier draft of this document
claimed the EM-band touch was actively harmful and that "no signal at all beats
the fade." That was a **basis error on my part** — I compared the fade (which
trades only touch days) against unconditional arms trading all days. On matched
days the fade *beats* unconditional entry, significantly (§5).

Two corrections in this document were forced by review, and both are preserved
rather than quietly folded in: the touch-definition correction (§1, §5) and the
control-basis correction (§5). They changed the conclusion twice.

- Claim source: YouTube `jvlGGTVE234`, Jamal Gi — spec in [spx_0dte_em_fade_spec.md](spx_0dte_em_fade_spec.md)
- Engine: [em_fade_core.py](../zerodte/backtest/em_fade_core.py) + [noarb.py](../zerodte/backtest/noarb.py)
- Driver: [backtest_em_fade.py](../scripts/backtest_em_fade.py) · tests: [test_em_fade_core.py](../tests/test_em_fade_core.py)
- Data: `data/cbbo_cache_fly_10min/` — 1,022 day files, real CBBO, **2022-05-11 → 2026-06-11**. Zero new spend.

---

## 1. Two touch definitions

"Sell when price **touches** the band" is not one testable rule at minute
resolution, and the choice is worth $12.50/trade.

| basis | definition | trades | bias |
|---|---|--:|---|
| **close** | parity spot at the minute *close* crosses the band | 518 | misses transient touches; **selects persistent moves** |
| **intrabar** | SPY 1-min high/low, each minute shifted by that minute's own measured parity-vs-SPY basis | 593 | catches transient touches; **adds SPY microstructure noise** |

**Intrabar is the headline** — a trader reacts to the wick, not the close — and
it is the more favourable to the claim. `close` is the conservative variant.

Close-only detection fires solely on moves *still beyond the band at the close*,
i.e. a continuation-enriched subsample. That manufactured a spurious "the touch
is a continuation signal" result which vanishes under intrabar detection (§5).

---

## 2. Headline (reference config: 20Δ, 10-wide, TP +50%, stop −100%, f=0.7, middle regime, $2.00 commission)

| metric | **intrabar** | close |
|---|--:|--:|
| touch days / 1,022 sessions | 597 (58.4%) | 520 (50.9%) |
| trades | **593** | 518 |
| win rate | **67.5%** | 62.0% |
| realised break-even win rate | **71.4%** | 71.3% |
| $/trade | **−$9.26** (t = −2.01) | −$21.76 (t = −4.27) |
| day-block bootstrap 95% CI | [−$17.40, −$0.98] | [−$31.37, −$11.92] |
| avg win / avg loss | +$67.06 / −$167.44 (1:2.50) | +$66.96 / −$166.33 (1:2.48) |
| max drawdown | −$5,873 | −$11,563 |
| at 1% / 2% of account risked per trade | −1.6% / −3.1%/yr | −3.2% / −6.4%/yr |

Buy-and-hold SPX over the identical span: **+16.4%/yr** (price only).

Pre-registered arithmetic said break-even needs **66.7%**. Realised break-even is
**71.4%** — commissions and stop overshoot add ~4.7pp. That gap is the strategy.

---

## 3. The execution bracket

30 cells = {entry depth 0.5 = mid … 1.0 = far side} × {optimistic, middle,
pessimistic exits} × {trail off, on}.

- **close basis: 30/30 negative**, −$9.10 … −$32.74.
- **intrabar basis: 29/30 negative**, −$0.15 … −$19.23. The one positive cell is
  *f*=0.5 / optimistic / trail on: **+$2.18, t = 0.63**, CI [−$3.70, +$8.21].

| f | regime | trail | close $/tr (t) | **intrabar $/tr (t)** |
|--:|---|---|--:|--:|
| 0.5 | optimistic | on | −9.10 (−2.27) | **+2.18 (+0.63)** |
| 0.5 | optimistic | off | −13.66 (−2.74) | −0.40 (−0.09) |
| 0.5 | middle | off | −16.52 (−3.19) | −2.91 (−0.63) |
| **0.7** | **middle** | **off** | **−21.76 (−4.27)** | **−9.26 (−2.01)** |
| 0.7 | pessimistic | off | −28.33 (−5.31) | −15.00 (−3.10) |
| 1.0 | middle | off | −26.20 (−5.38) | −12.37 (−2.80) |
| 1.0 | pessimistic | on | −27.93 (−6.70) | −19.23 (−5.21) |

---

## 4. The stop-fill problem — the study's binding limitation

**This dominates every other modelling choice.** A stop-market triggers on the
package mark and fills at the market. I observe the mark once per minute, so the
fill is booked at the *next observed* mark, which has already drifted past the
trigger. That is partly real slippage and partly an artifact of the observation
grid — and the two are separable by varying the grid:

| observation grid | reference: median stop loss / credit | reference $/tr | optimistic: median | optimistic $/tr |
|---|--:|--:|--:|--:|
| 1 min (this study) | 1.147 | −21.76 | 1.091 | −9.10 |
| 2 min | 1.186 | −23.26 | 1.150 | −9.70 |
| 3 min | 1.233 | −24.51 | 1.155 | −11.88 |
| 5 min | 1.268 | −22.76 | 1.194 | −11.55 |
| **continuous (fills at trigger)** | **1.014** | **−11.10** | **1.014** | **−0.56** |

Overshoot rises **monotonically with grid coarseness** and converges to 1.014
(= commission alone) under continuous monitoring. That is the signature of a
sampling artifact. **A 1-minute cache cannot measure stop slippage**, and mine
overstates it.

The effect on the answer, on the intrabar basis:

| arm | $/trade | t | 95% CI | %/yr @1% risk |
|---|--:|--:|---|--:|
| reference (realistic entry, 1-min stop) | −9.26 | −2.01 | [−17.40, −0.98] | −1.56% |
| reference + **perfect stop** | **−0.04** | −0.01 | [−7.02, +7.13] | −0.01% |
| optimistic corner (mid fills, 1-min stop) | +2.18 | 0.63 | [−3.70, +8.21] | +0.37% |
| **CEILING: mid fills + trail + perfect stop** | **+8.78** | **2.77** | **[+3.26, +14.53]** | **+1.48%** |
| CEILING + zero commission | +10.78 | 3.40 | [+5.26, +16.53] | +1.83% |

**A perfect stop is not attainable** — a stop-market fills at the market, never at
its trigger — so +$8.78 is a hard upper bound, not a result. But the reference
config with a perfect stop is **−$0.04/trade**, i.e. exactly zero. The truth lies
between −$9.26 and −$0.04 at realistic entry fills, and this data cannot say
where. Measured combo half-spread at exit is only ~0.075 pts (~$7.50/spread), so
realistic stop slippage is small — pushing the estimate toward the zero end.

**Consequence for claim C9** ("worst case with the stop is −100% of credit"): my
earlier statement that the realised tail is 2.4–5.1× worse is **withdrawn as a
finding about the strategy**. At 1-minute sampling the median stop loses 1.09–1.15×
the credit; under continuous monitoring it loses 1.014×, i.e. **C9 essentially
holds**. The overshoot I measured is mostly cache resolution.

---

## 5. Controls — corrected for day mix

**The error.** My first table compared the fade (520 touch days) against
unconditional and random arms trading all ~1,020 days. Days that touch ±1 EM are
systematically bad days to sell 0DTE premium — the same unconditional put spread
earns **+$3.89/day on non-touch days and −$29.33 on touch days**. Comparing
across those populations is a basis error (structural move 6), and it produced
the false conclusion that random entry beat the fade.

Corrected, restricted to the fade's own trade days, paired per day:

**intrabar basis (headline), fade trades 593 days**

| arm | all-day $/tr | **matched $/tr** | **paired vs fade** | t |
|---|--:|--:|--:|--:|
| **fade (the claim)** | −9.26 | **−9.26** | — | — |
| with-trend mirror | −10.81 | −11.01 | −1.75 | −0.23 |
| uncond 10:00 put | −12.98 | **−25.05** | **−15.67** | **−2.59** |
| uncond 10:00 call | −16.77 | **−30.87** | **−21.65** | **−3.11** |
| rand day+side (3 seeds) | −8.33 … −11.75 | −14.96 … −16.61 | −5.70 … −7.52 | −0.90 … −1.19 |

**close basis, fade trades 518 days**

| arm | all-day $/tr | matched $/tr | paired vs fade | t |
|---|--:|--:|--:|--:|
| **fade** | −21.76 | −21.76 | — | — |
| with-trend mirror | −3.94 | −3.85 | **+17.92** | **+2.13** |
| uncond 10:00 put | −12.98 | −29.33 | −7.57 | −1.15 |
| uncond 10:00 call | −16.77 | −28.41 | −6.80 | −0.88 |
| rand day+side (3 seeds) | −8.33 … −11.75 | −16.44 … −18.78 | +2.99 … +5.32 | 0.42 … 0.77 |

**What this actually shows.** On the headline basis the fade is the **best** arm
tested and significantly beats unconditional entry. The touch does two opposite
things: it **selects a worse-than-average day** (all arms lose more on touch days)
but a **better-than-arbitrary moment within it** (entering at the touch beats
entering at a fixed 10:00 by $16–22/trade). Those roughly cancel, leaving the
structure's standing cost.

The with-trend mirror's advantage is a **close-basis artifact**: +$17.92 (t=2.13)
on the continuation-enriched close sample, −$1.75 (t=−0.23) on the intrabar
sample. A skew-controlled test (same instrument, opposite trigger direction, so
vol skew and drift cancel) shows all four cells within noise on the intrabar
basis. **"The touch is a continuation signal" is withdrawn.**

*Consistency check:* the unconditional and random-day arms never read the touch,
so they must be bit-identical across touch bases. They are, despite independent
simulation under different config keys.

---

## 6. Can any knob fix it? — exit rules, strike selection, and the unifying number

### 6a. The exit rule is a dial, not an edge

The full TP × stop grid (5 × 6 = 30 cells, intrabar, realistic fills) is
**negative in every cell**, best −$5.39/trade at TP 75% / SL 0.5× (t = −2.23,
best-of-30 in-sample). Adding an **underlying-breach** stop — exit only when spot
actually reaches the short strike, which the loss diagnostic specifically pointed
to — does not help either:

| TP | stop rule | WR | avg win | avg loss | ratio | break-even WR | $/trade |
|--:|---|--:|--:|--:|--:|--:|--:|
| 25% | mark 0.5× | 44.4% | +102.07 | −93.93 | 1:0.92 | 47.9% | −7.01 |
| 50% | mark 1× (as specified) | 67.5% | +67.06 | −167.44 | 1:2.50 | 71.4% | −9.26 |
| 75% | mark 0.5× | 70.0% | +32.38 | −93.46 | 1:2.89 | 74.3% | **−5.39** |
| 50% | underlying breach | 77.2% | +66.79 | −289.77 | 1:4.34 | 81.3% | −14.39 |
| 75% | underlying breach | 86.2% | +32.40 | −300.71 | 1:9.28 | 90.3% | −13.67 |
| 75% | no stop, hold to EOD | 92.6% | +32.41 | −572.38 | 1:17.66 | 94.6% | −12.46 |

Read the last three columns. **The exit rule moves the win rate anywhere from 44%
to 93%**, with the payoff ratio moving in lockstep from 1:0.9 to 1:17.7 — and the
realised win rate lands **2–4 points below break-even in every single row**.

Two consequences. First, **any win rate can be manufactured by choosing an exit
rule**, so quoting one without its paired ratio carries no information — which is
exactly the trap in claim C1. Second, the exit rules slide along an iso-EV curve
that sits below the line everywhere; they cannot fix this.

**Why the specified stop is badly matched to the structure:** among the worst 60
trades, **98% never saw spot penetrate the short strike**. Median adverse move
after entry is 21.8 pts toward a strike sitting 32.4 pts away. The market travels
two-thirds of the way, the spread reprices on delta and vol, the 2×credit mark
stop fires — and a position that was never structurally threatened books a full
−100% of credit. Those trades cluster on high-EM (35.2 vs 26.3 pts) and high-VIX
(20.2 vs 17.2) days.

### 6b. Strike selection

Delta targeting is a *premium-normalising* rule: the ATM straddle falls 39%
across the entry window (27.82 → 16.95) but the credit on a 0.20Δ / 10-wide
vertical falls only 12% (1.46 → 1.28), because the strike migrates from 31 pts
OTM to 18 pts OTM. So the obvious question is whether a rule that *doesn't*
normalise premium away can harvest the richer early session.

Three alternatives were added and swept (21 configs, intrabar, realistic fills):
fixed **distance** (15–40 pts), fixed **credit** ($0.75–$2.50), and a fixed
fraction of the day's **EM** (0.4–1.0). They work as intended — a fixed 30-pt
strike collects **1.25 early vs 0.58 late**, more than double, where delta
targeting managed 1.45 vs 1.29.

**All 21 are negative**, −$4.29 to −$19.96/trade, and monotone: the more premium
the rule sells, the more it loses.

| strike rule | median Δ | median credit | mean risk | $/trade | t | RoR |
|---|--:|--:|--:|--:|--:|--:|
| delta 0.10 | 0.099 | 0.58 | $944 | **−4.29** | −2.11 | −0.45% |
| delta 0.20 (reference) | 0.199 | 1.38 | $864 | −9.26 | −2.01 | −1.07% |
| distance 20pt | 0.238 | 1.70 | $828 | −7.83 | −1.30 | −0.95% |
| credit $2.00 | 0.269 | 2.00 | $803 | −9.49 | −1.49 | −1.18% |
| distance 15pt | 0.298 | 2.22 | $781 | −14.46 | −1.93 | −1.85% |
| em_frac 0.4 | 0.351 | 2.72 | $735 | −19.96 | −2.33 | −2.72% |

**Entry-time interaction: the hypothesis is capturable and it does not pay.**
Early (09:35–10:30) is negative in **21/21** rules, and the loss scales with how
hard the rule reaches for early premium: em_frac 0.4 collects 2.96 pts early and
loses **−$31.09**; delta 0.10 collects 0.59 and loses **−$4.75**. Regressing
early-window $/trade on early-window credit across the 21 rules gives a slope of
**−$7.70 per point of credit sold** (r = −0.835). *Caveat: those 21 rules trade
the same days, so that r is close to mechanical and is not 21 independent
observations.*

Late (11:30–13:00) looks better and **isn't**: positive in only 9/21 rules, max
|t| = 1.45 across all of them, and per-trade the early-minus-late difference runs
−13.7% to +8.8% with |t| ≤ 1.71 and inconsistent sign. **No entry-time edge.**

### 6c. The unifying number

Normalise P&L by the premium actually sold — `pnl / (credit × 100)` — and the
whole study collapses to one constant:

| strike rule | median credit | edge, 1-min stop | $/trade | edge, perfect stop | $/trade |
|---|--:|--:|--:|--:|--:|
| delta 0.10 | 0.58 | **−7.8%** | −4.29 | +1.4% | +0.87 |
| delta 0.20 | 1.38 | **−7.4%** | −9.26 | −0.3% | −0.04 |
| credit $2.00 | 2.00 | **−4.8%** | −9.49 | +0.3% | +0.63 |
| distance 15pt | 2.22 | **−6.2%** | −14.46 | −1.0% | −3.99 |
| em_frac 0.4 | 2.72 | **−7.3%** | −19.96 | −3.0% | −8.89 |

Across a **4.7× range of premium sold**, and across every entry window, the edge
is a near-constant **−5% to −8% haircut on whatever you sell**. That single fact
explains every negative result in this document:

- the exit-rule dial (§6a) can put the win rate anywhere from 44% to 93%, but
  you pay the haircut on every setting;
- strike selection only changes *how much* premium you sell, and the edge is
  proportional to it with a negative constant;
- entry timing has no gradient because the haircut has no time structure;
- delta 0.10 loses least simply because it sells least.

**And under a perfect stop the haircut collapses to −3%…+1.4%, i.e. zero.** The
haircut *is* the friction, and §4 already showed it is dominated by stop-fill
modelling that 1-minute data cannot resolve. This is the same unresolved-sign
limitation seen from a different angle, not a new result — but it is the sharpest
statement of what the strategy actually is: **a mechanism for paying a spread,
scaled by how much premium you choose to sell.**

---

## 7. Claim-by-claim (intrabar, reference execution)

| # | Claim | Stated | Measured | Verdict |
|---|---|---|---|---|
| C1 | Win rate | **85%** | 67.5% with the stop; **86.9% without it** | **FAILS** (85% costs $16/trade) |
| C2 | Reward:risk | 1:2 | 1:2.50 | holds |
| C3 | Annual return | **~35%/yr** | −1.6%/yr; **+1.83%/yr at the study's absolute ceiling** | **FAILS** |
| C4 | Hold time | 20–30 min | median 34 min; 43.7% ≤30 min | roughly holds |
| C5 | Put beats call | yes | put −$10.13 · call −$8.34 | **FAILS — reversed** |
| C6 | Best in high vol | yes | VIX<15 −$5.41 · 15–20 −$7.37 · 20–28 −$8.75 · **≥28 −$41.06** | **FAILS** |
| C7 | Earlier touch better | yes | −$2.21 / −$12.51 / −$8.62 / −$4.73 by bucket | not monotone |
| C8 | Risk 3–4/10 | — | maxDD 6.8% / 13.6% of account at his sizing | plausible |
| C9 | Worst case = −100% of credit | — | 1.014× under continuous monitoring | **essentially HOLDS** (§4) |

---

## 8. Walk-forward

48-config grid, pre-simulated then folded, ≥12-trade gate.

| ladder | OOS re-opt (close) | OOS re-opt (**intrabar**) | OOS frozen (**intrabar**) |
|---|---|---|---|
| 100/50 | −$11.20 (t=−1.99) | +$3.31 (t=0.53) | −$9.29 (t=−1.93) |
| 130/65 | −$16.69 (t=−2.73) | +$7.66 (t=1.36) | −$8.55 (t=−1.73) |
| 160/80 | −$20.41 (t=−2.84) | −$4.67 (t=−0.70) | −$7.92 (t=−1.60) |

All intrabar OOS cells are indistinguishable from zero with the **sign flipping
on fold length**. Winner stability 38–44%; the recurring winner is
**0.25Δ / 20-wide**, the maximum-risk grid corner — the D3 leverage-artifact
signature (buying variance, not signal). *Caveat:* re-opt and frozen OOS samples
differ in size (n=646/565/527 vs 526/497/474) because winning configs often use
`em_multiplier=0.8`, which produces more touches — so the two columns are not a
matched comparison.

**No configuration survives out of sample.**

---

## 9. Support/resistance loophole — tested and closed

Spec A9 ("place the short strike just beyond a recent S/R level") is proxied
ex ante by the **prior regular session's high/low**, with the delta band *not*
relaxed (relaxing changes the instrument rather than applying the filter, D15).

| arm | n | $/trade | t |
|---|--:|--:|--:|
| reference, close | 518 | −21.76 | −4.27 |
| + S/R proxy | 448 | −23.41 | −4.33 |
| + S/R proxy, +10pt buffer | 412 | −24.39 | −4.34 |
| reference, intrabar | 593 | −9.26 | −2.01 |
| + S/R proxy | 501 | −7.48 | −1.53 |

Drops 70–106 trade days; improves nothing beyond noise. **Loophole closed.**

---

## 10. Direction of bias

| assumption | choice | favours |
|---|---|---|
| touch definition (headline) | intrabar, the more generous | **the claim** |
| commission | $2.00 vs the spec's $2.60 | **the claim** |
| entry latency | fill in the same minute the touch is seen | **the claim** |
| missing Jan–Apr 2022 | excluded by data availability; hostile to short premium | **the claim** |
| S/R nudge (A9) | proxied ex ante | neutral |
| trail (A7) | mechanised, tested both ways | **the claim** |
| **stop fill** | **next observed 1-minute mark** | **strongly against the claim (§4)** |
| EOD | flat at 15:50 rather than expiring worthless | against the claim |
| max 1 trade/day | per A3 | against |

The bias vector is **mixed**, not stacked — and the single largest term (stop
fill) runs *against* the claim, which is why the sign of the net result is not
resolved. Sensitivity arms (intrabar, *f*=0.7 middle): entry delay +1 min
−$19.94; EM = 0.8× straddle −$4.86; width 5 −$8.50; width 20 −$5.38;
delta 0.10 −$4.29; delta 0.25 −$7.96; **no stop 86.9% win, −$16.27**;
TP 25% −$7.95; TP 75% −$8.66.

---

## 11. Data quality

- **Whole-study invariant sweep**, all 130,991 trade rows / 205 config keys:
  `0 < credit < width`, `exit_debit ∈ [0, width]`,
  `pnl ∈ [−(width−credit), +credit]`, `exit > entry ≥ touch`, strike ordering and
  OTM-ness, stop never filling better than its trigger — **zero violations**.
  Only residual is a ≤$0.05 difference from persisting `credit`/`exit_debit`
  rounded to 3dp.
- **Look-ahead, tested by perturbation:** rescaling every quote strictly after the
  entry minute by a random 0.3–2.8× changed **0** entry decisions across 6 entry
  modes on a 70-day sample. Staleness fallbacks walk backward only.
- **Ladder edge:** interior on 130,983 / 130,991 (99.994%); the 8 exceptions are
  all 0.10Δ WF grid arms and random-day controls, **none in any reported config**
  (both reference configs 518/518 and 593/593 interior).
- **Quote integrity:** 0 crossed, 0 negative, 0 NaN across 4.18M sampled quotes.
- **No-arb guard is live:** rejects 4.7–11.3% of sampled verticals depending on
  how far OTM the sampler reaches; 0 within the traded ~20Δ region, so
  held-position glitch minutes are a verified zero.
- **Parity spot** matches `CbboStore.derive_spot` to median 0.0000 (max 0.15 on a
  40-day × 4-minute sample; **max 1.595** over 17,505 all-minute observations).
  This is intra-repo consistency, **not independent validation**.
- **Autocorrelation** of the trade series is −0.004 / −0.011 / −0.006 at lags
  1/2/3, so the i.i.d. t-test is appropriate; every headline also carries a
  moving-block bootstrap (block = 10, 20k draws).
- Sharpe/trade at the close reference is **−0.19** (× √518 = −4.27).

**Disclosures.** Window starts 2022-05-11, so **Jan–Apr 2022 is not covered**.
Intrabar touches are detected via SPY, not SPX, so some are ETF bounce the index
never made. Band anchor is 09:31, not the 09:30 open. Strike grid is 5 pts
(realised short delta median 0.199 vs 0.20 target, so not binding). 2026 is
partial.

---

## 12. Reconciliation with the prior in-repo fade result

`docs/orb_dir_debit_excursion_diag_2026-06-18.md` reports `fade_breakout`
195 trades / +$7,599 / $39 per trade / 74% win. **Re-run and reproduced exactly**
(along with with_trend 233 / +$31,207 / 80% and opposite_level 127 / +$7,148 /
77%), so both studies sit on a working pipeline.

That study priced exits at **mid** and fired take-profits on the **first mid mark
to cross** the target (failure-catalog B2), so its +$39/trade is an optimistic
bound. Its trigger (ORB breakout + retest), strikes (0.5×EM OTM, 20-wide) and
management (no stop, TP 0.75) all differ. Both studies agree there is no
demonstrable fade edge; neither establishes a strongly negative one.

---

## 13. Scope deviations

1. **No `zerodte/strategies/em_fade.py` or live config.** The plan routed this
   through `BacktestOrchestrator`/`PositionMonitor`, which fires exits on the
   first mark whose *mid* crosses the target (B2) and cannot express "a resting TP
   limit fills when the ask reaches it, at the limit" or "a stop-market fills
   worse than its trigger". A purpose-built minute-resolution path engine was
   written instead. It reuses `zerodte` rather than forking it: `CbboStore` for
   quotes, `flux/iv_compute.parity_spot` for the spot definition,
   `flux/iv_compute.compute_iv_rows` for per-strike IV + delta, and
   `backtest/vix_loader` for the regime split — all **delegated, not
   reimplemented**, so the study cannot drift from the repo's definitions of its
   two most load-bearing quantities. Every mark routes through a new **shared**
   `backtest/noarb.py` primitive, promoted so the guard cannot be lost between
   simulators again (F1). What is *not* reused is the backtest harness
   (`BacktestOrchestrator`, `PositionMonitor`, `OrbEngine`, `LiveStrategy`,
   `bt_tool_handlers`) — that is the whole point of the deviation.
2. **Walk-forward runs on the study engine's own grid**, not
   `run_wf_method_strategy.py`, for the same reason. The three-tier ladder and
   frozen-vs-reoptimised comparison are preserved.

---

## 14. Bottom line

The claim's arithmetic is internally consistent — 85% at TP+50%/SL−100% really
would give ~35%/yr — and **its numbers do not survive contact with real quotes**.
The 85% is reproducible only by deleting the stop, and that variant loses money.
The absolute ceiling of this study, built from a stack of deliberately generous
and partly unattainable assumptions, is **+1.83%/yr**.

What the study does **not** establish is that the structure loses money. At
realistic entry fills it sits between **−$9.26 and −$0.04 per trade** depending
entirely on how a stop-market is modelled, and 1-minute data cannot resolve that.
**Approximately break-even, sign unresolved.**

The EM-band touch is neither the mean-reversion signal claimed nor the harmful
signal an earlier draft of this document asserted. It selects a bad day and a
good moment within it, and those roughly cancel.

**Do not trade this as claimed.** Settling whether the residual is slightly
positive or slightly negative would need sub-minute quotes and a live resting-stop
study — and the prize for winning that argument is ~1%/yr.

### Reproduce

```bash
.venv/bin/python -m pytest tests/test_em_fade_core.py -q
.venv/bin/python -m scripts.backtest_em_fade sanity --jobs 22
for T in close intrabar; do
  for M in measure analyze controls wf close; do
    .venv/bin/python -m scripts.backtest_em_fade $M --touch $T --jobs 22
  done
done
.venv/bin/python -m scripts.orb_dir_fade_compare --start 2022-04-18 --end 2026-06-11
```
