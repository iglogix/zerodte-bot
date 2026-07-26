# Skyline (4-butterfly SPX 0DTE) — backtest verdict

**Source:** YouTube interview, Saurabh Khurana with John (`is_YUBFgR2U`).
**Study date:** 2026-07-26. **Verdict: REFUTED at any realistic fill.**
The backtest reproduces almost exactly at mid fills — and the entire edge is
smaller than one half-spread across the structure's 16 legs.

---

## 1. The strategy as specified

SPX 0DTE, entered 09:45 ET, **held to cash settlement**, no stop, no
adjustments. Four overlapping butterflies, 12 distinct strikes, 16 contracts.
With `A` = ATM strike (spot rounded to nearest 5):

| # | Type | Legs | Width | Note |
|---|---|---|---|---|
| BF1 | calls | +1 `A`, −2 `A+25`, +1 `A+50` | 25/25 | balanced |
| BF2 | calls | +1 `A+40`, −2 `A+60`, +1 `A+75` | 20/15 | 5pt "floor" (broken wing) |
| BF3 | puts | +1 `A−45`, −2 `A−20`, +1 `A+5` | 25/25 | balanced |
| BF4 | puts | +1 `A−80`, −2 `A−60`, +1 `A−35` | 25/20 | 5pt "floor" (broken wing) |

Overlaps 10 / 5 / 10 points; peaks at `A−60, A−20, A+25, A+60`; **deepest
valley at `A` itself**; outer bounds `A−80 … A+75` (≈ ±1% at SPX 7580).

**The structure parse reconciles to the penny with the video's stated
economics.** With the video's $1,255 debit the payoff gives max loss $755
(at the ATM valley *and* outside the range), max profit $1,245 at three peaks
and $745 at the 20-wide one — all four numbers exactly as stated on air.
Locked in `tests/test_skyline_structure.py` (6 tests).

`ASSUMPTION:` the two *balanced* flies can be puts or calls (the video says so);
BF1=calls / BF3=puts was used, i.e. each body kept OTM. Tested all four
assignments — identical at mid (parity), and this one is the **cheapest at real
fills**, so the verdict is not an artifact of the choice.

---

## 2. Data

| | |
|---|---|
| Marks | Databento **OPRA cbbo-1m**, real NBBO, `data/cbbo_cache_fly_10min` |
| Provenance | ±250pt @5pt around the 09:30 open, calls+puts; `scripts/backfill_butterfly_cbbo.py` @ `fa856f6`, fetched 2026-06-03…06-12 |
| Span | 2022-05-11 → 2026-06-11, **1,008 tradable days** (97.8% of cache days fully quotable) |
| Settlement | ^GSPC close — the actual SPXW PM-settlement basis (price index, no adjustment question) |
| Marginal cost | **$0** — existing cache reused, no new metered fetch |

**Sanity gate — all passed.**
- Second source: 33/33 sampled marks **penny-identical** to ThetaData NBBO.
- Settlement cross-check: parity-forward(15:59) − ^GSPC close = mean **+0.01pt**,
  median −0.04, sd 2.70 (≈ one minute of index drift — noise, not bias).
- No-arb scan: **0 violations / 7,521 rows** within `[500−debit, 2500−debit]`.
- Strike selection driven off the option tape's own parity spot (never bars).
- Smoke-run one month on real data before the full run.

---

## 3. Claim-by-claim

| Claim | Claimed | Measured (his window, **mid fills**) | |
|---|---|---|---|
| Structure economics | debit 1255 → maxloss 755, maxprofit 1245 | exact | ✅ |
| Win rate | 53% | **51.7%** | ✅ |
| $5,000 → ending capital, 2 contracts, ~2yr | $78,956 | **$70,177** | ✅ |
| CAGR | 297% | **288%** | ✅ |
| Range ≈ 1.8–2% of spot | 1.8–2% | 2.04% | ✅ |
| **MAR ratio** | **5.5** | **1.68** | ❌ |
| **Drawdown "27% at 1 contract"** | 27% | **171% of equity; account negative on day 7** | ❌ |
| 25-pt overnight-drop filter helps | helps | +$5.42/day, **t = +0.20** (noise) | ❌ |
| "back tested to perform in different market conditions" | robust | 2022 −$26/day, 2023 −$50/day *at mid* | ❌ |
| No stop loss needed | true | true — defined risk, cash settled | ✅ |

He is not inventing numbers. Win rate, ending capital and CAGR all reproduce
within ~11%. The failure is entirely in **fills** and **risk disclosure**.

---

## 4. Fill bracket — where it dies

Full cycle, 1,008 days, 1 lot, $0.65/contract commission:

| Fill | Total | $/day | Sharpe | Win% | maxDD | CVaR5 |
|---|---|---|---|---|---|---|
| **mid** (OptionOmega default) | +$10,861 | **+$10.77** | 0.29 | 48.9% | **$30,401** | −$929 |
| mid + 25% of spread | −$46,212 | −$45.84 | −1.21 | 46.5% | $58,477 | −$991 |
| half-spread (touch) | −$103,284 | −$102.46 | −2.68 | 43.2% | $103,284 | −$1,080 |

Median total quoted spread across the 16 legs = **1.33 pt = $133/lot**, so
crossing costs ~$66/lot. Over his own two-year window the mid-fill edge is
**+$66.37/day** — i.e. **the entire claimed edge is one half-spread.**

> Breakeven: you must fill within **30%** of the half-spread over his window,
> and within **5%** over the full cycle. On a 16-leg 0DTE package, every day,
> that is not achievable.

Per year at mid — the claimed window is the good half:

| Year | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| $/day @ mid | −26.47 | −50.49 | +18.05 | +65.66 | +64.46 |
| $/day @ mid+25% | −114.69 | −97.47 | −25.67 | +3.72 | +14.16 |

---

## 5. Walk-forward — nothing survives

5 rolling folds (train 252d → val 63d → test 126d), 96 configs
(entry × structure shift × fixed/level-scaled offsets × drop filter),
scored at mid+25%:

| | n | $/day | Sharpe | Win% |
|---|---|---|---|---|
| Re-optimised each fold | 524 | **−$48.37** | −1.32 | 43.3% |
| Frozen (video config) | 550 | **−$17.49** | −0.46 | 48.2% |

Frozen beats re-optimised again (failure-catalog D8). **The best of all 96
configs is −$13.42/day even in-sample.** Shifting the structure so a *peak*
(not the valley) sits at ATM does not rescue it; level-scaling the offsets to
% of spot makes it worse (−$51/day at mid+25%).

---

## 6. Why it loses

The deepest valley is deliberately placed at the money — which is the single
most likely place for SPX to settle.

| settle − ATM | days | % | avg P&L @mid |
|---|---|---|---|
| **−10 … +10 (ATM VALLEY)** | 243 | **24.1%** | **−$584** |
| −30…−10 (peak −20) | 168 | 16.7% | +$617 |
| +10…+35 (peak +25) | 267 | 26.5% | +$422 |
| beyond ±80 (outside range) | 54 | 5.0% | ≈ −$790 |

Median |settle − ATM| is 22 pt. The modal outcome sits on the max-loss point,
and ~29% of days are a near-full loss. The peaks pay well enough to offset that
*at mid* — but only just, and the margin is the spread.

---

## 7. Capital: the number that actually breaks it

"$5,000 with two contracts" risks 2 × ~$900 = **$1,800/day, 36% of the account,
every single day.** Starting on the first day of his own window, at mid fills:

```
2024-06-12  +968   equity 5,968      2024-06-20  −142   equity   878
2024-06-13 −1,831  equity 4,137      2024-06-21 −1,701  equity  −823   ← ruined, day 7
2024-06-14  −551   equity 3,587      2024-07-01 −1,803  equity −3,578
2024-06-18 −1,791  equity 1,020      worst 10-day stretch: −$7,594
```

The $78,956 endpoint is only reachable by a backtester that lets equity go
negative. **A 1-lot account needs ≈$35k to survive the full-cycle mid-fill
drawdown of $30,401** — ~14× the capital advertised for 2 lots. MAR 5.5 is
unreachable; the realised figure in his own framing is 1.68.

---

## 8. Verdict

**REFUTED as a tradable strategy. The backtest itself is honest and
reproducible; it is the fill assumption and the risk disclosure that fail.**

- **Real:** the structure, its stated economics, the ~53% win rate, ~290% CAGR
  at mid, and the claim that no stop loss is needed. Nothing here is fabricated.
- **Artifact:** the profit. It is the bid-ask on 16 legs, handed back by
  mid-price fills. Frozen and re-optimised walk-forward are both negative.
- **Understated:** the risk. 36% of capital at risk daily, ruin in 7 sessions
  at the advertised size, 2022–2023 losing even at mid.

The host's throwaway line — *"boy, is that a commission eater?"* — was directionally
right and an order of magnitude too kind: commissions are ~$10/lot, the spread
is ~$66/lot, and the whole edge is ~$66/lot.

**Caveats.** (a) Data ends 2026-06-11; the video's SPX-7580 example postdates
the cache, so that specific trade could not be reproduced (the structure's
economics were verified against it arithmetically instead). (b) His live "1,000%
YTD" is on a self-defined capital-at-risk denominator and explicitly includes
*variations* and discretionary intraday trimming — a different strategy from the
backtest, and not testable. (c) Gap-stressed Calmar is not reported: this is a
long-premium, defined-risk structure, so gaps cannot exceed the stated max loss.
(d) Fills are modelled leg-by-leg from consolidated NBBO; a skilled trader
working the package may beat mid+25% on some days — but must beat mid+30%
*on average, every day* for the strategy to break even.

*Backtests flatter live trading.*

---

### Reproduce

```bash
.venv/bin/python scripts/skyline_prep.py            # ^GSPC closes -> backtest_data/skyline.db
.venv/bin/python scripts/skyline_sanity.py          # data sanity gate
.venv/bin/python scripts/skyline_theta_check.py     # 2nd-source mark reconciliation
.venv/bin/python scripts/backtest_skyline.py --tag FULL --out sky_full_nofilter.csv
.venv/bin/python scripts/backtest_skyline.py --start 2024-06-12 --end 2026-06-11 \
    --tag repro --out sky_repro_nofilter.csv
.venv/bin/python scripts/skyline_analysis.py
.venv/bin/python scripts/skyline_wf.py
.venv/bin/python scripts/skyline_context.py         # market-context / event conditioning
.venv/bin/python scripts/skyline_opt.py             # structural variants + WF
.venv/bin/python -m pytest tests/test_skyline_structure.py -q
```

---

# Part II — Is it optimizable? Context, events, and what to avoid

**Short answer: not reliably.** A disciplined walk-forward (select on train, gate
on val, score OOS) over structural variants × context filters returns
**+$1.47/day, Sharpe +0.03** — zero. Every in-sample filter that looks better
than that is either selection-biased or, in two cases, a *date filter in
disguise*.

## 9. The bar any filter has to clear

Mean spread cost is **$43/lot** at mid+25% and **$87/lot** at half-spread,
against an unconditional mid edge of **$10.77/day**. A filter must roughly
quadruple the mid edge on the days it keeps, and keep enough days to matter.

## 10. Two traps hit and fixed (worth knowing generally)

A first pass found `VIX1D between 10 and 18` looked excellent (+$65/day at a
realistic fill). It was fake: **VIX1D history starts 2023-04-24**, so the filter
silently deleted all of 2022 and early 2023 — the worst period. A coverage guard
now asserts every conditioning series has >90% per-year coverage, and it caught
VIX9D too (series stops mid-2026). Filter retention by year exposes the softer
version of the same trap:

| filter | 2022 | 2023 | 2024 | 2025 | 2026 | |
|---|---|---|---|---|---|---|
| `no big event / OPEX` | 80% | 80% | 80% | 82% | 80% | honest |
| `EM 19–37pt` | 62% | 60% | 64% | 59% | 57% | honest |
| `VIX 13–22` | **19%** | 85% | 78% | 83% | 77% | ≈ "skip 2022" |
| `spread cap` | **16%** | 82% | 82% | 62% | 61% | ≈ "skip 2022" |

The two best-looking in-sample configs both leaned on the bottom two rows.

## 11. Walk-forward result

6 structural variants × 8 filters = 48 configs, 5 rolling folds, OOS at mid+25%:

| | n | $/day | Sharpe |
|---|---|---|---|
| **Re-optimised each fold (the honest number)** | 373 | **+$1.47** | **+0.03** |
| Frozen as-taught, no filter | 625 | −$24.19 | −0.63 |

Per-fold OOS: −55, −136, **+146**, 0, +50. One fold carries everything, and no
config had a positive *train* Sharpe in the first three folds — the whole family
loses through 2022–2024.

## 12. The geometry is the defect — and it is partly fixable

Full sample at mid+25%:

| variant | contracts | spread $/lot | $/day |
|---|---|---|---|
| as taught | 16 | $133 | −$45.84 |
| EM-scaled offsets (×EM/26.3) | 16 | $130 | −$19.03 |
| drop the two outer broken-wing flies | 8 | $85 | −$38.53 |
| **both: inner two flies, EM-scaled** | **8** | **$85** | **−$6.56** |

Two changes recover ~$39/day of the $46/day loss. The reason is visible in the
move distribution — P&L by realised move in units of that day's implied move (EM):

| \|settle−ATM\| / EM | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.2 | >1.2 |
|---|---|---|---|---|---|---|
| % of days | 13.2% | 11.3% | 11.1% | 12.3% | 19.8% | 32.1% |
| $/day @mid | **−$723** | −$241 | +$152 | +$384 | **+$442** | −$54 |

The structure wants a 0.6–1.2 EM move. Median EM at 09:45 is 26.3pt but ranges
17–44pt, so the *fixed* peaks (±20/25/60) sit at a different quantile every day:
the two outer peaks at ±60 are ~2.3 EM and almost never pay, while the valley at
±0 sits on the modal outcome. Even fixed, `inner_em` is still breakeven — and it
is no longer Skyline, it is a two-butterfly EM-scaled structure.

## 13. Patterns to avoid

Mid fills, full sample. **Caveat: only one row clears |t| > 2, and that is about
what 14 tests produce by chance.** The value here is the consistent direction
across independent categories, not any single cell.

| Condition | days | $/day *in* | $/day *out* | t |
|---|---|---|---|---|
| Flat open (\|09:32→09:45 drive\| < 1.8pt) | 200 | −$53 | +$27 | −1.71 |
| Overnight gap **down** > 20pt | 191 | −$46 | +$24 | −1.44 |
| Any of CPI / NFP / GDP / PPI / OPEX | 200 | −$45 | +$25 | −1.45 |
| Low implied move (EM < 19pt) | 185 | −$35 | +$21 | −1.16 |
| High implied move (EM > 37pt) | 211 | −$18 | +$18 | −0.79 |
| Monday or Tuesday | 396 | −$16 | +$28 | −1.14 |
| Widest-spread quintile (>2.0pt = $360/lot) | 202 | −$46 | — | — |
| **Overnight gap up > +24pt** | **200** | **+$118** | −$16 | **+2.86** |

Individually: CPI −$62/day, NFP −$57, GDP −$60, OPEX −$56, PPI −$24, FOMC +$35
(n=32). All five non-FOMC event types are negative, which is more persuasive than
any one of them. Mechanism is the same throughout — events and gaps produce moves
that overshoot ±80, quiet opens pin into the ATM valley, and both tails come with
wider spreads.

**The one candidate signal.** "Only trade after a >+20pt overnight gap up" gives
+$63/day at a realistic fill on 58 days/year, Sharpe 1.69, maxDD $7,169 — and it
has a smooth threshold plateau from +15 to +30 (not a grid corner). But: it is
positive in only 3 of 5 OOS folds and carried by one; its **level-invariant %
version is materially weaker** (+$43 vs +$62/day stitched), which suggests the
absolute point threshold is partly fitting the index-level trend; the first half
of the sample is breakeven at realistic fills (+$0.84/day) versus +$102/day in
the second; and it was chosen after seeing the whole sample. Note the claimant's
own filter has the right *sign* (avoid gap-downs) but the tradable asymmetry — if
it is real at all — is on the **up** side, which he explicitly does not condition on.

Stacking every avoid-condition leaves 69 days/year at +$65/day (Sharpe 1.73,
maxDD $9,195) at a realistic fill. Treat that as an **in-sample upper bound built
with hindsight**, not a forecast: the same machinery run honestly walk-forward
returns zero.

## 14. Bottom line on optimizability

The structure pays ~$133/lot of bid-ask every day to buy a lottery ticket whose
modal outcome is its own max-loss point. Fixing the geometry (EM-scale, drop the
outer flies) removes most of the loss but stops short of profit. No context or
event filter survives walk-forward. If something here is worth pursuing it is the
gap-up condition on an EM-scaled two-butterfly structure — as a *new hypothesis
needing forward OOS*, not as a validated edge, and not as Skyline.
