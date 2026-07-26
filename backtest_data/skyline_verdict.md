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
.venv/bin/python -m pytest tests/test_skyline_structure.py -q
```
