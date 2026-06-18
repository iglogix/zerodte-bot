# Volume-Profile Opening-Range Breakout (FRVP-ORB) — Backtest & Optimization

**Date:** 2026-06-17
**Strategy source:** [docs/volume_profile.md](volume_profile.md) (YouTube day-trading method)
**Objective:** Maximize Sharpe while remaining statistically sound.
**Verdict: REFUTED.** As-described loses outright; the optimized full-sample
plateau (Sharpe ~1.0–1.25 in-sample) does **not** survive walk-forward CV and is
slippage-fragile. No robust, exploitable edge on any cached underlying.

---

## The strategy (as specified)

1. Build a Fixed-Range Volume Profile over the **09:30–09:45 ET** opening block →
   POC, Value Area High (VAH), Value Area Low (VAL).
2. After 09:45, enter on the **first 5-min candle that closes its body** outside
   [VAL, VAH]. Close > VAH → long; close < VAL → short.
3. Stop = **2 ticks beyond POC**; fixed **2:1** target (reward = 2 × risk).
4. Hard entry cutoff **11:00 ET**. Designed for NASDAQ (NQ) futures.

Traded the **underlying directionally** (matches the video's index/futures
framing) — clean of options-pricing confounds. Returns are R-multiples
(fixed-fractional risk); Sharpe is the per-trade R series annualized by √252.

## Underlying selection (online research)

ORB + volume-profile works best where **liquidity / institutional participation**
produce clean levels and follow-through, with less whipsaw — i.e. index ETFs
(SPY/QQQ) and equity-index futures; high-beta single stocks trend hard but
whipsaw more. 15-min ranges are preferred over noisier 5-min. ([breakorb],
[litefinance], [buildalpha])

Cached underlyings span the spectrum: **SPY** (broad, most liquid), **IWM**
(small-cap, more volatile), **NVDA** (single-stock high-beta). 1-min bars,
2022–2026. The liquidity prediction **held directionally** in results — SPY was
least-bad OOS, NVDA worst — but none cleared the bar.

## Results

### Baseline (exactly as described: or=15, tf=5, rr=2, follow, 11:00 cutoff, 1-tick slip)
| Sym | Sharpe | Exp/trade | Win rate | PF | n |
|-----|-------:|----------:|---------:|----:|--:|
| SPY | −0.20 | −0.018R | 0.35 | 0.97 | 1021 |
| IWM | (neg) | neg | ~0.35 | <1 | ~950 |
| NVDA| (neg) | neg | ~0.34 | <1 | ~740 |

A 33–35% win rate cannot beat a 2:1 target (needs >33.3% *after* costs). The
breakouts mostly fail and revert to POC. **Fading** the breakout (POC as target)
is *also* negative — there is no symmetric reversion edge either.

### Full-sample grid optimization (384 cells, min 100 trades)
Best region is a **coherent plateau**, not a spike: `follow, or=15, entry_tf=15,
rr=3.0`, across all cutoffs/body-buffers.

| Sym | Best Sharpe (in-sample) | Structure | Median cell |
|-----|------------------------:|-----------|------------:|
| SPY | **+1.17** | or15 / tf15 / rr3 | −0.35 |
| IWM | +0.88 | or30 / tf15 / rr1 | −0.38 |
| NVDA| +0.61 | or15 / tf15 / rr3 | −0.94 |

The edge that survives optimization is "let a *confirmed* (15-min) breakout run to
**3R** with a wide POC stop" — 33% win × 3R. Note this already departs from the
spec (15-min not 5-min candle; 3:1 not 2:1). The **median cell is negative** — most
of the parameter space loses.

### Walk-forward CV (5 anchored OOS folds; optimizer re-selects per fold)
This validates the *method* (periodic re-optimization), not a hand-picked cell.

| Sym | OOS Sharpe mean | median | positive folds |
|-----|----------------:|-------:|---------------:|
| SPY | **−0.07** | −0.43 | 2/5 |
| IWM | −0.26 | −0.50 | 1/5 |
| NVDA| −0.46 | −0.51 | 1/5 |

Train Sharpe was +1.1 to +1.6 every fold; OOS collapsed to ~0 or negative. The
plateau is an **in-sample artifact** — the edge is regime-dependent (it appeared
in the SPX 2024–H1’2025 folds and vanished elsewhere), not stationary.

### Slippage fragility (SPY in-sample plateau cell)
| Slip/side | $ | Sharpe | Exp/trade |
|----------:|--:|-------:|----------:|
| 0 tick | $0.00 | +1.25 | +0.132R |
| 1 tick | $0.01 | +1.00 | +0.106R |
| 2 tick | $0.02 | +0.76 | +0.081R |
| 3 tick | $0.03 | +0.51 | +0.055R |
| 4 tick | $0.04 | +0.27 | +0.029R |

Each tick costs ~0.25 Sharpe. Market-style breakout entries + stop exits realistically
cost 2–4 ticks round-trip on SPY, ≈halving an edge that was already in-sample-only.

## Conclusion

The volume-profile ORB is **not statistically sound** and cannot be made so by
parameter optimization:
- As specified it is a **net loser** (33% win rate, 2:1 target).
- The only positive full-sample region is a regime artifact that **fails WF-CV**
  (OOS Sharpe ≈ 0 to −0.5) and is **slippage-fragile**.
- Holds across SPY / IWM / NVDA; SPY (most liquid) is least-bad, confirming the
  liquidity-selection logic but not rescuing the edge.

Consistent with the broader research program: naive opening-range breakout on
liquid indices is chopped out by mean reversion to value (POC), and a fixed R:R
without a real momentum/regime filter has no durable edge.

## Reproduce
```bash
python scripts/volume_profile_orb.py --symbol SPY              # baseline (as-described)
python scripts/volume_profile_sweep.py sweep --symbol SPY --min-trades 100
python scripts/volume_profile_sweep.py wfcv  --symbol SPY --folds 5 --min-trades 60
```
Engine: [scripts/volume_profile_orb.py](../scripts/volume_profile_orb.py) ·
Sweep/WF-CV: [scripts/volume_profile_sweep.py](../scripts/volume_profile_sweep.py)

---

## Addendum 1 — Faithful (trade-level) volume profile

The 1-min OHLCV profile smears each minute's volume across ~40 penny bins (only 15
observations build the 09:30–09:45 profile). Rebuilt the FRVP from **trade-level**
data (XNAS.ITCH `trades`, 09:30–09:45, ~136 true volume-at-price levels/day,
`data/or_trade_vp/SPY.pkl`, ~$2.37 Databento). POC shifts a median $0.12 (p90 $0.64)
vs the smear; VAH/VAL ~$0.06.

**Verdict unchanged.** Baseline still Sh −0.22 (vs −0.20 smeared). Best in-sample
cell +0.82 (vs +1.17). WF-CV OOS nudges from mean −0.07 to **+0.29 (median +0.04,
3/5)** — still ~0, regime-dependent, slippage-eroded. Sharper levels move the lines
without creating a durable edge; the REFUTED call was **not** an artifact of coarse
profiles. Use: `--trade-vp` flag on both scripts.

## Addendum 2 — Premium-SELLING variant (value-area iron condor)

Since breakouts fail ~67% (price respects value), tested SELLING a 0DTE SPX iron
condor with shorts anchored at the faithful VAH/VAL (mapped SPY→SPX via parity-spot
ratio at entry), priced/settled on real CBBO (`data/cbbo_cache`, 882 days). Script:
[scripts/backtest_vp_ic.py](../scripts/backtest_vp_ic.py).

- **In-sample looks good**: baseline (09:45, va=0.70, w=20, hold-to-expiry) Sh +1.63,
  +$65/trade, +$36k/4yr; best cell (va=0.90, buf=5) **Sh +1.93**, WR 56%, PF 1.24.
- **But fails WF-CV.** With TP cells: OOS mean −0.20 (2/5), train Sh up to +11
  (overfit). **Hold-to-expiry only: OOS mean −0.83, median −1.19, 1/5 folds**, and
  the most recent fold is the worst (Sh −1.99, WR 0.41). Optimal strikes aren't
  stationary and the edge has decayed recently.
- **REFUTED as deployable** — in-sample-attractive but not robust; same story as the
  directional version and the repo's prior `orb_em_ic`. Median sweep cell −0.46.

## Addendum 3 — MEIC start-time sweep (does earlier entry help "lately"?)

Tested the rumor that starting the MEIC ladder earlier (~09:40) works better lately.
Ran validated MEIC (hold-to-expiry, uniform qty=1) on `data/cbbo_cache` (09:31+,
which MEIC's own 11:00+ cache lacks); all variants identical except timing, so the
comparison is clean (absolute $ are NOT the deployed weight-map config). "Lately" =
since 2025-09-01. Script: [scripts/sweep_meic_start.py](../scripts/sweep_meic_start.py).

**Answer: partly true, but it's a return-vs-risk tradeoff, not a free win.**
- Per-trade, lately the **earliest slots are the most profitable**: 09:40 avg +$50,
  10:00 +$49 vs +$30–37 for later slots (all ~88–92% WR). So early entries *do* carry
  the best per-trade edge recently — the rumor has support.
- But **starting the whole ladder earlier raises drawdown sharply and lowers Sharpe**:
  late-period 09:40-start = +$160k, Sh 2.75, **DD $85k**; 11:30-start = +$124k,
  **Sh 3.41, DD $20k**; 12:00-start Sh 3.48, DD $16k. Earlier start = more raw PnL,
  much more variance.
- Net: early entries are a **return enhancer + risk adder**. Maximize raw P&L → start
  earlier; maximize Sharpe/limit drawdown → later start is better lately. Extending
  (not shifting) the ladder to add 09:40–10:40 is PnL-accretive but DD-accretive too.
- Caveat: every entry time benefited lately (regime favored premium sellers); this is
  hold-to-expiry on cash settlement (no fill fiction), so honest — but it's a recent,
  low-vol regime read, not a stationary law.

## Addendum 4 — Directional credit spreads (fade & with-trend)

Single defined-risk credit spread on the breakout, VP-anchored, real CBBO
([scripts/backtest_vp_dir_spread.py](../scripts/backtest_vp_dir_spread.py)).
Both weak in-sample, refuted OOS:
- **fade** (sell breakout side): best cell Sh +0.32 (PF 1.02); WF-CV OOS mean −0.43, 1/5.
- **with-trend** (sell far side, à la orb_em_directional): best +0.52 (PF 1.08); WF-CV
  OOS mean +0.46 but median −0.68, 2/5 — dominated by one n=40 fold (Sh +6.93, pure
  small-sample noise; SE≈√(252/40)≈2.5). No edge.

## Addendum 5 — Ablation: does the volume profile add anything? (no-VP control)

Identical structure (09:45 single IC, hold-to-expiry, same 882 days); only the
short-strike rule differs. `--strike-mode em` = no-VP control (shorts at spot ±
k×ATM-straddle). [scripts/backtest_vp_ic.py](../scripts/backtest_vp_ic.py).

| Strike rule | best in-sample Sh | WF-CV OOS mean | OOS median | +folds |
|---|---|---|---|---|
| VP (VAH/VAL) | +1.93 | −0.83 | −1.19 | 1/5 |
| no-VP EM (±k·straddle) | +0.26 | −0.85 | −0.33 | 1/5 |

**Verdict: the VP edge is real in-sample but not durable.** VP clearly beats no-VP
in-sample (+1.93 vs +0.26; fixed baselines +1.63 vs −1.99) — so VAH/VAL encodes
information, not noise. But **OOS they're statistically identical (≈ −0.85, both
refuted)** — the VP advantage is entirely in-sample/overfit and contributes no
tradable edge. Meanwhile the deployed **no-VP MEIC** (full-period Sh ~1.2, ~3 lately,
hold-to-expiry) earns its premium from **laddering many entries + cash settlement**,
not strike cleverness. So the durable edge is structural; the volume profile is a
marginal, non-robust strike-placement tweak that adds nothing out of sample.

## Addendum 6 — VP inside MEIC (multi-entry), vs no-VP MEIC

Applied VP to EVERY tranche of the MEIC ladder (11:00→15:00, 20-min spacing,
hold-to-expiry, uniform qty=1): each IC anchored to the DEVELOPING value area (FRVP
from 1-min bars, 09:30→entry or rolling window). No-VP control = same ladder but
EM-anchored (shorts at spot ± k·straddle). Real CBBO `data/cbbo_cache_meic` (wide
117+ strikes, 11:00+, 1023 days). Script:
[scripts/backtest_vp_meic.py](../scripts/backtest_vp_meic.py).
(NB: `data/cbbo_cache` only has 21 strikes ≈ ±50pts for 2022–2025 — too narrow for
wide ICs; use `cbbo_cache_meic` for any wide-strike SPX 0DTE work.)

| | best in-sample Sh | WF-CV OOS mean | OOS median | +folds |
|---|---|---|---|---|
| VP-MEIC (rolling-90, va0.8, w30) | +1.87 | **+1.94** | +1.46 | **5/5** |
| no-VP EM-MEIC (emX0.8, w30) | +2.58 | **+2.65** | +1.99 | **5/5** |

**Key results:**
- **The multi-entry MEIC SURVIVES WF-CV — both strike rules, 5/5 folds, stable params**
  (same cell every fold, no churn). Unlike every single-entry variant (all refuted),
  the structural edge (laddering + hold-to-expiry) is robust out-of-sample.
- **VP-MEIC genuinely works** (OOS +1.94, 5/5) — the FIRST VP variant in this study to
  survive WF-CV. Putting VP inside the multi-entry structure rescues it.
- **But VP is the worse strike rule**: no-VP EM-MEIC beats it on every fold (+2.65 vs
  +1.94, higher PF, lower DD). EM tracks the *remaining* expected move (straddle decays
  with theta), so each tranche stays calibrated; the developing value area ignores
  time-to-expiry. Rolling-90 VP >> cumulative VP, but still < EM.
- Caveat: single SPX, premium-friendly 2023–2026 regime, uniform qty. Relative VP-vs-EM
  ordering is the robust takeaway; see MEIC provenance caveats in [[project_meic_backtest]].

### Time-aware blend (VP skew + EM scale)
Tried fixing VP's one disadvantage — no time-awareness — by blending strike distance:
`dist = α·(VP distance) + (1−α)·(em_mult·straddle)`, keeping VP's asymmetry but EM's
theta-decaying scale. Result is **monotone in α**: in-sample Sh 0.0→+2.58, 0.25→+2.33,
0.5→+2.14, 0.75/1.0→+2.10. **Every bit of VP weight hurts.** WF-CV picks **α=0 (pure EM)
in all 5 folds**, OOS mean +2.64 (= pure EM). The blend does not rescue VP — optimal VP
weight is zero. (Endpoints validate the impl: α=0 ≡ EM-MEIC, α=1 ≡ VP-MEIC.)

**Overall conclusion across all addenda:** the volume profile carries mild in-sample
information but no out-of-sample edge of its own; the durable 0DTE premium edge is
structural (multi-entry ladder + hold-to-expiry), and EM strike placement beats VP
there — even a time-aware VP+EM blend optimizes to zero VP weight. VP neither creates
nor improves an edge in any structure tested.

[breakorb]: https://breakorb.com/blog/opening-range-breakout-strategy-guide.html
[litefinance]: https://www.litefinance.org/blog/for-beginners/trading-strategies/opening-range-breakout-strategy/
[buildalpha]: https://www.buildalpha.com/opening-range-breakout/
