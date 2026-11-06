# 0DTE Reverse Iron Condor — Backtest Findings (2026-06-10)

Backtest of the course's Reverse Iron Condor strategy spec (not published) over the SPX fly cache
(952 sessions, 2022-05-11 → 2026-02-27, 1-min CBBO). Engine:
`zerodte/backtest/reverse_ic.py`, runner: `scripts/backtest_reverse_ic.py`,
trades/reports under `results/ric_spx/`.

## Headline

**The instructor's numbers reproduce in his own window, but the as-taught
strategy is breakeven over the full period. The scale-out variant with his own
VIX caveat encoded as a floor is a real edge that survives worst-case
execution pricing.**

| Variant (mid pricing unless noted) | n | WR | PF | Total (qty 1 unless noted) |
|---|---|---|---|---|
| As taught: TP 25%, expire | 222 | 76.6% | 1.02 | +$943 |
| TP 30% | 222 | 74.8% | 1.09 | +$5,054 |
| TP 25%, close 15:55 | 222 | 75.2% | 0.98 | −$1,035 |
| Entry 09:45 | 233 | 77.3% | 0.98 | −$1,184 |
| Monday gap 0.10% | 246 | 74.8% | 0.89 | −$6,356 |
| No setup filter | 293 | 77.1% | 1.07 | +$4,634 |
| Scale-out (qty 4) | 222 | 68.9% | 1.35 | +$62,030 |
| **Natural pricing** (pay ask / hit bid, no extra slip): | | | | |
| TP 25% | 203 | 75.4% | 1.05 | +$2,441 |
| TP 30% | 203 | 74.4% | 1.18 | +$9,086 |
| Scale-out (qty 4) | 203 | 70.4% | 1.29 | +$51,913 |
| **Scale-out + VIX≥15, natural (production candidate)** | **144** | **75.0%** | **1.90** | **+$87,445** |

Production candidate by year (qty 4, natural pricing):
2022 +$27.6k · 2023 +$37.5k · 2024 +$14.0k · 2025 +$6.2k · 2026 (2mo) +$2.1k —
positive every year. Max drawdown $22.2k (vs ~$5.5k avg max loss per trade).

## What reproduces from the course

- **Instructor window (2022-05-11 → 2023-02-03): 91.2% win rate, +$7.7k** —
  matches his claimed >90% almost exactly. The strategy was real when recorded.
- **The gap-up filter matters**: 0.25–0.4% gaps lose money; 0.4–0.6% is the
  sweet spot. Gap-down days were never tested (hard skip).
- **His VIX caveat is the whole story**: VIX<15 buckets lose, 18–22 makes
  +$80/trade, >22 makes +$162/trade (single-target, qty 1). The edge decayed
  2023→2025 exactly as VIX ground lower.

## What does NOT reproduce

- **">90% win rate" today**: full-period WR is ~75–77%, and 2025 drops to
  60–65%.
- **"Average winner > average loser"**: false for single-target — avg winner
  $295 vs avg loser $946 (W/L 0.31). 38% of losers are full-debit losses.
  Single-target expectancy is ~zero after costs.
- **"Scaling performs about the same"**: strongly false in our data — the ¼
  runner riding to settlement captures the fat tail (trend days settle near
  width) and is what turns the strategy from breakeven into PF 1.3–1.9. The
  scale-out W/L ratio is 0.61 vs 0.31 single-target.
- **Monday 0.1% gap variant**: worse (PF 0.89). Mondays are the worst weekday
  outright (−$152/trade avg); the per-weekday "improvement" goes the other
  direction — a Monday *skip* would have helped (flagged as in-sample only).

## Modeling notes

- Gap is parity-pure: prior-day 15:59 parity forward vs 09:31 parity spot —
  no SPY×10 basis, no dividend distortion. Zero SPY-fallback days in practice.
- The "sets up" filter is exact, not an IV proxy: upside = call_width − debit
  ≥ 25% of debit ⇔ debit ≤ 16 pts on a 20-wide call spread. It removed 71
  trades; without it P&L is slightly better, but it caps the worst debits.
- `pricing: natural` buys at the ask / sells at the bid on every traded leg —
  the worst-case execution bound. The scale-out edge surviving it (PF 1.29
  unfiltered, 1.90 with the VIX floor) is the strongest robustness signal.
- The TP "limit" fill model fills at the resting level when the *mid* (or the
  natural liquidation value, in natural mode) crosses it.
- Sharpe in the reports annualizes per-trade at 252; with only ~40-60
  trades/yr, treat PF / yearly totals as primary.

## Parameter sweeps by VIX regime (2026-06-11)

All sweeps at qty 4 under natural pricing, computed from the trade logs (VIX
floor) or via `scripts/sweep_ric_params.py` (structure; single data pass over
16 engine variants, table in `results/ric_spx/param_sweep.csv`).

**VIX floor.** Swept 13/14/15/16/17/18/20/22 on the scale-out variant: 15 is
the optimum by total P&L (+$87.4k) and the only floor with zero negative
years; the profile is smooth (14: +$62.9k, 16: +$74.3k PF 2.05), not a
knife-edge — i.e. the peak sits exactly on the instructor's pre-registered
"dies sub-15" boundary. Per-bucket avg/trade: <13 +$90 · 13–15 **−$779** ·
15–18 +$615 · 18–22 +$530 · >22 +$767. The 13–15 bucket is where the runner
mechanic specifically backfires (pin days expire the runner worthless).
VIX-scaled sizing (0× <15, 1× 15–18, 2× ≥18) yields +$127k vs +$87k flat —
directionally supported by the monotone buckets but the 2× leg doubles a
66-trade sample; treat as a sizing suggestion only.

**Structure (long deltas, wing offsets, tiers), one-at-a-time around spec.**
The course spec (50Δ/35Δ, 20/35pt wings) is at or near the best in every
tradeable band; no robust case for regime-conditional structure:

- lc_delta 0.55 looks great in VIX 15–18 (+$820/trade, PF 2.67) but its richer
  debit fails the setup filter in high vol (n=3 above 22, losing) — trap.
- lp_delta 0.40 / put_wing 45 similarly shine in 15–18 and stop trading in
  high vol; put_wing 25 only works in high vol. Mechanism is mostly the setup
  filter *selecting different days*, not better structure.
- call_wing 15 is outright negative; 25/30 dilute every band.
- Tiers: a **bigger runner** (close only ½ @ +20%, let ½ ride) beats the base
  on VIX≥15 days: +$99k vs +$87k, best 15–18 band (+$726/trade) — at the cost
  of a 62% vs 75% win rate and lumpier equity. The defensible aggressive
  variant. Tighter tiers (15/25) hurt; 25/40 wins only in VIX≥22 (n=19).
- All 16 variants lose below VIX 15 (PF 0.48–0.78) — the floor is
  parameter-robust.

Every cell is in-sample and one-at-a-time; high-VIX cells are thin (n=19–76).

## Walk-forward CV (2026-06-11) — `scripts/run_ric_wf_cv.py`

Ratio-based 5:1, 4 folds (step ≈ 105 sessions): grid = 3 tier schemes × 6 VIX
floors, selected per fold on train by annualised daily Sharpe, scored on the
held-out next window. OOS span covers 2024-06 → 2026-02.

- **The VIX≥15 floor validates: selected in 4/4 folds** (never 14/16/17/18 or
  none). It is the robust component of the strategy.
- **The tier scheme does not validate**: base picked 2/4, the other schemes
  1/4 each — noise. The ½-runner "upgrade" was picked once (fold 4) and had
  the worst OOS window. **Do not promote it; keep the base tiers.**
- **OOS expectancy is much weaker than the full-window numbers**: selected
  combos made **+$3.7k over 66 OOS trades** (qty 4); the fixed production
  candidate made +$5.0k over the same windows. Two of four OOS folds were
  negative (Jun–Nov 2024: −$5.1k; Sep 2025–Feb 2026: −$11.3k).
- Interpretation: the headline +$87k is concentrated in 2022–2023, which
  ratio-based WF-CV only ever uses for training. In the most recent ~20
  months the edge is breakeven-to-slightly-positive (~$75/trade at qty 4
  ≈ $19/contract). The fold-3 window (Apr–Sep 2025, +$13.1k, PF 1.75) shows
  it still fires when vol returns — consistent with the regime story — but
  recent OOS evidence does NOT support sizing up.

## Caveats

- The VIX≥15 floor was spot-checked at {15, 18} on 2026-06-10 and fully swept
  on 2026-06-11 (see above) — the optimum lands on the instructor's own
  pre-registered boundary with smooth shoulders, which mitigates (but does not
  eliminate) the in-sample concern.
- The runner's profits concentrate in big trend days; the 2025 max drawdown
  ($22k at qty 4) requires accepting ~4 consecutive full-debit losses.
- FOMC blackout uses the static `_FOMC_DATES` table; CPI/PPI days are NOT
  excluded (the course only hard-excludes FOMC).

## Live bot

`zerodte/strategies/reverse_ic.py` (ScheduledEngine, 09:33 window) +
`config/config_ric_live.yaml` — production-candidate parameters: scale-out
½@+20% / ¼@+30% / ¼ runner, VIX floor 15, gap ≥0.25%, no stop,
eod_close_time 15:59 (runner rides to cash settlement). Graduate
dry_run → manual → live; TT cert cannot trade SPXW 0DTE.
