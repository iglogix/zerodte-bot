# SPX 0DTE Open Iron Condor — Findings (v2, analytical full-data)

*Last updated: 2026-05-16*

A 09:35 ET iron condor on SPX 0DTE expiries, evaluated analytically on a
**dense, multi-anchor CBBO cache** spanning 2022-04-18 → 2026-02-28
(952 trading days, ~3.78 years).

## Headline result

**`em_mult_put=0.30, em_mult_call=0.20, wing=30, TP=50% credit, SL=2× credit, hold-to-expiry otherwise`**

- 945 trade days (every CBBO-valid day, no calendar skips in the analytical eval)
- WR 61.6% per-IC-day
- **Total +$229,075** over 3.78 yr
- **Sharpe 3.35** (annualized)
- **Max drawdown $15,058** (6.6% of cumulative profits)
- Annualized **~$60,600/yr** on $50k notional = **~121% ROI**

Sharpe-best variant: `em_put=0.25, em_call=0.20, wing=20` → Sharpe 3.74, max DD $7,885, +$184k.

## Methodology

The earlier (v1) per-minute simulation had two problems:

1. **CBBO cache was sparse** — collected with ±50 pts of strike coverage to minimize Databento cost. At higher em_mults the required strikes weren't in the cache, biasing trade counts. We showed em_mult=1.0 had only 183/953 trade days while em_mult=0.10 had 548 — a data artifact, not market reality.

2. **Per-minute SL fired on intraday noise** at deep-OTM strikes where CBBO mid is noisy.

The v2 approach uses an **analytical evaluator** with a **multi-anchor entry-only CBBO backfill**:

- 5 anchor minutes per day: 09:35, 11:00, 13:00, 15:30, 15:59
- Full strike chain (±500 pts) at each anchor
- Cost: **$2.02 total** (Databento, single backfill)
- Output: `data/cbbo_cache_5anchor/` (952 day-pickles)

The evaluator:
- Picks strikes at `spot ± em_mult × EM` where EM = ATM-straddle-mid at 09:35 (model-free)
- Computes entry credit from 4 real CBBO mids
- Checks TP/SL at each subsequent anchor (4 intraday checkpoints)
- Falls back to expiry P&L (intrinsic from spot_close) if no TP/SL triggered
- No model assumptions (no BS), no per-minute simulation noise

## Full sweep results (TP=50%/SL=2×, symmetric IC)

| em_mult | wing | n | WR | Total | Sharpe | Max DD | avg_W | avg_L |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 30 | 946 | 58.2% | +$213,767 | 3.16 | -$11,445 | +$1,112 | -$1,011 |
| **0.20** | **30** | 947 | 60.4% | **+$219,012** | 3.21 | -$11,245 | +$1,088 | -$1,076 |
| 0.20 | 25 | 947 | 57.6% | +$198,257 | 3.38 | -$8,768 | +$990 | -$849 |
| 0.20 | 20 | 948 | 54.6% | +$178,156 | 3.73 | -$7,700 | +$866 | -$629 |
| 0.25 | 30 | 947 | 61.2% | +$211,711 | 3.08 | -$14,820 | +$1,076 | -$1,124 |
| 0.25 | 25 | 947 | 58.7% | +$195,927 | 3.27 | -$9,897 | +$981 | -$894 |
| **0.25** | **20** | 948 | 57.3% | +$183,574 | **3.74** | **-$7,885** | +$846 | -$681 |
| 0.30 | 30 | 946 | 62.2% | +$193,472 | 2.81 | -$15,910 | +$1,046 | -$1,177 |
| 0.50 | 30 | 947 | 65.8% | +$113,596 | 1.71 | -$20,548 | +$882 | -$1,345 |
| 0.70 | 30 | 945 | 68.0% | +$55,668 | 0.93 | -$34,865 | +$715 | -$1,338 |
| 1.00 | 30 | 947 | 70.5% | +$8,291 | 0.18 | -$28,882 | +$470 | -$1,097 |

Sweet spot: **em_mult 0.20–0.25 / wing 25–30**. Past em=0.50, PnL collapses because credits get too small relative to occasional max-loss events.

## Why avg loss is only ~$680 despite a "2× credit" SL

A natural objection: with SL = 2× credit, why is avg loss only $681 (em=0.25/wing=20)? Three reasons — reproduce with `--exit-breakdown`:

```bash
.venv/bin/python -m scripts.analytical_ic_eval \
  --em-mults 0.25 --wing-widths 20 --tp-pct 0.50 --sl-mult 2.0 --exit-breakdown
```

| exit | n | total PnL | #loss | avg loss | worst |
|---|---|---|---|---|---|
| TP | 316 | +$264,690 | 0 | — | — |
| SL | 26 | -$26,900 | 26 | -$1,035 | -$1,255 |
| Expiry | 606 | -$54,216 | 379 | -$656 | -$1,035 |

Blended avg loss = $680.

1. **"2× credit stop" = 1× credit LOSS, not 2×.** When the IC value reaches 2× credit, you've lost `2C − C = 1C`. At avg credit $12.16 (×100 = $1,216), a full SL hit ≈ $1,035 (slightly under, because SL-day credits ran below average).

2. **The SL barely fires — 26 of 405 losses.** The analytical evaluator checks the stop at only 4 anchors (11:00/13:00/15:30/15:59). Most losing days grind to a shallow ITM close without ever spiking to 2× credit at an anchor → they fall through to expiry.

3. **Expiry losses dominate and are shallow (avg -$656).** 379 of 405 losses are expiry exits where the close finished just past a short strike — debit is usually well below 1× credit. This blends *down* the average loss.

The real edge: em=0.25 puts shorts ~8 pts from spot (near-ATM, EM≈32) → rich credit ($12.16) on a 20-wide IC → max loss only (20−12.16)×100 ≈ $784. The SL is nearly cosmetic; the edge is rich near-ATM premium + tight wings.

**Caveat:** the coarse 4-anchor SL *under-stops* vs a real continuous 2× stop. A live bot checking every minute would fire the SL more often, cutting losers at ~1× credit ($1,035) instead of the -$656 avg expiry loss — i.e., a tighter live SL would likely REDUCE PnL. This matches the per-minute-sim finding that tight intraday SLs on noisy marks hurt.

## Asymmetric IC sweep

| put_em | call_em | wing | Total | Sharpe | Max DD |
|---|---|---|---|---|---|
| **0.30** | **0.20** | 30 | **+$229,075** | 3.35 | -$15,058 |
| 0.25 | 0.20 | 30 | +$228,715 | 3.33 | -$14,353 |
| 0.20 | 0.20 | 30 | +$219,012 | 3.21 | -$11,245 |
| 0.20 | 0.25 | 30 | +$209,093 | 3.05 | -$11,818 |
| 0.20 | 0.30 | 30 | +$191,904 | 2.79 | -$11,680 |
| 0.20 | 0.50 | 30 | +$104,574 | 1.52 | -$14,167 |

**Put-skew helps** (put further OTM than call) — counter-intuitive but consistent across the sample. The 65% call-side breach observation was misleading because moving calls closer captures more premium that overcomes the higher breach rate.

## TP vs no-TP comparison

| Variant | Best PnL | Sharpe | Max DD |
|---|---|---|---|
| Vanilla TT (no TP, no SL, hold to expiry) | +$36,824 | 1.78 | -$15,487 |
| With TP=50%/SL=2× (current) | **+$229,075** | **3.35** | **-$15,058** |

Vanilla TT numbers shown are on a 290-day sub-sample (most days where the analytical evaluator could compute non-management exits). Full-data comparison would scale similarly — management is the right call.

## Filter testing — most filters HURT

Tested gates (each tested individually + combined):

| Filter | Δ PnL | Δ Sharpe | Δ DD | Verdict |
|---|---|---|---|---|
| skip_cpi_next_day | -$760 | +0.05 | -$1,200 | **Marginal positive**, enabled in config for psych |
| skip_fomc_next_day | -$5,333 | -0.17 | flat | Hurts |
| skip both cpi/fomc next | -$6,095 | -0.12 | -$1,200 | Hurts |
| trend skip σ>1.5 (skip post-big-day) | -$6,830 | +0.10 | +$695 | Mixed |
| trend skip σ>2.0 | -$10,025 | -0.30 | -$2,123 | Hurts |
| low_em <80% of 5-day avg | -$7,835 | -0.21 | flat | Hurts |
| vix9d > 25 | -$20k | -0.40 | +$2,369 | Hurts (much fewer trades) |
| vix9d > 30 | -$17k | -0.58 | flat | Hurts |
| **ALL combined** | **-$37k** | -1.38 | -$1,546 | Catastrophically destroys edge |

**Streak analysis was anecdotal** — identifying loss-clustering days didn't translate to net-positive filters because those same calendar/regime days also include profitable days.

## Loss-streak diagnostic findings

From the analytical eval at em=0.20/wing=30 (290-day sample):

- **65% of losses are call-side breaches** (sample dominated by 2022-2023 recovery rally)
- Mean move on loss days: +0.37σ (bullish skew)
- Abs mean move: 1.59σ — large directional moves drive losses
- Max move: -4.43σ on SVB day (Mar 9 2023), +3.71σ on Oct 13 2022 CPI

Longest loss streaks were all multi-day directional trends:
- Sep 2022: 5 days (bear-market rally)
- Jul 2022 FOMC week: 3 days (post-Fed rally)
- Oct 2022 CPI week: 3 days (whipsaw)
- Mar 2023 SVB: SL fired

These aren't predictable from entry-time data — they're the structural cost of being a premium-seller in trending regimes.

## Critical bugs discovered along the way

1. **PositionMonitor double-close bug** ([src_v2/services/position_monitor.py](src_v2/services/position_monitor.py)) — when one side stop_lossed, the whole trade was marked closed, stranding the other side unmonitored. This was both a backtest distortion (fake 100% per-day WR) AND a live-bot bug. Fixed: trade.status=closed only fires when all legs are actually closed (`_any_legs_remain` helper).

2. **CBBO chain range filter** ([src_v2/backtest/bt_tool_handlers.py](src_v2/backtest/bt_tool_handlers.py)) — chain was capped at ±200 pts from spot for performance, excluding far-OTM strikes needed for em_mult ≥ 0.6 in volatile regimes. Widened to ±500 pts.

3. **Strategy breach check spammed log** ([src_v2/strategies/open_ic.py](src_v2/strategies/open_ic.py)) — after a side closed, breach check still fired with empty leg list on every bar. Fixed: require `_call_leg_symbols(trade)` non-empty before firing.

## Tooling

- [scripts/build_entry_manifest.py](../scripts/build_entry_manifest.py) — builds multi-anchor manifest from SPY bars
- [scripts/databento_backfill.py](../scripts/databento_backfill.py) — Databento backfill with cost cap (existing)
- [scripts/analytical_ic_eval.py](../scripts/analytical_ic_eval.py) — model-free IC evaluator on multi-anchor cache
- [scripts/test_asymmetric_ic.py](../scripts/test_asymmetric_ic.py) — asymmetric IC sweep
- [scripts/test_filters.py](../scripts/test_filters.py) — filter impact tester
- [scripts/analyze_ic_streaks.py](../scripts/analyze_ic_streaks.py) — loss-streak diagnostic
- [scripts/test_consec_put_losses_analytic.py](../scripts/test_consec_put_losses_analytic.py) — TT "no 2 consecutive losing puts" claim test

## Reproduce

```bash
# Build manifest
.venv/bin/python -m scripts.build_entry_manifest

# Estimate cost (free)
.venv/bin/python -m scripts.databento_backfill \
  --manifest data/signal_manifest_multi_anchor.jsonl \
  --half-width-pts 500 --before-min 0 --after-min 1 --dry-run

# Live backfill (~$2)
.venv/bin/python -m scripts.databento_backfill \
  --manifest data/signal_manifest_multi_anchor.jsonl \
  --half-width-pts 500 --before-min 0 --after-min 1 \
  --out data/cbbo_cache_5anchor --max-cost-usd 3

# Final evaluator (sweeps em_mult × wing × TP × SL)
.venv/bin/python -m scripts.analytical_ic_eval \
  --cache data/cbbo_cache_5anchor \
  --em-mults "0.15,0.20,0.25,0.30,0.40,0.50,0.70,1.00" \
  --wing-widths "20,25,30"

# Asymmetric variants
.venv/bin/python -m scripts.test_asymmetric_ic
```

## TT "never 2 consecutive losing puts at 25 delta" claim

Tested via [test_consec_put_losses_analytic.py](../scripts/test_consec_put_losses_analytic.py):

- At 25Δ equivalent (em=0.30): **7 consecutive losses** max (Sep 2022 bear rally)
- At em=2.0 (~5Δ, very far OTM): 2 consecutive losses — closest to TT's claim, but strategy is unprofitable here
- The claim doesn't hold for 0DTE SPX 2022-2026 at any actually-profitable delta

Likely the claim is from TT's older 45-DTE studies, where overnight theta decay smooths daily PnL. For 0DTE there's no carry — every day is a fresh coin flip.

## Recommended production config

[config/config_spx_0dte_open_ic.yaml](../config/config_spx_0dte_open_ic.yaml) currently set to (symmetric IC, em=0.20):

```yaml
options:
  strike_mode: em_mult
  em_mult: 0.20
  wing_width: 20
  tp_pct_of_credit: 0.50
  sl_mult_of_credit: 2.0
  sl_scope: per_side          # legacy — this is the WF-CV v7 winner from prior work
  ...

strategy:
  skip_nfp_fridays:    true
  skip_fomc_days:      true
  skip_cpi_days:       true
  skip_fomc_next_day:  true
  skip_cpi_next_day:   true    # enabled 2026-05-14 for psych/DD
  skip_weekdays:       ["Fri"]
```

**To get the analytical-best result in live trading**, the per-side SL would need to be switched to per-side _per the analytical hold-to-expiry model with hourly TP/SL checkpoints_. The current production config (`sl_scope=per_side`) was tuned on the per-minute simulation with the asymmetric leg-out edge. The analytical evaluator's exit logic (total-credit TP at any anchor, total-credit SL at any anchor) is different.

**Recommended upgrade path**: introduce an "anchor-based exit" mode in OpenIcStrategy that mirrors the analytical evaluator's logic, then live-trade that. This is a future task.

## Costs

- Databento data backfill (multi-anchor entry-only): **$2.02 one-time** for full 952-day coverage
- Per-trade commissions/slippage: as configured in YAML, currently $1/leg open + $0.10/spread

## What's NOT in this analysis

- **Survivorship bias**: SPX as a benchmark survives by definition. A regime where SPX is in persistent decline for years would change everything.
- **Liquidity / market impact at scale**: Single-contract sizing assumed. Scaling to 50+ contracts may move CBBO away from the displayed bid/ask.
- **Execution risk on entry**: 09:35 entry assumes fills at the displayed mid. Real-world: typically 1-2 ticks worse than mid.
- **Real-time data costs**: The backfill is historical. Live trading needs paid data subscription (Databento or tastytrade live feeds).
