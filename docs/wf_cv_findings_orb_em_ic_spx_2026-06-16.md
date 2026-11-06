# ORB EM IC — Full-Span WF METHOD Re-run (2026-06-16): REFUTED

**Strategy:** `orb_em_ic` (SPX 0DTE bias-adaptive iron condor; ES/SPY 08:00–08:15
ORB signal → SPXW 0DTE iron condor at scheduled windows)
**Data:** SPY 1-min signal bars + SPX CBBO 1-min quotes, **2022-06-08 → 2026-06-11
(~4 years, 350 trade-days)** — vs the original 2026-04-13 run's 125 days.
**Framework:** `zerodte/backtest/` orchestrator (real OrbEngine + PositionMonitor).
**Method:** three-tier walk-forward METHOD validation per `docs/wf-cv.md`
(`scripts/run_wf_method_strategy.py`), not the val-leaked `WalkForwardOptimizer`.

## TL;DR

**The strategy has no edge over the full cycle. REFUTED.** The 2026-04-13 result
(+$598 OOS, Sharpe 39 on 125 days) was a favorable-regime artifact. With ~4 years
of CBBO including the freshly-backfilled latest period, OOS PnL is negative under
*every* train/test configuration, and the per-fold winner is different every fold
(the doctrine's curve-fitting signature).

## What was done to "include the latest period"

The per-minute CBBO cache (`data/cbbo_cache/`) had grown to 2022-04 → 2026-06-11,
but the 56 days from **2026-03-23 → 2026-06-11 were truncated to 11:00–15:49**
(they had only been backfilled for the Power Hour strategy, which trades the
afternoon). ORB IC enters at 09:45/10:15/11:15…, so it needs the morning.

- `scripts/backfill_orb_ic_morning.py` fetched the missing 09:30–11:00 CBBO for
  those 56 days from Databento (OPRA.PILLAR cbbo-1m) and **merged** it into the
  existing pickles (union of keys → afternoon coverage for Power Hour preserved).
- Cost: **$0.16 total** (ledger: `results/databento_spend.jsonl`). ThetaData was
  considered as a cheaper fallback but Databento was already trivially cheap and
  keeps the cache homogeneous (all CBBO).
- Minor caveat: those 56 days end at 15:49 (Power Hour footprint), so ORB IC's
  15:50 EOD close falls back to the 15:49 quote. Negligible, ~56/350 days.

## Pipeline

1. `scripts/gen_orb_ic_combo_logs.py` — simulate-once-full-span (doctrine's
   day-independence shortcut), one `combo_*.csv` per param set under
   `results/orb_em_ic_wf/`. Grid = 144 combos (em_with × em_against × width_with ×
   width_against × tp × midline_tol) + the as-taught baseline = 145.
   - Run **serial** (workers=1): the 893-day preload is ~10.8 GB RAM; fork workers
     blow up via copy-on-write refcount churn (OOM'd at 3 workers). Serial: 6.5 min.
2. `scripts/run_wf_method_strategy.py --dir results/orb_em_ic_wf` over a train/test
   ladder, baseline = `ew0.5_ea0.75_ww5_wa10_tp0.5_mt0.25` (pre-opt config params).

## WF METHOD ladder — stitched UNTOUCHED test tier

| train / test | folds | method (deploy winner) | WFE-gated / as-taught | Sharpe |
|---|---|---|---|---|
| 80 / 40  | 5 | −$3,334 | −$4,611 | −1.2 to −1.8 |
| 100 / 50 | 4 | −$3,713 | −$4,611 | −1.3 to −1.8 |
| 130 / 65 | 2 | −$4,427 | −$2,371 | −1.5 to −1.9 |
| 160 / 80 | 1 | −$2,042 | −$2,869 | −1.7 to −2.7 |

- **Plateau is uniformly negative.** No train/test choice produces positive OOS.
- **WFE is negative in every fold** (train Sharpe ~4.8 but val Sharpe < 0): the
  train-winner always fails validation → the WFE gate correctly falls back to the
  as-taught baseline. But the baseline *also* loses money.
- **Winner instability = curve fitting:** every fold's train-winner is distinct
  (1/5, 1/4, 1/2, 1/1 — never 2+). Per `docs/wf-cv.md`, heavy fluctuation of the
  selected combo means the method is fitting noise.

## Full-span in-sample (sanity / not a pipeline bug)

Ranking all 145 combos over the full 4 years:

- **Only 7/145 combos are positive in-sample**; the best is **+$263 total
  (Sharpe 0.42)** over ~4 years (~$66/yr). Median combo −$2,809.
- The 7 survivors are all the *widest* against-wings (`em_against=2.5`, wing 15,
  tp=0.75); the tight-wing variants (`em_against=1.5`) bleed −$10k to −$12k.
- **Prior 125-day WF winner** `ew0.5_ea2_ww15_wa10_tp0.75_mt1` over full span:
  **−$2,393, Sharpe −0.80.**

The pipeline is sound: the timing probe over the favorable Jan–Mar 2026 window
gave +$937 for one combo, and a handful of combos scrape positive — there is just
no robust edge.

## Why it fails out-of-sample

Selling 0DTE premium with `hold_put_breach=true` (no stop; breached side rides to
EOD) wins in calm grind regimes and gives back everything in trending / high-vol
regimes (worst days −$1.2k to −$1.7k). The late-2025→early-2026 sample that the
original 125-day run covered was exactly such a calm grind. The full cycle
(2022 bear/high-vol, 2023–2024) erases the edge.

## Stop-fill model

N/A — there is no positive edge to defend, and ORB IC exits on breach/EOD/TP with
`sl_mult_of_credit=0` (no trigger-based stop), so the trigger-vs-quote fill
sensitivity that kills other strategies doesn't even apply here.

## Reproduce

```bash
# 1. backfill the latest period's morning CBBO (one-time, ~$0.16)
.venv/bin/python scripts/backfill_orb_ic_morning.py            # --dry-run to cost first

# 2. simulate all combos full-span (serial — fork OOMs on the 10.8GB preload)
.venv/bin/python -m scripts.gen_orb_ic_combo_logs \
    --start 2022-04-18 --end 2026-06-11 --out results/orb_em_ic_wf --workers 1

# 3. WF METHOD ladder
for c in "80 40" "100 50" "130 65"; do set -- $c; \
  .venv/bin/python scripts/run_wf_method_strategy.py --dir results/orb_em_ic_wf \
    --baseline ew0.5_ea0.75_ww5_wa10_tp0.5_mt0.25 --train-days $1 --test-days $2; done
```

## Put-dominance claim ("puts 84W/0L flawless") — REFUTED (per-side decomposition)

The es_8am CBBO IC reported the put leg as 84W/0L (+$11,696), Sharpe 14.6
(the earlier CBBO backtest). That was the es_8am **EOD-mislabeling bug**: with
`hold_put_breach=True` a breached put rides to EOD, and es_8am booked the ITM ones
as $0 ("expired_worthless"). The zerodte orchestrator emits one result per side
(`spread_type` = put_/call_credit_spread) with real intrinsic booking. Canonical IC
(em0.5/ea0.75/w5/w10/tp0.5, hold_put_breach=True) over the full 4yr:

| side | n | WR | total | breach-exits | worst trade |
|---|---|---|---|---|---|
| PUT spread | 338 | 82.0% | **−$2,810** | 0 (held to EOD) | −$855 |
| CALL spread | 338 | 67.8% | −$213 | 108 | −$355 |
| **combined** | | | **−$3,023** | | (= as-taught baseline) |

Put side by year: 2022 +$2,298 (28t stub) / 2023 −$1,949 / 2024 −$1,575 / 2025 −$756
/ 2026 −$828. **Puts are the bigger loser, not the edge** — the claim is inverted; a
put-only variant would be worse. The 82% WR is the short-premium trap (win small often,
lose big on held ITM). Sharpe 14.6 was pure bug.

## Supersedes

`docs/wf_cv_findings_orb_em_ic_spx_2026-04-13.md` — that run's positive OOS was a
125-day favorable-regime artifact. Conclusion now: **do not deploy orb_em_ic.**
