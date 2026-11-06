# ORB EM Directional — Full-Span WF METHOD Re-run (2026-06-16)

**Strategy:** `orb_em_directional` (single with-trend credit spread — PCS on long
breakout / CCS on short — after ORB retest). Re-runs the old es_8am
**"directional worst-case (0.20Δ, 15pt)"** optimizer winner on REAL CBBO.
**Original (BS-priced):** WR 92.9%, Sharpe 12.90, PF 5.22, +$12,460/contract
(the earlier worst-case optimization study, 2026-03-27) — `pricing_model: bs`.
**Data:** SPY 1-min signal + SPX CBBO 1-min, 2022-05 → 2026-06-11 (501 trade-days).
**Method:** three-tier WF METHOD validation (`scripts/run_wf_method_strategy.py`).

## TL;DR

- **The named strategy (0.20Δ-equiv / 15pt / midline) is REFUTED on real CBBO.**
  Held static, it loses **−$7.8k to −$9.6k OOS (Sharpe −1.8 to −2.4)**. The BS
  +$12.5k/Sharpe 12.9 was pricing fiction.
- **Periodic re-optimization rescues it only to ~breakeven** (OOS Sharpe ~0.05–0.10,
  plateau +$200–$515 for train ≥ 100). Large relative gain over the static baseline,
  but **not a deployable edge.**
- Per-fold winner churns (curve-fit on exact params); only the *family*
  (`breakout_level` retest + width 20 + em 0.4–0.5) is stable. Its in-sample
  Sharpe 2.0–2.4 collapses to OOS ~0.1 — overfit.

## Credit realism (important nuance vs ORB IC)

Unlike the thin-width IC (BS $5 vs CBBO $0.05), the **15–20pt directional spread
collects REAL CBBO credit of $2.5–$4.5** — the worst-case memo's "$3–4 credit"
rationale holds on real quotes. So this is NOT a pure BS-fiction artifact; the
credits are real. The edge fails for a different reason: breach losses (~−$300 on
breach-exit, up to −$1.6k worst-day) overwhelm the ~+$100 wins over the full cycle.
em_mult ≈ 0.5 lands ~20pt OTM ≈ 0.20–0.25Δ (matches the named target).

## WF METHOD ladder — stitched UNTOUCHED test tier

| train/test | folds | method (deploy winner) | WFE-gated | as-taught (static) |
|---|---|---|---|---|
| 80 / 40  | 9 | −$2,540  (Sh −0.51) | −$8,314 | −$9,033 (Sh −1.75) |
| 100 / 50 | 7 | **+$495   (Sh 0.10)** | −$7,328 | −$9,036 (Sh −1.79) |
| 130 / 65 | 4 | **+$192   (Sh 0.05)** | −$4,999 | −$9,628 (Sh −2.40) |
| 160 / 80 | 3 | **+$515   (Sh 0.10)** | −$7,815 | −$7,815 (Sh −2.06) |

- **as-taught (named worst-case params, static): firmly negative** — REFUTED.
- **method (re-opt): ~breakeven plateau** (train ≥ 100). train=80 too short → noisy.
- **WFE-gated worst**: the WFE gate keeps falling back to the losing static baseline,
  so it underperforms even the ungated method — i.e., the noisy train-winners still
  beat the static params OOS.
- **Winner stability:** every fold a different combo (1/9, 1/7, 1/4, 1/3) =
  curve-fitting on the exact cell. Family (breakout_level / w20 / em 0.4–0.5) stable.

## Full-span in-sample (context)

96/192 combos positive, 65 with Sharpe > 1 (median +$164). Top combos:
`breakout_level` + width 20 + em 0.4–0.5 + hold_put_breach → +$10k–$14k, Sharpe
2.0–2.4 (best `ew0.4_w20_tp0.5_me20_hpb1_breakout_level` +$13,832). **None survive
OOS** at that level — the method picking from this family nets only Sharpe ~0.1.
`hold_put_breach=true` averages +$681 vs −$941 for exit-on-breach. The named
baseline `ew0.5_w15_tp0.75_me0_hpb1_midline` = −$5,821 in-sample (Sharpe −0.96).

## Stop-fill

No trigger stop (`sl_mult=0`); exits on breach/TP/EOD at the CBBO quote. Breach-exit
fill could be optimistic, which would only push the already-breakeven method negative.
No robust edge to defend.

## Verdict

**Do not deploy.** The named worst-case directional is a BS-pricing artifact (refuted
on CBBO). The directional family is *less dead* than ORB IC — re-optimization lifts it
from −$9k to ~breakeven — but OOS Sharpe ~0.1 with curve-fit exact params is not a
tradeable edge by the WF doctrine (claim the plateau: ~$0).

## Reproduce

```bash
.venv/bin/python -m scripts.gen_orb_dir_combo_logs \
    --start 2022-04-18 --end 2026-06-11 --out results/orb_em_dir_wf --workers 1
for c in "100 50" "130 65" "160 80"; do set -- $c; \
  .venv/bin/python scripts/run_wf_method_strategy.py --dir results/orb_em_dir_wf \
    --baseline ew0.5_w15_tp0.75_me0_hpb1_midline --train-days $1 --test-days $2; done
```

Supersedes the BS-priced worst-case-optimization directional result.
Companion to `docs/wf_cv_findings_orb_em_ic_spx_2026-06-16.md` (ORB IC, also refuted).

---

## ADDENDUM (2026-06-16) — breakout-retest at mtol=3.0/min_em=20 SURVIVES

The primary run above fixed `midline_tolerance=1.0`. That undersold the
**breakout_level retest** variant. The config-file deployed winner
(`config/config_orb_dir_breakout_spx.yaml`) uses `mtol=3.0` + `min_em_pts=20`.
Re-running the breakout pool at mtol=3.0 (`results/orb_em_dir_bo_mt3/`, 96 combos):

- **Whole pool 96/96 positive.** Config winner `ew0.5_w20_tp0.75_me20_hpb1_breakout_level`:
  full-span **+$13,751, Sharpe 3.29**, worst day −$1,632.
- **By year (persistent, not 2022-only):** 2022 +$6,161 (Sh 7.5) / 2023 +$1,355 (1.5) /
  2024 +$3,109 (5.1) / 2025 +$3,525 (6.1) / 2026 −$399 (partial, −0.4).
- **WF untouched-test plateau — all positive:**

  | train/test | as-taught (static) | method (re-opt) |
  |---|---|---|
  | 80/40  | +$6,235 (Sh 2.01) | +$2,617 (0.67) |
  | 100/50 | +$5,417 (Sh 1.92) | +$6,704 (3.01) |
  | 130/65 | +$2,924 (Sh 1.47) | +$1,953 (1.28) |
  | 160/80 | +$1,447 (Sh 0.98) | +$1,235 (1.31) |

  Static params hold up as well as / better than re-opt (the edge is a stable family,
  not a re-opt artifact). maxDD −$963 to −$2.9k.

- **Fill stress (load-bearing) — PASSES.** The backtest priced exits at MID with no
  slippage (optimistic); entry at mid − flat slippage. Added `options.exit_fill_frac`
  to the orchestrator (0=mid [default, no regression]; 1=full bid/ask crossing — buy
  short back at ask, sell long at bid; hits breach exits hardest where quotes widen).
  Full-span stress matrix on the survivor config:

  | entry slip | exit crossing | total | Sharpe | $/trade |
  |---|---|---|---|---|
  | $0.10 | mid (baseline) | +$13,751 | 3.29 | $59 |
  | $0.10 | half-spread | +$12,339 | 2.94 | $53 |
  | $0.10 | **full bid/ask** | +$10,970 | **2.61** | $47 |
  | $0.15 | half (realistic) | +$11,795 | **2.80** | $51 |
  | $0.25 | full (harsh) | +$9,552 | **2.24** | $41 |

  Degrades gracefully, not catastrophically. Full bid/ask crossing on every exit
  (incl. breaches) only drops Sharpe 3.29→2.61 — the "trigger vs quote" artifact does
  NOT carry the edge. Modeled realistic round-trip cost ≈ $0.12–0.18/spread; the edge
  breaks only near ~$0.50/spread flat (≈$50/trade, ~3× realistic). Comfortable margin.
  Residual unmodeled: breach-exit execution latency (full-ask crossing partly covers
  it); confirm live SPX 0DTE fills before scaling (fill-realism caveat).

**Revised verdict:** the NAMED worst-case (midline) directional is refuted, BUT the
**breakout-retest directional at config settings (mtol=3.0, min_em=20) is a genuine,
persistent OOS edge** (Sharpe ~1–2 untouched-test, +4/5 yrs) — the only survivor of the
8AM ORB family. Caveats: needs ≤$0.25/spread fills; 2026 partial soft (watch decay);
breach-exit fill realism still wants a live study (fill-realism caveat).
Reproduce: `gen_orb_dir_combo_logs.py --retest breakout_level --mtol 3.0 --out results/orb_em_dir_bo_mt3`.
