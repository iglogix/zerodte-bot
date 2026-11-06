# 0DTE Iron Condor — Chris Pulver method (SPX), backtest + WF-CV

**Source:** Chris Pulver interview, "How He Wins 85% of His 0DTE Trades" (YouTube
`5MbQ_dnImR0`). Extracted, backtested on real CBBO quotes, optimized for
**maximum Sharpe**, and validated via walk-forward cross-validation (WF-CV).

**Date:** 2026-06-14. **Data:** `data/cbbo_cache` (real SPXW NBBO/CBBO mids,
882 trading days 2022-05-11 → ~2026-03-20). Fills = mid − $0.10/spread slippage,
$1/contract/leg commission. Engine: `OpenIcStrategy` + `WindowEntryEngine`.

---

## 1. The strategy as taught

Pulver runs three overlapping 0DTE structures on SPX/XSP, "to win the day":

1. **Core — open iron condor.** Enter once in the first ~5–10 min. Short call
   and short put at the **gap-adjusted expected-move box edge** (open ± broker
   plus/minus, ≈ **10–15 delta**), **10-pt wings**. Target ≥ $1.00–1.50 credit
   (≈ $850 max risk on a 10-wide). **No stop loss.** Hold to cash settlement.
2. **Defense — broken-wing butterfly** placed 5–10 min later as a 5–15-pt buffer
   above/below the condor. Discretionary, sized for a small credit/debit.
3. **Butterflies at "key levels"** (gamma HVL pivot via Tanuki Trade) for
   low-risk/high-reward "profit traps". Fully discretionary.

Only the **core open iron condor** is mechanically reproducible. The defense and
butterfly overlays depend on intraday gamma reads and discretionary level
selection that cannot be backtested without inventing the discretion — so the
honest mechanical test is the core IC. Pulver's claimed live results (89.6% IC
win rate, ~27%/yr) rest substantially on those discretionary overlays.

Mapping to `OpenIcStrategy`: model-free `strike_mode: em_mult` (EM = ATM straddle
mid = the broker expected move); `em_mult ≈ 1.25` ≈ his 10–15-delta box edge;
`wing_width: 10`; `sl_mult_of_credit: 0` + `hold_*_breach: true` = no stop;
`tp_pct_of_credit: 0` = hold to the 15:59 force-close; traded every day.

## 2. As-taught result — refuted

Full clean CBBO data, 326 trading days:

| Sharpe (ann) | Total P&L | Win rate | Profit factor | Avg win / loss | Max DD |
|---|---|---|---|---|---|
| **−1.01** | −$7,430 | 75.8% | 0.84 | +$164 / −$605 | $10,736 |

Classic narrow-wing / far-OTM iron condor trap: a high win rate, but each breach
loses nearly the full $850 wing while wins collect only ~$1.50 credit. **Net
loser without the discretionary defense overlay.** Consistent with the repo's
prior CBBO finding ("BS credits are fiction at low delta / narrow width; real
SPX 0DTE needs 25pt+ width, 0.20+ delta").

### 2a. Adding the broken-wing-butterfly defense (full as-taught system)

A video screenshot confirmed the defense is mechanically specifiable relative to
the IC short strikes (Kc = short call, Kp = short put):

- Call BWB: `+1 (Kc-5)  -2 (Kc+5)  +1 (Kc+25)`  (asymmetric 10/20)
- Put  BWB: `+1 (Kp+5)  -2 (Kp-5)  +1 (Kp-25)`  (mirror)

entered ~15 min after the IC (09:50), held to **intrinsic settlement**
(`scripts/overlay_defense_bwb.py`, entry priced from the CBBO cache):

| System | Sharpe | Total P&L | WR | PF | Avg win/loss | Max DD |
|---|---|---|---|---|---|---|
| IC only | −1.01 | −$7,430 | 75.8% | 0.84 | +$164 / −$605 | $10,736 |
| IC + BWB defense | −0.52 | −$5,420 | 78.8% | 0.90 | +$196 / −$807 | $11,874 |

The defense behaves exactly as Pulver describes — it **widens the win zone and
rescues moderate breaches** (WR 75.8%→78.8%, Sharpe −1.01→−0.52, **+$6.2/day**).
The per-settlement-bucket P&L makes the mechanism explicit:

| settlement vs short strike | n | defense P&L/day |
|---|---|---|
| inside (no breach) | 233 | −$13 (carry) |
| call breach 0–5 | 13 | **+$618** |
| call breach 5–25 | 22 | **+$149** |
| call breach >25 (tail) | 3 | −$676 |
| put breach 0–5 | 10 | **+$397** |
| put breach 5–25 | 18 | −$6 |
| put breach >25 (tail) | 16 | −$510 |

So on Pulver's **far 10-delta** strikes the protection zone (short → short+25)
captures most breaches and the >±25 tail is rare → defense is net **positive**.
It still doesn't make the weak as-taught IC profitable, but it genuinely helps.

**The defense is NOT placed in the engine or the winner.** Overlaid on the
per-side-leg-out winner it *costs* −$25.6/day (Sharpe 5.18 → 3.28) — but NOT
because the mechanism fails (it still pays in every moderate-breach bucket and
even on no-breach days, +$112 to +$773/day). It loses because the winner's
**close ~35-delta strikes (em_mult 0.20)** make settlement land beyond short±25
on **53 of 275 days**, repeatedly triggering the BWB's −$1,000 tail (−$46.9k
total vs +$39.9k from the helping buckets). Pulver's fixed +5/+25 BWB offsets are
calibrated to his far strikes; on close strikes the tail dominates. The per-side
leg-out is the better-matched defense for the optimized winner, so the BWB is
left off — but the earlier claim that the defense is simply "counterproductive"
was wrong: it works as designed; it's just miscalibrated to close strikes.

*(Correction: an earlier version of this analysis priced the BWB exit at 15:59
mids and lost 75 days to a `settlement_forward` gap, understating the defense.
The numbers above use intrinsic settlement and parity-derived spot.)* (226/326 days had the
call-BWB fully quoted, 175/326 the put-BWB — the Kc+25/Kp−25 far legs are often
unquoted; defense applied where liquid.) Pulver's real edge is the discretionary
**Plan-C gamma-"key-level" butterflies**, which cannot be mechanized.

### 2b. The gap-adjusted Expected-Move Box

Pulver places strikes on a **gap-adjusted** box: top = today's OPEN + EM, bottom
= PREVIOUS CLOSE − EM (EM = pre-market broker plus/minus). A gap pushes the
gap-away side further out (mean-reversion protection). The baseline backtest uses
**symmetric** placement (spot ± `em_mult`·EM, EM = 09:35 ATM straddle): it
captures the *distance* (via the `em_mult` sweep) but not the asymmetry.

Added `strike_mode: gap_box` to `OpenIcStrategy` (call anchor = open, put anchor
= prev close, both cbbo-derived in the chain's scale; injected by the
orchestrator) and isolated it:

| Framework | symmetric | gap-box (Pulver) |
|---|---|---|
| As-taught (no stop, hold; `scripts/em_box_compare.py`, k=1.25/W10) | −1.25 | **−0.08** |
| As-taught (k=1.0/W20) | −1.03 | **+0.46** |
| Optimized (per-side leg-out + gates, imbalance off) | **4.02** | 2.02 |

So the gap box is a **real improvement in Pulver's no-stop framework** (+0.5 to
+1.5 Sharpe — it lowers win rate but cuts reversal breaches), **but is dominated
by the per-side leg-out** in the Sharpe-optimal framework: the leg-out already
manages breach risk better, and the box's forced asymmetry can't use the
credit-balance filter (`max_credit_imbalance`, worth ~+1 Sharpe) and doubles the
drawdown ($10.7k vs $5.7k). **The validated winner is correct to use symmetric
placement;** the gap box is kept as a tested option for the no-stop variant.

### 2c. Joint re-optimization: IC + tunable defense

Since the defense is real (2a) and is matched to far strikes, the full system was
re-optimized jointly (`scripts/optimize_ic_plus_defense.py`): em_mult × wing ×
defense outer-wing offset (or off), with per-side leg-out + gates, full-period IS.

| em | wing | defense | Sharpe | total$ | WR | maxDD |
|---|---|---|---|---|---|---|
| 0.20 | 30 | **off** | **4.02** | +58,498 | 53% | $5,671 |
| 0.20 | 30 | out15 | 2.76 | +44,550 | 48% | $9,027 |
| 1.00 | 30 | off | 1.52 | +2,642 | 51% | $2,209 |
| 1.00 | 30 | **out30** | **2.12** | +9,271 | 66% | $3,373 |

- The defense **helps far-OTM ICs** (em1.0/wing30: 1.52 → 2.12, WR 51%→66%) —
  Pulver's zone, mechanism confirmed once more.
- But **no defense variant beats the close-strike bare IC (4.02)**; on the
  winner, defense only lowers Sharpe. The per-side leg-out is a *strictly better*
  version of the same defense (caps the breach loss without the BWB's >±25 tail
  or daily carry). They solve the same problem; the leg-out wins.

So the joint re-optimization leaves the winner unchanged, and no defense config
is worth WF-CV-ing (none beats the bare IC even in-sample). The defense earns its
keep only in Pulver's no-stop, far-strike framework.

### 2d. Full delta range: win rate vs Sharpe (the central tradeoff)

Sweeping em_mult from near-ATM to ~4-delta (`scripts/delta_tradeoff.py`,
realized delta from each day's ATM-straddle IV, wing 30, full period):

**A — Pulver's framework (no stop, hold to settlement):**

| em_mult | ~delta | WR% | Sharpe | total$ |
|---|---|---|---|---|
| 0.20 | 30d | 54% | 1.30 | +42,911 |
| 0.75 | 17d | 67% | 0.89 | +10,942 |
| 1.25 | **10d** | **80%** | 0.50 | +3,358 |
| 1.50 | 8d | 86% | 0.10 | +598 |
| 2.00 | 4d | 92% | 1.07 | +3,508 |

**B — leg-out (per-side SL + TP + gates):** 0.20/29d → Sharpe **4.02**;
1.25/10d → −0.44; 2.00/4d → −2.24.

Findings:
- **At Pulver's ~10-delta the win rate is 80%** (rising to 86–92% at 6–4 delta),
  matching his 85–89% claim. em_mult=1.25 is empirically ≈10-delta (an earlier
  analytic estimate of ~16d was wrong), so 10-delta *was* covered; this surfaces
  its WR/Sharpe explicitly and extends to 4-delta.
- **High WR ≠ high Sharpe.** At 10-delta Sharpe is only ~0.50; the 92%-WR
  4-delta is 1.07. None approach the near-ATM Sharpe (4.02). The high win rate is
  paid for by rare full-width losses → poor risk-adjusted return (the
  premium-selling inversion).
- **The leg-out only helps near-ATM and is destructive far-OTM** (1.25→−0.44,
  2.0→−2.24): a 2× stop on a tiny far-OTM credit triggers on noise and locks in
  losses that hold-to-expiry would have won. This is exactly *why* Pulver runs
  **no stop** at his delta — correct for his delta.
- So the max-Sharpe winner (~30-delta, 53% WR) is a fundamentally different trade
  from Pulver's (~10-delta, 80% WR). His high-WR IC is low-Sharpe on its own; his
  real returns come from the discretionary butterflies.

### 2e. Plan C — the "key-level" butterflies (modeled)

Pulver places 0DTE butterflies at gamma "key levels" (the HVL pivot / where price
gravitates) and manages them for 50%+ return or a end-of-day pin. Modeled with the
repo's existing butterfly evaluator + GEX engine (`scripts/analytical_butterfly_eval.py`,
`zerodte/backtest/gex.py`: `magnet_strike` = dealer-gamma pin = his "key level";
`gamma_flip` = HVL pivot). 20–25-wide, entry 10:00, CBBO-priced, full period.

**His exact recipe (gamma-magnet placement) has no edge:**

| placement (50% target) | Sharpe (zero-cost) | Sharpe ($0.05/leg) |
|---|---|---|
| gamma magnet (his "key level") | 1.93 | 0.57 |
| centered / ATM | 2.08 | 0.80 |
| anti (fade morning move) | 3.39 | 1.74 |

The gamma-magnet placement is the *weakest* — his "skate where the puck is going"
gamma read is not the source of edge (matches the repo's prior "GEX placement
refuted"). Hold-to-pin is a losing lottery (Sharpe −0.68); his 50%-return rule is
what turns a profit — but at 50% target it's cost-fragile (1.74 → 0.33 → −1.12 as
slip goes $0.05 → $0.10 → $0.15/leg).

**The one robust, WF-CV-validated, cost-robust butterfly is directional — and NOT
his method.** A sweep (120 combos × 4 expanding folds, selected on net train
Sharpe) picked **put-side / 2:1-target / 25-wide** in *every* fold; OOS net Sharpe
4.14 / 2.32 / 4.78 / 7.37, pooled **+$231/trade with costs**, and cost-robust
(Sharpe 3.87 → 3.41 → 2.95 at $0.05 → $0.10 → $0.15/leg — letting winners run to
2:1 makes per-leg slippage a small fraction). But the direction comparison shows
it is a **pure downside-drift bet**, not a structural pin edge:

| dir (2:1, 25-wide, $0.10) | Sharpe | P&L |
|---|---|---|
| put (downside fly) | +3.41 | +$156,847 |
| call (upside fly) | **−5.50** | −$238,468 |
| anti (mean-revert) | +1.56 | +$73,342 |
| coin (random) | −1.88 | −$83,924 |

Put wins, the call mirror loses catastrophically → the edge is direction (a
persistent downside-skew / pullback harvest over 2022–2026), expressed via a put
fly, with 47.6% WR and positive skew (a few big winners — so Sharpe flatters a
fat-tailed return). It carries melt-up bleed and gap-down tail risk and is
regime-dependent. The `anti` mean-reversion fly (1.56) is the more defensible,
less purely-directional edge.

**Verdict:** butterflies are fully modelable. Pulver's *neutral gamma-key-level*
butterfly has no mechanical edge; the profitable flies are a *directional*
downside/mean-reversion bet that is not what he teaches. So his discretionary
gamma read is, again, not the source of any edge — consistent with the IC, the
EM-box, and the defense: the mechanizable edges are different from his stated method.

## 3. Optimization for maximum Sharpe

Full-period landscape sweep (`scripts/sweep_open_ic_pulver.py`),
em_mult × wing × TP × SL, no gates, in-sample:

- The Sharpe-optimal **robust plateau** is the **opposite** of Pulver's recipe:
  **close-to-ATM** short strikes (`em_mult ≈ 0.20–0.25`, ~35–40 delta),
  **wider wings (20–30)**, a **50% take-profit**. In-sample Sharpe **≈ 2.0**.
- Tiny-n high-Sharpe cells (e.g. em1.25/wing50, n=45, Sharpe 4.2) are overfit
  noise — far-OTM wings are rarely quoted, so few days qualify.
- **Calendar gates do not help** the plateau in-sample (em0.25/wing20/tp0.5
  went 2.05 → 1.92 when gated). Simpler / trade-every-day wins on raw IS Sharpe.

## 4. WF-CV validation — the in-sample optimum is overfit; the real edge is the leg-out

WF-CV, ratio 4:1, 5 expanding folds (`scripts/run_wf_cv_open_ic_pulver.py`,
`scripts/run_v2_wf_cv_open_ic.py`). Mean **out-of-sample** fold Sharpe:

| Config | SL scope | Gates | OOS Sharpe | Profitable folds | Worst fold |
|---|---|---|---|---|---|
| em0.25 / wing20 / tp0.5 (IS optimum) | total | none | **0.09** | 2/5 | −$11,183 |
| em0.20 / wing30 / tp0.5 / sl2.0 | **per_side** | none | **1.66** | 3/5 | −$3,053 |
| em0.20 / wing30 / tp0.5 / sl2.0 | **per_side** | FOMC+1, Fri | **4.02** | 4/5 | −$74 |

Findings:

- **The in-sample Sharpe-2.0 cell is overfit** — under a conservative whole-IC
  stop (`sl_scope: total`) it collapses to OOS Sharpe **0.09** (2/5 folds).
- **`sl_scope: per_side` (asymmetric leg-out) is the primary edge** — it lifts
  OOS Sharpe 0.09 → **1.66** with no gates. Mechanism: when one side is breached,
  close that spread at the 2× stop and **hold the untested side**, which decays
  to ~0 by EOD and banks its full credit. This shrinks the average loss
  (−$248 vs −$605 as-taught) far more than it caps wins → higher Sharpe.
- **Calendar gates** (skip FOMC-next-day + Friday) add the rest: 1.66 → **4.02**
  and tighten robustness (worst fold −$3,053 → −$74).
- Two independent grids converge on the **same structural winner**:
  `em_mult=0.20, wing_width=30, sl_mult=2.0, tp=0.50, per_side`.

This also reconciles the repo's earlier note that `total` "won 1.6–1.7× PnL on
clean data" and per_side was "an artifact": **`total` maximizes raw P&L, but
`per_side` maximizes Sharpe** — it trades upside for much smaller drawdowns. The
user's objective here is Sharpe, so per_side is correct.

### Validated winner (config: `config/config_open_ic_pulver_wfcv.yaml`)

`em_mult 0.20 · wing 30 · TP 50% of credit · per-side SL 2× side-credit ·
skip FOMC+1 & Fridays · enter 09:35 · hold to 15:59`

Full-period in-sample: Sharpe **5.18**, PF **2.94**, +$60,124, max DD **$4,392**,
avg win +$607 / avg loss −$248. **WF-CV OOS: mean fold Sharpe 4.02, 4/5
profitable, worst fold −$74.**

## 5. Markov regime gate — tested, NOT necessary

Per the goal, the `markov-2-hedge-fund-method` (FILTER mode, causal trailing-20d
BULL/BEAR/SIDEWAYS state on prior-day close) was applied
(`scripts/wf_markov_gate_open_ic.py`, `scripts/markov_overlay_open_ic.py`):

- In-sample it *looks* powerful (IC after a BULL day → Sharpe 7.3; "trade only
  after BULL" → 9.0), but this is hindsight on small subsets (n≈25–86).
- **Walk-forward, the gate is not learnable.** On every fold's train window all
  three states show positive mean P&L, so the learned gate selects "trade all"
  and the gated OOS series is **identical** to ungated. On the validated winner,
  imposing BULL-only / skip-SIDEWAYS actually **hurts** OOS — the calendar gates
  already remove the bad days.
- Conclusion: the regime edge does not survive honest WF; the Markov gate adds
  nothing on top of the per-side + calendar-gate structure. **Not deployed.**

## 6. The execution caveat (this is the whole ballgame)

The entire validated edge lives in **`sl_scope: per_side`** — the asymmetric
leg-out. The backtest assumes (a) the breached side fills at ~2× its entry credit
on a per-minute mark, and (b) the untested side, once held, decays to ~0 without
a same-day reversal threatening it. Both are optimistic:

- 0DTE deltas move violently; a per-minute stop trigger can catch a transient
  spike whose actual fill is worse, and fast moves slip the tested leg.
- A whipsaw day (rally then sell-off) can breach **both** sides after you've
  already taken the first stop.

If live leg-out fills are materially worse than mid−$0.10, the edge degrades
toward the `total`-scope result (OOS Sharpe ~0). **Before risking capital this
needs a paper/live leg-out fill study** (cf. the open MEIC scale-out reconciliation
gap — scale-out/leg-out fills are exactly what the backtest can't yet prove).

## 7. Bottom line

- **Pulver's core IC as taught (far-OTM, 10-wide, no stop, hold) is a net loser
  on real SPX 0DTE fills** (Sharpe −1.01). His live edge is the discretionary
  butterfly/defense overlay, which is not mechanically reproducible.
- Re-optimized into the open-IC family and validated via WF-CV, the
  **maximum robust Sharpe is ≈ 4.0** (mean OOS, 4/5 folds) with
  `em0.20 / wing30 / 50% TP / per-side 2× SL / skip FOMC+1 & Fri` — but that
  number is **contingent on the per-side leg-out execution model** holding up in
  live fills. Under a conservative whole-IC stop there is **no robust edge**.
- The **Markov regime gate is unnecessary** here and not causally learnable.

### Reproduce

```bash
PY=.venv/bin/python
# as-taught (loser)
$PY -m scripts.run_v2_backtest --config config/config_open_ic_pulver.yaml
$PY -m scripts.ic_csv_stats results/open_ic_pulver/trades.csv "as-taught"
# landscape sweeps (in-sample)
$PY -m scripts.sweep_open_ic_pulver            # no gates
$PY -m scripts.sweep_open_ic_pulver --gates    # calendar gates
# WF-CV (OOS): total vs per_side, no gates
$PY -m scripts.run_wf_cv_open_ic_pulver --ratio 4 --folds 5
WF_SL_SCOPE=per_side $PY -m scripts.run_wf_cv_open_ic_pulver --ratio 4 --folds 5
# WF-CV with repo grid (per_side + gates) -> mean OOS Sharpe 4.02
$PY -m scripts.run_v2_wf_cv_open_ic --config config/config_spx_0dte_open_ic.yaml --ratio 4 --folds 5
# validated winner + Markov gate check
$PY -m scripts.run_v2_backtest --config config/config_open_ic_pulver_wfcv.yaml
$PY -m scripts.wf_markov_gate_open_ic results/open_ic_pulver_wfcv/trades.csv
```
