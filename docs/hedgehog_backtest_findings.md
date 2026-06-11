# Hedge Hog (SPY) — Backtest Findings

**Verdict: the Hedge Hog as taught is REFUTED on SPY 2020–2026.** The
two-cycle package loses money in every fill/delta/DTE/trigger variant
tested (−$1.3k to −$4.6k per 1-lot, PF 0.53–0.85). Both of the course's
signature ideas *subtract* value: the near-dated hedge costs far more than
the tail relief it buys, and the calendar split (30 DTE hedge vs 100 DTE
hog) makes the hedge bleed out 70 days before the risk it insures expires.
What's left after removing them — a plain 45–60 DTE ~10Δ short put with a
50% take-profit — is the only configuration with a real edge, and that is
just the classic short-put trade, not the course's structure.

Strategy spec: [Hedge-Hog-Options-Spread-Strategy.md](Hedge-Hog-Options-Spread-Strategy.md)
(swing skew-harvest: ~30 DTE 30Δ/20Δ long put vertical + ~100 DTE ~7Δ naked
short put, net credit, four management triggers, close-&-recycle).

## Data & engine

- **Data**: ThetaData (VALUE tier) EOD NBBO chains, all 524 Thu/Fri SPY
  expirations, **2020-01-02 → 2026-06** (tier history boundary: 2019-12
  returns 471/not-entitled). One pickle per expiration in
  `data/hedgehog_eod/SPY/` (207 MB), built by
  `scripts/backfill_hedgehog_eod.py` (bulk windows + day-by-day fallback
  for ten Q2-2022 weeklies whose bulk request crashes the terminal).
  Span includes the COVID crash, 2022 bear, Aug-2024 volmageddon and
  Apr-2025 tariff break — the fast-crash scenarios that are this
  structure's only real danger.
- **Quality**: 0 crossed / 0 zero-ask quotes in 142k sampled; hog-zone
  (10–25% OTM) put half-spreads median $0.01, p90 $0.035. Execution is a
  second-order effect here (unlike SPX 0DTE).
- **Granularity**: EOD only — matches the strategy (positional, daily
  triggers); intraday touches of trigger 4 are seen one close late, fills
  at the 16:00 NBBO snapshot.
- **Greeks**: no external rate/dividend inputs — implied forward per
  (date, expiry) from put-call parity, then Black-76 (discount≈1) IV/delta.
- **Engine**: `src_v2/backtest/hedgehog.py` (9 unit tests on synthetic BS
  surfaces). Runner `scripts/backtest_hedgehog.py`, sweep
  `scripts/sweep_hedgehog.py`, analyzer `scripts/analyze_hedgehog.py`.

## Encoding decisions (spec §14 ambiguities)

1. **Entry cadence**: enter whenever capacity allows (base 1 concurrent,
   re-enter same day after close). Ladder variant: 1 per 3 days, max 4.
2. **Hog selection**: start ~7Δ; step toward ATM (≤12Δ) until
   credit > debit, else skip.
3. **PoP gate**: P(S at hedge expiry > deep downside breakeven), breakeven
   = lower edge of the front-expiry profit zone (sticky-strike hog IV).
   In practice the gate never binds on SPY (base == no_pop_gate == pop_78):
   the deep breakeven sits 10–15% below spot, PoP ≈ 1.
4. **Exits**: all four triggers, base action = close package & recycle
   (the instructor's ~90% path). Rolls not modeled.
5. **Fills**: mid ± `fill_k`·half-spread (base 0.5), $1/contract/leg open
   commission + $0.15 fees per leg each way.
6. **Margin**: Reg-T naked-put formula + vertical debit.
7. **Not modeled**: early assignment, multi-symbol portfolio, portfolio
   margin, futures sleeve (/ZB /GC etc. — ThetaData is OPRA-only).

## Results (1-lot, fill_k 0.5 unless noted)

### As taught — base run, 2020-01 → 2026-06

118 trades, **24.6% win rate** (vs the marketed "95%" PoP and the
instructor's own 54% realized), **−$2,443**, PF 0.74, Sharpe −0.38,
avg hold 19.9 days, median trade −$52, max 14 consecutive losers.

By year: −$2,302 (2020), −$464 (2021), **+$2,690 (2022)**, +$127 (2023),
−$1,115 (2024), −$1,705 (2025), +$327 (2026 H1).

The **only good year was the slow 2022 bear** — exactly the regime the
mechanics §7 predict (slow grind down: vertical pays, hog buffered).
Fast crashes destroy it (COVID window −$2,115; Apr-2025 −$1,469;
Aug-2024 −$641) and bull years bleed (the vertical's debit decays faster
than the far hog's credit). Notably the instructor's real-money record
(Nov 2022 → 2023, +$22k claimed) coincides with the single best regime
window in our 6.5-year span.

**Trigger 1 (hog at 50% → close package) is the bleed engine: 62 of 118
exits, 6% win rate, −$3,171.** When the hog banks 50% fast (market up),
the 30-DTE vertical has lost more than the hog gained. Full-hold exits
(vertical <7 DTE) are the only positive group (+$1,479).

### Variant sweep (all full-span)

| variant | n | WR | P&L | PF | Sharpe |
|---|---|---|---|---|---|
| base (as taught) | 118 | 25% | −$2,443 | 0.74 | −0.38 |
| fill at mid | 119 | 26% | −$1,278 | 0.85 | −0.22 |
| fill at NBBO | 119 | 24% | −$3,753 | 0.63 | −0.55 |
| hog 5Δ / 10Δ | 118/119 | 25/38% | −$2,493 / −$1,809 | 0.74/0.76 | <0 |
| hog 75 DTE | 128 | 23% | −$3,307 | 0.61 | −0.54 |
| no hog-TP / no vert-TP / no fast-breach | | | −$1.3k…−$1.6k | | <0 |
| hold to 3 DTE | 112 | 21% | −$4,591 | 0.53 | −0.69 |
| ladder 4×3d | 427 | 23% | −$6,896 | 0.76 | −0.38 |
| **hog_only (no hedge, same entries)** | 118 | **79%** | **+$1,101** | 1.13 | +0.13 |
| **single_cycle 45–60 DTE ("1-1-1")** | 123 | **87%** | **+$3,542** | 1.96 | +0.61 |
| as taught, 3:1 R:R enforced | 133 | 70% | −$688 | 0.89 | −0.12 |
| single_cycle, 3:1 R:R enforced | 161 | 87% | −$960 | 0.87 | −0.15 |

**On the course's "~3:1 reward:risk" guide**: the natural 30Δ/20Δ
construction on SPY prices at a median **4.4:1** (only 8% of entries land
in the 2.5–3.5 band — the band reflects the course's 2023 example chains,
not SPY 2020-2026). Enforcing 3:1 via short-strike selection
(`hedge_rr_target=3.0`) raises the win rate to ~70% — much closer to the
instructor's claimed feel — while still losing money and worsening the
tail (worst −$2,182); it also flips the profitable single-cycle variant
negative. The refutation is robust to this construction choice.

A second 22-variant sweep around the 3:1-enforced base (fills, hog
delta/DTE, long delta 25–35Δ, R:R 2–4, TP/exit ablations, ladder,
single-cycle combinations) found no robust region: signs flip between
adjacent settings (2:1 +$607 vs 4:1 −$2,634; 25Δ +$241 vs 35Δ −$2,597;
5Δ hog −$2,002 vs 10Δ +$679) and profits only survive at pure mid fills —
the signature of noise around zero. The 20Δ-short single-cycle family and
its naked-put form remain the only neighborhoods that are profitable at
every tested perturbation.

Two inversions of the course's premise:

1. **The hedge is negative-value.** Removing it (same entries, naked put
   only) turns −$2,443 into +$1,101; the hedge improves the worst trade
   only −$2,558 → −$1,687 while costing ~$3.5k. Cheap convexity isn't
   cheap enough at SPY's skew.
2. **The calendar split is negative-value.** The same-cycle "1-1-1"
   (§13 — which the instructor explicitly calls inferior) is the only
   profitable package: with both legs in one expiry the hedge decays
   *with* the hog, trigger 1 flips to 90% WR (+$6,130), max losing streak
   drops 14 → 2, and it makes money in 5 of 7 years (incl. both 2021
   grind-up and 2022 bear; losses only COVID −$912 and 2025 −$198).

### Single-cycle robustness (not knife-edge, but hedge still drag)

| check | n | WR | P&L | PF | Sharpe | worst |
|---|---|---|---|---|---|---|
| sc base | 123 | 87% | +$3,542 | 1.96 | 0.61 | −$1,436 |
| sc NBBO fills | 120 | 87% | +$2,081 | 1.44 | 0.33 | −$1,725 |
| sc DTE 40–50 | 138 | 86% | +$1,209 | 1.19 | 0.19 | −$1,760 |
| sc DTE 55–70 | 91 | 85% | +$2,772 | 1.75 | 0.39 | −$2,163 |
| sc hog-TP 40% / 60% | 150/101 | 83/88% | +$3,546 / +$2,590 | 1.90/1.63 | 0.60/0.30 | |
| **sc hog_only (naked put alone)** | 123 | **94%** | **+$9,457** | **2.56** | **1.00** | −$1,784 |

The naked 45–60 DTE ~10Δ put with 50% TP and ~3-week time stop beats the
hedged version 2.7× (the vertical costs ~$5,900 over the span for ~$350 of
worst-trade relief). Profile: +28%/yr on avg Reg-T margin (~$5.3k/trade),
positive **every year except 2020 (−$103)**, 16-day avg hold. COVID window
itself −$2,063 — the naked tail is real and must be sized for (fixed
small fraction of capital; a cash-secured basis cuts the return to ~3%/yr).

## Caveats

- Single symbol (SPY); the course diversifies ~10 positions across ETFs,
  futures (/ZB /GC /NG), and stocks. The futures sleeve (different skew)
  is untested here and needs CME data. QQQ/IWM are one `--root` flag away.
- The profitable variants emerged from a 15-variant sweep — selection
  effect is mitigated by neighborhood robustness and a causal mechanism
  (calendar-mismatch removal), but treat as *promising*, not validated.
  ~19 trades/yr is too few for the usual WF-CV gate.
- EOD triggers see intraday breaches one close late; rolls (the
  instructor's occasional adjustment path) unmodeled; early assignment
  unmodeled.
- The 95% marketing number is the day-1 PoP of the structure; realized WR
  as taught is 25% with management, 79–94% only for the short-put-dominant
  variants — none of which is the taught structure.

## Bot decision

**No live bot for the Hedge Hog as taught** — refuted (consistent with the
jade-lizard and butterfly outcomes: no edge → no bot). The defensible
live candidate from this work is the **single-cycle naked-put recycle**
(45–60 DTE ~10Δ SPY put, 50% TP, close at 7 DTE / ~3 weeks, optional
30Δ/20Δ vertical if tail comfort is worth ~60% of the P&L), which is the
classic short-put strategy rather than the course's product. Build only
with explicit naked-margin sizing rules (≤30% capital, fixed-fractional).
