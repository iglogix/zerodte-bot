# ORB Directional — Debit-Spread Excursion Diagnostic (2026-06-18)

**Question:** the surviving ORB directional edge currently uses an OTM **credit**
spread (PCS/CCS ~0.5×EM OTM, theta tailwind). Should it instead use an ATM
**debit** spread to capture the directional continuation? Gate: does the
underlying continue with the breakout far enough AND fast enough to beat 0DTE
theta on a long-premium structure?

**Method:** ran the surviving credit config (`config_orb_dir_breakout_spx.yaml`,
mtol=3.0 / em0.5 / w20 / min_em=20 / breakout_level) to get the real entry set,
then overlaid the SPY×10 1-min path from each entry → 15:50 and measured signed
excursion in the trend direction. Underlying-behavior only, no option pricing —
the cheap gate before building a debit variant.
`scripts/orb_dir_excursion_diag.py`, 2022-04-18 → 2026-06-11, **n=233 entries**.

## Result

| metric | value |
|---|---|
| MFE (max favorable excursion) | median **22.1 pt** (p25 10.9, p75 37.4, p90 59) |
| MAE (max adverse excursion) | median **19.2 pt** (p75 38.5, p90 55.3) |
| **Net move at EOD** (signed, with-trend +) | **median +0.8 pt, mean 0.0, 52% positive** |
| Time to MFE | median **142 min** (p25 27, p75 301); only 38% within 60 min |
| Share of MFE captured by +30 min | median **5%** |
| Reach ≥15 pt favorable (any time) | 68% |
| Reach ≥20 pt favorable (any time) | 56% |

## Verdict — debit spread NOT justified

The favorable move exists in magnitude (median MFE 22 pt) but fails both things a
long-premium 0DTE structure needs:

1. **Too slow.** Median time-to-MFE is **142 min** and only **5%** of the eventual
   MFE shows up in the first 30 min. A 0DTE long debit spread bleeds theta for
   hours while waiting for the move — the decay races (and usually beats) the gain.
2. **Round-trips to zero.** Net directional move at EOD is **≈0 (median +0.8,
   52% positive)** and **MAE ≈ MFE** (19 vs 22 pt). The breakout doesn't persist;
   it whips both ways and mean-reverts. To extract the MFE you'd need near-perfect
   intraday exit timing — exactly the execution-contingent fragility we avoid.

This is the *opposite* of what long premium wants and *exactly* what the OTM
credit spread monetizes: slow, range-bound, mean-reverting price action lets the
short strike (placed ~0.5×EM OTM, breach-tail managed by the retest + min_em
filter) decay to profit. The credit structure is correctly matched to the data.

**Recommendation:** keep the credit spread. Do not build the ATM debit variant —
the underlying continuation is neither fast nor persistent enough to overcome
0DTE theta. (A same-strike debit would be a parity no-op anyway; only an ATM
reposition is a real change, and the data refutes it.)

Reproduce:
```bash
.venv/bin/python -m scripts.orb_dir_excursion_diag \
    --config config/config_orb_dir_breakout_spx.yaml --start 2022-04-18 --end 2026-06-11
```

---

## Follow-up: is there a FADE (mean-reversion) edge? — NO (2026-06-18)

The round-trip (net EOD ≈0, MAE≈MFE) raised the question: does betting AGAINST the
breakout beat betting with it? Compared full-span with identical structure params
(em0.5 / w20 / min_em20 / tp0.75 / mtol3.0, both breach sides held), changing only
the directional choice (`scripts/orb_dir_fade_compare.py`, real CBBO):

| variant | trades | total | $/trade | win% | Sharpe/tr | by year (total) |
|---|--:|--:|--:|--:|--:|---|
| **with_trend (survivor)** | 233 | **+$31,207** | 134 | 80% | 0.22 | '22 +9.9k / '23 +8.2k / '24 +5.9k / '25 −2.3k / '26 +9.6k |
| fade_breakout (invert side) | 195 | +$7,599 | 39 | 74% | 0.06 | '22 +4.5k / '23 −1.7k / '24 −0.3k / '25 −5.8k / '26 +10.9k |
| opposite_level (failed-breakout) | 127 | +$7,148 | 56 | 77% | 0.08 | '22 +1.6k / '23 +2.2k / '24 −4.2k / '25 −3.3k / '26 +10.8k |

(Sharpe here is a crude per-trade mean/sd; both-breach-hold differs from the
survivor's published 3.29 — read the table as a *relative* ranking only.)

**Verdict — no fade edge.** Both fade variants are positive in total only because
of a strong **2026 (partial year)**; across the bulk middle years (2023–2025) the
fade loses (with_trend +$11.8k vs fade_breakout −$7.8k / opposite −$5.3k), with
lower win% and ~1/3 the $/trade. The mean-reversion is real in the *underlying
path* but is **not monetizable** via these credit structures: the breakout keeps a
mild persistent continuation bias in the tails (MFE 22 > MAE 19), so an
against-trend spread sells into the slightly-more-likely-to-breach side and eats
more breach losses. The with-trend short-premium spread already captures the
round-trip better — it's centered to survive the ±20pt swings and collect theta.
Consistent with the original IC→directional finding (the against-trend leg was a
net drag). **Keep with_trend; fade is dead.**

Repro: `.venv/bin/python -m scripts.orb_dir_fade_compare --start 2022-04-18 --end 2026-06-11`
(adds opt-in `strategy.fade_breakout` flag in orb_engine, default off / no live change).
