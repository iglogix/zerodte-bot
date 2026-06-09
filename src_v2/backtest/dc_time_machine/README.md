# DC Time Machine — backtest verdict

Independent intraday backtest of the **"Double Calendar Time Machine"** strategy
(Steve Bernich / Navigation Trading, interviewed on Theta Profits, YouTube
`JtGW1wNFNIY`): enter a double calendar on SPX, then once it gains a little, *transform*
it in one order into a **risk-free iron condor** (sell the back-dated longs, buy
front-expiry wings) such that the collected credit `C ≥ debit D + wing W`.

Design: [`docs/superpowers/specs/2026-06-07-dc-time-machine-backtest-design.md`](../../../docs/superpowers/specs/2026-06-07-dc-time-machine-backtest-design.md)
· Plan: [`docs/superpowers/plans/2026-06-07-dc-time-machine-backtest.md`](../../../docs/superpowers/plans/2026-06-07-dc-time-machine-backtest.md)
· Raw numbers: [`docs/superpowers/specs/2026-06-07-dc-time-machine-verdict.md`](../../../docs/superpowers/specs/2026-06-07-dc-time-machine-verdict.md)

## TL;DR — the verdict

**The strategy is real but its edge lives entirely in execution quality, with
essentially zero margin.** It is profitable **only at mid-price fills**; under any
realistic slippage it loses heavily. **NO-GO for a naive automated bot** as specified.
The one path to viability is a live confirmation that the 4-leg transform combo can
be filled at/near mid — which is a discretionary-execution skill, not a given.

1. **The harness reproduces the author's published numbers at mid fills.** Over his
   exact window (Feb–Apr 2026, 62 sessions): our single-entry sim gives ~53–60 trades,
   **53–59% win**, **15–43% max-profit-hit** vs his published **~106 trades / 63% win /
   25% max-profit** (he scales into several entries per day; we do one DC per session).
   Same ballpark → the model and data are trustworthy, *and* his edge is real **at mid
   execution**.
2. **The edge is 100% fill-dependent.** `fill_k` = 0 is mid, 1 is full bid/ask cross:

   | delta | k=0 (mid) | k=0.5 | k=1 (cross) |
   |------:|----------:|------:|------------:|
   | 0.30  | **+$990** (tr 9%, PF 1.28) | −$28,460 | −$25,100 |
   | 0.35  | **+$1,380** (tr 17%, PF 1.46) | −$28,582 | −$25,710 |
   | 0.40  | **+$5,395** (tr 25%, PF 3.47) | −$33,260 | −$26,970 |
   | 0.45  | **+$6,485** (tr 54%, PF 3.57) | −$34,512 | −$31,130 |

   (84 sessions, Jan–May 2026, 1 contract/trade, front_dte=9, wing=5.) **Break-even is
   essentially `k≈0`** — you must capture ~95–100% of mid on the transform, or the
   strategy is deeply negative. This is exactly the trap behind the standing rule
   *"never trust optimizer results that ignore execution costs."*
3. **Delta is the key lever, and it's a tradeoff.** Higher delta ⇒ richer front
   verticals ⇒ the DC banks the wing with far less appreciation ⇒ **more transforms**
   (35Δ→17%, 45Δ→54%), but a **narrower risk-free tent** ⇒ **lower max-profit rate**
   (43%→13%). At mid, more transforms wins (net rises with delta); but every config dies
   at realistic fills.
4. **Spreads and transforms are positively correlated — the worst kind.** Transforms are
   most available in high vol (the DC appreciates fastest), but high-vol spreads blow out
   (empirically: SPX front-week half-spread median **$0.25 calm → $1.80, p90 $9+ in the
   Apr-2025 tariff week**), eating the entire transform margin precisely when the
   opportunity appears.

## Addendum (2026-06-09): the catastrophic-loss columns are two artifacts — the honest result is ~flat

A follow-up review of the `k≥0.5` columns in #2 found the large losses are **measurement
artifacts**, not the strategy. Corrected, the picture is a **near-zero edge**, not a blow-up
— which *strengthens* the NO-GO reasoning (no clear edge) while **retracting the "loses
heavily" magnitude**. Reproduce: `PYTHONPATH=. .venv/bin/python scripts/dc_slippage_audit.py`.

**Artifact 1 — stale cache quotes inflate the cross-fill cost.** The transform's 4-leg
bid/ask slippage `S_t` (the credit lost mid→cross) head-lined at **$647/contract**, but that
mean is dragged up by a few sessions with **16–38%-of-mid** quotes (e.g. a $9.50 spread on a
$54 option) — thin/stale ThetaData snapshots, not executable markets:

| `S_t` (transform slippage) | $/contract |
|---|---|
| raw mean (inflated) | ~~$647~~ |
| median | $380 |
| clean sessions only (<\$5/sh) | $325 |
| at realistic 1.5% SPX spreads | **~$171** |

Per-leg spreads are bimodal: median **3.8% of mid**, with 14/48 legs ≤2% (real SPX) and
10/48 legs >10% (stale). Defensible full-cross slippage is **~$170–325/contract** —
*comparable to* the **DC's profit at transform (~$157/contract, ≈16% of debit)**, not 4× it.

**Artifact 2 — the stop was marked at the adverse fill.** The simulator evaluated the mental
stop on the DC's *cross-fill* close value, so the DC looked instantly ~20% underwater and
stopped on bar 1 — that, not transform losses, produced the −$28k at `k≥0.5` (67–72 of 72
trades "stopped"). A mental stop should be monitored at mid and only pay the spread on action.

**The clean estimate** — stop monitored at **mid**, spreads **capped at 2% of mid**, only the
transform pays the spread (`k_tx`):

| transform fill | transforms | total P&L (72 trades) | avg/trade |
|---|---|---|---|
| mid (k=0) | 13 | +$1,325 | +$18.4 |
| half-spread (0.5) | 3 | +$367 | +$5.1 |
| full cross (capped) | 0 | +$125 | +$1.7 |

So with sane bookkeeping the strategy is **roughly flat**: a small positive that needs
near-mid transform fills (+$18/tr) and decays to ~0 at full cross — where essentially **no
transform clears `C ≥ D+W`**, so it just holds the DC to EOD (≈ breakeven). ~8 option-leg
round-trip commissions would push it negative. **Verdict unchanged (NO-GO); reasoning
corrected: a near-zero, fill-and-cost-dependent edge — not a heavy loser.**

## Why "risk-free" is misleading in practice

The transform *is* risk-free by construction (the floor `C−D−W ≥ 0` is asserted on every
transform). But the resulting IC almost never reaches **max** profit — at 45Δ the tent is
~45pt wide and SPX moves ~50pt/day, so over a ~9-day hold it is almost always breached and
the IC settles at its **~$0 floor**. So the advertised *"only thing that can happen is you
make a little or a lot"* collapses to *"you make a little"* — and that little is smaller
than the spread you cross to get it.

## What would change the verdict

The single open question is **achievable transform-fill price**. SPX complex orders can
sometimes price-improve toward mid via the CBOE complex-order book, and a patient limit
that only fires when the market is tight (low vol) might capture near-mid. The author is
a discretionary trader doing exactly this by hand. **Before any capital, the next step is
a live/paper measurement of realized net-credit vs mid on the 4-leg transform** (the
existing TastyTrade integration places the IC as one complex order). If real fills land
within a few cents of mid *and* an entry filter avoids wide-spread regimes, the strategy
could be marginally viable. If they don't, it isn't — and this backtest says they need to.

## Data coverage & honesty caveats

- **Vendor:** ThetaData (OPTION.VALUE intraday), 10-min NBBO bulk, SPXW. Spot/settlement
  via SPX put-call parity (no SPY basis). All quotes are SPX; SPY is not used.
- **Coverage:** focused on **Jan–May 2026** (his exact Feb–Apr cross-check window + a few
  weeks of settlement headroom each side), plus warm validation windows (calm Feb 2026,
  Apr-2025 tariff crisis). **Full 2022→2026 history was attempted but is impractical via
  ThetaData**: cold-storage reads are ~18–40s/call regardless of bar size, and the local
  terminal serializes — a full backfill is ~10h and fragile. The verdict therefore rests
  on a single-year, multi-regime sample that *does* reproduce his published numbers; it is
  **not** a 4-year walk-forward. Treat the in-sample sign (edge only at mid) as the robust
  finding, not the magnitude.
- **Resolution:** 10-min bars (transform fires over minutes-to-hours, so adequate; matches
  the project's fly/jl caches). **Settlement:** parity forward at the front expiry's own
  0DTE last-minute panel.
- **Single entry/session:** we model one DC per session; the author scales into several.
  This lowers our trade count vs his but does not change the per-trade economics.

## Reproduce

```bash
# Theta Terminal must be running (OPTION.VALUE): java -jar tools/theta/ThetaTerminal.jar "$EMAIL" "$PASS"
.venv/bin/python -m src_v2.backtest.dc_time_machine.run_backtest backfill \
    --start 2026-01-15 --end 2026-05-15 --out data/dc_cache_full --source theta --pad 0
PYTHONPATH=. .venv/bin/python scripts/dc_verdict.py --cache data/dc_cache_full \
    --md docs/superpowers/specs/2026-06-07-dc-time-machine-verdict.md
```

Tests: `.venv/bin/pytest tests/dc_time_machine/ -v` (57 unit tests, all synthetic — no network).

## Bottom line

A clever, genuinely-real options structure whose backtested edge **does not survive
realistic execution** and whose "risk-free" upside rarely materializes. Recommend **not**
building the live bot on these results; if pursued, gate it behind a live transform-fill
study and a wide-spread entry filter. The harness, data layer, and verdict tooling are in
place to re-run the moment that execution question is answered.
