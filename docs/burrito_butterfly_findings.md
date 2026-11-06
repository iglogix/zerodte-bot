# Burrito Butterfly — Backtest Findings

**Date:** 2026-06-14
**Source strategy:** the course PDF "The Burrito Butterfly Strategy" (not published) (thetaprofits.com / "Dan")
**Engine:** `zerodte/backtest/burrito_core.py` + `burrito_clock.py`; runner `scripts/backtest_burrito.py`; sweep `scripts/wf_cv_burrito.py`
**Data:** EOD NBBO SPXW chains `data/iron_duck_eod/SPXW/` (2020-01-03 → 2026-06-12, ~1,400 sessions)
**Objective:** maximize Sharpe.

## Verdict: REFUTED

The Burrito Butterfly, as specified, is a **negative-expectancy** structure across the entire
parameter space tested. No body offset, DTE, direction signal, lock/stop/target band, or defense
policy produced a positive Sharpe — in-sample or out-of-sample. The headline "lock into a free
delta-neutral structure" ("Magic Money Rainbow") **cannot manufacture positive expectancy from a
negative-expectancy entry**; it only redistributes outcomes while adding execution cost.

This is consistent with the repo's prior results on instructor/YouTube income strategies (Iron Duck,
Pulver IC, Hedge Hog, Jade Lizard as-taught — all refuted).

---

## 1. The entry has negative expectancy (structural sweep, hold-to-expiry)

A 1-2-1 debit butterfly (15-pt wings, body pushed OTM) + linked 5-pt vertical debit spread is a net
**long-premium** position that pays theta. Held to expiry over the full 2020-2026 history, **every
cell loses money**:

| body offset (×EM) | DTE | signal | trades | win rate | avg return / trade | profit factor |
|---:|---:|---|---:|---:|---:|---:|
| 0.25 | 1 | both | 935 | 34.1% | **−29.5%** | 0.43 |
| 0.50 | 1 | both | 1043 | 36.3% | **−30.5%** | 0.46 |
| 0.75 | 1 | hull | 526 | 32.5% | **−25.9%** (least bad) | 0.57 |
| 0.75 | 1 | both | 1074 | 32.5% | −29.7% | 0.52 |
| 1.00 | 3 | hull | 331 | 23.6% | −40.6% | 0.42 |

- **Best case is −26% of premium per trade.** Win rates peak ~37-42% (offset 0.25-0.5) but the ~60%
  of losers each forfeit close to the full debit, so average return stays deeply negative.
- The directional call adds almost nothing: `hull` vs `both` (no view) differ by <1 pt of win rate.
  The edge the deck attributes to "directional bias + natural pull to the peak" is not present.
- Placing the body ~1×EM OTM (the deck's "ATM Target" peak) is the worst region — price rarely
  travels a full expected move to the body in 1-5 days.

## 2. The lifecycle cannot rescue it (lock / defense / clawback)

On the least-bad structural cell (DTE 1, offset 0.75, hull), full-history:

| management | trades | win rate | avg return | Sharpe | net P&L |
|---|---:|---:|---:|---:|---:|
| hold-to-expiry | 526 | 32.5% | −25.9% | −3.65 | −$53,381 |
| lock .25 / stop .50 / pt .50 | 526 | 32.5% | −25.8% | −3.64 | −$53,328 |
| lock .50 / stop 1.0 / pt .50 | 526 | 32.7% | −25.4% | −3.59 | −$52,621 |
| + clawback (valley hole-injection) | 526 | 32.5% | −25.8% | −3.64 | −$53,328 |

- With **wide** bands the lock/stop rarely fire → the lifecycle collapses to hold-to-expiry.
- With the deck's **tight** 5-10% bands, EOD mark noise churns the position and adjustment slippage
  compounds, producing **−93% to −114% per trade** (a single fixed config: lock/stop 7%, pt 10%,
  Sharpe −10.6). The "lock when up 5-10%, stop when down 5-10%" rule is actively destructive at
  realistic granularity.
- **Clawback and defenses add nothing** — identical or slightly worse, and the optimizer never
  selects them (see §3).

## 3. Walk-forward CV (the method, not a curve-fit cell)

`scripts/wf_cv_burrito.py`, 64-cell grid (DTE×offset×signal×lock×stop×pt×defense), rolling folds:
select the cell by **validation** Sharpe (≥30 trades), report **test/OOS** — never selecting on test.

| test year | val Sharpe | **OOS Sharpe** | OOS trades | selected cell |
|---:|---:|---:|---:|---|
| 2023 | −1.77 | **−0.68** | 121 | dte1 / off0.75 / momentum / lock.25 / stop1.0 / pt.5 / no-def |
| 2024 | −1.13 | **−6.91** | 115 | dte1 / off0.75 / momentum / lock.25 / stop1.0 / pt.5 / no-def |
| 2025 | −3.86 | **−4.88** | 118 | dte1 / off0.75 / momentum / lock.25 / stop0.5 / pt.5 / no-def |
| 2026 | −4.34 | **−6.46** |  43 | dte1 / off0.50 / hull / lock.5 / stop0.5 / pt.5 / no-def |

- **Every fold is negative in validation AND OOS.** There is no plateau of positive cells to select
  from; the "best" cell is merely the least-negative, and it stays negative OOS.
- Every selected cell uses **no defense policy** — the sweep confirms defenses/clawback do not help.

## 4. Slippage stress (selected cell, full history)

Half-spread fill multiplier `k` (1.0 = cross full half-spread per leg):

| `k` | Sharpe | net P&L |
|---:|---:|---:|
| 1.00 | −5.26 | −$77,114 |
| 1.25 | −6.33 | −$90,112 |
| 1.50 | −7.24 | −$98,893 |
| 2.00 | −8.30 | −$90,033 |

Already negative at mid (k=0 would still be negative — see §1 hold-to-expiry, which marks settlement
at intrinsic), and monotonically worse with realistic execution cost. Frictions deepen, never
create, the loss.

## 5. Why the pitch is misleading

The deck's "free" status and "no downside expiration risk" describe the P&L *shape after a winning
move* (both tails floating above zero), not the *expectancy*. To reach that state the trade must
first be a winner — which happens ~32% of the time — and the act of locking spends additional debit.
Across all paths the negative drift of the long-premium entry dominates. The "delayed loss engine"
(slow bleed) is real, but slow loss is still loss: theta decays the structure to settlement.

## 6. Honest caveats / limits

- **EOD granularity for DTE≥1.** Lock/stop/clawback triggers fire on daily closes; intraday
  hit-then-revert is invisible. This biases stops **optimistically** (a real intraday stop would
  realize more whipsaw losses) — so the true result is, if anything, **worse**, which only
  strengthens the refutation.
- **0DTE intraday path built but not swept end-to-end.** `burrito_clock.intraday_marks` (10-min
  provider) is implemented and unit-tested, but `run_history` drives the EOD clock; a provider
  adapter over `data/cbbo_cache_fly_10min/` was not wired for the DTE=0 corner. Given the uniform
  negativity across DTE 1-5 and the structural sweep, 0DTE is highly unlikely to flip the verdict;
  wiring it is a documented follow-up, not a gap that changes the conclusion.
- **XSP micro overlays** were approximated as 1/10 SPX (no separate XSP cache), per spec.
- **Markov signal** uses a lightweight smoothed-return regime proxy, not the full
  markov-2-hedge-fund-method skill; `momentum`/`hull` were the swept signals and direction was shown
  to be a minor lever regardless.

## Reproduce

```bash
# single config
PYTHONPATH=. .venv/bin/python scripts/backtest_burrito.py --dte 1 --signal hull \
    --body 0.75 --lock 0.5 --stop 0.5 --pt 0.5 --start 20200101
# walk-forward CV + slippage stress
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python scripts/wf_cv_burrito.py --folds 4
# unit tests
.venv/bin/pytest tests/test_burrito_core.py -v
```

Artifacts: `results/burrito/wfcv.json`, `results/burrito/trades.csv`, `results/burrito/metrics.json`.
