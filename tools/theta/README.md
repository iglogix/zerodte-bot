# MEIC independent cross-check & strategy-realism toolkit

Independent validation of the SPX 0DTE **MEIC** backtest (`results/meic_spx/trades.csv`,
22,166 spreads, 2022-05 → 2026-02) plus the exit / regime / sizing research that grew
out of it. Everything here is read-only analysis driven off the project's own data
(`data/cbbo_cache_meic/*.pkl`, `data/vix9d_daily.csv`) and, optionally, an independent
ThetaData feed. The one production change this work produced is the **dislocation entry
guard** in `zerodte/strategies/meic.py` (+ `config_meic_spx.yaml`).

## TL;DR — what we learned

1. **The engine is faithful.** A from-scratch re-simulation reproduces the recorded
   backtest **to the penny on all 22,166 spreads** (P&L $507,885.50 == $507,885.50,
   100% exit-path match). No look-ahead, no accounting bug.
2. **The data is OPRA-real.** ThetaData NBBO == the Databento cache to the penny (both
   are OPRA — there is no truly independent SPX 0DTE tape). The 2025-04-09 13:20
   `0.05/45.00` tick is a *real* dislocated OPRA quote, not a bug → it's the **mid-fill
   assumption** that's wrong there, not the data. Hence the dislocation guard.
3. **The +$507k profit was an execution-fill artifact.** Under realistic stop fills the
   strategy loses **−$2.9M (combo)** to **−$9.4M (short_only, its actual spec)**. The
   net P&L is a razor-thin diff of two ~$4M flows, so the stop fill is load-bearing.
4. **The real edge is the *opposite* of the design:** these are defined-risk condors, so
   the stop *adds* risk. Removing the stop **and** the scale-outs and holding to cash
   settlement = **+$1.45M** (mid−0.10) / **+$2.05M** (realistic fills).
5. **VIX-regime sizing is overfit** — great in-sample (+$1.49M favor-mid), but **walk-forward
   CV rejects it**: OOS it *loses* to plain flat hold-to-expiry (regime 2.62 Sharpe / 3-of-4
   folds vs flat **2.71 / 4-of-4 / +$967k OOS**). Simplicity wins.
6. **Sizing is the binding constraint, not edge.** Defined-risk + fixed-fractional sizing
   makes ruin *impossible* (worst day ≈ −1.06·f). For ~20% drawdown: **f ≈ 3–5%**, full
   13-tranche structure needs **~$150–250k**; smaller accounts are granularity-constrained
   to fewer tranches. In-sample CAGR is high (~45–83%) but regime-dependent — anchor
   forward expectations on the WF-CV, not the headline.

**Bottom line:** trustworthy strategy = *plain hold-to-expiry MEIC, no stop, no scale-out*,
sized by fixed-fractional defined-risk at f≈3–5%. Execution-robust; the constraints are
capital and drawdown discipline.

## Scripts

Run with the project venv from repo root, e.g. `.venv/bin/python tools/theta/<script>.py`.
Quote-sourced scripts take `--source databento` (default) or `--source theta`.

| Script | Purpose |
|---|---|
| `reconcile_meic.py` | Entry-credit + accounting reconciliation vs recorded results (Databento\|Theta adapters). |
| `resim_meic.py` | Layer-C penny-perfect re-simulation of the engine (trigger stop). |
| `scan_cache_quality.py` | Scan the cache at entry minutes for garbage/locked/crossed ticks (found 2/22,166). |
| `compare_one_day.py` | Theta vs Databento quote diff for one day (cross-vendor sanity). |
| `resim_exits.py` | Alternative exits: `combo` / `short_only` stops, and hold-to-expiry (±scale-out). |
| `resim_exit_filter.py` | Intraday move-so-far entry filter (skip if move > k·EM); tail-vs-edge sweep. |
| `resim_regime.py` | VIX9D-regime sizing/strike variants (favor-mid, de-risk-hi, + move filter). |
| `wf_cv_regime.py` | Walk-forward CV (ratio 5:1, 4 folds, annualized Sharpe) — the overfitting gate. |
| `sizing_montecarlo.py` | Fixed-fractional defined-risk sizing; block-bootstrap maxDD vs CAGR per f. |
| `sizing_policy.py` | Per-account policy for a ~20% DD target (mid−0.10 baseline fills). |
| `entry_fill_sensitivity.py` | Entry-fill brackets: M1 both-mid / M2 shorts-mid+wings-ask / M3 no-improvement. |
| `sizing_policy_realistic.py` | Final policy on M2 realistic fills + missed-fill sensitivity. |
| `wait_upgrade.sh` | Polls the Theta Terminal until the option entitlement upgrades. |

Runtime artifacts (`ThetaTerminal.jar`, `terminal.log`, `*.out`) are git-ignored.

## Using the ThetaData path

ThetaData needs the local **Theta Terminal** (Java) authenticated with `THETADATA_EMAIL`/
`THETADATA_PASSWORD` from `.env`, serving REST on `127.0.0.1:25510`. Intraday 1-min NBBO
requires the **VALUE tier ($40/mo)**; the FREE tier is EOD-only. Launch:

```bash
java -jar tools/theta/ThetaTerminal.jar "$THETADATA_EMAIL" "$THETADATA_PASSWORD"
```

Then any script with `--source theta`. Note Theta and the Databento cache are byte-identical
(both OPRA), so Theta is a cross-vendor sanity check, not an independent edge confirmation.

## Key caveats

- **No compounding in the raw backtest** — fixed `tranche_sizing` (qty 3–6 by time-of-day).
  Headline P&L is an arithmetic sum; "CAGR on $100k" is meaningless without the BP base.
- **In-sample / regime-dependent.** 2022 + 2025 (high vol) carried it; 2023 (calm) lost under
  most variants. The WF-CV is the honest forward guide; the in-sample CAGR is a ceiling.
- **The tail is structural.** Short-gamma 0DTE has an irreducible surprise-gap tail (e.g.
  the 4/9 tariff move happened *after* entry). Fixed-fractional sizing bounds it; nothing
  removes it.
