I have everything needed. All four candidate gate designs have been adversarially rejected on both the hold-to-expiry and managed logs. Here is the scope document.

---

# SCOPE — Net-Delta / Trend Entry Gate for MEIC

**Status: NO-GO (build the WF harness, ship the gate default-OFF — do not enable it).**
Read-only investigation; nothing edited. Date: 2026-06-15.

---

## 1. Feasibility verdict

**The empirical signal does NOT justify building this as a P&L improver. Recommend AGAINST shipping any of the four gates enabled.** All four candidate designs were reconstructed under their own exact proxies and adversarially tested on the real combo logs. Every one is value-destructive on both the hold-to-expiry log and the managed (stop_pct) log — the latter being the log each design named as its rescue hypothesis.

The decisive numbers, on the baseline log `results/meic_wf_method/combo_eh_s1.0_w0.5_p0.7.csv` (11,956 tranches / 1,022 sessions, 2022-05→2026-06, baseline +1,400,362):

- **Net-skew / net-delta gate (`accum_skew_veto`)** — session-scope counterfactual loses at *every* threshold: X=0.20 → −31%, X=0.30 → −17%, X=0.50 → −7.5%. Skipped tranches are **77–79% winners** (mean +95 USD). Under the design's own EM-normalized qty-weighted proxy it is worse than the BS-delta version (fires more often). Verdict: **reject**.
- **Trend-run displacement gate (`trend_run_tranche_veto`)** — the one "+5%" cell (pts=40, idx≥4) is **5 catastrophic trades, not a signal**: blocked tranches have a *higher* median P&L (+692.8 vs +455 kept) and are 73% winners; the −98 mean is produced entirely by 5 trades summing −90,416. Excluding those 5, the cell destroys +20,512. Per-year sign flips violently (2023 −11k, 2025 −34k); bootstrap 95% CI on the cell mean is [−329, +128], P(mean<0)=0.80. In a 5-fold chronological WF the two earliest OOS folds lose money. Verdict: **reject**.
- **Trend-confirmed lean conjunction (`trend_confirmed_lean_throttle`)** — swept lean×trend grid; gate skips **72–87% winners**; delta_total −32,050 to +7,460 with positive cells firing on n=2–20 tranches (noise). Catches only 0.1–1.0% of the catastrophic tail it targets. Re-tested on the **managed** logs (`combo_em` p0.65 baseline 920,020; p0.75 baseline 325,394): **every** parameter cell reduces total P&L (−904 to −49,193). The "real upside under stop management" escape hatch is **empirically falsified**. Verdict: **reject**.
- **Protection-consumption gate (PCG)** — its "rare, tail-only" premise is false: protection-consumption fires in **97.5% of sessions** (mean fire at tranche index 7.0). Vetoed legs are **~88% winners** (call +50, put +53 mean). Mildest config forgoes +395,698; aggressive >1,000,000. Also fails live/backtest parity (longs not stored anywhere live). Verdict: **reject** (the only design that also fails the mirror-able axis).

**Root cause (consistent across all four):** the tranches these gates block are late-day, thinner-premium, but **still 73–88% winners**. The catastrophic tail (worst-5%, threshold ≈ −3,782 USD) is driven by *large whole-condor underlying moves*, and those blow-ups have **lower** mean |net lean| (0.121 vs 0.161) and **earlier** tranche index — they are mid-session full-condor losses, not late high-lean adds. The "net delta leans into a trend" mechanism the gate targets does not correlate with the losses it claims to prevent. The apparent Spearman(|net|, P&L) = −0.136 is a **time-of-day / premium-compression confound** (|net| ~ tranche-index Spearman ≈ 0.48–0.58), not a directional-risk edge.

---

## 2. Recommended gate (if one is built at all)

There is **no strong verified design** — all four were rejected. If the user insists on building the machinery anyway (the defensible reason: to let the WF *method* formally confirm "no gate" as optimal, and to be re-runnable on future managed/scale-out logs), build **`accum_skew_veto`** as the single, default-OFF, WF-swept parameter. It is the cleanest to mirror, has the simplest causal condition, and `max_abs_net_skew ≤ 0` reproduces the as-taught no-gate baseline **byte-identically**.

**Causal condition (within-session, backward-only):**
For each already-accepted tranche *i*, signed structural skew in EM units:
```
s_i = qty_i * (2*spot_i - short_put_K_i - short_call_K_i) / em_pts_i      # >0 = bullish-tilted shorts
```
Running net before the prospective tranche: `net = Σ s_i`. Prospective contribution `s_new` from candidate strikes/spot/em **before** entry.
**Veto (skip this tranche only):**
```
abs(net) >= max_abs_net_skew  AND  sign(s_new) == sign(net)   # deepen-only
```

**Tunable knobs:** `max_abs_net_skew` (≤0 = off; grid {0.0, 0.20, 0.30, 0.50, 0.80}, mirrors empirical p95≈0.45), `skew_qty_weighted` (bool; must match live↔backtest, requires `sizing_mode=static`), `skew_deepen_only` (bool), `gate_scope` ∈ {tranche → `skip_trade`, session → `skip_session`}.

**Hard rules for parity:** compute EM only from `setup.extra['em_pts']` (the strategy's own `_compute_em`, `meic.py:163`, nearest-strike straddle, no greeks) — **never** `get_expected_move()` (its ATM pick depends on flaky DXLink delta) and **never** `opt['delta']` (absent in CBBO). Spot must be parity/index spot, never SPYx10.

---

## 3. Insertion points

**Backtest hook (confirmed, ready):** `src_v2/backtest/engine.py`, `_process_day_meic()`.
- Initialize `running_skew = 0.0` at the top of the slot loop, **`engine.py:422`** (before `for slot in schedule:` at line 423).
- Insert the veto **after `engine.py:443`** (`if not val.get("passed"): continue`) and **before `engine.py:452`** (the `for side in ("put","call")` append). On veto → `continue` (optionally append a skipped `BacktestResult` with `skip_reason='accum_skew_gate'`; the field exists at `engine.py:51`).
- On a non-vetoed tranche, after the appends, `running_skew += s_new`.
- All inputs in scope: `setup.strikes`, `setup.extra['em_pts']`, `setup.extra['qty']`, `spot` (`engine.py:427`). Within-session-backward-only → 0DTE day-independence preserved → WF runner needs **zero** changes.

**Live hook:** `src_v2/agent/scheduled_engine.py`, `_run()`. Insert a "Step 6c" **after the candidate exists** (after `top = candidates['top_candidate']` ~line 319, before validate ~line 321). Returns `CycleResult(outcome='skip_trade', reason='accum_skew_gate: ...')`, mirroring the existing `regime_classifier` skip_trade branch (lines 256–279).

**Live data-wiring prerequisite (the real cost — see §5):** the engine cannot currently see accumulated open-tranche state. Two options:
- **(A) durable:** persist MEIC manual/live entries to the trades DB like `place_power_hour_ic` (`tool_handlers.py:487+`); MEIC's `place_two_spread_ic` (`tool_handlers.py:434–451`) writes **no DB row in manual mode** today. Then read net via `db.get_open_trades()` inside `risk_manager` (already in `EngineDeps`). This *also* closes the known "MEIC manual writes no DB row" gap and gives PositionMonitor a lifecycle.
- **(B) minimal:** add an `open_tranche_registry` field to `EngineDeps` (`base_engine.py:9–20`) and pass `self._quote_comparator` in `main.py:227`; read `QuoteComparator._active`. Caveat: `register()` (`quote_comparator.py:43–57`) stores only **short** strikes, **not qty and not em_pts** — must be extended for qty-weighting. `_active` is RAM-only/EOD-cleared, so `_rehydrate()` must run before the first post-restart window. Note: in **dry_run** mode `place_two_spread_ic` returns early and never registers, so `_active` is empty and the gate is silently OFF — validating the live gate via dry_run gives a false negative.

---

## 4. WF validation plan (three-tier method, per `docs/wf-cv.md`)

The pipeline already exists and needs only new parameter axes. **Do not trust any single in-sample threshold** — validate the *method*.

1. **Add the gate as a combo axis** in `scripts/gen_meic_combo_logs.py`, `combos()` (lines 42–51): add `max_abs_net_skew ∈ {0.0, 0.20, 0.30, 0.50, 0.80}` (and optionally `gate_scope`) to the tag tuple. Set it on `cfg['strategy']` in `run_chunk` near line 92, **keeping `st['sizing_mode']='static'`** (line 93) — qty-weighting couples to equity history under compounding and breaks fold-slicing. `max_abs_net_skew=0.0` must reproduce existing combo CSVs unchanged.
2. **Simulate ONCE over full span**, day-chunked, merged by date (existing flow). Run for **both** `exit_mode='hold_to_expiry'` *and* `exit_mode='managed'` so the managed-log rescue hypothesis is tested, not assumed.
3. **Run `scripts/run_wf_method_strategy.py`** unchanged — point `--dir` at the MEIC logs, set `--baseline` to the gate-OFF tag (e.g. `eh_s1.0_w0.5_p0.7`). It already produces the three tiers from the doctrine:
   - `always` (deploy fold winner),
   - `wfe` (WFE-gated; falls back to as-taught baseline when val/train Sharpe < `--wfe-floor 0.3`, doc step 5),
   - `base` (as-taught constants, no optimization).
   Verdict comes from the **stitched UNTOUCHED test tier only** (`run_wf_method_strategy.py:92–99`), with end-anchored folds (line 65) and winner-stability reporting (lines 101–103; heavy winner fluctuation = curve-fitting).
   **Success criterion:** the gated `wfe`/`always` tiers must beat `base` on OOS *total $ AND Sharpe AND maxDD*. Expected outcome: WFE selects `max_abs_net_skew ≤ 0` (gate OFF) and winner-stability shows the baseline dominating — i.e. the machinery **confirms no edge**.
4. **Stress-fill check (the optimism guard):** the existing `stress_eh_*` / `stress_em_*` logs already exist in `results/meic_wf_method/`. Regenerate the gated combos under the stress-fill model and re-run the WF on the stress logs. A gate "win" that survives only under optimistic fills is rejected. This matters acutely here because project memory (`project_meic_reconciliation`) notes **~47% of real MEIC P&L hinges on scale-out fills not in `trades.csv`** — the hold-to-expiry log understates management effects, and the managed log's stop-fill assumption is the single largest sensitivity (`project_power_hour_backtest`: "stop-fill model is the killer sensitivity").

---

## 5. Risks / caveats / what could make this a waste of time

- **It is a waste of time as a P&L play.** The empirical case is closed on both available logs (hold-to-expiry *and* managed). Every gate blocks 73–88%-winning premium. Building it to ship enabled would knowingly degrade returns 7–31%.
- **The live-wiring prerequisite is the bulk of the cost, not the gate.** Manual-mode MEIC writes no DB row; `QuoteComparator.register()` stores only shorts and lacks qty/em. Either Option (A) (DB persistence + PositionMonitor lifecycle) or Option (B) (extend register + EngineDeps + rehydrate) is real net-new wiring touching the live order path — risky to a running bot for a gate that won't fire profitably.
- **Confound, not signal:** the only robust correlation is a time-of-day / premium-compression artifact. A clean late-entry/time gate would dominate any lean gate — and even that mainly *compresses gains* rather than avoiding losses.
- **Untested-but-plausible niche:** a *scale-out-aware* gate ("don't add into a position about to trigger stops") is the only mechanism not yet falsified at the fill-level, because trades.csv lacks scale-out fills. But the managed-log test already covered the stop-level version and it lost. Pursuing this would require building a managed log that captures intraday scale-out fills (Theta VALUE tier $40/mo per `project_meic_reconciliation`) — a separate, larger project.
- **Asymmetric-tranche transfer risk:** all refutations were measured on balanced put+call pairs. If MEIC ever adds put-only/call-only tranches the lean distribution shifts and the refutation no longer transfers — but that's a different strategy.
- **Parity landmines that void the test if ignored:** mixing parity spot (numerator) with raw SPYx10 09:30 open (denominator) injects a ~0.3% basis comparable to a trend threshold; MEIC has no 09:30 entry so session_open must come from the raw bar both sides; EM must be `setup.extra['em_pts']` not `get_expected_move()`.

---

## 6. Effort estimate + go/no-go

**Recommendation: NO-GO on shipping a gate. CONDITIONAL-GO (≈0.5 day) only on the WF harness if the user wants the doctrine to formally close the question and leave a re-runnable test for future scale-out logs.**

| Task | Effort |
|---|---|
| Backtest hook + `running_skew` accumulator + combo axis in `gen_meic_combo_logs.py` | ~3–4 h |
| Re-simulate gated combos (hold + managed + stress) and run `run_wf_method_strategy.py` | ~2 h compute + review |
| **Subtotal — confirm-no-edge harness (backtest only)** | **~0.5–0.75 day** |
| Live wiring Option (B): extend `register()`, `EngineDeps`, rehydrate, dry_run caveat | ~1 day |
| Live wiring Option (A): DB persistence for MEIC manual + PositionMonitor lifecycle + risk_manager read | ~2–3 days |

**Decision:** Build the **backtest-only WF harness (default-OFF axis)** to let the method confirm the baseline wins OOS and to leave a re-runnable artifact. **Do not build the live wiring and do not ship the gate enabled** unless a *new* managed log that captures intraday scale-out fills produces a gate cell that beats baseline on the stitched untouched test tier under the stress-fill model. On current evidence, that is not expected.

**Relevant files:** `~/Dev/0dte_bot/src_v2/backtest/engine.py` (hook 422/443/452), `~/Dev/0dte_bot/src_v2/agent/scheduled_engine.py` (live hook ~319), `~/Dev/0dte_bot/scripts/gen_meic_combo_logs.py` (combos 42–51, run_chunk 88–93), `~/Dev/0dte_bot/scripts/run_wf_method_strategy.py` (three-tier, unchanged), `~/Dev/0dte_bot/docs/wf-cv.md`, `~/Dev/0dte_bot/results/meic_wf_method/combo_eh_s1.0_w0.5_p0.7.csv` (baseline log), `combo_em_*` (managed logs), `stress_*` (stress-fill logs), `~/Dev/0dte_bot/src_v2/strategies/meic.py:163` (`_compute_em`), `~/Dev/0dte_bot/src_v2/services/quote_comparator.py:43-57` (register, shorts-only), `~/Dev/0dte_bot/src_v2/agent/tool_handlers.py:434-451` (manual no-DB-row gap) and `:487+` (Power Hour DB-persist template).

---
*Scoped via 13-agent read-only workflow (wf_e5bbe55a-dc6), 2026-06-15.*
