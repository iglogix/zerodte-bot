# 0dte_bot

Research and execution code for SPX 0DTE options strategies: two generations of a
TastyTrade-connected bot, an early-morning ES/SPX options bot, a small
touch-and-turn backtester, and the backtest and walk-forward tooling used to
decide which strategies deserve a live bot at all.

Most of the strategies tested here did not survive honest backtesting, and the
findings reports say so. Nothing in this repository claims live profitability.
The bots default to simulated mode; see [Modes](#modes) for what actually
submits orders.

## Package map

| Package | What it is |
|---|---|
| `trading_core/` | The 11 modules both bots share: LLM advisors (`anomaly_detector`, `regime_classifier`), the legacy tool loop and dual validator, market-data cache, DB models, decision logger, notifications, fill listener. |
| `agentic_ic/` | Generation 1. A 0DTE iron-condor bot: deterministic entry pipeline (stability and large-move guards, risk manager) with LLM advisors for a pre-flight anomaly check and regime judgment. The earlier design, an LLM tool loop gated by an independent second-model validator, is still in `trading_core/agent/` but is no longer on the main path. |
| `zerodte/` | Generation 2. Strategy-plugin bot: each strategy owns its entry engine (scheduled windows or event-driven ORB). Strategies: iron condor, iron fly, MEIC, ORB-EM IC, ORB-EM directional, power hour, reverse IC, open IC, open strangle. Also holds the event-driven backtest framework. See [`zerodte/README.md`](zerodte/README.md). |
| `es_8am/` | 8am ES/SPX options bot and its backtests (futures and options variants). |
| `touch_turn/` | Small touch-and-turn backtester with its own config, optimizer and charts. |
| `web/` | Local FastAPI cockpit: live open P&L by strategy, payoff graph, bot start/stop. See [`web/README.md`](web/README.md) and [`docs/pnl-cockpit.md`](docs/pnl-cockpit.md). |
| `scripts/` | Backtests, parameter sweeps, walk-forward CV runners, data backfills, analysis. |
| `config/` | One YAML per strategy/variant. |
| `docs/` | Findings reports, design notes, specs. |

`llm_client.py` exists in both bots on purpose: the `zerodte` copy adds a Claude
Agent SDK (subscription) backend, see [`docs/llm-subscription-auth.md`](docs/llm-subscription-auth.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # TastyTrade and LLM credentials; not needed for dry_run or the tests
git config core.hooksPath .githooks   # protects tests_locked/ (see Tests)
```

## Modes

| Mode | Market data | Orders |
|---|---|---|
| `dry_run` (default) | Simulated | Never submitted |
| `manual` | Live (production API) | Generated and logged, not submitted |
| `sandbox` | TastyTrade cert environment | Submitted to the cert environment (cert cannot trade SPXW 0DTE) |
| `alpaca_paper` (`zerodte` only) | Live TastyTrade data | Submitted to an Alpaca paper account |
| `live` | Live (production API) | Submitted to the production brokerage account |

Priority: CLI flag, then `BOT_MODE` env var, then `bot.mode` in the config.
`live` is real money and is never the default.

```bash
# agentic_ic (gen 1)
python -m agentic_ic.main                    # dry_run
python -m agentic_ic.main --manual --once    # one cycle, orders not submitted
python -m agentic_ic.main --sandbox
python -m agentic_ic.main --live

# zerodte (gen 2): strategy chosen by bot.strategy in the config
python -m zerodte.main --config config/config_orb_em_ic_spx.yaml           # mode: manual (set in that config)
python -m zerodte.main --manual --config config/config_meic_spx.yaml
python -m zerodte.main --sandbox --config config/config_iron_fly_spx.yaml
python -m zerodte.main --alpaca  --config config/config_orb_dir_breakout_alpaca.yaml
python -m zerodte.main --live    --config config/config_power_hour_live.yaml

# es_8am
python -m es_8am.main --manual
python -m es_8am.main_options --manual --config config/config_8am_es_options.yaml

# touch_turn backtest
python -m touch_turn.run_backtest --config touch_turn/config.yaml
```

The systemd units in `deploy/systemd/` run one strategy in `manual` and one in
`alpaca_paper`; neither submits to a production brokerage account.

## Tests

```bash
python -m pytest -q          # tests_locked/ tests/ tests_web/ tests_v2/
```

The suite is hermetic: no credentials, network or local data are required.
Tests that need cached market data skip when it is absent. CI runs the full
suite on every push and pull request
(`.github/workflows/tests.yml`).

**`tests_locked/`** holds 279 tests written before the generation-1 implementation
and treated as immutable. `.githooks/pre-commit` rejects any staged change under
that directory. New tests go in `tests/` (shared and backtest code),
`tests_v2/` (`zerodte`) and `tests_web/` (`web`). `tests/test_layout.py` checks that
every `python -m ...` command in these docs, `scripts/*.sh` and `deploy/`
resolves to a real module.

## Findings

Backtests mostly use SPX 0DTE CBBO data; each report states its own
instrument and date range, and uses walk-forward cross-validation where it says so. The reports carry the numbers,
assumptions and caveats; this table carries only each report's own verdict.
Verdicts are about a strategy as specified by its source, under the stated
fill model, not a statement about options in general.

| Strategy / question | Verdict | Report |
|---|---|---|
| Burrito butterfly | REFUTED | [burrito_butterfly_findings](docs/burrito_butterfly_findings.md) |
| Hedge Hog (as published) | REFUTED | [hedgehog_backtest_findings](docs/hedgehog_backtest_findings.md) |
| Iron Duck (as published) | REFUTED; quarterly re-optimization not validated | [iron-duck-backtest-findings](docs/iron-duck-backtest-findings.md) |
| Jade Lizard (9:30 entry) | No edge as specified | [spx_0dte_jade_lizard_findings](docs/spx_0dte_jade_lizard_findings.md) |
| Expected-move fade | Performance claims REFUTED; sign of the structure unresolved | [spx_0dte_em_fade_findings](docs/spx_0dte_em_fade_findings.md) |
| GEX dealer-gamma regime filter | REFUTED | [spx_0dte_gex_regime_findings](docs/spx_0dte_gex_regime_findings.md) |
| ORB EM iron condor | REFUTED, do not deploy | [wf_cv_findings_orb_em_ic_spx_2026-06-16](docs/wf_cv_findings_orb_em_ic_spx_2026-06-16.md) |
| ORB EM directional | Named variant REFUTED, do not deploy; one breakout-retest variant reported to survive with caveats | [wf_cv_findings_orb_em_directional_spx_2026-06-16](docs/wf_cv_findings_orb_em_directional_spx_2026-06-16.md) |
| Volume-profile ORB | REFUTED | [wf_cv_findings_volume_profile_orb_2026-06-17](docs/wf_cv_findings_volume_profile_orb_2026-06-17.md) |
| Reverse iron condor | As published: breakeven over the full period. A VIX-floored scale-out variant is reported as a real edge that survives worst-case execution pricing, but the walk-forward section finds the profit concentrated in 2022-23 and says recent out-of-sample evidence does not support sizing up | [RIC-Backtest-Findings](docs/RIC-Backtest-Findings.md) |
| OTM butterfly | Published recipes show no edge; a fade-the-morning-move variant is net-positive but marginal | [spx_0dte_butterfly_otm_findings](docs/spx_0dte_butterfly_otm_findings.md) |
| Open strangle vs open IC | IC holds | [spx_0dte_open_strangle_findings](docs/spx_0dte_open_strangle_findings.md) |
| Open iron condor | Analytical study; read the exit-model caveats before relying on it | [spx_0dte_open_ic_findings](docs/spx_0dte_open_ic_findings.md) |

The earlier ORB-EM walk-forward report
([2026-04-13](docs/wf_cv_findings_orb_em_ic_spx_2026-04-13.md), 125 days) is
superseded by the full-span re-run above. The
`docs/*findings*.md` reports are research logs written as the work happened;
they include backtest figures and are not edited for presentation.

## Process notes

- `zerodte/backtest/orchestrator.py` runs the real `OrbEngine` and
  `PositionMonitor` rather than a parallel reimplementation; other backtest
  scripts are standalone analyses and are not checked against the live path.
- A reported number is expected to reproduce from the committed scripts, and
  findings that later turned out to be artifacts are corrected in place in the
  report rather than deleted.
- Cached market data, results and logs are not in the repository.

## License

MIT
