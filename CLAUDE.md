# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (includes dev tools)
pip install -e ".[dev]"

# Run the bot
python -m agentic_ic.main                  # dry_run (default, fully simulated)
python -m agentic_ic.main --manual         # live market data, orders NOT submitted
python -m agentic_ic.main --sandbox        # TastyTrade cert environment
python -m agentic_ic.main --live           # production (real money)
python -m agentic_ic.main --manual --once  # one cycle and exit

# Tests
pytest tests_locked/ -v                         # immutable suite only
pytest tests/ -v                                # mutable tests
pytest tests_locked/ -v --cov=agentic_ic --cov-report=html  # with coverage

# Lint / format
black agentic_ic/ tests/
ruff check agentic_ic/ tests/
```

Set up the git hook that protects `tests_locked/` from modification:
```bash
git config core.hooksPath .githooks
```

## Architecture

### Entry Flow

The bot runs on a scheduled entry loop (`BotOrchestrator.run_scheduled`). At each scheduled time (default: 09:45, 10:15, 11:15–15:15 ET), `DecisionEngine.run_cycle()` executes a deterministic pipeline:

1. **Candle fetch** — requires ≥3 five-minute candles
2. **ATM IV / expected daily move** — used to set volatility-relative thresholds
3. **Large-move guard** — blocks entry if last candle > `single_candle_fraction` × expected move
4. **Stability check** — blocks if 3-candle range > `stability_fraction` × expected move
5. **AnomalyDetector** (LLM, haiku) — pre-flight sanity check; returns `normal` or `anomaly`
6. **RegimeClassifier** (LLM, haiku) — returns `enter` / `skip_trade` / `skip_session`; `skip_session` suppresses all remaining entry windows for that day
7. **RiskManager** — enforces concurrent trade / buying power / daily risk limits
8. **`get_premium_balance_candidates`** — finds balanced call/put spread pairs at target delta
9. **`validate_ic_params`** — checks delta, credit, skew constraints
10. **`place_iron_condor`** — submits orders via `ToolHandlers`

`TradingAgent` (the old LLM tool-loop design) still exists in `trading_core/agent/trading_agent.py` but is **not used** by `BotOrchestrator`. The current engine is `DecisionEngine`.

### Concurrently Running Tasks

Three async tasks run in parallel after initialization:
- **Scheduler** — wakes at each scheduled entry window to call `DecisionEngine.run_cycle()`
- **PositionMonitor** — polls open positions every 30s, manages take-profit and EOD close
- **OrderFillListener** — listens to TastyTrade WebSocket for fill events

### Mode System

Mode controls whether market data is live/simulated and whether orders are actually submitted:

| Mode | Market Data | Orders |
|------|-------------|--------|
| `dry_run` | Simulated | Never submitted |
| `manual` | Live (prod API) | Generated but NOT submitted |
| `sandbox` | Live (cert API) | Submitted to TastyTrade cert |
| `live` | Live (prod API) | Submitted to TastyTrade prod |

Mode priority: CLI flag > `BOT_MODE` env var > `config/config.yaml` `bot.mode`

### Key Modules

| Path | Purpose |
|------|---------|
| `agentic_ic/main.py` | `BotOrchestrator`, CLI, entry point |
| `agentic_ic/agent/decision_engine.py` | Deterministic entry pipeline (current engine) |
| `trading_core/agent/anomaly_detector.py` | LLM pre-flight check |
| `trading_core/agent/regime_classifier.py` | LLM market regime judgment |
| `agentic_ic/agent/tool_handlers.py` | All tool implementations, mode-aware |
| `trading_core/agent/tool_definitions.py` | Tool schemas for LLM calls |
| `trading_core/agent/trading_agent.py` | Legacy LLM tool loop (not used in main flow) |
| `trading_core/agent/dual_validator.py` | Two-LLM approval gate (legacy, used by TradingAgent) |
| `agentic_ic/agent/eod_debrief.py` | EOD summary via LLM |
| `trading_core/services/market_data.py` | SQLite-cached market data layer |
| `agentic_ic/services/tastytrade_client.py` | TastyTrade REST + WebSocket client |
| `agentic_ic/services/risk_manager.py` | Pre-trade risk validation |
| `agentic_ic/services/position_monitor.py` | Open position lifecycle management |
| `trading_core/services/order_fill_listener.py` | WebSocket fill events → queue |
| `agentic_ic/services/llm_client.py` | LiteLLM wrapper used by AI advisors |
| `trading_core/services/decision_logger.py` | Per-cycle JSON files in `logs/decisions/YYYY-MM-DD/` |
| `trading_core/models/database.py` | SQLAlchemy ORM models |
| `config/config.yaml` | Primary configuration |
| `bot_prompt.md` | Strategy specification (read by LLM advisors) |

### LLM Usage

LLMs are used only as **advisors**, not as the decision-maker:
- `advisor_model` (claude-haiku-4-5 by default): `AnomalyDetector` and `RegimeClassifier`
- `debrief_model` (claude-sonnet-4-5 by default): `EodDebrief`
- `LLMClient` wraps LiteLLM; model names follow LiteLLM convention (e.g. `anthropic/claude-haiku-4-5`)

### Immutable Tests

`tests_locked/` contains 128 tests that **must not be modified**. The pre-commit hook (`.githooks/pre-commit`) enforces this. New tests go in `tests/`. The `conftest.py` at repo root patches `json.JSONEncoder` to handle `datetime` objects across all tests.

### Configuration

`config/config.yaml` is the primary config. Strategy-specific configs use the naming convention `config_8am_{symbol}_options.yaml` (e.g. `config_8am_es_options.yaml`, `config_8am_spx_options.yaml`). Optimizer and chart scripts auto-derive a results subdirectory from the config name (e.g. `results/8am_es_options/`). All strategy thresholds (delta, credit, skew) and stability parameters are configurable without code changes.
