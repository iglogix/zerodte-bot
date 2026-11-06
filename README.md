# 0DTE Iron Condor Trading Bot (AI Agentic)

AI-powered trading bot for the "0DTE break-even iron condor" strategy on SPX.
The LLM reads `bot_prompt.md` as its strategy specification and controls all execution via tool calls.
A second LLM independently validates every decision before any order is submitted.

## Architecture

- **AI Agentic** — LLM tool loop: reads market data → decides → calls tools → repeats (max 10 iterations)
- **Dual-LLM Validation** — Primary (`claude-sonnet-4.6`) + Validator (`gpt-5.2`) must independently agree on entry, timing, order sequence, and stop-loss before execution
- **4 Operating Modes** — `dry_run` → `manual` → `sandbox` → `live` (see below)
- **Market Data Caching** — SQLite-backed cache; 60s TTL for quotes/chains, permanent storage for 5-min candles
- **TDD** — 128 immutable tests in `tests_locked/` written before implementation

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure credentials
cp .env.example .env
# Edit .env with your Anthropic, OpenAI, and TastyTrade credentials

# 3. Set up git hooks (protects tests_locked/ from modification)
git config core.hooksPath .githooks

# 4. Run tests
pytest tests_locked/ -v

# 5. Run the bot
python -m agentic_ic.main                  # dry-run (default, no API calls)
python -m agentic_ic.main --manual         # live market data, orders NOT submitted
python -m agentic_ic.main --sandbox        # TastyTrade cert environment
python -m agentic_ic.main --live           # production (real money)
python -m agentic_ic.main --manual --once  # single cycle and exit
```

## Operating Modes

| Mode | Market Data | Orders | Use Case |
|------|-------------|--------|----------|
| `dry_run` | Simulated | Never submitted | Development / unit testing |
| `manual` | Live (production API) | Generated but NOT submitted | Live testing / human review |
| `sandbox` | Live (cert API) | Submitted to TastyTrade cert | Integration testing |
| `live` | Live (production API) | Submitted to TastyTrade production | Real trading |

**Mode priority:** CLI flag > `BOT_MODE` env var > `config/config.yaml` `bot.mode`

## Project Structure

```
0dte_bot/
├── bot_prompt.md               # Strategy specification — what the LLM reads
├── config/
│   └── config.yaml             # Bot configuration (mode, LLM, risk, schedule, etc.)
├── .env.example                # Credentials template — copy to .env
├── agentic_ic/
│   ├── main.py                 # Entry point, BotOrchestrator, CLI
│   ├── agent/
│   │   └── tool_handlers.py    # Tool implementations (mode-aware)
│   └── services/
│       ├── tastytrade_client.py# TastyTrade REST API client
│       ├── risk_manager.py     # Pre-trade risk validation
│       └── llm_client.py       # LiteLLM wrapper
├── trading_core/               # Modules shared by agentic_ic and zerodte
│   ├── agent/
│   │   ├── trading_agent.py    # LLM tool loop
│   │   ├── dual_validator.py   # Dual-LLM agreement check
│   │   └── tool_definitions.py # Tool schemas passed to LLM
│   ├── models/
│   │   └── database.py         # SQLAlchemy ORM models
│   └── services/
│       ├── market_data.py      # SQLite-cached market data layer
│       ├── db_service.py       # Async database service
│       └── notifications.py    # Email / webhook alerts
├── tests_locked/               # IMMUTABLE — 128 tests, protected by pre-commit hook
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_cache.py
│   ├── test_tools.py
│   ├── test_validation.py
│   ├── test_agent.py
│   └── test_risk.py
└── tests/                      # Mutable tests
```

## Strategy Summary (`bot_prompt.md`)

- **Structure**: Iron condor on SPX 0DTE options
- **Delta target**: 10–15 delta short strikes
- **Premium target**: $100–$200/side ($200–$400 total), equal on both sides
- **Wing width**: 30 points default (25–40 to equalise premium)
- **Entry**: ~hourly (9:45, 10:15, 11:15, 12:15, 13:15, 14:15, 15:15 ET), after 2–3 stable 5-min candles
- **Order sequence**: Call spread first, then put spread immediately
- **Stop-loss**: On short legs only; amount = total IC premium; OCO structure (stop-limit 40pt gap + stop-market 30pt backup)
- **Take-profit**: Close short at $0.05
- **Trailing stops**: Tighten as long legs decay to preserve break-even

## Dual-LLM Validation

Both LLMs are called **in parallel** with the same market data and prompt (virgin eyes — neither sees the other's output). They must agree on:

1. Entry decision (`enter_trade` vs `wait`)
2. Timing
3. Order sequence
4. Stop-loss amount
5. All four strike prices

**Any disagreement or LLM failure → execution blocked + user notified**

## Risk Management

- Max **10 concurrent** iron condors
- Max **1–2% daily risk** (worst-case: every open trade hits double stop-loss simultaneously)
- Max **50% buying power** allocation
- Stop-loss per side = total IC premium collected × 100
- Take-profit at **$0.05** per short option

## Configuration

All active config keys in `config/config.yaml`:

| Section | Controls |
|---------|---------|
| `bot.mode` | Operating mode (`dry_run` / `manual` / `sandbox` / `live`) |
| `llm.*` | Model names, timeout, max tool loop iterations |
| `entry.schedule` | Entry times (ET) |
| `risk.*` | Concurrent trade limit, daily risk %, buying power %, take-profit |
| `stop_loss.*` | OCO stop-limit gap (40pt) and stop-market gap (30pt) |
| `cache.*` | TTL seconds for option chain, quotes, account info |
| `notifications.*` | Email and webhook settings |
| `database.*` | SQLite DB paths |
| `logging.*` | Log level and directory |

## Development

```bash
# Run immutable test suite
pytest tests_locked/ -v

# Run with coverage
pytest tests_locked/ -v --cov=agentic_ic --cov-report=html

# Format and lint
black agentic_ic/ tests/
ruff check agentic_ic/ tests/
```

## License

MIT
