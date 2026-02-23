# 0DTE Iron Condor Trading Bot (AI Agentic)

AI-powered trading bot for the "0DTE break-even iron condor" strategy on SPX, using dual-LLM validation for safety.

## Architecture

- **AI Agentic**: LLM interprets strategy from `bot_prompt.md` and controls execution via tools
- **Dual-LLM Validation**: Primary (Claude Sonnet 4.5) + Validator (GPT-4.1) must agree on ALL aspects
- **TDD**: Tests written first, locked in `tests_locked/` after approval
- **Market Data Caching**: All data persisted for rate limiting + backtesting

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Setup git hooks (protects tests_locked/)
./setup_hooks.sh

# Run tests
pytest tests_locked/ -v --cov=src

# Run bot (dry-run mode)
python -m src.main

# Run bot (live mode - use with caution!)
python -m src.main --live
```

## Project Structure

```
0dte_bot/
├── bot_prompt.md           # Strategy specification (THE STRATEGY)
├── config/
│   └── config.yaml         # Bot configuration
├── src/
│   ├── agent/              # AI agent components
│   │   ├── trading_agent.py
│   │   ├── dual_validator.py
│   │   ├── tool_definitions.py
│   │   └── tool_handlers.py
│   ├── models/             # Database models
│   │   ├── database.py
│   │   └── cache.py
│   ├── services/           # External services
│   │   ├── market_data.py
│   │   ├── order_executor.py
│   │   └── risk_manager.py
│   └── main.py
├── tests_locked/           # IMMUTABLE test suite
│   ├── test_models.py
│   ├── test_cache.py
│   ├── test_tools.py
│   ├── test_validation.py
│   ├── test_agent.py
│   └── test_risk.py
└── tests/                  # Mutable tests
```

## Dual-LLM Validation

Both LLMs must agree on ALL aspects before execution:
- Entry decision (enter or wait)
- Trade timing
- Order sequence (call spread first, then put)
- Stop-loss placement

**Any disagreement → Block execution + Notify user**

## Risk Management

- Max 10 concurrent iron condors
- Max 1-2% daily risk (worst-case double stop-loss)
- Max 50% buying power allocation
- Stop-loss = total IC premium collected
- Take-profit at $0.05

## Configuration

Edit `config/config.yaml` for:
- LLM model selection
- Risk parameters
- Entry schedule
- Notification settings

## Development

```bash
# Run tests with coverage
pytest tests_locked/ -v --cov=src --cov-report=html

# Format code
black src/ tests/
ruff check src/ tests/
```

## License

MIT
