# src_v2 — Strategy-Based 0DTE Trading Bot

Strategy-agnostic bot where **each strategy owns its entry engine**. All strategy-specific logic (strike selection, entry timing, TP/SL rules) lives in pluggable strategy classes. The bot just calls `strategy.create_engine(deps)` and runs whatever it gets.

## Quick Start

```bash
# Iron Condor — scheduled entries (default)
python -m src_v2.main

# ORB EM Iron Condor — event-driven ORB breakout + midline retest
python -m src_v2.main --config config/config_orb_em_ic_spx.yaml

# Iron Fly
python -m src_v2.main --config config/config_iron_fly_spx.yaml

# MEIC — multiple-entry iron condor (tranche windows 11:00–15:00 ET)
python -m src_v2.main --config config/config_meic_spx.yaml

# Manual mode (live data, orders NOT submitted)
python -m src_v2.main --manual --config config/config_orb_em_ic_spx.yaml

# One cycle and exit
python -m src_v2.main --once --config config/config_iron_fly_spx.yaml

# Tests
python -m pytest tests_v2/ -v
```

## Strategy Selection

Set in config YAML:

```yaml
bot:
  strategy: "iron_condor"   # or "iron_fly", "orb_em_ic", or "meic"
```

## Architecture

The strategy decides how to enter. The bot doesn't know or care.

```
BotOrchestrator
  └── strategy.create_engine(deps)  →  engine.run()
        │                                    │
        ├── IronCondorStrategy  →  ScheduledEngine  (time-window loop)
        ├── IronFlyStrategy     →  ScheduledEngine  (time-window loop)
        ├── OrbEmIronCondorStrategy → OrbEngine   (3-phase ORB loop)
        └── MeicStrategy        →  ScheduledEngine  (tranche-window loop)
```

```
src_v2/
├── main.py                          # BotOrchestrator — wires strategy + services
├── strategies/
│   ├── base.py                      # LiveStrategy ABC + dataclasses
│   ├── iron_condor.py               # IronCondorStrategy
│   ├── iron_fly.py                  # IronFlyStrategy
│   ├── orb_em_ic.py                 # OrbEmIronCondorStrategy
│   ├── meic.py                      # MeicStrategy (multiple-entry iron condor)
│   └── utils.py                     # Shared helpers (mid_price, derive_spot_from_chain)
├── agent/
│   ├── base_engine.py               # BaseEngine ABC + EngineDeps dataclass
│   ├── scheduled_engine.py          # ScheduledEngine — time-window entries
│   ├── orb_engine.py                # OrbEngine — 3-phase ORB entries
│   ├── decision_engine.py           # CycleResult + backward-compat DecisionEngine alias
│   ├── tool_handlers.py             # Market data + ORB tools + order routing
│   ├── anomaly_detector.py          # LLM pre-flight check
│   └── regime_classifier.py         # LLM market regime judgment
├── services/
│   ├── position_monitor.py          # Exit logic (delegates to strategy)
│   ├── risk_manager.py              # Generic risk checks only
│   ├── tastytrade_client.py         # TastyTrade REST + WebSocket
│   ├── order_fill_listener.py       # WebSocket fill events
│   ├── market_data.py               # SQLite-cached market data
│   └── llm_client.py                # LiteLLM wrapper
└── models/
    └── database.py                  # Trade (generic) + IronCondorTrade (legacy)
```

## Engine System

The strategy owns its engine via `create_engine(deps) -> BaseEngine`:

| Engine | Entry Pattern | Used By |
|--------|--------------|---------|
| `ScheduledEngine` | Fixed time windows (09:45, 10:15, ...) | IronCondor, IronFly, MEIC |
| `OrbEngine` | Event-driven: ORB setup → breakout → midline retest | OrbEmIC |

Both implement `BaseEngine.run()`. The orchestrator calls `engine.run()` without knowing which type it is.

### ScheduledEngine Flow
1. Fetch candles, IV (generic guards)
2. Large-move guard, stability check
3. Anomaly detector, regime classifier (LLM)
4. Risk check
5. `strategy.search_candidates(chain, spot)` → validate → place

### OrbEngine Flow
- **Phase A** (08:15 ET): Fetch ORB candle → compute high/low/midline
- **Phase B** (09:30+ ET): Poll ~1s for breakout → AI preflight (once) → poll for midline retest → enter
- **Phase C**: Hand off to PositionMonitor for TP/SL/EOD

## LiveStrategy ABC

Every strategy implements these methods:

| Method | Phase | Purpose |
|--------|-------|---------|
| `configure(config)` | Init | Read strategy params from YAML |
| `create_engine(deps)` | Init | Return the entry engine this strategy needs |
| `search_candidates(chain, spot)` | Entry | Find trade setups in option chain |
| `validate(setup, chain)` | Entry | Pre-flight checks on a setup |
| `build_order(setup)` | Entry | Build API order payload |
| `compute_stop_loss(setup)` | Risk | Compute stop levels |
| `build_stop_orders(setup, stop_info)` | Risk | Build stop order payloads |
| `check_exit(trade, quotes)` | Exit | TP / SL / hold decision |
| `compute_trailing_stop(trade, quotes)` | Exit | Trailing stop update |
| `leg_pairs()` | Exit | Short/long leg mapping |

## Strategy Comparison

| Aspect | IronCondor | IronFly | OrbEmIC | MEIC |
|--------|-----------|---------|---------|------|
| Engine | Scheduled | Scheduled | OrbEngine | Scheduled |
| Entry signal | Time window + stability | Time window + EM | ORB breakout + midline retest | Tranche windows 11:00–15:00 ET (every 15–20 min) |
| Short strikes | Delta-based (balanced) | ATM (same strike) | EM-scaled (bias-adaptive) | Credit-band (AM/PM premium targets) |
| Wings | Fixed width (30pt) | EM x 0.5 | With-trend: 5pt, Against: 10pt | 80–100pt |
| TP | Per-leg at $0.05 | Fixed $ per contract | 50% of credit per spread | AM 55/65/75/85%, PM 65/85% scale-out |
| SL | Credit-based OCO | Wing breach | Breach (hold puts optional) | Stop-market on shorts at % of total credit |
| Backtest | — | — | 93.1% WR, Sharpe 14.44 | — |

## Adding a New Strategy

1. Create `src_v2/strategies/my_strategy.py`
2. Subclass `LiveStrategy`, implement all 10 abstract methods including `create_engine()`
3. Add routing in `BotOrchestrator._create_strategy()`:
   ```python
   elif strategy_name == "my_strategy":
       from src_v2.strategies.my_strategy import MyStrategy
       strategy = MyStrategy()
   ```
4. Create config YAML with `bot.strategy: "my_strategy"`
5. Write tests in `tests_v2/test_my_strategy.py`

For a new entry pattern, create a new engine class extending `BaseEngine` and return it from `create_engine()`. No changes needed in the orchestrator.

## Tests

```bash
python -m pytest tests_v2/ -v          # 117 strategy + engine tests
python -m pytest tests_locked/ -v      # locked tests (src/ unchanged)
```
