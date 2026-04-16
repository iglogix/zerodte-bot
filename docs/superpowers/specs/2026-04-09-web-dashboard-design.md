# Web Dashboard for 0DTE Bot Management

**Date:** 2026-04-09
**Stack:** FastAPI + HTMX + Jinja2 + Tailwind CSS (dark theme)
**Auth:** Bearer token (env var `BOT_WEB_TOKEN`, stored in session cookie)

## Goals

A web interface for a single user to:
- Start, stop, and manage multiple bot instances concurrently (each with its own config + mode)
- Monitor positions, P&L, and account state in real time
- Stream filtered logs (ERROR/WARNING/key INFO)
- Browse decision history
- Chat with an AI agent that can read bot state and take management actions

Must be accessible remotely.

## Architecture

### Separation of Concerns

The web server and bots run as **separate processes**. The web server:
- Manages bot lifecycles as **subprocesses** (`python -m src_v2.main --config X --mode Y`)
- Reads shared state: SQLite trading DB + filesystem (logs, decision JSONs)
- Maintains its own **read-only** TastyTrade API connection for displaying positions and account data
- Serves the dashboard UI and handles WebSocket connections

Each bot keeps its own `PositionMonitor` for trade management (take-profit, EOD close). The web server's TastyTrade connection is purely for display — no interference.

### Real-Time Data Flow

A **single multiplexed WebSocket** connection per browser tab handles all real-time data:

| Data | Method | Frequency |
|------|--------|-----------|
| Bot status (running/stopped/error) | WebSocket push | On change |
| Positions & P&L | WebSocket push | Every 10s |
| Account stats (buying power, risk) | WebSocket push | Every 30s |
| Logs | WebSocket push | As they appear (filtered) |
| Decisions | HTMX fetch | On demand (user navigates) |
| AI Chat | WebSocket streaming | On interaction |

Message format: `{"type": "positions", "data": [...]}`, `{"type": "log", "data": {...}}`, etc.

## Pages

### Dashboard (home)

- **Stat cards row:** Running bots count, open positions count, daily P&L, buying power
- **Bot grid:** All bots (running, stopped, errored) as cards in a responsive wrapping grid. Each card shows: name, config, mode, status badge, position count, P&L
- **Recent errors feed:** Last N ERROR/WARNING lines across all bots, monospace, color-coded

### Bots

Full bot management page:
- **Start new bot:** Pick config file (dropdown of `config/*.yaml`) + mode (dry_run/manual/sandbox/live) + optional display name + "Start" button. If no name given, auto-generate from config (e.g., `iron_fly_spx` from `config_iron_fly_spx.yaml`) + short ID
- **Bot list/table:** All bots with columns: name, config, mode, status, uptime, last cycle outcome, actions (stop/restart)
- Status badges: RUNNING (green), STOPPED (gray), ERROR (red)

### Positions

- Table of all open positions across all bots
- Columns: bot name, symbol, type (IC/IF), legs, entry credit, current P&L, greeks (delta, theta), time to expiry
- Manual "Close" button per position (with confirmation)
- Auto-refreshes via WebSocket every 10s

### Decisions

- Date picker + bot filter
- Lists decision JSON files for selected date/bot
- Click to expand: shows full cycle pipeline result (candle data, IV, guards, regime, outcome, reason)
- No real-time updates needed — static files read on demand via HTMX

### Logs

- Per-bot tabs
- Live-streaming filtered log lines: ERROR, WARNING, and INFO lines that match meaningful patterns (cycle outcomes, trade fills, position events, connection state changes). Excludes DEBUG and routine polling INFO
- Color-coded by level: red for ERROR, yellow for WARNING, white for INFO
- Auto-scroll with "pin to bottom" toggle
- Search/filter within displayed logs

### AI Chat

- Chat interface with message history
- Claude agent with streaming responses via WebSocket
- Tool use is transparent: when the agent calls a tool, the UI shows what it did (e.g., "Queried positions..." → result)

## AI Chat Agent

**Model:** Claude via LiteLLM (same setup as existing advisors)

### Tools

| Tool | Type | Description |
|------|------|-------------|
| `list_bots` | read | List all bots with status, config, mode, uptime |
| `get_bot_logs` | read | Get recent filtered logs for a specific bot |
| `get_positions` | read | Get open positions with P&L, greeks |
| `get_decisions` | read | Read decision history for a date/bot |
| `get_account_info` | read | Buying power, balances, risk utilization |
| `start_bot` | action | Start a bot with a specified config + mode |
| `stop_bot` | action | Stop a running bot by name/ID |
| `restart_bot` | action | Restart a bot (stop + start) |
| `close_position` | action | Close a specific position via TastyTrade API |
| `change_mode` | action | Restart a bot with a different mode (mode is set at init, so this stops + starts with new mode) |

### Safety

Action tools require a **confirmation step**: the agent proposes the action in chat, the user confirms with "yes" before execution. Read-only tools execute immediately.

## Project Structure

```
web/
  app.py              # FastAPI app, routes, WebSocket handlers
  bot_manager.py      # Subprocess lifecycle (start/stop/list bots)
  chat_agent.py       # Claude agent with bot management tools
  log_streamer.py     # Tail + filter bot log files
  auth.py             # Token middleware
  templates/
    base.html         # Layout (sidebar, HTMX/WebSocket setup)
    dashboard.html
    bots.html
    positions.html
    decisions.html
    logs.html
    chat.html
  static/
    style.css         # Dark theme
    app.js            # WebSocket client, chat UI logic
```

## Auth

- Token set via env var `BOT_WEB_TOKEN` or in config
- Login page: single password field, submits token
- On success: token stored in HTTP-only session cookie
- All HTTP routes and WebSocket connections validate the token
- No user management, sessions, or roles — single user

## Configuration

Addition to `config/config.yaml` (or standalone `config/web.yaml`):

```yaml
web:
  port: 8080
  host: 0.0.0.0
  token: ${BOT_WEB_TOKEN}
  tastytrade_poll_interval: 10
  log_filter_levels: [ERROR, WARNING, INFO]
```

## Running

```bash
python -m web.app                    # default port 8080
python -m web.app --port 9000       # custom port
```

The web server starts independently. Bots are started/stopped from the UI or AI chat.

## Dependencies

New packages:
- `fastapi` + `uvicorn` — web server
- `jinja2` — templates
- `python-multipart` — form handling
- `htmx` — CDN, no install needed
- `tailwindcss` — CDN or standalone CLI for dev

No new LLM dependencies — reuses existing `LLMClient` and LiteLLM setup.
