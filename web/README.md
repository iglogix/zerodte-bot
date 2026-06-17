# Web Dashboard

FastAPI app for monitoring and controlling the 0DTE bot: bot start/stop, live
logs, decisions, a chat agent, and the **Live P&L Cockpit**.

## Run

```bash
# from the repo root
BOT_WEB_TOKEN=<token> python -m web                  # http://0.0.0.0:8080
BOT_WEB_TOKEN=<token> python -m web --port 8085      # custom port
BOT_WEB_TOKEN=<token> python -m web --reload         # auto-reload (dev)
```

Then open the host/port and log in with `BOT_WEB_TOKEN` (the value is the login
token; the session cookie is derived from it). Entry point: [`web/__main__.py`](__main__.py).

## Live P&L Cockpit

Real-time open P&L for the bot's positions, grouped/selectable by strategy, at
`/cockpit`. Quick start with the offline session-replay harness (no market /
no positions needed):

```bash
BOT_WEB_TOKEN=<token> COCKPIT_SIM=1 python -m web --port 8085
# open http://localhost:8085/cockpit  (log in, then it replays 09:30 -> 16:00)
```

Full details — live mode, options_api wiring, env vars, replay speed, reset,
tests — are in **[docs/pnl-cockpit.md](../docs/pnl-cockpit.md)**.

## Key environment variables

| Var | Purpose |
|-----|---------|
| `BOT_WEB_TOKEN` | **required** — login token for the dashboard |
| `COCKPIT_SIM` | `1` to run the cockpit's offline session-replay harness |
| `SIM_MINUTES_PER_POLL` | cockpit replay speed (session-minutes per poll; default 6) |
| `COCKPIT_DB` | bot trading DB the cockpit reads open trades from (default `data/meic_bot.db`) |
| `COCKPIT_PNL_DB` | where the cockpit persists intraday P&L snapshots |
| `OPTIONS_API_URL` / `OPTIONS_API_TOKEN` | live quote source for the cockpit (options_api) |

## Layout

| Path | Purpose |
|------|---------|
| `app.py` | FastAPI app factory (`create_app`), routes, WebSocket |
| `__main__.py` | CLI entry point (`python -m web`) |
| `auth.py` | token auth middleware |
| `cockpit.py` / `cockpit_data.py` / `cockpit_sim.py` | P&L cockpit routes, data layer, sim harness |
| `bot_manager.py` | start/stop/monitor bot processes |
| `chat_agent.py` | LLM chat agent backing `/chat` |
| `log_streamer.py` / `ws_hub.py` | log tailing + WebSocket broadcast |
| `static/` / `templates/` | assets and Jinja templates (cockpit SPA in `static/cockpit/`) |
