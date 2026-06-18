# Live P&L Cockpit

A browser cockpit for real-time open P&L on the bot's 0DTE option positions —
faster than the TastyTrade GUI and aware of *which strategy* each position
belongs to (MEIC, Power Hour, ORB, …). Three coordinated views, all scoped to
the strategies you check on/off:

1. **Positions table** — grouped by strategy, collapsible, with select
   checkboxes, per-leg marks, per-trade / per-strategy open P&L and % of max.
2. **Intraday P&L curve** — cumulative open P&L over the day, seeded from
   persisted snapshots and ticking live.
3. **Payoff / risk graph** — P&L vs SPX (now via Black-Scholes + at expiry),
   with breakeven shape and a spot marker.

Design spec: [docs/superpowers/specs/2026-06-17-pnl-cockpit-design.md](superpowers/specs/2026-06-17-pnl-cockpit-design.md).

## Architecture

Served by the existing web dashboard ([web/app.py](../web/app.py)). All chart
logic is client-side (vanilla JS + Plotly). The backend only supplies data:

| Endpoint | Purpose |
|----------|---------|
| `GET /cockpit` | the single-page app |
| `GET /api/positions` | open trades grouped by strategy (+ legs, credits) |
| `POST /api/quotes` | proxies live marks from `options_api`, computes & persists per-position P&L |
| `GET /api/pnl_history` | persisted intraday P&L samples (seeds the curve) |

- **Positions + strategy grouping** come from the bot SQLite (the `trade_id`
  prefix names the strategy; `strategy_data` holds the leg symbols/credits).
  See [web/cockpit_data.py](../web/cockpit_data.py).
- **Live quotes** come from the sibling `options_api` service
  (`POST /market-data/options-snapshot`), proxied so the bearer token stays
  server-side.
- **P&L history** is persisted on every poll to a small SQLite table
  ([src_v2/services/pnl_snapshot.py](../src_v2/services/pnl_snapshot.py)).

## Running

```bash
# 1. Simulation harness (market closed / no positions) — replays a full
#    trading session on a simulated clock: a pre-generated 1-minute SPX path
#    from 09:30 to 16:00, Black-Scholes marks priced at the simulated
#    time-to-close (theta decays through the day), and positions that activate
#    at their entry times (ORB 09:45, MEIC 10:05/10:30, Power Hour 14:30).
#    No options_api needed.
BOT_WEB_TOKEN=<token> COCKPIT_SIM=1 python -m web --port 8080
# then open http://localhost:8080/cockpit  (log in with the token)
#
#  - Watch the whole day in ~2 min; SIM_MINUTES_PER_POLL controls the speed
#    (session-minutes advanced per ~2s poll; default 6).
#  - POST /api/sim/reset restarts the session from the open and clears history.
```

For live/production, see **[Production (live)](#production-live)** below.

### Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `COCKPIT_SIM` | _(off)_ | `1` to use the offline session-replay harness |
| `SIM_MINUTES_PER_POLL` | `6` | session-minutes advanced per poll (replay speed) |
| `COCKPIT_DB` | `data/meic_bot.db` | bot trading DB to read open trades from |
| `COCKPIT_PNL_DB` | `data/cockpit_pnl.db` | where intraday P&L snapshots are stored |
| `OPTIONS_API_URL` | `http://127.0.0.1:8101` | options_api base URL |
| `OPTIONS_API_TOKEN` | _(empty)_ | bearer token for the snapshot endpoint |

## Production (live)

Live mode reads open trades from the bot's SQLite and proxies real option marks
from the sibling `options_api` service. Do **not** set `COCKPIT_SIM`.

### Prerequisites
1. The **live bot** is running and writing open trades to its `trading_db`
   (e.g. `data/meic_bot.db`).
2. The **options_api** service is running (default port `8101`).

### 1. Configure the shared quote token (both sides, same value)
The snapshot endpoint is bearer-protected; `options_api` rejects it with `503`
until `EXTERNAL_API_TOKEN` is set, and the cockpit must send the same value.

In `../options_api/.env`:
```
EXTERNAL_API_TOKEN=<long-random-token>
```
In `0dte_bot/.env` (same token + cockpit wiring):
```
OPTIONS_API_TOKEN=<same-long-random-token>
OPTIONS_API_URL=http://127.0.0.1:8101
COCKPIT_DB=data/meic_bot.db        # the live bot's trading DB
COCKPIT_PNL_DB=data/cockpit_pnl.db
# leave COCKPIT_SIM unset
```
`BOT_WEB_TOKEN` must also be set (login token). `python -m web` loads `.env`
via `load_dotenv()`, so these are picked up automatically.

### 2. Start the services
```bash
# options_api (if not already up)
cd ../options_api && python run.py            # serves :8101

# dashboard + cockpit
cd ~/Dev/0dte_bot
python -m web --host 0.0.0.0 --port 8080
# open http://<host>:8080/cockpit  (log in with BOT_WEB_TOKEN)
```

Run it under `systemd` for a managed service (see `deploy/systemd/`).

### Caveats
- **Auth is a single shared token over plain HTTP.** Don't expose port 8080
  directly to the internet — bind to localhost / a VPN, or front it with a TLS
  reverse proxy (caddy/nginx). The cockpit page is `no-store`; the session
  cookie is `httponly`.
- **One trading DB at a time.** The cockpit reads a single `COCKPIT_DB`. If you
  run MEIC / Power Hour / ORB as separate bots with separate `trading_db`
  files, point `COCKPIT_DB` at the one you want — it won't merge multiple DBs.

## Tests

- Python: `pytest tests/test_cockpit.py` (data layer, sim, snapshot store, routes).
- JS pure functions: open `http://localhost:8080/static/cockpit/tests.html`
  (Black-Scholes, payoff, P&L aggregation, greeks).
