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

# 2. Live (bot running + options_api up):
BOT_WEB_TOKEN=<token> \
COCKPIT_DB=data/meic_bot.db \
OPTIONS_API_URL=http://127.0.0.1:8101 \
OPTIONS_API_TOKEN=<external_api_token> \
python -m web --port 8080
```

### Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `COCKPIT_SIM` | _(off)_ | `1` to use the offline session-replay harness |
| `SIM_MINUTES_PER_POLL` | `6` | session-minutes advanced per poll (replay speed) |
| `COCKPIT_DB` | `data/meic_bot.db` | bot trading DB to read open trades from |
| `COCKPIT_PNL_DB` | `data/cockpit_pnl.db` | where intraday P&L snapshots are stored |
| `OPTIONS_API_URL` | `http://127.0.0.1:8101` | options_api base URL |
| `OPTIONS_API_TOKEN` | _(empty)_ | bearer token for the snapshot endpoint |

## Tests

- Python: `pytest tests/test_cockpit.py` (data layer, sim, snapshot store, routes).
- JS pure functions: open `http://localhost:8080/static/cockpit/tests.html`
  (Black-Scholes, payoff, P&L aggregation, greeks).
