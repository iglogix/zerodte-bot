# Live P&L Cockpit — Design Spec

**Date:** 2026-06-17
**Status:** Approved (design); pending implementation plan
**Author:** brainstormed with Claude Code

## Problem

The TastyTrade web/desktop GUI lags when displaying real-time P&L for 0DTE option
positions — the P&L curve is not up to date. We also cannot manage positions by
strategy in TastyTrade: when multiple bot strategies (e.g. MEIC, ORB, Power Hour)
hold positions on the same underlying, TastyTrade shows a flat list of legs with no
notion of which legs belong to which strategy entry.

We want a ToS-like browser cockpit that shows live open P&L, lets us select/deselect
positions grouped by strategy, and reflects that selection across all charts.

## Goals

- Real-time open P&L that updates faster than the TastyTrade GUI (~2s cadence).
- Three coordinated views, all scoped to the selected strategies:
  1. **Live positions table** — ToS-style position statement, grouped by strategy,
     collapsible, with select checkboxes, per-leg mark / open P&L / % to TP / greeks,
     and per-group + grand totals.
  2. **Intraday cumulative P&L line chart** — the "curve that lags" in TastyTrade,
     summed over selected strategies, ticking live and seeded with the day's history.
  3. **Payoff / risk graph** — P&L vs underlying price (ToS Analyze-tab style): a
     "now" curve and an "expiry" curve, breakevens, and a current-spot marker.
- Group/select positions by strategy, reflected in all three views and the net-greeks
  header.

## Non-Goals

- No order entry / position management actions (close, roll). View-only cockpit.
- No multi-account support beyond the bot's configured account.
- No historical/backtest P&L beyond the current trading day.
- Not a replacement for the bot's own risk/exit logic — display only.

## Architecture

A single-page app served by the **existing bot FastAPI app** ([web/app.py](../../../web/app.py)).
All UI and chart logic run client-side in vanilla JS; the backend only supplies data.

Two data sources:

- **Positions + strategy grouping** → the bot's SQLite DB. Strategy grouping exists
  *only* here (TastyTrade and `options_api` do not know which legs belong to which
  strategy). Encoded as the `trade_id` prefix (`meic` / `ph` / `orbdir`) plus the
  `strategy_data` JSON blob of leg symbols and credits.
- **Live quotes + greeks** → sibling project `options_api`
  (`~/Dev/options_api`), endpoint `POST /market-data/options-snapshot`
  (Bearer `EXTERNAL_API_TOKEN`; port 8101 dev / 8080 prod; CORS enabled). Given a list
  of legs `[{symbol (OSI), qty, side, multiplier, ...}]` it returns bid/ask/mid/mark +
  greeks + IV + underlying quote + market_regime. Poll-only (no websocket exposed).

```
Browser (cockpit page, vanilla JS + Plotly)
  │  GET  /api/positions        → open trades grouped by strategy (+ legs, entry credit, TP/stop)
  │  POST /api/quotes  ──proxy──→ options_api /market-data/options-snapshot  (Bearer added server-side)
  │  GET  /api/pnl_history?date= → persisted intraday P&L series
  │  GET  /cockpit              → serves the static page
  ▼
bot FastAPI (web/app.py)  ── reads ──→ SQLite (IronCondorTrade, new PnlSnapshot table)
                          ── writes ─→ PnlSnapshot (via PositionMonitor hook)
```

### Decisions (approved 2026-06-17)

1. **Proxy quotes** browser → bot → `options_api`, not a direct browser→options_api
   call. Keeps `EXTERNAL_API_TOKEN` server-side, sidesteps CORS, and lets the proxy
   persist a P&L snapshot on each poll. The browser only ever talks to the bot app.
2. **Payoff "now" curve via Black-Scholes reprice** using the per-leg IV returned in
   the snapshot; the expiry curve is plain intrinsic value. Accurate for 0DTE with no
   extra data.
3. **Charting via Plotly.js**, already shipped in the repo for the Flux cockpit
   ([src_v2/flux](../../../src_v2/flux)) — no new dependency, handles both chart types.
   (uPlot would be faster for the tick curve but adds a dependency; Plotly is fine at
   2s cadence.)

## Components

Each unit has one purpose, a defined interface, and is independently testable.

### 1. `src_v2/services/pnl_snapshot.py` (new)
- New `PnlSnapshot` table: `(id, ts, trade_id, strategy, open_pnl, underlying_price)`.
- `record(trade_id, strategy, open_pnl, underlying_price, ts=None)` — persist one sample.
- `series_for_date(date)` — return ordered samples for the day, suitable for seeding
  the intraday curve (per-strategy and total).
- Depends on: the SQLAlchemy session / DB layer in `src_v2/models`.

### 2. PositionMonitor hook
- PositionMonitor already computes per-leg open P&L every 30s. Add one call to
  `pnl_snapshot.record(...)` per open trade per cycle. No new polling loop.

### 3. `web/app.py` routes (new, thin — marshal data only)
- `GET /api/positions` — open trades (`status='open'`) grouped by strategy. Each entry:
  `{trade_id, strategy, legs: [...], entry_credit, tp_target, stop_target}`.
- `POST /api/quotes` — proxy: accepts a leg list, adds the Bearer header, forwards to
  `options_api /market-data/options-snapshot`, returns the snapshot. Persists a P&L
  snapshot per poll. Passes through upstream errors with clear status.
- `GET /api/pnl_history?date=` — persisted intraday P&L series (delegates to
  `pnl_snapshot.series_for_date`).
- `GET /cockpit` — serves the static cockpit page.

### 4. Leg extractor (new helper, used by `/api/positions`)
- Turns a trade's `strategy_data` into a generic leg list
  `[{symbol, qty, side, strike, right, expiration, multiplier}]` by reading `*_symbol`
  keys, matching `*_open_qty`, and inferring side from `short_` / `long_` prefixes.
- Strategy name derived from the `trade_id` prefix (`meic` → MEIC, `ph` → Power Hour,
  `orbdir` → ORB Directional).

### 5. Static client `web/cockpit/` (new)
- `index.html` — layout: left positions table; top-right intraday curve; bottom-right
  payoff graph; header with net delta/gamma/theta/vega, total open P&L, and a stale
  badge.
- `app.js` — poll loop (~2s) + app state + rendering + strategy-selection wiring.
- `pnl.js` — **pure functions** for open-P&L math (current spread value vs entry credit).
- `payoff.js` — **pure functions** for Black-Scholes "now" reprice and intrinsic
  "expiry" payoff across a spot grid.
- `styles.css` — styling.
- `tests.html` — tiny in-browser assertion page for `pnl.js` / `payoff.js` (no Node build).

## Data Flow (poll loop)

1. **On load:** `GET /api/positions` → render grouped table; `GET /api/pnl_history` →
   seed the intraday curve.
2. **Every ~2s:** POST the legs of the **selected** strategies to `/api/quotes` → marks
   + greeks + spot.
3. **Compute (client):** open P&L per strategy (current spread value vs stored entry
   credit); update table; append a point to the intraday curve; recompute the payoff
   graph and net greeks — all scoped to the strategy checkboxes.

## Error Handling

- `options_api` down / timeout → keep last marks, show a stale badge, do not corrupt
  the intraday curve.
- Leg quote `missing` / `partial` → flag that leg, exclude it from totals with a
  warning rather than silently zeroing it.
- `EXTERNAL_API_TOKEN` unconfigured → explicit error surfaced in the header.

## Testing

- **Python (`tests/`, never `tests_locked/`):**
  - `/api/positions` grouping — prefix→strategy mapping and leg extraction from
    `strategy_data`.
  - `/api/quotes` proxy — request shaping and upstream-error passthrough.
  - `pnl_snapshot` — `record` / `series_for_date` round-trip.
- **JS:** `pnl.js` and `payoff.js` are pure functions exercised by
  `web/cockpit/tests.html` (in-browser assertions, no build step).

## Key Code References

- Open trades: [db_service.py:64](../../../src_v2/services/db_service.py#L64) `get_open_trades()`
- Trade-id prefixes: [tool_handlers.py:420](../../../src_v2/agent/tool_handlers.py#L420) (meic),
  :506 (ph), :721 (orbdir)
- `strategy_data` leg symbols: [tool_handlers.py:523](../../../src_v2/agent/tool_handlers.py#L523)
- DXLink streamer: [tastytrade_client.py:368](../../../src_v2/services/tastytrade_client.py#L368)
- Existing FastAPI app + WS hub: [web/app.py](../../../web/app.py)
- Flux cockpit (Plotly precedent): [src_v2/flux](../../../src_v2/flux)
- options_api snapshot endpoint: `~/Dev/options_api/app/routes/market_data.py`
