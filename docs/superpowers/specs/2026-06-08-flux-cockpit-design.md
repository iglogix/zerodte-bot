# Flux — DC Time Machine cockpit (corrected design)

**Date:** 2026-06-08
**Supersedes the framing of** `2026-06-08-flux-iv-dashboard-design.md` (that built the IV
*history* piece, which is correct and reused; this corrects the tool's purpose).

## What Flux actually is

Not a passive IV chart — a **cockpit for trading the DC Time Machine** on a selected
front/back expiration pair. From the real tool (`flux.navigationtrading.com`) + the video:

1. **Live 1-min intraday IV scanner** for the selected F/B pair — front IV / back IV /
   ratio, x-axis = **time of day**, updating every minute. *(new — the current "Today"
   view is wrong; it falls back to EOD history.)*
2. **History (5D / 20D)** of the ratio for the pair. **Already implemented** via
   marketdata.app — keep as-is.
3. **Find** the front/back expirations **and the DC strikes** (at target delta).
4. **Build the DC order** (4 legs, sell front / buy back) → manual ticket (default) or
   submit via TastyTrade.
5. **Build the transformation order** (close backs + buy front wings, credit ≥ debit +
   wing) for an **existing DC auto-detected from open TastyTrade positions** → ticket or
   submit.
6. (A further section exists in the real tool — out of scope for now.)

The IV ratio is *context*; the payload is **strikes + DC order + transform order**. This is
human-in-the-loop execution near mid — the one regime the backtest showed actually works.

## Decisions (from the user)

- **Live intraday data:** TastyTrade (DXLink). **History:** marketdata.app (done).
- **Order submission:** default **manual** — generate the ticket (legs + editable limit);
  submitting requires a non-manual bot mode (`sandbox`/`live`) + an explicit action.
- **Existing DC for transform:** **auto-detected from open TastyTrade positions.**

## Architecture — reuse-heavy, built in 3 increments

Keep: `flux/engine.py` (atm_iv at ref-delta, ratio_point), `flux/history.py` +
`sources/marketdata.py` (5D/20D), the Flask shell + dashboard. Reuse from the bot:
`services/tastytrade_client.py` (DXLink Quote streamer, `_build_leg`, complex-order POST),
`backtest/dc_time_machine/{pricing,iv,trading_calendar}.py`.

### Increment 1 — Live intraday scanner (TastyTrade)
- `flux/sources/tastytrade_live.py` — `TastyTradeLiveSource`: given a front & back expiry,
  subscribe the ~ATM strikes via the bot's persistent **DXLink Quote streamer**, take the
  bid/ask mids, and compute **front ATM IV / back ATM IV via Black–Scholes** (same `iv.py`
  path as history → consistent IV). Returns a live `RatioPoint`.
- `flux/live.py` (extend the existing poller): a 1-min loop, market-hours-aware, appending
  `RatioPoint`s to an in-memory buffer keyed by (front, back, ref_delta); persisted to
  `data/flux_intraday/<date>.jsonl` so the day's line survives a restart.
- Endpoints: `GET /api/intraday?front=YYYY-MM-DD&back=...&ref_delta=` → today's 1-min
  series; `GET /api/expirations_iv?symbol=SPX` → each available expiry with **live ATM IV%
  + day-change** (for the expiration list).
- Dashboard "**Today**" view: time-of-day x-axis, live front IV / back IV / ratio, top
  metrics (IV ratio + today-change, front IV% [DTE], back IV% [DTE]); an **expirations list**
  with ATM IV% + day-change and **click-F / click-B** to select the pair; a "Live (1m)"
  status with last-snapshot time. 5D/20D toggle keeps the existing marketdata history.

### Increment 2 — Find strikes + build the DC order
- `flux/strikes.py` — `select_dc_strikes(front_chain, spot, target_delta=0.35)` → (K_put,
  K_call) nearest ±target_delta (reuse the delta logic; live chain quotes + BS delta).
- `flux/orders.py` — `build_dc_order(front_exp, back_exp, k_put, k_call, qty)` → 4
  `OrderLeg`s (Sell-to-Open front put@Kp, Buy-to-Open back put@Kp, Sell front call@Kc, Buy
  back call@Kc) + a **suggested limit debit** from live mids; returns an `OrderTicket`
  (legs, net side=debit, limit, est cost). Pure/testable.
- `services/tastytrade_client.py` — add `place_complex_order(legs, price, price_effect,
  tif)` generalizing `_place_iron_condor` to **arbitrary multi-expiry** legs (legs already
  carry their own expiry via `format_osi`).
- `GET/POST /api/dc_order` → returns the ticket; in non-manual mode a submit posts the
  complex order. Dashboard "**Build DC**" panel: strikes + ticket + copy-for-manual /
  Submit (mode-gated).

### Increment 3 — Build the transform order (from positions)
- `services/tastytrade_client.py` — add `get_positions()` (`GET
  /accounts/{acct}/positions`).
- `flux/transform.py` — `detect_double_calendars(positions)` → for each long-back +
  short-front put&call pair on matching strikes across two expiries, infer
  `(front_exp, back_exp, k_put, k_call, debit_paid, qty)`; `build_transform_order(dc,
  wing=5, epsilon=0.0, live_chain)` → legs (Sell back put@Kp, Sell back call@Kc, Buy front
  put@Kp−W, Buy front call@Kc+W) with **required limit credit = debit + wing + epsilon**
  and the **current achievable credit** from live mids (reuse `dc_time_machine.pricing`).
- `GET /api/positions` (detected DCs + current transform-credit vs threshold), `POST
  /api/transform_order` (ticket / submit). Dashboard "**Open DCs**" panel: each detected DC
  with current credit vs `D+W`, ★ when transformable, "Build Transform" → ticket / Submit.

## Safety

Submission is **off by default** (manual mode → tickets only). The Submit path is gated by
the bot's mode (`tastytrade_client` + `BOT_MODE`): `manual` = never submit, `sandbox` =
cert, `live` = real money — and always requires an explicit user action with the ticket
shown first. No auto-submission, ever.

## Testing

Pure pieces are TDD'd with fake quotes/positions, no network: `select_dc_strikes`,
`build_dc_order` (legs + limit), `detect_double_calendars`, `build_transform_order`
(credit math via `pricing`), live-source IV from canned quotes. The DXLink stream and live
order submission are verified manually (need a broker session).

## Out of scope (now)

Real-time *broker* IV (we compute from streamed mids; dxfeed `Greeks` is a later
refinement), non-SPX underlyings, the extra tool section, and any automated/unattended
trading (this is a manual cockpit).
