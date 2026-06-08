# Flux — SPX IV term-structure dashboard

A reconstruction of the **"Flux"** tool Steve Bernich describes in the Double Calendar
Time Machine interview (YouTube `JtGW1wNFNIY`): it shows the relationship between
**front-expiration IV** and **back-expiration IV** so a calendar/diagonal trader can time
entries (he enters when the front/back ratio is elevated and starting to revert *down* —
a falling ratio profits a short-front calendar).

Design: [`docs/superpowers/specs/2026-06-08-flux-iv-dashboard-design.md`](../../../docs/superpowers/specs/2026-06-08-flux-iv-dashboard-design.md)
· Plan: [`docs/superpowers/plans/2026-06-08-flux-iv-dashboard.md`](../../../docs/superpowers/plans/2026-06-08-flux-iv-dashboard.md)

## What it shows

- **Three lines** (his blue/green/orange): **front IV**, **back IV**, and **ratio =
  front/back** (`>1` = backwardation, front richer than back).
- **Views:** 5-day and 20-day daily history (and an intraday view — see *Limitations*).
- **Scanner:** ranks expiration-pairs across 0–30 DTE and several gaps by how much the
  ratio **dropped** (reverting down — entry candidates) or **rose**, flagging "elevated &
  reverting" pairs.

## Reference delta (the `ref Δ` toggle)

IV varies by strike (skew), so Flux reads "the" IV at a chosen delta on each expiration:
- **0.35** (default) — the strikes he actually trades ("the front IV — the options you're
  selling"); most faithful to his Flux and to the DC strategy.
- **0.50** — clean at-the-money term structure.

## Data source — marketdata.app

Uses **marketdata.app**. Token: `MARKET_DATA_TOKEN` from the environment, else
`../options_api/.env` (the existing repo convention; same token used for SPY candles).

- **IV source:** the **live** feed returns `iv`/`delta` directly; the **historical EOD**
  feed returns only prices (iv/delta are null). So for historical, Flux **computes IV and
  delta via Black–Scholes** (`dc_time_machine.iv`) from the bid/ask mid, a put-call-parity
  spot, and DTE. Live uses the API's IV as-is.
- **Historical** is **EOD** (`&date=` → one snapshot/day at 16:00 ET) → powers the
  5/20-day views and the scanner. Fully offline once cached. (Today's EOD doesn't exist
  yet → that session is skipped, not an error.)
- **Live** is **delayed/cached** (HTTP 203 on this plan) — fine for an IV ratio, which
  moves slowly.
- **Cost:** marketdata.app bills **per option returned**, so every request uses
  `strikeLimit=25` (~50 options/chain ≈ ~50 credits) and EOD snapshots are **SQLite-cached**
  (`data/flux_cache.db`) — a (date, expiry) is fetched at most once. A 20-day history of
  one front/back pair is ~40 chain fetches **on first run**, instant thereafter.

## Usage

```bash
# Headless scanner (prints ranked expiry-pairs for a date)
.venv/bin/python -m src_v2.flux scan --on-date 2026-06-05 --min-dte 5 --max-dte 12 --gaps 1,2 --ref-delta 0.35

# Web dashboard
.venv/bin/python -m src_v2.flux serve --port 8787
#   → open http://127.0.0.1:8787
```

Controls: front DTE, gap, ref-Δ (0.35↔0.50), and the 1-min / 5-day / 20-day view toggle.

## Limitations (v1)

- **Intraday view** currently falls back to the 5-day daily series. The live poller
  (`live.py`, market-hours-only, tested) is built but **not yet mounted** into `serve` as a
  background thread + `/api/intraday` endpoint — a small follow-up (the spec's optional
  Task 12). marketdata.app's live feed is delayed anyway.
- **SPX only** (the source is parameterized by underlying but v1 targets/tests SPX).
- **No order placement** — this is an observation/scan tool.

## Tests

`.venv/bin/pytest tests/flux/ -v` — 21 unit tests, all synthetic (canned marketdata JSON /
a `FakeSource`); **no network**.

## Architecture

Pure-Python, unit-tested compute layer behind a thin Flask shell:
`models → sources/{base,marketdata} → engine → cache → history → scanner → live`, with
`app.py` (Flask JSON endpoints) + `static/flux.html` (plotly.js) + `cli.py`. Reuses
`dc_time_machine`'s `front_back_ratio`, `ratio_zscore`/`rank_candidates`, and
`trading_calendar`.
