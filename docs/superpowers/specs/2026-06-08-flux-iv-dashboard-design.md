# Flux — SPX IV term-structure dashboard (design)

**Date:** 2026-06-08
**Origin:** Reconstruction of the "Flux" tool described by Steve Bernich in the
"Double Calendar Time Machine" interview (YouTube `JtGW1wNFNIY`). Flux monitors the
relationship between **front-expiration IV** and **back-expiration IV** — the signal a
calendar/diagonal trader uses to time entries (he enters when the front/back ratio is
elevated and starting to revert down, since a falling ratio profits a short-front
calendar).

**Status:** Approved design, pending spec review → implementation plan.

---

## 1. Purpose & scope

A small **web dashboard** that shows, for SPX, the three lines from his tool — **front
IV (blue)**, **back IV (green)**, and their **ratio = front/back (orange)** — across an
**intraday**, **5-day**, and **20-day** view, plus a **scanner** that ranks
expiration-pairs by how much the ratio has dropped (or risen) so the user can find entry
candidates.

Data comes from **marketdata.app**, which returns implied vol and delta **directly** (no
Black-Scholes inversion). Built **historical-first** (daily EOD snapshots are fully
offline-testable) with a **live-ready seam** (intraday polling during market hours).

### Non-goals (v1)
- No order placement, no strategy execution — this is an observation/scan tool.
- No historical *intraday* replay (marketdata.app is EOD for history; ThetaData could
  provide intraday replay later — deferred behind the `FluxSource` interface).
- No alerting/notifications, no non-SPX underlyings (the source is parameterized by
  underlying but v1 targets and tests SPX only).

---

## 2. Data reality (verified against marketdata.app)

The `MARKET_DATA_TOKEN` (in `../options_api/.env`, already used by the repo for SPY
candles) works for SPX options:

- **Endpoint:** `GET https://api.marketdata.app/v1/options/chain/SPX/` with
  `Authorization: Bearer <token>`.
- **Fields available directly:** `optionSymbol, side, strike, bid, ask, mid, iv, delta,
  dte, updated` (verified sample: iv 0.176, delta 0.53).
- **Live/near-live:** no `date=` param → current chain (HTTP **203** = delayed/cached on
  this plan; acceptable — the IV ratio moves slowly).
- **Historical:** `&date=YYYY-MM-DD` → **EOD snapshot** (one per day, `updated` = 16:00
  ET). No intraday history.
- **Cost control (important):** marketdata.app bills **per option returned**. We never
  fetch the full chain — only a **narrow ATM band** via `minDelta=0.25&maxDelta=0.75`
  (~15–25 options/expiration), which is all `atm_iv` needs. All fetches are **cached**
  (SQLite) so a day/expiry is pulled at most once.

Mapping to Flux:
- **5-day / 20-day views + scanner** → daily EOD ratios via `date=` (offline-testable).
- **Intraday (1-min) view** → poll the live chain every ~60 s during market hours.

---

## 3. Architecture (Approach A: Flask + plotly.js)

Pure-Python data/compute layer (unit-tested, no network in tests) behind a thin Flask
JSON shell, with a single plotly.js page.

```
src_v2/flux/
  __init__.py
  models.py            # OptionRow, ChainSnapshot, RatioPoint, ScanRow (dataclasses)
  sources/
    base.py            # FluxSource ABC: chain(), expirations()
    marketdata.py      # MarketDataSource — marketdata.app adapter (injectable fetch_json)
  engine.py            # atm_iv(chain, ref_delta) ; ratio_point(front, back, ref_delta)
  cache.py             # SQLite cache: (underlying, expiry, on_date) -> ChainSnapshot JSON
  history.py           # daily_ratio_series(front_dte, gap, n_days, ref_delta, end_date)
  scanner.py           # scan(min_dte, max_dte, gaps, ref_delta, by) -> [ScanRow]
  live.py              # LivePoller — intraday RatioPoint buffer (market-hours only)
  app.py               # Flask: /api/ratio, /api/scan, /api/expirations, / (dashboard)
  static/flux.html     # single-page plotly.js dashboard
  cli.py               # `python -m src_v2.flux serve|scan`
tests/flux/
  test_models.py test_engine.py test_marketdata.py test_cache.py
  test_history.py test_scanner.py test_app.py
```

**Reuse (DRY):** `dc_time_machine.iv.front_back_ratio`,
`dc_time_machine.flux_scanner.ratio_zscore` / `rank_candidates`,
`dc_time_machine.trading_calendar` (`sessions_between`, `nth_trading_day_after`,
`trading_days_after`). The marketdata token-loading pattern is reused from
`src/strategies/metf/metf_backtester.py`. Because marketdata.app gives IV directly,
`dc_time_machine.iv.implied_vol` (BS inversion) is **not** used here.

---

## 4. Data model (`models.py`)

```python
@dataclass(frozen=True)
class OptionRow:
    strike: float
    is_call: bool
    bid: float
    ask: float
    iv: float          # from marketdata.app, decimal (0.176 = 17.6%)
    delta: float       # signed (calls +, puts −)

@dataclass(frozen=True)
class ChainSnapshot:
    underlying: str            # "SPX"
    expiry: date
    as_of: datetime            # quote timestamp (UTC)
    rows: tuple[OptionRow, ...]

@dataclass(frozen=True)
class RatioPoint:
    as_of: datetime
    front_expiry: date
    back_expiry: date
    ref_delta: float
    front_iv: float
    back_iv: float
    ratio: float               # front_iv / back_iv

@dataclass(frozen=True)
class ScanRow:
    front_expiry: date
    back_expiry: date
    gap_td: int                # trading-day gap
    ratio: float
    ratio_prev: float          # prior session EOD ratio
    drop: float                # ratio_prev − ratio (positive = reverting down)
    zscore: float
    flag: bool                 # elevated AND reverting (entry candidate)
```

---

## 5. Data source layer (`sources/`)

```python
class FluxSource(ABC):
    def chain(self, expiry: date, on_date: Optional[date] = None) -> ChainSnapshot: ...
    def expirations(self, ref: date, min_dte: int, max_dte: int) -> list[date]: ...
```
- `on_date=None` ⇒ live/delayed; `on_date=D` ⇒ EOD snapshot for D.
- **Expiry resolution avoids a dedicated API endpoint.** SPX has (near-)daily expiries, so
  `expirations(ref, min_dte, max_dte)` is derived from `trading_calendar` —
  the candidate expiry dates are simply the NYSE sessions `min_dte..max_dte` trading days
  after `ref`. A `chain()` fetch for a candidate that doesn't exist returns `no_data` →
  treated as "not an expiry" (no error). This reuses the DC trading-day-distance approach
  and removes any dependency on a marketdata expirations endpoint.
- `MarketDataSource(FluxSource)`:
  - Builds `…/options/chain/SPX/?expiration=<e>&[date=<d>&]minDelta=0.25&maxDelta=0.75&columns=…`.
  - Accepts an injected `fetch_json(url) -> dict` (default urllib + Bearer token) so unit
    tests pass canned responses with **no network**.
  - Parses marketdata's **columnar** JSON (`{"strike":[...],"iv":[...],"delta":[...],…}`)
    into `ChainSnapshot`. Treats HTTP 200 and 203 identically; `s == "no_data"` → empty
    chain (not an error). Drops rows with null iv/delta.
  - Token from `MARKET_DATA_TOKEN` env, else `../options_api/.env` (same loader pattern as
    the existing repo code).

---

## 6. Engine (`engine.py`)

```python
def atm_iv(chain: ChainSnapshot, ref_delta: float = 0.35) -> Optional[float]:
    # pick the call nearest |delta-ref_delta| and the put nearest |delta+ref_delta|,
    # average their IVs; None if the band has no usable rows.

def ratio_point(front: ChainSnapshot, back: ChainSnapshot, ref_delta: float) -> Optional[RatioPoint]:
    f, b = atm_iv(front, ref_delta), atm_iv(back, ref_delta)
    # ratio via dc_time_machine.iv.front_back_ratio(f, b); None if either is None/NaN
```
- **ref_delta:** the strike on each expiration's skew at which IV is read. **Default
  0.35** (matches the strikes he trades and his "the front IV — the options you're
  selling" framing); the dashboard also offers **0.50** (clean ATM term structure). `>1`
  ratio = backwardation (front richer than back).

---

## 7. Cache (`cache.py`)

SQLite at `data/flux_cache.db`: table `chains(underlying, expiry, on_date, json, fetched_at)`
keyed by `(underlying, expiry, on_date)`. `on_date` is the session date for EOD; live
snapshots are **not** cached (always refetched). `MarketDataSource` consults the cache
before any EOD fetch. This bounds marketdata.app credit spend and makes history/scan
re-runs instant.

---

## 8. History (`history.py`)

```python
def daily_ratio_series(source, front_dte, gap, n_days, ref_delta,
                       end_date) -> list[RatioPoint]:
    # for each of the last n_days NYSE sessions ending end_date:
    #   front = session's EOD expiry nearest front_dte trading days out
    #   back  = nearest front_dte+gap trading days out
    #   fetch EOD chains (on_date=session), compute ratio_point
```
Powers **5-day** (`n_days=5`) and **20-day** (`n_days=20`) views. Expiry resolution reuses
the DC trading-day-distance logic (nearest available expiry within a tolerance).

---

## 9. Scanner (`scanner.py`)

```python
def scan(source, on_date, min_dte, max_dte, gaps, ref_delta, by="drop") -> list[ScanRow]:
    # for each front expiry in [min,max] DTE × each gap in gaps:
    #   ratio_today = EOD ratio_point(on_date)
    #   ratio_prev  = EOD ratio_point(prior session)
    #   zscore via ratio_zscore(today, recent history)
    #   flag = ratio_today > 1 (backwardation) AND drop > 0 (reverting down)
    # rank via rank_candidates(..., by)  ("drop" = reverting down first; "rise" = opposite)
```
Mirrors his scanner ("scan all expirations 0–30 DTE, any gap, by minimum ratio drop/rise
that day"). Output rendered as the dashboard's candidate table.

---

## 10. Live intraday (`live.py`)

`LivePoller(source, front_dte, gap, ref_delta, poll_seconds=60)`:
- A background thread that, **only when the NYSE session is open** (09:30–16:00 ET via
  `trading_calendar` + a clock), fetches live front/back chains, computes a `RatioPoint`,
  and appends it to an in-memory ring buffer (the intraday series). Optionally persisted
  to `data/flux_intraday/<date>.jsonl`.
- When the market is closed the buffer holds the most recent session; the dashboard labels
  the intraday view "delayed / last session" accordingly.
- The clock is injectable (`now_et` callable) so tests don't depend on wall time.

---

## 11. Web app (`app.py` + `static/flux.html`)

**Flask JSON endpoints** (each takes a `FluxSource`, injectable for tests):
- `GET /api/ratio?front_dte=9&gap=1&ref_delta=0.35&view=intraday|5day|20day`
  → `{points:[{as_of, front_iv, back_iv, ratio}], meta:{front_expiry, back_expiry, delayed}}`.
- `GET /api/scan?on_date=&min_dte=0&max_dte=30&gaps=1,2,3&ref_delta=0.35&by=drop`
  → `{rows:[ScanRow…]}`.
- `GET /api/expirations?min_dte=0&max_dte=30` → `{expirations:[…]}`.
- `GET /` → `static/flux.html`.

**Dashboard (`flux.html`, plotly.js):** one chart with three traces — front IV (blue),
back IV (green) on the left axis, ratio (orange) on a right axis — a **view toggle**
(intraday / 5-day / 20-day), a **controls bar** (front DTE, gap, ref-delta 0.35↔0.50),
and a **scanner table** below with the candidate pairs (★ on flagged rows). The intraday
view auto-refreshes every 60 s; daily views refresh on control change. A status banner
surfaces "delayed feed" / "market closed" / source errors.

**Launch:** `python -m src_v2.flux serve --port 8787` (also `… scan --on-date … --by drop`
for a headless scan).

---

## 12. Error handling

- Missing token → endpoints return `{error:"MARKET_DATA_TOKEN not set"}`, dashboard shows a
  banner; never a stack trace to the user.
- API down / timeout → retry once, then surface a clean error for that endpoint; other
  panels keep working.
- `s=="no_data"` for a date/expiry → that point is omitted (gap in the series), not an
  error.
- Delayed (203) clearly labeled; market-closed clearly labeled.

---

## 13. Testing

Unit tests, **no network** (canned marketdata JSON / a `FakeSource`):
- `test_marketdata`: columnar JSON → `ChainSnapshot`; band/columns in the URL; 203 == ok;
  no_data → empty; token-loader fallback.
- `test_engine`: `atm_iv` picks the right call/put by delta and averages; ratio via
  `front_back_ratio`; None handling.
- `test_cache`: round-trip; EOD cached, live not cached.
- `test_history`: series assembles N sessions, resolves expiries, uses cache.
- `test_scanner`: ranking by drop/rise, z-score, flag logic.
- `test_app`: Flask test client with a `FakeSource` → endpoint JSON shapes + error paths.

Frontend (`flux.html`) is verified manually against the live dashboard.

---

## 14. Deliverables

- `src_v2/flux/` package + `tests/flux/`.
- `src_v2/flux/README.md` — token setup, `serve`/`scan` usage, what each view/colour means,
  the ref-delta meaning, and the marketdata.app credit/cost note.
- Defaults: SPX, front_dte 9 / gap 1, ref_delta 0.35 (0.50 toggle), ATM-band fetch.
