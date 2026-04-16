# src_v2 Backtest Engine — Design Spec

**Date:** 2026-04-11
**Status:** Draft
**Scope:** ORB-based entry only (orb_em_ic strategy); scheduled entry strategies deferred.

## Goal

Build a backtest engine in `src_v2/backtest/` that runs the **actual `LiveStrategy` code** against historical SPY bars + SPX CBBO option quotes. The engine is self-contained — no imports from `src_8am`.

This validates that the live trading code (`OrbEmIronCondorStrategy.search_candidates()`, `.check_exit()`, etc.) produces correct results on historical data, using the same interfaces the live bot uses.

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `src_v2/backtest/__init__.py` | Package init |
| `src_v2/backtest/data_loader.py` | SPY 1-min CSV loader → per-day bar lists |
| `src_v2/backtest/cbbo_store.py` | CBBO quote store — loads pickle cache, serves bid/ask per (minute, strike, type) |
| `src_v2/backtest/engine.py` | Backtest engine — ORB detection, entry via `LiveStrategy`, exit simulation via `LiveStrategy` |
| `src_v2/backtest/report.py` | Stats computation: WR, Sharpe, PF, drawdown, exit breakdown |
| `scripts/run_v2_backtest.py` | CLI runner: loads config, runs engine, prints report |

### Data Flow

```
config YAML
    ↓
CLI runner (scripts/run_v2_backtest.py)
    ↓
SpyLoader reads SPY 1-min CSV → List[DayBars]
CbboStore preloads pickle cache → Dict[date, QuoteMap]
    ↓
BacktestEngine.run(days) → for each day:
    1. Load CBBO quotes for day
    2. Build ORB candle (08:00–08:15 bars)
    3. Filter: range (min/max pts), NFP/FOMC/CPI
    4. Detect breakout: first post-open bar closing above ORB high (bullish) or below ORB low (bearish)
    5. Detect midline retest: bar touching midline ± tolerance
    6. Fill bar: next bar after retest
    7. Build option chain from CBBO at fill minute → List[Dict] with real OCC symbols
    8. Derive spot via put-call parity
    9. Set strategy._orb_bias = bias
   10. strategy.search_candidates(chain, spot) → LiveTradeSetup
   11. strategy.validate(setup, chain)
   12. Record entry (IC = put spread + call spread)
   13. Walk remaining bars: build quotes dict from CBBO, call strategy.check_exit(trade, quotes)
   14. EOD (15:50): expire worthless
   15. Compute PnL = (credit - debit) × multiplier × qty
    ↓
List[BacktestResult]
    ↓
generate_report(results) → summary stats
```

## Module Details

### data_loader.py

Copied from `src_8am/backtest/spy_loader.py`, simplified.

```python
@dataclass
class DayBars:
    session_date: date
    symbol: str
    bars: List[Dict]  # each: {open, high, low, close, volume, ts_et: datetime}

class SpyLoader:
    def __init__(self, csv_path: str, price_scale: float = 10.0, symbol: str = "SPX")
    def iter_days(self, start_date, end_date) -> Iterator[DayBars]
```

- Reads SPY 1-min OHLCV CSV
- Multiplies prices by `price_scale` (10.0 for SPY→SPX)
- Groups bars by trading date
- Attaches timezone-aware ET timestamps

### cbbo_store.py

Copied from `src_8am/backtest/cbbo_store.py` — the full class including:
- `preload_cache(cache_dir)` — loads all pickle files into memory
- `load_day(date)` — loads single day from pickle or zst
- `lookup(minute, strike, is_call)` — returns `(bid, ask)` with stale-minute fallback
- `atm_straddle(minute, spot)` — ATM put+call mid (expected move)
- `derive_spot(minute, bar_price)` — put-call parity spot derivation
- `available_strikes` property

No changes to the interface. Self-contained copy.

### engine.py

Core backtest engine.

```python
@dataclass
class BacktestResult:
    trade_id: str
    session_date: date
    symbol: str

    # ORB
    orb_high: float
    orb_low: float
    orb_midline: float
    orb_range_pts: float

    bias: Optional[str] = None
    breakout_time: Optional[datetime] = None
    breakout_price: Optional[float] = None

    # Spread
    spread_type: Optional[str] = None       # "put_credit_spread" | "call_credit_spread"
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None
    spread_width: float = 0.0

    # Entry
    entry_time: Optional[datetime] = None
    underlying_at_entry: Optional[float] = None
    credit_received: Optional[float] = None
    short_delta_at_entry: Optional[float] = None

    # Exit
    exit_time: Optional[datetime] = None
    underlying_at_exit: Optional[float] = None
    debit_paid: Optional[float] = None
    exit_reason: Optional[str] = None       # expired_worthless | tp_50pct | short_breached | eod_close

    realized_pnl_usd: Optional[float] = None

    status: str = "skipped"                 # skipped | closed
    skip_reason: Optional[str] = None

    qty: int = 1


class BacktestEngine:
    def __init__(self, config: Dict, strategy: LiveStrategy)
    def run(self, days: List[DayBars]) -> List[BacktestResult]
```

#### Constructor

Reads from config:
- `strategy.*`: ORB window times, midline tolerance, range filter, event skips
- `options.*`: multiplier, quantity, slippage
- `market.*`: trading hours, market close time
- `backtest.*`: cbbo_cache_dir

Creates `CbboStore` from `backtest.cbbo_cache_dir`.
Calls `strategy.configure(config)`.

#### run(days) → List[BacktestResult]

Groups days by date (picks highest-volume contract if multiple).
Skips NFP/FOMC/CPI days per config.
Calls `_process_day()` for each.

#### _process_day(day_bars) → List[BacktestResult]

Returns 2 results for IC (put side + call side), or skip result.

**ORB detection** (same logic as `src_8am`):
1. Collect bars in `[orb_start, orb_end)` → compute high, low, midline, range
2. Filter range: skip if < min_pts or > max_pts

**Breakout detection:**
1. Walk post-open bars until `close > orb_high` (long) or `close < orb_low` (short)
2. If none before EOD → skip

**Midline retest:**
1. Walk post-breakout bars until bar's low/high touches `midline ± tolerance`
2. If none before EOD → skip

**Entry:**
1. Fill bar = next bar after retest bar
2. ES entry price = fill bar open (used for ORB reference)
3. Derive SPX spot from CBBO put-call parity at fill minute
4. Build chain from CBBO (all strikes within ±200 pts of spot, both put/call)
5. Set `strategy._orb_bias = bias`
6. Call `strategy.search_candidates(chain, spot)` → `LiveTradeSetup` or empty
7. Call `strategy.validate(setup, chain)` → pass/fail
8. Extract per-side credits from `setup.extra["put_credit"]` and `setup.extra["call_credit"]`

**Chain building:**

```python
def _build_chain(self, minute: str, spot: float, session_date: date) -> List[Dict]:
    chain = []
    for strike in self._cbbo.available_strikes:
        if abs(strike - spot) > 200:
            continue
        for is_call in [False, True]:
            q = self._cbbo.lookup(minute, strike, is_call)
            if q is None:
                continue
            bid, ask = q
            if bid <= 0 and ask <= 0:
                continue
            sym = self._make_symbol(session_date, strike, is_call)
            chain.append({
                "strike": strike,
                "option_type": "call" if is_call else "put",
                "bid": bid,
                "ask": ask,
                "symbol": sym,
                "delta": 0,  # CBBO doesn't provide Greeks
            })
    return chain
```

Symbol format: `SPXW YYMMDDCSSSSSSSS` (standard OCC, e.g. `SPXW 260411P05900000`).

**Exit simulation:**

Build a `trade` dict matching what `check_exit()` expects:
```python
trade = {
    "short_put_symbol": setup.legs[0].symbol,    # from LiveTradeSetup
    "long_put_symbol": setup.legs[1].symbol,
    "short_call_symbol": setup.legs[2].symbol,
    "long_call_symbol": setup.legs[3].symbol,
    "short_put_strike": setup.strikes["short_put"],
    "short_call_strike": setup.strikes["short_call"],
    "put_credit": setup.extra["put_credit"],
    "call_credit": setup.extra["call_credit"],
}
```

For each bar after fill:
1. If bar time >= EOD → expire worthless, debit = 0
2. Build quotes dict:
   ```python
   quotes = {}
   for leg in setup.legs:
       q = cbbo.lookup(bar_minute, leg.strike, leg.option_type == "call")
       if q:
           quotes[leg.symbol] = {"bid": q[0], "ask": q[1]}
   quotes["_underlying"] = {"price": bar_spot}
   ```
3. Call `strategy.check_exit(trade, quotes)` → `ExitAction`
4. On `take_profit`: debit = credit × (1 - tp_pct) for the closed side; other side continues
5. On `stop_loss` (breach): debit = spread width for breached side
6. On `hold`: continue to next bar

**Per-side tracking:** Since IC has two independent sides (put spread + call spread) that can exit at different times, the engine tracks each side separately. A side is "closed" when `check_exit()` returns an action targeting its legs. The other side continues until it also exits or reaches EOD.

This produces **two `BacktestResult` records per IC trade** (one per side), matching the `src_8am` convention.

**Debit calculation on exit:**
- `expired_worthless`: debit = 0 (full credit kept)
- `tp_*`: debit = credit × (1 - tp_pct) — the residual value at TP
- `short_breached`: debit = spread mark from CBBO at breach bar, capped at spread width
- `eod_close`: debit = spread mark from CBBO at EOD bar

Spread mark from CBBO: `short_mid - long_mid` (same as entry pricing).

### report.py

```python
def generate_report(results: List[BacktestResult]) -> Dict[str, Any]:
    """Compute summary statistics from backtest results."""
```

Returns:
- `trades_placed`: count of closed trades
- `win_rate`: fraction with PnL >= 0
- `total_pnl_usd`: sum of realized PnL
- `avg_pnl_per_trade_usd`: mean PnL
- `sharpe_annualised`: annualized Sharpe (daily PnL series, 252 days)
- `profit_factor`: gross profit / gross loss
- `max_drawdown_usd`: peak-to-trough on cumulative equity
- `by_exit_reason`: {reason: {count, pnl_usd}}

Same output format as `src_8am/backtest/options_report.py` so existing chart code is compatible.

### scripts/run_v2_backtest.py

```
Usage:
  python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml
  python -m scripts.run_v2_backtest --config config/config_orb_em_ic_spx.yaml --start 2025-10-01 --end 2026-01-01
```

Flow:
1. Parse args (--config, --start, --end)
2. Load YAML config
3. Instantiate strategy from `config.bot.strategy` (currently only `orb_em_ic`)
4. Load SPY bars via `SpyLoader`
5. Preload CBBO cache
6. Create `BacktestEngine(config, strategy)`
7. Run engine, get results
8. Generate and print report

## Config Changes

Add `backtest:` section to `config/config_orb_em_ic_spx.yaml`:

```yaml
backtest:
  data_file: "data/spy_1m.csv"
  data_source: "spy"
  price_scale: 10.0
  cbbo_cache_dir: "data/cbbo_cache"
  start_date: "2025-09-22"
  end_date: "2026-03-20"
```

## Validation

The backtest should reproduce the `src_8am` results on the same data/config:
- 72 trades, 93.1% WR, Sharpe 14.44, PF 4.35, 5 breaches

If numbers diverge, the delta reveals bugs in either the live strategy code or the backtest adapter.

## Out of Scope

- Scheduled entry mode (iron_condor, iron_fly strategies)
- Optimizer / grid search / walk-forward CV
- Chart generation (can be added later using the report output)
- Databento loader (SPY CSV + CBBO pickle is the active data path)
