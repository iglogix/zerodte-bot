# 8AM ORB Options Strategy Reference

## Overview

Sell 0DTE SPX credit spreads based on the 8:00-8:15 AM ET Opening Range Breakout (ORB) pattern on ES futures. The strategy uses ES price action for entry signals but prices SPX options for execution.

---

## Entry Rules

### 1. ORB Window (8:00-8:15 AM ET)
- Capture the high/low of the first 15-minute candle after ES futures open
- Calculate midline: `(ORB_high + ORB_low) / 2`
- Calculate range: `ORB_high - ORB_low`

### 2. Range Filter
| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Min Range | `strategy.orb_range_filter.min_pts` | 2.0 | Skip if ORB range < this (no volatility) |
| Max Range | `strategy.orb_range_filter.max_pts` | 30.0 | Skip if ORB range > this (gap/event day) |

### 2b. NFP Filter
| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Skip NFP | `strategy.skip_nfp_fridays` | true | Skip first Friday of month |

NFP (Non-Farm Payrolls) releases at 8:30 AM ET on first Fridays, right after the ORB window. Backtest shows **31.6% loss rate** on NFP days vs 9.2% overall.

### 3. Breakout Detection (after 9:30 AM ET)
- **LONG bias**: 1-min bar closes above ORB high
- **SHORT bias**: 1-min bar closes below ORB low
- First breakout direction determines trade bias

### 4. Midline Retest (Entry Trigger)
- After breakout, wait for price to return to midline
- Entry triggered when bar touches midline within tolerance

| Parameter | Config Key | Default |
|-----------|------------|---------|
| Midline Tolerance | `strategy.midline_tolerance_pts` | 0.50 pts |

### 5. Spread Selection

| Bias | Spread Type | Short Strike Location |
|------|-------------|----------------------|
| LONG (bullish) | PUT credit spread | Below ORB low |
| SHORT (bearish) | CALL credit spread | Above ORB high |

---

## Options Parameters

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Target Delta | `options.target_short_delta` | 0.16 | Short strike delta (~1 SD OTM) |
| Max Delta | `options.max_short_delta` | 0.25 | Reject if actual delta exceeds this |
| Spread Width | `options.spread_width_pts` | 5 | Distance between short/long strikes |
| Strike Step | `options.strike_step_pts` | 5 | Round strikes to nearest N points |
| Min Credit | `options.min_credit_per_spread` | 0.50 | Minimum credit to accept ($50/contract) |

### Strike Selection
1. Use Black-Scholes to find strike at target delta
2. Round to nearest 5-point increment
3. Long strike = short strike ± spread_width (based on spread type)

---

## Exit Rules

### Take Profit (TP)
| Parameter | Config Key | Default |
|-----------|------------|---------|
| TP % of Credit | `options.tp_pct_of_credit` | 0.50 |

- Exit when spread mark decays to 50% of credit received
- Example: Received $0.80 credit → exit at $0.40 debit

### Stop Loss (SL)
| Parameter | Config Key | Default |
|-----------|------------|---------|
| SL Multiplier | `options.sl_mult_of_credit` | 2.0 |

- Exit when spread value reaches 200% of credit
- Example: Received $0.80 credit → exit at $1.60 debit (max loss = $0.80)

### Short Strike Breach
- If underlying price touches short strike intraday
- Reprice spread via Black-Scholes at breach moment
- Exit immediately

### End of Day (EOD)
| Parameter | Config Key | Default |
|-----------|------------|---------|
| Force Close Time | `strategy.eod_force_close` | 15:50 ET |

- Close any open position at 15:50 ET
- 0DTE options expire at market close (16:00 ET)

---

## Risk Management

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Max Loss/Trade | `risk.max_loss_per_trade_usd` | $500 | Skip trade if (width - credit) × 100 > this |
| Max Daily Loss | `risk.max_daily_loss_usd` | $1000 | Halt trading if cumulative loss exceeds |
| Max BP Usage | `risk.max_buying_power_pct` | 50% | Max % of buying power per trade |

---

## Backtest Data Sources

| Parameter | Config Key | Options | Description |
|-----------|------------|---------|-------------|
| Underlying Source | `options.underlying_source` | `es`, `spy` | Price source for option pricing |
| IV Source | `options.iv_source` | `fixed`, `vix` | IV source for Black-Scholes |
| Data Directory | `backtest.data_dir` | `data` | Location of SPY/VIX CSV files |

### Underlying Source
- **`es`**: Use ES futures price directly (introduces ~20-40pt basis error)
- **`spy`**: Use SPY × 10 as SPX proxy (more accurate for SPX options)

### IV Source
- **`fixed`**: Use `options.assumed_iv` (default 0.15)
- **`vix`**: Use daily VIX9D/VIX from market data (realistic)

---

## Optimization Grid

Parameters swept during walkforward optimization:

```python
DEFAULT_GRID = {
    "target_short_delta":  [0.10, 0.12, 0.16, 0.20, 0.25],
    "spread_width_pts":    [5, 10, 15],
    "tp_pct_of_credit":    [0.25, 0.50, 0.75],
    "min_orb_range_pts":   [2.0, 3.0, 5.0],
    "midline_tolerance":   [0.25, 0.50, 1.0],
}
```

**Note:** `assumed_iv` removed from grid — IV is now fetched from daily VIX data.

---

## Trade Flow Summary

```
08:00-08:15  Build ORB candle (high, low, midline)
             ↓
             Check range filter (skip if too narrow/wide)
             ↓
09:30+       Wait for breakout above high OR below low
             ↓
             Detect bias (LONG if break up, SHORT if break down)
             ↓
             Wait for midline retest
             ↓
             Select spread: PUT spread (LONG) or CALL spread (SHORT)
             ↓
             Price via Black-Scholes, check min credit
             ↓
             Enter trade
             ↓
             Monitor: TP (50%) | SL (200%) | Breach | EOD (15:50)
             ↓
             Exit and record P&L
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `config/config_8am_options.yaml` | Primary configuration |
| `src_8am/backtest/options_engine.py` | Backtest engine |
| `src_8am/backtest/optimizer.py` | Walkforward optimizer |
| `src_8am/backtest/market_data.py` | SPY/VIX data loader |
| `src_8am/backtest/data_loader.py` | ES bars loader (Databento) |
| `data/spy_daily.csv` | SPY OHLC (SPX proxy) |
| `data/vix_daily.csv` | VIX daily close |
| `data/vix9d_daily.csv` | VIX9D daily close (preferred) |
