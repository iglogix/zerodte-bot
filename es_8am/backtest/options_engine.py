"""
ORB Options Backtest Engine — simulates 0DTE credit spreads using ES/SPX bar data.

The engine handles common infrastructure (ORB detection, signal generation,
exit simulation, result tracking) while delegating strike selection and
pricing to a pluggable Strategy.

Built-in strategies:
  - DeltaStrategy (default): selects short strike at target BS delta
  - EMStrategy: selects short strike at N × expected move from spot
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

from es_8am.backtest.strategy_base import Strategy, TradeSetup

# Re-export pricing functions under their old underscore names for backward compat.
# Any external code that imported `_bs_call` etc. from this module will still work.
from es_8am.backtest.pricing import (
    ncdf as _ncdf, npdf as _npdf,
    bs_call as _bs_call, bs_put as _bs_put,
    bs_call_delta as _bs_call_delta, bs_put_delta as _bs_put_delta,
    bachelier_call as _bachelier_call, bachelier_put as _bachelier_put,
    bachelier_call_delta as _bachelier_call_delta, bachelier_put_delta as _bachelier_put_delta,
    select_strike as _select_strike,
    spread_credit as _spread_credit, spread_value as _spread_value,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptionsBacktestResult:
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
    spread_type: Optional[str] = None
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None
    spread_width: float = 5.0
    iv_assumed: float = 0.15

    # Entry
    entry_time: Optional[datetime] = None
    underlying_at_entry: Optional[float] = None
    credit_received: Optional[float] = None     # per-share (e.g. 0.75)
    max_profit_usd: Optional[float] = None       # credit * multiplier * qty
    max_loss_usd: Optional[float] = None         # (width - credit) * multiplier * qty
    short_delta_at_entry: Optional[float] = None

    # Exit
    exit_time: Optional[datetime] = None
    underlying_at_exit: Optional[float] = None
    debit_paid: Optional[float] = None           # cost to close
    exit_reason: Optional[str] = None            # expired_worthless|tp_50pct|sl_breach|eod_close|short_breached

    realized_pnl_usd: Optional[float] = None    # (credit - debit) * multiplier * qty

    status: str = "skipped"
    skip_reason: Optional[str] = None

    qty: int = 1


# ---------------------------------------------------------------------------
# Options Backtest Engine
# ---------------------------------------------------------------------------

class OrbOptionsBacktestEngine:
    """
    Backtests the ORB credit spread strategy using ES 1-min bar data.

    Strategy recap:
      LONG  bias → sell PUT  credit spread (short strike below ORB low)
      SHORT bias → sell CALL credit spread (short strike above ORB high)

    Exit logic (in priority order):
      1. EOD (15:50 ET): spread expires worthless → max profit.
      2. Intraday short-strike breach: compute spread value at breach bar → record loss/profit.
      3. TP at 50% of credit: detected by checking BS spread mark on each bar.
    """

    def __init__(self, config: Dict[str, Any], strategy: Optional[Strategy] = None):
        cfg = config

        # --- Pluggable strategy ---
        if strategy is not None:
            self._strategy = strategy
        else:
            strategy_type = cfg.get("options", {}).get("strategy_type", "delta")
            if strategy_type == "cbbo":
                from es_8am.backtest.strategy_cbbo import CbboDeltaStrategy
                self._strategy = CbboDeltaStrategy()
            elif strategy_type == "cbbo_em":
                from es_8am.backtest.strategy_cbbo_em import CbboEMStrategy
                self._strategy = CbboEMStrategy()
            elif strategy_type == "em":
                from es_8am.backtest.strategy_em import EMStrategy
                self._strategy = EMStrategy()
            else:
                from es_8am.backtest.strategy_delta import DeltaStrategy
                self._strategy = DeltaStrategy()
        self._strategy.configure(cfg)

        # --- CBBO store (real option quotes) ---
        self._cbbo_store = None
        self._last_bar_time: Optional[str] = None  # "HH:MM" for CBBO lookups
        cbbo_dir = cfg.get("options", {}).get("cbbo_dir", "")
        cbbo_cache = cfg.get("options", {}).get("cbbo_cache_dir", "")
        if cbbo_dir:
            from es_8am.backtest.cbbo_store import CbboStore
            self._cbbo_store = CbboStore(cbbo_dir, cache_dir=cbbo_cache)
            if hasattr(self._strategy, "set_cbbo_store"):
                self._strategy.set_cbbo_store(self._cbbo_store)

        strat = cfg.get("strategy", {})
        orb_start = strat.get("orb_window_start", "08:00")
        orb_end_s = strat.get("orb_window_end", "08:15")
        market_open_s = strat.get("market_open", "09:30")
        eod_s = strat.get("eod_force_close", "15:50")

        self._orb_start = time(*[int(x) for x in orb_start.split(":")])
        self._orb_end = time(*[int(x) for x in orb_end_s.split(":")])
        self._market_open = time(*[int(x) for x in market_open_s.split(":")])
        self._eod = time(*[int(x) for x in eod_s.split(":")])

        range_cfg = strat.get("orb_range_filter", {})
        self._min_range = range_cfg.get("min_pts", 2.0)
        self._max_range = range_cfg.get("max_pts", 30.0)

        self._midline_tol = strat.get("midline_tolerance_pts", 0.50)
        self._skip_nfp = strat.get("skip_nfp_fridays", False)
        self._skip_fomc = strat.get("skip_fomc_days", False)
        self._skip_cpi = strat.get("skip_cpi_days", False)

        opt = cfg.get("options", {})
        self._target_delta = opt.get("target_short_delta", 0.16)
        self._spread_width = opt.get("spread_width_pts", 5.0)
        self._strike_step = opt.get("strike_step_pts", 5.0)
        self._min_credit = opt.get("min_credit_per_spread", 0.50)
        self._tp_pct = opt.get("tp_pct_of_credit", 0.50)
        self._sl_mult = opt.get("sl_mult_of_credit", 0.0)  # 0 = disabled (exit on breach only)
        self._slippage = opt.get("slippage_per_spread", 0.0)  # deducted from credit at entry
        # Pricing model uncertainty: worst-case adjustment per leg (in points).
        # At entry: credit reduced by 2x (one per leg). At MTM: spread value inflated by 2x.
        # Based on ANN delta-zone MAE analysis (~$0.55 at 16-delta).
        self._pricing_uncertainty = opt.get("pricing_uncertainty_per_leg", 0.0)
        self._hold_put_breach = opt.get("hold_put_breach", False)  # True = don't exit on PUT breach
        self._iv = opt.get("assumed_iv", 0.15)

        # Spread mode: "directional" (PCS or CCS based on bias) or "iron_condor" (both sides)
        self._spread_mode = opt.get("spread_mode", "directional")
        # Pricing model: "bs" (Black-Scholes) or "bachelier" (normal model, better for 0DTE)
        self._pricing_model = opt.get("pricing_model", "bs")
        # IV correction factor: multiplier on IV to account for BS underpricing 0DTE options
        # 1.0 = no correction, >1.0 = inflate IV (more conservative backtest)
        self._iv_correction = opt.get("iv_correction_factor", 1.0)

        # Configurable data sources for more accurate backtesting
        # underlying_source: "bars" (use bar prices directly) or "daily_csv" (use SPY daily CSV * 10)
        # iv_source: "fixed" (use assumed_iv) or "vix" (use daily VIX9D/VIX) or "ann" (IV surface ANN)
        self._underlying_source = opt.get("underlying_source", "bars")
        self._iv_source = opt.get("iv_source", "fixed")

        # IV skew table: empirical moneyness → skew_ratio lookup from BBO data
        # Applies per-strike IV correction: effective_iv = atm_iv * skew_ratio(moneyness)
        skew_path = opt.get("iv_skew_table", "")
        self._skew_table: list[dict] | None = None
        if skew_path and Path(skew_path).exists():
            with open(skew_path) as f:
                data = json.load(f)
            self._skew_table = data.get("all", [])
            if self._skew_table:
                # Pre-sort by moneyness for binary search
                self._skew_table.sort(key=lambda r: r["moneyness_mid"])

        # IV surface ANN lookup table: precomputed IV = f(moneyness, TTE, is_call)
        # When iv_source="ann", loads JSON table (no PyTorch needed at runtime)
        self._ann_table: dict | None = None
        self._ann_tte_fine: list[float] = []
        self._ann_tte_coarse: list[float] = []
        self._ann_moneyness_vals: list[float] = []
        ann_table_path = opt.get("iv_ann_table", "") if self._iv_source == "ann" else ""
        if ann_table_path and Path(ann_table_path).exists():
            with open(ann_table_path) as f:
                ann_data = json.load(f)
            self._ann_table = ann_data.get("compact", {})
            # Build TTE buckets for nearest-neighbor lookup
            import numpy as np
            fine = ann_data.get("tte_fine_range", [0.0001, 0.05])
            fine_step = ann_data.get("tte_fine_step", 0.001)
            coarse = ann_data.get("tte_coarse_range", [0.05, 4.0])
            coarse_step = ann_data.get("tte_coarse_step", 0.05)
            self._ann_tte_fine = list(np.arange(fine[0], fine[1], fine_step))
            self._ann_tte_coarse = list(np.arange(coarse[0], coarse[1] + coarse_step, coarse_step))
            self._ann_tte_all = self._ann_tte_fine + self._ann_tte_coarse
            m_range = ann_data.get("moneyness_range", [0.9, 1.1])
            m_step = ann_data.get("moneyness_step", 0.001)
            self._ann_moneyness_vals = list(np.arange(m_range[0], m_range[1] + m_step, m_step))
            # When using ANN, skew table is redundant
            self._skew_table = None
        # Price ANN: direct option pricing via trained neural network
        # When iv_source="price_ann", bypasses BS/Bachelier for spread pricing
        self._price_ann = None
        self._price_ann_symbol: str = opt.get("price_ann_symbol", "IWM")
        if self._iv_source == "price_ann":
            ann_model_path = opt.get("price_ann_model", "models/price_ann_v2_otm_vol.pt")
            if Path(ann_model_path).exists():
                try:
                    from scripts.train_price_ann import PriceANN
                    self._price_ann = PriceANN.load(ann_model_path)
                    # Detect if model uses vol feature from checkpoint
                    import torch
                    ckpt = torch.load(ann_model_path, map_location="cpu", weights_only=False)
                    self._price_ann_use_vol = ckpt.get("use_vol", False)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("Failed to load price ANN: %s", e)

        self._option_root = opt.get("option_root", "")
        self._data_dir = cfg.get("backtest", {}).get("data_dir", "data")
        self._market_data = None  # Lazy-loaded

        self._multiplier = opt.get("contract_multiplier", 50)  # ES=$50, SPX=$100

        self._qty = opt.get("quantity", cfg.get("bot", {}).get("quantity", 1))

        risk = cfg.get("risk", {})
        self._max_loss_usd = risk.get("max_loss_per_trade_usd", 500)

        # Market constants for BS time-to-expiration calculations
        market = cfg.get("market", {})
        self._trading_days_year = market.get("trading_days_per_year", 252)
        self._trading_hours_day = market.get("trading_hours_per_day", 6.5)
        market_close_s = market.get("market_close", "16:00")
        self._market_close_time = time(*[int(x) for x in market_close_s.split(":")])
        self._session_end_minutes = self._trading_hours_day * 60

    # -----------------------------------------------------------------------
    # Market data helpers
    # -----------------------------------------------------------------------

    def _get_market_data(self):
        """Lazy-load market data loader for SPY/VIX data."""
        if self._market_data is None:
            from es_8am.backtest.market_data import MarketDataLoader
            self._market_data = MarketDataLoader(data_dir=self._data_dir)
        return self._market_data

    def _get_underlying_price(self, session_date: date, bar_price: float) -> float:
        """
        Get the underlying price to use for option pricing.

        If underlying_source is "daily_csv", fetches SPY daily close * 10 (SPX proxy).
        Otherwise ("bars") returns the bar price as-is.
        """
        if self._underlying_source == "daily_csv":
            md = self._get_market_data()
            spy_data = md.get_spy_ohlc(session_date)
            if spy_data:
                return spy_data["close"]  # Already scaled by 10
        return bar_price

    def _get_session_iv(self, session_date: date) -> float:
        """
        Get the IV to use for option pricing.

        If iv_source is "vix", returns daily VIX9D/VIX (as decimal).
        If iv_source is "realized", returns fixed IV (caller should use
        _compute_realized_vol with bars instead).
        Otherwise returns the fixed assumed_iv.
        """
        if self._iv_source == "vix":
            md = self._get_market_data()
            return md.get_daily_iv(session_date, fallback=self._iv)
        return self._iv

    @staticmethod
    def _compute_realized_vol(bars: list, annualize: bool = True) -> float:
        """Compute realized volatility from 1-min bar close prices.

        Uses log returns of consecutive closes, then annualizes assuming
        390 trading minutes per day and 252 trading days per year.

        Args:
            bars: List of bar dicts with 'close' key.
            annualize: If True, return annualized vol. If False, return per-minute stdev.

        Returns:
            Annualized realized volatility (decimal, e.g. 0.18 for 18%).
            Returns 0.15 fallback if fewer than 10 bars.
        """
        if len(bars) < 10:
            return 0.15  # not enough data, use fallback

        closes = [b["close"] for b in bars]
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i] > 0 and closes[i - 1] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))

        if len(log_returns) < 5:
            return 0.15

        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        stdev = math.sqrt(variance)

        if not annualize:
            return stdev

        # Annualize: stdev_per_minute * sqrt(minutes_per_day * trading_days_per_year)
        # 390 minutes per trading day (6.5 hours), 252 trading days
        return stdev * math.sqrt(390 * 252)

    def _get_strike_iv(self, base_iv: float, s: float, k: float,
                       t: float, is_call: int) -> float:
        """Get IV for a specific strike, using ANN table, skew table, or flat IV.

        Priority: ANN table > skew table > flat base_iv.
        """
        if self._ann_table is not None:
            return self._ann_lookup(s / k, t, is_call)
        return self._skew_iv(base_iv, s, k)

    def _ann_lookup(self, moneyness: float, tte: float, is_call: int) -> float:
        """Look up IV from precomputed ANN table (nearest-neighbor)."""
        opt_key = "call" if is_call else "put"
        sub = self._ann_table.get(opt_key, {})
        if not sub:
            return 0.15  # fallback

        # Snap moneyness to nearest 0.001
        m_snap = round(max(0.900, min(1.100, moneyness)), 3)

        # Snap TTE to nearest bucket
        best_tte = self._ann_tte_all[0]
        best_dist = abs(best_tte - tte)
        for tv in self._ann_tte_all:
            d = abs(tv - tte)
            if d < best_dist:
                best_dist = d
                best_tte = tv

        key = f"{m_snap:.3f}_{best_tte:.4f}"
        iv = sub.get(key)
        if iv is not None:
            return iv

        # Fallback: try adjacent moneyness
        for offset in [0.001, -0.001, 0.002, -0.002]:
            alt_key = f"{m_snap + offset:.3f}_{best_tte:.4f}"
            iv = sub.get(alt_key)
            if iv is not None:
                return iv
        return 0.15  # ultimate fallback

    def _skew_iv(self, base_iv: float, s: float, k: float) -> float:
        """Apply empirical IV skew correction based on moneyness.

        Returns base_iv * skew_ratio(S/K) if skew table is loaded,
        otherwise returns base_iv unchanged.
        """
        if self._skew_table is None:
            return base_iv
        moneyness = s / k
        # Find closest bin by moneyness
        best = self._skew_table[0]
        best_dist = abs(best["moneyness_mid"] - moneyness)
        for row in self._skew_table:
            dist = abs(row["moneyness_mid"] - moneyness)
            if dist < best_dist:
                best_dist = dist
                best = row
        return base_iv * best["skew_ratio_median"]

    def _ann_option_price(self, spot: float, strike: float, tte_hours: float,
                          is_call: bool, vol: float = 0.0) -> float:
        """Price a single option using the trained ANN model.

        Returns option mid price in points (not normalized).
        """
        if self._price_ann is None:
            return 0.0
        import math
        import numpy as np
        from scripts.train_price_ann import SYMBOL_ORDER, N_BASE_FEATURES

        m = strike / spot
        sym = self._price_ann_symbol
        sym_idx = SYMBOL_ORDER.index(sym) if sym in SYMBOL_ORDER else 0

        use_vol = getattr(self, "_price_ann_use_vol", False)
        n_symbols = len(SYMBOL_ORDER)
        base = N_BASE_FEATURES + (1 if use_vol else 0)
        n_feat = base + n_symbols

        x = np.zeros((1, n_feat), dtype=np.float32)
        x[0, 0] = m
        x[0, 1] = math.log(m) if m > 0 else 0
        x[0, 2] = tte_hours
        x[0, 3] = math.sqrt(tte_hours) if tte_hours > 0 else 0
        x[0, 4] = 1.0 if is_call else 0.0
        if use_vol:
            x[0, 5] = vol
        x[0, base + sym_idx] = 1.0

        norm_price = float(self._price_ann.predict(x)[0])
        return max(0.0, norm_price * spot)

    def _ann_spread_credit(self, spot: float, short_k: float, long_k: float,
                           tte_hours: float, spread_type: str, vol: float = 0.0) -> float:
        """Compute spread credit using ANN option prices.

        Applies pricing_uncertainty as worst-case: short leg priced lower,
        long leg priced higher -> less credit received.
        """
        is_call = spread_type == "call_credit_spread"
        short_price = self._ann_option_price(spot, short_k, tte_hours, is_call, vol)
        long_price = self._ann_option_price(spot, long_k, tte_hours, is_call, vol)
        unc = self._pricing_uncertainty
        return max(0.0, (short_price - unc) - (long_price + unc))

    def _ann_spread_value(self, spot: float, short_k: float, long_k: float,
                          tte_hours: float, spread_type: str, vol: float = 0.0) -> float:
        """Current mark-to-market spread value using ANN.

        Applies pricing_uncertainty as worst-case: costs more to close
        (short leg higher, long leg lower -> wider debit).
        """
        is_call = spread_type == "call_credit_spread"
        short_price = self._ann_option_price(spot, short_k, tte_hours, is_call, vol)
        long_price = self._ann_option_price(spot, long_k, tte_hours, is_call, vol)
        unc = self._pricing_uncertainty
        return max(0.0, (short_price + unc) - (long_price - unc))

    def _is_nfp_friday(self, session_date: date) -> bool:
        """First Friday of the month — NFP release at 8:30 AM ET."""
        if session_date.weekday() != 4:  # Not Friday
            return False
        return session_date.day <= 7

    # FOMC announcement dates (statement release day, 2:00 PM ET)
    _FOMC_DATES: set[date] = {
        # 2021
        date(2021,1,27), date(2021,3,17), date(2021,4,28), date(2021,6,16),
        date(2021,7,28), date(2021,9,22), date(2021,11,3), date(2021,12,15),
        # 2022
        date(2022,1,26), date(2022,3,16), date(2022,5,4), date(2022,6,15),
        date(2022,7,27), date(2022,9,21), date(2022,11,2), date(2022,12,14),
        # 2023
        date(2023,2,1), date(2023,3,22), date(2023,5,3), date(2023,6,14),
        date(2023,7,26), date(2023,9,20), date(2023,11,1), date(2023,12,13),
        # 2024
        date(2024,1,31), date(2024,3,20), date(2024,5,1), date(2024,6,12),
        date(2024,7,31), date(2024,9,18), date(2024,11,7), date(2024,12,18),
        # 2025
        date(2025,1,29), date(2025,3,19), date(2025,5,7), date(2025,6,18),
        date(2025,7,30), date(2025,9,17), date(2025,10,29), date(2025,12,10),
        # 2026
        date(2026,1,28), date(2026,3,18), date(2026,4,29), date(2026,6,17),
        date(2026,7,29), date(2026,9,16), date(2026,10,28), date(2026,12,9),
    }

    # CPI release dates (8:30 AM ET)
    _CPI_DATES: set[date] = {
        # 2021
        date(2021,1,13), date(2021,2,10), date(2021,3,10), date(2021,4,13),
        date(2021,5,12), date(2021,6,10), date(2021,7,13), date(2021,8,11),
        date(2021,9,14), date(2021,10,13), date(2021,11,10), date(2021,12,10),
        # 2022
        date(2022,1,12), date(2022,2,10), date(2022,3,10), date(2022,4,12),
        date(2022,5,11), date(2022,6,10), date(2022,7,13), date(2022,8,10),
        date(2022,9,13), date(2022,10,13), date(2022,11,10), date(2022,12,13),
        # 2023
        date(2023,1,12), date(2023,2,14), date(2023,3,14), date(2023,4,12),
        date(2023,5,10), date(2023,6,13), date(2023,7,12), date(2023,8,10),
        date(2023,9,13), date(2023,10,12), date(2023,11,14), date(2023,12,12),
        # 2024
        date(2024,1,11), date(2024,2,13), date(2024,3,12), date(2024,4,10),
        date(2024,5,15), date(2024,6,12), date(2024,7,11), date(2024,8,14),
        date(2024,9,11), date(2024,10,10), date(2024,11,13), date(2024,12,11),
        # 2025
        date(2025,1,15), date(2025,2,12), date(2025,3,12), date(2025,4,10),
        date(2025,5,13), date(2025,6,11), date(2025,7,15), date(2025,8,12),
        date(2025,9,11), date(2025,10,24), date(2025,12,18),
        # 2026
        date(2026,1,13), date(2026,2,13), date(2026,3,11), date(2026,4,10),
        date(2026,5,12), date(2026,6,10), date(2026,7,14), date(2026,8,12),
        date(2026,9,11), date(2026,10,14), date(2026,11,10), date(2026,12,10),
    }

    def _is_fomc_day(self, session_date: date) -> bool:
        return session_date in self._FOMC_DATES

    def _is_cpi_day(self, session_date: date) -> bool:
        return session_date in self._CPI_DATES

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        zip_path: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[OptionsBacktestResult]:
        from es_8am.backtest.data_loader import DatabentoLoader
        loader = DatabentoLoader(zip_path)
        days = list(loader.iter_days(start_date=start_date, end_date=end_date))
        return self.run_on_days(days)

    def run_on_days(self, days) -> List[OptionsBacktestResult]:
        """Run backtest on a pre-loaded list of DayBars (avoids re-parsing zip)."""
        # Group by date and select highest-volume contract per day
        # This handles roll periods where multiple contracts trade simultaneously
        from collections import defaultdict

        by_date: Dict[date, List] = defaultdict(list)
        for day_bars in days:
            by_date[day_bars.session_date].append(day_bars)

        results = []
        # Rolling buffer of prior days' bars for realized vol (when iv_source="realized")
        from collections import deque
        _realized_lookback = 5  # number of prior trading days
        prior_days_bars: deque = deque(maxlen=_realized_lookback)

        for session_date in sorted(by_date.keys()):
            # Skip NFP Fridays (first Friday of month) - high loss rate due to 8:30 AM release
            if self._skip_nfp and self._is_nfp_friday(session_date):
                continue
            # Skip FOMC announcement days — elevated vol after 2 PM statement
            if self._skip_fomc and self._is_fomc_day(session_date):
                continue
            # Skip CPI release days — 8:30 AM release causes gap moves through ORB
            if self._skip_cpi and self._is_cpi_day(session_date):
                continue

            candidates = by_date[session_date]

            # Select contract with highest volume (most liquid)
            best_day = max(candidates, key=lambda d: sum(b["volume"] for b in d.bars))

            # Skip AM expiration contracts (3rd Friday, contract expires at 9:30 AM)
            # These have symbol like ESH4 expiring in March - check if bars end early
            if self._is_am_expiration(best_day):
                # Try to use next contract instead
                non_expiring = [d for d in candidates if not self._is_am_expiration(d)]
                if non_expiring:
                    best_day = max(non_expiring, key=lambda d: sum(b["volume"] for b in d.bars))
                else:
                    continue  # Skip this day entirely

            day_results = self._process_day(best_day, prior_days_bars=prior_days_bars)
            if day_results is not None:
                if isinstance(day_results, list):
                    results.extend(r for r in day_results if r is not None)
                else:
                    results.append(day_results)

            # Add this day's bars to the rolling buffer (regardless of trade outcome)
            prior_days_bars.append(best_day.bars)

        return results

    def _is_am_expiration(self, day_bars) -> bool:
        """
        Check if this is an AM expiration day for this contract.

        ES futures expire at 9:30 AM ET on the 3rd Friday of contract month.
        We detect this by checking if the contract's bars end before market hours.
        """
        if not day_bars.bars:
            return False

        symbol = day_bars.symbol
        session_date = day_bars.session_date

        # ES contract months: H=Mar, M=Jun, U=Sep, Z=Dec
        contract_months = {"H": 3, "M": 6, "U": 9, "Z": 12}

        # Extract contract month from symbol (e.g., ESH4 -> H)
        if len(symbol) >= 3 and symbol[2] in contract_months:
            contract_month = contract_months[symbol[2]]

            # Check if this is the expiration month
            if session_date.month == contract_month:
                # 3rd Friday check: day 15-21 and Friday
                if 15 <= session_date.day <= 21 and session_date.weekday() == 4:
                    # Verify bars end early (before 10:00 AM typical for AM exp)
                    last_bar_time = day_bars.bars[-1]["ts_et"].time()
                    if last_bar_time.hour < 10:
                        return True

        return False

    # -----------------------------------------------------------------------
    # Day processing
    # -----------------------------------------------------------------------

    def _process_day(self, day_bars, prior_days_bars=None) -> Optional[OptionsBacktestResult]:
        from es_8am.backtest.data_loader import DayBars
        bars: List[Dict] = day_bars.bars
        session_date: date = day_bars.session_date
        symbol: str = day_bars.symbol

        # Load CBBO quotes for this day if available
        if self._cbbo_store is not None:
            n_quotes = self._cbbo_store.load_day(session_date)
            if n_quotes == 0:
                return None  # Skip days without CBBO data

        # --- Step 1: Build ORB candle ---
        orb_bars = [b for b in bars if self._orb_start <= b["ts_et"].time() < self._orb_end]
        if not orb_bars:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=0, orb_low=0, orb_midline=0, orb_range_pts=0,
                status="skipped", skip_reason="no_orb_data",
            )

        orb_high = max(b["high"] for b in orb_bars)
        orb_low = min(b["low"] for b in orb_bars)
        orb_midline = (orb_high + orb_low) / 2
        orb_range = round(orb_high - orb_low, 4)

        if orb_range < self._min_range:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason=f"range_too_narrow_{orb_range}pts",
            )
        if orb_range > self._max_range:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason=f"range_too_wide_{orb_range}pts",
            )

        # --- Step 2: Post-open bars ---
        post_open_bars = [b for b in bars if b["ts_et"].time() >= self._market_open]
        if not post_open_bars:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason="no_post_open_bars",
            )

        # --- Step 3: Detect breakout ---
        breakout_bar = None
        bias = None
        for bar in post_open_bars:
            if bar["ts_et"].time() >= self._eod:
                break
            if bar["close"] > orb_high:
                bias = "long"
                breakout_bar = bar
                break
            if bar["close"] < orb_low:
                bias = "short"
                breakout_bar = bar
                break

        if breakout_bar is None:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                status="skipped", skip_reason="no_breakout",
            )

        post_breakout_bars = post_open_bars[post_open_bars.index(breakout_bar):]

        # --- Step 4: Midline retest ---
        retest_bar = None
        for bar in post_breakout_bars:
            if bar["ts_et"].time() >= self._eod:
                break
            if bar["low"] <= orb_midline + self._midline_tol and bar["high"] >= orb_midline - self._midline_tol:
                retest_bar = bar
                break

        if retest_bar is None:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                status="skipped", skip_reason="no_retest",
            )

        # --- Step 5: Fill on next bar's open ---
        retest_idx = post_breakout_bars.index(retest_bar)
        post_retest = post_breakout_bars[retest_idx + 1:]
        if not post_retest:
            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-SKIP",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                status="skipped", skip_reason="no_bar_after_retest",
            )

        fill_bar = post_retest[0]
        es_entry_price = fill_bar["open"]  # ES price from bars (used for ORB logic)
        entry_time = fill_bar["ts_et"]

        # --- Step 6: Delegate to strategy for strike selection + pricing ---
        option_underlying = self._get_underlying_price(session_date, es_entry_price)

        # For index options with CBBO: derive true spot from put-call parity
        if self._option_root and self._cbbo_store is not None:
            minute_et = entry_time.strftime("%H:%M")
            derived = self._cbbo_store.derive_spot(minute_et, option_underlying)
            if derived is not None:
                option_underlying = derived
        # Get session IV
        if self._iv_source == "realized":
            lookback_bars = []
            if prior_days_bars:
                for day in prior_days_bars:
                    lookback_bars.extend(day)
            if len(lookback_bars) < 30:
                session_iv = self._iv
            else:
                session_iv = self._compute_realized_vol(lookback_bars)
        else:
            session_iv = self._get_session_iv(session_date)

        tte_hours_entry = self._tte_hours(entry_time)
        t_entry = tte_hours_entry / (self._trading_days_year * self._trading_hours_day)
        iv_adj = session_iv * self._iv_correction

        # --- Helper: execute one leg (select + simulate) ---
        def _run_leg(leg_bias: str, tag: str = "") -> Optional[OptionsBacktestResult]:
            setup = self._strategy.select_trade(
                spot=option_underlying,
                bias=leg_bias,
                entry_time=entry_time,
                session_iv=session_iv,
                tte_years=t_entry,
                tte_hours=tte_hours_entry,
                engine=self,
            )

            if setup is None:
                return OptionsBacktestResult(
                    trade_id=f"OPT-{session_date}-SKIP{tag}",
                    session_date=session_date, symbol=symbol,
                    orb_high=orb_high, orb_low=orb_low,
                    orb_midline=orb_midline, orb_range_pts=orb_range,
                    bias=leg_bias,
                    breakout_time=breakout_bar["ts_et"],
                    breakout_price=breakout_bar["close"],
                    underlying_at_entry=option_underlying,
                    iv_assumed=session_iv,
                    status="skipped", skip_reason="strategy_rejected",
                )

            credit = setup.credit
            max_profit = round(credit * self._multiplier * self._qty, 2)
            max_loss = round((setup.spread_width - credit) * self._multiplier * self._qty, 2)

            if max_loss > self._max_loss_usd:
                return OptionsBacktestResult(
                    trade_id=f"OPT-{session_date}-SKIP{tag}",
                    session_date=session_date, symbol=symbol,
                    orb_high=orb_high, orb_low=orb_low,
                    orb_midline=orb_midline, orb_range_pts=orb_range,
                    bias=leg_bias,
                    breakout_time=breakout_bar["ts_et"],
                    breakout_price=breakout_bar["close"],
                    spread_type=setup.spread_type,
                    short_strike=setup.short_strike, long_strike=setup.long_strike,
                    spread_width=setup.spread_width, iv_assumed=session_iv,
                    underlying_at_entry=option_underlying,
                    credit_received=credit, max_profit_usd=max_profit, max_loss_usd=max_loss,
                    status="skipped", skip_reason="risk_limit",
                )

            tp_debit = round(credit * (1 - self._tp_pct), 2)
            exit_time, exit_price, exit_reason, debit = self._simulate_options_exit(
                bars=post_retest[1:],
                fill_bar=fill_bar,
                spread_type=setup.spread_type,
                short_k=setup.short_strike,
                long_k=setup.long_strike,
                credit=credit,
                tp_debit=tp_debit,
                entry_time=entry_time,
                session_iv=iv_adj,
            )

            pnl = round((credit - debit) * self._multiplier * self._qty, 2)

            return OptionsBacktestResult(
                trade_id=f"OPT-{session_date}-{uuid.uuid4().hex[:6].upper()}{tag}",
                session_date=session_date, symbol=symbol,
                orb_high=orb_high, orb_low=orb_low,
                orb_midline=orb_midline, orb_range_pts=orb_range,
                bias=leg_bias,
                breakout_time=breakout_bar["ts_et"],
                breakout_price=breakout_bar["close"],
                spread_type=setup.spread_type,
                short_strike=setup.short_strike, long_strike=setup.long_strike,
                spread_width=setup.spread_width, iv_assumed=session_iv,
                entry_time=entry_time,
                underlying_at_entry=option_underlying,
                credit_received=credit,
                max_profit_usd=max_profit,
                max_loss_usd=max_loss,
                short_delta_at_entry=round(setup.short_delta, 4),
                exit_time=exit_time,
                underlying_at_exit=exit_price,
                debit_paid=debit,
                exit_reason=exit_reason,
                realized_pnl_usd=pnl,
                status="closed",
                qty=self._qty,
            )

        # --- Step 6+7: Execute trade(s) ---
        if self._spread_mode == "iron_condor":
            # IC: place both PCS and CCS; pass actual ORB bias for adaptive strategies
            self._orb_bias = bias  # store for strategy to read
            put_result = _run_leg("long", "-P")    # long bias -> put credit spread
            call_result = _run_leg("short", "-C")   # short bias -> call credit spread
            return [put_result, call_result]
        else:
            # Directional: single leg based on breakout bias
            return _run_leg(bias)

    # -----------------------------------------------------------------------
    # Options exit simulation
    # -----------------------------------------------------------------------

    def _simulate_options_exit(
        self,
        bars: List[Dict],
        fill_bar: Dict,
        spread_type: str,
        short_k: float,
        long_k: float,
        credit: float,
        tp_debit: float,
        entry_time: datetime,
        session_iv: float,
    ) -> Tuple[datetime, float, str, float]:
        """
        Walk bars after fill, checking in order:
          1. Short strike breached intrabar → reprice spread at breach → exit.
          2. SL: spread mark >= sl_mult × credit → exit (cut loss before full breach).
          3. TP: spread mark <= tp_debit → exit at tp_debit.
          4. EOD bar → expire worthless.

        Returns (exit_time, underlying_price, exit_reason, debit_paid).
        """
        sl_debit = round(credit * self._sl_mult, 2) if self._sl_mult > 0 else 0.0

        # Check fill bar itself for immediate breach
        check_bars = [fill_bar] + list(bars)

        for bar in check_bars:
            ts = bar["ts_et"]
            if ts.time() >= self._eod:
                return ts, bar["close"], "expired_worthless", 0.0

            # Set current bar time for CBBO strategy lookups
            self._last_bar_time = ts.strftime("%H:%M")

            t_now = self._tte_hours(ts) / (self._trading_days_year * self._trading_hours_day)
            t_now = max(t_now, 1e-6)
            tte_hours_now = self._tte_hours(ts)

            # Check short strike breach
            breached, breach_price = self._check_breach(bar, spread_type, short_k)
            if breached:
                if self._hold_put_breach and spread_type == "put_credit_spread":
                    continue
                debit = self._strategy.compute_exit_value(
                    breach_price, short_k, long_k, t_now, tte_hours_now,
                    spread_type, session_iv, self,
                )
                debit = round(min(debit, self._spread_width), 2)
                return ts, breach_price, "short_breached", debit

            # Mark-to-market
            mark = self._strategy.compute_exit_value(
                bar["close"], short_k, long_k, t_now, tte_hours_now,
                spread_type, session_iv, self,
            )

            # SL check
            if sl_debit > 0 and mark >= sl_debit:
                sl_label = f"sl_{int(self._sl_mult * 100)}pct"
                return ts, bar["close"], sl_label, round(mark, 2)

            # TP check
            if mark <= tp_debit:
                tp_label = f"tp_{int(self._tp_pct * 100)}pct"
                return ts, bar["close"], tp_label, round(tp_debit, 2)

        # No exit found — expire worthless at EOD
        last = bars[-1] if bars else fill_bar
        return last["ts_et"], last["close"], "expired_worthless", 0.0

    def _check_breach(
        self, bar: Dict, spread_type: str, short_k: float
    ) -> Tuple[bool, float]:
        """
        Returns (breached, breach_price) where breach_price is the short strike
        (worst-case assumption: touched exactly).
        """
        if spread_type == "put_credit_spread":
            if bar["low"] <= short_k:
                return True, short_k
        else:
            if bar["high"] >= short_k:
                return True, short_k
        return False, 0.0

    # -----------------------------------------------------------------------
    # Time-to-expiry helpers
    # -----------------------------------------------------------------------

    def _tte_hours(self, ts: datetime) -> float:
        """Trading hours remaining until market close on the same day."""
        close_dt = ts.replace(
            hour=self._market_close_time.hour,
            minute=self._market_close_time.minute,
            second=0,
            microsecond=0
        )
        if ts.tzinfo is None:
            close_dt = close_dt.replace(tzinfo=ET)
            ts = ts.replace(tzinfo=ET)
        remaining = (close_dt - ts).total_seconds() / 3600.0
        return max(remaining, 0.0)
