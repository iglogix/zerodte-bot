"""
METF (Multiple Entry Trend Following) Backtester

Backtests the METF 0DTE SPX strategy using historical 1-minute bar data.
Simulates credit spread outcomes based on EMA trend and entry times.

Features:
- EMA-based trend detection (20/40 EMA)
- Scheduled entries at specific times
- Credit spread simulation with 1x stop loss
- FOMC day filtering
- Performance metrics and reporting

Author: METF Backtester v1.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CBBO Pricing Helper
# =============================================================================

class CbboPricer:
    """
    Real-pricing helper that wraps CbboStore for METF spread quotes.

    One instance per backtest run. Call `load_day()` at the start of each
    trading day; if it returns False the day must be skipped.
    """

    def __init__(self, cbbo_dir: str):
        from src_v2.backtest.cbbo_store import CbboStore
        self._store = CbboStore(opra_dir=cbbo_dir, cache_dir=cbbo_dir)
        self.cbbo_dir = cbbo_dir

    def load_day(self, trading_date: date) -> bool:
        """Load a day's quotes. Returns True if quotes are available."""
        n = self._store.load_day(trading_date)
        return n > 0

    @property
    def available_strikes(self):
        return self._store.available_strikes

    def nearest_strike(self, target: float, available: Optional[set] = None) -> Optional[float]:
        """Snap target price to the nearest available strike on the loaded day."""
        strikes = available if available is not None else self._store.available_strikes
        if not strikes:
            return None
        return min(strikes, key=lambda k: abs(k - target))

    def lookup(self, minute_et: str, strike: float, is_call: bool):
        """Return (bid, ask) or None if quote unavailable / stale > 5 min."""
        return self._store.lookup(minute_et, strike, is_call)

    def entry_credit(
        self,
        minute_et: str,
        short_strike: float,
        long_strike: float,
        is_call: bool,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Conservative credit at entry: sell short @ bid, buy long @ ask.
        Returns (credit, short_bid, long_ask) or None if either quote is missing.
        """
        short_q = self.lookup(minute_et, short_strike, is_call)
        long_q = self.lookup(minute_et, long_strike, is_call)
        if short_q is None or long_q is None:
            return None
        short_bid, _ = short_q
        _, long_ask = long_q
        credit = short_bid - long_ask
        return credit, short_bid, long_ask

    def close_mark(
        self,
        minute_et: str,
        short_strike: float,
        long_strike: float,
        is_call: bool,
    ) -> Optional[float]:
        """
        Conservative cost-to-close mark: buy short @ ask, sell long @ bid.
        Returns None if either quote is unavailable.
        """
        short_q = self.lookup(minute_et, short_strike, is_call)
        long_q = self.lookup(minute_et, long_strike, is_call)
        if short_q is None or long_q is None:
            return None
        _, short_ask = short_q
        long_bid, _ = long_q
        return short_ask - long_bid


# =============================================================================
# Configuration Classes
# =============================================================================

class SpreadType(Enum):
    """Type of credit spread"""
    PUT_CREDIT_SPREAD = "PCS"
    CALL_CREDIT_SPREAD = "CCS"


class ExitReason(Enum):
    """Reason for position exit"""
    EXPIRATION_WIN = "EXPIRATION_WIN"
    EXPIRATION_LOSS = "EXPIRATION_LOSS"
    STOP_LOSS = "STOP_LOSS"
    NO_ENTRY = "NO_ENTRY"


class TrendDirection(Enum):
    """Market trend based on EMA relationship"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class METFBacktestConfig:
    """Configuration for METF backtester"""
    # EMA settings
    ema_fast_period: int = 20
    ema_slow_period: int = 40
    ema_min_separation_pct: float = 0.001
    
    # Entry times (HH:MM format)
    entry_times: List[str] = field(default_factory=lambda: [
        "12:30", "13:00", "13:30", "14:00", "14:30", "14:45"
    ])
    
    # Spread settings
    spread_widths: List[int] = field(default_factory=lambda: [20, 30, 35])
    default_width: int = 20
    min_credit: float = 1.25
    max_credit: float = 2.50
    default_credit: float = 1.75  # Used for simulation (ignored when use_em_strikes=True)
    use_em_strikes: bool = True  # Place short strike at Expected Move boundary
    em_method: str = "historical_vol"  # "historical_vol", "atr", or "fixed_pct"
    em_fixed_pct: float = 0.01  # 1% EM if using fixed_pct method
    em_lookback: int = 20  # Lookback period for historical vol calculation
    
    # Risk settings
    stop_loss_multiplier: float = 1.0
    
    # Market hours
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    
    # Filters
    fomc_rate_days: List[str] = field(default_factory=list)
    fomc_minutes_days: List[str] = field(default_factory=list)
    
    # Data settings
    symbol: str = "SPX"
    timezone: str = "America/New_York"

    # CBBO real-pricing mode (replaces synthetic credit + heuristic exit walk)
    use_cbbo: bool = False
    cbbo_dir: str = "data/cbbo_cache"
    bar_csv_path: str = "data/spy_1m.csv"
    bar_price_scale: float = 10.0  # SPY * 10 ≈ SPX
    quote_max_stale_min: int = 5
    
    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)
    
    def get_entry_times(self) -> List[time]:
        """Parse entry times to time objects"""
        times = []
        for t in self.entry_times:
            h, m = map(int, t.split(":"))
            times.append(time(h, m))
        return times


@dataclass
class TradeResult:
    """Result of a single trade"""
    trade_date: str
    entry_time: datetime
    entry_number: int
    trend: TrendDirection
    spread_type: SpreadType
    underlying_at_entry: float
    ema_fast: float
    ema_slow: float
    credit: float
    width: int
    exit_time: datetime
    exit_reason: ExitReason
    pnl: float
    max_adverse_excursion: float  # Maximum loss during trade
    # CBBO-mode fields (None in synthetic mode)
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None
    short_bid_at_entry: Optional[float] = None
    long_ask_at_entry: Optional[float] = None
    exit_mark: Optional[float] = None
    underlying_at_exit: Optional[float] = None

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class DailyResult:
    """Results for a single trading day"""
    trade_date: str
    trades: List[TradeResult]
    total_pnl: float
    num_winners: int
    num_losers: int
    is_fomc_day: bool
    skip_reasons: Dict[str, int] = field(default_factory=dict)  # per-entry skip counts

    @property
    def win_rate(self) -> float:
        total = self.num_winners + self.num_losers
        return self.num_winners / total if total > 0 else 0.0


@dataclass
class BacktestResults:
    """Complete backtest results"""
    config: METFBacktestConfig
    start_date: str
    end_date: str
    daily_results: List[DailyResult]
    skipped_days_no_cbbo: List[str] = field(default_factory=list)

    @property
    def aggregate_skip_reasons(self) -> Dict[str, int]:
        agg: Dict[str, int] = {}
        for d in self.daily_results:
            for k, v in d.skip_reasons.items():
                agg[k] = agg.get(k, 0) + v
        return agg

    @property
    def total_trades(self) -> int:
        return sum(len(d.trades) for d in self.daily_results)
    
    @property
    def total_pnl(self) -> float:
        return sum(d.total_pnl for d in self.daily_results)
    
    @property
    def total_winners(self) -> int:
        return sum(d.num_winners for d in self.daily_results)
    
    @property
    def total_losers(self) -> int:
        return sum(d.num_losers for d in self.daily_results)
    
    @property
    def win_rate(self) -> float:
        total = self.total_winners + self.total_losers
        return self.total_winners / total if total > 0 else 0.0
    
    @property
    def avg_winner(self) -> float:
        winners = [t.pnl for d in self.daily_results for t in d.trades if t.is_winner]
        return np.mean(winners) if winners else 0.0
    
    @property
    def avg_loser(self) -> float:
        losers = [t.pnl for d in self.daily_results for t in d.trades if not t.is_winner]
        return np.mean(losers) if losers else 0.0
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for d in self.daily_results for t in d.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for d in self.daily_results for t in d.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for daily in self.daily_results:
            cumulative += daily.total_pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (annualized)"""
        daily_pnls = [d.total_pnl for d in self.daily_results]
        if len(daily_pnls) < 2:
            return 0.0
        
        mean_daily = np.mean(daily_pnls)
        std_daily = np.std(daily_pnls)
        
        if std_daily == 0:
            return 0.0
        
        # Annualize (252 trading days)
        return (mean_daily / std_daily) * np.sqrt(252)


# =============================================================================
# Core Backtester
# =============================================================================

class METFBacktester:
    """
    METF Strategy Backtester
    
    Tests the METF strategy on historical 1-minute bar data.
    """
    
    def __init__(
        self,
        config: Optional[METFBacktestConfig] = None,
        pricer: Optional[CbboPricer] = None,
    ):
        self.config = config or METFBacktestConfig()
        self.logger = logging.getLogger(__name__)
        self._pricer = pricer
        if self.config.use_cbbo and self._pricer is None:
            self._pricer = CbboPricer(self.config.cbbo_dir)
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def get_trend(self, ema_fast: float, ema_slow: float) -> TrendDirection:
        """Determine trend from EMA values"""
        separation_pct = (ema_fast - ema_slow) / ema_slow * 100
        min_sep = self.config.ema_min_separation_pct * 100
        
        if separation_pct > min_sep:
            return TrendDirection.BULLISH
        elif separation_pct < -min_sep:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL
    
    def is_fomc_day(self, trade_date: str) -> bool:
        """Check if date is an FOMC day"""
        return (trade_date in self.config.fomc_rate_days or 
                trade_date in self.config.fomc_minutes_days)
    
    def calculate_expected_move(self, bars: pd.DataFrame, current_idx: int) -> float:
        """
        Calculate Expected Move (EM) for strike placement.
        
        Methods:
        - historical_vol: Use historical volatility scaled to intraday
        - atr: Use Average True Range
        - fixed_pct: Use fixed percentage of underlying
        
        Returns:
            Expected move in points (e.g., 45 for SPY at 600 with 0.75% EM)
        """
        if current_idx < self.config.em_lookback:
            # Not enough data, use fixed percentage
            current_price = bars.iloc[current_idx]['close']
            return current_price * self.config.em_fixed_pct
        
        lookback_bars = bars.iloc[max(0, current_idx - self.config.em_lookback):current_idx]
        current_price = bars.iloc[current_idx]['close']
        
        if self.config.em_method == "fixed_pct":
            return current_price * self.config.em_fixed_pct
        
        elif self.config.em_method == "atr":
            # Calculate ATR
            high = lookback_bars['high']
            low = lookback_bars['low']
            close = lookback_bars['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.mean()
            
            # Scale ATR for remaining trading time (afternoon entries = ~3-4 hours left)
            # Full day ATR covers ~6.5 hours, so scale by sqrt(remaining/total)
            time_scale = np.sqrt(3.5 / 6.5)
            return atr * time_scale
        
        else:  # historical_vol
            # Calculate historical volatility from returns
            returns = lookback_bars['close'].pct_change().dropna()
            if len(returns) < 2:
                return current_price * self.config.em_fixed_pct
            
            # For 0DTE options, EM is typically calculated from ATM straddle
            # which represents ~1 standard deviation move for remaining time
            # 
            # Typical 0DTE EM for SPY/SPX:
            # - Low vol day: 0.3-0.5% (~$1.5-2.5 for SPY at $500)
            # - Normal day: 0.5-0.8% (~$2.5-4 for SPY at $500)
            # - High vol day: 0.8-1.2% (~$4-6 for SPY at $500)
            #
            # Calculate intraday vol and scale to remaining time
            vol_1min = returns.std()
            
            # Remaining bars from afternoon entry (~3.5 hours = 210 bars)
            # Scale by sqrt(time) for volatility
            remaining_bars = 210
            remaining_vol = vol_1min * np.sqrt(remaining_bars)
            
            # EM is typically 1 standard deviation, but cap it to realistic range
            em_pct = remaining_vol
            
            # Cap EM to realistic 0DTE range (0.3% to 1.5%)
            em_pct = max(0.003, min(0.015, em_pct))
            
            return current_price * em_pct
    
    def get_credit_for_em_distance(self, em_distance: float, width: int) -> float:
        """
        Estimate credit based on how far OTM the short strike is.
        
        Per METF rules: credit should be between $1.25 and $2.50.
        Closer to ATM = higher credit, further OTM = lower credit.
        
        This is a simplified model - real credit depends on IV, time to expiry, etc.
        """
        import random
        
        # Base credit scales with how close we are to ATM
        # At EM boundary, we expect mid-range credit
        # Use random within the valid range to simulate market conditions
        base_credit = random.uniform(self.config.min_credit, self.config.max_credit)
        
        return round(base_credit, 2)
    
    # -------------------------------------------------------------------------
    # Real-pricing (CBBO) selection + simulation
    # -------------------------------------------------------------------------

    def _select_real_spread(
        self,
        spread_type: SpreadType,
        underlying: float,
        expected_move: float,
        minute_et: str,
    ) -> Tuple[Optional[Tuple[float, float, int, float, float, float]], str]:
        """
        Pick (short_strike, long_strike, width, credit, short_bid, long_ask) using real cbbo quotes.

        Iterates configured widths; for each width, snaps the EM-OTM short to the
        nearest cached strike and verifies the long strike is also in the cache.
        Picks the width that yields a credit closest to (but not above) max_credit,
        within [min_credit, max_credit].

        Returns ((short, long, width, credit, short_bid, long_ask), '') on success
        or (None, reason) on failure where reason is one of:
          'no_strikes' | 'no_short_quote' | 'no_long_strike' | 'no_long_quote' | 'credit_band'
        """
        if self._pricer is None:
            return None, "no_pricer"

        strikes = self._pricer.available_strikes
        if not strikes:
            return None, "no_strikes"

        is_call = spread_type == SpreadType.CALL_CREDIT_SPREAD
        # EM-OTM short strike target
        short_target = (underlying + expected_move) if is_call else (underlying - expected_move)
        short_strike = self._pricer.nearest_strike(short_target, available=strikes)
        if short_strike is None:
            return None, "no_strikes"

        best: Optional[Tuple[float, float, int, float, float, float]] = None
        last_reason = "credit_band"

        for width in self.config.spread_widths:
            long_strike = (short_strike + width) if is_call else (short_strike - width)
            if long_strike not in strikes:
                last_reason = "no_long_strike"
                continue

            quote = self._pricer.entry_credit(minute_et, short_strike, long_strike, is_call)
            if quote is None:
                short_q = self._pricer.lookup(minute_et, short_strike, is_call)
                last_reason = "no_short_quote" if short_q is None else "no_long_quote"
                continue

            credit, short_bid, long_ask = quote
            if not (self.config.min_credit <= credit <= self.config.max_credit):
                last_reason = "credit_band"
                continue

            # Prefer higher credit (closer to max_credit / shorter time-to-expiry capture)
            if best is None or credit > best[3]:
                best = (short_strike, long_strike, width, round(credit, 2), short_bid, long_ask)

        if best is None:
            return None, last_reason
        return best, ""

    def _simulate_real_spread(
        self,
        spread_type: SpreadType,
        short_strike: float,
        long_strike: float,
        width: int,
        credit: float,
        entry_datetime: datetime,
        bars_after_entry: pd.DataFrame,
    ) -> Tuple[ExitReason, float, float, datetime, Optional[float], Optional[float]]:
        """
        Walk forward minute-by-minute through cbbo_cache marks.

        Returns:
            (exit_reason, pnl, max_adverse_excursion_$, exit_time, exit_mark, underlying_at_exit)

        Stop-loss fires when close-mark (short_ask − long_bid) ≥ credit × (1 + stop_mult).
        At expiration, settles based on terminal SPX vs short_strike.
        """
        assert self._pricer is not None
        is_call = spread_type == SpreadType.CALL_CREDIT_SPREAD
        stop_mark = credit * (1 + self.config.stop_loss_multiplier)
        max_adverse_dollars = 0.0
        exit_mark: Optional[float] = None

        for ts, row in bars_after_entry.iterrows():
            minute_et = ts.strftime("%H:%M")
            mark = self._pricer.close_mark(minute_et, short_strike, long_strike, is_call)
            if mark is None:
                continue
            # Adverse excursion in $ (per contract = ×100)
            adverse = max(0.0, mark - credit) * 100.0
            if adverse > max_adverse_dollars:
                max_adverse_dollars = adverse

            if mark >= stop_mark:
                # Pay `mark` to close → pnl = credit − mark
                pnl = (credit - mark) * 100.0
                return ExitReason.STOP_LOSS, pnl, max_adverse_dollars, ts, mark, float(row['close'])

        # Held to expiration — settle on terminal SPX
        if len(bars_after_entry) == 0:
            return ExitReason.EXPIRATION_WIN, credit * 100.0, max_adverse_dollars, entry_datetime, None, None

        final_ts = bars_after_entry.index[-1]
        final_price = float(bars_after_entry.iloc[-1]['close'])

        if is_call:
            itm = max(0.0, final_price - short_strike)
        else:
            itm = max(0.0, short_strike - final_price)
        itm_capped = min(itm, float(width))
        pnl = (credit - itm_capped) * 100.0
        reason = ExitReason.EXPIRATION_WIN if pnl > 0 else ExitReason.EXPIRATION_LOSS
        return reason, pnl, max_adverse_dollars, final_ts, itm_capped, final_price

    def simulate_spread_outcome(
        self,
        spread_type: SpreadType,
        entry_price: float,
        entry_time: datetime,
        bars_after_entry: pd.DataFrame,
        credit: float,
        width: int,
        expected_move: float
    ) -> Tuple[ExitReason, float, float, datetime]:
        """
        Simulate credit spread outcome with Expected Move (EM) based strike placement.
        
        Based on METF strategy rules:
        - Short strike is placed at the Expected Move boundary
        - Credit target: $1.25 - $2.50
        - Stop loss: 1x credit (spread value reaches 2x credit)
        - Hold to expiration unless stopped
        
        Strike Placement Logic:
        - PUT CREDIT SPREAD: Short put at entry_price - expected_move
        - CALL CREDIT SPREAD: Short call at entry_price + expected_move
        
        Win/Loss at Expiration:
        - WIN: Price stays above short put (PCS) or below short call (CCS)
        - LOSS: Price breaches short strike
        
        Returns:
            Tuple of (exit_reason, pnl, max_adverse_excursion, exit_time)
        """
        # Use Expected Move for OTM distance (short strike at EM boundary)
        otm_distance = expected_move
        
        # Place strikes based on spread type
        if spread_type == SpreadType.PUT_CREDIT_SPREAD:
            # Bullish: sell put spread below current price
            short_strike = entry_price - otm_distance
            long_strike = short_strike - width
        else:
            # Bearish: sell call spread above current price
            short_strike = entry_price + otm_distance
            long_strike = short_strike + width
        
        # Stop loss price (1x credit = 2x initial spread value)
        stop_spread_value = credit * (1 + self.config.stop_loss_multiplier)
        max_loss = width - credit
        
        # Track adverse movement
        max_adverse = 0.0
        
        # Simulate through each bar
        for idx, row in bars_after_entry.iterrows():
            current_price = row['close']
            current_low = row['low']
            current_high = row['high']
            
            # Estimate spread value based on intrinsic + time value decay
            # As we approach expiration, spread value is mostly intrinsic
            if spread_type == SpreadType.PUT_CREDIT_SPREAD:
                # Put spread value increases as price drops toward/below short strike
                if current_low <= short_strike:
                    # Price touched or breached short strike - spread is ITM
                    intrinsic = min(short_strike - current_low, width)
                    # Add some extrinsic value early, decays toward expiration
                    time_remaining = len(bars_after_entry) - bars_after_entry.index.get_loc(idx)
                    time_pct = time_remaining / max(len(bars_after_entry), 1)
                    extrinsic = credit * 0.3 * time_pct  # Small extrinsic component
                    spread_value = intrinsic + extrinsic
                else:
                    # OTM - spread value is mostly extrinsic, decaying
                    distance_to_strike = current_price - short_strike
                    distance_pct = distance_to_strike / otm_distance
                    time_remaining = len(bars_after_entry) - bars_after_entry.index.get_loc(idx)
                    time_pct = time_remaining / max(len(bars_after_entry), 1)
                    # Spread value decays as we stay OTM and time passes
                    spread_value = credit * (0.3 + 0.7 * time_pct) * max(0.2, 1 - distance_pct * 0.5)
            else:
                # Call spread value increases as price rises toward/above short strike
                if current_high >= short_strike:
                    # Price touched or breached short strike - spread is ITM
                    intrinsic = min(current_high - short_strike, width)
                    time_remaining = len(bars_after_entry) - bars_after_entry.index.get_loc(idx)
                    time_pct = time_remaining / max(len(bars_after_entry), 1)
                    extrinsic = credit * 0.3 * time_pct
                    spread_value = intrinsic + extrinsic
                else:
                    # OTM - spread value decaying
                    distance_to_strike = short_strike - current_price
                    distance_pct = distance_to_strike / otm_distance
                    time_remaining = len(bars_after_entry) - bars_after_entry.index.get_loc(idx)
                    time_pct = time_remaining / max(len(bars_after_entry), 1)
                    spread_value = credit * (0.3 + 0.7 * time_pct) * max(0.2, 1 - distance_pct * 0.5)
            
            # Track max adverse excursion
            if spread_value > credit:
                max_adverse = max(max_adverse, spread_value - credit)
            
            # Check stop loss (spread value reaches 2x credit)
            if spread_value >= stop_spread_value:
                pnl = -credit * 100  # Loss equals credit received (1x stop)
                return ExitReason.STOP_LOSS, pnl, max_adverse, idx
        
        # Held to expiration - determine final outcome
        if len(bars_after_entry) == 0:
            return ExitReason.EXPIRATION_WIN, credit * 100, 0.0, entry_time
        
        final_bar = bars_after_entry.iloc[-1]
        final_price = final_bar['close']
        exit_time = bars_after_entry.index[-1]
        
        if spread_type == SpreadType.PUT_CREDIT_SPREAD:
            itm_amount = max(0.0, min(short_strike - final_price, width))
        else:
            itm_amount = max(0.0, min(final_price - short_strike, width))
        pnl = (credit - itm_amount) * 100
        reason = ExitReason.EXPIRATION_WIN if pnl > 0 else ExitReason.EXPIRATION_LOSS
        return reason, pnl, max_adverse, exit_time
    
    def backtest_day(
        self,
        trade_date: str,
        bars: pd.DataFrame
    ) -> DailyResult:
        """
        Backtest a single trading day
        
        Args:
            trade_date: Date string (YYYY-MM-DD)
            bars: DataFrame with 1-minute bars (timestamp index, OHLCV columns)
            
        Returns:
            DailyResult with all trades for the day
        """
        trades: List[TradeResult] = []
        skip_reasons: Dict[str, int] = {}

        # Check FOMC filter
        is_fomc = self.is_fomc_day(trade_date)
        if is_fomc:
            self.logger.info(f"{trade_date}: FOMC day - skipping")
            return DailyResult(
                trade_date=trade_date,
                trades=[],
                total_pnl=0.0,
                num_winners=0,
                num_losers=0,
                is_fomc_day=True
            )
        
        # Ensure bars are sorted by time
        bars = bars.sort_index()
        
        # Calculate EMAs
        bars['ema_fast'] = self.calculate_ema(bars['close'], self.config.ema_fast_period)
        bars['ema_slow'] = self.calculate_ema(bars['close'], self.config.ema_slow_period)
        
        # Get entry times
        entry_times = self.config.get_entry_times()
        
        # Process each entry time
        for entry_num, entry_time in enumerate(entry_times, 1):
            # Find bar at entry time
            entry_datetime = None
            entry_bar = None
            
            for idx in bars.index:
                bar_time = idx.time() if hasattr(idx, 'time') else idx
                if isinstance(bar_time, datetime):
                    bar_time = bar_time.time()
                
                # Match entry time (within same minute)
                if hasattr(idx, 'hour'):
                    if idx.hour == entry_time.hour and idx.minute == entry_time.minute:
                        entry_datetime = idx
                        entry_bar = bars.loc[idx]
                        break
                else:
                    # Handle case where index is already a time
                    dt = datetime.combine(date.fromisoformat(trade_date), bar_time)
                    if dt.hour == entry_time.hour and dt.minute == entry_time.minute:
                        entry_datetime = dt
                        entry_bar = bars.loc[idx]
                        break
            
            if entry_bar is None:
                self.logger.debug(f"{trade_date} {entry_time}: No bar found at entry time")
                continue
            
            # Get EMA values at entry
            ema_fast = entry_bar['ema_fast']
            ema_slow = entry_bar['ema_slow']
            
            # Skip if EMAs not yet calculated (early in day)
            if pd.isna(ema_fast) or pd.isna(ema_slow):
                self.logger.debug(f"{trade_date} {entry_time}: EMAs not ready")
                continue
            
            # Determine trend
            trend = self.get_trend(ema_fast, ema_slow)
            
            if trend == TrendDirection.NEUTRAL:
                self.logger.debug(f"{trade_date} {entry_time}: Neutral trend - no entry")
                continue
            
            # Determine spread type
            if trend == TrendDirection.BULLISH:
                spread_type = SpreadType.PUT_CREDIT_SPREAD
            else:
                spread_type = SpreadType.CALL_CREDIT_SPREAD
            
            # Get bars after entry until market close
            bars_after = bars.loc[entry_datetime:]
            if len(bars_after) < 2:
                continue
            
            # Get entry bar index for EM calculation
            entry_price = entry_bar['close']
            entry_idx = bars.index.get_loc(entry_datetime)
            expected_move = self.calculate_expected_move(bars, entry_idx)

            short_strike = long_strike = None
            short_bid_at_entry = long_ask_at_entry = None
            exit_mark = underlying_at_exit = None

            if self.config.use_cbbo:
                # Real-pricing path: pick strikes from cache, walk forward through real marks
                minute_et = entry_datetime.strftime("%H:%M")
                pick, reason = self._select_real_spread(
                    spread_type, entry_price, expected_move, minute_et
                )
                if pick is None:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    self.logger.debug(f"{trade_date} {entry_time}: skip ({reason})")
                    continue
                short_strike, long_strike, width, credit, short_bid_at_entry, long_ask_at_entry = pick

                (exit_reason, pnl, max_adverse,
                 exit_time, exit_mark, underlying_at_exit) = self._simulate_real_spread(
                    spread_type=spread_type,
                    short_strike=short_strike,
                    long_strike=long_strike,
                    width=width,
                    credit=credit,
                    entry_datetime=entry_datetime,
                    bars_after_entry=bars_after.iloc[1:],
                )
            else:
                # Synthetic path (legacy): random credit, heuristic decay
                if self.config.use_em_strikes:
                    credit = self.get_credit_for_em_distance(expected_move, self.config.default_width)
                else:
                    expected_move = entry_price * 0.0075
                    credit = self.config.default_credit
                width = self.config.default_width

                exit_reason, pnl, max_adverse, exit_time = self.simulate_spread_outcome(
                    spread_type=spread_type,
                    entry_price=entry_price,
                    entry_time=entry_datetime,
                    bars_after_entry=bars_after.iloc[1:],
                    credit=credit,
                    width=width,
                    expected_move=expected_move,
                )

            trade = TradeResult(
                trade_date=trade_date,
                entry_time=entry_datetime,
                entry_number=entry_num,
                trend=trend,
                spread_type=spread_type,
                underlying_at_entry=entry_price,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                credit=credit,
                width=width,
                exit_time=exit_time,
                exit_reason=exit_reason,
                pnl=pnl,
                max_adverse_excursion=max_adverse,
                short_strike=short_strike,
                long_strike=long_strike,
                short_bid_at_entry=short_bid_at_entry,
                long_ask_at_entry=long_ask_at_entry,
                exit_mark=exit_mark,
                underlying_at_exit=underlying_at_exit,
            )

            trades.append(trade)
            self.logger.debug(
                f"{trade_date} Entry #{entry_num}: {spread_type.value} @ ${entry_price:.2f}, "
                f"PnL: ${pnl:.2f} ({exit_reason.value})"
            )
        
        # Compile daily results
        total_pnl = sum(t.pnl for t in trades)
        num_winners = sum(1 for t in trades if t.is_winner)
        num_losers = sum(1 for t in trades if not t.is_winner)
        
        return DailyResult(
            trade_date=trade_date,
            trades=trades,
            total_pnl=total_pnl,
            num_winners=num_winners,
            num_losers=num_losers,
            is_fomc_day=False,
            skip_reasons=skip_reasons,
        )
    
    def run(
        self,
        bars: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResults:
        """
        Run backtest on historical data
        
        Args:
            bars: DataFrame with 1-minute bars (timestamp index, OHLCV columns)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            BacktestResults with all metrics
        """
        self.logger.info(f"Starting METF backtest...")
        
        # Ensure timestamp index
        if not isinstance(bars.index, pd.DatetimeIndex):
            if 'timestamp' in bars.columns:
                bars = bars.set_index('timestamp')
            else:
                raise ValueError("Bars must have timestamp index or 'timestamp' column")
        
        # Filter by date range
        if start_date:
            bars = bars[bars.index >= start_date]
        if end_date:
            bars = bars[bars.index <= end_date]
        
        # Get unique trading days
        bars['date'] = bars.index.date
        trading_days = sorted(bars['date'].unique())
        
        self.logger.info(f"Processing {len(trading_days)} trading days")
        
        # Backtest each day
        daily_results: List[DailyResult] = []
        skipped_no_cbbo: List[str] = []

        for trade_date in trading_days:
            date_str = trade_date.isoformat()
            day_bars = bars[bars['date'] == trade_date].copy()
            day_bars = day_bars.drop(columns=['date'])

            if len(day_bars) < self.config.ema_slow_period:
                self.logger.debug(f"{date_str}: Insufficient bars ({len(day_bars)})")
                continue

            # Load CBBO quotes for the day if real-pricing mode is on.
            if self.config.use_cbbo:
                assert self._pricer is not None
                if not self._pricer.load_day(trade_date):
                    self.logger.debug(f"{date_str}: No cbbo cache - skipping day")
                    skipped_no_cbbo.append(date_str)
                    continue

            daily_result = self.backtest_day(date_str, day_bars)
            daily_results.append(daily_result)

        # Compile results
        actual_start = trading_days[0].isoformat() if trading_days else start_date or ""
        actual_end = trading_days[-1].isoformat() if trading_days else end_date or ""

        results = BacktestResults(
            config=self.config,
            start_date=actual_start,
            end_date=actual_end,
            daily_results=daily_results,
            skipped_days_no_cbbo=skipped_no_cbbo,
        )
        
        self.logger.info(f"Backtest complete: {results.total_trades} trades")
        
        return results


# =============================================================================
# Results Reporter
# =============================================================================

class METFResultsReporter:
    """Generate reports from backtest results"""
    
    def __init__(self, results: BacktestResults):
        self.results = results
    
    def print_summary(self):
        """Print summary statistics to console"""
        r = self.results
        
        print(f"\n{'='*60}")
        print("METF Backtest Results")
        print(f"{'='*60}")
        print(f"Period:          {r.start_date} to {r.end_date}")
        print(f"Trading Days:    {len(r.daily_results)}")
        print(f"Total Trades:    {r.total_trades}")
        print()
        
        print("Performance:")
        print(f"  Total P&L:     ${r.total_pnl:,.2f}")
        print(f"  Win Rate:      {r.win_rate:.1%}")
        print(f"  Winners:       {r.total_winners}")
        print(f"  Losers:        {r.total_losers}")
        print(f"  Avg Winner:    ${r.avg_winner:,.2f}")
        print(f"  Avg Loser:     ${r.avg_loser:,.2f}")
        print(f"  Profit Factor: {r.profit_factor:.2f}")
        print(f"  Max Drawdown:  ${r.max_drawdown:,.2f}")
        print(f"  Sharpe Ratio:  {r.sharpe_ratio:.2f}")
        print()
        
        # Entry time analysis
        print("By Entry Time:")
        entry_stats: Dict[int, Dict] = {}
        for daily in r.daily_results:
            for trade in daily.trades:
                if trade.entry_number not in entry_stats:
                    entry_stats[trade.entry_number] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
                if trade.is_winner:
                    entry_stats[trade.entry_number]['wins'] += 1
                else:
                    entry_stats[trade.entry_number]['losses'] += 1
                entry_stats[trade.entry_number]['pnl'] += trade.pnl
        
        entry_times = r.config.entry_times
        for entry_num in sorted(entry_stats.keys()):
            stats = entry_stats[entry_num]
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total if total > 0 else 0
            time_str = entry_times[entry_num - 1] if entry_num <= len(entry_times) else f"#{entry_num}"
            print(f"  {time_str}: {stats['wins']}W/{stats['losses']}L ({wr:.0%}) P&L: ${stats['pnl']:,.2f}")
        
        print()
        
        # Trend analysis
        print("By Trend Direction:")
        trend_stats: Dict[str, Dict] = {}
        for daily in r.daily_results:
            for trade in daily.trades:
                trend = trade.trend.value
                if trend not in trend_stats:
                    trend_stats[trend] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
                if trade.is_winner:
                    trend_stats[trend]['wins'] += 1
                else:
                    trend_stats[trend]['losses'] += 1
                trend_stats[trend]['pnl'] += trade.pnl
        
        for trend, stats in trend_stats.items():
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total if total > 0 else 0
            print(f"  {trend}: {stats['wins']}W/{stats['losses']}L ({wr:.0%}) P&L: ${stats['pnl']:,.2f}")

        if r.config.use_cbbo:
            print()
            print("CBBO Skip Reasons:")
            agg = r.aggregate_skip_reasons
            if not agg and not r.skipped_days_no_cbbo:
                print("  (none)")
            else:
                for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
                    print(f"  {k}: {v}")
                if r.skipped_days_no_cbbo:
                    print(f"  days_no_cbbo_cache: {len(r.skipped_days_no_cbbo)}")

        print(f"\n{'='*60}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        rows = []
        for daily in self.results.daily_results:
            for trade in daily.trades:
                rows.append({
                    'date': trade.trade_date,
                    'entry_time': trade.entry_time,
                    'entry_number': trade.entry_number,
                    'trend': trade.trend.value,
                    'spread_type': trade.spread_type.value,
                    'underlying': trade.underlying_at_entry,
                    'ema_fast': trade.ema_fast,
                    'ema_slow': trade.ema_slow,
                    'short_strike': trade.short_strike,
                    'long_strike': trade.long_strike,
                    'short_bid_at_entry': trade.short_bid_at_entry,
                    'long_ask_at_entry': trade.long_ask_at_entry,
                    'credit': trade.credit,
                    'width': trade.width,
                    'exit_time': trade.exit_time,
                    'exit_reason': trade.exit_reason.value,
                    'exit_mark': trade.exit_mark,
                    'underlying_at_exit': trade.underlying_at_exit,
                    'pnl': trade.pnl,
                    'is_winner': trade.is_winner,
                    'max_adverse': trade.max_adverse_excursion
                })
        
        return pd.DataFrame(rows)
    
    def export_csv(self, filepath: str):
        """Export results to CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
    
    def export_summary(self, filepath: str):
        """Export summary to text file"""
        import io
        import sys
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.print_summary()
        sys.stdout = old_stdout
        
        with open(filepath, 'w') as f:
            f.write(buffer.getvalue())
        
        print(f"Summary exported to {filepath}")


# =============================================================================
# Data Loading
# =============================================================================

def load_bars_from_csv(filepath: str) -> pd.DataFrame:
    """Load 1-minute bars from CSV file"""
    df = pd.read_csv(filepath)
    
    # Try to parse timestamp column
    timestamp_cols = ['timestamp', 'datetime', 'date', 'time']
    for col in timestamp_cols:
        if col in df.columns:
            df['timestamp'] = pd.to_datetime(df[col])
            break
    
    if 'timestamp' not in df.columns:
        raise ValueError(f"Could not find timestamp column. Available: {df.columns.tolist()}")
    
    df = df.set_index('timestamp')
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            # Try lowercase
            if col.lower() in df.columns:
                df[col] = df[col.lower()]
            else:
                raise ValueError(f"Missing required column: {col}")
    
    return df


def load_bars_from_db(
    db_path: str,
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Load 1-minute bars from SQLite database"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM underlying_bars
        WHERE symbol = ? AND date(timestamp) BETWEEN ? AND ?
        ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
    conn.close()
    
    if df.empty:
        raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    return df


def load_bars_from_marketdata(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1min",
    cache_db: str = "backtest_data/underlying_bars.db",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Load bars from MarketData.app API with SQLite caching.
    
    - Checks cache first for existing bars
    - Only fetches missing days from API
    - Saves new data to cache for future runs
    
    Requires MARKET_DATA_TOKEN environment variable.
    
    Args:
        symbol: Stock symbol (SPX, SPY, etc.)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1min, 5min, 15min)
        cache_db: Path to SQLite cache database
        force_refresh: If True, bypass cache and fetch all from API
        
    Returns:
        DataFrame with OHLCV data
    """
    import os
    import sqlite3
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("MARKET_DATA_TOKEN")
    if not api_key:
        raise ValueError("MARKET_DATA_TOKEN environment variable not set")
    
    # Get list of trading days needed
    trading_days = _get_trading_days(start_date, end_date)
    logger.info(f"Requested {len(trading_days)} trading days for {symbol}")
    
    # Check cache for existing data (unless force_refresh)
    cached_days = set()
    cached_bars = pd.DataFrame()
    
    if not force_refresh and os.path.exists(cache_db):
        try:
            conn = sqlite3.connect(cache_db)
            # Find which days we already have in cache
            query = """
                SELECT DISTINCT date(timestamp) as trade_date
                FROM underlying_bars
                WHERE symbol = ? AND date(timestamp) BETWEEN ? AND ?
            """
            result = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            cached_days = set(result['trade_date'].tolist())
            
            if cached_days:
                # Load cached bars
                bars_query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM underlying_bars
                    WHERE symbol = ? AND date(timestamp) BETWEEN ? AND ?
                    ORDER BY timestamp
                """
                cached_bars = pd.read_sql_query(bars_query, conn, params=(symbol, start_date, end_date))
                if not cached_bars.empty:
                    cached_bars['timestamp'] = pd.to_datetime(cached_bars['timestamp'])
                    cached_bars = cached_bars.set_index('timestamp')
                logger.info(f"Found {len(cached_days)} days in cache ({len(cached_bars)} bars)")
            
            conn.close()
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
    
    # Determine which days need to be fetched
    missing_days = [d for d in trading_days if d not in cached_days]
    
    if not missing_days:
        logger.info(f"All {len(trading_days)} days found in cache, no API calls needed")
        return cached_bars
    
    logger.info(f"Need to fetch {len(missing_days)} missing days from MarketData.app")
    
    # Fetch missing days from API
    new_bars = _fetch_missing_days(symbol, missing_days, interval, api_key)
    
    if not new_bars.empty:
        # Cache the new bars
        _cache_bars_to_db(new_bars, symbol, cache_db, interval)
        logger.info(f"Cached {len(new_bars)} new bars to {cache_db}")
    
    # Combine cached and new bars
    if cached_bars.empty:
        all_bars = new_bars
    elif new_bars.empty:
        all_bars = cached_bars
    else:
        all_bars = pd.concat([cached_bars, new_bars])
        all_bars = all_bars.sort_index()
        # Remove any duplicates
        all_bars = all_bars[~all_bars.index.duplicated(keep='last')]
    
    if all_bars.empty:
        raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")
    
    logger.info(f"Total: {len(all_bars)} bars ({len(cached_days)} cached + {len(missing_days)} fetched)")
    return all_bars


def _get_trading_days(start_date: str, end_date: str) -> List[str]:
    """Get list of trading days (weekdays) between dates"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            days.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return days


def _fetch_missing_days(
    symbol: str,
    days: List[str],
    interval: str,
    api_key: str
) -> pd.DataFrame:
    """Fetch specific days from MarketData.app API"""
    import requests
    from datetime import time as time_obj
    import time as time_module
    
    BASE_URL = "https://api.marketdata.app/v1"
    resolution_map = {"1min": "1", "5min": "5", "15min": "15", "1day": "D"}
    resolution = resolution_map.get(interval, "1")
    
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {api_key}"})
    
    market_open = time_obj(9, 30)
    market_close = time_obj(16, 0)
    
    all_bars = []
    fetched_count = 0
    
    for day in days:
        try:
            current_date = datetime.strptime(day, '%Y-%m-%d').date()
            market_open_dt = datetime.combine(current_date, market_open)
            market_close_dt = datetime.combine(current_date, market_close)
            
            from_ts = int(market_open_dt.timestamp())
            to_ts = int(market_close_dt.timestamp())
            
            params = {"from": from_ts, "to": to_ts}
            
            response = session.get(f"{BASE_URL}/stocks/candles/{resolution}/{symbol}/", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("s") == "ok" and 't' in data:
                for i in range(len(data['t'])):
                    all_bars.append({
                        'timestamp': datetime.fromtimestamp(data['t'][i]),
                        'open': data['o'][i],
                        'high': data['h'][i],
                        'low': data['l'][i],
                        'close': data['c'][i],
                        'volume': data['v'][i]
                    })
                fetched_count += 1
                logger.debug(f"Fetched {len(data['t'])} bars for {day}")
            
            time_module.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            logger.warning(f"Failed to fetch {day}: {e}")
    
    logger.info(f"Successfully fetched {fetched_count}/{len(days)} days from API")
    
    if not all_bars:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    
    return df


def _cache_bars_to_db(df: pd.DataFrame, symbol: str, db_path: str, interval: str):
    """Save bars to SQLite cache database"""
    import sqlite3
    from pathlib import Path
    
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS underlying_bars (
            symbol TEXT,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT DEFAULT 'marketdata',
            PRIMARY KEY (symbol, timestamp)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON underlying_bars(symbol, timestamp)")
    
    # Prepare records
    records = []
    for timestamp, row in df.iterrows():
        records.append((
            symbol,
            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            int(row['volume']),
            'marketdata'
        ))
    
    # Insert with replace for any duplicates
    conn.executemany("""
        INSERT OR REPLACE INTO underlying_bars 
        (symbol, timestamp, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    conn.close()


def load_bars_via_bar_loader(
    csv_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    price_scale: float = 10.0,
    symbol: str = "SPX",
) -> pd.DataFrame:
    """
    Load 1-min bars via src_v2.backtest.data_loader.BarLoader.

    Produces a DataFrame with a tz-naive ET DatetimeIndex and OHLCV columns,
    compatible with METFBacktester.run().
    """
    from src_v2.backtest.data_loader import BarLoader

    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None
    loader = BarLoader(csv_path, price_scale=price_scale, symbol=symbol)

    rows: List[Dict] = []
    rth_open = time(9, 30)
    rth_close = time(16, 0)
    for day in loader.iter_days(start_date=sd, end_date=ed):
        for bar in day.bars:
            ts_et = bar["ts_et"].replace(tzinfo=None)  # tz-naive ET for downstream compat
            t = ts_et.time()
            if t < rth_open or t > rth_close:
                continue  # Skip extended-hours bars to keep EMAs RTH-only
            rows.append({
                "timestamp": ts_et,
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            })

    if not rows:
        raise ValueError(f"BarLoader returned no bars for {csv_path} {start_date}..{end_date}")
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def generate_sample_data(
    symbol: str,
    start_date: str,
    end_date: str,
    base_price: float = 6000.0
) -> pd.DataFrame:
    """Generate sample 1-minute bar data for testing"""
    import random
    
    tz = ZoneInfo('America/New_York')
    start = datetime.fromisoformat(start_date).replace(tzinfo=tz)
    end = datetime.fromisoformat(end_date).replace(tzinfo=tz)
    
    bars = []
    current_date = start.date()
    price = base_price
    
    while current_date <= end.date():
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Generate bars for trading hours
        current_time = datetime.combine(current_date, time(9, 30), tzinfo=tz)
        market_close = datetime.combine(current_date, time(16, 0), tzinfo=tz)
        
        # Daily trend (random)
        daily_trend = random.choice([-1, 0, 1]) * random.uniform(0.001, 0.003)
        
        while current_time < market_close:
            # Random walk with trend
            change = random.gauss(daily_trend * price, price * 0.0005)
            price = max(price + change, price * 0.95)  # Prevent extreme drops
            
            bar_high = price + random.uniform(0, price * 0.001)
            bar_low = price - random.uniform(0, price * 0.001)
            
            bars.append({
                'timestamp': current_time,
                'open': price,
                'high': bar_high,
                'low': bar_low,
                'close': price,
                'volume': random.randint(1000, 10000)
            })
            
            current_time += timedelta(minutes=1)
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(bars)
    df = df.set_index('timestamp')
    
    return df


# =============================================================================
# CLI
# =============================================================================

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Backtest METF (Multiple Entry Trend Following) strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m app.backtesting.metf_backtester --start 2024-01-01 --end 2024-03-31
    python -m app.backtesting.metf_backtester --csv data/spx_1min.csv
    python -m app.backtesting.metf_backtester --db backtest_data/underlying_bars.db --symbol SPX
        """
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='SPX',
        help='Symbol to backtest (default: SPX)'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to CSV file with 1-minute bars'
    )
    
    parser.add_argument(
        '--db',
        type=str,
        default=None,
        help='Path to SQLite database with bars'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use generated sample data for testing'
    )
    
    parser.add_argument(
        '--marketdata',
        action='store_true',
        help='Fetch real data from MarketData.app (requires MARKET_DATA_TOKEN). Uses cache by default.'
    )
    
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Bypass cache and fetch all data from API (use with --marketdata)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1min',
        choices=['1min', '5min', '15min'],
        help='Data interval for MarketData.app (default: 1min)'
    )
    
    parser.add_argument(
        '--cache-db',
        type=str,
        default='backtest_data/underlying_bars.db',
        help='Path to SQLite cache database (default: backtest_data/underlying_bars.db)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for trade results'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Strategy parameters
    parser.add_argument(
        '--ema-fast',
        type=int,
        default=20,
        help='Fast EMA period (default: 20)'
    )
    
    parser.add_argument(
        '--ema-slow',
        type=int,
        default=40,
        help='Slow EMA period (default: 40)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=20,
        help='Default spread width (default: 20)'
    )
    
    parser.add_argument(
        '--credit',
        type=float,
        default=1.75,
        help='Default credit for simulation (default: 1.75)'
    )
    
    parser.add_argument(
        '--otm-pct',
        type=float,
        default=0.75,
        help='OTM distance as %% of underlying price (default: 0.75 = 0.75%% OTM ≈ 15-20 delta). Ignored when --use-em is set.'
    )
    
    parser.add_argument(
        '--use-em',
        action='store_true',
        default=True,
        help='Place short strike at Expected Move boundary (default: True)'
    )
    
    parser.add_argument(
        '--no-em',
        action='store_true',
        help='Disable EM-based strikes, use fixed OTM percentage instead'
    )
    
    parser.add_argument(
        '--em-method',
        type=str,
        default='historical_vol',
        choices=['historical_vol', 'atr', 'fixed_pct'],
        help='Method for calculating Expected Move (default: historical_vol)'
    )
    
    parser.add_argument(
        '--em-fixed-pct',
        type=float,
        default=1.0,
        help='Fixed EM as %% of underlying (default: 1.0 = 1%%). Used when --em-method=fixed_pct'
    )

    # Real-pricing (CBBO) flags
    parser.add_argument(
        '--use-cbbo',
        action='store_true',
        help='Use real cbbo_cache option quotes for credit + exit simulation (replaces synthetic).'
    )

    parser.add_argument(
        '--cbbo-dir',
        type=str,
        default='data/cbbo_cache',
        help='Directory of cbbo_cache pickle day-files (default: data/cbbo_cache)'
    )

    parser.add_argument(
        '--bar-csv',
        type=str,
        default='data/spy_1m.csv',
        help='1-min OHLCV CSV used by BarLoader (default: data/spy_1m.csv, scaled to SPX)'
    )

    parser.add_argument(
        '--price-scale',
        type=float,
        default=10.0,
        help='Multiplier applied to bar prices (default 10.0 — SPY * 10 ≈ SPX)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_cli_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    if args.use_cbbo:
        if not args.start or not args.end:
            print("Error: --start and --end required when using --use-cbbo")
            return 1
        print(f"Loading bars via BarLoader: {args.bar_csv} (scale={args.price_scale})")
        bars = load_bars_via_bar_loader(
            args.bar_csv,
            start_date=args.start,
            end_date=args.end,
            price_scale=args.price_scale,
            symbol=args.symbol,
        )
    elif args.csv:
        print(f"Loading data from CSV: {args.csv}")
        bars = load_bars_from_csv(args.csv)
    elif args.db:
        if not args.start or not args.end:
            print("Error: --start and --end required when using --db")
            return 1
        print(f"Loading data from database: {args.db}")
        bars = load_bars_from_db(args.db, args.symbol, args.start, args.end)
    elif args.marketdata:
        if not args.start or not args.end:
            print("Error: --start and --end required when using --marketdata")
            return 1
        cache_msg = "(bypassing cache)" if args.force_refresh else f"(cache: {args.cache_db})"
        print(f"Fetching data from MarketData.app for {args.symbol} {cache_msg}...")
        try:
            bars = load_bars_from_marketdata(
                args.symbol, 
                args.start, 
                args.end, 
                args.interval,
                cache_db=args.cache_db,
                force_refresh=args.force_refresh
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.sample:
        start = args.start or '2024-01-01'
        end = args.end or '2024-01-31'
        print(f"Generating sample data from {start} to {end}")
        base_price = 6000.0 if args.symbol == 'SPX' else 600.0
        bars = generate_sample_data(args.symbol, start, end, base_price)
    else:
        print("Error: Must specify --csv, --db, --marketdata, or --sample")
        return 1
    
    print(f"Loaded {len(bars)} bars")
    
    # Create config
    use_em = args.use_em and not args.no_em  # --no-em overrides --use-em
    config = METFBacktestConfig(
        symbol=args.symbol,
        ema_fast_period=args.ema_fast,
        ema_slow_period=args.ema_slow,
        default_width=args.width,
        default_credit=args.credit,
        use_em_strikes=use_em,
        em_method=args.em_method,
        em_fixed_pct=args.em_fixed_pct / 100.0,  # Convert from % to decimal
        use_cbbo=args.use_cbbo,
        cbbo_dir=args.cbbo_dir,
        bar_csv_path=args.bar_csv,
        bar_price_scale=args.price_scale,
    )
    
    # Run backtest
    backtester = METFBacktester(config)
    results = backtester.run(bars, args.start, args.end)
    
    # Print results
    reporter = METFResultsReporter(results)
    reporter.print_summary()
    
    # Export if requested
    if args.output:
        reporter.export_csv(args.output)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
