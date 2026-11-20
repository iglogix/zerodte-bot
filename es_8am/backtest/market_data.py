"""
Market data loaders for SPY and VIX data used in backtesting.

Provides SPX proxy (SPY * 10) and daily IV lookup from VIX/VIX9D.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Dict, Optional

import pandas as pd


class MarketDataLoader:
    """
    Loads and caches SPY and VIX daily data for backtest use.

    SPY is scaled by 10 to approximate SPX index levels.
    IV priority: VIX1D (1-day) > VIX9D (9-day) > VIX (30-day).
    """

    def __init__(
        self,
        data_dir: str = "data",
        spy_file: str = "spy_daily.csv",
        vix_file: str = "vix_daily.csv",
        vix9d_file: str = "vix9d_daily.csv",
        vix1d_file: str = "vix1d_daily.csv",
    ):
        self._data_dir = data_dir
        self._spy_file = spy_file
        self._vix_file = vix_file
        self._vix9d_file = vix9d_file
        self._vix1d_file = vix1d_file

        # Lazy-loaded caches
        self._spy_df: Optional[pd.DataFrame] = None
        self._vix_df: Optional[pd.DataFrame] = None
        self._vix9d_df: Optional[pd.DataFrame] = None
        self._vix1d_df: Optional[pd.DataFrame] = None

        # Date -> values lookup caches
        self._spy_cache: Dict[date, Dict[str, float]] = {}
        self._iv_cache: Dict[date, float] = {}

    def _load_spy(self) -> pd.DataFrame:
        if self._spy_df is None:
            path = os.path.join(self._data_dir, self._spy_file)
            self._spy_df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return self._spy_df

    def _load_vix(self) -> pd.DataFrame:
        if self._vix_df is None:
            path = os.path.join(self._data_dir, self._vix_file)
            self._vix_df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return self._vix_df

    def _load_vix9d(self) -> Optional[pd.DataFrame]:
        if self._vix9d_df is None:
            path = os.path.join(self._data_dir, self._vix9d_file)
            if os.path.exists(path):
                self._vix9d_df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return self._vix9d_df

    def _load_vix1d(self) -> Optional[pd.DataFrame]:
        if self._vix1d_df is None:
            path = os.path.join(self._data_dir, self._vix1d_file)
            if os.path.exists(path):
                self._vix1d_df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return self._vix1d_df

    def get_spy_ohlc(self, session_date: date) -> Optional[Dict[str, float]]:
        """
        Get SPY OHLC for a given date, scaled by 10 to approximate SPX.

        Returns dict with keys: open, high, low, close (all scaled by 10).
        Returns None if date not found.
        """
        if session_date in self._spy_cache:
            return self._spy_cache[session_date]

        df = self._load_spy()

        # Handle both date and datetime index
        if isinstance(session_date, datetime):
            session_date = session_date.date()

        # Find matching row
        matches = df[df.index.date == session_date]
        if matches.empty:
            return None

        row = matches.iloc[0]
        result = {
            "open": float(row["Open"]) * 10,
            "high": float(row["High"]) * 10,
            "low": float(row["Low"]) * 10,
            "close": float(row["Close"]) * 10,
        }
        self._spy_cache[session_date] = result
        return result

    def get_daily_iv(self, session_date: date, fallback: float = 0.15) -> float:
        """
        Get implied volatility for a given date.

        Priority: VIX1D (1-day) > VIX9D (9-day) > VIX (30-day).
        Returns value as decimal (e.g., 0.15 for 15% IV).

        Args:
            session_date: The trading date
            fallback: Default IV if no data found

        Returns:
            IV as decimal (VIX value / 100)
        """
        if session_date in self._iv_cache:
            return self._iv_cache[session_date]

        if isinstance(session_date, datetime):
            session_date = session_date.date()

        # Try VIX1D first (best for 0DTE — actual 1-day implied vol)
        vix1d_df = self._load_vix1d()
        if vix1d_df is not None:
            matches = vix1d_df[vix1d_df.index.date == session_date]
            if not matches.empty:
                iv = float(matches.iloc[0]["Close"]) / 100.0
                self._iv_cache[session_date] = iv
                return iv

        # Try VIX9D (9-day IV)
        vix9d_df = self._load_vix9d()
        if vix9d_df is not None:
            matches = vix9d_df[vix9d_df.index.date == session_date]
            if not matches.empty:
                iv = float(matches.iloc[0]["Close"]) / 100.0
                self._iv_cache[session_date] = iv
                return iv

        # Fall back to VIX (30-day)
        vix_df = self._load_vix()
        matches = vix_df[vix_df.index.date == session_date]
        if not matches.empty:
            iv = float(matches.iloc[0]["Close"]) / 100.0
            self._iv_cache[session_date] = iv
            return iv

        # No data found
        self._iv_cache[session_date] = fallback
        return fallback

    def has_spy_data(self, session_date: date) -> bool:
        """Check if SPY data exists for a given date."""
        return self.get_spy_ohlc(session_date) is not None

    def has_iv_data(self, session_date: date) -> bool:
        """Check if IV data exists for a given date."""
        if isinstance(session_date, datetime):
            session_date = session_date.date()

        vix1d_df = self._load_vix1d()
        if vix1d_df is not None:
            if not vix1d_df[vix1d_df.index.date == session_date].empty:
                return True

        vix9d_df = self._load_vix9d()
        if vix9d_df is not None:
            if not vix9d_df[vix9d_df.index.date == session_date].empty:
                return True

        vix_df = self._load_vix()
        return not vix_df[vix_df.index.date == session_date].empty


# Module-level singleton for convenience
_default_loader: Optional[MarketDataLoader] = None


def get_market_data_loader(data_dir: str = "data") -> MarketDataLoader:
    """Get or create the default MarketDataLoader instance."""
    global _default_loader
    if _default_loader is None or _default_loader._data_dir != data_dir:
        _default_loader = MarketDataLoader(data_dir=data_dir)
    return _default_loader
