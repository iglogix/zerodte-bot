"""
Market data service with SQLite caching layer.
Prevents rate limiting and enables future backtesting via permanent candle storage.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, create_engine, event, text
from sqlalchemy.orm import sessionmaker

from src.models.database import Base, MarketDataCache, get_engine, get_session

logger = logging.getLogger(__name__)


class CachedMarketData:
    """
    Market data layer with SQLite-backed caching.

    - Option prices / quotes: 60s TTL
    - 5-min candles: permanent (no TTL), append-only, deduplicated by timestamp
    """

    def __init__(
        self,
        cache_db_path: str = "data/market_cache.db",
        api_client: Optional[Any] = None,
    ):
        self.cache_db_path = cache_db_path
        self.api_client = api_client
        self._engine = None
        self._SessionLocal = None

    async def initialize(self):
        """Initialize the cache database."""
        self._engine = get_engine(self.cache_db_path)
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)
        logger.debug("CachedMarketData initialized at %s", self.cache_db_path)

    async def close(self):
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _session(self):
        return get_session(self._engine)

    def _write(self, symbol: str, data_type: str, data: Any, ttl_seconds: Optional[int]):
        with self._session() as session:
            entry = MarketDataCache(
                symbol=symbol,
                data_type=data_type,
                timestamp=datetime.now(),
                data_json=json.dumps(data, default=str),
                ttl_seconds=ttl_seconds,
            )
            session.add(entry)

    def _read_latest(
        self,
        symbol: str,
        data_type: str,
        max_age_seconds: Optional[int] = None,
    ) -> Optional[Any]:
        with self._session() as session:
            q = (
                session.query(MarketDataCache)
                .filter_by(symbol=symbol, data_type=data_type)
                .order_by(MarketDataCache.timestamp.desc())
                .first()
            )
            if q is None:
                return None
            if max_age_seconds is not None:
                age = (datetime.now() - q.timestamp).total_seconds()
                if age > max_age_seconds:
                    return None
            return json.loads(q.data_json)

    # -------------------------------------------------------------------------
    # Option chain
    # -------------------------------------------------------------------------

    async def cache_option_chain(
        self,
        symbol: str,
        chain: List[Dict],
        ttl_seconds: int = 60,
    ):
        self._write(symbol, "option_chain", chain, ttl_seconds)

    async def get_cached_option_chain(
        self,
        symbol: str,
        max_age_seconds: int = 60,
    ) -> Optional[List[Dict]]:
        return self._read_latest(symbol, "option_chain", max_age_seconds)

    async def get_option_chain_with_fallback(
        self,
        symbol: str,
        max_age_seconds: int = 60,
    ) -> List[Dict]:
        cached = await self.get_cached_option_chain(symbol, max_age_seconds)
        if cached is not None:
            logger.debug("Cache hit: option_chain for %s", symbol)
            return cached

        logger.debug("Cache miss: fetching option_chain for %s from API", symbol)
        data = await self.api_client.get_option_chain(symbol)
        await self.cache_option_chain(symbol, data)
        return data

    # -------------------------------------------------------------------------
    # Quote
    # -------------------------------------------------------------------------

    async def cache_quote(
        self,
        symbol: str,
        quote: Dict,
        ttl_seconds: int = 60,
    ):
        self._write(symbol, "quote", quote, ttl_seconds)

    async def get_cached_quote(
        self,
        symbol: str,
        max_age_seconds: int = 60,
    ) -> Optional[Dict]:
        return self._read_latest(symbol, "quote", max_age_seconds)

    async def get_quote_with_fallback(
        self,
        symbol: str,
        max_age_seconds: int = 60,
    ) -> Dict:
        cached = await self.get_cached_quote(symbol, max_age_seconds)
        if cached is not None:
            return cached

        data = await self.api_client.get_quote(symbol)
        await self.cache_quote(symbol, data)
        return data

    # -------------------------------------------------------------------------
    # Account info
    # -------------------------------------------------------------------------

    async def cache_account_info(self, info: Dict, ttl_seconds: int = 60):
        self._write("ACCOUNT", "account_info", info, ttl_seconds)

    async def get_cached_account_info(
        self,
        max_age_seconds: int = 60,
    ) -> Optional[Dict]:
        return self._read_latest("ACCOUNT", "account_info", max_age_seconds)

    async def get_account_info_with_fallback(
        self,
        max_age_seconds: int = 60,
    ) -> Dict:
        cached = await self.get_cached_account_info(max_age_seconds)
        if cached is not None:
            return cached

        data = await self.api_client.get_account_info()
        await self.cache_account_info(data)
        return data

    # -------------------------------------------------------------------------
    # 5-min candles (permanent, append-only, deduplicated)
    # -------------------------------------------------------------------------

    async def cache_5min_candle(
        self,
        symbol: str,
        candle: Dict,
        permanent: bool = True,
    ):
        """Store a 5-min candle. Deduplicates by (symbol, data_type, timestamp)."""
        ts = candle.get("timestamp")
        if ts is None:
            logger.warning("Candle missing timestamp, skipping")
            return

        candle_ts = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
        candle_ts = candle_ts.replace(microsecond=0)

        with self._session() as session:
            existing = (
                session.query(MarketDataCache)
                .filter_by(symbol=symbol, data_type="5min_candle", timestamp=candle_ts)
                .first()
            )
            if existing:
                return

            entry = MarketDataCache(
                symbol=symbol,
                data_type="5min_candle",
                timestamp=candle_ts,
                data_json=json.dumps(candle, default=str),
                ttl_seconds=None,
            )
            session.add(entry)

    async def get_5min_candles(
        self,
        symbol: str,
        count: int = 10,
        ignore_ttl: bool = True,
    ) -> List[Dict]:
        with self._session() as session:
            rows = (
                session.query(MarketDataCache)
                .filter_by(symbol=symbol, data_type="5min_candle")
                .order_by(MarketDataCache.timestamp.asc())
                .all()
            )
            result = [json.loads(r.data_json) for r in rows]
            return result[-count:] if count < len(result) else result

    async def get_5min_candles_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict]:
        with self._session() as session:
            rows = (
                session.query(MarketDataCache)
                .filter(
                    MarketDataCache.symbol == symbol,
                    MarketDataCache.data_type == "5min_candle",
                    MarketDataCache.timestamp >= start_time,
                    MarketDataCache.timestamp <= end_time,
                )
                .order_by(MarketDataCache.timestamp.asc())
                .all()
            )
            return [json.loads(r.data_json) for r in rows]

    # -------------------------------------------------------------------------
    # Cache validity check
    # -------------------------------------------------------------------------

    async def is_cache_valid(
        self,
        symbol: str,
        data_type: str,
        max_age_seconds: int = 60,
    ) -> bool:
        result = self._read_latest(symbol, data_type, max_age_seconds)
        return result is not None
