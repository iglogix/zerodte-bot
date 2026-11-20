"""
Market data service with SQLite caching layer.
Prevents rate limiting and enables future backtesting via permanent candle storage.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import sessionmaker

from trading_core.models.database import Base, MarketDataCache, get_engine, get_session

logger = logging.getLogger(__name__)


class CachedMarketData:
    """
    Market data layer with:
    - In-memory cache for option chains and quotes (short TTL, never persisted)
    - SQLite-backed permanent storage for 5-min candles (backtesting / fallback)

    Option prices change every second — they are held in memory only long enough
    to deduplicate repeated tool calls within a single LLM decision loop (~10s).
    """

    def __init__(
        self,
        cache_db_path: str = "data/market_cache.db",
        api_client: Optional[Any] = None,
        option_chain_ttl: int = 10,
        quote_ttl: int = 10,
        account_info_ttl: int = 60,
    ):
        self.cache_db_path = cache_db_path
        self.api_client = api_client
        self.option_chain_ttl = option_chain_ttl
        self.quote_ttl = quote_ttl
        self.account_info_ttl = account_info_ttl
        self._engine = None
        self._SessionLocal = None
        # In-memory stores: {symbol: (timestamp, data)}
        self._mem_chains: Dict[str, tuple] = {}
        self._mem_quotes: Dict[str, tuple] = {}
        self._mem_account: Optional[tuple] = None

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
    # Option chain  (in-memory only — prices are stale after ~10s)
    # -------------------------------------------------------------------------

    async def cache_option_chain(
        self,
        symbol: str,
        chain: List[Dict],
        ttl_seconds: int = 10,
    ):
        self._mem_chains[symbol] = (datetime.now(), chain)

    async def get_cached_option_chain(
        self,
        symbol: str,
        max_age_seconds: int = 10,
    ) -> Optional[List[Dict]]:
        entry = self._mem_chains.get(symbol)
        if entry is None:
            return None
        ts, data = entry
        if (datetime.now() - ts).total_seconds() > max_age_seconds:
            del self._mem_chains[symbol]
            return None
        return data

    async def get_option_chain_with_fallback(
        self,
        symbol: str,
        max_age_seconds: Optional[int] = None,
    ) -> List[Dict]:
        max_age_seconds = max_age_seconds if max_age_seconds is not None else self.option_chain_ttl
        cached = await self.get_cached_option_chain(symbol, max_age_seconds)
        if cached is not None:
            logger.debug("Memory cache hit: option_chain for %s (age < %ds)", symbol, max_age_seconds)
            return cached

        logger.debug("Fetching fresh option_chain for %s from API", symbol)
        data = await self.api_client.get_option_chain(symbol)
        await self.cache_option_chain(symbol, data)
        return data

    # -------------------------------------------------------------------------
    # Quote  (in-memory only)
    # -------------------------------------------------------------------------

    async def cache_quote(
        self,
        symbol: str,
        quote: Dict,
        ttl_seconds: int = 10,
    ):
        self._mem_quotes[symbol] = (datetime.now(), quote)

    async def get_cached_quote(
        self,
        symbol: str,
        max_age_seconds: int = 10,
    ) -> Optional[Dict]:
        entry = self._mem_quotes.get(symbol)
        if entry is None:
            return None
        ts, data = entry
        if (datetime.now() - ts).total_seconds() > max_age_seconds:
            del self._mem_quotes[symbol]
            return None
        return data

    async def get_quote_with_fallback(
        self,
        symbol: str,
        max_age_seconds: Optional[int] = None,
    ) -> Dict:
        max_age_seconds = max_age_seconds if max_age_seconds is not None else self.quote_ttl
        cached = await self.get_cached_quote(symbol, max_age_seconds)
        if cached is not None:
            return cached

        data = await self.api_client.get_quote(symbol)
        await self.cache_quote(symbol, data)
        return data

    # -------------------------------------------------------------------------
    # Account info  (in-memory, 60s TTL — changes slowly)
    # -------------------------------------------------------------------------

    async def cache_account_info(self, info: Dict, ttl_seconds: int = 60):
        self._mem_account = (datetime.now(), info)

    async def get_cached_account_info(
        self,
        max_age_seconds: int = 60,
    ) -> Optional[Dict]:
        if self._mem_account is None:
            return None
        ts, data = self._mem_account
        if (datetime.now() - ts).total_seconds() > max_age_seconds:
            self._mem_account = None
            return None
        return data

    async def get_account_info_with_fallback(
        self,
        max_age_seconds: Optional[int] = None,
    ) -> Dict:
        max_age_seconds = max_age_seconds if max_age_seconds is not None else self.account_info_ttl
        cached = await self.get_cached_account_info(max_age_seconds)
        if cached is not None:
            return cached

        data = await self.api_client.get_account_info()
        await self.cache_account_info(data)
        return data

    # -------------------------------------------------------------------------
    # Candle fetch (delegates to API client, caches results)
    # -------------------------------------------------------------------------

    async def get_candles(
        self,
        symbol: str,
        interval: str = "5m",
        start: Optional[str] = None,
        end: Optional[str] = None,
        count: int = 10,
    ) -> List[Dict]:
        """Fetch candles from API client by time range or count.

        Args:
            symbol: Ticker symbol
            interval: Candle interval (e.g. "5m")
            start: Start time as "HH:MM" (today assumed) or ISO string
            end: End time as "HH:MM" (today assumed) or ISO string
            count: Number of candles if no start/end given
        """
        if self.api_client is None:
            return []

        from_time = None
        if start:
            from_time = self._parse_time(start)

        candles = await self.api_client.get_candles(
            symbol, interval=interval, from_time=from_time, count=count,
        )

        if candles and end:
            end_dt = self._parse_time(end)
            candles = [
                c for c in candles
                if datetime.fromisoformat(str(c.get("timestamp", ""))) < end_dt
            ]

        return candles or []

    @staticmethod
    def _parse_time(t: str) -> datetime:
        """Parse 'HH:MM' (today) or full ISO timestamp."""
        if len(t) <= 5:
            now = datetime.now()
            h, m = int(t[:2]), int(t[3:5])
            return now.replace(hour=h, minute=m, second=0, microsecond=0)
        return datetime.fromisoformat(t)

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
            logger.error("Candle missing timestamp, skipping")
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
        import logging as _logging
        # Always fetch fresh candles from API — candles are time-series and
        # the cache goes stale immediately after the first fetch.
        if self.api_client and hasattr(self.api_client, "get_5min_candles"):
            try:
                candles = await self.api_client.get_5min_candles(symbol, count=count)
                if candles:
                    for c in candles:
                        await self.cache_5min_candle(symbol, c)
                    return candles
            except Exception as exc:
                _logging.getLogger(__name__).warning(
                    "get_5min_candles API fetch failed for %s: %s — falling back to cache", symbol, exc
                )

        # API unavailable — serve from cache
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
        if data_type == "quote":
            return await self.get_cached_quote(symbol, max_age_seconds) is not None
        if data_type == "option_chain":
            return await self.get_cached_option_chain(symbol, max_age_seconds) is not None
        if data_type == "account_info":
            return await self.get_cached_account_info(max_age_seconds) is not None
        result = self._read_latest(symbol, data_type, max_age_seconds)
        return result is not None
