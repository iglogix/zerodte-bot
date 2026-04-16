"""
CBBO Price Store — loads Databento SPXW CBBO-1m zst files for backtesting.

Provides real bid/ask quotes per (minute, strike, is_call) for 0DTE options.
Loads one day at a time to keep memory bounded.

Usage:
    store = CbboStore("/path/to/OPRA-dir")
    store.load_day(date(2025, 9, 22))
    bid, ask = store.lookup(minute_et="09:45", strike=6445.0, is_call=False)
"""

from __future__ import annotations

import csv
import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")

# "SPXW  250922P06445000" -> exp=2025-09-22, put, strike=6445.000
SYM_RE = re.compile(r"SPXW\s+(\d{6})([CP])(\d{8})")


def _parse_symbol(raw: str) -> Optional[Tuple[date, bool, float]]:
    """Parse SPXW symbol -> (expiry, is_call, strike) or None."""
    m = SYM_RE.match(raw.strip())
    if not m:
        return None
    date_str, cp, strike_raw = m.groups()
    yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
    return date(2000 + yy, mm, dd), cp == "C", int(strike_raw) / 1000.0


# Key: (minute_str "HH:MM", strike, is_call) -> (bid, ask)
QuoteMap = Dict[Tuple[str, float, bool], Tuple[float, float]]


class CbboStore:
    """Loads and serves 0DTE SPX option quotes from Databento CBBO zst files.

    Supports fast pickle cache: if cache_dir is set, loads pre-extracted
    pickle files instead of decompressing zst (100x faster).
    """

    def __init__(self, opra_dir: str | Path, cache_dir: str | Path = "",
                 preloaded: Optional[Dict] = None):
        self.opra_dir = Path(opra_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._quotes: QuoteMap = {}
        self._loaded_date: Optional[date] = None
        self._strikes: set[float] = set()
        # In-memory cache: {date: QuoteMap} — avoids disk I/O per day
        self._mem_cache: Optional[Dict] = preloaded

    @staticmethod
    def preload_cache(cache_dir: str | Path) -> Dict:
        """Load all pickle files into memory. Returns {date: QuoteMap} dict.

        ~625MB for 125 days. Call once at startup, pass to constructor as preloaded=.
        """
        import pickle
        cache_path = Path(cache_dir)
        mem: Dict = {}
        for pkl in sorted(cache_path.glob("*.pkl")):
            d = date.fromisoformat(pkl.stem)
            with open(pkl, "rb") as f:
                mem[d] = pickle.load(f)
        return mem

    @property
    def loaded_date(self) -> Optional[date]:
        return self._loaded_date

    @property
    def available_strikes(self) -> set[float]:
        return self._strikes

    def load_day(self, trading_date: date) -> int:
        """Load 0DTE quotes for a single trading day. Returns record count.

        Tries pickle cache first (fast), falls back to zst decompression (slow).
        """
        self._quotes = {}
        self._strikes = set()
        self._loaded_date = trading_date

        # Fastest path: in-memory preloaded cache (reference, not copy — do not mutate)
        if self._mem_cache is not None and trading_date in self._mem_cache:
            self._quotes = self._mem_cache[trading_date]
            self._strikes = {k[1] for k in self._quotes}
            return len(self._quotes)

        # Fast path: pickle cache on disk
        if self.cache_dir:
            import pickle
            pkl_path = self.cache_dir / f"{trading_date.isoformat()}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    self._quotes = pickle.load(f)
                self._strikes = {k[1] for k in self._quotes}
                return len(self._quotes)

        # Slow path: decompress zst
        ds = trading_date.strftime("%Y%m%d")
        zst_path = self.opra_dir / f"opra-pillar-{ds}.cbbo-1m.csv.zst"
        if not zst_path.exists():
            return 0

        exp_date_str = trading_date.strftime("%y%m%d")  # "250922" for matching

        # Stream decompress and parse
        proc = subprocess.Popen(
            ["zstd", "-d", "--stdout", str(zst_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        count = 0
        reader = csv.DictReader(proc.stdout)
        for row in reader:
            # Quick filter: only 0DTE symbols
            sym_raw = row.get("symbol", "")
            if exp_date_str not in sym_raw:
                continue

            bid = float(row.get("bid_px_00") or 0)
            ask = float(row.get("ask_px_00") or 0)
            if bid <= 0 and ask <= 0:
                continue

            parsed = _parse_symbol(sym_raw)
            if parsed is None:
                continue
            expiry, is_call, strike = parsed
            if expiry != trading_date:
                continue

            # Convert UTC timestamp to ET minute
            ts_str = row.get("ts_recv", "")
            if not ts_str:
                continue
            # "2025-09-22T13:31:00.000000000Z" -> truncate to second
            try:
                dt_utc = datetime.fromisoformat(ts_str[:19]).replace(tzinfo=UTC)
                dt_et = dt_utc.astimezone(ET)
            except (ValueError, IndexError):
                continue

            minute_key = dt_et.strftime("%H:%M")

            # Store quote (last write wins for same minute — CBBO is already 1m agg)
            key = (minute_key, strike, is_call)
            self._quotes[key] = (bid, ask)
            self._strikes.add(strike)
            count += 1

        proc.wait()
        return count

    def lookup(
        self,
        minute_et: str,
        strike: float,
        is_call: bool,
        max_stale_minutes: int = 5,
    ) -> Optional[Tuple[float, float]]:
        """Look up (bid, ask) for given minute/strike/type.

        If exact minute not found, searches back up to max_stale_minutes.
        Returns None if no quote found.
        """
        key = (minute_et, strike, is_call)
        if key in self._quotes:
            return self._quotes[key]

        # Search backward for stale quote
        hh, mm = int(minute_et[:2]), int(minute_et[3:5])
        for offset in range(1, max_stale_minutes + 1):
            prev_mm = mm - offset
            prev_hh = hh
            while prev_mm < 0:
                prev_mm += 60
                prev_hh -= 1
            if prev_hh < 0:
                break
            prev_key = (f"{prev_hh:02d}:{prev_mm:02d}", strike, is_call)
            if prev_key in self._quotes:
                return self._quotes[prev_key]

        return None

    def spread_quote(
        self,
        minute_et: str,
        short_strike: float,
        long_strike: float,
        is_call: bool,
        max_stale_minutes: int = 5,
    ) -> Optional[Tuple[float, float]]:
        """Get credit spread entry and exit quotes.

        Returns (entry_credit, current_debit) or None if quotes missing.
        Entry credit = short_bid - long_ask (conservative: sell at bid, buy at ask)
        Current debit = short_ask - long_bid (conservative: buy at ask, sell at bid)
        """
        short_q = self.lookup(minute_et, short_strike, is_call, max_stale_minutes)
        long_q = self.lookup(minute_et, long_strike, is_call, max_stale_minutes)
        if short_q is None or long_q is None:
            return None

        short_bid, short_ask = short_q
        long_bid, long_ask = long_q

        # Credit spread: sell short, buy long
        entry_credit = short_bid - long_ask  # worst-case entry
        current_debit = short_ask - long_bid  # worst-case exit (buy back)

        return (round(entry_credit, 2), round(current_debit, 2))

    def mid_spread_quote(
        self,
        minute_et: str,
        short_strike: float,
        long_strike: float,
        is_call: bool,
        max_stale_minutes: int = 5,
    ) -> Optional[Tuple[float, float]]:
        """Get credit spread quotes at mid prices.

        Returns (mid_credit, mid_debit) or None.
        """
        short_q = self.lookup(minute_et, short_strike, is_call, max_stale_minutes)
        long_q = self.lookup(minute_et, long_strike, is_call, max_stale_minutes)
        if short_q is None or long_q is None:
            return None

        short_mid = (short_q[0] + short_q[1]) / 2
        long_mid = (long_q[0] + long_q[1]) / 2

        credit = short_mid - long_mid
        debit = credit  # at mid, entry credit == exit debit (same mid)

        return (round(credit, 2), round(debit, 2))

    def atm_straddle(
        self,
        minute_et: str,
        spot: float,
        max_stale_minutes: int = 5,
    ) -> Optional[float]:
        """Get ATM straddle price (put_mid + call_mid at closest strike to spot).

        This IS the market's 0DTE expected move — no model needed.
        Returns straddle price in points, or None if quotes unavailable.
        """
        if not self._strikes:
            return None
        atm_k = min(self._strikes, key=lambda k: abs(k - spot))
        put_q = self.lookup(minute_et, atm_k, False, max_stale_minutes)
        call_q = self.lookup(minute_et, atm_k, True, max_stale_minutes)
        if put_q is None or call_q is None:
            return None
        put_mid = (put_q[0] + put_q[1]) / 2
        call_mid = (call_q[0] + call_q[1]) / 2
        return put_mid + call_mid

    def derive_spot(
        self,
        minute_et: str,
        bar_price: float,
        max_stale_minutes: int = 5,
    ) -> Optional[float]:
        """Derive index spot from put-call parity at the ATM strike.

        For 0DTE: S ≈ K + C_mid - P_mid (discount ≈ 1).
        Uses *bar_price* only to find the nearest ATM strike.
        Returns None if quotes are unavailable.
        """
        if not self._strikes:
            return None
        atm_k = min(self._strikes, key=lambda k: abs(k - bar_price))
        put_q = self.lookup(minute_et, atm_k, False, max_stale_minutes)
        call_q = self.lookup(minute_et, atm_k, True, max_stale_minutes)
        if put_q is None or call_q is None:
            return None
        c_mid = (call_q[0] + call_q[1]) / 2
        p_mid = (put_q[0] + put_q[1]) / 2
        return round(atm_k + c_mid - p_mid, 2)

    def has_day(self, trading_date: date) -> bool:
        """Check if zst file exists for this date."""
        ds = trading_date.strftime("%Y%m%d")
        return (self.opra_dir / f"opra-pillar-{ds}.cbbo-1m.csv.zst").exists()

    def available_days(self) -> list[date]:
        """List all trading dates with available CBBO data."""
        days = []
        for f in sorted(self.opra_dir.glob("opra-pillar-*.cbbo-1m.csv.zst")):
            ds = f.name.split("-")[2].replace(".cbbo", "")
            try:
                days.append(date(int(ds[:4]), int(ds[4:6]), int(ds[6:8])))
            except (ValueError, IndexError):
                continue
        return days
