"""Live SPX IV source backed by TastyTrade DXLink quotes.

For a given expiry, subscribes the ~ATM strike band, takes bid/ask mids, and computes
ATM IV/delta via Black–Scholes (same path as the historical source → identical IV
definition). The DXLink/quote access is injected as `quote_fn` so the source is fully
unit-testable offline; the real wiring (the bot's persistent DXLink Quote streamer) is
supplied by `make_tt_quote_fn` / `make_tt_spot_fn` and verified live.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from src_v2.flux.iv_compute import RATE, compute_iv_rows
from src_v2.flux.models import ChainSnapshot
from src_v2.flux.sources.base import FluxSource
from src_v2.backtest.osi_symbol import build_strike_window, parse_osi

QuoteFn = Callable[[List[str]], Dict[str, Tuple[float, float]]]


class TastyTradeLiveSource(FluxSource):
    def __init__(self, quote_fn: QuoteFn, spot_fn: Callable[[], Optional[float]],
                 half_width_pts: int = 80, step_pts: int = 5,
                 ref_date: Optional[date] = None, rate: float = RATE):
        self.quote_fn = quote_fn
        self.spot_fn = spot_fn
        self.half_width_pts = half_width_pts
        self.step_pts = step_pts
        self.ref_date = ref_date
        self.rate = rate

    def chain(self, expiry: date, on_date: Optional[date] = None) -> ChainSnapshot:
        # `on_date` is ignored: this source is always live ("now").
        now = datetime.now(timezone.utc)
        spot = self.spot_fn()
        if spot is None:
            return ChainSnapshot("SPX", expiry, now, ())
        symbols = build_strike_window(expiry=expiry, atm_strike=spot,
                                      half_width_pts=self.half_width_pts, step_pts=self.step_pts)
        quotes = self.quote_fn(symbols) or {}
        price_rows: List[Tuple[float, bool, float, float]] = []
        for sym, ba in quotes.items():
            if not ba:
                continue
            try:
                strike, is_call = parse_osi(sym.ljust(21))
            except (ValueError, Exception):
                continue
            price_rows.append((strike, is_call, float(ba[0]), float(ba[1])))
        ref = self.ref_date or date.today()
        T = max((expiry - ref).days, 0) / 365.0
        rows, _ = compute_iv_rows(price_rows, T, spot, self.rate)
        return ChainSnapshot("SPX", expiry, now, tuple(rows))
