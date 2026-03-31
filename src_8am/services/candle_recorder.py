"""
CandleRecorder — saves 1-min ES candles to CSV during live trading.

Also logs a 3-way option price comparison (Market vs BS vs ANN) every 5 min.

The CSV file is appended to throughout the trading day:
  data/candles_1m/YYYY-MM-DD.csv

Price comparison log:
  data/iv_comparison/YYYY-MM-DD.jsonl
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class CandleRecorder:
    """Background service that records 1-min candles and compares pricing models."""

    def __init__(
        self,
        market_data,       # OrbMarketData
        config: Dict[str, Any],
        options_service=None,  # OrbOptionsService (for option quotes)
        tt_client=None,
    ):
        self._md = market_data
        self._config = config
        self._options_service = options_service
        self._tt = tt_client
        self._task: Optional[asyncio.Task] = None

        # Candle CSV
        candle_dir = Path("data/candles_1m")
        candle_dir.mkdir(parents=True, exist_ok=True)
        self._candle_path = candle_dir / f"{date.today().isoformat()}.csv"
        self._candle_written = set()  # track timestamps to avoid duplicates
        self._reauth_interval = 600  # re-authenticate every 10 min
        self._reauth_counter = 0  # ticks since last re-auth (each tick = 60s)

        # Price comparison log
        iv_dir = Path("data/iv_comparison")
        iv_dir.mkdir(parents=True, exist_ok=True)
        self._iv_path = iv_dir / f"{date.today().isoformat()}.jsonl"

        # Load PriceANN v3 model (direct bid/ask prediction)
        self._price_ann = None
        opt = config.get("options", {})
        ann_model_path = opt.get("price_ann_model", "models/price_ann_v3.pt")
        if ann_model_path and Path(ann_model_path).exists():
            try:
                from scripts.train_price_ann import PriceANN
                self._price_ann = PriceANN.load(ann_model_path)
                logger.info("CandleRecorder: loaded PriceANN from %s", ann_model_path)
            except Exception as exc:
                logger.warning("CandleRecorder: failed to load PriceANN: %s", exc)

    def start(self):
        """Start the background recording loop."""
        self._task = asyncio.ensure_future(self._record_loop())
        logger.info("CandleRecorder started — saving to %s", self._candle_path)

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CandleRecorder stopped")

    async def _record_loop(self):
        """Poll 1-min candles every 60s, save to CSV, compare IV every 5 min."""
        iv_counter = 0
        while True:
            try:
                # Re-authenticate every 10 min to avoid token expiry (15 min lifetime)
                self._reauth_counter += 1
                if self._reauth_counter >= 10 and self._tt:
                    self._reauth_counter = 0
                    await self._tt.refresh_session()

                await self._save_candles()
                iv_counter += 1
                if iv_counter >= 5:  # IV comparison every 5 minutes
                    await self._compare_prices()
                    iv_counter = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("CandleRecorder error")
            await asyncio.sleep(60)

    async def _save_candles(self):
        """Fetch latest 1-min candle(s) and append to CSV."""
        candles = await self._md.get_1min_candles(count=3)  # last 3 to catch any missed
        if not candles:
            return

        write_header = not self._candle_path.exists()
        with open(self._candle_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "open", "high", "low", "close", "volume"
            ])
            if write_header:
                writer.writeheader()
            for c in candles:
                ts = c.get("timestamp", "")
                if ts in self._candle_written:
                    continue
                self._candle_written.add(ts)
                writer.writerow({
                    "timestamp": ts,
                    "open": c.get("open"),
                    "high": c.get("high"),
                    "low": c.get("low"),
                    "close": c.get("close"),
                    "volume": c.get("volume"),
                })
        logger.debug("Saved %d candle(s) to %s", len(candles), self._candle_path)

    async def _compare_prices(self):
        """Fetch option quotes and compare: Market bid/ask vs BS theo vs ANN bid/ask."""
        if not self._tt:
            return

        try:
            # Get current underlying price
            price = await self._md.get_current_price()
            if not price or price <= 0:
                return

            # Compute TTE
            now_et = datetime.now(ET)
            close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            tte_hours = max(0.01, (close_dt - now_et).total_seconds() / 3600)
            tte_years = tte_hours / (252 * 6.5)

            # Get option chain
            product_code = self._config.get("bot", {}).get("symbol", "ES")
            chain = await self._tt.get_future_option_chain(
                product_code=product_code,
                underlying_price=price,
            )
            if not chain:
                return

            comparisons = []

            for opt in chain:
                strike = opt.get("strike", 0)
                bid = opt.get("bid", 0)
                ask = opt.get("ask", 0)
                opt_type = opt.get("option_type", "")
                delta = opt.get("delta", 0)

                if bid <= 0 or ask <= 0:
                    continue
                mid = (bid + ask) / 2
                if mid < 0.05:
                    continue

                moneyness = price / strike if strike > 0 else 1.0
                if moneyness < 0.95 or moneyness > 1.05:
                    continue

                # --- BS price (using market IV) ---
                market_iv = self._bs_iv(mid, price, strike, tte_years, opt_type)
                bs_price = self._bs_price(price, strike, tte_years, market_iv, opt_type) if market_iv else None

                # --- ANN price (direct bid/ask prediction) ---
                ann_bid = ann_ask = ann_mid = None
                if self._price_ann:
                    try:
                        is_call = opt_type == "call"
                        ann_bid, ann_ask = self._price_ann.predict_bid_ask(
                            product_code, price, strike, tte_hours, is_call
                        )
                        ann_mid = (ann_bid + ann_ask) / 2
                    except Exception:
                        pass

                rec = {
                    "timestamp": now_et.isoformat(),
                    "underlying": round(price, 2),
                    "strike": strike,
                    "type": opt_type,
                    "moneyness": round(moneyness, 4),
                    "tte_hours": round(tte_hours, 2),
                    # Market
                    "market_bid": bid,
                    "market_ask": ask,
                    "market_mid": round(mid, 2),
                    # BS (using market IV)
                    "market_iv": round(market_iv, 5) if market_iv else None,
                    "bs_price": round(bs_price, 2) if bs_price is not None else None,
                    # ANN (direct prediction)
                    "ann_bid": round(ann_bid, 2) if ann_bid is not None else None,
                    "ann_ask": round(ann_ask, 2) if ann_ask is not None else None,
                    "ann_mid": round(ann_mid, 2) if ann_mid is not None else None,
                    # TT greeks
                    "delta": round(delta, 4) if delta else None,
                    # Errors (vs market mid)
                    "bs_err": round(bs_price - mid, 2) if bs_price is not None else None,
                    "ann_err": round(ann_mid - mid, 2) if ann_mid is not None else None,
                    "ann_bid_err": round(ann_bid - bid, 2) if ann_bid is not None else None,
                    "ann_ask_err": round(ann_ask - ask, 2) if ann_ask is not None else None,
                }
                comparisons.append(rec)

            if comparisons:
                with open(self._iv_path, "a") as f:
                    for c in comparisons:
                        f.write(json.dumps(c) + "\n")

                # Summary: avg absolute errors
                ann_errs = [abs(c["ann_err"]) for c in comparisons if c["ann_err"] is not None]
                bs_errs = [abs(c["bs_err"]) for c in comparisons if c["bs_err"] is not None]
                avg_ann = sum(ann_errs) / len(ann_errs) if ann_errs else 0
                avg_bs = sum(bs_errs) / len(bs_errs) if bs_errs else 0
                logger.info(
                    "Price comparison: %d options | BS avg err=%.2f | ANN avg err=%.2f",
                    len(comparisons), avg_bs, avg_ann,
                )

        except Exception:
            logger.exception("Price comparison failed")

    @staticmethod
    def _bs_price(s, k, t, iv, opt_type):
        """BS theoretical price given IV."""
        if t <= 0 or iv is None or iv <= 0:
            return None

        def _ncdf(x):
            a = abs(x)
            t2 = 1.0 / (1.0 + 0.2316419 * a)
            poly = t2 * (0.319381530 + t2 * (-0.356563782 + t2 * (
                1.781477937 + t2 * (-1.821255978 + t2 * 1.330274429))))
            cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
            return cdf if x >= 0 else 1.0 - cdf

        sqrt_t = math.sqrt(t)
        d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
        d2 = d1 - iv * sqrt_t
        if opt_type == "call":
            return s * _ncdf(d1) - k * _ncdf(d2)
        else:
            return k * _ncdf(-d2) - s * _ncdf(-d1)

    @staticmethod
    def _bs_iv(market_price, s, k, t, opt_type, tol=1e-6, max_iter=50):
        """Newton-Raphson IV solver."""
        if t <= 0 or market_price <= 0:
            return None
        intrinsic = max(s - k, 0.0) if opt_type == "call" else max(k - s, 0.0)
        if market_price < intrinsic - 0.01:
            return None

        def _ncdf(x):
            a = abs(x)
            t2 = 1.0 / (1.0 + 0.2316419 * a)
            poly = t2 * (0.319381530 + t2 * (-0.356563782 + t2 * (
                1.781477937 + t2 * (-1.821255978 + t2 * 1.330274429))))
            cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
            return cdf if x >= 0 else 1.0 - cdf

        iv = 0.20
        diff = 0.0
        for _ in range(max_iter):
            sqrt_t = math.sqrt(t)
            d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
            d2 = d1 - iv * sqrt_t
            if opt_type == "call":
                price = s * _ncdf(d1) - k * _ncdf(d2)
            else:
                price = k * _ncdf(-d2) - s * _ncdf(-d1)
            vega = s * sqrt_t * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
            if vega < 1e-12:
                return None
            diff = price - market_price
            iv -= diff / vega
            if iv <= 0.001:
                iv = 0.001
            if abs(diff) < tol:
                return iv
        return iv if abs(diff) < tol * 10 else None
