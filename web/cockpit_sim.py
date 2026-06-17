"""Simulation harness for the Live P&L Cockpit.

The market is closed and there are no live 0DTE positions, so this module
fabricates a realistic, *internally consistent* world:

* a set of open trades across strategies (MEIC x2 tranches, Power Hour, ORB),
  shaped exactly like ``DatabaseService.get_open_trades()`` output; and
* a quote generator that prices every requested leg with Black-Scholes off a
  random-walking SPX spot, returning the same wire shape as the options_api
  ``/market-data/options-snapshot`` endpoint.

Because marks come from one moving spot, the intraday P&L curve, the payoff
graph and the positions table all move together and stay coherent.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime
from typing import Any, Dict, List

from src_v2.backtest.osi_symbol import format_osi

_MULT = 100
_NORM = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return _NORM * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(
    spot: float, strike: float, t_years: float, iv: float, is_call: bool
) -> Dict[str, float]:
    """Black-Scholes price + greeks (r=0). Returns price and per-contract greeks."""
    t = max(t_years, 1e-6)
    sig = max(iv, 1e-4)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + 0.5 * sig * sig * t) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    if is_call:
        price = spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
    gamma = _norm_pdf(d1) / (spot * sig * sqrt_t)
    vega = spot * _norm_pdf(d1) * sqrt_t / 100.0  # per 1 vol point
    theta = (-(spot * _norm_pdf(d1) * sig) / (2.0 * sqrt_t)) / 365.0  # per day
    return {
        "price": max(price, 0.0),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


class CockpitSim:
    """Holds the simulated world: a moving spot and a fixed set of positions."""

    def __init__(self, spot: float = 6000.0, iv: float = 0.22, seed: int = 7):
        self._rng = random.Random(seed)
        self.spot = spot
        self.iv = iv
        self.expiry = date.today()
        self._trades = self._make_trades()

    # ------------------------------------------------------------------ #
    #  Positions
    # ------------------------------------------------------------------ #

    def _ic_trade(
        self,
        trade_id: str,
        sp: float,
        lp: float,
        sc: float,
        lc: float,
        put_credit: float,
        call_credit: float,
        entry_hhmm: str,
        strangle_credit: float | None = None,
        stop_dollars: float | None = None,
    ) -> Dict[str, Any]:
        return {
            "trade_id": trade_id,
            "entry_time": datetime.combine(self.expiry, datetime.min.time()).replace(
                hour=int(entry_hhmm[:2]), minute=int(entry_hhmm[2:])
            ),
            "expiration_date": self.expiry,
            "underlying": "SPX",
            "underlying_price_at_entry": 6000.0,
            "short_put_strike": sp,
            "long_put_strike": lp,
            "short_call_strike": sc,
            "long_call_strike": lc,
            "put_spread_premium": put_credit,
            "call_spread_premium": call_credit,
            "total_premium": put_credit + call_credit,
            "call_wing_width": int(lc - sc),
            "put_wing_width": int(sp - lp),
            "stop_loss_amount": stop_dollars,
            "status": "open",
            # strategy_data fields (merged onto the dict, as the DB service does)
            "short_put_symbol": format_osi(self.expiry, sp, is_call=False),
            "long_put_symbol": format_osi(self.expiry, lp, is_call=False),
            "short_call_symbol": format_osi(self.expiry, sc, is_call=True),
            "long_call_symbol": format_osi(self.expiry, lc, is_call=True),
            "put_credit": put_credit,
            "call_credit": call_credit,
            "put_open_qty": 1,
            "call_open_qty": 1,
            "strangle_credit": strangle_credit,
            "stop_dollars": stop_dollars,
        }

    def _make_trades(self) -> List[Dict[str, Any]]:
        s = self.spot
        trades = [
            # Two MEIC tranches at different times / widths
            self._ic_trade("meic-100500", s - 50, s - 70, s + 50, s + 70, 1.10, 1.05, "1005"),
            self._ic_trade("meic-103000", s - 60, s - 80, s + 60, s + 80, 0.95, 0.98, "1030"),
            # Power Hour IC with combined-strangle stop
            self._ic_trade(
                "ph-T1-143000",
                s - 40,
                s - 60,
                s + 40,
                s + 60,
                1.30,
                1.25,
                "1430",
                strangle_credit=2.55,
                stop_dollars=510.0,
            ),
        ]
        # ORB directional: single put credit spread
        orb = {
            "trade_id": "orbdir-put-094500",
            "entry_time": datetime.combine(self.expiry, datetime.min.time()).replace(
                hour=9, minute=45
            ),
            "expiration_date": self.expiry,
            "underlying": "SPX",
            "underlying_price_at_entry": 6000.0,
            "short_put_strike": s - 30,
            "long_put_strike": s - 50,
            "short_call_strike": 0,
            "long_call_strike": 0,
            "put_spread_premium": 1.40,
            "call_spread_premium": 0.0,
            "total_premium": 1.40,
            "call_wing_width": 0,
            "put_wing_width": 20,
            "status": "open",
            "short_put_symbol": format_osi(self.expiry, s - 30, is_call=False),
            "long_put_symbol": format_osi(self.expiry, s - 50, is_call=False),
            "put_credit": 1.40,
            "put_open_qty": 1,
            "side": "put",
            "strategy_type": "orb_em_directional",
        }
        trades.append(orb)
        return trades

    def trades(self) -> List[Dict[str, Any]]:
        return self._trades

    # ------------------------------------------------------------------ #
    #  Quotes
    # ------------------------------------------------------------------ #

    def _t_years(self) -> float:
        """Fraction of a year to 4pm ET today; floored so 0DTE keeps extrinsic."""
        now = datetime.now()
        close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        # Floor at ~2h so 0DTE legs keep visible extrinsic value off-hours.
        hours = max((close - now).total_seconds() / 3600.0, 2.0)
        return hours / (24.0 * 365.0)

    def step_spot(self) -> None:
        """Advance the simulated SPX spot by a small random walk."""
        self.spot += self._rng.gauss(0.0, 2.5)

    def snapshot(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Return an options_api-shaped SnapshotResponse for the requested legs."""
        self.step_spot()
        t = self._t_years()
        legs_out: List[Dict[str, Any]] = []
        for leg in request_body.get("legs", []):
            sym = leg["symbol"]
            strike = float(leg.get("strike") or 0.0)
            is_call = leg.get("right") == "call"
            bs = black_scholes(self.spot, strike, t, self.iv, is_call)
            mark = round(bs["price"], 2)
            spread = max(0.05, mark * 0.02)
            legs_out.append(
                {
                    "symbol": sym,
                    "status": "ok",
                    "bid": round(max(mark - spread, 0.0), 2),
                    "ask": round(mark + spread, 2),
                    "mid": mark,
                    "last": mark,
                    "mark": mark,
                    "delta": round(bs["delta"], 4),
                    "gamma": round(bs["gamma"], 6),
                    "theta": round(bs["theta"], 4),
                    "vega": round(bs["vega"], 4),
                    "iv": self.iv,
                    "multiplier": leg.get("multiplier", _MULT),
                }
            )
        return {
            "as_of": datetime.now().isoformat(),
            "underlyings": {
                "SPX": {
                    "symbol": "SPX",
                    "last": round(self.spot, 2),
                    "bid": round(self.spot - 0.1, 2),
                    "ask": round(self.spot + 0.1, 2),
                    "mid": round(self.spot, 2),
                    "day_change_pct": round((self.spot - 6000.0) / 6000.0 * 100, 3),
                }
            },
            "legs": legs_out,
            "market_regime": {
                "spy_day_change_pct": round((self.spot - 6000.0) / 6000.0 * 100, 3),
                "vix": 15.5,
            },
            "errors": [],
        }
