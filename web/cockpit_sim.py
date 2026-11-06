"""Simulation harness for the Live P&L Cockpit.

The market is closed and there are no live 0DTE positions, so this module
replays a *full trading session* on a simulated clock:

* a pre-generated 1-minute SPX path from the open (09:30) to the close (16:00);
* a simulated clock that advances a few session-minutes per poll, so you can
  watch the whole day unfold in a couple of minutes of wall-clock time;
* Black-Scholes marks priced at the *simulated* time-to-close, so theta decays
  realistically as the session progresses and legs converge to intrinsic at 16:00;
* positions that activate at their real entry times (ORB 09:45, MEIC 10:05/10:30,
  Power Hour 14:30) — before entry a position has no quotes and no P&L.

Everything is driven by the one simulated spot path, so the intraday P&L curve,
the payoff graph and the positions table stay coherent. The wire shape matches
the options_api ``/market-data/options-snapshot`` response.
"""

from __future__ import annotations

import math
import os
import random
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List

from zerodte.backtest.osi_symbol import format_osi

_MULT = 100
_NORM = 1.0 / math.sqrt(2.0 * math.pi)

_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(16, 0)
_SESSION_MINUTES = 390  # 09:30 -> 16:00
_MINUTES_PER_YEAR = 525600.0


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
    """Replays one trading session: a simulated clock, spot path and positions."""

    def __init__(
        self,
        spot: float = 6000.0,
        iv: float = 0.22,
        seed: int = 7,
        minutes_per_poll: int | None = None,
        daily_vol: float = 0.012,
    ):
        self._rng = random.Random(seed)
        self.start_spot = spot
        self.iv = iv
        self.expiry = date.today()
        if minutes_per_poll is None:
            minutes_per_poll = int(os.environ.get("SIM_MINUTES_PER_POLL", "6"))
        self.minutes_per_poll = max(1, minutes_per_poll)
        self._path = self._gen_path(spot, daily_vol)
        self.sim_minute = 0
        self._trades = self._make_trades()

    # ------------------------------------------------------------------ #
    #  Session clock + spot path
    # ------------------------------------------------------------------ #

    def _gen_path(self, spot: float, daily_vol: float) -> List[float]:
        """Pre-generate a 1-minute GBM spot path for the whole session."""
        sig_min = daily_vol / math.sqrt(_SESSION_MINUTES)
        path = [spot]
        for _ in range(_SESSION_MINUTES):
            shock = self._rng.gauss(0.0, sig_min) - 0.5 * sig_min * sig_min
            path.append(path[-1] * math.exp(shock))
        return path

    @property
    def spot(self) -> float:
        return self._path[min(self.sim_minute, _SESSION_MINUTES)]

    def current_dt(self) -> datetime:
        return datetime.combine(self.expiry, _SESSION_OPEN) + timedelta(minutes=self.sim_minute)

    def _t_years(self) -> float:
        minutes_left = max(_SESSION_MINUTES - self.sim_minute, 0)
        return max(minutes_left / _MINUTES_PER_YEAR, 1e-6)

    def at_close(self) -> bool:
        return self.sim_minute >= _SESSION_MINUTES

    def seek(self, minute: int) -> None:
        self.sim_minute = max(0, min(minute, _SESSION_MINUTES))

    def reset(self) -> None:
        self.sim_minute = 0

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
        s = self.start_spot
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

    def _entry_minute_of(self, trade: Dict[str, Any]) -> int:
        et = trade["entry_time"]
        minute = (et.hour - _SESSION_OPEN.hour) * 60 + (et.minute - _SESSION_OPEN.minute)
        return max(0, min(minute, _SESSION_MINUTES))

    def trades(self) -> List[Dict[str, Any]]:
        return self._trades

    # ------------------------------------------------------------------ #
    #  Quotes
    # ------------------------------------------------------------------ #

    def snapshot(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Return an options_api-shaped SnapshotResponse at the current sim time."""
        spot = self.spot
        t = self._t_years()
        as_of = self.current_dt()
        # Every option has a live quote all session — entry timing gates the
        # *position's* P&L (below), not the quote feed. (Strategies can share a
        # strike, so gating quotes by symbol would be wrong.)
        legs_out: List[Dict[str, Any]] = []
        for leg in request_body.get("legs", []):
            sym = leg["symbol"]
            strike = float(leg.get("strike") or 0.0)
            is_call = leg.get("right") == "call"
            bs = black_scholes(spot, strike, t, self.iv, is_call)
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

        active_trade_ids = [
            t["trade_id"] for t in self._trades if self._entry_minute_of(t) <= self.sim_minute
        ]
        day_change = (spot - self._path[0]) / self._path[0] * 100
        result = {
            "as_of": as_of.isoformat(),
            "sim": {
                "clock": as_of.isoformat(),
                "minute": self.sim_minute,
                "total": _SESSION_MINUTES,
                "minutes_per_poll": self.minutes_per_poll,
                "at_close": self.at_close(),
                "active_trade_ids": active_trade_ids,
            },
            "underlyings": {
                "SPX": {
                    "symbol": "SPX",
                    "last": round(spot, 2),
                    "bid": round(spot - 0.1, 2),
                    "ask": round(spot + 0.1, 2),
                    "mid": round(spot, 2),
                    "day_change_pct": round(day_change, 3),
                }
            },
            "legs": legs_out,
            "market_regime": {
                "spy_day_change_pct": round(day_change, 3),
                "vix": 15.5,
            },
            "errors": [],
        }
        # Advance the simulated clock for the next poll (hold at the close).
        self.sim_minute = min(self.sim_minute + self.minutes_per_poll, _SESSION_MINUTES)
        return result
