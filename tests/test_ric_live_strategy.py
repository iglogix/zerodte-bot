"""Tests for the live ReverseIcStrategy (src_v2/strategies/reverse_ic.py)."""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src_v2.strategies.reverse_ic import ReverseIcStrategy

ET = ZoneInfo("America/New_York")
SPOT = 4000.0


def make_config(**strategy_overrides):
    strat = {
        "gap_min_pct": 0.25,
        "gap_min_pct_by_weekday": {},
        "vix_floor": 15,
        "long_call_delta": 0.50,
        "long_put_delta": 0.35,
        "call_wing_offset_pts": 20,
        "put_wing_offset_pts": 35,
        "profit_target_pct": 0.30,
        "setup_min_upside_pct": 0.25,
        "skip_fomc_days": True,
        "qty": 4,
        "scale_out": {"enabled": False},
    }
    strat.update(strategy_overrides)
    return {"strategy": strat, "options": {"contract_multiplier": 100}}


def bs_price(s, k, sigma, t_years, is_call):
    sq = sigma * math.sqrt(t_years)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t_years) / sq
    d2 = d1 - sq

    def n(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if is_call:
        return s * n(d1) - k * n(d2)
    return k * n(-d2) - s * n(-d1)


def make_chain(spot=SPOT, iv=0.20):
    """BS-priced 0DTE chain at ~09:33 (no greeks — the strategy inverts mids)."""
    t_years = 387 / (60 * 24 * 365)
    chain = []
    for k in [3800.0 + 5 * i for i in range(81)]:
        for ot, is_call in (("call", True), ("put", False)):
            mid = bs_price(spot, k, iv, t_years, is_call)
            chain.append({"strike": k, "option_type": ot,
                          "bid": round(max(0.0, mid - 0.05), 3),
                          "ask": round(mid + 0.05, 3)})
    return chain


def make_strategy(prior_close=3980.0, vix=18.0, at="09:33",
                  day=(2026, 6, 9), **overrides):
    s = ReverseIcStrategy()
    s.configure(make_config(**overrides))
    h, m = int(at[:2]), int(at[3:5])
    s._now_fn = lambda: datetime(*day, h, m, tzinfo=ET)

    async def _pc():
        return prior_close

    async def _vix():
        return vix

    s._prior_daily_close = _pc
    s._vix_quote = _vix
    return s


async def test_search_gap_up_builds_ric():
    s = make_strategy()                       # +0.50% gap, VIX 18
    out = await s.search_candidates(make_chain(), spot=SPOT)
    setup = out["top_candidate"]
    assert setup is not None
    st = setup.strikes
    assert abs(st["long_call"] - SPOT) <= 10              # ~50-delta
    assert SPOT - 20 <= st["long_put"] < SPOT             # ~35-delta
    assert st["short_call"] == st["long_call"] + 20
    assert st["short_put"] == st["long_put"] - 35
    assert setup.extra["net_debit"] > 0
    assert setup.total_credit == -setup.extra["net_debit"]
    assert setup.extra["gap_pct"] == pytest.approx(0.50, abs=0.05)
    val = await s.validate(setup, make_chain())
    assert val["passed"], val["failures"]


async def test_search_skips_gap_down():
    s = make_strategy(prior_close=4005.0)
    out = await s.search_candidates(make_chain(), spot=SPOT)
    assert out["top_candidate"] is None


async def test_search_skips_small_gap():
    s = make_strategy(prior_close=3996.0)     # +0.10%
    out = await s.search_candidates(make_chain(), spot=SPOT)
    assert out["top_candidate"] is None


async def test_search_skips_low_vix():
    s = make_strategy(vix=13.0)
    out = await s.search_candidates(make_chain(), spot=SPOT)
    assert out["top_candidate"] is None


async def test_search_fails_closed_without_prior_close():
    s = make_strategy(prior_close=None)
    out = await s.search_candidates(make_chain(), spot=SPOT)
    assert out["top_candidate"] is None


async def test_search_skips_fomc_day():
    s = make_strategy(day=(2023, 6, 14))      # FOMC
    out = await s.search_candidates(make_chain(), spot=SPOT)
    assert out["top_candidate"] is None


async def test_setup_filter_rejects_pumped_iv():
    s = make_strategy()
    out = await s.search_candidates(make_chain(iv=0.80), spot=SPOT)
    assert out["top_candidate"] is None


# ---------------------------------------------------------------------------
# check_exit
# ---------------------------------------------------------------------------

def _trade(debit=10.0, qty=4, scale_out=False):
    return {
        "trade_id": "t1",
        "long_call_symbol": "LC", "short_call_symbol": "SC",
        "long_put_symbol": "LP", "short_put_symbol": "SP",
        "net_debit": debit,
        "quantity": qty,
        "call_open_qty": qty, "put_open_qty": qty,
        "scale_out": scale_out,
    }


def _quotes(value):
    """Quotes such that the combo (LC-SC)+(LP-SP) mids equal `value`."""
    half = value / 2
    return {
        "LC": {"bid": half + 1.0 - 0.05, "ask": half + 1.0 + 0.05},
        "SC": {"bid": 1.0 - 0.05, "ask": 1.0 + 0.05},
        "LP": {"bid": half + 2.0 - 0.05, "ask": half + 2.0 + 0.05},
        "SP": {"bid": 2.0 - 0.05, "ask": 2.0 + 0.05},
    }


def test_exit_profit_target_single():
    s = make_strategy()
    act = s.check_exit(_trade(), _quotes(13.1))   # >= 10 x 1.30
    assert act.action == "take_profit"
    assert act.leg_quantities == {sym: 4 for sym in ("LC", "SC", "LP", "SP")}


def test_exit_holds_below_target_and_never_stops():
    s = make_strategy()
    assert s.check_exit(_trade(), _quotes(12.9)).action == "hold"
    # deep loss: NO stop — hold to settlement
    assert s.check_exit(_trade(), _quotes(0.5)).action == "hold"


def test_exit_scale_out_tiers_then_runner_holds():
    s = make_strategy(scale_out={"enabled": True, "tiers": [
        {"profit": 0.20, "close_frac": 0.50},
        {"profit": 0.30, "close_frac": 0.25},
    ]})
    t = _trade(scale_out=True)
    act1 = s.check_exit(t, _quotes(12.1))         # +21% -> close half
    assert act1.action == "take_profit"
    assert act1.leg_quantities["LC"] == 2
    t["call_open_qty"] = t["put_open_qty"] = 2
    act2 = s.check_exit(t, _quotes(13.1))         # +31% -> close quarter
    assert act2.action == "take_profit"
    assert act2.leg_quantities["LC"] == 1
    t["call_open_qty"] = t["put_open_qty"] = 1
    # runner rides — no further exits even deeper in profit
    assert s.check_exit(t, _quotes(18.0)).action == "hold"
