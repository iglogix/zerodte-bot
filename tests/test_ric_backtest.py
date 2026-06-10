"""Regression tests for the Reverse Iron Condor backtest engine
(src_v2/backtest/reverse_ic.py).

Synthetic Black-Scholes chains over a spot path verify the spec mechanics from
docs/0DTE-Reverse-Iron-Condor-Strategy.md: the parity-based gap gate (flat and
per-weekday), FOMC blackout, 50-delta / 35-delta long selection with fixed-point
wings, the "sets up" upside filter, profit-target fills, pin-day full-debit loss
at expiry, close-mode partial loss, scale-out tiers, and width-capped settlement.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src_v2.backtest.cbbo_store import CbboStore
from src_v2.backtest.reverse_ic import RicEngine

ET = ZoneInfo("America/New_York")

PRIOR = date(2023, 6, 5)   # Monday
DAY = date(2023, 6, 6)     # Tuesday, not FOMC
SPOT = 4000.0
STRIKES = [3800.0 + 5 * i for i in range(81)]   # 3800..4200 step 5


def minutes(start="09:31", end="15:59"):
    h0, m0 = int(start[:2]), int(start[3:5])
    h1, m1 = int(end[:2]), int(end[3:5])
    t, out = h0 * 60 + m0, []
    while t <= h1 * 60 + m1:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += 1
    return out


def bs_price(s, k, sigma, t_years, is_call):
    if t_years <= 0 or sigma <= 0:
        return max(0.0, (s - k) if is_call else (k - s))
    sq = sigma * math.sqrt(t_years)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t_years) / sq
    d2 = d1 - sq

    def n(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if is_call:
        return s * n(d1) - k * n(d2)
    return k * n(-d2) - s * n(-d1)


def make_day_quotes(spot_path=None, iv=0.20, half_spread=0.05):
    """BS-priced 0DTE chain for DAY over a {minute: spot} path (default: pinned)."""
    qs = {}
    for m in minutes():
        s = (spot_path or {}).get(m, SPOT)
        h, mm = int(m[:2]), int(m[3:5])
        t_years = max(16 * 60 - (h * 60 + mm), 1) / (60 * 24 * 365)
        for k in STRIKES:
            for is_call in (True, False):
                mid = bs_price(s, k, iv, t_years, is_call)
                bid = max(0.0, mid - half_spread)
                qs[(m, k, is_call)] = (round(bid, 3), round(mid + half_spread, 3))
    return qs


def make_prior_quotes(prior_settle):
    """Minimal prior-day chain: one late minute with parity-consistent quotes,
    so settlement_forward() recovers prior_settle for the gap reference."""
    qs = {}
    t_years = 1 / (60 * 24 * 365)
    for k in STRIKES:
        for is_call in (True, False):
            mid = bs_price(prior_settle, k, 0.20, t_years, is_call)
            qs[("15:59", k, is_call)] = (max(0.0, mid - 0.05), mid + 0.05)
    return qs


def make_bars(d, spot_path=None, scale_close=None):
    bars = []
    for m in minutes("09:30", "15:59"):
        s = (spot_path or {}).get(m, SPOT)
        ts = datetime(d.year, d.month, d.day, int(m[:2]), int(m[3:5]), tzinfo=ET)
        bars.append({"ts_et": ts, "open": s, "high": s, "low": s,
                     "close": scale_close or s, "volume": 1})
    return bars


def make_config(**strategy_overrides):
    strat = {
        "entry_time": "09:33",
        "gap_check_minute": "09:31",
        "gap_min_pct": 0.25,
        "gap_min_pct_by_weekday": {},
        "long_call_delta": 0.50,
        "long_put_delta": 0.35,
        "call_wing_offset_pts": 20,
        "put_wing_offset_pts": 35,
        "profit_target_pct": 0.25,
        "tp_fill": "limit",
        "setup_min_upside_pct": 0.25,
        "final_exit": "15:55",
        "final_exit_mode": "expire",
        "skip_fomc_days": True,
        "qty": 1,
        "scale_out": {"enabled": False},
        "dislocation_min_spread_pts": 1.0,
        "dislocation_spread_mult": 1.5,
    }
    strat.update(strategy_overrides)
    return {
        "strategy": strat,
        "options": {
            "contract_multiplier": 100,
            "slippage_per_spread": 0.10,
            "commission_open_per_contract_per_leg": 1.0,
            "commission_close_per_contract_per_leg": 1.0,
        },
    }


def run_engine(config, prior_settle=3980.0, spot_path=None, iv=0.20,
               day=DAY, prior=PRIOR, day_quotes=None):
    qs = day_quotes if day_quotes is not None else make_day_quotes(spot_path, iv)
    store = CbboStore("", preloaded={prior: make_prior_quotes(prior_settle),
                                     day: qs})
    engine = RicEngine(config)
    engine.set_cbbo_store(store)
    results = engine.run([(prior, make_bars(prior)),
                          (day, make_bars(day, spot_path))])
    assert len(results) == 2
    return results[1]


# ---------------------------------------------------------------------------
# Entry gates
# ---------------------------------------------------------------------------

def test_gap_up_enters_and_selects_strikes():
    # prior settle 3980 -> spot 4000 = +0.50% gap up
    res = run_engine(make_config())
    assert res.status == "closed"
    assert res.gap_pct == pytest.approx(0.50, abs=0.05)
    # 50-delta long call lands at/near ATM; 35-delta put just below
    assert abs(res.long_call - SPOT) <= 10
    assert SPOT - 20 <= res.long_put < SPOT
    # fixed-point wings beyond the longs
    assert res.short_call == res.long_call + 20
    assert res.short_put == res.long_put - 35
    assert res.call_width == 20 and res.put_width == 35
    assert res.debit and res.debit > 0
    assert res.max_loss_usd >= res.debit * 100


def test_gap_down_skipped():
    res = run_engine(make_config(), prior_settle=4005.0)
    assert res.status == "skipped"
    assert res.skip_reason == "gap_down"


def test_small_gap_skipped():
    res = run_engine(make_config(), prior_settle=3996.0)  # +0.10%
    assert res.status == "skipped"
    assert res.skip_reason == "gap_below_min"
    assert res.gap_pct == pytest.approx(0.10, abs=0.03)


def test_weekday_gap_override_enters_monday():
    # +0.15% gap fails the flat 0.25% gate but passes a Monday 0.10% override.
    cfg = make_config(gap_min_pct_by_weekday={"mon": 0.10})
    res = run_engine(cfg, prior_settle=3994.0, day=date(2023, 6, 5),
                     prior=date(2023, 6, 2))
    assert res.status == "closed"


def test_fomc_day_skipped():
    res = run_engine(make_config(), day=date(2023, 6, 14),
                     prior=date(2023, 6, 13))
    assert res.status == "skipped"
    assert res.skip_reason == "fomc_day"


def test_first_day_has_no_gap_reference():
    store = CbboStore("", preloaded={DAY: make_day_quotes()})
    engine = RicEngine(make_config())
    engine.set_cbbo_store(store)
    res = engine.run([(DAY, make_bars(DAY))])[0]
    assert res.status == "skipped"
    assert res.skip_reason == "no_gap_reference"


def test_setup_filter_rejects_pumped_iv():
    # IV high enough that debit leaves < 25% upside on the 20-wide call spread
    res = run_engine(make_config(), iv=0.80)
    assert res.status == "skipped"
    assert res.skip_reason == "setup_filter"


def test_dislocated_leg_skips():
    qs = make_day_quotes()
    # blow out every call quote at the entry minute so whichever strike the
    # delta selector picks trips the spread > 1.5 x mid dislocation guard
    for k in STRIKES:
        bid, ask = qs[("09:33", k, True)]
        qs[("09:33", k, True)] = (max(0.0, bid - 8), ask + 20)
    res = run_engine(make_config(), day_quotes=qs)
    assert res.status == "skipped"
    assert res.skip_reason == "dislocated_quote"


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

def rally_path(target=4045.0, by="11:00"):
    """Linear spot ramp from SPOT to target between 09:34 and `by`, then flat."""
    path = {}
    ms = minutes()
    ramp = [m for m in ms if "09:34" <= m <= by]
    for m in ms:
        if m < "09:34":
            path[m] = SPOT
        elif m in ramp:
            i = ramp.index(m)
            path[m] = SPOT + (target - SPOT) * (i + 1) / len(ramp)
        else:
            path[m] = target
    return path


def test_profit_target_limit_fill():
    res = run_engine(make_config(), spot_path=rally_path())
    assert res.status == "closed"
    assert res.exit_reason == "profit_target"
    target = res.debit * 1.25
    assert res.exit_value == pytest.approx(target, abs=0.01)
    # pnl = (target - debit - entry slip - exit slip) * 100 - 8 legs commission
    expected = (0.25 * res.debit - 0.40) * 100 - 8.0
    assert res.realized_pnl_usd == pytest.approx(expected, abs=1.0)


def test_pin_day_full_debit_loss_at_expiry():
    res = run_engine(make_config())  # pinned at 4000 all day
    assert res.exit_reason == "expired"
    # settle ~4000 sits between the longs -> both spreads worthless
    assert res.exit_value == pytest.approx(0.0, abs=0.5)
    expected = (-res.debit - 0.20) * 100 - 4.0   # no close commission on expiry
    assert res.realized_pnl_usd == pytest.approx(expected, abs=60.0)
    assert res.realized_pnl_usd >= -res.max_loss_usd


def test_close_mode_partial_loss_beats_full_loss():
    cfg = make_config(final_exit_mode="close")
    res = run_engine(cfg)
    assert res.exit_reason == "time_exit"
    assert res.exit_time == "15:55"
    # residual time value means the close-out recovers more than expiring at 0
    assert res.exit_value > 0


def test_settlement_capped_at_width():
    # huge rally, TP effectively disabled -> runner rides to settle past the wing
    cfg = make_config(profit_target_pct=5.0)
    res = run_engine(cfg, spot_path=rally_path(target=4120.0))
    assert res.exit_reason == "expired"
    assert res.exit_value == pytest.approx(res.call_width, abs=0.5)


def test_scale_out_tiers():
    cfg = make_config(
        qty=4,
        scale_out={"enabled": True,
                   "tiers": [{"profit": 0.20, "close_frac": 0.50},
                             {"profit": 0.30, "close_frac": 0.25}]},
        profit_target_pct=0.25,   # ignored when scale_out is enabled
    )
    res = run_engine(cfg, spot_path=rally_path(target=4060.0))
    assert res.status == "closed"
    assert res.tiers_filled == 2
    assert res.exit_reason == "expired"      # the runner rides to settlement
    assert res.realized_pnl_usd > 0
