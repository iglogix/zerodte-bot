"""Regression tests for the Power Hour backtest engine (zerodte/backtest/power_hour.py).

Synthetic quote days verify the spec mechanics from docs/0DTE-Power-Hour-Strategy.md:
fixed-credit strike selection, never-invert floor, the §6 worked ratchet example,
the entry_plus ambiguity reading, T3 no-ratchet, time exit, and wing selection.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.power_hour import PowerHourEngine

ET = ZoneInfo("America/New_York")
DAY = date(2023, 6, 6)  # not a FOMC day


def minutes(start="14:50", end="15:59"):
    h0, m0 = int(start[:2]), int(start[3:5])
    h1, m1 = int(end[:2]), int(end[3:5])
    t = h0 * 60 + m0
    out = []
    while t <= h1 * 60 + m1:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += 1
    return out


def make_quotes(spot=4000.0, per_minute_strangle=None):
    """Build a synthetic 0DTE chain around `spot` on a 5-pt grid.

    Shorts at 3990P / 4010C are priced to a $4.00 mid at entry; their combined
    path over time is driven by per_minute_strangle: {minute: combined_price}
    split evenly between put and call. All other strikes get static decoration
    prices so selection logic has a full ladder to choose from.
    """
    qs = {}
    short_put, short_call = 3990.0, 4010.0
    strikes = [spot + d for d in range(-100, 105, 5)]
    base = {}  # static mids by distance for non-short strikes
    for k in strikes:
        for is_call in (True, False):
            otm = (k - spot) if is_call else (spot - k)
            if otm <= -50:
                mid = 55.0   # deep ITM
            elif otm <= 0:
                mid = 8.3 - otm * 0.1
            else:
                # decreasing ladder that never lands exactly on 4.00, so the
                # planted $4.00 shorts are the unambiguous closest-to-target
                mid = max(0.025, 7.3 - otm * 0.13)
            if otm >= 40:
                mid = 0.025  # zero-bid wing territory: (0, 0.05)
            base[(k, is_call)] = mid

    for m in minutes():
        for k in strikes:
            for is_call in (True, False):
                mid = base[(k, is_call)]
                if (k, is_call) in ((short_put, False), (short_call, True)):
                    combined = (per_minute_strangle or {}).get(m, 8.0)
                    mid = combined / 2
                bid = max(0.0, mid - 0.025)
                ask = mid + 0.025
                qs[(m, k, is_call)] = (round(bid, 3), round(ask, 3))
    return qs


def make_bars():
    bars = []
    for m in minutes("09:30", "15:59"):
        h, mm = int(m[:2]), int(m[3:5])
        ts = datetime(DAY.year, DAY.month, DAY.day, h, mm, tzinfo=ET)
        bars.append({"ts_et": ts, "open": 4000.0, "high": 4000.0,
                     "low": 4000.0, "close": 4000.0, "volume": 1})
    return bars


def make_config(**strategy_overrides):
    strat = {
        "tranches": [
            {"name": "T1", "entry": "14:55", "target_credit": 4.00,
             "stop": 4.00, "ratchet_add": 3.00, "qty": 1},
        ],
        "ratchet_mode": "trigger_plus",
        "ratchet_trigger_profit_pct": 0.40,
        "profit_target_pct": 0.90,
        "final_exit": "15:55",
        "final_exit_mode": "close",
        "wing_target_price": 0.05,
        "stop_execution": "trigger",
        "skip_fomc_days": True,
        "regime_overlay": {"enabled": False},
    }
    strat.update(strategy_overrides)
    return {
        "strategy": strat,
        "options": {
            "contract_multiplier": 100,
            "slippage_per_spread": 0.10,
            "stop_slippage_per_spread": 0.25,
            "commission_open_per_contract_per_leg": 1.0,
            "commission_close_per_contract_per_leg": 1.0,
        },
    }


def run_engine(quotes, config):
    engine = PowerHourEngine(config)
    engine.set_cbbo_store(CbboStore("", preloaded={DAY: quotes}))
    return engine.run([(DAY, make_bars())])


def test_fixed_credit_selection_and_wings():
    qs = make_quotes()
    res = run_engine(qs, make_config())[0]
    assert res.status == "closed"
    # $4 mids were planted at 3990P / 4010C
    assert res.short_put == 3990.0
    assert res.short_call == 4010.0
    assert res.put_credit == pytest.approx(4.0, abs=0.01)
    assert res.call_credit == pytest.approx(4.0, abs=0.01)
    # wings: nearest strike beyond the short with ask <= 0.05 (otm >= 40 zone)
    assert res.long_call == 4040.0
    assert res.long_put == 3960.0


def test_ratchet_worked_example_trigger_plus():
    """Lecture 1.7: entry $8 -> ratchet when price hits $4.80 -> stop $7.80."""
    path = {m: 8.0 for m in minutes()}
    for m in minutes("15:10", "15:20"):
        path[m] = 4.8           # 40% profit -> ratchet: stop = 4.8 + 3 = 7.8
    for m in minutes("15:21", "15:59"):
        path[m] = 7.9           # breaches the ratcheted stop
    res = run_engine(make_quotes(per_minute_strangle=path), make_config())[0]
    assert res.ratcheted
    assert res.stop_final == pytest.approx(7.8, abs=0.01)
    assert res.exit_reason == "stopped"
    # trigger-mode fill: stop level + 2 spreads x 0.25 stop slippage
    assert res.exit_debit == pytest.approx(7.8 + 0.5, abs=0.01)


def test_ratchet_entry_plus_reading():
    """Literal OO reading: stop moves to entry + $3 = $11, so $7.90 doesn't stop."""
    path = {m: 8.0 for m in minutes()}
    for m in minutes("15:10", "15:20"):
        path[m] = 4.8
    for m in minutes("15:21", "15:59"):
        path[m] = 7.9
    res = run_engine(make_quotes(per_minute_strangle=path),
                     make_config(ratchet_mode="entry_plus"))[0]
    assert res.ratcheted
    assert res.stop_final == pytest.approx(11.0, abs=0.01)
    assert res.exit_reason == "time_exit"


def test_initial_stop_at_entry_plus_4():
    path = {m: 8.0 for m in minutes()}
    for m in minutes("15:05", "15:59"):
        path[m] = 12.5          # breaches 8 + 4
    res = run_engine(make_quotes(per_minute_strangle=path), make_config())[0]
    assert res.exit_reason == "stopped"
    assert not res.ratcheted
    assert res.stop_initial == pytest.approx(12.0, abs=0.01)
    assert res.exit_debit == pytest.approx(12.5, abs=0.01)  # 12 + 0.5 slip


def test_t3_has_no_ratchet():
    cfg = make_config()
    cfg["strategy"]["tranches"] = [
        {"name": "T3", "entry": "15:25", "target_credit": 4.00,
         "stop": 3.00, "ratchet_add": None, "qty": 1},
    ]
    path = {m: 8.0 for m in minutes()}
    for m in minutes("15:30", "15:40"):
        path[m] = 4.0           # deep profit; would ratchet if enabled
    for m in minutes("15:41", "15:59"):
        path[m] = 10.6          # above E+stop(11)? no: 10.6 < 11 -> survives
    res = run_engine(make_quotes(per_minute_strangle=path), cfg)[0]
    assert not res.ratcheted
    assert res.stop_final == pytest.approx(11.0, abs=0.01)
    assert res.exit_reason == "time_exit"


def test_time_exit_pnl_accounting():
    """Flat decay to $2 at 15:55: P&L = (8 - wings - slip - debit)*100 - comm."""
    path = {m: 8.0 for m in minutes("14:50", "15:30")}
    for m in minutes("15:31", "15:59"):
        path[m] = 5.0           # 37.5% profit: no ratchet (needs 40%), no TP
    res = run_engine(make_quotes(per_minute_strangle=path), make_config())[0]
    assert res.exit_reason == "time_exit"
    # debit = 5.0 + 0.2 exit slip; wings = 2 x 0.05 ask
    expected = (8.0 - 0.10 - 0.20 - 5.20) * 100 - 6.0  # 4 open + 2 close legs
    assert res.realized_pnl_usd == pytest.approx(expected, abs=1.0)


def test_never_invert_floors_at_atm():
    """If the $4-credit put strike crosses above the call strike, both go ATM."""
    qs = {}
    # low-vol chain: every OTM option is cheap, ITM expensive -> the $4 mids
    # land ITM on both sides and would invert.
    for m in minutes():
        for k in [4000.0 + d for d in range(-50, 55, 5)]:
            for is_call in (True, False):
                otm = (k - 4000.0) if is_call else (4000.0 - k)
                mid = max(0.025, 2.0 - otm * 0.12) if otm >= 0 else 2.0 - otm * 0.8
                qs[(m, k, is_call)] = (round(mid - 0.025, 3), round(mid + 0.025, 3))
    res = run_engine(qs, make_config())[0]
    assert res.status == "closed"
    assert res.short_put == res.short_call == 4000.0


def test_fomc_day_skipped():
    run_engine(make_quotes(), make_config())
    engine_day = date(2023, 6, 14)  # FOMC
    engine = PowerHourEngine(make_config())
    engine.set_cbbo_store(CbboStore("", preloaded={engine_day: make_quotes()}))
    bars = [dict(b, ts_et=b["ts_et"].replace(day=14)) for b in make_bars()]
    out = engine.run([(engine_day, bars)])
    assert all(r.status == "skipped" and r.skip_reason == "fomc_day" for r in out)
