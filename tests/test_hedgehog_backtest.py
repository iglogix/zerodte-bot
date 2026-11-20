"""Tests for the Hedge Hog backtest engine on synthetic BS-priced chains."""

from __future__ import annotations

import datetime as dt
import pickle

import pytest

from zerodte.backtest.hedgehog import (
    HedgeHogEngine,
    HedgeHogStore,
    HhParams,
    black_put,
    black_put_iv,
    parity_forward,
    summarize,
)

# ---------------------------------------------------------------- helpers

def _to_int(d: dt.date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def _black_call(F, K, T, sigma):
    # parity with discount 1: C - P = F - K
    return black_put(F, K, T, sigma) + F - K


def skewed_iv(strike: float, spot: float, base_iv: float) -> float:
    """Simple put-skew surface: IV rises as strikes fall below spot."""
    m = (spot - strike) / spot
    return base_iv + max(m, 0.0) * 0.35 + max(-m, 0.0) * 0.05


def make_chain(spot: float, expiry: dt.date, on: dt.date, base_iv: float) -> dict:
    """Synthetic chain: 5-wide strikes 50%..110% of spot, both rights."""
    T = max((expiry - on).days, 0) / 365.0
    chain = {}
    lo, hi = int(spot * 0.5 // 5 * 5), int(spot * 1.1 // 5 * 5)
    for k in range(lo, hi + 5, 5):
        iv = skewed_iv(k, spot, base_iv)
        p = black_put(spot, k, T, iv)
        c = _black_call(spot, k, T, iv)
        for right, px in (("P", p), ("C", c)):
            bid = max(px - 0.03, 0.0)
            ask = px + 0.03
            if bid > 0:
                chain[(float(k), right)] = (round(bid, 2), round(ask, 2), 100)
    return chain


def build_store(tmp_path, spot_path: dict[dt.date, float], base_iv=0.18) -> HedgeHogStore:
    """Write synthetic per-expiry pickles covering the dates in spot_path."""
    dates = sorted(spot_path)
    first, last = dates[0], dates[-1]
    # Friday expiries from just after start to 130 days past the end
    expiries = []
    d = first
    while d <= last + dt.timedelta(days=130):
        if d.weekday() == 4:
            expiries.append(d)
        d += dt.timedelta(days=1)
    cache = tmp_path / "SPY"
    cache.mkdir(parents=True)
    for exp in expiries:
        days = {}
        for qd in dates:
            if qd >= exp:
                continue
            days[_to_int(qd)] = make_chain(spot_path[qd], exp, qd, base_iv)
        if days:
            with open(cache / f"{_to_int(exp)}.pkl", "wb") as f:
                pickle.dump({"expiry": _to_int(exp), "days": days}, f)
    return HedgeHogStore(cache)


def trading_days(start: dt.date, n: int) -> list[dt.date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


START = dt.date(2024, 1, 2)


# ---------------------------------------------------------------- greeks

def test_black_iv_round_trip():
    px = black_put(500.0, 450.0, 0.25, 0.22)
    iv = black_put_iv(px, 500.0, 450.0, 0.25)
    assert abs(iv - 0.22) < 1e-4


def test_parity_forward_recovers_spot():
    chain = make_chain(500.0, START + dt.timedelta(days=30), START, 0.18)
    f = parity_forward(chain)
    assert abs(f - 500.0) < 1.0


# ---------------------------------------------------------------- entry

def test_entry_structure():
    """Entry picks ~30/20-delta vertical and ~7-delta hog with credit > debit."""
    days = trading_days(START, 5)
    store = build_store(pytest.tmp_path, {d: 500.0 for d in days})
    engine = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0))
    tr = engine.try_enter(_to_int(days[0]))
    assert tr is not None
    assert tr.k_long > tr.k_short > tr.k_hog
    assert tr.hog_credit > tr.debit
    assert tr.net_credit > 0
    assert 0.04 <= tr.hog_delta <= 0.12
    assert 0.0 <= tr.pop_entry <= 1.0
    # vertical strikes near 30/20 delta of a put-skewed surface
    assert tr.k_long < 500.0
    assert tr.margin_usd > 0


def test_flat_market_closes_at_dte_trigger():
    """Flat market for a month: position closes when the vertical hits <7 DTE.

    Whether a flat tape is profitable is an empirical question for real
    chains; here we only assert trigger mechanics and accounting sanity.
    """
    days = trading_days(START, 35)
    store = build_store(pytest.tmp_path, {d: 500.0 for d in days})
    engine = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0, fees_per_leg=0.0,
                                            commission_open=0.0))
    trades = engine.run()
    assert trades, "expected at least one closed trade"
    t = trades[0]
    assert t.exit_reason in ("vertical_lt7dte", "hog_50pct")
    assert t.days_held <= 31
    width = t.k_long - t.k_short
    # P&L bounded by structure: can't lose more than debit+hog move, can't
    # win more than width + full credit (per contract, 100 multiplier)
    assert -(t.debit + t.hog_credit) * 100 < t.pnl_usd < (width + t.hog_credit) * 100


def test_fast_crash_triggers_breach_exit():
    """Crash through the vertical short strike within 10 days => fast_breach."""
    days = trading_days(START, 15)
    path = {}
    for i, d in enumerate(days):
        path[d] = 500.0 if i < 3 else 500.0 * (1 - 0.04 * min(i - 2, 5))
    store = build_store(pytest.tmp_path, path)
    engine = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0))
    trades = engine.run(end=_to_int(days[-1]))
    breaches = [t for t in trades if t.exit_reason == "fast_breach"]
    assert breaches
    assert breaches[0].days_held < 10


def test_rally_triggers_hog_tp():
    """Strong rally: far OTM hog put collapses to <50% of credit quickly."""
    days = trading_days(START, 30)
    path = {d: 500.0 * (1 + 0.005 * i) for i, d in enumerate(days)}
    store = build_store(pytest.tmp_path, path)
    engine = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0))
    trades = engine.run(end=_to_int(days[-1]))
    assert trades
    t = trades[0]
    assert t.exit_reason == "hog_50pct"
    # the hog leg itself banked >= 50% of its credit (doc allows the package
    # to show a tiny loss on rallies: "+10% -> tiny loss (~$77)")
    assert t.days_held < 25


def test_skip_when_credit_cannot_exceed_debit():
    """With no skew (cheap far puts), the 7-12 delta credit can't beat the debit."""
    days = trading_days(START, 3)
    cache = pytest.tmp_path / "flat"

    # build a no-skew store manually
    def make_flat_chain(spot, expiry, on):
        T = max((expiry - on).days, 0) / 365.0
        chain = {}
        for k in range(int(spot * 0.5), int(spot * 1.1), 5):
            iv = 0.10  # flat, low IV
            p = black_put(spot, k, T, iv)
            c = _black_call(spot, k, T, iv)
            for right, px in (("P", p), ("C", c)):
                if px - 0.03 > 0:
                    chain[(float(k), right)] = (round(px - 0.03, 2), round(px + 0.03, 2), 100)
        return chain

    cache.mkdir(parents=True)
    expiries = [d for d in (START + dt.timedelta(days=i) for i in range(140))
                if d.weekday() == 4]
    for exp in expiries:
        day_map = {}
        for qd in days:
            if qd < exp:
                day_map[_to_int(qd)] = make_flat_chain(500.0, exp, qd)
        if day_map:
            with open(cache / f"{_to_int(exp)}.pkl", "wb") as f:
                pickle.dump({"expiry": _to_int(exp), "days": day_map}, f)

    store = HedgeHogStore(cache)
    engine = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0))
    tr = engine.try_enter(_to_int(days[0]))
    assert tr is None
    assert engine.skips and engine.skips[-1].reason == "credit_le_debit"


def test_summarize_basic():
    days = trading_days(START, 35)
    store = build_store(pytest.tmp_path, {d: 500.0 for d in days})
    engine = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0))
    trades = engine.run()
    stats = summarize(trades, engine.daily_equity)
    assert stats["trades"] == len(trades) > 0
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert stats["avg_days_held"] > 0


@pytest.fixture(autouse=True)
def _tmp_path_on_pytest(tmp_path, monkeypatch):
    # convenience: expose tmp_path as pytest.tmp_path for the helpers above
    pytest.tmp_path = tmp_path
    yield


def test_hog_only_ablation_accounting():
    """trade_hedge=False books only the naked-put leg (1 leg of costs)."""
    days = trading_days(START, 35)
    store = build_store(pytest.tmp_path, {d: 500.0 for d in days})
    full = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0))
    solo = HedgeHogEngine(store, HhParams(fill_k=0.0, pop_min=0.0, trade_hedge=False))
    t_full = full.run()
    t_solo = solo.run()
    assert t_full and t_solo
    a, b = t_full[0], t_solo[0]
    # identical structure selection
    assert (a.k_long, a.k_short, a.k_hog) == (b.k_long, b.k_short, b.k_hog)
    assert b.net_credit == pytest.approx(b.hog_credit)
    # hog-only P&L = credit - buyback - 1 leg of costs; in a flat decaying
    # market the naked put alone must be profitable
    assert b.pnl_usd > 0
