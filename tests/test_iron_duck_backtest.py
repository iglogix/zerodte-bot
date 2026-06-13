"""Tests for the Iron Duck backtest engine (src_v2/backtest/iron_duck.py).

Uses synthetic Black-76 chains so construction, gates, stop, settlement and
sequential lifecycle are all verified against known closed-form values.
"""

import datetime as dt
import math

import pytest

from src_v2.backtest.hedgehog import black_put
from src_v2.backtest.iron_duck import (
    DuckParams,
    DuckTrade,
    IronDuckEngine,
    Smile,
    black_call,
    black_call_delta,
    summarize,
)

# ---------------------------------------------------------------- fixtures


def make_chain(
    F: float,
    T: float,
    vol: float = 0.20,
    spread: float = 0.10,
    lo: float = 0.70,
    hi: float = 1.15,
    step: float = 5.0,
):
    """Synthetic SPX-like chain: strikes every `step`, flat vol, fixed spread."""
    chain = {}
    k = math.floor(F * lo / step) * step
    while k <= F * hi:
        c = black_call(F, k, T, vol)
        p = black_put(F, k, T, vol)
        chain[(k, "C")] = (max(c - spread / 2, 0.0), c + spread / 2, 100)
        chain[(k, "P")] = (max(p - spread / 2, 0.0), p + spread / 2, 100)
        k += step
    return chain


def _to_date(d: int) -> dt.date:
    return dt.date(d // 10000, d // 100 % 100, d % 100)


class FakeStore:
    """Duck-typed DuckStore over {date: {expiry: (F, vol)}} synthetic specs."""

    def __init__(self, spec, T_override=None):
        # spec: {quote_date: {expiry: (F, vol)}}
        self.spec = spec
        self.expiries = sorted({e for day in spec.values() for e in day})
        self.T_override = T_override

    def quote_dates(self):
        return sorted(self.spec)

    def _T(self, qd, exp):
        if self.T_override is not None:
            return self.T_override
        return max((_to_date(exp) - _to_date(qd)).days, 0) / 365.0

    def chain(self, qd, exp):
        cell = self.spec.get(qd, {}).get(exp)
        if cell is None:
            return None
        F, vol = cell
        return make_chain(F, max(self._T(qd, exp), 1e-9), vol)

    def expiries_in_dte(self, qd, lo, hi):
        out = []
        for exp in self.spec.get(qd, {}):
            dte = (_to_date(exp) - _to_date(qd)).days
            if lo <= dte <= hi:
                out.append((exp, dte))
        return out

    def spot(self, qd):
        day = self.spec.get(qd, {})
        return next(iter(day.values()))[0] if day else None


def base_params(**kw):
    defaults = dict(
        dte_target=7,
        dte_min=3,
        dte_max=10,
        fill_k=0.0,
        commission_per_leg=0.0,
        fees_per_leg=0.0,
    )
    defaults.update(kw)
    return DuckParams(**defaults)


# ---------------------------------------------------------------- payoff


def _trade(credit=6.0, kcs=4000, kcl=4005, kps=3900, kpl=3850) -> DuckTrade:
    return DuckTrade(
        entry_date=20230103,
        expiry=20230110,
        dte=7,
        spot_entry=3995.0,
        k_call_short=kcs,
        k_call_long=kcl,
        k_put_short=kps,
        k_put_long=kpl,
        credit=credit,
        credit_mid=credit,
        call_width=kcl - kcs,
        put_width=kps - kpl,
        pop_entry=0.9,
        be_down=kps - credit,
        be_loss=0.0,
        atm_iv_entry=0.2,
        margin_usd=(kps - kpl - credit) * 100,
    )


class TestPayoff:
    def test_head_full_credit(self):
        tr = _trade()
        assert IronDuckEngine.payoff(tr, 3950) == pytest.approx(6.0)

    def test_beak_locked_profit(self):
        tr = _trade()  # credit 6 > call width 5 -> beak = +1 everywhere up
        assert IronDuckEngine.payoff(tr, 4100) == pytest.approx(1.0)
        assert IronDuckEngine.payoff(tr, 9999) == pytest.approx(1.0)

    def test_downside_breakeven_and_max_loss(self):
        tr = _trade()
        assert IronDuckEngine.payoff(tr, tr.be_down) == pytest.approx(0.0)
        assert IronDuckEngine.payoff(tr, 3000) == pytest.approx(6.0 - 50.0)

    def test_upside_risk_when_credit_below_width(self):
        tr = _trade(credit=3.0)
        assert IronDuckEngine.payoff(tr, 4100) == pytest.approx(-2.0)


# ---------------------------------------------------------------- smile


class TestSmile:
    def test_recovers_flat_vol_and_deltas(self):
        F, T, vol = 4000.0, 7 / 365.0, 0.20
        sm = Smile(make_chain(F, T, vol, spread=0.0), F, T)
        assert sm.ok()
        assert sm.vol(4000) == pytest.approx(vol, abs=0.005)
        assert sm.vol(3900) == pytest.approx(vol, abs=0.01)
        d = sm.call_delta(3980)
        assert d == pytest.approx(black_call_delta(F, 3980, T, vol), abs=0.01)
        assert sm.put_delta_abs(3900) == pytest.approx(
            1 - black_call_delta(F, 3900, T, vol), abs=0.01
        )


# ---------------------------------------------------------------- entry


class TestEntry:
    def test_construction_targets_deltas(self):
        store = FakeStore({20230103: {20230110: (4000.0, 0.20)}})
        eng = IronDuckEngine(store, base_params())
        tr = eng.try_enter(20230103)
        assert tr is not None
        T, vol, F = 7 / 365.0, 0.20, 4000.0
        # short call ~65 delta (ITM, below spot)
        assert tr.k_call_short < F
        d_cs = black_call_delta(F, tr.k_call_short, T, vol)
        assert abs(d_cs - 0.65) < 0.05
        # long call exactly one strike above
        assert tr.k_call_long == tr.k_call_short + 5
        # puts ~15 / ~5 abs delta
        d_ps = 1 - black_call_delta(F, tr.k_put_short, T, vol)
        d_pl = 1 - black_call_delta(F, tr.k_put_long, T, vol)
        assert abs(d_ps - 0.15) < 0.04
        assert abs(d_pl - 0.05) < 0.03
        assert tr.put_width > tr.call_width
        assert tr.credit > 0
        assert 0.5 < tr.pop_entry < 1.0

    def test_no_upside_risk_gate_skips(self):
        store = FakeStore({20230103: {20230110: (4000.0, 0.04)}})  # tiny vol -> tiny credit
        eng = IronDuckEngine(store, base_params(require_no_upside_risk=True, put_long_delta=0.10))
        tr = eng.try_enter(20230103)
        if tr is not None:
            # if it entered, the gate must genuinely hold
            assert tr.credit >= tr.call_width
        else:
            assert any(s.reason == "upside_risk" for s in eng.skips)

    def test_pop_gate(self):
        store = FakeStore({20230103: {20230110: (4000.0, 0.20)}})
        eng = IronDuckEngine(store, base_params(pop_min=0.999))
        assert eng.try_enter(20230103) is None
        assert eng.skips[-1].reason == "pop"


# ---------------------------------------------------------------- lifecycle


def week_spec(F_path, vol=0.20, start=20230102, expiry=20230110):
    """Mon..Fri then Mon-Tue of next week, expiry Tuesday 20230110."""
    days = [20230102, 20230103, 20230104, 20230105, 20230106, 20230109, 20230110]
    return {d: {expiry: (F_path[i], vol)} for i, d in enumerate(days[: len(F_path)])}


class TestLifecycle:
    def test_hold_to_expiry_in_head_books_full_credit(self):
        spec = week_spec([4000, 4000, 4000, 4000, 4000, 4000, 3990])
        eng = IronDuckEngine(FakeStore(spec), base_params(dte_target=8))
        trades = eng.run()
        assert len(trades) >= 1
        tr = trades[0]
        assert tr.exit_reason == "expiry"
        # settled in the head (between short put and short call)? then pnl == credit
        if tr.k_put_short < 3990 < tr.k_call_short:
            assert tr.pnl_usd == pytest.approx(tr.credit * 100, abs=1.0)

    def test_crash_triggers_stop(self):
        spec = week_spec([4000, 3700, 3700, 3700, 3700, 3700, 3700])
        eng = IronDuckEngine(FakeStore(spec), base_params(dte_target=8, stop_frac=1.0))
        trades = eng.run()
        assert trades, "crash should still produce a closed trade"
        assert trades[0].exit_reason == "stop"
        assert trades[0].exit_date == 20230103
        assert trades[0].pnl_usd < 0

    def test_no_stop_holds_to_expiry(self):
        spec = week_spec([4000, 3700, 3700, 3700, 3700, 3700, 3700])
        eng = IronDuckEngine(FakeStore(spec), base_params(dte_target=8, stop_frac=None))
        trades = eng.run()
        assert trades[0].exit_reason in ("expiry", "eod_force")

    def test_sequential_reentry_next_day(self):
        # two expiries; first trade expires 20230110, next entry 20230111+
        spec = {}
        days1 = [20230102, 20230103, 20230104, 20230105, 20230106, 20230109, 20230110]
        for d in days1:
            spec.setdefault(d, {})[20230110] = (4000.0, 0.20)
        days2 = [20230111, 20230112, 20230113, 20230117, 20230118]
        for d in days2 + days1:
            spec.setdefault(d, {})[20230118] = (4000.0, 0.20)
        eng = IronDuckEngine(FakeStore(spec), base_params(dte_min=1, dte_max=16))
        trades = eng.run()
        assert len(trades) == 2
        assert trades[0].exit_date == 20230110
        assert trades[1].entry_date == 20230111

    def test_beak_exit(self):
        spec = week_spec([4000, 4400, 4400, 4400, 4400, 4400, 4400])
        eng = IronDuckEngine(
            FakeStore(spec),
            base_params(dte_target=8, stop_frac=None, beak_exit_prob=0.10),
        )
        trades = eng.run()
        assert trades[0].exit_reason == "beak"
        # beak exit books a small profit (credit - width-ish), small loss at worst
        assert trades[0].pnl_usd > -100


# ---------------------------------------------------------------- summary


class TestSummary:
    def test_summarize_fields(self):
        spec = week_spec([4000, 4000, 4000, 4000, 4000, 4000, 3990])
        eng = IronDuckEngine(FakeStore(spec), base_params(dte_target=8))
        eng.run()
        s = summarize(eng.trades, eng.daily_equity)
        assert s["trades"] >= 1
        assert "net_pnl" in s and "win_rate" in s and "max_dd_usd" in s
        assert s["max_margin_usd"] > 0
