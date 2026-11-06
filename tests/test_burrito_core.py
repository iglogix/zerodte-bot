"""Unit tests for the Burrito Butterfly backtest engine."""
import pytest

from zerodte.backtest.burrito_core import (
    BurritoConfig, Leg, build_legs, payoff, position_value, entry_debit,
    passes_gates, direction_from_series, BurritoTrade, lifecycle_decision,
    lock_legs, choo_choo_legs, _nearest_valley, summarize_burrito, run_trade,
)
from zerodte.backtest.iron_duck import Smile


# ---------------------------------------------------------------- geometry

def test_build_legs_bullish_call_outer_link():
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    legs = build_legs(forward=5000.0, em=20.0, cfg=cfg, side="call")  # body 5020
    by = {(l.strike, l.right): l.qty for l in legs}
    body = 5020.0
    assert by[(body, "C")] == -2
    assert by[(body - 15.0, "C")] == 1
    assert by[(body + 15.0, "C")] == 2          # fly outer + spread long merged
    assert by[(body + 15.0 + 5.0, "C")] == -1
    assert all(l.right == "C" for l in legs)


def test_build_legs_bearish_put_mirror():
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    legs = build_legs(forward=5000.0, em=20.0, cfg=cfg, side="put")
    by = {(l.strike, l.right): l.qty for l in legs}
    body = 4980.0
    assert by[(body, "P")] == -2
    assert by[(body + 15.0, "P")] == 1
    assert by[(body - 15.0, "P")] == 2
    assert by[(body - 15.0 - 5.0, "P")] == -1


# ---------------------------------------------------------------- payoff

def test_payoff_call_fly_peaks_at_body():
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    legs = build_legs(5000.0, 20.0, cfg, "call")
    assert abs(payoff(legs, 4000.0)) < 1e-9
    assert payoff(legs, 5020.0) == 15.0
    assert payoff(legs, 5035.0) < payoff(legs, 5020.0)
    assert payoff(legs, 5040.0) > payoff(legs, 5035.0) - 1e-9


# ---------------------------------------------------------------- value

def test_position_value_from_chain_fill():
    chain = {(5000.0, "C"): (10.0, 12.0, 0)}
    legs = [Leg(1, 5000.0, "C")]
    val = position_value(legs, chain, smile=None, F=None, slippage_k=1.0)
    assert abs(val - 10.0) < 1e-9


def test_position_value_model_fallback_when_missing():
    smile_chain = {(4800.0, "P"): (5.0, 6.0, 0), (5200.0, "C"): (5.0, 6.0, 0),
                   (4900.0, "P"): (8.0, 9.0, 0), (5100.0, "C"): (8.0, 9.0, 0)}
    sm = Smile(smile_chain, F=5000.0, T=5 / 365)
    legs = [Leg(1, 5050.0, "C")]
    val = position_value(legs, {}, smile=sm, F=5000.0, slippage_k=0.0)
    assert val is not None and val > 0


# ---------------------------------------------------------------- entry

def test_entry_debit_and_gate():
    chain = {(5000.0, "C"): (10.0, 11.0, 0), (5005.0, "C"): (7.0, 8.0, 0)}
    legs = [Leg(1, 5000.0, "C"), Leg(-1, 5005.0, "C")]
    debit = entry_debit(legs, chain, slippage_k=1.0, commission_per_leg=0.0)
    assert abs(debit - (11.0 - 7.0)) < 1e-9
    cfg = BurritoConfig(fly_wing_width=15.0, max_debit_pct=0.40, gates_on=True)
    assert passes_gates(4.0, cfg) is True
    assert passes_gates(7.0, cfg) is False


# ---------------------------------------------------------------- direction

def test_direction_signal_hull_and_momentum():
    rising = [100 + i for i in range(40)]
    falling = [100 - i for i in range(40)]
    assert direction_from_series(rising, "hull") == "call"
    assert direction_from_series(falling, "hull") == "put"
    assert direction_from_series(rising, "momentum") == "call"
    assert direction_from_series(falling, "momentum") == "put"
    assert direction_from_series(rising, "both") == "both"
    assert direction_from_series([100.0] * 3, "hull") == "skip"


# ---------------------------------------------------------------- clock

def test_eod_marks_walk_to_expiry():
    from zerodte.backtest.burrito_clock import Mark, eod_marks

    class FakeStore:
        def __init__(self, chains, dates):
            self._c = chains
            self._d = dates

        def quote_dates(self):
            return self._d

        def chain(self, d, e):
            return self._c.get((d, e))

    chain = {(5000.0, "C"): (1.0, 2.0, 0), (5000.0, "P"): (1.0, 2.0, 0),
             (4900.0, "P"): (0.5, 1.0, 0), (5100.0, "C"): (0.5, 1.0, 0)}
    store = FakeStore({(20240101, 20240105): chain, (20240102, 20240105): chain,
                       (20240105, 20240105): chain},
                      [20240101, 20240102, 20240105])
    marks = list(eod_marks(store, entry_date=20240101, expiry=20240105))
    assert [m.date for m in marks] == [20240101, 20240102, 20240105]
    assert marks[-1].is_last and not marks[0].is_last
    assert all(isinstance(m, Mark) for m in marks)
    assert marks[0].T > marks[1].T


def test_intraday_marks_for_0dte():
    from zerodte.backtest.burrito_clock import intraday_marks

    class FakeIntraday:
        def bars(self, date):
            return ["09:40", "10:00", "15:50"]

        def chain_at(self, date, minute):
            return {(5000.0, "C"): (1.0, 2.0, 0)}

        def forward_at(self, date, minute):
            return 5000.0

    marks = list(intraday_marks(FakeIntraday(), date=20240705, entry_min="09:40"))
    assert [m.intraday_min for m in marks] == ["09:40", "10:00", "15:50"]
    assert marks[-1].is_last and not marks[0].is_last
    assert marks[0].T > marks[-1].T >= 0


# ---------------------------------------------------------------- lifecycle

def test_lifecycle_lock_then_target_and_uncle_stop():
    cfg = BurritoConfig(lock_trigger_pct=0.07, stop_pct=0.07, pt_pct=0.10)
    tr = BurritoTrade(side="call", legs=[], debit=10.0, debit_mid=10.0,
                      entry_date=20240101, expiry=20240105)
    d = lifecycle_decision(tr, "DIRECTIONAL", 10.8, cfg, is_last=False)
    assert d.action == "lock" and d.next_state == "FLOATING"
    d = lifecycle_decision(tr, "DIRECTIONAL", 9.2, cfg, is_last=False)
    assert d.action == "close" and d.reason == "uncle_stop"
    d = lifecycle_decision(tr, "FLOATING", 11.1, cfg, is_last=False)
    assert d.action == "close" and d.reason == "target"
    d = lifecycle_decision(tr, "DIRECTIONAL", 10.0, cfg, is_last=True)
    assert d.action == "settle"


# ---------------------------------------------------------------- adjustments

def test_adjustment_legs_lock_and_choo_choo():
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    lock = lock_legs(side="call", forward=5000.0, em=20.0, cfg=cfg)
    assert all(l.right == "P" for l in lock) and any(l.qty > 0 for l in lock)
    extra = choo_choo_legs(side="call", forward=4900.0, em=20.0, cfg=cfg)
    assert any(l.qty == -2 for l in extra)


def test_nearest_valley_finds_subzero_gap():
    cfg = BurritoConfig(fly_wing_width=15.0, spread_width=5.0, spread_link="outer", step=5.0)
    legs = build_legs(5000.0, 20.0, cfg, "call")
    v2 = _nearest_valley(legs, forward=4700.0, cfg=cfg, debit=10.0)
    assert v2 is not None


# ---------------------------------------------------------------- summarize

def test_summarize_sharpe_basic():
    class T:
        def __init__(self, pnl, d, reason="target"):
            self.pnl_usd = pnl
            self.exit_date = d
            self.exit_reason = reason
            self.debit = 5.0
            self.extra_cost = 0.0
    trades = [T(100, 20240102), T(-50, 20240103), T(80, 20240104)]
    out = summarize_burrito(trades)
    assert out["trades"] == 3
    assert "sharpe" in out and isinstance(out["sharpe"], float)
    assert out["win_rate"] == round(2 / 3, 4)
    assert out["profit_factor"] > 1.0


# ---------------------------------------------------------------- integration

def test_run_trade_settles_to_pnl():
    from zerodte.backtest.iron_duck import DuckStore
    import os
    cache = "data/iron_duck_eod/SPXW"
    if not os.path.isdir(cache):
        pytest.skip("SPXW cache not present")
    store = DuckStore(cache)
    exp = [e for e in store.expiries if e >= 20240601][0]
    cands = [d for d in store.quote_dates() if d < exp]
    entry = cands[-4]
    tr = run_trade(store, entry_date=entry, expiry=exp, side="call",
                   em=None, cfg=BurritoConfig(direction_signal="both", gates_on=False))
    assert tr is not None
    assert tr.exit_reason in {"target", "uncle_stop", "expiry"}
    assert tr.pnl_usd is not None
