"""Tests for the SPX 0DTE expected-move fade simulator (src_v2/backtest/em_fade_core.py).

Synthetic days are built from Black-Scholes so per-strike deltas are realistic
and strike selection is exercised against a real chain shape.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from src_v2.backtest.dc_time_machine.iv import bs_price
from src_v2.backtest.em_fade_core import (
    DayContext,
    EmFadeConfig,
    minutes_between,
    simulate_day,
)
from src_v2.backtest.noarb import quote_ok, vertical_mark, check_pnl_bounds

SD = date(2024, 6, 12)


# ----------------------------------------------------------------------
# synthetic chain construction
# ----------------------------------------------------------------------

def _T(minute: str) -> float:
    hh, mm = int(minute[:2]), int(minute[3:5])
    return max((16 * 60 - (hh * 60 + mm)) / (365.0 * 24.0 * 60.0), 1e-6)


def synth_day(
    spot_path: dict[str, float],
    iv: float = 0.16,
    half_spread: float = 0.05,
    n_strikes: int = 60,
    step: float = 5.0,
    strike_center: float | None = None,
) -> dict:
    """Build a {(minute, strike, is_call): (bid, ask)} map from a spot path."""
    quotes: dict = {}
    center = strike_center if strike_center is not None else list(spot_path.values())[0]
    base = round(center / step) * step
    strikes = [base + (i - n_strikes // 2) * step for i in range(n_strikes)]
    for minute, spot in spot_path.items():
        T = _T(minute)
        for k in strikes:
            for is_call in (True, False):
                px = bs_price(spot, k, T, iv, 0.04, is_call)
                bid = max(round(px - half_spread, 2), 0.0)
                ask = round(max(px + half_spread, 0.05), 2)
                quotes[(minute, k, is_call)] = (bid, ask)
    return quotes


def flat_path(minutes, spot):
    return {m: spot for m in minutes}


ALL_MIN = minutes_between("09:31", "15:59")


# ----------------------------------------------------------------------
# no-arb primitive
# ----------------------------------------------------------------------

def test_quote_ok_rejects_crossed_and_negative():
    assert quote_ok((1.0, 1.2))
    assert not quote_ok(None)
    assert not quote_ok((1.5, 1.2))          # crossed
    assert not quote_ok((-0.1, 1.2))         # negative bid
    assert not quote_ok((float("nan"), 1.0))


def test_vertical_mark_rejects_out_of_band_mid():
    # short mid 5.0, long mid 1.0 -> pkg mid 4.0 on a 10-wide: fine
    assert vertical_mark((4.9, 5.1), (0.9, 1.1), 10.0) is not None
    # pkg mid 12 on a 10-wide is structurally impossible -> None (skip, not clamp)
    assert vertical_mark((14.9, 15.1), (2.9, 3.1), 10.0) is None
    # negative pkg mid is impossible for a short vertical
    assert vertical_mark((0.9, 1.1), (4.9, 5.1), 10.0) is None


def test_check_pnl_bounds():
    assert check_pnl_bounds(1.0, credit=1.0, width=10.0)
    assert check_pnl_bounds(-9.0, credit=1.0, width=10.0)
    assert not check_pnl_bounds(1.5, credit=1.0, width=10.0)
    assert not check_pnl_bounds(-9.5, credit=1.0, width=10.0)


# ----------------------------------------------------------------------
# entry logic
# ----------------------------------------------------------------------

def test_no_touch_day_is_skipped():
    quotes = synth_day(flat_path(ALL_MIN, 5000.0))
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig())
    assert out.trade is None
    assert out.skip_reason == "no_touch"


def test_first_touch_wins_when_both_bands_hit():
    """Down through the lower band first, then up through the upper band."""
    path = {}
    for i, m in enumerate(ALL_MIN):
        if m < "09:40":
            path[m] = 5000.0
        elif m < "10:00":
            path[m] = 4900.0     # lower band touch first
        else:
            path[m] = 5120.0     # upper band later
    quotes = synth_day(path, strike_center=5000.0, n_strikes=120)
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig())
    assert out.trade is not None, out.skip_reason
    assert out.trade.side == "put"           # fade a lower touch -> put spread
    assert out.trade.touch_minute < "10:00"


def test_entry_cutoff_enforced():
    path = {m: (5000.0 if m < "13:05" else 4900.0) for m in ALL_MIN}
    quotes = synth_day(path, strike_center=5000.0, n_strikes=120)
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig(entry_cutoff_et="13:00"))
    assert out.trade is None
    assert out.skip_reason == "no_touch"


def test_mirror_control_inverts_the_side():
    path = {m: (5000.0 if m < "09:40" else 4900.0) for m in ALL_MIN}
    quotes = synth_day(path, strike_center=5000.0, n_strikes=120)
    ctx = DayContext.build(SD, quotes)
    fade = simulate_day(ctx, EmFadeConfig(direction="fade"))
    trend = simulate_day(ctx, EmFadeConfig(direction="with_trend"))
    assert fade.trade.side == "put"
    assert trend.trade.side == "call"
    assert fade.trade.touch_minute == trend.trade.touch_minute


def test_delta_targeting_picks_the_nearest_listed_delta():
    path = {m: (5000.0 if m < "09:40" else 4900.0) for m in ALL_MIN}
    quotes = synth_day(path, strike_center=5000.0, n_strikes=120)
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig(target_delta=0.20, delta_min=0.05, delta_max=0.40))
    assert out.trade is not None, out.skip_reason
    assert 0.05 <= out.trade.short_delta <= 0.40
    assert abs(out.trade.short_delta - 0.20) < 0.05
    assert out.trade.short_strike < out.trade.spot_at_entry     # OTM put


def test_delta_out_of_range_is_recorded_as_a_skip():
    path = {m: (5000.0 if m < "09:40" else 4900.0) for m in ALL_MIN}
    quotes = synth_day(path, strike_center=5000.0, n_strikes=120)
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig(target_delta=0.20, delta_min=0.199, delta_max=0.2001))
    assert out.trade is None
    assert out.skip_reason == "delta_out_of_range"


def test_missing_wing_is_rejected():
    path = {m: (5000.0 if m < "09:40" else 4900.0) for m in ALL_MIN}
    quotes = synth_day(path, strike_center=5000.0, n_strikes=120)
    # delete every put strike below 4800 so the wing cannot be listed
    for key in [k for k in quotes if k[1] < 4800.0 and k[2] is False]:
        del quotes[key]
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig(target_delta=0.20, delta_min=0.05,
                                         delta_max=0.40, width_pts=200.0))
    assert out.trade is None
    assert out.skip_reason == "wing_unlisted"


# ----------------------------------------------------------------------
# exit paths
# ----------------------------------------------------------------------

def _touch_then(after_spot: float, iv_after: float = 0.16):
    """Path: flat 5000 until 09:40, drop to 4900 (lower touch), then `after_spot`."""
    path = {}
    for m in ALL_MIN:
        if m < "09:40":
            path[m] = 5000.0
        elif m < "09:45":
            path[m] = 4900.0
        else:
            path[m] = after_spot
    return path


def test_take_profit_path():
    quotes = synth_day(_touch_then(5050.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40))
    assert out.trade is not None, out.skip_reason
    assert out.trade.exit_reason == "take_profit"
    assert out.trade.pnl_usd > 0


def test_stop_loss_path_and_fill_is_worse_than_trigger():
    quotes = synth_day(_touch_then(4700.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    cfg = EmFadeConfig(delta_min=0.05, delta_max=0.40, exit_regime="pessimistic")
    out = simulate_day(ctx, cfg)
    assert out.trade is not None, out.skip_reason
    assert out.trade.exit_reason == "stop_loss"
    # a stop-market fills by crossing: the debit paid exceeds the trigger level
    assert out.trade.exit_debit >= out.trade.sl_trigger_debit
    assert out.trade.pnl_usd < 0


def test_stop_fill_regimes_bracket_each_other():
    quotes = synth_day(_touch_then(4700.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    base = dict(delta_min=0.05, delta_max=0.40)
    opt = simulate_day(ctx, EmFadeConfig(exit_regime="optimistic", **base)).trade
    pess = simulate_day(ctx, EmFadeConfig(exit_regime="pessimistic", **base)).trade
    assert opt.exit_debit <= pess.exit_debit
    assert opt.pnl_usd >= pess.pnl_usd


def test_eod_flat_path():
    """Price parks just short of both TP and SL -> forced close at eod_flat_et."""
    quotes = synth_day(_touch_then(4880.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40,
                                         tp_pct_of_credit=0.01,
                                         sl_mult_of_credit=50.0))
    assert out.trade is not None, out.skip_reason
    assert out.trade.exit_reason == "eod_flat"
    assert out.trade.exit_minute == "15:50"


def test_trailing_stop_arms_and_fires():
    """Move favourably enough to arm the trail, then back to the entry level."""
    path = {}
    for m in ALL_MIN:
        if m < "09:40":
            path[m] = 5000.0
        elif m < "09:45":
            path[m] = 4900.0     # touch
        elif m < "11:00":
            path[m] = 4960.0     # favourable -> arms the trail
        else:
            path[m] = 4800.0     # sharp reversal -> trail should fire
    quotes = synth_day(path, strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    base = dict(delta_min=0.05, delta_max=0.40, tp_pct_of_credit=0.0,
                sl_mult_of_credit=50.0)
    no_trail = simulate_day(ctx, EmFadeConfig(trail_enabled=False, **base)).trade
    trail = simulate_day(ctx, EmFadeConfig(trail_enabled=True, trail_arm_pct=0.25,
                                           trail_stop_pct=0.10, **base)).trade
    assert no_trail.exit_reason == "eod_flat"
    assert trail.exit_reason == "trail_stop"
    assert trail.exit_minute < "15:50"


# ----------------------------------------------------------------------
# glitch handling + invariants
# ----------------------------------------------------------------------

def test_glitch_mark_is_skipped_not_traded_on():
    """A crossed/absurd mark at 10:00 must not fire the take-profit."""
    quotes = synth_day(_touch_then(4880.0), strike_center=5000.0, n_strikes=140)
    ctx0 = DayContext.build(SD, dict(quotes))
    cfg = EmFadeConfig(delta_min=0.05, delta_max=0.40, tp_pct_of_credit=0.50,
                       sl_mult_of_credit=50.0)
    clean = simulate_day(ctx0, cfg).trade
    # Poison the short leg at 10:00 with a zero print (would look like a free TP)
    k = clean.short_strike
    quotes[("10:00", k, False)] = (0.0, 0.0)
    ctx1 = DayContext.build(SD, quotes)
    dirty = simulate_day(ctx1, cfg).trade
    assert dirty.exit_minute != "10:00"
    assert dirty.glitch_minutes >= 1


def test_pnl_never_escapes_the_structural_band():
    for after in (4700.0, 4880.0, 5050.0, 5300.0):
        quotes = synth_day(_touch_then(after), strike_center=5000.0, n_strikes=160)
        ctx = DayContext.build(SD, quotes)
        for regime in ("optimistic", "middle", "pessimistic"):
            out = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40,
                                                 exit_regime=regime))
            if out.trade is None:
                continue
            t = out.trade
            max_loss = t.width - t.credit
            pnl_pts = t.credit - t.exit_debit
            assert -max_loss - 1e-6 <= pnl_pts <= t.credit + 1e-6, (regime, after, t)


def test_entry_depth_monotonically_reduces_credit():
    quotes = synth_day(_touch_then(4880.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    credits = []
    for f in (0.5, 0.7, 1.0):
        out = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40,
                                             entry_depth_f=f))
        credits.append(out.trade.credit)
    assert credits[0] > credits[1] > credits[2], credits


def test_parity_spot_recovers_the_synthetic_spot():
    quotes = synth_day(flat_path(ALL_MIN, 5137.0), strike_center=5137.0)
    ctx = DayContext.build(SD, quotes)
    assert abs(ctx.spot["10:00"] - 5137.0) < 0.2


def test_band_shift_changes_the_touch_decision():
    """Pushing the bands out past the path's extreme must kill the touch."""
    quotes = synth_day(_touch_then(4890.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    em = ctx.em_pts
    assert em and em > 0
    assert ctx.open_spot is not None
    lower = ctx.open_spot - em
    assert lower > 4900.0, "test path must penetrate the unshifted band"

    touched = simulate_day(ctx, EmFadeConfig(band_shift_pct=0.0))
    assert touched.touched

    # push both bands 2% of spot further out -> below the path's 4890 low
    pushed = simulate_day(ctx, EmFadeConfig(band_shift_pct=2.0))
    assert pushed.skip_reason == "no_touch"

    # the pre-registered ±0.05% sensitivity must not move a non-marginal touch
    for shift in (-0.05, 0.05):
        s = simulate_day(ctx, EmFadeConfig(band_shift_pct=shift))
        assert s.touched
        assert s.trade.touch_minute == touched.trade.touch_minute


def test_max_one_trade_per_day():
    path = {}
    for m in ALL_MIN:
        if m < "09:40":
            path[m] = 5000.0
        elif m < "10:00":
            path[m] = 4900.0
        elif m < "11:00":
            path[m] = 5000.0
        else:
            path[m] = 4900.0
    quotes = synth_day(path, strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    out = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40))
    assert out.trade is not None
    assert out.trade.touch_minute < "10:00"


# ----------------------------------------------------------------------
# Mutation guards. Each of these was added because an independent audit
# mutated the engine in this exact way and the suite still passed.
# ----------------------------------------------------------------------

from src_v2.backtest.em_fade_core import _buy_fill, _sell_fill, EmFadeConfig as _C


def _small_day(spot=5000.0):
    return synth_day(flat_path(minutes_between("09:31", "12:00"), spot),
                     strike_center=spot, n_strikes=40)


def test_lookup_staleness_walks_backward_only():
    """M1: a forward-walking staleness fallback is a literal look-ahead bug."""
    quotes = _small_day()
    k = sorted({key[1] for key in quotes})[20]
    # make the three minutes distinguishable, then delete the middle one
    quotes[("10:00", k, True)] = (1.00, 1.10)
    quotes[("10:02", k, True)] = (9.00, 9.10)
    del quotes[("10:01", k, True)]
    ctx = DayContext.build(SD, quotes)
    got = ctx.lookup("10:01", k, True)
    assert got == (1.00, 1.10), f"resolved forward to a future quote: {got}"


def test_spot_staleness_walks_backward_only():
    """M2: same look-ahead risk on the spot series."""
    quotes = _small_day()
    ctx = DayContext.build(SD, quotes)
    ctx.spot["10:00"] = 4444.0
    ctx.spot["10:02"] = 8888.0
    ctx.spot.pop("10:01", None)
    assert ctx.spot_at("10:01") == 4444.0


def test_entry_fill_is_a_true_function_of_depth():
    """M3: 'always fill at the bid' must not pass. f=0.5 is exactly the mid."""
    q = (1.00, 1.20)
    assert _sell_fill(q, 0.0) == pytest.approx(1.10)     # mid
    assert _sell_fill(q, 1.0) == pytest.approx(1.00)     # bid
    assert _buy_fill(q, 0.0) == pytest.approx(1.10)      # mid
    assert _buy_fill(q, 1.0) == pytest.approx(1.20)      # ask  (M4: sign flip)
    # buying is never cheaper than mid; selling never richer than mid
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert _buy_fill(q, d) >= 1.10 - 1e-12
        assert _sell_fill(q, d) <= 1.10 + 1e-12


def test_commission_is_actually_charged():
    """M5: deleting the commission must fail a test."""
    quotes = synth_day(_touch_then(4880.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    a = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40,
                                       commission_per_spread=0.0)).trade
    b = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40,
                                       commission_per_spread=7.0)).trade
    assert a.pnl_usd - b.pnl_usd == pytest.approx(7.0)
    assert b.pnl_usd == pytest.approx(round((b.credit - b.exit_debit) * 100 - 7.0, 2), abs=0.11)


def test_exit_cannot_happen_in_the_entry_minute():
    """M6: the exit walk must start strictly after the entry minute (I5).

    The dangerous case is a WIDE market entered at full depth: the credit is
    struck at the bid side while the package mark sits at the mid, so the mark
    can already exceed the 2x-credit stop level in the entry minute itself. If
    the walk included that minute the trade would exit before it existed.
    """
    wide = synth_day(_touch_then(4880.0), strike_center=5000.0,
                     n_strikes=160, half_spread=0.5)
    ctxw = DayContext.build(SD, wide)
    tw = simulate_day(ctxw, EmFadeConfig(delta_min=0.05, delta_max=0.40,
                                         width_pts=20.0, entry_depth_f=1.0)).trade
    assert tw is not None
    # this configuration stops out on the very first observed minute, so it is
    # exactly one minute away from an impossible same-minute exit
    assert tw.exit_reason == "stop_loss" and tw.hold_minutes == 1
    assert tw.exit_minute > tw.entry_minute

    for after in (4700.0, 4880.0, 5050.0):
        quotes = synth_day(_touch_then(after), strike_center=5000.0, n_strikes=140)
        ctx = DayContext.build(SD, quotes)
        for regime in ("optimistic", "middle", "pessimistic"):
            t = simulate_day(ctx, EmFadeConfig(delta_min=0.05, delta_max=0.40,
                                               exit_regime=regime)).trade
            if t is None:
                continue
            assert t.exit_minute > t.entry_minute
            assert t.hold_minutes >= 1


def test_touch_scan_respects_entry_start():
    """M7: a touch before entry_start_et must not be tradeable."""
    # dip below the band only between 09:32 and 09:34, then recover. The 09:31
    # anchor must stay at 5000 or the bands move with the dip.
    path = {m: (4900.0 if "09:32" <= m <= "09:34" else 5000.0) for m in ALL_MIN}
    quotes = synth_day(path, strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    late = simulate_day(ctx, EmFadeConfig(entry_start_et="09:35"))
    early = simulate_day(ctx, EmFadeConfig(entry_start_et="09:31"))
    assert late.skip_reason == "no_touch"
    assert early.touched


def test_config_key_separates_every_field():
    """M8: reintroducing a partial hash must fail. This bug already shipped once."""
    import dataclasses
    base = _C()
    bumps = {}
    for f in dataclasses.fields(_C):
        v = getattr(base, f.name)
        if isinstance(v, bool):
            nv = not v
        elif isinstance(v, int) and not isinstance(v, bool):
            nv = v + 3
        elif isinstance(v, float):
            nv = v + 0.123
        elif isinstance(v, str):
            nv = v + "_x"
        else:
            continue
        bumps[f.name] = dataclasses.replace(base, **{f.name: nv}).key()
    assert len(set(bumps.values())) == len(bumps), "two fields collide in key()"
    assert base.key() not in bumps.values(), "a field is missing from key()"
    assert len(bumps) >= 20, f"only {len(bumps)} fields exercised"


def test_stop_fill_at_trigger_is_never_worse_than_the_crossing_fill():
    """The idealised-stop arm must bracket the realistic one, not replace it."""
    quotes = synth_day(_touch_then(4700.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    base = dict(delta_min=0.05, delta_max=0.40)
    real = simulate_day(ctx, EmFadeConfig(**base)).trade
    ideal = simulate_day(ctx, EmFadeConfig(stop_fill_at_trigger=True, **base)).trade
    assert real.exit_reason == ideal.exit_reason == "stop_loss"
    assert ideal.exit_debit <= real.exit_debit
    assert ideal.exit_debit == pytest.approx(ideal.sl_trigger_debit, abs=1e-6)


def test_coarser_monitor_grid_never_exits_earlier():
    """monitor_every_min must subsample the walk, not change its start."""
    quotes = synth_day(_touch_then(4700.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    base = dict(delta_min=0.05, delta_max=0.40)
    fine = simulate_day(ctx, EmFadeConfig(monitor_every_min=1, **base)).trade
    coarse = simulate_day(ctx, EmFadeConfig(monitor_every_min=5, **base)).trade
    assert coarse.exit_minute >= fine.exit_minute
    assert coarse.entry_minute == fine.entry_minute


def test_control_entry_modes_are_covered():
    """The control arms carried a headline conclusion with zero test coverage."""
    quotes = synth_day(_touch_then(4880.0), strike_center=5000.0, n_strikes=140)
    ctx = DayContext.build(SD, quotes)
    base = dict(delta_min=0.05, delta_max=0.40)
    fixed = simulate_day(ctx, EmFadeConfig(entry_mode="fixed", fixed_side="call", **base)).trade
    assert fixed.entry_minute == "10:00" and fixed.side == "call"
    # random minute must never precede the touch, and must be seed-deterministic
    seen = set()
    for seed in (1, 2, 3, 4, 5):
        t = simulate_day(ctx, EmFadeConfig(entry_mode="random_minute",
                                           random_seed=seed, **base)).trade
        assert t.entry_minute >= t.touch_minute
        assert t.side == "put"                      # same side as the fade rule
        seen.add(t.entry_minute)
        again = simulate_day(ctx, EmFadeConfig(entry_mode="random_minute",
                                               random_seed=seed, **base)).trade
        assert again.entry_minute == t.entry_minute  # deterministic
    assert len(seen) > 1, "seed has no effect"
    unc = simulate_day(ctx, EmFadeConfig(entry_mode="random_uncond", **base)).trade
    assert unc is not None
