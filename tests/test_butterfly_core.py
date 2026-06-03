# tests/test_butterfly_core.py
from src_v2.backtest import butterfly_core as bf


def test_round_to_step():
    assert bf.round_to_step(5022.0, 5.0) == 5020.0
    assert bf.round_to_step(5023.0, 5.0) == 5025.0


def test_select_fly_strikes_call_side():
    cfg = bf.ButterflyConfig(side="call", body_em_mult=1.0, width=30.0)
    # forward 5000, em 20 -> body round(5020)=5020; wings +/-30
    assert bf.select_fly_strikes(5000.0, 20.0, cfg) == (4990.0, 5020.0, 5050.0)


def test_select_fly_strikes_put_side():
    cfg = bf.ButterflyConfig(side="put", body_em_mult=1.0, width=30.0)
    # forward 5000, em 20 -> body round(4980)=4980; wings +/-30
    assert bf.select_fly_strikes(5000.0, 20.0, cfg) == (4950.0, 4980.0, 5010.0)


def test_fly_intrinsic_call():
    strikes = (4990.0, 5020.0, 5050.0)
    assert bf.fly_intrinsic(strikes, True, 5020.0) == 30.0   # peak at body = width
    assert bf.fly_intrinsic(strikes, True, 4980.0) == 0.0    # below low wing
    assert bf.fly_intrinsic(strikes, True, 5060.0) == 0.0    # above high wing
    assert bf.fly_intrinsic(strikes, True, 5005.0) == 15.0   # midway low->body


def test_fly_intrinsic_put():
    strikes = (4950.0, 4980.0, 5010.0)
    assert bf.fly_intrinsic(strikes, False, 4980.0) == 30.0  # peak at body
    assert bf.fly_intrinsic(strikes, False, 5020.0) == 0.0   # above high wing
    assert bf.fly_intrinsic(strikes, False, 4940.0) == 0.0   # below low wing


def _call_legs_quotes(minute):
    # call mids: low=15.0, body=8.0, high=3.0 -> value = 15 + 3 - 2*8 = 2.0
    return {
        (minute, 4990.0, True): (14.5, 15.5),
        (minute, 5020.0, True): (7.5, 8.5),
        (minute, 5050.0, True): (2.5, 3.5),
    }


def test_fly_value_call():
    q = _call_legs_quotes("09:30")
    assert bf.fly_value(q, "09:30", (4990.0, 5020.0, 5050.0), True) == 2.0


def test_fly_value_none_when_leg_missing():
    q = {("09:30", 4990.0, True): (14.5, 15.5)}
    assert bf.fly_value(q, "09:30", (4990.0, 5020.0, 5050.0), True) is None


def test_fly_mark_clamps_negative_to_zero():
    # noisy mids producing a slightly negative raw value clamp to 0
    q = {
        ("10:00", 4990.0, True): (1.0, 1.0),
        ("10:00", 5020.0, True): (1.0, 1.0),
        ("10:00", 5050.0, True): (0.0, 0.0),   # mid None -> treated as missing
    }
    # high leg missing -> mark None
    assert bf.fly_mark(q, "10:00", (4990.0, 5020.0, 5050.0), True) is None
    q[("10:00", 5050.0, True)] = (0.4, 0.6)    # mid 0.5; raw = 1 + 0.5 - 2 = -0.5 -> clamp 0
    assert bf.fly_mark(q, "10:00", (4990.0, 5020.0, 5050.0), True) == 0.0


def test_passes_gates():
    cfg = bf.ButterflyConfig(width=30.0, max_debit_pct=0.10, min_r2r=9.0)
    assert bf.passes_gates(2.0, cfg) is True      # 2/30 = 6.7%, r2r 14 -> ok
    assert bf.passes_gates(3.0, cfg) is True       # 3/30 = 10%, r2r 9 -> exactly ok
    assert bf.passes_gates(4.0, cfg) is False      # 4/30 = 13.3% -> too rich
    assert bf.passes_gates(0.0, cfg) is False       # non-positive debit
    cfg_off = bf.ButterflyConfig(width=30.0, gates_on=False)
    assert bf.passes_gates(9.9, cfg_off) is True    # gates disabled


# --- Task 3: Direction filter (Hull MA), coin-flip, regime width ---
from datetime import date as _date


def test_wma_weights_recent_most():
    # WMA of [1,2,3] with n=3: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
    assert abs(bf.wma([1.0, 2.0, 3.0], 3) - (14.0 / 6.0)) < 1e-9


def test_hull_trend_up_and_down():
    up = [float(x) for x in range(1, 40)]          # strictly rising
    assert bf.hull_trend(up, 21) > 0
    down = [float(x) for x in range(40, 1, -1)]     # strictly falling
    assert bf.hull_trend(down, 21) < 0


def test_hull_trend_insufficient_history_returns_zero():
    assert bf.hull_trend([1.0, 2.0, 3.0], 21) == 0


def test_coin_flip_side_deterministic_by_date():
    # parity of ordinal: even -> call, odd -> put; same date always same side
    d = _date(2025, 1, 2)
    assert bf.coin_flip_side(d) in ("call", "put")
    assert bf.coin_flip_side(d) == bf.coin_flip_side(d)
    assert bf.coin_flip_side(_date(2025, 1, 2)) != bf.coin_flip_side(_date(2025, 1, 3))


def test_resolve_side_modes():
    up = [float(x) for x in range(1, 40)]
    assert bf.resolve_side("call", _date(2025, 1, 2), up) == "call"
    assert bf.resolve_side("put", _date(2025, 1, 2), up) == "put"
    assert bf.resolve_side("hull", _date(2025, 1, 2), up) == "call"
    assert bf.resolve_side("anti", _date(2025, 1, 2), up) == "put"


def test_resolve_side_rejects_unknown_mode():
    import pytest
    up = [float(x) for x in range(1, 40)]
    with pytest.raises(ValueError):
        bf.resolve_side("bogus", _date(2025, 1, 2), up)


def test_regime_width():
    assert bf.regime_width(14.0) == 25.0    # Zombieland (<17)
    assert bf.regime_width(22.0) == 40.0    # Goldilocks (17-32)
    assert bf.regime_width(40.0) == 55.0    # Chaos (>=32)
    assert bf.regime_width(None) == 40.0    # missing VIX -> mid default


# --- Task 4: Per-day evaluation with three exit models ---

def _atm_chain_5000(minute):
    """Chain so atm=5000, em=20, forward=5000 (call_mid==put_mid==10 at 5000)."""
    return {
        (minute, 4990.0, True): (14.5, 15.5), (minute, 4990.0, False): (4.5, 5.5),
        (minute, 5000.0, True): (9.5, 10.5),  (minute, 5000.0, False): (9.5, 10.5),
        (minute, 5010.0, True): (4.5, 5.5),   (minute, 5010.0, False): (14.5, 15.5),
    }


def _entry_chain_fly():
    """09:30 chain: atm/em/forward = 5000/20/5000, call fly debit = 2.0.

    body_em_mult 1.0, width 30 -> strikes (4990, 5020, 5050).
    call mids low=15 (from atm chain), body=8, high=3 -> debit 15+3-16 = 2.0.
    """
    q = _atm_chain_5000("09:30")
    q.update({
        ("09:30", 5020.0, True): (7.5, 8.5),   # body mid 8.0
        ("09:30", 5050.0, True): (2.5, 3.5),   # high mid 3.0
    })
    return q


def _final_anchor_forward_5020(q):
    """Give the 16:00 anchor an ATM chain at 5020 -> settlement forward 5020."""
    q.update({
        ("16:00", 5015.0, True): (12.0, 13.0), ("16:00", 5015.0, False): (7.0, 8.0),
        ("16:00", 5020.0, True): (9.5, 10.5),  ("16:00", 5020.0, False): (9.5, 10.5),
        ("16:00", 5025.0, True): (7.0, 8.0),   ("16:00", 5025.0, False): (12.0, 13.0),
    })
    return q


def test_evaluate_day_hold_to_expiry_win():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold")
    q = _final_anchor_forward_5020(_entry_chain_fly())
    r = bf.evaluate_day(q, cfg)
    assert r["skip"] is None
    assert r["kind"] == "expiry"
    assert r["debit"] == 2.0
    assert r["exit_mark"] == 30.0           # intrinsic at S=5020 (body) = width
    assert r["pnl"] == 2800.0               # (30 - 2) * 100


def test_evaluate_day_gate_blocks_rich_debit():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold")
    q = _entry_chain_fly()
    q[("09:30", 5020.0, True)] = (6.0, 6.0)   # body mid 6 -> debit 15+3-12 = 6.0 (20% of 30)
    r = bf.evaluate_day(q, cfg)
    assert r["skip"] == "gate"


def test_evaluate_day_target_exit():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "12:00", "16:00"],
                             exit_model="target", target_mult=2.0)
    q = _final_anchor_forward_5020(_entry_chain_fly())
    # 12:00 mark: low=9, body=1, high=0.5 -> value = 9 + 0.5 - 2 = 7.5; profit 5.5 >= 2*2=4
    # Note: _mid(0.0,1.0) returns ask=1.0 (bid<=0 shortcut), so use (0.25,0.75) to get mid=0.5
    q.update({
        ("12:00", 4990.0, True): (8.5, 9.5), ("12:00", 5020.0, True): (0.5, 1.5),
        ("12:00", 5050.0, True): (0.25, 0.75),
    })
    r = bf.evaluate_day(q, cfg)
    assert r["kind"] == "target"
    assert r["exit_anchor"] == "12:00"
    assert r["exit_mark"] == 7.5
    assert r["pnl"] == 550.0                # (7.5 - 2) * 100


def test_evaluate_day_trail_exit():
    cfg = bf.ButterflyConfig(side="call",
                             anchors=["09:30", "10:00", "10:30", "16:00"],
                             exit_model="trail", trail_arm_pct=0.50)
    q = _final_anchor_forward_5020(_entry_chain_fly())
    # 10:00 profit peaks; 10:30 retraces past the AM giveback (0.625 of peak) -> exit
    # 10:00 value: low=10, body=1, high~0 -> ~8.0 ; profit ~6 (armed, peak ~6)
    # Note: _mid(0.0,0.0) returns None, so use (0.01,0.01) for the near-zero high wing
    q.update({
        ("10:00", 4990.0, True): (9.5, 10.5), ("10:00", 5020.0, True): (0.5, 1.5),
        ("10:00", 5050.0, True): (0.01, 0.01),
    })
    # 10:30: value must be exactly 2.0 for pnl=(2-2)*100=0.
    # low=4, high=0.01, body=1.005 -> 4 + 0.01 - 2*1.005 = 2.00
    # profit 0.0 <= peak*(1-0.625)~2.25 -> trail fires
    q.update({
        ("10:30", 4990.0, True): (3.5, 4.5), ("10:30", 5020.0, True): (1.005, 1.005),
        ("10:30", 5050.0, True): (0.01, 0.01),
    })
    r = bf.evaluate_day(q, cfg)
    assert r["kind"] == "trail"
    assert r["exit_anchor"] == "10:30"
    assert r["exit_mark"] == 2.0
    assert r["pnl"] == 0.0                   # (2.0 - 2.0) * 100


def test_evaluate_day_with_costs():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold",
                             slippage_per_leg=0.05, commission_per_leg=1.0)
    q = _final_anchor_forward_5020(_entry_chain_fly())
    r = bf.evaluate_day(q, cfg)
    # gross 2800; expiry cash-settle: entry slip 0.05*4*100 = 20; no exit slip;
    # commission 4 legs * 1.0 (open only) = 4 -> net 2800 - 24 = 2776
    assert r["pnl"] == 2776.0


def test_evaluate_day_skips_no_atm():
    cfg = bf.ButterflyConfig()
    assert bf.evaluate_day({}, cfg)["skip"] == "no_atm"


def test_trail_giveback_by_time_of_day():
    assert bf.trail_giveback("11:30") == 0.625
    assert bf.trail_giveback("12:30") == 0.35
    assert bf.trail_giveback("14:30") == 0.30


def test_evaluate_day_put_side_hold_win():
    # Downtrend day: put fly. forward 5000, em 20, body_em_mult 1.0 -> body 4980.
    # width 30 -> put strikes (4950, 4980, 5010). Put mids: low=3, body=8, high=15
    # -> debit = 3 + 15 - 16 = 2.0 (gates pass: 2/30, r2r 14).
    cfg = bf.ButterflyConfig(side="put", anchors=["09:30", "16:00"], exit_model="hold")
    q = _atm_chain_5000("09:30")
    q.update({
        ("09:30", 4950.0, False): (2.5, 3.5),   # low put mid 3.0
        ("09:30", 4980.0, False): (7.5, 8.5),    # body put mid 8.0
        ("09:30", 5010.0, False): (14.5, 15.5),  # high put mid 15.0
    })
    # 16:00 chain with forward at 4980 (body) -> put fly intrinsic = width = 30
    q.update({
        ("16:00", 4975.0, True): (12.0, 13.0), ("16:00", 4975.0, False): (7.0, 8.0),
        ("16:00", 4980.0, True): (9.5, 10.5),  ("16:00", 4980.0, False): (9.5, 10.5),
        ("16:00", 4985.0, True): (7.0, 8.0),   ("16:00", 4985.0, False): (12.0, 13.0),
    })
    r = bf.evaluate_day(q, cfg)
    assert r["skip"] is None
    assert r["side"] == "put"
    assert r["kind"] == "expiry"
    assert r["debit"] == 2.0
    assert r["exit_mark"] == 30.0      # put fly intrinsic at S=4980 (body) = width
    assert r["pnl"] == 2800.0          # (30 - 2) * 100


# --- Task 5: Aggregation, per-regime breakdown, and WF folds ---

def test_make_wf_folds_ratio_5_4folds():
    dates = [_date(2025, 1, 1).fromordinal(_date(2025, 1, 1).toordinal() + i) for i in range(90)]
    folds = bf.make_wf_folds(dates, ratio=5, folds=4)
    assert len(folds) == 4
    # expanding train, 1-step OOS val (step = 90 // 9 = 10)
    train0, val0 = folds[0]
    assert len(train0) == 50 and len(val0) == 10
    train3, val3 = folds[3]
    assert len(train3) == 80 and len(val3) == 10
    # val windows are disjoint and strictly after their train
    assert max(train0) < min(val0)


def test_evaluate_aggregates_pnl_and_winrate():
    cfg = bf.ButterflyConfig(side="call", anchors=["09:30", "16:00"], exit_model="hold")
    win_q = _final_anchor_forward_5020(_entry_chain_fly())          # +2800
    loss_q = _entry_chain_fly()
    # final forward at 4900 -> call fly intrinsic 0 -> full debit loss -200
    loss_q.update({
        ("16:00", 4895.0, True): (12.0, 13.0), ("16:00", 4895.0, False): (7.0, 8.0),
        ("16:00", 4900.0, True): (9.5, 10.5),  ("16:00", 4900.0, False): (9.5, 10.5),
        ("16:00", 4905.0, True): (7.0, 8.0),   ("16:00", 4905.0, False): (12.0, 13.0),
    })
    by_date = {
        _date(2025, 1, 2): {"quotes": win_q, "spot_close": 5020.0, "vix": 14.0},
        _date(2025, 1, 3): {"quotes": loss_q, "spot_close": 4900.0, "vix": 14.0},
    }
    res = bf.evaluate(by_date, cfg)
    assert res["n"] == 2
    assert res["total"] == 2600.0      # 2800 - 200
    assert res["wr"] == 50.0
    assert res["exits"] == {"expiry": 2}


# --- Task 7: Per-day config builder ---

def test_build_day_cfg_sets_side_and_width():
    base = bf.ButterflyConfig(exit_model="hold")
    up = [float(x) for x in range(1, 40)]
    cfg = bf.build_day_cfg(base, dir_mode="hull", width_mode="regime",
                           d=_date(2025, 1, 2), closes_before=up, vix=14.0, fixed_width=30.0)
    assert cfg.side == "call"        # rising Hull -> call
    assert cfg.width == 25.0          # VIX 14 -> Zombieland 25
    cfg2 = bf.build_day_cfg(base, dir_mode="put", width_mode="fixed",
                            d=_date(2025, 1, 2), closes_before=up, vix=40.0, fixed_width=35.0)
    assert cfg2.side == "put"
    assert cfg2.width == 35.0          # fixed mode ignores regime
