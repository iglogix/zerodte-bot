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
