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
