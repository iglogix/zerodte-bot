"""Tests for derive_spot_from_chain."""
from src_v2.strategies.utils import derive_spot_from_chain


def _opt(strike, otype, bid, ask):
    return {"strike": strike, "option_type": otype, "bid": bid, "ask": ask}


def test_derive_spot_basic():
    chain = [_opt(6580, "call", 8.0, 8.4), _opt(6580, "put", 2.8, 3.2)]
    assert derive_spot_from_chain(chain, 6613.0) == 6585.2


def test_derive_spot_returns_none_no_pairs():
    assert derive_spot_from_chain([_opt(6580, "call", 8.0, 8.4)], 6613.0) is None


def test_derive_spot_returns_none_empty():
    assert derive_spot_from_chain([], 6613.0) is None


def test_derive_spot_skips_zero_bid():
    chain = [_opt(6580, "call", 0, 1.0), _opt(6580, "put", 3.0, 3.2)]
    assert derive_spot_from_chain(chain, 6613.0) is None


def test_derive_spot_picks_nearest_strike():
    chain = [
        _opt(6575, "call", 3.1, 3.3), _opt(6575, "put", 3.0, 3.2),
        _opt(6610, "call", 5.0, 5.2), _opt(6610, "put", 2.0, 2.2),
    ]
    assert abs(derive_spot_from_chain(chain, 6613.0) - 6613.0) < 0.1
