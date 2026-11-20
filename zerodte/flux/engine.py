# zerodte/flux/engine.py
"""Read 'the' front/back IV at a reference delta off each expiration's skew, and the ratio."""
from __future__ import annotations

from typing import Optional

from zerodte.backtest.dc_time_machine.iv import front_back_ratio
from zerodte.flux.models import ChainSnapshot, RatioPoint


def atm_iv(chain: ChainSnapshot, ref_delta: float = 0.35) -> Optional[float]:
    """IV at ref_delta: nearest call to +ref_delta and nearest put to −ref_delta, averaged."""
    calls = [r for r in chain.rows if r.is_call]
    puts = [r for r in chain.rows if not r.is_call]
    ivs = []
    if calls:
        ivs.append(min(calls, key=lambda r: abs(r.delta - ref_delta)).iv)
    if puts:
        ivs.append(min(puts, key=lambda r: abs(r.delta + ref_delta)).iv)
    return sum(ivs) / len(ivs) if ivs else None


def ratio_point(front: ChainSnapshot, back: ChainSnapshot,
                ref_delta: float) -> Optional[RatioPoint]:
    f, b = atm_iv(front, ref_delta), atm_iv(back, ref_delta)
    if f is None or b is None:
        return None
    ratio = front_back_ratio(f, b)
    if ratio != ratio:   # NaN
        return None
    return RatioPoint(as_of=front.as_of, front_expiry=front.expiry, back_expiry=back.expiry,
                      ref_delta=ref_delta, front_iv=f, back_iv=b, ratio=ratio)
