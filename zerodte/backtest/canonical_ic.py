"""Canonical single-entry 0DTE IC: EM-relative short strikes, fixed wing, hold-to-expiry,
mid-minus-slippage fills. Mirrors zerodte/strategies/meic.py + engine.py settlement."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def _mid(q: Optional[Tuple[float, float]]) -> Optional[float]:
    if not q or q[0] <= 0 or q[1] <= 0:
        return None
    return (q[0] + q[1]) / 2


def price_ic(chain: Dict[Tuple[float, bool], Tuple[float, float]], spot: float, em: float,
             em_mult: float, wing: int, step: int, slippage: float,
             settle_spot: Optional[float], mult: int = 100,
             commission_per_leg: float = 1.0) -> Optional[Dict]:
    short_call = math.ceil((spot + em_mult * em) / step) * step
    short_put = math.floor((spot - em_mult * em) / step) * step
    long_call, long_put = short_call + wing, short_put - wing
    legs = {"sc": _mid(chain.get((float(short_call), True))),
            "lc": _mid(chain.get((float(long_call), True))),
            "sp": _mid(chain.get((float(short_put), False))),
            "lp": _mid(chain.get((float(long_put), False)))}
    if any(v is None for v in legs.values()):
        return None
    call_credit = legs["sc"] - legs["lc"] - slippage
    put_credit = legs["sp"] - legs["lp"] - slippage
    credit = call_credit + put_credit
    if settle_spot is None:
        return {"credit": credit, "pnl": None}
    call_loss = min(max(0.0, settle_spot - short_call), wing)
    put_loss = min(max(0.0, short_put - settle_spot), wing)
    commission = 4 * commission_per_leg  # 4 legs
    pnl = (credit - call_loss - put_loss) * mult - commission
    return {"short_call": short_call, "short_put": short_put, "credit": credit,
            "call_loss": call_loss, "put_loss": put_loss, "pnl": pnl}
