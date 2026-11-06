"""OSI symbol formatter for SPXW 0DTE options.

OPRA raw_symbol format: 21 chars = 6-char root (left-justified) + YYMMDD + C/P + strike*1000 (8-digit zero-pad).
Example: 'SPXW  250922P06445000' = SPXW Sep 22 2025 6445.000 Put.
"""
from __future__ import annotations

from datetime import date
from typing import List


def format_osi(expiry: date, strike: float, is_call: bool, root: str = "SPXW") -> str:
    """Build a 21-char OSI raw_symbol string for an SPXW option."""
    root_padded = f"{root:<6}"  # 'SPXW  '
    expiry_str = expiry.strftime("%y%m%d")
    cp = "C" if is_call else "P"
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"
    return f"{root_padded}{expiry_str}{cp}{strike_str}"


def parse_osi(symbol: str) -> tuple[float, bool]:
    """Parse a 21-char OSI raw_symbol into (strike, is_call).

    Inverse of `format_osi`: positions 12 = C/P, 13:21 = strike*1000.
    """
    if len(symbol) != 21:
        raise ValueError(f"not a 21-char OSI symbol: {symbol!r}")
    is_call = symbol[12] == "C"
    strike = int(symbol[13:21]) / 1000.0
    return strike, is_call


def build_strike_window(
    expiry: date,
    atm_strike: float,
    half_width_pts: int = 50,
    step_pts: int = 5,
    root: str = "SPXW",
) -> List[str]:
    """Build OSI symbols for ATM ± half_width_pts in step_pts increments, both calls and puts.

    `atm_strike` is snapped to the nearest step before windowing so all
    returned strikes are step-aligned.
    """
    base = int(round(atm_strike / step_pts) * step_pts)
    n_steps = half_width_pts // step_pts
    strikes = [base + i * step_pts for i in range(-n_steps, n_steps + 1)]
    out: List[str] = []
    for k in strikes:
        out.append(format_osi(expiry, float(k), is_call=False, root=root))
        out.append(format_osi(expiry, float(k), is_call=True, root=root))
    return out
