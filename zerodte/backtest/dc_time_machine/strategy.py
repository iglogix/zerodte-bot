"""DcParams — the single sweepable parameter object for the DC Time Machine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DcParams:
    entry_minute: str = "09:40"        # ET; baseline = shortly after open
    target_delta: float = 0.35         # abs delta for both K_p (put) and K_c (call)
    front_dte: int = 9                 # front expiry = Nth available expiry after session
    gap: int = 1                       # back expiry = front index + gap
    wing_width: float = 5.0            # IC wing width in points (W)
    transform_epsilon: float = 0.0     # extra credit required over (D + W), in $/share
    stop_loss_frac: float = 0.20       # mental stop on the un-transformed DC
    fill_k: float = 1.0                # 0 = mid, 1 = full cross
    entry_filter: str = "always"       # always | backwardation | fri_mon | spike
    overnight_policy: str = "close"    # close | hold
    rate: float = 0.04                 # risk-free rate for BS/IV
