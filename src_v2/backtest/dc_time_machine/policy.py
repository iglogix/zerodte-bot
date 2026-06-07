# src_v2/backtest/dc_time_machine/policy.py
"""Entry filters and the overnight-hold decision seam.

`features` keys produced by the simulator: ratio, ratio_z, ratio_falling,
is_backwardation, is_fri_mon. `overnight_decision` is the swappable seam; the
v1 default is "close" (no overnight hold).
"""
from __future__ import annotations

from typing import Dict


def entry_allowed(filter_name: str, features: Dict) -> bool:
    if filter_name == "always":
        return True
    if filter_name == "backwardation":
        return features.get("ratio", 0.0) > 1.0
    if filter_name == "fri_mon":
        return bool(features.get("is_fri_mon", False))
    if filter_name == "spike":
        return features.get("ratio_z", 0.0) >= 1.0 and bool(features.get("ratio_falling", False))
    raise ValueError(f"unknown entry_filter: {filter_name!r}")


def overnight_decision(policy_name: str, state: Dict) -> str:
    """Return 'close' or 'hold' for an un-transformed DC at EOD."""
    if policy_name == "close":
        return "close"
    if policy_name == "hold":
        shallow = state.get("dc_pnl_frac", -1.0) > -0.10
        backwardated = state.get("ratio", 0.0) > 1.0
        return "hold" if (shallow and backwardated) else "close"
    raise ValueError(f"unknown overnight_policy: {policy_name!r}")
