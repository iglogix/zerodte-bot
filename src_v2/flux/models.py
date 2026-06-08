"""Immutable data types for Flux."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Tuple


@dataclass(frozen=True)
class OptionRow:
    strike: float
    is_call: bool
    bid: float
    ask: float
    iv: float       # decimal (0.18 = 18%)
    delta: float    # signed: calls +, puts −


@dataclass(frozen=True)
class ChainSnapshot:
    underlying: str
    expiry: date
    as_of: datetime
    rows: Tuple[OptionRow, ...]


@dataclass(frozen=True)
class RatioPoint:
    as_of: datetime
    front_expiry: date
    back_expiry: date
    ref_delta: float
    front_iv: float
    back_iv: float
    ratio: float    # front_iv / back_iv


@dataclass(frozen=True)
class ScanRow:
    front_expiry: date
    back_expiry: date
    gap_td: int
    ratio: float
    ratio_prev: float
    drop: float     # ratio_prev − ratio (positive = reverting down)
    zscore: float
    flag: bool      # elevated (>1) AND reverting (drop>0)
