"""Shared fixtures for src_v2 tests."""

from __future__ import annotations

import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock


def _make_opt(
    strike: float, option_type: str, delta: float, bid: float, ask: float,
    iv: float = 0.15,
) -> Dict[str, Any]:
    """Build a minimal option dict for testing."""
    return {
        "strike": strike,
        "option_type": option_type.upper(),
        "delta": -delta if option_type.lower() == "put" else delta,
        "bid": bid,
        "ask": ask,
        "iv": iv,
    }


@pytest.fixture
def sample_ic_chain() -> List[Dict[str, Any]]:
    """10-strike chain suitable for IC candidate search.

    Short calls at 5890 (delta 0.12), short puts at 5810 (delta 0.12).
    Wing width 30pt: long call 5920, long put 5780.
    """
    return [
        # Calls (ascending strike)
        _make_opt(5890.0, "call", 0.12, 1.40, 1.60),
        _make_opt(5920.0, "call", 0.04, 0.20, 0.40),
        _make_opt(5950.0, "call", 0.02, 0.05, 0.15),
        # Puts (descending strike by delta)
        _make_opt(5810.0, "put", 0.12, 1.50, 1.70),
        _make_opt(5780.0, "put", 0.04, 0.35, 0.55),
        _make_opt(5750.0, "put", 0.02, 0.10, 0.20),
        # Extra strikes for wing width variations
        _make_opt(5885.0, "call", 0.13, 1.55, 1.75),
        _make_opt(5915.0, "call", 0.05, 0.30, 0.50),
        _make_opt(5815.0, "put", 0.11, 1.35, 1.55),
        _make_opt(5785.0, "put", 0.035, 0.30, 0.50),
    ]


@pytest.fixture
def sample_if_chain() -> List[Dict[str, Any]]:
    """Chain for iron fly testing — ATM options around spot 5850.

    Center strike: 5850 (first above spot if spot < 5850).
    Wings at ±15pt: long call 5865, long put 5835.
    """
    return [
        # ATM calls
        _make_opt(5845.0, "call", 0.52, 8.00, 8.40),
        _make_opt(5850.0, "call", 0.50, 7.50, 7.90),
        _make_opt(5855.0, "call", 0.48, 7.00, 7.40),
        _make_opt(5860.0, "call", 0.45, 6.50, 6.90),
        _make_opt(5865.0, "call", 0.42, 6.00, 6.40),
        _make_opt(5870.0, "call", 0.39, 5.50, 5.90),
        # ATM puts
        _make_opt(5835.0, "put", 0.48, 6.80, 7.20),
        _make_opt(5840.0, "put", 0.50, 7.30, 7.70),
        _make_opt(5845.0, "put", 0.52, 7.80, 8.20),
        _make_opt(5850.0, "put", 0.50, 7.50, 7.90),
        _make_opt(5855.0, "put", 0.48, 7.00, 7.40),
    ]


@pytest.fixture
def ic_config() -> Dict[str, Any]:
    """Config dict for IronCondorStrategy."""
    return {
        "strategy": {
            "credit_equality_threshold": 0.20,
            "skew_threshold": 1.15,
            "min_short_delta": 0.10,
            "max_short_delta": 0.15,
            "min_side_credit": 0.50,
            "max_side_credit": 2.50,
            "wing_width": 30,
        },
        "stop_loss": {
            "stop_limit_gap": 40,
            "stop_market_gap": 30,
        },
        "risk": {
            "take_profit_threshold": 0.05,
        },
    }


@pytest.fixture
def if_config() -> Dict[str, Any]:
    """Config dict for IronFlyStrategy."""
    return {
        "options": {
            "strike_step_pts": 5,
            "em_wing_fraction": 0.50,
            "tp_fixed_dollars": 200,
            "sl_fixed_dollars": 0,
            "slippage_per_leg": 0.05,
            "contract_multiplier": 100,
            "quantity": 1,
            "em_source": "chain",
        },
    }


@pytest.fixture
def orb_em_ic_config() -> Dict[str, Any]:
    return {
        "bot": {"symbol": "ES", "strategy": "orb_em_ic"},
        "strategy": {
            "orb_window_start": "08:00", "orb_window_end": "08:15",
            "market_open": "09:30", "eod_force_close": "15:50",
            "midline_tolerance_pts": 0.25,
            "orb_range_filter": {"min_pts": 2.0, "max_pts": 30.0},
        },
        "options": {
            "option_root": "SPXW", "strike_step_pts": 5,
            "em_with_trend": 0.50, "em_against_trend": 0.75,
            "width_with_trend": 5, "width_against_trend": 10,
            "min_credit_per_spread": 0.05, "tp_pct_of_credit": 0.50,
            "sl_mult_of_credit": 0, "hold_put_breach": True,
            "slippage_per_spread": 0.10, "contract_multiplier": 100,
            "quantity": 1, "index_quote_symbol": "",
        },
        "entry": {"schedule": ["09:45"]},
        "risk": {"max_concurrent_trades": 1},
    }


@pytest.fixture
def sample_orb_chain() -> List[Dict[str, Any]]:
    """Chain around spot 5850 with 5-pt strikes, paired puts/calls."""
    chain = []
    for k in range(5820, 5885, 5):
        tv = 3.0
        c_mid = max(5850 - k, 0) + tv
        p_mid = max(k - 5850, 0) + tv
        chain.append(_make_opt(float(k), "call", 0.10, round(c_mid - 0.1, 2), round(c_mid + 0.1, 2)))
        chain.append(_make_opt(float(k), "put", 0.10, round(p_mid - 0.1, 2), round(p_mid + 0.1, 2)))
    return chain


@pytest.fixture
def orb_levels() -> Dict[str, Any]:
    return {"high": 5860.0, "low": 5840.0, "midline": 5850.0, "range_pts": 20.0}


@pytest.fixture
def meic_config() -> Dict[str, Any]:
    """Config dict for MeicStrategy."""
    return {
        "bot": {"strategy": "meic"},
        "entry": {
            "schedule": ["11:00", "11:20", "11:40", "12:00", "12:40",
                         "13:00", "13:20", "14:00", "14:40", "15:00"],
        },
        "strategy": {
            "am_pm_cutoff": "12:30",
            "am_min_credit": 1.25,
            "am_max_credit": 1.65,
            "pm_min_credit": 0.80,
            "pm_max_credit": 1.30,
            "wing_widths": [80, 90, 100],
            "stop_pct_of_total_credit": 0.70,
            "am_scale_out_tiers": [0.55, 0.65, 0.75, 0.85],
            "am_scale_out_fractions": [0.25, 0.25, 0.25, 0.25],
            "pm_scale_out_tiers": [0.65, 0.85],
            "pm_scale_out_fractions": [0.50, 0.50],
            "abandon_long_below": 0.05,
            "credit_equality_threshold": 0.30,
            "default_tranche_qty": 3,
            "tranche_sizing": {"11:00": 3, "11:40": 4, "12:00": 5, "13:00": 6},
        },
    }


@pytest.fixture
def sample_meic_chain() -> List[Dict[str, Any]]:
    """Chain supporting 80-100pt MEIC wings around spot 5850.

    Short call 5950 mid ~1.45; long calls at 6030/6040/6050.
    Short put 5750 mid ~1.45; long puts at 5670/5660/5650.
    Credits land in the AM band [1.25, 1.65].
    """
    return [
        # Short call + its three possible long-call wings (80/90/100)
        _make_opt(5950.0, "call", 0.10, 1.40, 1.50),
        _make_opt(6030.0, "call", 0.03, 0.02, 0.10),   # wing 80 -> credit ~1.39
        _make_opt(6040.0, "call", 0.025, 0.01, 0.09),  # wing 90
        _make_opt(6050.0, "call", 0.02, 0.01, 0.07),   # wing 100
        # Short put + its three possible long-put wings
        _make_opt(5750.0, "put", 0.10, 1.40, 1.50),
        _make_opt(5670.0, "put", 0.03, 0.02, 0.10),    # wing 80 -> credit ~1.39
        _make_opt(5660.0, "put", 0.025, 0.01, 0.09),   # wing 90
        _make_opt(5650.0, "put", 0.02, 0.01, 0.07),    # wing 100
    ]
