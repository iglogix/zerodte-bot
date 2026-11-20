"""
ORB Anomaly Detector — thin wrapper delegating to the existing 0DTE anomaly detector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OrbAnomalyDetector:
    """
    Lightweight anomaly detector for the ORB bot.
    Checks for conditions that should block trade entry regardless of regime.
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._risk = config.get("risk", {})
        orb_filter = config.get("strategy", {}).get("orb_range_filter", {})
        self._min_range_pts: float = orb_filter.get("min_pts", 2.0)
        self._max_range_pts: float = orb_filter.get("max_pts", 30.0)

    async def check(
        self,
        current_price: Optional[float],
        orb_levels: Optional[Dict[str, Any]],
        account_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run anomaly checks. Returns dict with 'safe' bool and 'reasons' list.
        """
        reasons = []

        if current_price is None:
            reasons.append("no_price_data")

        if orb_levels:
            range_pts = orb_levels.get("range_pts", 0)
            if range_pts < self._min_range_pts:
                reasons.append(f"orb_range_too_narrow_{range_pts:.1f}pts")
            if range_pts > self._max_range_pts:
                reasons.append(f"orb_range_too_wide_{range_pts:.1f}pts")

        if account_info:
            net_liq = account_info.get("net_liq", 0)
            if net_liq <= 0:
                reasons.append("zero_net_liq")

            buying_power = account_info.get("buying_power", 0)
            max_bp_pct = self._risk.get("max_buying_power_pct", 50.0)
            if net_liq > 0 and (buying_power / net_liq * 100) < (100 - max_bp_pct):
                reasons.append("insufficient_buying_power")

        return {
            "safe": len(reasons) == 0,
            "reasons": reasons,
        }
