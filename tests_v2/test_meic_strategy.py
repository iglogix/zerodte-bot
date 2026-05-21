"""TDD tests: MeicStrategy."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src_v2.strategies.base import ExitAction

ET = ZoneInfo("America/New_York")


class TestExitActionWidening:
    def test_exit_action_has_partial_fields_with_defaults(self):
        a = ExitAction(action="hold")
        assert a.leg_quantities == {}
        assert a.quantity_fraction == 1.0

    def test_exit_action_accepts_leg_quantities(self):
        a = ExitAction(
            action="take_profit",
            legs_to_close=["SPXW_C"],
            leg_quantities={"SPXW_C": 2},
            quantity_fraction=0.25,
        )
        assert a.leg_quantities == {"SPXW_C": 2}
        assert a.quantity_fraction == 0.25
