# tests/flux/test_source_base.py
from datetime import date

from src_v2.flux.models import ChainSnapshot
from src_v2.flux.sources.base import FluxSource


class _Fake(FluxSource):
    def chain(self, expiry, on_date=None):
        return ChainSnapshot("SPX", expiry, None, ())


def test_expirations_are_calendar_days_after_ref():
    # ref Mon 2026-06-08; 1..3 trading days out = 06-09,06-10,06-11
    exps = _Fake().expirations(date(2026, 6, 8), min_dte=1, max_dte=3)
    assert exps == [date(2026, 6, 9), date(2026, 6, 10), date(2026, 6, 11)]
