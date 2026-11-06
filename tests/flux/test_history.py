# tests/flux/test_history.py
from datetime import date, datetime, timezone

from zerodte.flux.models import OptionRow, ChainSnapshot
from zerodte.flux.sources.base import FluxSource
from zerodte.flux.history import daily_ratio_series
from zerodte.backtest.dc_time_machine.trading_calendar import nth_trading_day_after


class FakeSource(FluxSource):
    """front IV 0.22, back IV 0.20 on every session → ratio 1.1."""
    def chain(self, expiry, on_date=None):
        # front expiry is sooner (front_dte=9 below) → give it higher IV
        front = nth_trading_day_after(on_date, 9) if on_date is not None else None
        is_front = expiry == front
        iv = 0.22 if is_front else 0.20
        rows = (OptionRow(7000, True, 1, 1, iv, 0.5), OptionRow(7000, False, 1, 1, iv, -0.5))
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), rows)


def test_daily_series_has_one_point_per_session():
    # 3 sessions ending Wed 2026-06-10 → 06-08,06-09,06-10
    pts = daily_ratio_series(FakeSource(), front_dte=9, gap=1, n_days=3,
                             ref_delta=0.5, end_date=date(2026, 6, 10))
    assert len(pts) == 3
    assert all(abs(p.ratio - 1.1) < 1e-9 for p in pts)
    assert pts[0].as_of.date() < pts[-1].as_of.date()  # chronological
