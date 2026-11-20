# tests/flux/test_scanner.py
from datetime import date, datetime, timezone

from zerodte.flux.models import ChainSnapshot, OptionRow
from zerodte.flux.scanner import scan
from zerodte.flux.sources.base import FluxSource


class FakeSource(FluxSource):
    """Ratio falls from 1.20 (prior sessions) to 1.05 today for every pair → reverting down."""
    def chain(self, expiry, on_date=None):
        front_iv, back_iv = (0.21, 0.20) if on_date == date(2026, 6, 10) else (0.24, 0.20)
        is_front = expiry.day % 2 == 1
        iv = front_iv if is_front else back_iv
        rows = (OptionRow(7000, True, 1, 1, iv, 0.5), OptionRow(7000, False, 1, 1, iv, -0.5))
        return ChainSnapshot("SPX", expiry, datetime(2026, 6, 8, tzinfo=timezone.utc), rows)


def test_scan_ranks_by_drop_and_flags_reverting_backwardation():
    rows = scan(FakeSource(), on_date=date(2026, 6, 10), min_dte=8, max_dte=10,
                gaps=[1], ref_delta=0.5, by="drop")
    assert rows, "expected candidates"
    top = rows[0]
    assert top.ratio < top.ratio_prev      # reverting down
    assert top.drop > 0
    assert top.flag is True                 # ratio>1 (backwardation) AND drop>0
    # sorted by drop descending
    assert all(rows[i].drop >= rows[i + 1].drop for i in range(len(rows) - 1))
