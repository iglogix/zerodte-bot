# tests/flux/test_live.py
from datetime import date, datetime, timezone

from zerodte.flux.models import OptionRow, ChainSnapshot
from zerodte.flux.sources.base import FluxSource
from zerodte.flux.live import LivePoller

ET_OPEN = datetime(2026, 6, 8, 14, 35, tzinfo=timezone.utc)   # 10:35 ET (open)
ET_CLOSED = datetime(2026, 6, 8, 2, 0, tzinfo=timezone.utc)   # overnight (closed)


class FakeSource(FluxSource):
    def chain(self, expiry, on_date=None):
        is_front = expiry.day % 2 == 0
        iv = 0.22 if is_front else 0.20
        return ChainSnapshot("SPX", expiry, datetime.now(timezone.utc),
                             (OptionRow(7000, True, 1, 1, iv, 0.5),
                              OptionRow(7000, False, 1, 1, iv, -0.5)))


def test_poll_appends_when_open():
    p = LivePoller(FakeSource(), front_dte=9, gap=1, ref_delta=0.5, now_utc=lambda: ET_OPEN)
    p.poll_once()
    assert len(p.buffer) == 1 and abs(p.buffer[0].ratio - 1.1) < 1e-9


def test_poll_noop_when_closed():
    p = LivePoller(FakeSource(), front_dte=9, gap=1, ref_delta=0.5, now_utc=lambda: ET_CLOSED)
    p.poll_once()
    assert p.buffer == []
