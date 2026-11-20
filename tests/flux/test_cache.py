# tests/flux/test_cache.py
from datetime import date, datetime, timezone

from zerodte.flux.cache import ChainCache
from zerodte.flux.models import ChainSnapshot, OptionRow


def _snap():
    return ChainSnapshot("SPX", date(2026, 6, 12),
                         datetime(2026, 4, 9, 20, tzinfo=timezone.utc),
                         (OptionRow(7000, True, 40, 41, 0.18, 0.5),))


def test_put_then_get_roundtrip(tmp_path):
    cache = ChainCache(tmp_path / "f.db")
    assert cache.get("SPX", date(2026, 6, 12), date(2026, 4, 9)) is None
    cache.put(_snap(), on_date=date(2026, 4, 9))
    got = cache.get("SPX", date(2026, 6, 12), date(2026, 4, 9))
    assert got is not None and got.rows[0].iv == 0.18 and got.expiry == date(2026, 6, 12)
