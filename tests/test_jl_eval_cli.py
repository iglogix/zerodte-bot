import pickle
from datetime import date

from scripts.analytical_jl_eval import load_by_date


def test_load_by_date_reads_pickles(tmp_path, monkeypatch):
    import scripts.analytical_jl_eval as mod

    cache = tmp_path / "cache"
    cache.mkdir()
    quotes = {("09:30", 5000.0, True): (10.0, 11.0)}
    with open(cache / "2025-01-02.pkl", "wb") as f:
        pickle.dump(quotes, f)

    class _Day:
        session_date = date(2025, 1, 2)
        bars = [{"ts_et": __import__("datetime").datetime(2025, 1, 2, 15, 59),
                 "close": 5012.0}]

    class _Loader:
        def __init__(self, *a, **k): pass
        def iter_days(self, start_date=None, end_date=None): return [_Day()]

    monkeypatch.setattr(mod, "BarLoader", _Loader)
    by_date = load_by_date(cache, "ignored.csv", date(2025, 1, 1), date(2025, 1, 31))
    assert set(by_date) == {date(2025, 1, 2)}
    assert by_date[date(2025, 1, 2)]["spot_close"] == 5012.0
    assert by_date[date(2025, 1, 2)]["quotes"] == quotes
