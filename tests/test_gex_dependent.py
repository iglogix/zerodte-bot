# tests/test_gex_dependent.py
from src_v2.backtest.gex_dependent import realized_move

def _bar(h, m, o, hi, lo, c):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return {"ts_et": datetime(2024,1,2,h,m,tzinfo=ZoneInfo("America/New_York")),
            "open": o, "high": hi, "low": lo, "close": c}

def test_realized_move_terminal_and_excursion():
    bars = [
        _bar(9,30, 5000, 5000, 5000, 5000),
        _bar(12,0, 5000, 5040, 4950, 5010),   # excursion 50 down / 40 up
        _bar(15,55, 5000, 5005, 4995, 5030),  # close 5030 -> terminal +30
    ]
    r = realized_move(bars, entry_anchor="09:30", em=40.0)
    assert r["entry_spot"] == 5000
    assert abs(r["terminal_abs_em"] - 30/40) < 1e-9
    assert abs(r["max_excursion_em"] - 50/40) < 1e-9   # max(|5040-5000|,|4950-5000|)=50

def test_realized_move_none_when_no_entry_bar():
    assert realized_move([_bar(9,31,1,1,1,1)], "09:30", 40.0) is None
