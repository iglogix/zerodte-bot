import pickle
from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from src_v2.backtest.dc_time_machine.strategy import DcParams
from src_v2.backtest.dc_time_machine.run_backtest import backfill_range, session_expiries


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    return df.set_index("ts_recv")


def test_session_expiries_targets_front_and_back_dtes():
    # favorite config front_dte=9, gap=1, pad=1 → trading days 8..11 after Mon 2026-02-02
    exps = session_expiries(date(2026, 2, 2), DcParams(front_dte=9, gap=1), pad=1)
    assert date(2026, 2, 13) in exps   # 9 td out = front target
    assert date(2026, 2, 17) in exps   # 10 td out = back target
    assert all(e > date(2026, 2, 2) for e in exps)


def test_backfill_range_writes_one_pickle_per_session(tmp_path):
    client = MagicMock()
    client.timeseries.get_range.return_value.to_df.return_value = _df([
        {"ts_recv": "2026-02-02T14:40:00Z", "symbol": "SPXW  260213P05000000",
         "bid_px_00": 40.0, "ask_px_00": 41.0},
    ])
    atm_for = lambda s: 5000.0
    paths = backfill_range(client, [date(2026, 2, 2)], atm_for,
                           DcParams(front_dte=9, gap=1), tmp_path, pad=1)
    assert len(paths) == 1
    panel = pickle.load(open(paths[0], "rb"))
    assert date(2026, 2, 13) in panel   # front target expiry present
    assert panel[date(2026, 2, 13)][("09:40", 5000.0, False)] == (40.0, 41.0)


def test_backfill_range_skips_sessions_without_atm(tmp_path):
    client = MagicMock()
    client.timeseries.get_range.return_value.to_df.return_value = _df([
        {"ts_recv": "2026-02-02T14:40:00Z", "symbol": "SPXW  260213P05000000",
         "bid_px_00": 40.0, "ask_px_00": 41.0},
    ])
    paths = backfill_range(client, [date(2026, 2, 2)], lambda s: None, DcParams(), tmp_path)
    assert paths == []
