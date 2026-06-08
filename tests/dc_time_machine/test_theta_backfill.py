import pickle
from datetime import date

from src_v2.backtest.dc_time_machine.theta_backfill import (
    fetch_expiry_session_theta, backfill_session_theta,
)

_FMT = ["ms_of_day", "bid_size", "bid_exchange", "bid", "bid_condition",
        "ask_size", "ask_exchange", "ask", "ask_condition", "date"]


def _bulk(contracts):
    """contracts: list of (strike_x1000, right, [(ms_of_day, bid, ask), ...])."""
    resp = []
    for strike, right, ticks in contracts:
        rows = [[ms, 1, 5, bid, 50, 1, 5, ask, 50, 20260202] for (ms, bid, ask) in ticks]
        resp.append({"ticks": rows,
                     "contract": {"root": "SPXW", "expiration": 20260213,
                                  "strike": strike, "right": right}})
    return {"header": {"format": _FMT}, "response": resp}


def test_fetch_filters_strike_window_and_parses_minutes():
    data = _bulk([
        (6950000, "C", [(34200000, 0.0, 0.0), (34260000, 40.0, 41.0)]),  # 09:30 zero→drop, 09:31 keep
        (6870000, "P", [(34260000, 5.0, 5.2)]),                          # lower window edge → keep
        (6000000, "C", [(34260000, 900.0, 905.0)]),                      # far ITM, out of window → drop
    ])
    out = fetch_expiry_session_theta(
        date(2026, 2, 2), date(2026, 2, 13), atm_strike=6950.0,
        half_width_pts=80, fetch_json=lambda url: data,
    )
    assert out[("09:31", 6950.0, True)] == (40.0, 41.0)
    assert out[("09:31", 6870.0, False)] == (5.0, 5.2)
    assert ("09:30", 6950.0, True) not in out          # zero/zero dropped
    assert all(strike != 6000.0 for (_, strike, _) in out)   # out-of-window excluded


def test_fetch_keeps_one_sided_wing_quotes():
    # deep-OTM wing: bid 0.0, ask 0.05 must be kept (not both-zero)
    data = _bulk([(6900000, "P", [(34260000, 0.0, 0.05)])])
    out = fetch_expiry_session_theta(date(2026, 2, 2), date(2026, 2, 13), atm_strike=6900.0,
                                     half_width_pts=80, fetch_json=lambda url: data)
    assert out[("09:31", 6900.0, False)] == (0.0, 0.05)


def test_backfill_session_theta_writes_expiry_panel(tmp_path):
    data = _bulk([(6950000, "C", [(34260000, 40.0, 41.0)])])
    path = backfill_session_theta(
        date(2026, 2, 2), [date(2026, 2, 13)], atm_strike=6950.0, out_dir=tmp_path,
        fetch_json=lambda url: data,
    )
    panel = pickle.load(open(path, "rb"))
    assert date(2026, 2, 13) in panel
    assert panel[date(2026, 2, 13)][("09:31", 6950.0, True)] == (40.0, 41.0)


def test_backfill_centers_on_data_derived_forward_not_hint(tmp_path):
    sess, expF = date(2026, 2, 2), date(2026, 2, 13)

    def bulk_for(url):
        if "20260202" in url:   # 0DTE panel: parity forward at 09:40 (34800000ms) is 7000
            return _bulk([(7000000, "C", [(34800000, 10.0, 10.2)]),
                          (7000000, "P", [(34800000, 10.0, 10.2)]),
                          (6000000, "C", [(34800000, 1000.0, 1001.0)])])
        return _bulk([(7000000, "C", [(34800000, 40.0, 41.0)]),
                      (6000000, "C", [(34800000, 1000.0, 1001.0)])])

    # atm_strike hint is deliberately wrong (9999); data-derived center (7000) must win
    path = backfill_session_theta(sess, [sess, expF], atm_strike=9999.0, out_dir=tmp_path,
                                  ref_minute="09:40", half_width_pts=100, fetch_json=bulk_for)
    panel = pickle.load(open(path, "rb"))
    assert any(s == 7000.0 for (_, s, _) in panel[expF])      # near the real forward → kept
    assert all(s != 6000.0 for (_, s, _) in panel[expF])      # far ITM → dropped by centered window
