from datetime import date

from zerodte.flux.sources.theta_intraday import ThetaIntradaySource

_FMT = ["ms_of_day", "bid_size", "bid_exchange", "bid", "bid_condition",
        "ask_size", "ask_exchange", "ask", "ask_condition", "date"]


def _bulk(contracts):
    """contracts: list of (strike_x1000, right, [(ms_of_day, bid, ask), ...])."""
    resp = []
    for strike, right, ticks in contracts:
        rows = [[ms, 1, 5, bid, 50, 1, 5, ask, 50, 20260605] for (ms, bid, ask) in ticks]
        resp.append({"ticks": rows,
                     "contract": {"root": "SPXW", "expiration": 20260617,
                                  "strike": strike, "right": right}})
    return {"header": {"format": _FMT}, "response": resp}


# 09:40 = 34800000 ms, 09:50 = 35400000 ms
def _panel_for(expiry_iv_offset):
    """ATM 7000 call+put priced to imply ~given IV at two minutes."""
    # use real-ish ATM straddle prices; exact IV value isn't asserted, just monotonic/positive
    return _bulk([
        (7000000, "C", [(34800000, 40.0 + expiry_iv_offset, 41.0 + expiry_iv_offset),
                        (35400000, 42.0 + expiry_iv_offset, 43.0 + expiry_iv_offset)]),
        (7000000, "P", [(34800000, 40.0 + expiry_iv_offset, 41.0 + expiry_iv_offset),
                        (35400000, 42.0 + expiry_iv_offset, 43.0 + expiry_iv_offset)]),
    ])


def test_intraday_series_one_point_per_common_minute():
    front, back = date(2026, 6, 17), date(2026, 6, 18)

    def fetch_json(url):
        # front expiry exp=20260617 ; back exp=20260618 — distinguish by url
        return _panel_for(2.0) if "20260617" in url else _panel_for(0.0)

    src = ThetaIntradaySource(fetch_json=fetch_json, half_width_pts=80)
    pts = src.intraday(session=date(2026, 6, 5), front_exp=front, back_exp=back,
                       ref_delta=0.50, atm_hint=7000.0)
    assert len(pts) == 2                       # two common minutes (09:40, 09:50)
    p = pts[0]
    assert p.front_expiry == front and p.back_expiry == back
    assert p.front_iv > p.back_iv              # front straddle richer → higher IV → ratio>1
    assert p.ratio > 1.0
    assert pts[0].as_of < pts[1].as_of         # chronological


def test_intraday_drops_opening_auction_bar():
    # 09:30 = 34200000 ms (opening artifact), 09:40 = 34800000 ms
    def panel():
        return _bulk([(7000000, "C", [(34200000, 40, 41), (34800000, 42, 43)]),
                      (7000000, "P", [(34200000, 40, 41), (34800000, 42, 43)])])
    src = ThetaIntradaySource(fetch_json=lambda url: panel(), half_width_pts=80)
    pts = src.intraday(date(2026, 6, 5), date(2026, 6, 17), date(2026, 6, 18), 0.50, 7000.0)
    assert len(pts) == 1 and pts[0].as_of.strftime("%H:%M") == "09:40"   # 09:30 dropped


def _counting_fetch(calls):
    def fetch_json(url):
        calls["n"] += 1
        return _panel_for(2.0) if "20260617" in url else _panel_for(0.0)
    return fetch_json


def test_intraday_caches_today_within_ttl():
    calls = {"n": 0}
    clk = {"t": 1000.0}
    src = ThetaIntradaySource(fetch_json=_counting_fetch(calls), ttl_seconds=60,
                              clock=lambda: clk["t"], today_fn=lambda: date(2026, 6, 5))
    f, b = date(2026, 6, 17), date(2026, 6, 18)
    src.intraday(date(2026, 6, 5), f, b, 0.5, 7000.0)
    assert calls["n"] == 2                      # front + back fetched
    src.intraday(date(2026, 6, 5), f, b, 0.5, 7000.0)
    assert calls["n"] == 2                      # within TTL → served from cache
    clk["t"] += 120
    src.intraday(date(2026, 6, 5), f, b, 0.5, 7000.0)
    assert calls["n"] == 4                      # past TTL → refetched


def test_intraday_past_session_cached_forever():
    calls = {"n": 0}
    clk = {"t": 1000.0}
    src = ThetaIntradaySource(fetch_json=_counting_fetch(calls), ttl_seconds=1,
                              clock=lambda: clk["t"], today_fn=lambda: date(2026, 6, 10))
    f, b = date(2026, 6, 17), date(2026, 6, 18)
    src.intraday(date(2026, 6, 5), f, b, 0.5, 7000.0)   # session in the past
    assert calls["n"] == 2
    clk["t"] += 10_000                           # way past TTL
    src.intraday(date(2026, 6, 5), f, b, 0.5, 7000.0)
    assert calls["n"] == 2                        # immutable past session → still cached


def test_intraday_series_empty_when_no_data():
    src = ThetaIntradaySource(fetch_json=lambda url: {}, half_width_pts=80)
    pts = src.intraday(session=date(2026, 6, 5), front_exp=date(2026, 6, 17),
                       back_exp=date(2026, 6, 18), ref_delta=0.5, atm_hint=7000.0)
    assert pts == []
