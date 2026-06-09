from datetime import date, datetime

from src_v2.flux.models import RatioPoint
from src_v2.flux.sources.theta_intraday import ET, _ts
from src_v2.flux.sources.tastytrade_intraday import TastyTradeIntradaySource

FRONT, BACK = date(2026, 6, 16), date(2026, 6, 18)
TODAY = date(2026, 6, 9)


def _chain(call_mid, put_mid, strike=7000.0):
    """ATM straddle → parity spot ≈ strike; front richer than back drives ratio>1."""
    return [{"strike": strike, "option_type": "call", "bid": call_mid - 0.5, "ask": call_mid + 0.5},
            {"strike": strike, "option_type": "put", "bid": put_mid - 0.5, "ask": put_mid + 0.5}]


def _chain_fn(exp):
    return _chain(42, 42) if exp == FRONT else _chain(40, 40)   # front straddle richer → IV higher


class _Seed:
    """Stand-in morning backfill (Theta): two early minutes + a past-session series."""
    def intraday(self, session, f, b, rd, atm):
        return [RatioPoint(_ts(session, "09:40"), f, b, rd, 0.22, 0.20, 1.10),
                RatioPoint(_ts(session, "09:50"), f, b, rd, 0.23, 0.20, 1.15)]

    def latest_chain(self, session, expiry, atm_hint=None):
        from src_v2.flux.models import ChainSnapshot, OptionRow
        return ChainSnapshot("SPX", expiry, _ts(session, "16:00"),
                             (OptionRow(7000, True, 1, 2, 0.2, 0.5),))


def _src(**kw):
    return TastyTradeIntradaySource(chain_fn=_chain_fn, seed=_Seed(), today_fn=lambda: TODAY,
                                    now_fn=lambda: datetime(2026, 6, 9, 15, 4, tzinfo=ET), **kw)


def test_tt_intraday_seeds_morning_then_appends_live_minute():
    pts = _src().intraday(TODAY, FRONT, BACK, 0.50, 7000.0)
    mins = [p.as_of.astimezone(ET).strftime("%H:%M") for p in pts]
    assert mins == ["09:40", "09:50", "15:04"]          # 2 seeded + 1 live, chronological
    live = pts[-1]
    assert live.front_iv > live.back_iv and live.ratio > 1.0   # front straddle richer → ratio>1
    assert live.front_expiry == FRONT and live.back_expiry == BACK


def test_tt_intraday_live_minute_updates_in_place():
    src = _src()
    src.intraday(TODAY, FRONT, BACK, 0.50, 7000.0)       # adds 15:04
    pts = src.intraday(TODAY, FRONT, BACK, 0.50, 7000.0)  # same minute → overwrite, not duplicate
    assert sum(1 for p in pts if p.as_of.astimezone(ET).strftime("%H:%M") == "15:04") == 1


def test_tt_intraday_past_session_delegates_to_seed():
    pts = _src().intraday(date(2026, 6, 5), FRONT, BACK, 0.50, 7000.0)
    assert [p.as_of.astimezone(ET).strftime("%H:%M") for p in pts] == ["09:40", "09:50"]


def test_tt_latest_chain_is_live_today_priced_from_tt_quotes():
    snap = _src().latest_chain(TODAY, FRONT)
    assert snap.rows and any(r.is_call for r in snap.rows)   # built from the TT chain_fn quotes


def test_tt_latest_chain_past_delegates_to_seed():
    snap = _src().latest_chain(date(2026, 6, 5), FRONT)
    assert snap.as_of.astimezone(ET).strftime("%H:%M") == "16:00"   # the seed's chain


def test_guard_retries_then_succeeds():
    calls = {"n": 0}
    def flaky(exp):
        calls["n"] += 1
        if calls["n"] <= 2:                       # first two attempts fail
            raise RuntimeError("DXLink dropped")
        return _chain_fn(exp)
    src = TastyTradeIntradaySource(chain_fn=flaky, seed=_Seed(), today_fn=lambda: TODAY,
                                   now_fn=lambda: datetime(2026, 6, 9, 15, 4, tzinfo=ET),
                                   retries=3, sleep_fn=lambda s: None)   # don't actually sleep
    snap = src.latest_chain(TODAY, FRONT)
    assert snap.rows and calls["n"] == 3          # retried past the two failures


def test_guard_falls_back_to_last_good_chain_on_total_failure():
    state = {"ok": True}
    def sometimes(exp):
        if not state["ok"]:
            raise RuntimeError("DXLink wedged")
        return _chain_fn(exp)
    src = TastyTradeIntradaySource(chain_fn=sometimes, seed=_Seed(), today_fn=lambda: TODAY,
                                   now_fn=lambda: datetime(2026, 6, 9, 15, 4, tzinfo=ET),
                                   retries=1, chain_ttl=0.0, sleep_fn=lambda s: None)
    good = src.latest_chain(TODAY, FRONT)
    assert good.rows                              # first fetch succeeds → cached
    state["ok"] = False
    stale = src.latest_chain(TODAY, FRONT)        # now failing → serve last good (non-empty)
    assert stale.rows == good.rows                # chart doesn't blank out


def test_intraday_keeps_morning_when_live_fetch_fails():
    def dead(exp):
        raise RuntimeError("no broker")
    src = TastyTradeIntradaySource(chain_fn=dead, seed=_Seed(), today_fn=lambda: TODAY,
                                   now_fn=lambda: datetime(2026, 6, 9, 15, 4, tzinfo=ET),
                                   retries=1, sleep_fn=lambda s: None)
    pts = src.intraday(TODAY, FRONT, BACK, 0.50, 7000.0)   # live dead → still returns seeded morning
    assert [p.as_of.astimezone(ET).strftime("%H:%M") for p in pts] == ["09:40", "09:50"]
