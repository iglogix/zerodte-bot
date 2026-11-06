from datetime import date, datetime

from zerodte.flux.models import ChainSnapshot, OptionRow, RatioPoint
from zerodte.flux.sources.theta_intraday import ET, _ts
from zerodte.flux.sources.tastytrade_intraday import TastyTradeIntradaySource

FRONT, BACK = date(2026, 6, 16), date(2026, 6, 18)
TODAY = date(2026, 6, 9)
NOW = lambda: datetime(2026, 6, 9, 15, 4, tzinfo=ET)
KEY = (TODAY, FRONT, BACK, 0.5)


def _chain(call_mid, put_mid, strike=7000.0):
    """ATM straddle → parity spot ≈ strike; front richer than back drives ratio>1."""
    return [{"strike": strike, "option_type": "call", "bid": call_mid - 0.5, "ask": call_mid + 0.5},
            {"strike": strike, "option_type": "put", "bid": put_mid - 0.5, "ask": put_mid + 0.5}]


def _chain_fn(exp):
    return _chain(42, 42) if exp == FRONT else _chain(40, 40)   # front straddle richer → IV higher


class _Seed:
    def intraday(self, session, f, b, rd, atm):
        return [RatioPoint(_ts(session, "09:40"), f, b, rd, 0.22, 0.20, 1.10),
                RatioPoint(_ts(session, "09:50"), f, b, rd, 0.23, 0.20, 1.15)]

    def latest_chain(self, session, expiry, atm_hint=None):
        return ChainSnapshot("SPX", expiry, _ts(session, "16:00"),
                             (OptionRow(7000, True, 1, 2, 0.2, 0.5),))


def _src(seed=None, **kw):
    kw.setdefault("seed_async", False)        # synchronous seed → deterministic content tests
    return TastyTradeIntradaySource(chain_fn=_chain_fn, seed=_Seed() if seed is None else seed,
                                    today_fn=lambda: TODAY, now_fn=NOW, sleep_fn=lambda s: None, **kw)


def _mins(pts):
    return [p.as_of.astimezone(ET).strftime("%H:%M") for p in pts]


def test_seeds_morning_then_appends_live_minute():
    pts = _src().intraday(TODAY, FRONT, BACK, 0.50, 7000.0)
    assert _mins(pts) == ["09:40", "09:50", "15:04"]       # 2 seeded + 1 live, chronological
    assert pts[-1].front_iv > pts[-1].back_iv and pts[-1].ratio > 1.0


def test_live_minute_updates_in_place():
    src = _src()
    src.intraday(TODAY, FRONT, BACK, 0.50, 7000.0)
    pts = src.intraday(TODAY, FRONT, BACK, 0.50, 7000.0)
    assert sum(1 for m in _mins(pts) if m == "15:04") == 1   # overwrite, not duplicate


def test_past_session_delegates_to_seed():
    assert _mins(_src().intraday(date(2026, 6, 5), FRONT, BACK, 0.50, 7000.0)) == ["09:40", "09:50"]


def test_latest_chain_live_today():
    assert _src().latest_chain(TODAY, FRONT).rows                      # built from TT chain_fn


def test_latest_chain_past_delegates_to_seed():
    assert _src().latest_chain(date(2026, 6, 5), FRONT).as_of.astimezone(ET).strftime("%H:%M") == "16:00"


# --- reconnect guard + Theta fallback ---
def test_guard_retries_then_succeeds():
    calls = {"n": 0}
    def flaky(exp):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("DXLink dropped")
        return _chain_fn(exp)
    src = TastyTradeIntradaySource(chain_fn=flaky, seed=_Seed(), today_fn=lambda: TODAY, now_fn=NOW,
                                   retries=3, sleep_fn=lambda s: None, seed_async=False)
    assert src.latest_chain(TODAY, FRONT).rows and calls["n"] == 3


def test_tt_failure_falls_back_to_theta_live_chain():
    def dead(exp):
        raise RuntimeError("DXLink wedged")
    src = TastyTradeIntradaySource(chain_fn=dead, seed=_Seed(), today_fn=lambda: TODAY, now_fn=NOW,
                                   retries=1, sleep_fn=lambda s: None, seed_async=False)
    snap = src.latest_chain(TODAY, FRONT)                  # TT down → Theta latest_chain (real prices)
    assert snap.rows and snap.as_of.astimezone(ET).strftime("%H:%M") == "16:00"   # the seed's chain


def test_stale_fallback_when_no_seed():
    state = {"ok": True}
    def sometimes(exp):
        if not state["ok"]:
            raise RuntimeError("down")
        return _chain_fn(exp)
    src = TastyTradeIntradaySource(chain_fn=sometimes, seed=None, today_fn=lambda: TODAY, now_fn=NOW,
                                   retries=1, chain_ttl=0.0, sleep_fn=lambda s: None)
    good = src.latest_chain(TODAY, FRONT)
    state["ok"] = False
    assert src.latest_chain(TODAY, FRONT).rows == good.rows   # no seed → last good (stale) cache


# --- backgrounded seed ---
def test_async_seed_is_non_blocking_and_merges():
    src = TastyTradeIntradaySource(chain_fn=_chain_fn, seed=_Seed(), today_fn=lambda: TODAY,
                                   now_fn=NOW, sleep_fn=lambda s: None, seed_async=True)
    src.intraday(TODAY, FRONT, BACK, 0.50, 7000.0)          # starts seed thread, returns live
    src._run_seed(KEY, TODAY, FRONT, BACK, 0.50, 7000.0)    # idempotent merge (thread-safe w/ thread)
    assert _mins(src.intraday(TODAY, FRONT, BACK, 0.50, 7000.0)) == ["09:40", "09:50", "15:04"]


def test_merge_seed_does_not_clobber_live_point():
    src = TastyTradeIntradaySource(chain_fn=_chain_fn, seed=None, today_fn=lambda: TODAY, now_fn=NOW)
    src._acc[KEY] = {"15:04": RatioPoint(_ts(TODAY, "15:04"), FRONT, BACK, 0.5, 0.30, 0.20, 1.5)}
    src._merge_seed(KEY, [RatioPoint(_ts(TODAY, "15:04"), FRONT, BACK, 0.5, 0.9, 0.9, 1.0),
                          RatioPoint(_ts(TODAY, "09:40"), FRONT, BACK, 0.5, 0.22, 0.20, 1.1)])
    assert src._acc[KEY]["15:04"].ratio == 1.5             # live preserved
    assert src._acc[KEY]["09:40"].ratio == 1.1             # morning added
