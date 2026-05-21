"""MEIC backtest smoke tests with a fake CBBO store."""

import pytest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src_v2.strategies.meic import MeicStrategy
from src_v2.backtest.engine import BacktestEngine

ET = ZoneInfo("America/New_York")


class _FakeCbbo:
    """Minimal CbboStore stand-in with optional time-shifted marks.

    Before `shift_minute` (or always, if None) returns `marks`. From
    `shift_minute` onward, any key present in `shift_marks` overrides.
    This lets a trade enter with a valid in-band credit, then have a
    leg's mark move (breach or decay) later in the session.
    """
    def __init__(self, marks, shift_minute=None, shift_marks=None):
        self._marks = marks                       # {(strike, is_call): (bid, ask)}
        self._shift_minute = shift_minute         # "HH:MM" or None
        self._shift_marks = shift_marks or {}

    def load_day(self, d):
        return len(self._marks)

    def derive_spot(self, minute, ref):
        return 5850.0

    def lookup(self, minute, strike, is_call):
        key = (float(strike), is_call)
        if self._shift_minute and minute >= self._shift_minute:
            if key in self._shift_marks:
                return self._shift_marks[key]
        return self._marks.get(key)


class _Day:
    def __init__(self, bars, sd, symbol="SPX"):
        self.bars = bars
        self.session_date = sd
        self.symbol = symbol


def _bars_flat(d, start="11:00", end="15:50", price=5850.0):
    """One bar per minute, flat price (never breaches any short)."""
    out = []
    h0, m0 = (int(x) for x in start.split(":"))
    h1, m1 = (int(x) for x in end.split(":"))
    t = datetime(d.year, d.month, d.day, h0, m0, tzinfo=ET)
    end_t = datetime(d.year, d.month, d.day, h1, m1, tzinfo=ET)
    while t <= end_t:
        out.append({"ts_et": t, "open": price, "high": price,
                    "low": price, "close": price})
        t = t.replace(minute=t.minute + 1) if t.minute < 59 else \
            t.replace(hour=t.hour + 1, minute=0)
    return out


def _meic_marks():
    """Marks giving call & put credits ~1.40 in AM band; longs cheap."""
    return {
        (5950.0, True): (1.40, 1.50),   # short call
        (6040.0, True): (0.01, 0.09),   # long call (wing 90)
        (5750.0, False): (1.40, 1.50),  # short put
        (5660.0, False): (0.01, 0.09),  # long put (wing 90)
    }


def _config():
    return {
        "bot": {"strategy": "meic"},
        "entry": {"schedule": ["11:00"]},
        "strategy": {
            "am_pm_cutoff": "12:30",
            "am_min_credit": 1.25, "am_max_credit": 1.65,
            "pm_min_credit": 0.80, "pm_max_credit": 1.30,
            "wing_widths": [90],
            "stop_pct_of_total_credit": 0.70,
            "am_scale_out_tiers": [0.55, 0.65, 0.75, 0.85],
            "am_scale_out_fractions": [0.25, 0.25, 0.25, 0.25],
            "pm_scale_out_tiers": [0.65, 0.85],
            "pm_scale_out_fractions": [0.50, 0.50],
            "abandon_long_below": 0.05,
            "credit_equality_threshold": 0.30,
            "default_tranche_qty": 3,
            "tranche_sizing": {"11:00": 3},
            "eod_force_close": "15:50",
            "market_open": "09:30",
        },
        "options": {
            "contract_multiplier": 100, "quantity": 1,
            "strike_step_pts": 5, "option_root": "SPXW",
            "tp_pct_of_credit": 0.50,
        },
        "backtest": {"cbbo_cache_dir": ""},
    }


def _make_engine(monkeypatch, fake_cbbo, config=None):
    # Patch CbboStore construction inside BacktestEngine.__init__
    import src_v2.backtest.engine as eng
    monkeypatch.setattr(eng, "CbboStore", lambda *a, **k: fake_cbbo,
                        raising=False)
    return BacktestEngine(config or _config(), MeicStrategy())


def test_meic_backtest_worthless_expiry(monkeypatch):
    d = date(2026, 5, 20)
    engine = _make_engine(monkeypatch, _FakeCbbo(_meic_marks()))
    days = [_Day(_bars_flat(d), d)]
    results = engine.run(days)

    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2          # one per side
    for r in closed:
        # qty 3, credit ~1.40 -> 1.40 * 100 * 3 = 420
        assert r.realized_pnl_usd == pytest.approx(1.40 * 100 * 3, abs=1.0)
        assert r.exit_reason == "expired_worthless"
        assert r.qty == 3


def test_meic_backtest_stop_loss(monkeypatch):
    d = date(2026, 5, 20)
    # Enter at 11:00 with valid 1.40 credit; from 11:30 the call short ask
    # jumps to 3.60 (>= stop trigger 1.45 + 0.70*2.80 = 3.41) -> stop fills.
    fake = _FakeCbbo(
        _meic_marks(),
        shift_minute="11:30",
        shift_marks={(5950.0, True): (3.50, 3.60)},
    )
    engine = _make_engine(monkeypatch, fake)
    days = [_Day(_bars_flat(d), d)]
    results = engine.run(days)

    call = next(r for r in results if r.spread_type == "call")
    # stop fills at stop_amount 1.96 -> pnl per contract (1.40-1.96)*100 = -56, x3 = -168
    assert call.exit_reason == "short_stopped"
    assert call.realized_pnl_usd == pytest.approx((1.40 - 1.96) * 100 * 3, abs=1.0)
    # Put side never moved -> expires worthless for full credit.
    put = next(r for r in results if r.spread_type == "put")
    assert put.exit_reason == "expired_worthless"


def _config_qty4():
    """Config with tranche qty 4 so 25% AM tiers close 1 contract each."""
    cfg = _config()
    cfg["strategy"]["tranche_sizing"] = {"11:00": 4}
    cfg["strategy"]["default_tranche_qty"] = 4
    return cfg


def test_meic_backtest_scale_out_full_exit(monkeypatch):
    d = date(2026, 5, 20)
    # qty=4: floor(4*0.25)=1 per AM tier. Enter at 11:00 (credit 1.40);
    # from 11:30 the call short collapses to ask 0.20 -> pnl ~86% -> all
    # four AM tiers fire in one bar -> 4 contracts closed -> full exit.
    fake = _FakeCbbo(
        _meic_marks(),
        shift_minute="11:30",
        shift_marks={(5950.0, True): (0.15, 0.20)},
    )
    engine = _make_engine(monkeypatch, fake, config=_config_qty4())
    days = [_Day(_bars_flat(d), d)]
    results = engine.run(days)

    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "scaled_out"
    assert call.qty == 4
    # Each of 4 contracts bought back at ask 0.20: (1.40-0.20)*100*4 = 480
    assert call.realized_pnl_usd == pytest.approx((1.40 - 0.20) * 100 * 4, abs=1.0)


def test_meic_backtest_partial_then_expiry(monkeypatch):
    d = date(2026, 5, 20)
    # qty=4. Enter at 11:00 (credit 1.40). From 11:30 the call short sits at
    # ask 0.60 -> pnl ~57% -> only AM tier 0 (55%) fires, closing floor(4*0.25)=1
    # contract; tier 1 (65%) is NOT crossed. The remaining 3 contracts ride to
    # EOD and expire worthless.
    fake = _FakeCbbo(
        _meic_marks(),
        shift_minute="11:30",
        shift_marks={(5950.0, True): (0.55, 0.60)},
    )
    engine = _make_engine(monkeypatch, fake, config=_config_qty4())
    days = [_Day(_bars_flat(d), d)]
    results = engine.run(days)

    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "partial_then_expiry"
    # tier 0 closes 1 @ 0.60: (1.40-0.60)*100*1 = 80
    # remaining 3 expire worthless: 1.40*100*3 = 420  -> total 500
    assert call.realized_pnl_usd == pytest.approx(80 + 420, abs=1.0)
