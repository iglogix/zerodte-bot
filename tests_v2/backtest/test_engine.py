"""Tests for src_v2.backtest.engine.BacktestEngine."""

import pickle
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def _make_bar(hour, minute, open_, high, low, close, volume=100, d=date(2025, 10, 1)):
    return {
        "ts_et": datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }


def _make_orb_bars(d=date(2025, 10, 1)):
    """ORB candle: 08:00-08:14, range = 5920-5930, midline = 5925."""
    return [_make_bar(8, m, 5924, 5930, 5920, 5925, d=d) for m in range(15)]


def _make_breakout_bars(bias="long", d=date(2025, 10, 1)):
    """Post-open bars: breakout + midline retest + fill + monitoring bars."""
    bars = []
    if bias == "long":
        bars.append(_make_bar(9, 30, 5925, 5928, 5922, 5926, d=d))
        bars.append(_make_bar(9, 31, 5926, 5935, 5926, 5932, d=d))
        bars.append(_make_bar(9, 32, 5932, 5932, 5924, 5926, d=d))
        bars.append(_make_bar(9, 33, 5927, 5935, 5926, 5933, d=d))
        for m in range(34, 41):
            bars.append(_make_bar(9, m, 5933, 5938, 5930, 5935, d=d))
    return bars


def _make_eod_bar(d=date(2025, 10, 1)):
    return _make_bar(15, 50, 5935, 5936, 5934, 5935, d=d)


@pytest.fixture
def cbbo_cache(tmp_path):
    """CBBO cache with quotes for orb_em_ic strategy."""
    quotes = {}
    for minute in ["09:33", "09:34", "09:35", "09:36", "09:37", "09:38", "09:39", "09:40", "15:50"]:
        # ATM quotes at 5935 (nearest strike to derived spot)
        quotes[(minute, 5935.0, False)] = (12.0, 13.0)
        quotes[(minute, 5935.0, True)] = (13.0, 14.0)
        # Also at 5925 and 5930 for put-call parity derivation
        quotes[(minute, 5925.0, False)] = (15.0, 16.0)
        quotes[(minute, 5925.0, True)] = (10.0, 11.0)
        quotes[(minute, 5930.0, False)] = (14.0, 15.0)
        quotes[(minute, 5930.0, True)] = (11.0, 12.0)
        # Put spread candidates (OTM puts)
        quotes[(minute, 5920.0, False)] = (5.00, 5.50)
        quotes[(minute, 5915.0, False)] = (3.50, 4.00)
        quotes[(minute, 5910.0, False)] = (2.50, 3.00)
        quotes[(minute, 5905.0, False)] = (1.50, 2.00)
        # Call spread candidates (OTM calls)
        quotes[(minute, 5950.0, True)] = (5.00, 5.50)
        quotes[(minute, 5955.0, True)] = (3.50, 4.00)
        quotes[(minute, 5960.0, True)] = (2.00, 2.50)
        quotes[(minute, 5965.0, True)] = (1.00, 1.50)
        quotes[(minute, 5970.0, True)] = (0.50, 1.00)
        # Fill remaining strikes with stubs
        for k in range(5800, 6100, 5):
            if (minute, float(k), False) not in quotes:
                quotes[(minute, float(k), False)] = (0.05, 0.15)
            if (minute, float(k), True) not in quotes:
                quotes[(minute, float(k), True)] = (0.05, 0.15)

    pkl_path = tmp_path / "2025-10-01.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(quotes, f)
    return str(tmp_path)


@pytest.fixture
def base_config(cbbo_cache):
    return {
        "bot": {"symbol": "ES", "strategy": "orb_em_ic"},
        "strategy": {
            "orb_window_start": "08:00",
            "orb_window_end": "08:15",
            "market_open": "09:30",
            "eod_force_close": "15:50",
            "midline_tolerance_pts": 1.0,
            "orb_range_filter": {"min_pts": 2.0, "max_pts": 30.0},
            "skip_nfp_fridays": False,
            "skip_fomc_days": False,
            "skip_cpi_days": False,
        },
        "options": {
            "option_root": "SPXW",
            "strike_step_pts": 5,
            "min_credit_per_spread": 0.05,
            "tp_pct_of_credit": 0.50,
            "sl_mult_of_credit": 0,
            "hold_put_breach": True,
            "em_with_trend": 0.50,
            "em_against_trend": 0.75,
            "width_with_trend": 5,
            "width_against_trend": 10,
            "slippage_per_spread": 0.10,
            "contract_multiplier": 100,
            "quantity": 1,
        },
        "market": {
            "trading_days_per_year": 252,
            "trading_hours_per_day": 6.5,
            "market_close": "16:00",
        },
        "risk": {"max_loss_per_trade_usd": 50000},
        "backtest": {"cbbo_cache_dir": cbbo_cache},
    }


def test_skip_no_orb_data(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)
    day = DayBars(date(2025, 10, 1), "SPX", [_make_bar(10, 0, 5925, 5930, 5920, 5925)])
    results = engine.run([day])
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].skip_reason == "no_orb_data"


def test_skip_range_too_narrow(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)
    bars = [_make_bar(8, m, 5925, 5925.5, 5924.5, 5925) for m in range(15)]
    bars.append(_make_bar(9, 30, 5925, 5926, 5924, 5925))
    day = DayBars(date(2025, 10, 1), "SPX", bars)
    results = engine.run([day])
    assert results[0].status == "skipped"
    assert "range_too_narrow" in results[0].skip_reason


def test_skip_no_breakout(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)
    orb = _make_orb_bars()
    post = [_make_bar(9, 30+m, 5925, 5928, 5922, 5925) for m in range(10)]
    post.append(_make_eod_bar())
    day = DayBars(date(2025, 10, 1), "SPX", orb + post)
    results = engine.run([day])
    assert results[0].status == "skipped"
    assert results[0].skip_reason == "no_breakout"


def test_full_trade_expired_worthless(base_config):
    from src_v2.backtest.data_loader import DayBars
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.orb_em_ic import OrbEmIronCondorStrategy

    strategy = OrbEmIronCondorStrategy()
    engine = BacktestEngine(base_config, strategy)

    orb = _make_orb_bars()
    post = _make_breakout_bars("long")
    post.append(_make_eod_bar())
    day = DayBars(date(2025, 10, 1), "SPX", orb + post)

    results = engine.run([day])
    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2

    for r in closed:
        assert r.bias == "long"
        assert r.credit_received is not None
        assert r.credit_received > 0
        assert r.exit_reason in ("expired_worthless", "tp_50pct")
        if r.exit_reason == "expired_worthless":
            assert r.debit_paid == 0.0
            assert r.realized_pnl_usd > 0
