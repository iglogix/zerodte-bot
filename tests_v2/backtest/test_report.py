"""Tests for zerodte.backtest.report."""

from datetime import date
import pytest


def _make_result(**overrides):
    from zerodte.backtest.engine import BacktestResult
    defaults = dict(
        trade_id="T1", session_date=date(2025, 10, 1), symbol="SPX",
        orb_high=5930, orb_low=5920, orb_midline=5925, orb_range_pts=10,
        status="closed",
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def test_empty_report():
    from zerodte.backtest.report import generate_report
    report = generate_report([])
    assert report["trades_placed"] == 0
    assert report["win_rate"] == 0.0


def test_basic_stats():
    from zerodte.backtest.report import generate_report

    results = [
        _make_result(trade_id="T1", realized_pnl_usd=100.0, exit_reason="tp_50pct",
                     credit_received=1.0, short_delta_at_entry=0.10, bias="long"),
        _make_result(trade_id="T2", realized_pnl_usd=80.0, exit_reason="expired_worthless",
                     credit_received=0.80, short_delta_at_entry=0.12, bias="long"),
        _make_result(trade_id="T3", realized_pnl_usd=-200.0, exit_reason="short_breached",
                     credit_received=1.20, short_delta_at_entry=0.15, bias="short",
                     session_date=date(2025, 10, 2)),
    ]

    report = generate_report(results)
    assert report["trades_placed"] == 3
    assert report["wins"] == 2
    assert report["losses"] == 1
    assert report["win_rate"] == pytest.approx(0.6667, abs=0.001)
    assert report["total_pnl_usd"] == pytest.approx(-20.0)
    assert report["profit_factor"] == pytest.approx(0.9)
    assert report["max_drawdown_usd"] == pytest.approx(200.0)
    assert report["by_exit_reason"]["tp_50pct"]["count"] == 1
    # Daily correlated-day metrics: 2025-10-01 net +180, 2025-10-02 net -200.
    assert report["worst_day_pnl_usd"] == pytest.approx(-200.0)
    assert report["best_day_pnl_usd"] == pytest.approx(180.0)
    assert report["max_drawdown_daily_usd"] == pytest.approx(200.0)


def test_worst_day_aggregates_same_day_losses():
    from zerodte.backtest.report import generate_report
    # Three losing legs on ONE day should aggregate into the worst-day metric,
    # even though no single trade is that large.
    results = [
        _make_result(trade_id="A", session_date=date(2025, 11, 7),
                     realized_pnl_usd=-150.0, exit_reason="short_stopped"),
        _make_result(trade_id="B", session_date=date(2025, 11, 7),
                     realized_pnl_usd=-200.0, exit_reason="short_stopped"),
        _make_result(trade_id="C", session_date=date(2025, 11, 7),
                     realized_pnl_usd=-250.0, exit_reason="short_stopped"),
        _make_result(trade_id="D", session_date=date(2025, 11, 10),
                     realized_pnl_usd=400.0, exit_reason="expired_worthless"),
    ]
    report = generate_report(results)
    assert report["worst_day_pnl_usd"] == pytest.approx(-600.0)   # 3 stops summed
    assert report["max_drawdown_daily_usd"] == pytest.approx(600.0)


def test_sharpe_calculation():
    from zerodte.backtest.report import generate_report

    results = [
        _make_result(trade_id="T1", session_date=date(2025, 10, 1),
                     realized_pnl_usd=50.0, exit_reason="tp", bias="long"),
        _make_result(trade_id="T2", session_date=date(2025, 10, 1),
                     realized_pnl_usd=50.0, exit_reason="tp", bias="long"),
        _make_result(trade_id="T3", session_date=date(2025, 10, 2),
                     realized_pnl_usd=100.0, exit_reason="tp", bias="short"),
    ]
    report = generate_report(results)
    assert report["sharpe_annualised"] > 0


def test_skipped_results():
    from zerodte.backtest.report import generate_report

    results = [
        _make_result(trade_id="S1", status="skipped", skip_reason="no_breakout"),
        _make_result(trade_id="S2", status="skipped", skip_reason="no_breakout"),
        _make_result(trade_id="S3", status="skipped", skip_reason="range_too_narrow"),
    ]
    report = generate_report(results)
    assert report["trades_placed"] == 0
    assert report["skipped_days"] == 3
    assert report["skip_reasons"]["no_breakout"] == 2
