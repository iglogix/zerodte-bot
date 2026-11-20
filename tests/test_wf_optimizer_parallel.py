"""Regression test: WF-CV produces identical winner with workers=1 vs workers=4.

Backtest is fully deterministic — parallelization must preserve identical results.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zerodte.backtest.cbbo_store import CbboStore
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest.wf_optimizer import (
    WalkForwardOptimizer,
    _suggest_workers,
    _wait_for_memory,
)
from zerodte.strategies.orb_em_directional import OrbEmDirectionalStrategy

CONFIG = {
    "bot": {"name": "test", "timezone": "America/New_York", "symbol": "ES",
            "strategy": "orb_em_directional", "mode": "manual"},
    "strategy": {
        "orb_window_start": "08:00", "orb_window_end": "08:15",
        "market_open": "09:30", "eod_force_close": "15:50",
        "midline_tolerance_pts": 3.0, "retest_mode": "breakout_level",
        "orb_range_filter": {"min_pts": 2.0, "max_pts": 30.0},
        "skip_nfp_fridays": False, "skip_fomc_days": False, "skip_cpi_days": False,
    },
    "options": {
        "option_root": "SPXW", "strike_step_pts": 5,
        "min_credit_per_spread": 0.05, "tp_pct_of_credit": 0.75,
        "sl_mult_of_credit": 0, "hold_put_breach": True,
        "em_with_trend": 0.5, "width_with_trend": 20, "min_em_pts": 20,
        "slippage_per_spread": 0.10, "contract_multiplier": 100,
        "commission_per_contract_per_leg": 1.0, "quantity": 1,
        "assumed_iv": 0.15,
    },
    "entry": {"schedule": ["09:45"]},
    "risk": {"max_loss_per_trade_usd": 50000, "max_concurrent_trades": 1,
             "max_daily_risk_percent": 5.0, "max_buying_power_percent": 50.0},
    "market": {"trading_days_per_year": 252, "trading_hours_per_day": 6.5,
               "market_close": "16:00"},
    "logging": {"level": "WARNING"},
    "backtest": {"data_file": "data/spy_1m.csv", "data_source": "spy",
                 "price_scale": 10.0, "cbbo_cache_dir": "data/cbbo_cache"},
}

# Tiny grid — 4 combos × ratio 3:1 × 2 folds = 16 backtests, fast
GRID = {
    "options.tp_pct_of_credit": [0.65, 0.75],
    "options.width_with_trend": [15, 20],
}


@pytest.fixture(scope="module")
def real_data():
    """Load a 90-day window of real data — must exist in repo."""
    cbbo_dir = Path("data/cbbo_cache")
    if not cbbo_dir.exists() or len(list(cbbo_dir.glob("*.pkl"))) < 60:
        pytest.skip("data/cbbo_cache missing — run backfill first")
    data_file = Path("data/spy_1m.csv")
    if not data_file.exists():
        pytest.skip("data/spy_1m.csv missing")

    mem = CbboStore.preload_cache(cbbo_dir)
    cbbo = CbboStore(cbbo_dir, preloaded=mem)
    loader = BarLoader(str(data_file), price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=date(2025, 9, 22), end_date=date(2026, 1, 31)))
    if len(days) < 30:
        pytest.skip(f"only {len(days)} days loaded, need 30+")
    return cbbo, days


def _build_optimizer(real_data, n_workers):
    cbbo, days = real_data
    return WalkForwardOptimizer(
        base_config=CONFIG, strategy_cls=OrbEmDirectionalStrategy,
        cbbo_store=cbbo, days=days, param_grid=GRID,
        ratio=3, n_folds=2, top_k=2, min_trades=2,
        objective="total_pnl", min_sharpe=None,
        require_all_folds_profitable=False,
        n_workers=n_workers,
    )


def test_wf_optimizer_serial_baseline(real_data):
    """Sanity: serial run produces a winner."""
    opt = _build_optimizer(real_data, n_workers=1)
    result = opt.run()
    assert result.best_params is not None
    assert result.best_appearances >= 1


def test_wf_optimizer_parallel_matches_serial(real_data):
    """workers=4 produces IDENTICAL winner + aggregate metrics as workers=1."""
    serial = _build_optimizer(real_data, n_workers=1).run()
    parallel = _build_optimizer(real_data, n_workers=4).run()

    assert parallel.best_params == serial.best_params
    assert parallel.best_appearances == serial.best_appearances
    assert parallel.best_n_profitable == serial.best_n_profitable
    assert parallel.best_min_pnl == pytest.approx(serial.best_min_pnl, abs=0.01)
    assert parallel.best_mean_val_sharpe == pytest.approx(serial.best_mean_val_sharpe, abs=0.01)
    assert parallel.best_mean_val_pnl == pytest.approx(serial.best_mean_val_pnl, abs=0.01)

    # Aggregate OOS must match exactly
    for k in ("n_folds_appeared", "total_oos_trades", "total_oos_pnl",
              "aggregate_win_rate", "mean_fold_sharpe", "max_fold_drawdown"):
        assert parallel.aggregate[k] == pytest.approx(serial.aggregate[k], abs=0.01), \
            f"mismatch on {k}: serial={serial.aggregate[k]} parallel={parallel.aggregate[k]}"


# ----------------------------------------------------------------------------
# Memory throttle tests (no real backtest needed — pure unit tests with mocks)
# ----------------------------------------------------------------------------

def _fake_vmem(percent: float, available_gb: float = 20.0):
    """Build a mock psutil.virtual_memory() return."""
    from types import SimpleNamespace
    return SimpleNamespace(percent=percent, available=int(available_gb * 1e9),
                           total=int(45e9))


def test_wait_for_memory_returns_immediately_when_below_threshold():
    with patch("zerodte.backtest.wf_optimizer.psutil.virtual_memory",
               return_value=_fake_vmem(50.0)):
        t0 = time.monotonic()
        _wait_for_memory(threshold_pct=85, sleep_s=0.1, timeout_s=5)
        assert time.monotonic() - t0 < 0.1


def test_wait_for_memory_blocks_then_returns_when_pressure_subsides():
    """Mock returns 90% (above threshold), then 60% (below) — must sleep at least once."""
    seq = iter([_fake_vmem(90.0), _fake_vmem(90.0), _fake_vmem(60.0)])
    with patch("zerodte.backtest.wf_optimizer.psutil.virtual_memory",
               side_effect=lambda: next(seq)):
        t0 = time.monotonic()
        _wait_for_memory(threshold_pct=85, sleep_s=0.1, timeout_s=5)
        elapsed = time.monotonic() - t0
        assert 0.08 <= elapsed < 0.5  # at least one sleep


def test_wait_for_memory_warns_and_returns_after_timeout(caplog):
    """Always above threshold → log warning and return (no raise — avoids deadlock)."""
    with patch("zerodte.backtest.wf_optimizer.psutil.virtual_memory",
               return_value=_fake_vmem(95.0)):
        t0 = time.monotonic()
        with caplog.at_level("WARNING", logger="zerodte.backtest.wf_optimizer"):
            _wait_for_memory(threshold_pct=85, sleep_s=0.05, timeout_s=0.2)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.2
        assert any("throttle timeout" in r.message for r in caplog.records)


def test_suggest_workers_caps_when_memory_low():
    """Available=4GB, est 2GB/worker × 1.5 safety = 1 worker max even if user asks 22."""
    with patch("zerodte.backtest.wf_optimizer.psutil.virtual_memory",
               return_value=_fake_vmem(50.0, available_gb=4.0)):
        capped = _suggest_workers(requested=22, gb_per_worker=2.0, safety_factor=1.5)
        assert capped == 1


def test_suggest_workers_unchanged_when_memory_plentiful():
    """Available=80GB → easily fits 22 workers."""
    with patch("zerodte.backtest.wf_optimizer.psutil.virtual_memory",
               return_value=_fake_vmem(20.0, available_gb=80.0)):
        capped = _suggest_workers(requested=22, gb_per_worker=2.0, safety_factor=1.5)
        assert capped == 22


def test_suggest_workers_minimum_one():
    """Even with no memory, return at least 1."""
    with patch("zerodte.backtest.wf_optimizer.psutil.virtual_memory",
               return_value=_fake_vmem(99.0, available_gb=0.0)):
        capped = _suggest_workers(requested=22, gb_per_worker=2.0, safety_factor=1.5)
        assert capped == 1
