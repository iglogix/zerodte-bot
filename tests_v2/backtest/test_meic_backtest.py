"""MEIC backtest smoke tests with a fake CBBO store."""

import pytest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src_v2.strategies.meic import MeicStrategy
from src_v2.backtest.engine import BacktestEngine

ET = ZoneInfo("America/New_York")


class _LinearCbbo:
    """Synthetic CBBO: ATM straddle == em; premium decays linearly at `rate`/pt
    from peak em/2, floored at `floor`. Optional time-shifted per-(strike,is_call)
    overrides from `shift_minute` onward (to drive stops / scale-outs)."""
    def __init__(self, spot, em, rate=0.10, floor=0.05, step=5, span=400,
                 shift_minute=None, shift_overrides=None,
                 eod_spot=None, eod_from="15:50"):
        self._spot = spot
        self._peak = em / 2.0
        self._rate = rate
        self._floor = floor
        self._span = span
        self._shift_minute = shift_minute
        self._shift = shift_overrides or {}
        self._eod_spot = eod_spot          # settlement spot from eod_from onward
        self._eod_from = eod_from

    def load_day(self, d):
        return 1

    def derive_spot(self, minute, ref):
        if self._eod_spot is not None and minute >= self._eod_from:
            return self._eod_spot
        return self._spot

    def settlement_forward(self):
        return self._eod_spot if self._eod_spot is not None else self._spot

    def lookup(self, minute, strike, is_call):
        key = (float(strike), is_call)
        if self._shift_minute and minute >= self._shift_minute and key in self._shift:
            return self._shift[key]
        dist = abs(strike - self._spot)
        if dist > self._span:
            return None
        mid = max(self._floor, self._peak - self._rate * dist)
        return (round(mid - 0.05, 2), round(mid + 0.05, 2))


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


def _config(tranche=None):
    return {
        "bot": {"strategy": "meic"},
        "entry": {"schedule": ["11:00"]},
        "strategy": {
            "am_pm_cutoff": "12:30",
            "short_em_mult": 1.0,
            "wing_em_frac": 0.5,
            "min_credit_width_ratio": 0.08,
            "strike_step_pts": 5,
            "stop_pct_of_total_credit": 0.70,
            "am_scale_out_tiers": [0.55, 0.65, 0.75, 0.85],
            "am_scale_out_fractions": [0.25, 0.25, 0.25, 0.25],
            "pm_scale_out_tiers": [0.65, 0.85],
            "pm_scale_out_fractions": [0.50, 0.50],
            "abandon_long_below": 0.05,
            "credit_equality_threshold": 0.30,
            "default_tranche_qty": (tranche or 3),
            "tranche_sizing": {"11:00": (tranche or 3)},
            "eod_force_close": "15:50",
            "market_open": "09:30",
        },
        "options": {
            "contract_multiplier": 100, "quantity": 1,
            "strike_step_pts": 5, "option_root": "SPXW",
            "tp_pct_of_credit": 0.50,
            "slippage_per_spread": 0.0,
            "commission_per_contract_per_leg": 0.0,
            "stop_slippage_per_spread": 0.0,
            "chain_half_span_pts": 400,
        },
        "backtest": {"cbbo_cache_dir": ""},
    }


def _make_engine(monkeypatch, fake_cbbo, config=None):
    # Patch CbboStore construction inside BacktestEngine.__init__
    import src_v2.backtest.engine as eng
    monkeypatch.setattr(eng, "CbboStore", lambda *a, **k: fake_cbbo,
                        raising=False)
    return BacktestEngine(config or _config(), MeicStrategy())


def test_meic_backtest_hold_to_expiry_settles_at_intrinsic(monkeypatch):
    # Hold-to-expiry: no stop / no scale-out; remaining qty settles at intrinsic.
    # entry spot 6700 -> call short 6820 / long 6880 (wing 60), put short 6580.
    # Rally to 6850 at settlement: call short 30 ITM (loss 30, < width 60);
    # put far OTM -> keeps full credit. credit per side ~6.0 (em 120 linear chain).
    d = date(2026, 1, 5)
    cfg = _config()
    cfg["strategy"]["exit_mode"] = "hold_to_expiry"
    cbbo = _LinearCbbo(6700, 120.0, eod_spot=6850.0)
    engine = _make_engine(monkeypatch, cbbo, cfg)
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    closed = {r.spread_type: r for r in results if r.status == "closed"}
    assert closed["call"].exit_reason == "held_to_expiry"
    assert closed["put"].exit_reason == "held_to_expiry"
    # call: (credit 6.0 - intrinsic 30) * 100 * 3 = -7200
    assert closed["call"].realized_pnl_usd == pytest.approx((6.0 - 30) * 100 * 3, abs=1.0)
    # put: OTM -> full credit kept = 6.0 * 100 * 3 = 1800
    assert closed["put"].realized_pnl_usd == pytest.approx(6.0 * 100 * 3, abs=1.0)


def test_meic_backtest_fixed_fractional_scales_qty(monkeypatch):
    # Fixed-fractional sizing: qty scales with starting equity, not the static map.
    # em 120 -> wing 60; total_credit ~12 -> IC margin (60-12)*100 = 4800/contract.
    # budget = f 0.10 * equity 2,000,000 * (weight 3 / total 3) = 200,000
    # -> floor(200000 / 4800) = 41 lots (vs static 3).
    d = date(2026, 1, 5)
    cfg = _config()
    cfg["strategy"]["sizing_mode"] = "fixed_fractional"
    cfg["strategy"]["sizing_fraction"] = 0.10
    cfg["strategy"]["exit_mode"] = "hold_to_expiry"
    cfg["backtest"]["starting_equity"] = 2_000_000
    engine = _make_engine(monkeypatch, _LinearCbbo(6700, 120.0, eod_spot=6700.0), cfg)
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    closed = [r for r in results if r.status == "closed"]
    assert closed, "expected trades"
    assert closed[0].qty == 41


def test_meic_backtest_worthless_expiry(monkeypatch):
    d = date(2026, 1, 5)
    engine = _make_engine(monkeypatch, _LinearCbbo(6700, 120.0))
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2
    for r in closed:
        assert r.realized_pnl_usd == pytest.approx(6.0 * 100 * 3, abs=1.0)
        assert r.exit_reason == "expired_worthless"
        assert r.qty == 3


def test_meic_backtest_stop_loss(monkeypatch):
    d = date(2026, 1, 5)
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (56.5, 57.0)})
    engine = _make_engine(monkeypatch, fake)
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "short_stopped"
    assert call.realized_pnl_usd == pytest.approx((6.0 - 8.4) * 100 * 3, abs=1.0)
    put = next(r for r in results if r.spread_type == "put")
    assert put.exit_reason == "expired_worthless"


def test_meic_backtest_scale_out_full_exit(monkeypatch):
    d = date(2026, 1, 5)
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (0.85, 0.90)})
    engine = _make_engine(monkeypatch, fake, config=_config(tranche=4))
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "scaled_out"
    assert call.qty == 4
    assert call.realized_pnl_usd == pytest.approx((6.0 - 0.90) * 100 * 4, abs=1.0)


def test_meic_backtest_partial_then_expiry(monkeypatch):
    d = date(2026, 1, 5)
    # short ask 2.69 -> pnl_pct (6.0-2.69)/6.0 = 0.5517: fires AM tier 0 (0.55)
    # only, not tier 1 (0.65). 2.70 lands exactly on the 0.55 boundary and
    # floats to just under, so use 2.69 to fire tier 0 cleanly.
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (2.64, 2.69)})
    engine = _make_engine(monkeypatch, fake, config=_config(tranche=4))
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "partial_then_expiry"
    # tier 0 closes 1 @ 2.69: (6.0-2.69)*100*1 = 331
    # remaining 3 expire worthless: 6.0*100*3 = 1800  -> total 2131
    assert call.realized_pnl_usd == pytest.approx(331 + 1800, abs=1.0)


def _config_costs(tranche=3):
    cfg = _config(tranche=tranche)
    cfg["options"]["slippage_per_spread"] = 0.10
    cfg["options"]["commission_per_contract_per_leg"] = 1.0
    cfg["options"]["stop_slippage_per_spread"] = 0.25
    return cfg


def test_meic_backtest_applies_costs_on_worthless_expiry(monkeypatch):
    d = date(2026, 1, 5)
    engine = _make_engine(monkeypatch, _LinearCbbo(6700, 120.0), config=_config_costs())
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2
    for r in closed:
        assert r.realized_pnl_usd == pytest.approx(5.90 * 100 * 3 - 6.0, abs=0.5)


def test_meic_backtest_applies_stop_slippage(monkeypatch):
    d = date(2026, 1, 5)
    fake = _LinearCbbo(6700, 120.0, shift_minute="11:30",
                       shift_overrides={(6820.0, True): (56.5, 57.0)})
    engine = _make_engine(monkeypatch, fake, config=_config_costs())
    results = engine.run([_Day(_bars_flat(d, price=6700.0), d)])
    call = next(r for r in results if r.spread_type == "call")
    assert call.exit_reason == "short_stopped"
    assert call.realized_pnl_usd == pytest.approx((5.90 - 8.65) * 100 * 3 - 6.0, abs=0.5)


def test_build_chain_covers_wide_span():
    import yaml
    cfg = yaml.safe_load(open("config/config_meic_spx.yaml"))

    class _AllStrikes:
        def load_day(self, d): return 1
        def derive_spot(self, m, ref): return 6700.0
        def lookup(self, m, k, is_call): return (1.0, 1.1)

    e = BacktestEngine(cfg, MeicStrategy())
    e._cbbo = _AllStrikes()
    chain = e._build_chain("11:00", 6700.0, date(2026, 1, 5))
    strikes = [o["strike"] for o in chain]
    assert max(strikes) >= 7050
    assert min(strikes) <= 6350


def test_meic_backtest_low_vol_regime_trades(monkeypatch):
    # Low-vol regime: spot 4285, EM 24 -> short_call 4310, short_put 4260,
    # wing 10, each-side credit 1.0 (cw 0.10 >= gate). The absolute-band
    # version produced ZERO trades here; EM-relative must produce trades.
    d = date(2026, 1, 5)
    engine = _make_engine(monkeypatch, _LinearCbbo(4285, 24.0))
    results = engine.run([_Day(_bars_flat(d, price=4285.0), d)])
    closed = [r for r in results if r.status == "closed"]
    assert len(closed) == 2  # one per side, both entered
    for r in closed:
        # flat low-vol day -> expires worthless for full credit (qty 3)
        assert r.exit_reason == "expired_worthless"
        assert r.realized_pnl_usd == pytest.approx(1.0 * 100 * 3, abs=1.0)
