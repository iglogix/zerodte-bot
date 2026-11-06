"""Unit tests for OpenStrangleStrategy."""

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from zerodte.strategies.open_strangle import OpenStrangleStrategy

ET = ZoneInfo("America/New_York")


def _make_chain(spot: float, atm_credit: float = 10.0):
    """Build a synthetic chain centered on spot.

    ATM straddle = atm_credit (so EM = atm_credit).
    Symmetric prices decreasing linearly with strike distance.
    """
    strikes = [spot + i * 5 for i in range(-30, 31)]
    chain = []
    for k in strikes:
        d = abs(k - spot)
        # Linear taper, floor at $0.05
        c_mid = max(0.05, atm_credit / 2 - d * 0.10)
        p_mid = max(0.05, atm_credit / 2 - d * 0.10)
        chain.append(
            {
                "strike": k,
                "option_type": "call",
                "bid": c_mid - 0.02,
                "ask": c_mid + 0.02,
            }
        )
        chain.append(
            {
                "strike": k,
                "option_type": "put",
                "bid": p_mid - 0.02,
                "ask": p_mid + 0.02,
            }
        )
    return chain


def _make_strategy(**opt_overrides):
    s = OpenStrangleStrategy()
    cfg = {
        "options": {
            "strike_step_pts": 5.0,
            "strike_mode": "em_mult",
            "em_mult": 0.30,
            "min_credit_per_spread": 0.05,
            "slippage_per_spread": 0.05,
            "contract_multiplier": 100,
            "quantity": 1,
            "tp_pct_of_credit": 0.50,
            "sl_dollars_csv": None,
            "sl_dollars_fallback": 300.0,
            **opt_overrides,
        },
        "market": {"market_close": "16:00"},
        "strategy": {"eod_force_close": "15:30"},
    }
    s.configure(cfg)
    s._now_dt = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    return s


def test_search_candidates_returns_two_legs():
    s = _make_strategy()
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    assert out["candidates_found"] == 1
    setup = out["top_candidate"]
    assert len(setup.legs) == 2, f"expected 2 legs, got {len(setup.legs)}"
    sides = sorted(leg.option_type for leg in setup.legs)
    assert sides == ["call", "put"]
    assert all(leg.side == "sell" for leg in setup.legs)


def test_gross_credit_no_wing_subtraction():
    s = _make_strategy()
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    setup = out["top_candidate"]
    # Strangle credit = put_mid + call_mid - 2 * slippage; no wings.
    assert setup.total_credit > 0
    assert "put_credit" in setup.extra and "call_credit" in setup.extra
    # Total = put_credit + call_credit (post-slippage)
    expected = round(setup.extra["put_credit"] + setup.extra["call_credit"], 2)
    assert setup.total_credit == expected


def test_em_mult_strike_placement():
    """At em_mult=0.30 with EM=10pt and 5pt strike step:
    0.30 × 10 = 3.0 pt offset.
    Put: floor((5000 - 3) / 5) * 5 = 4995.
    Call: ceil((5000 + 3) / 5) * 5 = 5005."""
    s = _make_strategy(em_mult=0.30)
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    setup = out["top_candidate"]
    assert setup.strikes["short_put"] == 4995.0
    assert setup.strikes["short_call"] == 5005.0
    assert "long_put" not in setup.strikes
    assert "long_call" not in setup.strikes


def _write_sl_artifact(tmp_path, rows, p75):
    """Write a temporary IC SL-by-day artifact + meta.json."""
    import json

    csv_path = tmp_path / "sl.csv"
    with open(csv_path, "w") as f:
        f.write("trade_date,ic_total_credit,sl_mult,ic_sl_dollars\n")
        for d, sl in rows:
            f.write(f"{d},{sl/2/100:.2f},2.0,{sl:.2f}\n")
    meta_path = tmp_path / "sl.meta.json"
    meta_path.write_text(json.dumps({"p75": p75, "median": 250, "mean": 280}))
    return str(csv_path)


def test_sl_lookup_uses_per_day_value(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 240.0), ("2025-06-03", 280.0)],
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    when = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    assert s._sl_dollars_for(when) == 240.0  # per-day < p75 → per-day


def test_sl_lookup_caps_at_p75(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 900.0)],  # per-day > p75
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    when = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    assert s._sl_dollars_for(when) == 350.0  # capped at p75


def test_sl_lookup_fallback_when_date_missing(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 240.0)],
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    when = datetime(2025, 7, 15, 9, 35, tzinfo=ET)  # not in artifact
    assert s._sl_dollars_for(when) == 350.0  # falls back to p75


def test_sl_lookup_no_artifact_uses_fallback():
    s = _make_strategy(sl_dollars_csv=None, sl_dollars_fallback=300.0)
    when = datetime(2025, 6, 2, 9, 35, tzinfo=ET)
    assert s._sl_dollars_for(when) == 300.0


def test_setup_records_sl_dollars(tmp_path):
    csv = _write_sl_artifact(
        tmp_path,
        rows=[("2025-06-02", 275.0)],
        p75=350.0,
    )
    s = _make_strategy(sl_dollars_csv=csv, sl_dollars_fallback=300.0)
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    out = asyncio.run(s.search_candidates(chain, spot=5000.0))
    setup = out["top_candidate"]
    assert setup.extra["sl_dollars"] == 275.0


# ---------------- exit logic ----------------


def _entered_trade(setup, sl_dollars=300.0, entry_dt=None):
    """Build a trade dict shaped the way PositionMonitor/backtest passes it."""
    return {
        "trade_id": "t1",
        "short_put_symbol": "SPX_P_4995",
        "short_call_symbol": "SPX_C_5005",
        "short_put_strike": setup.strikes["short_put"],
        "short_call_strike": setup.strikes["short_call"],
        "put_credit": setup.extra["put_credit"],
        "call_credit": setup.extra["call_credit"],
        "total_credit": setup.total_credit,
        "sl_dollars": setup.extra.get("sl_dollars", sl_dollars),
        "entry_time": entry_dt or datetime(2025, 6, 2, 9, 35, tzinfo=ET),
    }


def _quotes(put_mid: float, call_mid: float):
    return {
        "SPX_P_4995": {"bid": put_mid - 0.02, "ask": put_mid + 0.02},
        "SPX_C_5005": {"bid": call_mid - 0.02, "ask": call_mid + 0.02},
    }


def test_check_exit_hold_when_quiet():
    s = _make_strategy(tp_pct_of_credit=0.50)
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    trade = _entered_trade(setup)
    # Marks at entry credit → no TP or SL trigger.
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] + s._slippage,
        call_mid=setup.extra["call_credit"] + s._slippage,
    )
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    assert action.action == "hold"


def test_check_exit_total_tp_fires():
    s = _make_strategy(tp_pct_of_credit=0.50, tp_mode="total")
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    trade = _entered_trade(setup)
    # Marks at half the credit → total_mark = 0.5*total_credit → TP fires.
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] * 0.5,
        call_mid=setup.extra["call_credit"] * 0.5,
    )
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    assert action.action == "take_profit"
    assert "SPX_P_4995" in action.legs_to_close
    assert "SPX_C_5005" in action.legs_to_close


def test_check_exit_sl_dollars_fires():
    s = _make_strategy(tp_pct_of_credit=0.25, tp_mode="total")
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    # Manually set per-trade SL$ to $50 for a clean trigger.
    setup.extra["sl_dollars"] = 50.0
    trade = _entered_trade(setup)
    trade["sl_dollars"] = 50.0
    # SL fires when (total_mark - total_credit) * 100 >= sl_dollars.
    # total_credit ≈ put_credit + call_credit (per-contract dollars).
    # Choose marks so total_mark = total_credit + 0.51 → loss = $51 > $50 SL.
    bump = (50.0 / 100) + 0.01  # 0.51 in per-contract terms
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] + bump / 2,
        call_mid=setup.extra["call_credit"] + bump / 2,
    )
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    assert action.action == "stop_loss"
    assert "sl_dollars" in action.reason or "SL" in action.reason


def test_check_exit_no_breach_action():
    """Strangle does not exit on spot-breach — relies on $-SL only."""
    s = _make_strategy(tp_pct_of_credit=0.25, tp_mode="total")
    chain = _make_chain(spot=5000.0, atm_credit=10.0)
    setup = asyncio.run(s.search_candidates(chain, 5000.0))["top_candidate"]
    trade = _entered_trade(setup)
    trade["sl_dollars"] = 1_000_000.0  # effectively unlimited
    # Spot crosses short call but SL$ huge → still hold.
    quotes = _quotes(
        put_mid=setup.extra["put_credit"] * 0.5,
        call_mid=setup.extra["call_credit"] * 0.5,
    )
    quotes["_underlying"] = {"price": setup.strikes["short_call"] + 50}
    s._now_dt = datetime(2025, 6, 2, 10, 0, tzinfo=ET)
    action = s.check_exit(trade, quotes)
    # tp_pct=0.25 means TP requires mark <= 0.25*credit. With marks at 0.5*credit, no TP.
    # No breach action ever fires for strangle → must be hold.
    assert action.action == "hold", f"expected hold, got {action.action}"
