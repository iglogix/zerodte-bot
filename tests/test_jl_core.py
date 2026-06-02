from src_v2.backtest import jl_core as jl


def _chain_0930():
    """Synthetic 09:30 chain: ATM at 5000 (call_mid == put_mid == 10.5)."""
    return {
        ("09:30", 4990.0, True): (15.0, 16.0),  ("09:30", 4990.0, False): (5.0, 6.0),
        ("09:30", 4995.0, True): (12.0, 13.0),  ("09:30", 4995.0, False): (7.0, 8.0),
        ("09:30", 5000.0, True): (10.0, 11.0),  ("09:30", 5000.0, False): (10.0, 11.0),
        ("09:30", 5005.0, True): (8.0, 9.0),    ("09:30", 5005.0, False): (12.0, 13.0),
        ("09:30", 5010.0, True): (6.0, 7.0),    ("09:30", 5010.0, False): (15.0, 16.0),
    }


def test_mid_handles_one_sided_and_empty():
    assert jl._mid(10.0, 11.0) == 10.5
    assert jl._mid(0.0, 4.0) == 4.0   # bid missing -> ask
    assert jl._mid(3.0, 0.0) == 3.0   # ask missing -> bid
    assert jl._mid(0.0, 0.0) is None  # both missing


def test_quote_mid_exact_and_fallback():
    q = _chain_0930()
    assert jl.quote_mid(q, "09:30", 5000.0, True) == 10.5
    # 09:33 missing -> falls back to 09:30 within +/-5 min
    assert jl.quote_mid(q, "09:33", 5000.0, True) == 10.5
    # nothing within window
    assert jl.quote_mid(q, "12:00", 5000.0, True) is None


def test_atm_em_forward_from_chain():
    q = _chain_0930()
    assert jl.atm_from_chain(q, "09:30") == 5000.0
    assert jl.compute_em(q, "09:30", 5000.0) == 21.0          # 10.5 + 10.5
    assert jl.forward_from_chain(q, "09:30", 5000.0) == 5000.0  # 5000 + (10.5-10.5)


def test_make_anchors_10min():
    a = jl.make_anchors("09:30", "16:00", 10)
    assert a[0] == "09:30" and a[1] == "09:40" and a[-1] == "16:00"
    assert len(a) == 40   # 09:30..16:00 inclusive, step 10


def _legs_for_default_cfg():
    """With forward=5000, em=21, em_mult=0.25 -> offset 5.25:
       sp=floor(4994.75/5)*5=4990, lp=4960, sc=ceil(5005.25/5)*5=5010, lc=5020."""
    return (4990.0, 4960.0, 5010.0, 5020.0)


def test_select_strikes_default():
    cfg = jl.JlConfig()
    assert jl.select_strikes(5000.0, 21.0, cfg) == _legs_for_default_cfg()


def test_entry_credit():
    cfg = jl.JlConfig()
    q = {
        ("09:30", 4990.0, False): (5.0, 6.0),   # short put mid 5.5
        ("09:30", 4960.0, False): (2.0, 3.0),   # long put  mid 2.5 -> put credit 3.0
        ("09:30", 5010.0, True): (6.0, 7.0),    # short call mid 6.5
        ("09:30", 5020.0, True): (3.0, 4.0),    # long call mid 3.5 -> call credit 3.0
    }
    c = jl.entry_credit(q, "09:30", _legs_for_default_cfg(), cfg)
    assert c["put_credit"] == 3.0
    assert c["call_credit"] == 3.0
    assert c["total_credit"] == 6.0


def test_entry_credit_none_when_leg_missing_or_inverted():
    cfg = jl.JlConfig()
    q = {("09:30", 4990.0, False): (5.0, 6.0)}  # only one leg present
    assert jl.entry_credit(q, "09:30", _legs_for_default_cfg(), cfg) is None


def test_combined_mark():
    legs = _legs_for_default_cfg()
    q = {
        ("10:00", 4990.0, False): (1.4, 1.6),   # 1.5
        ("10:00", 4960.0, False): (0.4, 0.6),   # 0.5 -> put mark 1.0
        ("10:00", 5010.0, True): (1.9, 2.1),    # 2.0
        ("10:00", 5020.0, True): (0.4, 0.6),    # 0.5 -> call mark 1.5
    }
    assert jl.combined_mark(q, "10:00", legs) == 2.5


def _entry_chain():
    """09:30 chain that yields default legs and total_credit 6.0."""
    q = _chain_0930()                       # ATM/EM/forward = 5000/21/5000
    q.update({
        ("09:30", 4990.0, False): (5.0, 6.0),
        ("09:30", 4960.0, False): (2.0, 3.0),
        ("09:30", 5010.0, True): (6.0, 7.0),
        ("09:30", 5020.0, True): (3.0, 4.0),
    })
    return q


def test_evaluate_day_take_profit_at_mid():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"])
    q = _entry_chain()
    q.update({  # 10:00 combined mark = 1.0 + 1.5 = 2.5 <= 0.5*6 = 3.0 -> TP
        ("10:00", 4990.0, False): (1.4, 1.6), ("10:00", 4960.0, False): (0.4, 0.6),
        ("10:00", 5010.0, True): (1.9, 2.1),  ("10:00", 5020.0, True): (0.4, 0.6),
    })
    r = jl.evaluate_day(q, cfg)
    assert r["skip"] is None
    assert r["kind"] == "tp"
    assert r["exit_anchor"] == "10:00"
    assert r["pnl"] == 350.0    # (6.0 - 2.5) * 100


def test_evaluate_day_expiry_at_mid_loss():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"])
    q = _entry_chain()
    # no anchor reaches TP; final 16:00 combined mid = 8.0 -> loss
    q.update({
        ("10:00", 4990.0, False): (5.0, 5.0), ("10:00", 4960.0, False): (1.0, 1.0),
        ("10:00", 5010.0, True): (5.0, 5.0),  ("10:00", 5020.0, True): (1.0, 1.0),
        ("16:00", 4990.0, False): (9.0, 9.0), ("16:00", 4960.0, False): (1.0, 1.0),
        ("16:00", 5010.0, True): (1.0, 1.0),  ("16:00", 5020.0, True): (1.0, 1.0),
    })
    r = jl.evaluate_day(q, cfg)
    assert r["kind"] == "expiry"
    assert r["exit_anchor"] == "16:00"
    assert r["pnl"] == -200.0   # (6.0 - 8.0) * 100  [put mark 8.0 + call mark 0.0]


def test_evaluate_day_intrinsic_mode():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"], exit_mode="intrinsic")
    q = _entry_chain()
    q.update({  # never TP
        ("10:00", 4990.0, False): (5.0, 5.0), ("10:00", 4960.0, False): (1.0, 1.0),
        ("10:00", 5010.0, True): (5.0, 5.0),  ("10:00", 5020.0, True): (1.0, 1.0),
    })
    # spot_close 4980 -> put_debit = (4990-4980) - 0 = 10; call_debit 0 -> exit 10
    r = jl.evaluate_day(q, cfg, spot_close=4980.0)
    assert r["kind"] == "expiry"
    assert r["pnl"] == -400.0   # (6.0 - 10.0) * 100


def test_evaluate_day_with_costs():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"],
                      slippage_per_spread=0.10, commission_per_contract_per_leg=1.0)
    q = _entry_chain()
    q.update({
        ("10:00", 4990.0, False): (1.4, 1.6), ("10:00", 4960.0, False): (0.4, 0.6),
        ("10:00", 5010.0, True): (1.9, 2.1),  ("10:00", 5020.0, True): (0.4, 0.6),
    })
    r = jl.evaluate_day(q, cfg)
    # gross 350; entry+exit slip = 0.40*100 = 40; commission 4 legs * 1.0 = 4
    assert r["pnl"] == 306.0


def test_evaluate_day_skips():
    cfg = jl.JlConfig()
    assert jl.evaluate_day({}, cfg)["skip"] == "no_atm"


from datetime import date


def test_evaluate_aggregate():
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"])
    win_q = _entry_chain()
    win_q.update({
        ("10:00", 4990.0, False): (1.4, 1.6), ("10:00", 4960.0, False): (0.4, 0.6),
        ("10:00", 5010.0, True): (1.9, 2.1),  ("10:00", 5020.0, True): (0.4, 0.6),
    })  # TP +350
    loss_q = _entry_chain()
    loss_q.update({
        ("10:00", 4990.0, False): (5.0, 5.0), ("10:00", 4960.0, False): (1.0, 1.0),
        ("10:00", 5010.0, True): (5.0, 5.0),  ("10:00", 5020.0, True): (1.0, 1.0),
        ("16:00", 4990.0, False): (9.0, 9.0), ("16:00", 4960.0, False): (1.0, 1.0),
        ("16:00", 5010.0, True): (1.0, 1.0),  ("16:00", 5020.0, True): (1.0, 1.0),
    })  # expiry -200
    by_date = {
        date(2025, 1, 2): {"quotes": win_q, "spot_close": 5000.0},
        date(2025, 1, 3): {"quotes": loss_q, "spot_close": 4985.0},
    }
    res = jl.evaluate(by_date, cfg)
    assert res["n"] == 2
    assert res["total"] == 150.0          # 350 - 200
    assert res["wr"] == 50.0
    assert res["exits"] == {"tp": 1, "expiry": 1}
    assert res["skips"] == {}


def test_settlement_forward_prefers_chain_over_spot_close():
    # Final anchor 16:00 has an ATM chain (4995) -> settlement uses the chain
    # forward, IGNORING a deliberately-wrong spot_close (the SPY*10 basis bias).
    cfg = jl.JlConfig(anchors=["09:30", "16:00"], exit_mode="intrinsic")
    q = _entry_chain()
    q.update({
        ("16:00", 4990.0, True): (10.0, 11.0), ("16:00", 4990.0, False): (5.0, 6.0),
        ("16:00", 4995.0, True): (7.0, 8.0),   ("16:00", 4995.0, False): (7.0, 8.0),
        ("16:00", 5000.0, True): (5.0, 6.0),   ("16:00", 5000.0, False): (9.0, 10.0),
    })
    assert jl.settlement_forward(q, cfg.anchors) == 4995.0
    # At S=4995 both spreads expire worthless -> exit_mark 0, full credit kept.
    r = jl.evaluate_day(q, cfg, spot_close=4900.0)   # wrong spot_close must be ignored
    assert r["kind"] == "expiry"
    assert r["exit_mark"] == 0.0
    assert r["pnl"] == 600.0


def test_intrinsic_falls_back_to_spot_close_when_no_final_chain():
    # Final anchor 16:00 absent from quotes -> settlement_forward None -> spot_close.
    cfg = jl.JlConfig(anchors=["09:30", "10:00", "16:00"], exit_mode="intrinsic")
    q = _entry_chain()
    q.update({
        ("10:00", 4990.0, False): (5.0, 5.0), ("10:00", 4960.0, False): (1.0, 1.0),
        ("10:00", 5010.0, True): (5.0, 5.0),  ("10:00", 5020.0, True): (1.0, 1.0),
    })
    assert jl.settlement_forward(q, cfg.anchors) is None
    r = jl.evaluate_day(q, cfg, spot_close=4980.0)
    assert r["pnl"] == -400.0


def test_max_consec_losses_and_drawdown():
    assert jl._max_consec_losses([1.0, -1.0, -1.0, 2.0, -1.0]) == 2
    assert jl._max_drawdown([100.0, -50.0, -80.0, 30.0]) == -130.0
