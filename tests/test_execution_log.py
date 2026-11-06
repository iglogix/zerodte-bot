"""Tests for the MEIC execution-comparison log + join."""

from zerodte.services.execution_log import ExecutionLog, compare, compare_quotes


def _assumed(trade_id="T1", put_trig=3.41, call_trig=3.41):
    return {
        "type": "assumed", "trade_id": trade_id, "kind": "entry",
        "legs": {
            "short_put": {"symbol": "SPXW  260522P05750000",
                          "assumed_credit": 1.40, "short_mid": 1.45,
                          "stop_trigger": put_trig},
            "short_call": {"symbol": "SPXW  260522C05950000",
                           "assumed_credit": 1.40, "short_mid": 1.45,
                           "stop_trigger": call_trig},
        },
    }


def _fill(symbol, price, is_stop=True, order_type="Stop"):
    return {"type": "fill", "symbol": symbol, "fill_price": price,
            "order_type": order_type, "is_stop_fill": is_stop, "quantity": 3}


def test_compare_matches_stop_fill_to_trigger():
    recs = [_assumed(call_trig=3.41),
            _fill("SPXW  260522C05950000", 3.45)]  # filled 0.04 past trigger
    rows = compare(recs)
    assert len(rows) == 1
    r = rows[0]
    assert r["match"] == "matched"
    assert r["role"] == "short_call"
    assert r["slippage"] == 0.04
    assert "trigger" in r["implied_model"]


def test_compare_flags_deep_slippage_as_combo():
    recs = [_assumed(call_trig=3.41),
            _fill("SPXW  260522C05950000", 6.50)]  # 3.09 past trigger
    rows = compare(recs)
    assert rows[0]["slippage"] == 3.09
    assert "combo" in rows[0]["implied_model"]


def test_compare_ignores_non_stop_fills():
    recs = [_assumed(),
            _fill("SPXW  260522C05950000", 1.45, is_stop=False, order_type="Limit")]
    assert compare(recs) == []


def test_compare_unmatched_stop_fill():
    recs = [_fill("SPXW  260522C09999000", 5.00)]
    rows = compare(recs)
    assert rows[0]["match"] == "unmatched"


def test_log_round_trip(tmp_path):
    el = ExecutionLog(base_dir=str(tmp_path))
    payload = {
        "quantity": 3, "total_credit": 2.80, "stop_amount": 1.96,
        "put_spread": {"credit": 1.40, "short_mid": 1.45, "stop_trigger": 3.41},
        "call_spread": {"credit": 1.40, "short_mid": 1.45, "stop_trigger": 3.41},
    }
    el.log_assumed_entry("T1", payload, "SPXW  260522P05750000", "SPXW  260522C05950000")
    el.log_fill("SPXW  260522C05950000", 3.50, "Stop", True, 3)

    import json
    from datetime import date
    lines = (tmp_path / "execution_comparison" / f"{date.today().isoformat()}.jsonl").read_text().splitlines()
    recs = [json.loads(x) for x in lines]
    assert recs[0]["type"] == "assumed" and recs[0]["trade_id"] == "T1"
    assert recs[1]["type"] == "fill" and recs[1]["is_stop_fill"] is True
    rows = compare(recs)
    assert rows[0]["slippage"] == 0.09


def test_read_today_assumed_round_trip(tmp_path):
    el = ExecutionLog(base_dir=str(tmp_path))
    payload = {
        "quantity": 3, "total_credit": 2.80, "stop_amount": 1.96,
        "put_spread": {"credit": 1.40, "short_mid": 1.45, "stop_trigger": 3.41},
        "call_spread": {"credit": 1.40, "short_mid": 1.45, "stop_trigger": 3.41},
    }
    el.log_assumed_entry("T1", payload, "SPXW  260522P05750000", "SPXW  260522C05950000")
    el.log_fill("SPXW  260522C05950000", 3.50, "Stop", True, 3)  # non-assumed noise
    el.log_assumed_entry("T2", payload, "SPXW  260522P05700000", "SPXW  260522C06000000")

    rows = el.read_today_assumed()
    assert [r["trade_id"] for r in rows] == ["T1", "T2"]
    assert rows[0]["legs"]["short_put"]["symbol"] == "SPXW  260522P05750000"
    assert rows[0]["legs"]["short_call"]["stop_trigger"] == 3.41


def test_read_today_assumed_empty_when_no_file(tmp_path):
    el = ExecutionLog(base_dir=str(tmp_path))
    assert el.read_today_assumed() == []


def _quote(trade_id, role, symbol, ask, trigger, ts):
    return {"type": "quote", "trade_id": trade_id, "role": role, "symbol": symbol,
            "real_bid": ask - 0.05, "real_ask": ask, "trigger": trigger,
            "crossed": ask >= trigger, "ts": ts}


def test_compare_quotes_first_cross_and_overshoot():
    recs = [
        _quote("T1", "short_call", "C", 1.0, 3.41, "t1"),   # below
        _quote("T1", "short_call", "C", 3.80, 3.41, "t2"),  # first cross, overshoot 0.39
        _quote("T1", "short_call", "C", 5.0, 3.41, "t3"),   # still crossed
    ]
    rows = compare_quotes(recs)
    assert len(rows) == 1
    r = rows[0]
    assert r["crossed"] is True
    assert r["first_cross_ask"] == 3.80
    assert r["overshoot_at_cross"] == 0.39
    assert r["max_ask"] == 5.0


def test_compare_quotes_never_crossed():
    recs = [_quote("T1", "short_put", "P", 1.0, 3.41, "t1"),
            _quote("T1", "short_put", "P", 1.5, 3.41, "t2")]
    rows = compare_quotes(recs)
    assert rows[0]["crossed"] is False
    assert rows[0]["overshoot_at_cross"] is None
