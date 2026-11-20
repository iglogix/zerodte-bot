"""Tests for the manual-mode QuoteComparator (chain-based quote source)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from zerodte.services.execution_log import ExecutionLog
from zerodte.services.quote_comparator import QuoteComparator


def _payload():
    return {
        "put_spread": {"short_strike": 5750, "stop_trigger": 3.41, "credit": 1.40, "short_mid": 1.45},
        "call_spread": {"short_strike": 5950, "stop_trigger": 3.41, "credit": 1.40, "short_mid": 1.45},
    }


def _chain():
    return [
        {"strike": 5750.0, "option_type": "P", "bid": 0.55, "ask": 0.60},
        {"strike": 5950.0, "option_type": "C", "bid": 0.70, "ask": 0.75},
        {"strike": 5800.0, "option_type": "P", "bid": 1.0, "ask": 1.1},  # unrelated
    ]


@pytest.mark.asyncio
async def test_poll_logs_both_shorts_from_chain():
    log = MagicMock()
    th = MagicMock()
    th.get_option_chain = AsyncMock(return_value=_chain())
    qc = QuoteComparator(tool_handlers=th, execution_log=log)
    qc.register("T1", _payload(), "SPXW  260522P05750000", "SPXW  260522C05950000")

    await qc._poll_once()

    assert log.log_quote.call_count == 2
    by_role = {c.args[1]["role"]: c.args for c in log.log_quote.call_args_list}
    assert set(by_role) == {"short_put", "short_call"}
    # short_put got the 5750P bid/ask
    _, leg, bid, ask = by_role["short_put"]
    assert (bid, ask) == (0.55, 0.60)
    _, leg, bid, ask = by_role["short_call"]
    assert (bid, ask) == (0.70, 0.75)


def test_rehydrate_rebuilds_active_from_log(tmp_path):
    # A prior run logged two entries; a fresh comparator must recover them.
    el = ExecutionLog(base_dir=str(tmp_path))
    el.log_assumed_entry("meic-124015", _meic_payload(7465, 7510),
                         "SPXW  260522P07465000", "SPXW  260522C07510000")
    el.log_assumed_entry("meic-130013", _meic_payload(7470, 7515),
                         "SPXW  260522P07470000", "SPXW  260522C07515000")

    qc = QuoteComparator(tool_handlers=MagicMock(), execution_log=el)
    n = qc.rehydrate()

    assert n == 2
    assert set(qc._active) == {"meic-124015", "meic-130013"}
    legs = {leg["role"]: leg for leg in qc._active["meic-124015"]}
    assert legs["short_put"]["strike"] == 7465.0
    assert legs["short_put"]["is_call"] is False
    assert legs["short_call"]["strike"] == 7510.0
    assert legs["short_call"]["is_call"] is True
    assert legs["short_put"]["trigger"] == 3.41


def test_rehydrate_does_not_duplicate_already_tracked(tmp_path):
    el = ExecutionLog(base_dir=str(tmp_path))
    el.log_assumed_entry("meic-124015", _meic_payload(7465, 7510),
                         "SPXW  260522P07465000", "SPXW  260522C07510000")
    qc = QuoteComparator(tool_handlers=MagicMock(), execution_log=el)
    qc.register("meic-124015", _meic_payload(7465, 7510),
                "SPXW  260522P07465000", "SPXW  260522C07510000")

    n = qc.rehydrate()

    assert n == 0  # already tracked → skipped
    assert len(qc._active["meic-124015"]) == 2


def test_rehydrate_noop_without_execution_log():
    qc = QuoteComparator(tool_handlers=MagicMock(), execution_log=None)
    assert qc.rehydrate() == 0


def _meic_payload(put_k, call_k):
    return {
        "put_spread": {"short_strike": put_k, "stop_trigger": 3.41, "credit": 1.40, "short_mid": 1.45},
        "call_spread": {"short_strike": call_k, "stop_trigger": 3.41, "credit": 1.40, "short_mid": 1.45},
    }


@pytest.mark.asyncio
async def test_poll_skips_when_chain_empty():
    log = MagicMock()
    th = MagicMock()
    th.get_option_chain = AsyncMock(return_value=[])
    qc = QuoteComparator(tool_handlers=th, execution_log=log)
    qc.register("T1", _payload(), "P", "C")
    await qc._poll_once()
    assert log.log_quote.call_count == 0


@pytest.mark.asyncio
async def test_poll_skips_invalid_zero_quotes():
    # Chain enrichment failure returns 0.0 bid/ask — must not be logged as a quote.
    chain = [
        {"strike": 5750.0, "option_type": "P", "bid": 0.0, "ask": 0.0},
        {"strike": 5950.0, "option_type": "C", "bid": 0.70, "ask": 0.75},
    ]
    log = MagicMock()
    th = MagicMock()
    th.get_option_chain = AsyncMock(return_value=chain)
    qc = QuoteComparator(tool_handlers=th, execution_log=log)
    qc.register("T1", _payload(), "SPXW  260522P05750000", "SPXW  260522C05950000")

    await qc._poll_once()

    # Only the valid call leg is logged; the zero-priced put leg is skipped.
    assert log.log_quote.call_count == 1
    _, leg, bid, ask = log.log_quote.call_args_list[0].args
    assert leg["role"] == "short_call"
    assert (bid, ask) == (0.70, 0.75)
