"""Tests for the manual-mode QuoteComparator."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src_v2.services.quote_comparator import QuoteComparator


def _payload():
    return {
        "put_spread": {"short_strike": 5750, "stop_trigger": 3.41, "credit": 1.40, "short_mid": 1.45},
        "call_spread": {"short_strike": 5950, "stop_trigger": 3.41, "credit": 1.40, "short_mid": 1.45},
    }


@pytest.mark.asyncio
async def test_register_then_poll_logs_both_short_legs():
    log = MagicMock()
    cache = MagicMock()
    cache.get_cached_quote = AsyncMock(return_value={"bid": 0.55, "ask": 0.60})
    qc = QuoteComparator(cache=cache, execution_log=log)
    qc.register("T1", _payload(), "SPXW  260522P05750000", "SPXW  260522C05950000")

    await qc._poll_once()

    assert log.log_quote.call_count == 2  # short put + short call
    roles = {c.args[1]["role"] for c in log.log_quote.call_args_list}
    assert roles == {"short_put", "short_call"}


@pytest.mark.asyncio
async def test_poll_skips_when_no_quote():
    log = MagicMock()
    cache = MagicMock()
    cache.get_cached_quote = AsyncMock(return_value=None)
    qc = QuoteComparator(cache=cache, api_client=None, execution_log=log)
    qc.register("T1", _payload(), "P", "C")
    await qc._poll_once()
    assert log.log_quote.call_count == 0
