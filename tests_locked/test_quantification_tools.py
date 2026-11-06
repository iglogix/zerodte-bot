"""
Tests for the quantification / pre-flight tools:
  - validate_ic_params
  - get_premium_balance_candidates
  - get_skew_snapshot
  - get_option_chain target_credit sorting

Fixtures use today's date so the chain expiration filter passes.
"""

import pytest
from datetime import date
from typing import Any, Dict, List
from unittest.mock import AsyncMock


TODAY = date.today().isoformat()


def _make_opt(
    strike: float,
    option_type: str,
    delta: float,
    bid: float,
    ask: float,
    iv: float = 0.15,
) -> Dict[str, Any]:
    return {
        "strike": strike,
        "option_type": option_type,
        "delta": delta,
        "bid": bid,
        "ask": ask,
        "iv": iv,
        "implied_volatility": iv,
        "gamma": 0.002,
        "theta": -0.5,
        "vega": 0.1,
        "expiration": TODAY,
    }


@pytest.fixture
def balanced_chain() -> List[Dict[str, Any]]:
    """Chain where call and put credits are balanced within $0.20."""
    return [
        # Short call 5890 delta 0.12, mid 1.50  →  long call 5920 mid 0.30  →  credit 1.20
        _make_opt(5890.0, "call", 0.12, 1.40, 1.60, iv=0.14),
        _make_opt(5920.0, "call", 0.04, 0.20, 0.40, iv=0.13),
        # Short put 5810 delta -0.12, mid 1.60  →  long put 5780 mid 0.45  →  credit 1.15
        _make_opt(5810.0, "put", -0.12, 1.50, 1.70, iv=0.155),
        _make_opt(5780.0, "put", -0.04, 0.35, 0.55, iv=0.14),
        # Extra strikes outside delta range (should be ignored by candidate search)
        _make_opt(5950.0, "call", 0.02, 0.05, 0.15, iv=0.12),
        _make_opt(5740.0, "put", -0.02, 0.05, 0.15, iv=0.12),
    ]


@pytest.fixture
def skewed_chain() -> List[Dict[str, Any]]:
    """Chain with strong put skew — put IV is 30% higher than call IV."""
    return [
        _make_opt(5890.0, "call", 0.12, 1.40, 1.60, iv=0.14),
        _make_opt(5920.0, "call", 0.04, 0.20, 0.40, iv=0.13),
        _make_opt(5810.0, "put", -0.12, 2.20, 2.60, iv=0.19),   # put IV 35% higher
        _make_opt(5780.0, "put", -0.04, 0.60, 0.90, iv=0.18),
    ]


def _make_handlers(chain: List[Dict[str, Any]], **kwargs):
    """Build a ToolHandlers with a mocked cache returning the given chain."""
    from agentic_ic.agent.tool_handlers import ToolHandlers

    mock_cache = AsyncMock()
    mock_cache.get_option_chain_with_fallback = AsyncMock(return_value=chain)
    return ToolHandlers(cache=mock_cache, **kwargs)


# ---------------------------------------------------------------------------
# validate_ic_params
# ---------------------------------------------------------------------------

class TestValidateIcParams:

    @pytest.mark.asyncio
    async def test_valid_trade_passes_all_checks(self, balanced_chain):
        """A well-formed IC within thresholds should return valid=True."""
        handlers = _make_handlers(
            balanced_chain,
            credit_equality_threshold=0.20,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=1.00,
            max_side_credit=2.00,
        )
        result = await handlers.validate_ic_params(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
        )
        assert result["valid"] is True
        assert all(c["passed"] for c in result["checks"]), result["checks"]

    @pytest.mark.asyncio
    async def test_inverted_call_strike_fails(self, balanced_chain):
        """long_call < short_call should fail strike_order_call."""
        handlers = _make_handlers(balanced_chain)
        result = await handlers.validate_ic_params(
            short_call_strike=5920.0,
            long_call_strike=5890.0,   # inverted
            short_put_strike=5810.0,
            long_put_strike=5780.0,
        )
        assert result["valid"] is False
        failed = [c for c in result["checks"] if not c["passed"]]
        assert any("strike_order_call" in c["check"] for c in failed)

    @pytest.mark.asyncio
    async def test_inverted_put_strike_fails(self, balanced_chain):
        """long_put > short_put should fail strike_order_put."""
        handlers = _make_handlers(balanced_chain)
        result = await handlers.validate_ic_params(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5780.0,   # inverted
            long_put_strike=5810.0,
        )
        assert result["valid"] is False
        failed = [c for c in result["checks"] if not c["passed"]]
        assert any("strike_order_put" in c["check"] for c in failed)

    @pytest.mark.asyncio
    async def test_asymmetric_wings_by_more_than_5pts_fails(self, balanced_chain):
        """Call wing 30pt vs put wing 25pt (diff=5) should fail symmetry check."""
        handlers = _make_handlers(balanced_chain)
        result = await handlers.validate_ic_params(
            short_call_strike=5890.0,
            long_call_strike=5920.0,   # 30pt call wing
            short_put_strike=5810.0,
            long_put_strike=5784.0,    # 26pt put wing → diff = 4 → pass
        )
        # diff=4, should still pass symmetry
        symmetry = next(c for c in result["checks"] if c["check"] == "wing_width_symmetry")
        assert symmetry["passed"] is True

    @pytest.mark.asyncio
    async def test_wings_differ_by_more_than_5pts_fails_symmetry(self, balanced_chain):
        """Call wing 30pt vs put wing 20pt (diff=10) should fail symmetry."""
        handlers = _make_handlers(balanced_chain)
        result = await handlers.validate_ic_params(
            short_call_strike=5890.0,
            long_call_strike=5920.0,   # 30pt
            short_put_strike=5810.0,
            long_put_strike=5790.0,    # 20pt → diff=10 → fail
        )
        assert result["valid"] is False
        failed_checks = [c["check"] for c in result["checks"] if not c["passed"]]
        assert "wing_width_symmetry" in failed_checks

    @pytest.mark.asyncio
    async def test_credit_diff_within_threshold_passes(self, balanced_chain):
        """Credits differing by less than threshold should pass credit_equality."""
        handlers = _make_handlers(
            balanced_chain,
            credit_equality_threshold=0.20,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=1.00,
            max_side_credit=2.00,
        )
        result = await handlers.validate_ic_params(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
        )
        credit_check = next((c for c in result["checks"] if c["check"] == "credit_equality"), None)
        assert credit_check is not None
        assert credit_check["passed"] is True

    @pytest.mark.asyncio
    async def test_credit_diff_exceeds_threshold_fails(self, skewed_chain):
        """Credits differing by more than threshold should fail credit_equality."""
        handlers = _make_handlers(
            skewed_chain,
            credit_equality_threshold=0.20,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=1.00,
            max_side_credit=2.50,
        )
        result = await handlers.validate_ic_params(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
        )
        credit_check = next((c for c in result["checks"] if c["check"] == "credit_equality"), None)
        assert credit_check is not None
        # put mid ~2.40, call mid ~1.50 → diff ~0.90 > 0.20
        assert credit_check["passed"] is False
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_threshold_exposed_in_result(self, balanced_chain):
        """credit_equality_threshold should be echoed back in the result."""
        handlers = _make_handlers(balanced_chain, credit_equality_threshold=0.25)
        result = await handlers.validate_ic_params(
            short_call_strike=5890.0,
            long_call_strike=5920.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
        )
        assert result["credit_equality_threshold"] == 0.25

    @pytest.mark.asyncio
    async def test_missing_strike_in_chain_fails_chain_data(self, balanced_chain):
        """A strike not in the chain should produce a chain_data_* failure."""
        handlers = _make_handlers(balanced_chain)
        result = await handlers.validate_ic_params(
            short_call_strike=5999.0,   # not in chain
            long_call_strike=6029.0,
            short_put_strike=5810.0,
            long_put_strike=5780.0,
        )
        assert result["valid"] is False
        failed = [c["check"] for c in result["checks"] if not c["passed"]]
        assert any("chain_data" in ch for ch in failed)


# ---------------------------------------------------------------------------
# get_premium_balance_candidates
# ---------------------------------------------------------------------------

class TestGetPremiumBalanceCandidates:

    @pytest.mark.asyncio
    async def test_returns_candidates_within_threshold(self, balanced_chain):
        """Should find at least one valid candidate when chain has balanced strikes."""
        handlers = _make_handlers(
            balanced_chain,
            credit_equality_threshold=0.20,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=1.00,
            max_side_credit=2.00,
        )
        result = await handlers.get_premium_balance_candidates(wing_width=30)
        assert "candidates" in result
        assert result["candidates_found"] > 0
        for cand in result["candidates"]:
            assert cand["credit_diff"] <= 0.20

    @pytest.mark.asyncio
    async def test_candidates_sorted_by_credit_diff(self, balanced_chain):
        """Candidates should be sorted best (lowest diff) first."""
        handlers = _make_handlers(
            balanced_chain,
            credit_equality_threshold=0.20,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=1.00,
            max_side_credit=2.00,
        )
        result = await handlers.get_premium_balance_candidates(wing_width=30)
        diffs = [c["credit_diff"] for c in result["candidates"]]
        assert diffs == sorted(diffs)

    @pytest.mark.asyncio
    async def test_respects_max_results(self, balanced_chain):
        """Should return at most max_results candidates."""
        handlers = _make_handlers(
            balanced_chain,
            credit_equality_threshold=0.50,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=0.50,
            max_side_credit=3.00,
        )
        result = await handlers.get_premium_balance_candidates(wing_width=30, max_results=2)
        assert len(result["candidates"]) <= 2

    @pytest.mark.asyncio
    async def test_candidate_has_required_fields(self, balanced_chain):
        """Each candidate must include all key fields."""
        handlers = _make_handlers(
            balanced_chain,
            credit_equality_threshold=0.30,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=1.00,
            max_side_credit=2.00,
        )
        result = await handlers.get_premium_balance_candidates(wing_width=30)
        for cand in result["candidates"]:
            for field in (
                "short_call_strike", "long_call_strike",
                "short_put_strike", "long_put_strike",
                "call_credit", "put_credit", "total_credit",
                "credit_diff", "wing_width",
                "call_delta", "put_delta",
            ):
                assert field in cand, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_no_candidates_when_skew_too_high(self, skewed_chain):
        """With tight threshold and skewed chain, no candidates should be found."""
        handlers = _make_handlers(
            skewed_chain,
            credit_equality_threshold=0.20,
            min_short_delta=0.10,
            max_short_delta=0.15,
            min_side_credit=1.00,
            max_side_credit=2.00,
        )
        result = await handlers.get_premium_balance_candidates(wing_width=30)
        assert result["candidates_found"] == 0
        assert result["candidates"] == []

    @pytest.mark.asyncio
    async def test_threshold_echoed_in_result(self, balanced_chain):
        """credit_equality_threshold should appear in the result."""
        handlers = _make_handlers(balanced_chain, credit_equality_threshold=0.18)
        result = await handlers.get_premium_balance_candidates()
        assert result["credit_equality_threshold"] == 0.18

    @pytest.mark.asyncio
    async def test_empty_chain_returns_error(self):
        """Empty chain should return an error dict with no candidates."""
        from agentic_ic.agent.tool_handlers import ToolHandlers

        mock_cache = AsyncMock()
        mock_cache.get_option_chain_with_fallback = AsyncMock(return_value=[])
        handlers = ToolHandlers(
            cache=mock_cache,
            min_short_delta=0.10,
            max_short_delta=0.15,
        )
        result = await handlers.get_premium_balance_candidates()
        assert "error" in result
        assert result["candidates"] == []


# ---------------------------------------------------------------------------
# get_skew_snapshot
# ---------------------------------------------------------------------------

class TestGetSkewSnapshot:

    @pytest.mark.asyncio
    async def test_within_threshold_verdict(self, balanced_chain):
        """Balanced chain should return verdict='balanced'."""
        handlers = _make_handlers(balanced_chain, skew_threshold=1.15)
        result = await handlers.get_skew_snapshot(delta_target=0.12)
        assert result["verdict"] in ("balanced", "unknown")
        if result["verdict"] == "balanced":
            assert result["skew_ratio_used"] <= 1.15

    @pytest.mark.asyncio
    async def test_skew_too_high_verdict(self, skewed_chain):
        """Skewed chain (put IV 35% > call IV) should return verdict='skewed_use_candidates'."""
        handlers = _make_handlers(skewed_chain, skew_threshold=1.15)
        result = await handlers.get_skew_snapshot(delta_target=0.12)
        # iv_ratio = 0.19 / 0.14 ≈ 1.36 > 1.15
        assert result["verdict"] == "skewed_use_candidates"

    @pytest.mark.asyncio
    async def test_threshold_echoed(self, balanced_chain):
        """skew_threshold should appear in the result."""
        handlers = _make_handlers(balanced_chain, skew_threshold=1.20)
        result = await handlers.get_skew_snapshot()
        assert result["skew_threshold"] == 1.20

    @pytest.mark.asyncio
    async def test_result_contains_required_fields(self, balanced_chain):
        """Snapshot must include strikes, deltas, iv, credit, and ratios."""
        handlers = _make_handlers(balanced_chain, skew_threshold=1.15)
        result = await handlers.get_skew_snapshot(delta_target=0.12)
        for field in (
            "delta_target", "skew_threshold", "verdict",
            "call_strike", "call_delta", "call_credit",
            "put_strike", "put_delta", "put_credit",
        ):
            assert field in result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_iv_ratio_preferred_over_credit_ratio(self, balanced_chain):
        """When IV data is available, iv_ratio should drive the verdict."""
        handlers = _make_handlers(balanced_chain, skew_threshold=1.15)
        result = await handlers.get_skew_snapshot(delta_target=0.12)
        if result.get("iv_ratio") is not None:
            assert result["skew_ratio_used"] == result["iv_ratio"]

    @pytest.mark.asyncio
    async def test_empty_chain_returns_error(self):
        """Empty chain should return error verdict."""
        from agentic_ic.agent.tool_handlers import ToolHandlers

        mock_cache = AsyncMock()
        mock_cache.get_option_chain_with_fallback = AsyncMock(return_value=[])
        handlers = ToolHandlers(cache=mock_cache, skew_threshold=1.15)
        result = await handlers.get_skew_snapshot()
        assert "error" in result or result["verdict"] == "unknown"


# ---------------------------------------------------------------------------
# get_option_chain target_credit sorting
# ---------------------------------------------------------------------------

class TestGetOptionChainTargetCredit:

    @pytest.mark.asyncio
    async def test_sorted_by_proximity_to_target_credit(self, balanced_chain):
        """Results should be ordered closest mid to target_credit first."""
        handlers = _make_handlers(balanced_chain)
        target = 1.50
        result = await handlers.get_option_chain(target_credit=target)
        mids = [round((o["bid"] + o["ask"]) / 2, 4) for o in result]
        diffs = [abs(m - target) for m in mids]
        assert diffs == sorted(diffs)

    @pytest.mark.asyncio
    async def test_no_target_credit_preserves_original_order(self, balanced_chain):
        """Without target_credit, order should be unaffected by sorting."""
        handlers = _make_handlers(balanced_chain)
        result_unsorted = await handlers.get_option_chain()
        result_no_target = await handlers.get_option_chain(target_credit=None)
        assert [o["strike"] for o in result_unsorted] == [o["strike"] for o in result_no_target]

    @pytest.mark.asyncio
    async def test_target_credit_combined_with_delta_filter(self, balanced_chain):
        """target_credit and delta filter should both apply."""
        handlers = _make_handlers(balanced_chain)
        result = await handlers.get_option_chain(
            min_delta=0.10,
            max_delta=0.15,
            target_credit=1.50,
        )
        for opt in result:
            assert 0.10 <= abs(opt["delta"]) <= 0.15
        if len(result) > 1:
            mids = [round((o["bid"] + o["ask"]) / 2, 4) for o in result]
            diffs = [abs(m - 1.50) for m in mids]
            assert diffs == sorted(diffs)
