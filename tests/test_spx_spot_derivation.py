"""
Tests for SPX spot price derivation from put-call parity and the
get_spread_candidates flow that uses derived spot instead of ES price.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src_8am.services.orb_options_service import OrbOptionsService
from src_8am.agent.options_tool_handlers import OrbOptionsToolHandlers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_option(strike: float, option_type: str, bid: float, ask: float) -> dict:
    """Build a minimal option chain record."""
    return {
        "strike": strike,
        "option_type": option_type,
        "bid": bid,
        "ask": ask,
        "symbol": f"SPXW{option_type[0].upper()}{int(strike)}",
        "delta": 0.0,
    }


def _make_chain_around(spot: float, n_strikes: int = 5, step: float = 5.0) -> list:
    """
    Build a synthetic 0DTE chain centered near *spot*.
    Puts get more expensive below spot; calls get more expensive above.
    Put-call parity holds: C - P = spot - K at each strike.
    """
    import math
    atm_k = round(spot / step) * step
    chain = []
    for i in range(-n_strikes, n_strikes + 1):
        k = atm_k + i * step
        # Intrinsic + small time-value so bid > 0
        c_mid = max(spot - k, 0) + 2.0
        p_mid = max(k - spot, 0) + 2.0
        # Verify parity: C - P = spot - K
        # c_mid - p_mid = (spot - k + 2) - (k - spot + 2) = 2*(spot-k) ... only when both OTM
        # Simpler: just force parity exactly
        # Set call = (spot - k) + tv, put = (k - spot) + tv → C - P = spot - k ✓ when tv equal
        tv = 3.0  # time value
        c_mid = max(spot - k, 0) + tv
        p_mid = max(k - spot, 0) + tv
        spread = 0.20
        chain.append(_make_option(k, "call", round(c_mid - spread / 2, 2), round(c_mid + spread / 2, 2)))
        chain.append(_make_option(k, "put", round(p_mid - spread / 2, 2), round(p_mid + spread / 2, 2)))
    return chain


def _build_service(option_root: str = "SPXW", **overrides) -> OrbOptionsService:
    """Build an OrbOptionsService with sensible defaults for SPX."""
    config = {
        "bot": {"symbol": "ES"},
        "options": {
            "option_root": option_root,
            "spread_width_pts": 5,
            "strike_step_pts": 5,
            "target_short_delta": 0.16,
            "max_short_delta": 0.30,
            "min_credit_per_spread": 0.05,
            "tp_pct_of_credit": 0.50,
            "sl_mult_of_credit": 0,
            "use_straddle_em": True,
            "spread_mode": "iron_condor",
            "em_with_trend": 0.50,
            "em_against_trend": 0.75,
            "width_with_trend": 5,
            "width_against_trend": 10,
            "atm_filter_width_pts": 150,
            "contract_multiplier": 100,
            **overrides,
        },
    }
    tt = MagicMock()
    return OrbOptionsService(tt, config)


def _build_tool_handlers(
    option_root: str = "SPXW",
    es_price: float = 6613.0,
    chain: list | None = None,
    derive_spot: float | None = 6575.0,
    api_quote: dict | None = None,
    api_quote_error: Exception | None = None,
    spread_mode: str = "iron_condor",
    use_em: bool = True,
) -> OrbOptionsToolHandlers:
    """Build OrbOptionsToolHandlers with mocked dependencies."""
    config = {
        "bot": {"symbol": "ES", "quantity": 1},
        "options": {
            "option_root": option_root,
            "index_quote_symbol": "SPX" if option_root in ("SPXW", "SPX") else "",
            "spread_mode": spread_mode,
            "use_straddle_em": use_em,
        },
        "strategy": {},
    }

    md = AsyncMock()
    md.get_current_price = AsyncMock(return_value=es_price)

    opts = MagicMock()
    opts._opt_cfg = config["options"]
    opts._use_em = use_em
    opts._spread_mode = spread_mode
    opts.get_0dte_chain = AsyncMock(return_value=chain or [{"stub": True}])
    opts.derive_spot_from_chain = MagicMock(return_value=derive_spot)
    opts.select_em_spread = MagicMock(return_value={
        "spread_type": "put_credit_spread",
        "short_strike": 6550.0,
        "long_strike": 6545.0,
        "credit": 1.50,
    })

    tt = AsyncMock()
    if api_quote_error:
        tt.get_quote = AsyncMock(side_effect=api_quote_error)
    else:
        tt.get_quote = AsyncMock(return_value=api_quote if api_quote is not None else {"price": 6575.0})

    db = AsyncMock()

    handlers = OrbOptionsToolHandlers(md, opts, tt, db, config, mode="manual")
    handlers._orb_levels = {"high": 6640, "low": 6600, "midline": 6620}
    handlers._breakout_bias = "long"
    return handlers


# ===========================================================================
# derive_spot_from_chain tests
# ===========================================================================


class TestDeriveSpotFromChain:
    """Tests for OrbOptionsService.derive_spot_from_chain."""

    def test_derives_correct_spot(self):
        """Put-call parity at ATM strike gives correct SPX spot."""
        svc = _build_service()
        spx_spot = 6575.0
        es_price = 6613.0
        chain = _make_chain_around(spx_spot, n_strikes=5, step=5.0)

        result = svc.derive_spot_from_chain(chain, es_price)

        assert result is not None
        assert abs(result - spx_spot) < 1.0  # within rounding

    def test_returns_none_for_empty_chain(self):
        """Returns None when chain is empty."""
        svc = _build_service()
        assert svc.derive_spot_from_chain([], 6613.0) is None

    def test_returns_none_for_puts_only(self):
        """Returns None when chain has only puts (no call pairing)."""
        svc = _build_service()
        chain = [_make_option(6575, "put", 3.0, 3.20)]
        assert svc.derive_spot_from_chain(chain, 6613.0) is None

    def test_returns_none_for_calls_only(self):
        """Returns None when chain has only calls (no put pairing)."""
        svc = _build_service()
        chain = [_make_option(6575, "call", 3.0, 3.20)]
        assert svc.derive_spot_from_chain(chain, 6613.0) is None

    def test_skips_zero_bid_options(self):
        """Options with bid=0 are excluded from parity calculation."""
        svc = _build_service()
        chain = [
            _make_option(6575, "put", 0.0, 1.0),   # bid=0, skipped
            _make_option(6575, "call", 3.0, 3.20),
        ]
        assert svc.derive_spot_from_chain(chain, 6613.0) is None

    def test_picks_strike_nearest_es_price(self):
        """ATM strike is selected as closest to ES price, not SPX spot."""
        svc = _build_service()
        # Chain with two strikes: 6575 (near SPX) and 6610 (near ES)
        # The one near ES should be picked as ATM for parity calc
        spx_actual = 6575.0
        chain = [
            # Strike 6575 — far from ES 6613
            _make_option(6575, "call", 3.10, 3.30),  # mid=3.20
            _make_option(6575, "put", 3.00, 3.20),   # mid=3.10
            # Strike 6610 — close to ES 6613
            _make_option(6610, "call", 5.00, 5.20),   # mid=5.10
            _make_option(6610, "put", 2.00, 2.20),    # mid=2.10
        ]
        result = svc.derive_spot_from_chain(chain, es_price=6613.0)
        # At K=6610: spot = 6610 + 5.10 - 2.10 = 6613.0
        assert result is not None
        assert abs(result - 6613.0) < 0.1

    def test_basis_gap_detected(self):
        """Derived SPX spot differs from ES price by the basis gap."""
        svc = _build_service()
        spx_spot = 6575.0
        es_price = 6613.0
        chain = _make_chain_around(spx_spot)

        result = svc.derive_spot_from_chain(chain, es_price)

        assert result is not None
        basis = es_price - result
        assert basis > 30  # should detect the ~38pt gap

    def test_single_paired_strike(self):
        """Works with just one strike that has both put and call."""
        svc = _build_service()
        chain = [
            _make_option(6580, "call", 8.00, 8.40),  # mid=8.20
            _make_option(6580, "put", 2.80, 3.20),   # mid=3.00
        ]
        result = svc.derive_spot_from_chain(chain, es_price=6613.0)
        # spot = 6580 + 8.20 - 3.00 = 6585.20
        assert result == 6585.2


# ===========================================================================
# get_spread_candidates — SPX spot derivation flow
# ===========================================================================


class TestGetSpreadCandidatesSPXSpot:
    """Tests for the SPX spot derivation path in get_spread_candidates."""

    @pytest.mark.asyncio
    async def test_uses_derived_spot_for_spx(self):
        """When option_root=SPXW, derived spot is passed to select_em_spread."""
        h = _build_tool_handlers(option_root="SPXW", derive_spot=6575.0)
        result = await h.get_spread_candidates()

        assert "error" not in result
        # select_em_spread should have been called with spot=6575, not es_price=6613
        calls = h._opts.select_em_spread.call_args_list
        for call in calls:
            assert call.args[1] == 6575.0  # spot arg

    @pytest.mark.asyncio
    async def test_uses_es_price_for_futures_options(self):
        """When option_root=EW (ES futures), es_price is used directly."""
        h = _build_tool_handlers(option_root="EW", es_price=6613.0)
        result = await h.get_spread_candidates()

        assert "error" not in result
        calls = h._opts.select_em_spread.call_args_list
        for call in calls:
            assert call.args[1] == 6613.0

    @pytest.mark.asyncio
    async def test_falls_back_to_api_quote_when_parity_fails(self):
        """When derive_spot returns None, fetches SPX quote from API."""
        h = _build_tool_handlers(
            derive_spot=None,
            api_quote={"price": 6572.0},
        )
        result = await h.get_spread_candidates()

        assert "error" not in result
        h._tt.get_quote.assert_called_once_with("SPX")
        calls = h._opts.select_em_spread.call_args_list
        for call in calls:
            assert call.args[1] == 6572.0

    @pytest.mark.asyncio
    async def test_errors_when_both_parity_and_api_fail(self):
        """Returns error when neither parity nor API can provide SPX spot."""
        h = _build_tool_handlers(
            derive_spot=None,
            api_quote_error=Exception("connection timeout"),
        )
        result = await h.get_spread_candidates()

        assert result["error"] == "cannot_determine_index_spot"
        assert "connection timeout" in result["detail"]

    @pytest.mark.asyncio
    async def test_errors_when_api_returns_empty_quote(self):
        """Returns error when API quote has no usable price fields."""
        h = _build_tool_handlers(
            derive_spot=None,
            api_quote={},  # no price/mark/last/bid
        )
        result = await h.get_spread_candidates()

        assert result["error"] == "cannot_determine_index_spot"
        assert "both failed" in result["detail"]

    @pytest.mark.asyncio
    async def test_underlying_price_in_result_uses_spot(self):
        """The underlying_price in results reflects SPX spot, not ES."""
        h = _build_tool_handlers(derive_spot=6575.0)
        result = await h.get_spread_candidates()

        assert result["underlying_price"] == 6575.0

    @pytest.mark.asyncio
    async def test_api_quote_mark_field(self):
        """API fallback works with 'mark' field."""
        h = _build_tool_handlers(
            derive_spot=None,
            api_quote={"mark": 6574.5},
        )
        result = await h.get_spread_candidates()

        assert "error" not in result
        calls = h._opts.select_em_spread.call_args_list
        for call in calls:
            assert call.args[1] == 6574.5

    @pytest.mark.asyncio
    async def test_api_quote_last_field(self):
        """API fallback works with 'last' field when price/mark absent."""
        h = _build_tool_handlers(
            derive_spot=None,
            api_quote={"last": 6573.0},
        )
        result = await h.get_spread_candidates()

        assert "error" not in result
        calls = h._opts.select_em_spread.call_args_list
        for call in calls:
            assert call.args[1] == 6573.0

    @pytest.mark.asyncio
    async def test_no_api_call_when_parity_succeeds(self):
        """When put-call parity works, no API quote is fetched."""
        h = _build_tool_handlers(derive_spot=6575.0)
        await h.get_spread_candidates()

        h._tt.get_quote.assert_not_called()

    @pytest.mark.asyncio
    async def test_directional_mode_uses_spot(self):
        """Directional (non-IC) mode also uses derived spot for SPX."""
        h = _build_tool_handlers(
            derive_spot=6575.0,
            spread_mode="directional",
        )
        result = await h.get_spread_candidates()

        assert "error" not in result
        assert result["underlying_price"] == 6575.0
        call = h._opts.select_em_spread.call_args
        assert call.args[1] == 6575.0
