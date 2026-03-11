"""
Guard against hardcoded SPX/SPXW symbols in the 8am options bot code.

The 8am bot trades ES futures options — all symbols must come from config
or the option chain, never hardcoded as 'SPX' or 'SPXW'.
"""

import ast
import os
from pathlib import Path

import pytest

SRC_8AM = Path(__file__).resolve().parent.parent / "src_8am"

# Strings that should NOT appear as hardcoded literals in src_8am/ code
FORBIDDEN_LITERALS = {"SPX", "SPXW", '"SPX"', "'SPX'"}


def _collect_python_files():
    """Yield all .py files under src_8am/."""
    for root, _, files in os.walk(SRC_8AM):
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def _find_hardcoded_strings(filepath: Path):
    """Return list of (lineno, value) for any forbidden string literals."""
    source = filepath.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if val in FORBIDDEN_LITERALS:
                hits.append((node.lineno, val))
    return hits


@pytest.mark.parametrize(
    "pyfile",
    list(_collect_python_files()),
    ids=lambda p: str(p.relative_to(SRC_8AM)),
)
def test_no_hardcoded_spx_symbols(pyfile):
    """No src_8am/ file should contain hardcoded 'SPX' or 'SPXW' string literals."""
    hits = _find_hardcoded_strings(pyfile)
    if hits:
        details = "\n".join(f"  line {ln}: {val!r}" for ln, val in hits)
        pytest.fail(
            f"Hardcoded symbol(s) in {pyfile.relative_to(SRC_8AM)}:\n{details}\n"
            "Use config values or chain-provided symbols instead."
        )


class TestOrbOptionsServiceConfig:
    """Verify OrbOptionsService reads symbols from config."""

    def _make_service(self, **overrides):
        from src_8am.services.orb_options_service import OrbOptionsService

        config = {
            "bot": {"symbol": "ES"},
            "options": {"option_root": "EW", "spread_width_pts": 5},
        }
        config.update(overrides)
        return OrbOptionsService(tt_client=None, config=config)

    def test_product_code_from_config(self):
        svc = self._make_service(bot={"symbol": "NQ"})
        assert svc._product_code == "NQ"

    def test_product_code_default(self):
        svc = self._make_service(bot={})
        assert svc._product_code == "ES"

    def test_build_symbol_uses_config_root(self):
        svc = self._make_service(options={"option_root": "NQEW", "spread_width_pts": 5})
        sym = svc._build_symbol("C", 5800.0)
        assert sym.startswith("NQEW")
        assert "C" in sym
        assert "00005800" in sym

    def test_build_symbol_default_root(self):
        svc = self._make_service()
        sym = svc._build_symbol("P", 5700.0)
        assert sym.startswith("EW")


class TestTastyTradeClientLeg:
    """Verify _build_leg supports configurable instrument type."""

    def test_default_instrument_type(self):
        from src.services.tastytrade_client import TastyTradeClient

        leg = TastyTradeClient._build_leg("SPXW260311C05800", "Sell to Open", 1)
        assert leg["instrument-type"] == "Equity Option"

    def test_future_option_instrument_type(self):
        from src.services.tastytrade_client import TastyTradeClient

        leg = TastyTradeClient._build_leg(
            "./EW1H26C5800:XCME", "Sell to Open", 1,
            instrument_type="Future Option",
        )
        assert leg["instrument-type"] == "Future Option"
        assert leg["symbol"] == "./EW1H26C5800:XCME"


class TestOrderPlacementLogging:
    """Verify successful order placement is logged at INFO level."""

    @pytest.mark.asyncio
    async def test_iron_condor_success_logs_info(self):
        import logging
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.services.tastytrade_client import TastyTradeClient

        tt = TastyTradeClient.__new__(TastyTradeClient)
        tt.mode = "sandbox"
        tt.base_url = "https://api.cert.tastyworks.com"
        tt.account_number = "TEST123"
        tt._sdk_session = None
        tt._market_sdk_session = None
        tt._access_token = "fake-token"

        mock_resp = AsyncMock()
        mock_resp.status = 201
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value={
            "data": {"order": {"id": "IC-42"}}
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        tt._http = MagicMock()
        tt._http.post = MagicMock(return_value=mock_resp)

        order = {
            "order_type": "iron_condor",
            "short_call_strike": 5900,
            "long_call_strike": 5930,
            "short_put_strike": 5800,
            "long_put_strike": 5770,
            "quantity": 1,
            "credit": 2.50,
        }

        with patch("src.services.tastytrade_client.logger") as mock_logger:
            result = await tt._place_iron_condor(order)
            mock_logger.info.assert_any_call(
                "Iron condor order placed successfully: order_id=%s", "IC-42"
            )
        assert result["order_id"] == "IC-42"

    @pytest.mark.asyncio
    async def test_credit_spread_success_logs_info(self):
        import logging
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.services.tastytrade_client import TastyTradeClient

        tt = TastyTradeClient.__new__(TastyTradeClient)
        tt.mode = "sandbox"
        tt.base_url = "https://api.cert.tastyworks.com"
        tt.account_number = "TEST123"
        tt._sdk_session = None
        tt._market_sdk_session = None
        tt._access_token = "fake-token"

        mock_resp = AsyncMock()
        mock_resp.status = 201
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value={
            "data": {"order": {"id": "CS-99"}}
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        tt._http = MagicMock()
        tt._http.post = MagicMock(return_value=mock_resp)

        order = {
            "order_type": "call_credit_spread",
            "short_symbol": "./EW1H26C5900:XCME",
            "long_symbol": "./EW1H26C5930:XCME",
            "instrument_type": "Future Option",
            "quantity": 1,
            "credit": 1.20,
        }

        with patch("src.services.tastytrade_client.logger") as mock_logger:
            result = await tt._place_credit_spread(order)
            mock_logger.info.assert_any_call(
                "Credit spread order placed successfully: order_id=%s", "CS-99"
            )
        assert result["order_id"] == "CS-99"
