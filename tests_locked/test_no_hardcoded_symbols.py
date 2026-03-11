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
