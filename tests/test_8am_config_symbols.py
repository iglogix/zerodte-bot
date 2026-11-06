"""Symbols come from config, never from hardcoded defaults (see tests_locked guard)."""
import asyncio
import glob
from datetime import date
from pathlib import Path

import pytest
import yaml

from es_8am.config_keys import require
from es_8am.services.orb_options_service import OrbOptionsService

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted(glob.glob(str(ROOT / "config" / "config_8am*.yaml")))


def test_require_names_the_missing_key():
    with pytest.raises(KeyError) as exc:
        require({"options": {}}, "options.option_root")
    assert "options.option_root" in str(exc.value)


def test_require_returns_value():
    assert require({"a": {"b": 3}}, "a.b") == 3


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_configs_found():
    assert len(CONFIGS) == 10


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: Path(p).name)
def test_every_8am_config_declares_symbols(path):
    cfg = _load(path)
    if "options" not in cfg:
        pytest.skip("futures-only config (no options section)")
    require(cfg, "bot.symbol")
    require(cfg, "options.option_root")
    chain_type = require(cfg, "options.chain_type")
    assert chain_type in ("index", "futures")
    if chain_type == "index":
        require(cfg, "options.chain_symbol")


class _StubClient:
    def __init__(self):
        self.calls = []

    async def get_option_chain(self, symbol):
        self.calls.append(("index", symbol))
        return []

    async def get_future_option_chain(self, product_code, underlying_price=None):
        self.calls.append(("futures", product_code))
        return []


def _chain_calls(bot, options):
    stub = _StubClient()
    svc = OrbOptionsService(stub, {"bot": bot, "options": options})
    asyncio.run(svc.get_0dte_chain(5000.0))
    return stub.calls


def test_index_roots_fetch_the_configured_chain():
    calls = _chain_calls(
        {"symbol": "ES"},
        {"option_root": "SPXW", "chain_symbol": "SPX", "chain_type": "index"},
    )
    assert calls == [("index", "SPX")]


def test_futures_chain_type_takes_the_futures_path():
    calls = _chain_calls(
        {"symbol": "ES"},
        {"option_root": "EW", "chain_type": "futures"},
    )
    assert calls == [("futures", "ES")]


def test_missing_chain_type_is_loud_not_swallowed():
    stub = _StubClient()
    svc = OrbOptionsService(stub, {"bot": {"symbol": "ES"}, "options": {"option_root": "EW"}})
    with pytest.raises(KeyError, match="options.chain_type"):
        asyncio.run(svc.get_0dte_chain(5000.0))
