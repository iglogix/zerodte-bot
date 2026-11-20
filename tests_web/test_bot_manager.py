import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from web.bot_manager import BotInfo, BotManager


@pytest.fixture
def manager():
    return BotManager()


def test_list_bots_empty(manager):
    assert manager.list_bots() == []


def test_list_configs(manager, tmp_path):
    (tmp_path / "config_iron_fly_spx.yaml").write_text("a: 1")
    (tmp_path / "config_orb_em_ic.yaml").write_text("b: 2")
    (tmp_path / "notes.txt").write_text("not a config")
    configs = BotManager.list_configs(str(tmp_path))
    assert len(configs) == 2
    assert all(c.endswith(".yaml") for c in configs)


def test_start_bot_creates_subprocess(manager, tmp_path):
    cfg = tmp_path / "config_test.yaml"
    cfg.write_text("a: 1")

    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.returncode = None

    async def _run():
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            bot_id = await manager.start_bot(str(cfg), mode="dry_run")
            assert bot_id is not None
            bots = manager.list_bots()
            assert len(bots) == 1
            assert bots[0].status == "running"
            assert bots[0].pid == 12345
            mock_exec.assert_called_once()

    asyncio.run(_run())


def test_stop_bot(manager, tmp_path):
    cfg = tmp_path / "config_test.yaml"
    cfg.write_text("a: 1")

    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.returncode = None
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    async def _run():
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            bot_id = await manager.start_bot(str(cfg), mode="dry_run")

        mock_proc.returncode = 0
        result = await manager.stop_bot(bot_id)
        assert result is True
        bots = manager.list_bots()
        assert bots[0].status == "stopped"

    asyncio.run(_run())


def test_stop_nonexistent_bot(manager):
    async def _run():
        result = await manager.stop_bot("nonexistent")
        assert result is False

    asyncio.run(_run())


def test_bot_name_auto_generated(manager):
    assert BotManager._derive_name("config/config_iron_fly_spx.yaml") == "iron_fly_spx"


def test_bot_name_fallback(manager):
    assert BotManager._derive_name("config/config.yaml") == "config"


# ---------------------------------------------------------------------------
# Input validation: start_bot builds a command line from caller-supplied values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["live --config=/etc/passwd", "help", "LIVE", "", "config=x"])
def test_start_bot_rejects_modes_outside_the_allow_list(manager, tmp_path, mode):
    cfg = tmp_path / "config_test.yaml"
    cfg.write_text("a: 1")

    async def _run():
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            with pytest.raises(ValueError, match="mode"):
                await manager.start_bot(str(cfg), mode=mode)
            mock_exec.assert_not_called()

    asyncio.run(_run())


@pytest.mark.parametrize("mode, flag", [("manual", "--manual"), ("sandbox", "--sandbox"),
                                        ("alpaca", "--alpaca"), ("live", "--live")])
def test_start_bot_maps_each_allowed_mode_to_its_flag(manager, tmp_path, mode, flag):
    cfg = tmp_path / "config_test.yaml"
    cfg.write_text("a: 1")
    mock_proc = AsyncMock()
    mock_proc.pid = 1
    mock_proc.returncode = None

    async def _run():
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await manager.start_bot(str(cfg), mode=mode)
            assert flag in mock_exec.call_args.args

    asyncio.run(_run())


@pytest.mark.parametrize("name", ["missing.yaml", "notes.txt"])
def test_start_bot_rejects_paths_that_are_not_existing_yaml_files(manager, tmp_path, name):
    (tmp_path / "notes.txt").write_text("not a config")

    async def _run():
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            with pytest.raises(ValueError, match="config"):
                await manager.start_bot(str(tmp_path / name), mode="dry_run")
            mock_exec.assert_not_called()

    asyncio.run(_run())


def test_dashboard_binds_to_loopback_by_default(monkeypatch):
    import web.__main__ as entry

    seen = {}
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr("sys.argv", ["web"])
    entry.main()
    assert seen["host"] == "127.0.0.1"
