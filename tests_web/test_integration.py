"""Integration test: full app startup and page navigation."""


def test_full_navigation_flow(authed_client):
    """Visit every page and verify no 500 errors."""
    pages = ["/", "/bots", "/positions", "/decisions", "/logs", "/chat"]
    for page in pages:
        resp = authed_client.get(page)
        assert resp.status_code == 200, f"Page {page} returned {resp.status_code}"


def test_start_and_list_bot_flow(authed_client):
    """Start a bot via form, verify it appears on dashboard."""
    from unittest.mock import AsyncMock, patch, MagicMock
    from web.bot_manager import BotInfo
    from datetime import datetime, timezone

    mock_info = BotInfo(
        bot_id="test1", name="test_bot", config_path="config/config.yaml",
        mode="dry_run", status="running", pid=9999,
        started_at=datetime.now(timezone.utc),
    )

    with patch("web.app._bot_manager") as mock_mgr:
        mock_mgr.start_bot = AsyncMock(return_value="test1")
        mock_mgr.get_bot.return_value = mock_info
        mock_mgr.list_bots.return_value = [mock_info]
        mock_mgr.check_health.return_value = None
        mock_mgr.list_configs = MagicMock(return_value=["config/config.yaml"])

        # Start a bot
        resp = authed_client.post(
            "/bots/start",
            data={"config_path": "config/config.yaml", "mode": "dry_run"},
            follow_redirects=False,
        )
        assert resp.status_code == 303

        # Dashboard should show the bot
        resp = authed_client.get("/")
        assert resp.status_code == 200
        assert "test_bot" in resp.text
