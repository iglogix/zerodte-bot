"""Tests for web dashboard routes."""

from unittest.mock import AsyncMock, patch


def test_dashboard_page(authed_client):
    resp = authed_client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


def test_bots_page(authed_client):
    resp = authed_client.get("/bots")
    assert resp.status_code == 200
    assert "Bots" in resp.text


def test_positions_page(authed_client):
    resp = authed_client.get("/positions")
    assert resp.status_code == 200
    assert "Positions" in resp.text


def test_decisions_page(authed_client):
    resp = authed_client.get("/decisions")
    assert resp.status_code == 200


def test_logs_page(authed_client):
    resp = authed_client.get("/logs")
    assert resp.status_code == 200


def test_chat_page(authed_client):
    resp = authed_client.get("/chat")
    assert resp.status_code == 200


def test_start_bot_endpoint(authed_client):
    with patch("web.app._bot_manager") as mock_mgr:
        mock_mgr.start_bot = AsyncMock(return_value="abc123")
        resp = authed_client.post(
            "/bots/start",
            data={"config_path": "config/config.yaml", "mode": "dry_run", "name": "test"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/bots"


def test_stop_bot_endpoint(authed_client):
    with patch("web.app._bot_manager") as mock_mgr:
        mock_mgr.stop_bot = AsyncMock(return_value=True)
        resp = authed_client.post("/bots/stop/abc123", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/bots"
