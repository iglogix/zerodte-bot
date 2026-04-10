import os
import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def _set_web_token(monkeypatch):
    monkeypatch.setenv("BOT_WEB_TOKEN", "test-secret-token")


@pytest.fixture()
def app():
    from web.app import create_app

    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def authed_client(app):
    c = TestClient(app)
    c.post("/login", data={"token": "test-secret-token"})
    return c
