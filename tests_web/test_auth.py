"""Tests for web authentication."""


def test_unauthenticated_redirects_to_login(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_login_page_accessible_without_auth(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "login" in resp.text.lower()


def test_login_with_correct_token_sets_cookie(client):
    resp = client.post("/login", data={"token": "test-secret-token"}, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"
    assert "session" in resp.cookies


def test_login_with_wrong_token_returns_401(client):
    resp = client.post("/login", data={"token": "wrong-token"})
    assert resp.status_code == 401


def test_authenticated_request_succeeds(authed_client):
    resp = authed_client.get("/")
    assert resp.status_code == 200
