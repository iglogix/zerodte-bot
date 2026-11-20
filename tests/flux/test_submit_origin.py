"""Order-submit routes refuse cross-site requests.

The Flux app binds to loopback, but a page open in the same browser can still
POST to it. A browser always sends Origin on a cross-site POST, so the submit
routes reject any Origin that is not the app's own host.
"""
import pytest

from zerodte.flux.app import create_app

SUBMIT_ROUTES = ["/api/dc_submit", "/api/transform_submit"]


@pytest.fixture
def client():
    return create_app().test_client()


@pytest.mark.parametrize("route", SUBMIT_ROUTES)
def test_cross_site_post_is_refused(client, route):
    r = client.post(route, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


@pytest.mark.parametrize("route", SUBMIT_ROUTES)
def test_same_origin_post_is_processed(client, route):
    r = client.post(route, headers={"Origin": "http://localhost"})
    assert r.status_code == 200


@pytest.mark.parametrize("route", SUBMIT_ROUTES)
def test_non_browser_post_without_origin_is_processed(client, route):
    assert client.post(route).status_code == 200
