"""Token-based authentication middleware for the web dashboard."""

import hashlib
import hmac
import os
from http.cookies import SimpleCookie

from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send


def _get_token() -> str:
    """Read the web token from the BOT_WEB_TOKEN environment variable."""
    token = os.environ.get("BOT_WEB_TOKEN", "")
    if not token:
        raise RuntimeError("BOT_WEB_TOKEN environment variable is not set")
    return token


def _check_token(provided: str) -> bool:
    """Constant-time comparison of the provided token against the expected one."""
    expected = _get_token()
    return hmac.compare_digest(provided, expected)


def _get_session_token(token: str) -> str:
    """Derive a session cookie value from the token (SHA-256, first 32 chars)."""
    return hashlib.sha256(token.encode()).hexdigest()[:32]


def _get_cookie(scope: Scope, name: str) -> str:
    """Extract a cookie value from ASGI scope headers."""
    for header_name, header_value in scope.get("headers", []):
        if header_name == b"cookie":
            cookies = SimpleCookie(header_value.decode("latin-1"))
            if name in cookies:
                return cookies[name].value
    return ""


class AuthMiddleware:
    """Pure ASGI middleware — redirects unauthenticated HTTP requests to /login.

    Passes WebSocket connections through unconditionally (WS does its own auth).
    BaseHTTPMiddleware breaks WebSocket in Starlette 1.0, so we use raw ASGI.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only intercept HTTP requests — let WebSocket and lifespan through
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Allow public paths
        if path == "/login" or path.startswith("/static"):
            await self.app(scope, receive, send)
            return

        # Check session cookie
        session_cookie = _get_cookie(scope, "session")
        try:
            expected = _get_session_token(_get_token())
        except RuntimeError:
            expected = ""

        if hmac.compare_digest(session_cookie, expected):
            await self.app(scope, receive, send)
            return

        # Redirect to login
        response = RedirectResponse(url="/login", status_code=303)
        await response(scope, receive, send)
