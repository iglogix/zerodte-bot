"""Token-based authentication middleware for the web dashboard."""

import hashlib
import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse


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


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated requests to /login.

    Allows /login and /static paths without authentication.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow login page and static assets without auth
        if path == "/login" or path.startswith("/static"):
            return await call_next(request)

        # Check session cookie
        session_cookie = request.cookies.get("session", "")
        expected = _get_session_token(_get_token())

        if not hmac.compare_digest(session_cookie, expected):
            return RedirectResponse(url="/login", status_code=303)

        return await call_next(request)
