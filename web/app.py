"""FastAPI application factory for the web dashboard."""

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.auth import AuthMiddleware, _check_token, _get_session_token, _get_token

_BASE_DIR = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    app = FastAPI(title="0DTE Bot Dashboard")

    # Static files
    app.mount("/static", StaticFiles(directory=_BASE_DIR / "static"), name="static")

    # Templates
    templates = Jinja2Templates(directory=_BASE_DIR / "templates")

    # Auth middleware
    app.add_middleware(AuthMiddleware)

    # --- Routes ---

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html")

    @app.post("/login")
    async def login_submit(request: Request, token: str = Form(...)):
        if not _check_token(token):
            return templates.TemplateResponse(
                request, "login.html", {"error": "Invalid token"}, status_code=401
            )
        session_value = _get_session_token(_get_token())
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="session", value=session_value, httponly=True, samesite="lax")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse(request, "dashboard.html")

    return app
