"""FastAPI application factory for the web dashboard."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)

from web.auth import AuthMiddleware, _check_token, _get_session_token, _get_token
from web.bot_manager import BotManager
from web.chat_agent import ChatAgent
from web.log_streamer import LogFilter, LogTailer, tail_lines
from web.ws_hub import WSHub

_BASE_DIR = Path(__file__).resolve().parent

# Module-level singletons (re-initialised in create_app for testing)
_bot_manager = BotManager()
_ws_hub = WSHub()
_log_filter = LogFilter()
_log_tailers: dict[str, LogTailer] = {}
_chat_agent: Optional[ChatAgent] = None


async def _start_log_tailer(bot_id: str, log_file: str, bot_name: str) -> None:
    """Start tailing a bot's log file and broadcast filtered lines via WSHub."""
    tailer = LogTailer(log_file)
    _log_tailers[bot_id] = tailer

    async def on_line(line: str, level: str) -> None:
        await _ws_hub.broadcast("log", {
            "bot_id": bot_id,
            "bot_name": bot_name,
            "line": line,
            "level": level,
        })

    asyncio.create_task(tailer.tail(on_line))


def _stop_log_tailer(bot_id: str) -> None:
    tailer = _log_tailers.pop(bot_id, None)
    if tailer:
        tailer.stop()


def create_app() -> FastAPI:
    global _bot_manager, _ws_hub, _log_filter, _log_tailers, _chat_agent
    _bot_manager = BotManager()
    _ws_hub = WSHub()
    _log_filter = LogFilter()
    _log_tailers = {}
    _chat_agent = ChatAgent(bot_manager=_bot_manager)

    app = FastAPI(title="0DTE Bot Dashboard")

    # Static files
    app.mount("/static", StaticFiles(directory=_BASE_DIR / "static"), name="static")

    # Templates
    templates = Jinja2Templates(directory=_BASE_DIR / "templates")
    templates.env.filters["basename"] = lambda p: Path(p).name

    # Auth middleware
    app.add_middleware(AuthMiddleware)

    # Live P&L cockpit (routes: /cockpit, /api/positions, /api/quotes, /api/pnl_history)
    from web.cockpit import build_router as build_cockpit_router
    app.include_router(build_cockpit_router())

    # ------------------------------------------------------------------ #
    #  Auth routes
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Dashboard
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        _bot_manager.check_health()
        bots = _bot_manager.list_bots()
        return templates.TemplateResponse(request, "dashboard.html", {"bots": bots})

    # ------------------------------------------------------------------ #
    #  Bots
    # ------------------------------------------------------------------ #

    @app.get("/bots", response_class=HTMLResponse)
    async def bots_page(request: Request):
        _bot_manager.check_health()
        bots = _bot_manager.list_bots()
        configs = BotManager.list_configs()
        return templates.TemplateResponse(
            request, "bots.html", {"bots": bots, "configs": configs}
        )

    @app.post("/bots/start")
    async def start_bot(
        config_path: str = Form(...),
        mode: str = Form("dry_run"),
        name: Optional[str] = Form(None),
    ):
        bot_id = await _bot_manager.start_bot(config_path, mode=mode, name=name or None)
        bot = _bot_manager.get_bot(bot_id)
        if bot and bot.log_file:
            await _start_log_tailer(bot_id, bot.log_file, bot.name)
        return RedirectResponse(url="/bots", status_code=303)

    @app.post("/bots/stop/{bot_id}")
    async def stop_bot(bot_id: str):
        _stop_log_tailer(bot_id)
        await _bot_manager.stop_bot(bot_id)
        return RedirectResponse(url="/bots", status_code=303)

    @app.post("/bots/restart/{bot_id}")
    async def restart_bot(bot_id: str):
        await _bot_manager.restart_bot(bot_id)
        return RedirectResponse(url="/bots", status_code=303)

    # ------------------------------------------------------------------ #
    #  Positions
    # ------------------------------------------------------------------ #

    @app.get("/positions", response_class=HTMLResponse)
    async def positions_page(request: Request):
        return templates.TemplateResponse(request, "positions.html")

    # ------------------------------------------------------------------ #
    #  Decisions
    # ------------------------------------------------------------------ #

    @app.get("/decisions", response_class=HTMLResponse)
    async def decisions_page(request: Request, d: Optional[str] = None):
        date_str = d or date.today().isoformat()
        decisions_dir = Path("logs/decisions") / date_str
        files: list[str] = []
        if decisions_dir.is_dir():
            files = sorted(
                [f.name for f in decisions_dir.glob("*.json")], reverse=True
            )
        return templates.TemplateResponse(
            request,
            "decisions.html",
            {"date_str": date_str, "files": files, "selected": None, "detail": None},
        )

    @app.get("/decisions/{date_str}/{filename}", response_class=HTMLResponse)
    async def decision_detail(request: Request, date_str: str, filename: str):
        decisions_dir = Path("logs/decisions") / date_str
        files: list[str] = []
        if decisions_dir.is_dir():
            files = sorted(
                [f.name for f in decisions_dir.glob("*.json")], reverse=True
            )

        detail = None
        file_path = decisions_dir / filename
        if file_path.is_file():
            try:
                detail = json.loads(file_path.read_text())
            except (json.JSONDecodeError, OSError):
                detail = {"error": "Failed to read decision file"}

        return templates.TemplateResponse(
            request,
            "decisions.html",
            {
                "date_str": date_str,
                "files": files,
                "selected": filename,
                "detail": json.dumps(detail, indent=2, default=str) if detail else None,
            },
        )

    # ------------------------------------------------------------------ #
    #  Logs
    # ------------------------------------------------------------------ #

    @app.get("/logs", response_class=HTMLResponse)
    async def logs_page(request: Request):
        bots = _bot_manager.list_bots()
        return templates.TemplateResponse(request, "logs.html", {"bots": bots})

    # ------------------------------------------------------------------ #
    #  Chat
    # ------------------------------------------------------------------ #

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page(request: Request):
        return templates.TemplateResponse(request, "chat.html")

    # ------------------------------------------------------------------ #
    #  WebSocket
    # ------------------------------------------------------------------ #

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        # Verify auth via session cookie
        session_cookie = ws.cookies.get("session", "")
        try:
            expected = _get_session_token(_get_token())
        except RuntimeError:
            await ws.close(code=1008)
            return

        if not hmac.compare_digest(session_cookie, expected):
            await ws.close(code=1008)
            return

        await ws.accept()
        await _ws_hub.connect(ws)

        # Backfill: send recent log lines from all running bots
        # so the client doesn't miss logs from fast-exiting bots
        _bot_manager.check_health()
        _lf = LogFilter()
        for bot in _bot_manager.list_bots():
            if bot.log_file:
                for line in tail_lines(bot.log_file, n=100):
                    if _lf.accept(line):
                        try:
                            await ws.send_json({"type": "log", "data": {
                                "bot_id": bot.bot_id,
                                "bot_name": bot.name,
                                "line": line,
                                "level": _lf.classify(line),
                            }})
                        except Exception:
                            break

        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "chat_message":
                    user_msg = data.get("message", "")
                    try:
                        response = await _chat_agent.chat(user_msg)
                        await _ws_hub.send_to(
                            ws, "chat_response", {"type": "text", "content": response}
                        )
                    except Exception as exc:
                        await _ws_hub.send_to(
                            ws, "chat_response",
                            {"type": "text", "content": f"Error: {exc}"},
                        )

                elif msg_type == "chat_confirm":
                    tool_name = data.get("tool", "")
                    args = data.get("args", {})
                    confirmed = data.get("confirmed", False)
                    try:
                        result = await _chat_agent.execute_tool(
                            tool_name, args, confirmed=confirmed
                        )
                        await _ws_hub.send_to(
                            ws, "chat_response",
                            {"type": "text", "content": json.dumps(result, default=str)},
                        )
                    except Exception as exc:
                        await _ws_hub.send_to(
                            ws, "chat_response",
                            {"type": "text", "content": f"Error: {exc}"},
                        )
        except WebSocketDisconnect:
            _ws_hub.disconnect(ws)
        except Exception:
            _ws_hub.disconnect(ws)

    @app.on_event("startup")
    async def on_startup():
        await _bot_manager.recover()
        # Start log tailers for recovered bots
        for bot in _bot_manager.list_bots():
            if bot.status == "running" and bot.log_file:
                await _start_log_tailer(bot.bot_id, bot.log_file, bot.name)

    @app.on_event("shutdown")
    async def on_shutdown():
        for bot_id in list(_log_tailers):
            _stop_log_tailer(bot_id)
        # Don't stop_all — bots should keep running across web server restarts
        # User can explicitly stop them from the UI

    return app
