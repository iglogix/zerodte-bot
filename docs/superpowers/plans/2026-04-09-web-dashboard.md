# Web Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + HTMX web dashboard for managing multiple 0DTE bot instances, monitoring positions/logs, and chatting with an AI agent that can inspect and control bots.

**Architecture:** The web server runs as a separate process from bots. It manages bot lifecycles as subprocesses, reads shared state (SQLite DB + log files), maintains its own read-only TastyTrade connection for display, and serves a dark-themed HTMX dashboard with a multiplexed WebSocket for real-time updates and AI chat.

**Tech Stack:** FastAPI, Uvicorn, Jinja2, HTMX (CDN), Tailwind CSS (CDN), WebSocket, LiteLLM (existing)

**Spec:** `docs/superpowers/specs/2026-04-09-web-dashboard-design.md`

---

## File Structure

```
web/
  __init__.py           # Package marker
  app.py                # FastAPI app, routes, startup/shutdown
  bot_manager.py        # Subprocess lifecycle (start/stop/list)
  ws_hub.py             # WebSocket hub: multiplexed connections, broadcast
  log_streamer.py       # Tail + filter bot log files
  market_poller.py      # Read-only TastyTrade poller for positions/account
  chat_agent.py         # Claude agent with bot management tools
  auth.py               # Token middleware + login
  templates/
    base.html           # Layout: sidebar nav, HTMX/WS setup, Tailwind
    login.html          # Token login page
    dashboard.html      # Home: stats, bot grid, recent errors
    bots.html           # Bot management: start/stop/restart
    positions.html      # Open positions table
    decisions.html      # Decision history browser
    logs.html           # Live log stream
    chat.html           # AI chat interface
  static/
    style.css           # Dark theme overrides
    ws.js               # WebSocket client: connect, dispatch by type, reconnect
    chat.js             # Chat UI: send/receive messages, render tool calls
tests_web/
  __init__.py
  conftest.py           # Shared fixtures (test app, mock bot_manager, etc.)
  test_auth.py          # Auth middleware tests
  test_bot_manager.py   # Bot subprocess management tests
  test_routes.py        # HTTP route tests
  test_ws_hub.py        # WebSocket hub tests
  test_log_streamer.py  # Log filtering tests
  test_chat_agent.py    # AI chat agent tool tests
```

---

## Task 1: Project Setup and Auth

**Files:**
- Create: `web/__init__.py`
- Create: `web/auth.py`
- Create: `web/app.py`
- Create: `web/templates/login.html`
- Create: `web/templates/base.html`
- Create: `web/static/style.css`
- Create: `tests_web/__init__.py`
- Create: `tests_web/conftest.py`
- Create: `tests_web/test_auth.py`
- Modify: `pyproject.toml` — add fastapi, uvicorn, jinja2, python-multipart to deps

- [ ] **Step 1: Add web dependencies to pyproject.toml**

Add to the `dependencies` list in `pyproject.toml`:

```
"fastapi>=0.115",
"uvicorn[standard]>=0.30",
"jinja2>=3.1",
"python-multipart>=0.0.9",
```

Then install:

```bash
pip install -e ".[dev]"
```

- [ ] **Step 2: Write auth tests**

Create `tests_web/__init__.py` (empty) and `tests_web/conftest.py`:

```python
"""Shared fixtures for web tests."""
from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _set_web_token(monkeypatch):
    """Ensure a known token is set for all tests."""
    monkeypatch.setenv("BOT_WEB_TOKEN", "test-secret-token")


@pytest.fixture
def app():
    """Create a fresh FastAPI app for testing."""
    # Import inside fixture so env var is already set
    from web.app import create_app
    return create_app()


@pytest.fixture
def client(app):
    """TestClient with no auth."""
    return TestClient(app)


@pytest.fixture
def authed_client(app):
    """TestClient with valid auth cookie."""
    c = TestClient(app)
    c.post("/login", data={"token": "test-secret-token"})
    return c
```

Create `tests_web/test_auth.py`:

```python
"""Tests for token auth middleware."""
from __future__ import annotations


def test_unauthenticated_redirects_to_login(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 303
    assert "/login" in resp.headers["location"]


def test_login_page_accessible_without_auth(client):
    resp = client.get("/login")
    assert resp.status_code == 200


def test_login_with_correct_token_sets_cookie(client):
    resp = client.post(
        "/login",
        data={"token": "test-secret-token"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "/" == resp.headers["location"]
    assert "session" in resp.cookies


def test_login_with_wrong_token_returns_401(client):
    resp = client.post("/login", data={"token": "wrong"})
    assert resp.status_code == 401


def test_authenticated_request_succeeds(authed_client):
    resp = authed_client.get("/")
    assert resp.status_code == 200
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests_web/test_auth.py -v
```

Expected: FAIL — `web.app` module does not exist yet.

- [ ] **Step 4: Implement auth module**

Create `web/__init__.py` (empty).

Create `web/auth.py`:

```python
"""Token-based auth middleware for the web dashboard."""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware


_PUBLIC_PATHS = {"/login", "/static"}


def _get_token() -> str:
    return os.environ.get("BOT_WEB_TOKEN", "")


def _check_token(provided: str) -> bool:
    expected = _get_token()
    if not expected:
        return False
    return hmac.compare_digest(provided, expected)


def _get_session_token(token: str) -> str:
    """Derive a session value from the token (not the raw token itself)."""
    return hashlib.sha256(token.encode()).hexdigest()[:32]


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated requests to /login."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Allow public paths
        if any(path.startswith(p) for p in _PUBLIC_PATHS):
            return await call_next(request)

        # Check session cookie
        session = request.cookies.get("session", "")
        expected = _get_session_token(_get_token())
        if hmac.compare_digest(session, expected):
            return await call_next(request)

        return RedirectResponse(url="/login", status_code=303)
```

- [ ] **Step 5: Implement app.py with login route and base template**

Create `web/app.py`:

```python
"""FastAPI web dashboard application."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.auth import AuthMiddleware, _check_token, _get_session_token, _get_token

_WEB_DIR = Path(__file__).parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="0DTE Bot Dashboard")

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.add_middleware(AuthMiddleware)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

    @app.post("/login")
    async def login_submit(request: Request, token: str = Form(...)):
        if not _check_token(token):
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid token"},
                status_code=401,
            )
        session_value = _get_session_token(_get_token())
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            "session",
            session_value,
            httponly=True,
            max_age=86400 * 7,  # 7 days
            samesite="lax",
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

    return app
```

- [ ] **Step 6: Create base template and login page**

Create `web/templates/base.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}0DTE Bot{% endblock %}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <script src="https://unpkg.com/htmx-ext-ws@2.0.3/ws.js"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            dark: { bg: '#0f172a', card: '#1e293b', border: '#334155' },
            accent: { green: '#22c55e', red: '#ef4444', yellow: '#eab308', cyan: '#06b6d4' },
          }
        }
      }
    }
  </script>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body class="bg-dark-bg text-gray-200 min-h-screen flex">
  {% block body %}
  <!-- Sidebar -->
  <nav class="w-56 bg-dark-card border-r border-dark-border flex flex-col p-4 fixed h-full">
    <div class="text-xl font-bold text-accent-cyan mb-8">0DTE Bot</div>
    <a href="/" class="nav-link {% if active == 'dashboard' %}active{% endif %}">Dashboard</a>
    <a href="/bots" class="nav-link {% if active == 'bots' %}active{% endif %}">Bots</a>
    <a href="/positions" class="nav-link {% if active == 'positions' %}active{% endif %}">Positions</a>
    <a href="/decisions" class="nav-link {% if active == 'decisions' %}active{% endif %}">Decisions</a>
    <a href="/logs" class="nav-link {% if active == 'logs' %}active{% endif %}">Logs</a>
    <a href="/chat" class="nav-link {% if active == 'chat' %}active{% endif %}">AI Chat</a>
  </nav>

  <!-- Main content -->
  <main class="ml-56 flex-1 p-6">
    {% block content %}{% endblock %}
  </main>

  <script src="/static/ws.js"></script>
  {% block scripts %}{% endblock %}
  {% endblock %}
</body>
</html>
```

Create `web/templates/login.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login — 0DTE Bot</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body class="bg-slate-900 text-gray-200 min-h-screen flex items-center justify-center">
  <form method="post" action="/login" class="bg-slate-800 p-8 rounded-lg shadow-lg w-80">
    <h1 class="text-xl font-bold text-cyan-400 mb-6 text-center">0DTE Bot</h1>
    {% if error %}
    <div class="bg-red-900/50 text-red-400 text-sm p-2 rounded mb-4">{{ error }}</div>
    {% endif %}
    <input
      type="password"
      name="token"
      placeholder="Access token"
      class="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm mb-4 focus:outline-none focus:border-cyan-400"
      autofocus
    >
    <button type="submit" class="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-2 rounded text-sm font-medium">
      Login
    </button>
  </form>
</body>
</html>
```

Create `web/templates/dashboard.html`:

```html
{% extends "base.html" %}
{% set active = "dashboard" %}
{% block title %}Dashboard — 0DTE Bot{% endblock %}
{% block content %}
<h1 class="text-2xl font-semibold mb-6">Dashboard</h1>
<p class="text-gray-400">Coming soon.</p>
{% endblock %}
```

Create `web/static/style.css`:

```css
/* Nav links */
.nav-link {
  display: block;
  padding: 0.5rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  color: #94a3b8;
  text-decoration: none;
  margin-bottom: 0.125rem;
}
.nav-link:hover { background: #334155; color: #e2e8f0; }
.nav-link.active { background: #0e7490; color: #ffffff; }

/* Stat cards */
.stat-card {
  background: #1e293b;
  border-radius: 0.5rem;
  padding: 1rem;
}
.stat-label {
  font-size: 0.6875rem;
  text-transform: uppercase;
  color: #64748b;
  letter-spacing: 0.05em;
}
.stat-value {
  font-size: 1.75rem;
  font-weight: 700;
  margin-top: 0.25rem;
}

/* Bot cards */
.bot-card {
  background: #1e293b;
  border-radius: 0.5rem;
  padding: 1rem;
  border-left: 3px solid #334155;
}
.bot-card.running { border-left-color: #22c55e; }
.bot-card.stopped { border-left-color: #64748b; }
.bot-card.error { border-left-color: #ef4444; }

/* Status badges */
.badge {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.6875rem;
  font-weight: 600;
  text-transform: uppercase;
}
.badge-running { background: #064e3b; color: #22c55e; }
.badge-stopped { background: #1e293b; color: #64748b; }
.badge-error { background: #450a0a; color: #ef4444; }

/* Log lines */
.log-line { font-family: ui-monospace, monospace; font-size: 0.75rem; padding: 0.125rem 0; }
.log-error { color: #ef4444; }
.log-warning { color: #eab308; }
.log-info { color: #94a3b8; }
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests_web/test_auth.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 8: Create ws.js placeholder**

Create `web/static/ws.js`:

```javascript
// WebSocket client — connects to /ws and dispatches messages by type.
// Implemented in Task 3.
```

- [ ] **Step 9: Commit**

```bash
git add web/ tests_web/ pyproject.toml
git commit -m "feat(web): add FastAPI app with token auth, login page, base template"
```

---

## Task 2: Bot Manager (subprocess lifecycle)

**Files:**
- Create: `web/bot_manager.py`
- Create: `tests_web/test_bot_manager.py`

- [ ] **Step 1: Write bot manager tests**

Create `tests_web/test_bot_manager.py`:

```python
"""Tests for bot subprocess manager."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from web.bot_manager import BotManager, BotInfo


@pytest.fixture
def manager():
    return BotManager()


def test_list_bots_empty(manager):
    assert manager.list_bots() == []


def test_list_configs(manager, tmp_path):
    (tmp_path / "config_iron_fly_spx.yaml").write_text("bot: {strategy: iron_fly}")
    (tmp_path / "config_orb_em_ic_spx.yaml").write_text("bot: {strategy: orb_em_ic}")
    (tmp_path / "not_a_config.txt").write_text("nope")
    configs = BotManager.list_configs(str(tmp_path))
    assert len(configs) == 2
    assert any("iron_fly" in c for c in configs)


@pytest.mark.asyncio
async def test_start_bot_creates_subprocess(manager, tmp_path):
    config_file = tmp_path / "config_test.yaml"
    config_file.write_text("bot: {strategy: iron_condor}")

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_exec.return_value = mock_proc

        bot_id = await manager.start_bot(
            config_path=str(config_file),
            mode="dry_run",
        )

    assert bot_id is not None
    bots = manager.list_bots()
    assert len(bots) == 1
    assert bots[0].status == "running"
    assert bots[0].config_path == str(config_file)
    assert bots[0].mode == "dry_run"


@pytest.mark.asyncio
async def test_stop_bot(manager, tmp_path):
    config_file = tmp_path / "config_test.yaml"
    config_file.write_text("bot: {strategy: iron_condor}")

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_exec.return_value = mock_proc

        bot_id = await manager.start_bot(config_path=str(config_file), mode="dry_run")
        stopped = await manager.stop_bot(bot_id)

    assert stopped is True
    bots = manager.list_bots()
    assert bots[0].status == "stopped"


def test_stop_nonexistent_bot(manager):
    result = asyncio.get_event_loop().run_until_complete(
        manager.stop_bot("nonexistent")
    )
    assert result is False


def test_bot_name_auto_generated(manager):
    name = BotManager._derive_name("config/config_iron_fly_spx.yaml")
    assert name == "iron_fly_spx"


def test_bot_name_fallback(manager):
    name = BotManager._derive_name("config/config.yaml")
    assert name == "config"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests_web/test_bot_manager.py -v
```

Expected: FAIL — `web.bot_manager` does not exist.

- [ ] **Step 3: Implement bot manager**

Create `web/bot_manager.py`:

```python
"""Bot subprocess lifecycle manager."""
from __future__ import annotations

import asyncio
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BotInfo:
    """Snapshot of a managed bot instance."""
    bot_id: str
    name: str
    config_path: str
    mode: str
    status: str  # "running" | "stopped" | "error"
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    error: Optional[str] = None
    log_file: Optional[str] = None


class BotManager:
    """Manages bot subprocesses — start, stop, list, status."""

    def __init__(self):
        self._bots: Dict[str, _BotEntry] = {}

    def list_bots(self) -> List[BotInfo]:
        """Return info for all managed bots."""
        return [e.to_info() for e in self._bots.values()]

    def get_bot(self, bot_id: str) -> Optional[BotInfo]:
        entry = self._bots.get(bot_id)
        return entry.to_info() if entry else None

    @staticmethod
    def list_configs(config_dir: str = "config") -> List[str]:
        """List available config YAML files."""
        p = Path(config_dir)
        if not p.is_dir():
            return []
        return sorted(
            str(f) for f in p.glob("config*.yaml")
        )

    @staticmethod
    def _derive_name(config_path: str) -> str:
        """Derive a display name from config filename."""
        stem = Path(config_path).stem  # e.g. "config_iron_fly_spx"
        name = re.sub(r"^config_?", "", stem)
        return name or stem

    async def start_bot(
        self,
        config_path: str,
        mode: str,
        name: Optional[str] = None,
    ) -> str:
        """Start a bot subprocess. Returns bot_id."""
        bot_id = uuid.uuid4().hex[:8]
        display_name = name or self._derive_name(config_path)

        cmd = [
            sys.executable, "-m", "src_v2.main",
            f"--config={config_path}",
        ]
        if mode != "dry_run":
            cmd.append(f"--{mode}")

        log_dir = Path("logs") / "web_bots"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{display_name}_{bot_id}.log"

        log_fh = open(log_file, "w")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
        )

        entry = _BotEntry(
            bot_id=bot_id,
            name=display_name,
            config_path=config_path,
            mode=mode,
            process=proc,
            log_fh=log_fh,
            log_file=str(log_file),
            started_at=datetime.now(timezone.utc),
        )
        self._bots[bot_id] = entry
        logger.info("Started bot %s (pid=%s, config=%s, mode=%s)", bot_id, proc.pid, config_path, mode)
        return bot_id

    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot subprocess. Returns True if stopped."""
        entry = self._bots.get(bot_id)
        if not entry:
            return False
        if entry.process.returncode is not None:
            entry.stopped_at = datetime.now(timezone.utc)
            return True

        entry.process.terminate()
        try:
            await asyncio.wait_for(entry.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            entry.process.kill()
            await entry.process.wait()

        entry.stopped_at = datetime.now(timezone.utc)
        if entry.log_fh:
            entry.log_fh.close()
        logger.info("Stopped bot %s", bot_id)
        return True

    async def restart_bot(self, bot_id: str) -> Optional[str]:
        """Restart a bot — stop then start with same config/mode. Returns new bot_id."""
        entry = self._bots.get(bot_id)
        if not entry:
            return None
        config_path = entry.config_path
        mode = entry.mode
        name = entry.name
        await self.stop_bot(bot_id)
        return await self.start_bot(config_path=config_path, mode=mode, name=name)

    def check_health(self):
        """Update status for bots whose processes have exited."""
        for entry in self._bots.values():
            if entry.process.returncode is not None and entry.stopped_at is None:
                entry.stopped_at = datetime.now(timezone.utc)
                if entry.process.returncode != 0:
                    entry.error = f"Exited with code {entry.process.returncode}"

    async def stop_all(self):
        """Stop all running bots — called on web server shutdown."""
        for bot_id in list(self._bots):
            await self.stop_bot(bot_id)


@dataclass
class _BotEntry:
    """Internal mutable state for a managed bot."""
    bot_id: str
    name: str
    config_path: str
    mode: str
    process: asyncio.subprocess.Process
    log_fh: object
    log_file: str
    started_at: datetime
    stopped_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_info(self) -> BotInfo:
        self_status = "running"
        if self.stopped_at:
            self_status = "stopped"
        if self.error:
            self_status = "error"
        if self.process.returncode is not None and self.stopped_at is None:
            if self.process.returncode != 0:
                self_status = "error"
                self.error = f"Exited with code {self.process.returncode}"
            else:
                self_status = "stopped"
        return BotInfo(
            bot_id=self.bot_id,
            name=self.name,
            config_path=self.config_path,
            mode=self.mode,
            status=self_status,
            pid=self.process.pid,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            error=self.error,
            log_file=self.log_file,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests_web/test_bot_manager.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add web/bot_manager.py tests_web/test_bot_manager.py
git commit -m "feat(web): add BotManager for subprocess lifecycle management"
```

---

## Task 3: WebSocket Hub

**Files:**
- Create: `web/ws_hub.py`
- Create: `tests_web/test_ws_hub.py`
- Update: `web/static/ws.js`

- [ ] **Step 1: Write WebSocket hub tests**

Create `tests_web/test_ws_hub.py`:

```python
"""Tests for WebSocket hub."""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from web.ws_hub import WSHub


@pytest.fixture
def hub():
    return WSHub()


@pytest.mark.asyncio
async def test_connect_and_disconnect(hub):
    ws = AsyncMock()
    await hub.connect(ws)
    assert len(hub.connections) == 1
    hub.disconnect(ws)
    assert len(hub.connections) == 0


@pytest.mark.asyncio
async def test_broadcast_sends_to_all(hub):
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    await hub.connect(ws1)
    await hub.connect(ws2)

    await hub.broadcast("positions", {"trades": []})

    for ws in (ws1, ws2):
        ws.send_json.assert_called_once()
        msg = ws.send_json.call_args[0][0]
        assert msg["type"] == "positions"
        assert msg["data"] == {"trades": []}


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections(hub):
    ws_alive = AsyncMock()
    ws_dead = AsyncMock()
    ws_dead.send_json.side_effect = Exception("disconnected")
    await hub.connect(ws_alive)
    await hub.connect(ws_dead)

    await hub.broadcast("test", {})

    assert len(hub.connections) == 1
    assert ws_alive in hub.connections
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests_web/test_ws_hub.py -v
```

Expected: FAIL — `web.ws_hub` does not exist.

- [ ] **Step 3: Implement WebSocket hub**

Create `web/ws_hub.py`:

```python
"""WebSocket hub for multiplexed real-time updates."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WSHub:
    """Manages WebSocket connections and broadcasts typed messages."""

    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg_type: str, data: Any):
        """Send a typed message to all connected clients."""
        message = {"type": msg_type, "data": data}
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, msg_type: str, data: Any):
        """Send a typed message to a specific client."""
        await ws.send_json({"type": msg_type, "data": data})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests_web/test_ws_hub.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Implement ws.js client**

Create `web/static/ws.js`:

```javascript
/**
 * WebSocket client — connects to /ws, dispatches messages by type,
 * and auto-reconnects on disconnect.
 */
(function () {
  let ws = null;
  let reconnectTimer = null;
  const handlers = {};

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
      console.log("[ws] connected");
      if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const handler = handlers[msg.type];
        if (handler) handler(msg.data);
      } catch (e) {
        console.error("[ws] parse error", e);
      }
    };

    ws.onclose = () => {
      console.log("[ws] disconnected, reconnecting...");
      if (!reconnectTimer) {
        reconnectTimer = setInterval(connect, 3000);
      }
    };
  }

  /** Register a handler for a message type. */
  window.wsOn = function (type, callback) {
    handlers[type] = callback;
  };

  /** Send a message to the server. */
  window.wsSend = function (type, data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type, data }));
    }
  };

  // Auto-connect when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", connect);
  } else {
    connect();
  }
})();
```

- [ ] **Step 6: Commit**

```bash
git add web/ws_hub.py web/static/ws.js tests_web/test_ws_hub.py
git commit -m "feat(web): add WebSocket hub with multiplexed broadcasts and JS client"
```

---

## Task 4: Log Streamer

**Files:**
- Create: `web/log_streamer.py`
- Create: `tests_web/test_log_streamer.py`

- [ ] **Step 1: Write log streamer tests**

Create `tests_web/test_log_streamer.py`:

```python
"""Tests for log streaming and filtering."""
from __future__ import annotations

import pytest
from web.log_streamer import LogFilter, tail_lines


def test_filter_accepts_error():
    f = LogFilter()
    line = "2026-04-09 10:30:00 [ERROR] src.main: Connection failed"
    assert f.accept(line) is True


def test_filter_accepts_warning():
    f = LogFilter()
    line = "2026-04-09 10:30:00 [WARNING] src.main: Retrying..."
    assert f.accept(line) is True


def test_filter_accepts_meaningful_info():
    f = LogFilter()
    assert f.accept("[INFO] Cycle complete: outcome=executed") is True
    assert f.accept("[INFO] Trade filled: order_id=abc123") is True
    assert f.accept("[INFO] Bot initialized — mode: LIVE") is True


def test_filter_rejects_debug():
    f = LogFilter()
    line = "2026-04-09 10:30:00 [DEBUG] src.cache: Cache hit for SPX"
    assert f.accept(line) is False


def test_filter_rejects_routine_info():
    f = LogFilter()
    line = "2026-04-09 10:30:00 [INFO] src.cache: Polling market data"
    assert f.accept(line) is False


def test_tail_lines(tmp_path):
    log = tmp_path / "test.log"
    log.write_text("line1\nline2\nline3\nline4\nline5\n")
    result = tail_lines(str(log), n=3)
    assert result == ["line3", "line4", "line5"]


def test_tail_lines_missing_file():
    result = tail_lines("/nonexistent/file.log", n=10)
    assert result == []


def test_classify_level():
    f = LogFilter()
    assert f.classify("[ERROR] something") == "error"
    assert f.classify("[WARNING] something") == "warning"
    assert f.classify("[INFO] something") == "info"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests_web/test_log_streamer.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement log streamer**

Create `web/log_streamer.py`:

```python
"""Log file tailing and filtering for bot log streams."""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# INFO lines matching these patterns are considered meaningful
_MEANINGFUL_INFO = re.compile(
    r"(cycle complete|outcome=|trade filled|order.*(filled|placed|cancel)"
    r"|position.*(open|close|expir)|bot init|mode:|strategy:|shutdown"
    r"|session skip|entry watch|breakout|stop.*(trigger|hit|loss))",
    re.IGNORECASE,
)


class LogFilter:
    """Filter log lines to ERROR, WARNING, and meaningful INFO."""

    def accept(self, line: str) -> bool:
        """Return True if line should be shown to the user."""
        if "[DEBUG]" in line:
            return False
        if "[ERROR]" in line or "[WARNING]" in line:
            return True
        if "[INFO]" in line:
            return bool(_MEANINGFUL_INFO.search(line))
        return False

    def classify(self, line: str) -> str:
        """Return log level class: 'error', 'warning', or 'info'."""
        if "[ERROR]" in line:
            return "error"
        if "[WARNING]" in line:
            return "warning"
        return "info"


def tail_lines(file_path: str, n: int = 50) -> List[str]:
    """Read the last n lines from a file."""
    p = Path(file_path)
    if not p.exists():
        return []
    try:
        lines = p.read_text().splitlines()
        return lines[-n:] if len(lines) > n else lines
    except Exception:
        return []


class LogTailer:
    """Async generator that tails a log file and yields filtered lines."""

    def __init__(self, file_path: str, log_filter: Optional[LogFilter] = None):
        self.file_path = file_path
        self.filter = log_filter or LogFilter()
        self._running = False

    async def tail(self, callback: Callable[[str, str], None]):
        """
        Continuously tail the log file.
        Calls callback(line, level) for each accepted line.
        """
        self._running = True
        p = Path(self.file_path)

        # Wait for file to exist
        while self._running and not p.exists():
            await asyncio.sleep(1)

        pos = p.stat().st_size if p.exists() else 0

        while self._running:
            try:
                if not p.exists():
                    await asyncio.sleep(1)
                    continue
                size = p.stat().st_size
                if size < pos:
                    pos = 0  # File was truncated/rotated
                if size > pos:
                    with open(p) as f:
                        f.seek(pos)
                        new_data = f.read()
                        pos = f.tell()
                    for line in new_data.splitlines():
                        line = line.strip()
                        if line and self.filter.accept(line):
                            level = self.filter.classify(line)
                            await callback(line, level)
            except Exception as e:
                logger.error("LogTailer error for %s: %s", self.file_path, e)
            await asyncio.sleep(0.5)

    def stop(self):
        self._running = False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests_web/test_log_streamer.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add web/log_streamer.py tests_web/test_log_streamer.py
git commit -m "feat(web): add log streamer with filtering for ERROR/WARNING/meaningful INFO"
```

---

## Task 5: Market Poller (read-only TastyTrade connection)

**Files:**
- Create: `web/market_poller.py`

- [ ] **Step 1: Implement market poller**

Create `web/market_poller.py`:

```python
"""Read-only TastyTrade poller for positions and account data."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketPoller:
    """
    Periodically polls TastyTrade for positions and account info,
    then broadcasts via a callback.

    Uses a separate read-only TastyTradeClient connection — does not
    interfere with bot-managed positions.
    """

    def __init__(
        self,
        api_client,
        db,
        on_positions: Callable[[List[Dict]], Coroutine],
        on_account: Callable[[Dict], Coroutine],
        position_interval: float = 10.0,
        account_interval: float = 30.0,
    ):
        self.api_client = api_client
        self.db = db
        self._on_positions = on_positions
        self._on_account = on_account
        self._position_interval = position_interval
        self._account_interval = account_interval
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        self._running = True
        self._tasks = [
            asyncio.create_task(self._poll_positions()),
            asyncio.create_task(self._poll_account()),
        ]

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    async def _poll_positions(self):
        while self._running:
            try:
                trades = await self.db.get_open_trades()
                # Enrich with live quotes if available
                enriched = []
                for trade in trades:
                    enriched.append({
                        "trade_id": trade.get("trade_id", ""),
                        "strategy_type": trade.get("strategy_type", ""),
                        "underlying": trade.get("underlying", "SPX"),
                        "total_credit": trade.get("total_credit", 0),
                        "unrealized_pnl": trade.get("unrealized_pnl", 0),
                        "status": trade.get("status", "open"),
                        "entry_time": str(trade.get("entry_time", "")),
                        "strategy_data": trade.get("strategy_data", {}),
                    })
                await self._on_positions(enriched)
            except Exception as e:
                logger.error("Position poll error: %s", e)
            await asyncio.sleep(self._position_interval)

    async def _poll_account(self):
        while self._running:
            try:
                info = await self.api_client.get_account_info()
                await self._on_account(info)
            except Exception as e:
                logger.error("Account poll error: %s", e)
            await asyncio.sleep(self._account_interval)
```

- [ ] **Step 2: Commit**

```bash
git add web/market_poller.py
git commit -m "feat(web): add MarketPoller for read-only position/account polling"
```

---

## Task 6: Dashboard and Bots Pages (routes + templates)

**Files:**
- Modify: `web/app.py` — add routes, WebSocket endpoint, startup/shutdown hooks
- Create: `web/templates/dashboard.html` (replace placeholder)
- Create: `web/templates/bots.html`
- Create: `web/templates/positions.html`
- Create: `web/templates/decisions.html`
- Create: `web/templates/logs.html`
- Create: `tests_web/test_routes.py`

- [ ] **Step 1: Write route tests**

Create `tests_web/test_routes.py`:

```python
"""Tests for HTTP routes."""
from __future__ import annotations

import pytest


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


def test_start_bot_endpoint(authed_client, tmp_path, monkeypatch):
    """POST /bots/start should create a bot (mocked subprocess)."""
    from unittest.mock import AsyncMock, patch

    with patch("web.app._bot_manager") as mock_mgr:
        mock_mgr.start_bot = AsyncMock(return_value="abc123")
        mock_mgr.list_bots.return_value = []
        resp = authed_client.post(
            "/bots/start",
            data={"config_path": "config/config.yaml", "mode": "dry_run"},
            follow_redirects=False,
        )
    assert resp.status_code in (200, 303)


def test_stop_bot_endpoint(authed_client):
    from unittest.mock import AsyncMock, patch

    with patch("web.app._bot_manager") as mock_mgr:
        mock_mgr.stop_bot = AsyncMock(return_value=True)
        mock_mgr.list_bots.return_value = []
        resp = authed_client.post(
            "/bots/stop/abc123",
            follow_redirects=False,
        )
    assert resp.status_code in (200, 303)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests_web/test_routes.py -v
```

Expected: FAIL — routes don't exist yet.

- [ ] **Step 3: Update app.py with all routes and WebSocket endpoint**

Replace `web/app.py` with:

```python
"""FastAPI web dashboard application."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.auth import AuthMiddleware, _check_token, _get_session_token, _get_token
from web.bot_manager import BotManager
from web.ws_hub import WSHub
from web.log_streamer import LogFilter, tail_lines

_WEB_DIR = Path(__file__).parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

# Module-level singletons (set in create_app or overridden in tests)
_bot_manager: BotManager = BotManager()
_ws_hub: WSHub = WSHub()


def create_app() -> FastAPI:
    global _bot_manager, _ws_hub
    _bot_manager = BotManager()
    _ws_hub = WSHub()

    app = FastAPI(title="0DTE Bot Dashboard")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.add_middleware(AuthMiddleware)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # --- Auth ---

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

    @app.post("/login")
    async def login_submit(request: Request, token: str = Form(...)):
        if not _check_token(token):
            return templates.TemplateResponse(
                "login.html", {"request": request, "error": "Invalid token"},
                status_code=401,
            )
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            "session", _get_session_token(_get_token()),
            httponly=True, max_age=86400 * 7, samesite="lax",
        )
        return response

    # --- Pages ---

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        _bot_manager.check_health()
        bots = _bot_manager.list_bots()
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "active": "dashboard",
            "bots": bots,
        })

    @app.get("/bots", response_class=HTMLResponse)
    async def bots_page(request: Request):
        _bot_manager.check_health()
        bots = _bot_manager.list_bots()
        configs = BotManager.list_configs()
        return templates.TemplateResponse("bots.html", {
            "request": request,
            "active": "bots",
            "bots": bots,
            "configs": configs,
        })

    @app.post("/bots/start")
    async def start_bot(
        request: Request,
        config_path: str = Form(...),
        mode: str = Form("dry_run"),
        name: str = Form(""),
    ):
        await _bot_manager.start_bot(
            config_path=config_path,
            mode=mode,
            name=name or None,
        )
        return RedirectResponse(url="/bots", status_code=303)

    @app.post("/bots/stop/{bot_id}")
    async def stop_bot(bot_id: str):
        await _bot_manager.stop_bot(bot_id)
        return RedirectResponse(url="/bots", status_code=303)

    @app.post("/bots/restart/{bot_id}")
    async def restart_bot(bot_id: str):
        await _bot_manager.restart_bot(bot_id)
        return RedirectResponse(url="/bots", status_code=303)

    @app.get("/positions", response_class=HTMLResponse)
    async def positions_page(request: Request):
        return templates.TemplateResponse("positions.html", {
            "request": request,
            "active": "positions",
        })

    @app.get("/decisions", response_class=HTMLResponse)
    async def decisions_page(request: Request):
        selected_date = request.query_params.get("date", date.today().isoformat())
        decisions_dir = Path("logs/decisions") / selected_date
        files = []
        if decisions_dir.exists():
            files = sorted(decisions_dir.glob("*.json"), reverse=True)
        return templates.TemplateResponse("decisions.html", {
            "request": request,
            "active": "decisions",
            "selected_date": selected_date,
            "decision_files": files,
        })

    @app.get("/decisions/{date_str}/{filename}", response_class=HTMLResponse)
    async def decision_detail(request: Request, date_str: str, filename: str):
        fpath = Path("logs/decisions") / date_str / filename
        data = {}
        if fpath.exists():
            data = json.loads(fpath.read_text())
        return templates.TemplateResponse("decisions.html", {
            "request": request,
            "active": "decisions",
            "selected_date": date_str,
            "decision_files": sorted(fpath.parent.glob("*.json"), reverse=True) if fpath.parent.exists() else [],
            "detail": data,
            "detail_filename": filename,
        })

    @app.get("/logs", response_class=HTMLResponse)
    async def logs_page(request: Request):
        _bot_manager.check_health()
        bots = _bot_manager.list_bots()
        return templates.TemplateResponse("logs.html", {
            "request": request,
            "active": "logs",
            "bots": bots,
        })

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page(request: Request):
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "active": "chat",
        })

    # --- WebSocket ---

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        # Verify auth via cookie
        session = ws.cookies.get("session", "")
        import hmac
        expected = _get_session_token(_get_token())
        if not hmac.compare_digest(session, expected):
            await ws.close(code=4001)
            return

        await ws.accept()
        await _ws_hub.connect(ws)
        try:
            while True:
                data = await ws.receive_json()
                # Handle client messages (e.g., chat) — Task 8
        except WebSocketDisconnect:
            _ws_hub.disconnect(ws)

    return app
```

- [ ] **Step 4: Create dashboard template**

Replace `web/templates/dashboard.html`:

```html
{% extends "base.html" %}
{% set active = "dashboard" %}
{% block title %}Dashboard — 0DTE Bot{% endblock %}
{% block content %}
<h1 class="text-2xl font-semibold mb-6">Dashboard</h1>

<!-- Stat cards (updated via WebSocket) -->
<div id="stats" class="grid grid-cols-4 gap-4 mb-6">
  <div class="stat-card">
    <div class="stat-label">Running Bots</div>
    <div class="stat-value text-cyan-400" id="stat-running">{{ bots|selectattr('status','equalto','running')|list|length }}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Open Positions</div>
    <div class="stat-value" id="stat-positions">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Daily P&L</div>
    <div class="stat-value" id="stat-pnl">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Buying Power</div>
    <div class="stat-value" id="stat-bp">—</div>
  </div>
</div>

<!-- Bot grid -->
<h2 class="text-lg font-semibold mb-3">All Bots</h2>
<div id="bot-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
  {% for bot in bots %}
  <div class="bot-card {{ bot.status }}">
    <div class="flex justify-between items-center mb-1">
      <span class="font-semibold text-sm">{{ bot.name }}</span>
      <span class="badge badge-{{ bot.status }}">{{ bot.status }}</span>
    </div>
    <div class="text-xs text-gray-400">
      Mode: {{ bot.mode }} · Config: {{ bot.config_path | basename }}
    </div>
    {% if bot.error %}
    <div class="text-xs text-red-400 mt-1">{{ bot.error }}</div>
    {% endif %}
  </div>
  {% endfor %}
  {% if not bots %}
  <p class="text-gray-500 text-sm col-span-3">No bots running. <a href="/bots" class="text-cyan-400 underline">Start one</a>.</p>
  {% endif %}
</div>

<!-- Recent errors -->
<h2 class="text-lg font-semibold mb-3">Recent Errors</h2>
<div id="recent-errors" class="bg-dark-card rounded-lg p-3 font-mono text-xs max-h-48 overflow-y-auto">
  <p class="text-gray-500">Errors will appear here in real-time.</p>
</div>

<script>
wsOn("positions", function(data) {
  document.getElementById("stat-positions").textContent = data.length || "0";
  let pnl = data.reduce((sum, t) => sum + (t.unrealized_pnl || 0), 0);
  let el = document.getElementById("stat-pnl");
  el.textContent = (pnl >= 0 ? "+" : "") + "$" + pnl.toFixed(0);
  el.className = "stat-value " + (pnl >= 0 ? "text-green-400" : "text-red-400");
});

wsOn("account", function(data) {
  let bp = data.buying_power || 0;
  document.getElementById("stat-bp").textContent = "$" + (bp / 1000).toFixed(1) + "K";
});

wsOn("log", function(data) {
  if (data.level !== "error") return;
  let el = document.getElementById("recent-errors");
  if (el.querySelector("p.text-gray-500")) el.innerHTML = "";
  let line = document.createElement("div");
  line.className = "log-line log-" + data.level;
  line.textContent = data.line;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
});
</script>
{% endblock %}
```

- [ ] **Step 5: Create bots page template**

Create `web/templates/bots.html`:

```html
{% extends "base.html" %}
{% set active = "bots" %}
{% block title %}Bots — 0DTE Bot{% endblock %}
{% block content %}
<h1 class="text-2xl font-semibold mb-6">Bots</h1>

<!-- Start new bot -->
<div class="bg-dark-card rounded-lg p-4 mb-6">
  <h2 class="text-lg font-semibold mb-3">Start New Bot</h2>
  <form method="post" action="/bots/start" class="flex gap-3 items-end flex-wrap">
    <div>
      <label class="block text-xs text-gray-400 mb-1">Config</label>
      <select name="config_path" class="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm">
        {% for cfg in configs %}
        <option value="{{ cfg }}">{{ cfg }}</option>
        {% endfor %}
      </select>
    </div>
    <div>
      <label class="block text-xs text-gray-400 mb-1">Mode</label>
      <select name="mode" class="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm">
        <option value="dry_run">dry_run</option>
        <option value="manual">manual</option>
        <option value="sandbox">sandbox</option>
        <option value="live">live</option>
      </select>
    </div>
    <div>
      <label class="block text-xs text-gray-400 mb-1">Name (optional)</label>
      <input name="name" type="text" placeholder="auto" class="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm w-40">
    </div>
    <button type="submit" class="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded text-sm font-medium">
      Start
    </button>
  </form>
</div>

<!-- Bot list -->
<div class="bg-dark-card rounded-lg overflow-hidden">
  <table class="w-full text-sm">
    <thead class="bg-slate-800 text-gray-400 text-xs uppercase">
      <tr>
        <th class="px-4 py-2 text-left">Name</th>
        <th class="px-4 py-2 text-left">Config</th>
        <th class="px-4 py-2 text-left">Mode</th>
        <th class="px-4 py-2 text-left">Status</th>
        <th class="px-4 py-2 text-left">PID</th>
        <th class="px-4 py-2 text-left">Actions</th>
      </tr>
    </thead>
    <tbody>
      {% for bot in bots %}
      <tr class="border-t border-dark-border">
        <td class="px-4 py-2 font-medium">{{ bot.name }}</td>
        <td class="px-4 py-2 text-gray-400">{{ bot.config_path | basename }}</td>
        <td class="px-4 py-2">{{ bot.mode }}</td>
        <td class="px-4 py-2"><span class="badge badge-{{ bot.status }}">{{ bot.status }}</span></td>
        <td class="px-4 py-2 text-gray-400">{{ bot.pid or '—' }}</td>
        <td class="px-4 py-2">
          {% if bot.status == 'running' %}
          <form method="post" action="/bots/stop/{{ bot.bot_id }}" class="inline">
            <button class="text-red-400 hover:text-red-300 text-xs mr-2">Stop</button>
          </form>
          <form method="post" action="/bots/restart/{{ bot.bot_id }}" class="inline">
            <button class="text-yellow-400 hover:text-yellow-300 text-xs">Restart</button>
          </form>
          {% else %}
          <span class="text-gray-500 text-xs">—</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
      {% if not bots %}
      <tr><td colspan="6" class="px-4 py-4 text-gray-500 text-center">No bots yet.</td></tr>
      {% endif %}
    </tbody>
  </table>
</div>
{% endblock %}
```

- [ ] **Step 6: Create positions template**

Create `web/templates/positions.html`:

```html
{% extends "base.html" %}
{% set active = "positions" %}
{% block title %}Positions — 0DTE Bot{% endblock %}
{% block content %}
<h1 class="text-2xl font-semibold mb-6">Positions</h1>

<div class="bg-dark-card rounded-lg overflow-hidden">
  <table class="w-full text-sm" id="positions-table">
    <thead class="bg-slate-800 text-gray-400 text-xs uppercase">
      <tr>
        <th class="px-4 py-2 text-left">Trade ID</th>
        <th class="px-4 py-2 text-left">Strategy</th>
        <th class="px-4 py-2 text-left">Underlying</th>
        <th class="px-4 py-2 text-right">Credit</th>
        <th class="px-4 py-2 text-right">P&L</th>
        <th class="px-4 py-2 text-left">Entry</th>
        <th class="px-4 py-2 text-left">Status</th>
      </tr>
    </thead>
    <tbody id="positions-body">
      <tr><td colspan="7" class="px-4 py-4 text-gray-500 text-center">Waiting for data...</td></tr>
    </tbody>
  </table>
</div>

<script>
wsOn("positions", function(data) {
  const tbody = document.getElementById("positions-body");
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="px-4 py-4 text-gray-500 text-center">No open positions.</td></tr>';
    return;
  }
  tbody.innerHTML = data.map(t => `
    <tr class="border-t border-dark-border">
      <td class="px-4 py-2 font-mono text-xs">${t.trade_id}</td>
      <td class="px-4 py-2">${t.strategy_type}</td>
      <td class="px-4 py-2">${t.underlying}</td>
      <td class="px-4 py-2 text-right">$${(t.total_credit || 0).toFixed(2)}</td>
      <td class="px-4 py-2 text-right ${t.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
        ${t.unrealized_pnl >= 0 ? '+' : ''}$${(t.unrealized_pnl || 0).toFixed(2)}
      </td>
      <td class="px-4 py-2 text-gray-400 text-xs">${t.entry_time}</td>
      <td class="px-4 py-2"><span class="badge badge-running">${t.status}</span></td>
    </tr>
  `).join("");
});
</script>
{% endblock %}
```

- [ ] **Step 7: Create decisions template**

Create `web/templates/decisions.html`:

```html
{% extends "base.html" %}
{% set active = "decisions" %}
{% block title %}Decisions — 0DTE Bot{% endblock %}
{% block content %}
<h1 class="text-2xl font-semibold mb-6">Decision History</h1>

<form method="get" class="mb-4 flex gap-3 items-end">
  <div>
    <label class="block text-xs text-gray-400 mb-1">Date</label>
    <input type="date" name="date" value="{{ selected_date }}"
      class="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm"
      onchange="this.form.submit()">
  </div>
</form>

<div class="flex gap-4">
  <!-- File list -->
  <div class="w-64 flex-shrink-0">
    <div class="bg-dark-card rounded-lg p-3 max-h-[70vh] overflow-y-auto">
      {% for f in decision_files %}
      <a href="/decisions/{{ selected_date }}/{{ f.name }}"
         class="block px-2 py-1 text-xs font-mono rounded hover:bg-slate-700 {% if detail_filename == f.name %}bg-slate-700 text-cyan-400{% else %}text-gray-400{% endif %}">
        {{ f.name }}
      </a>
      {% endfor %}
      {% if not decision_files %}
      <p class="text-gray-500 text-xs">No decisions for this date.</p>
      {% endif %}
    </div>
  </div>

  <!-- Detail view -->
  <div class="flex-1">
    {% if detail %}
    <div class="bg-dark-card rounded-lg p-4">
      <div class="flex justify-between items-center mb-3">
        <span class="text-sm font-semibold">{{ detail_filename }}</span>
        <span class="badge badge-{{ 'running' if detail.outcome == 'executed' else 'stopped' }}">{{ detail.outcome }}</span>
      </div>
      <pre class="text-xs text-gray-300 overflow-x-auto max-h-[60vh] overflow-y-auto">{{ detail | tojson(indent=2) }}</pre>
    </div>
    {% else %}
    <div class="bg-dark-card rounded-lg p-4 text-gray-500 text-sm">
      Select a decision file to view details.
    </div>
    {% endif %}
  </div>
</div>
{% endblock %}
```

- [ ] **Step 8: Create logs template**

Create `web/templates/logs.html`:

```html
{% extends "base.html" %}
{% set active = "logs" %}
{% block title %}Logs — 0DTE Bot{% endblock %}
{% block content %}
<h1 class="text-2xl font-semibold mb-6">Logs</h1>

<!-- Bot tabs -->
<div class="flex gap-1 mb-4">
  <button class="log-tab active" data-bot="all" onclick="selectLogTab('all', this)">All</button>
  {% for bot in bots %}
  <button class="log-tab" data-bot="{{ bot.bot_id }}" onclick="selectLogTab('{{ bot.bot_id }}', this)">
    {{ bot.name }}
  </button>
  {% endfor %}
</div>

<div class="bg-dark-card rounded-lg p-3">
  <div id="log-container" class="font-mono text-xs max-h-[70vh] overflow-y-auto">
    <p class="text-gray-500">Waiting for log events...</p>
  </div>
  <div class="flex justify-between items-center mt-2">
    <label class="text-xs text-gray-400">
      <input type="checkbox" id="auto-scroll" checked class="mr-1"> Auto-scroll
    </label>
    <button onclick="document.getElementById('log-container').innerHTML=''" class="text-xs text-gray-400 hover:text-white">Clear</button>
  </div>
</div>

<style>
.log-tab { padding: 0.375rem 0.75rem; border-radius: 0.375rem; font-size: 0.75rem; color: #94a3b8; background: #1e293b; border: none; cursor: pointer; }
.log-tab:hover { background: #334155; }
.log-tab.active { background: #0e7490; color: white; }
</style>

<script>
let activeBot = "all";

function selectLogTab(botId, el) {
  activeBot = botId;
  document.querySelectorAll(".log-tab").forEach(t => t.classList.remove("active"));
  el.classList.add("active");
}

wsOn("log", function(data) {
  if (activeBot !== "all" && data.bot_id !== activeBot) return;
  const container = document.getElementById("log-container");
  if (container.querySelector("p.text-gray-500")) container.innerHTML = "";
  const line = document.createElement("div");
  line.className = "log-line log-" + data.level;
  line.textContent = `[${data.bot_name || "?"}] ${data.line}`;
  container.appendChild(line);
  if (document.getElementById("auto-scroll").checked) {
    container.scrollTop = container.scrollHeight;
  }
});
</script>
{% endblock %}
```

- [ ] **Step 9: Create chat template (placeholder)**

Create `web/templates/chat.html`:

```html
{% extends "base.html" %}
{% set active = "chat" %}
{% block title %}AI Chat — 0DTE Bot{% endblock %}
{% block content %}
<h1 class="text-2xl font-semibold mb-6">AI Chat</h1>

<div class="flex flex-col h-[calc(100vh-10rem)]">
  <!-- Messages -->
  <div id="chat-messages" class="flex-1 bg-dark-card rounded-lg p-4 overflow-y-auto mb-4">
    <div class="text-gray-500 text-sm">Ask me about your bots, positions, or market conditions. I can also start/stop bots and close positions.</div>
  </div>

  <!-- Input -->
  <form id="chat-form" class="flex gap-2" onsubmit="sendChat(event)">
    <input id="chat-input" type="text" placeholder="Ask something..."
      class="flex-1 bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-400"
      autocomplete="off">
    <button type="submit" class="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded text-sm font-medium">
      Send
    </button>
  </form>
</div>

<script src="/static/chat.js"></script>
{% endblock %}
```

- [ ] **Step 10: Add basename filter to Jinja2 env**

In `web/app.py`, after `templates = Jinja2Templates(...)`, add:

```python
    templates.env.filters["basename"] = lambda p: Path(p).name
```

- [ ] **Step 11: Run tests**

```bash
.venv/bin/python -m pytest tests_web/test_routes.py tests_web/test_auth.py -v
```

Expected: All tests PASS.

- [ ] **Step 12: Commit**

```bash
git add web/app.py web/templates/ tests_web/test_routes.py
git commit -m "feat(web): add all page routes and templates (dashboard, bots, positions, decisions, logs, chat)"
```

---

## Task 7: Wire Up Real-Time Updates (log streaming + market polling + WebSocket)

**Files:**
- Modify: `web/app.py` — add startup/shutdown lifecycle hooks

- [ ] **Step 1: Add startup/shutdown hooks to app.py**

Add to `create_app()`, after all route definitions but before `return app`:

```python
    @app.on_event("startup")
    async def on_startup():
        """Start background tasks: log tailers, market poller."""
        # Log tailers will be started dynamically when bots start.
        # For now, wire the bot manager's start_bot to also create a tailer.
        pass

    @app.on_event("shutdown")
    async def on_shutdown():
        await _bot_manager.stop_all()
```

- [ ] **Step 2: Add log broadcasting to bot start/stop**

Add a helper in `web/app.py` that creates a LogTailer when a bot starts and broadcasts via WSHub:

```python
import asyncio
from web.log_streamer import LogTailer, LogFilter

_log_tailers: dict = {}  # bot_id -> LogTailer


async def _start_log_tailer(bot_id: str, log_file: str, bot_name: str):
    """Start tailing a bot's log file and broadcast filtered lines."""
    tailer = LogTailer(log_file)
    _log_tailers[bot_id] = tailer

    async def on_line(line: str, level: str):
        await _ws_hub.broadcast("log", {
            "bot_id": bot_id,
            "bot_name": bot_name,
            "line": line,
            "level": level,
        })

    asyncio.create_task(tailer.tail(on_line))


def _stop_log_tailer(bot_id: str):
    tailer = _log_tailers.pop(bot_id, None)
    if tailer:
        tailer.stop()
```

Update the `start_bot` route to also start a tailer:

```python
    @app.post("/bots/start")
    async def start_bot(
        request: Request,
        config_path: str = Form(...),
        mode: str = Form("dry_run"),
        name: str = Form(""),
    ):
        bot_id = await _bot_manager.start_bot(
            config_path=config_path, mode=mode, name=name or None,
        )
        bot = _bot_manager.get_bot(bot_id)
        if bot and bot.log_file:
            await _start_log_tailer(bot_id, bot.log_file, bot.name)
        return RedirectResponse(url="/bots", status_code=303)
```

Update `stop_bot` to also stop the tailer:

```python
    @app.post("/bots/stop/{bot_id}")
    async def stop_bot(bot_id: str):
        _stop_log_tailer(bot_id)
        await _bot_manager.stop_bot(bot_id)
        return RedirectResponse(url="/bots", status_code=303)
```

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(web): wire log streaming and lifecycle hooks to WebSocket broadcasts"
```

---

## Task 8: AI Chat Agent

**Files:**
- Create: `web/chat_agent.py`
- Create: `web/static/chat.js`
- Create: `tests_web/test_chat_agent.py`
- Modify: `web/app.py` — handle chat messages in WebSocket

- [ ] **Step 1: Write chat agent tests**

Create `tests_web/test_chat_agent.py`:

```python
"""Tests for AI chat agent tools."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from web.chat_agent import ChatAgent


@pytest.fixture
def mock_bot_manager():
    mgr = MagicMock()
    mgr.list_bots.return_value = [
        MagicMock(bot_id="abc", name="iron_fly_spx", status="running", mode="manual", config_path="config/config_iron_fly_spx.yaml", pid=1234),
    ]
    mgr.start_bot = AsyncMock(return_value="def456")
    mgr.stop_bot = AsyncMock(return_value=True)
    mgr.restart_bot = AsyncMock(return_value="ghi789")
    return mgr


@pytest.fixture
def agent(mock_bot_manager):
    return ChatAgent(bot_manager=mock_bot_manager)


def test_tool_definitions(agent):
    tools = agent.get_tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert "list_bots" in names
    assert "stop_bot" in names
    assert "start_bot" in names
    assert "get_bot_logs" in names


@pytest.mark.asyncio
async def test_execute_list_bots(agent):
    result = await agent.execute_tool("list_bots", {})
    assert "bots" in result
    assert len(result["bots"]) == 1
    assert result["bots"][0]["name"] == "iron_fly_spx"


@pytest.mark.asyncio
async def test_execute_stop_bot_needs_confirmation(agent):
    result = await agent.execute_tool("stop_bot", {"bot_id": "abc"})
    # Action tools return a confirmation prompt, not the result
    assert result.get("needs_confirmation") is True


@pytest.mark.asyncio
async def test_execute_stop_bot_confirmed(agent, mock_bot_manager):
    result = await agent.execute_tool("stop_bot", {"bot_id": "abc"}, confirmed=True)
    assert result.get("stopped") is True
    mock_bot_manager.stop_bot.assert_awaited_once_with("abc")


@pytest.mark.asyncio
async def test_execute_unknown_tool(agent):
    result = await agent.execute_tool("nonexistent", {})
    assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests_web/test_chat_agent.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement chat agent**

Create `web/chat_agent.py`:

```python
"""AI chat agent with bot management tools."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tools that modify state — require user confirmation
_ACTION_TOOLS = {"start_bot", "stop_bot", "restart_bot", "close_position", "change_mode"}


class ChatAgent:
    """
    Claude-powered chat agent with tools for inspecting and controlling bots.
    """

    def __init__(self, bot_manager, db=None, api_client=None):
        self.bot_manager = bot_manager
        self.db = db
        self.api_client = api_client
        self._conversations: Dict[str, List[Dict]] = {}  # session_id -> messages

    def get_tool_definitions(self) -> List[Dict]:
        """Return tool definitions in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_bots",
                    "description": "List all managed bot instances with their status, config, and mode.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bot_logs",
                    "description": "Get recent filtered log lines for a specific bot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "Bot ID to get logs for"},
                            "lines": {"type": "integer", "description": "Number of recent lines (default 50)"},
                        },
                        "required": ["bot_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_positions",
                    "description": "Get all open trading positions with P&L.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_decisions",
                    "description": "Get decision history for a specific date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                        },
                        "required": ["date"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_account_info",
                    "description": "Get account balances, buying power, and risk utilization.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "start_bot",
                    "description": "Start a new bot instance with a config file and mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_path": {"type": "string", "description": "Path to config YAML file"},
                            "mode": {"type": "string", "enum": ["dry_run", "manual", "sandbox", "live"]},
                            "name": {"type": "string", "description": "Optional display name"},
                        },
                        "required": ["config_path", "mode"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "stop_bot",
                    "description": "Stop a running bot instance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "Bot ID to stop"},
                        },
                        "required": ["bot_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restart_bot",
                    "description": "Restart a bot (stop then start with same config/mode).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "Bot ID to restart"},
                        },
                        "required": ["bot_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "close_position",
                    "description": "Close a specific trading position.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trade_id": {"type": "string", "description": "Trade ID to close"},
                        },
                        "required": ["trade_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "change_mode",
                    "description": "Restart a bot with a different mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string", "description": "Bot ID"},
                            "new_mode": {"type": "string", "enum": ["dry_run", "manual", "sandbox", "live"]},
                        },
                        "required": ["bot_id", "new_mode"],
                    },
                },
            },
        ]

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        confirmed: bool = False,
    ) -> Dict[str, Any]:
        """Execute a tool call. Action tools need confirmation unless confirmed=True."""

        # Action tools require confirmation
        if tool_name in _ACTION_TOOLS and not confirmed:
            return {
                "needs_confirmation": True,
                "tool": tool_name,
                "args": args,
                "message": f"Please confirm: {tool_name} with {json.dumps(args)}",
            }

        handler = getattr(self, f"_tool_{tool_name}", None)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return await handler(**args)
        except Exception as e:
            logger.exception("Chat tool %s error", tool_name)
            return {"error": str(e)}

    # --- Read tools ---

    async def _tool_list_bots(self) -> Dict:
        bots = self.bot_manager.list_bots()
        return {"bots": [
            {
                "bot_id": b.bot_id,
                "name": b.name,
                "status": b.status,
                "mode": b.mode,
                "config": b.config_path,
                "pid": b.pid,
                "error": b.error,
            }
            for b in bots
        ]}

    async def _tool_get_bot_logs(self, bot_id: str, lines: int = 50) -> Dict:
        from web.log_streamer import tail_lines, LogFilter
        bot = self.bot_manager.get_bot(bot_id)
        if not bot or not bot.log_file:
            return {"error": f"Bot {bot_id} not found or has no log file"}
        all_lines = tail_lines(bot.log_file, n=lines * 3)  # Read more, then filter
        f = LogFilter()
        filtered = [l for l in all_lines if f.accept(l)][-lines:]
        return {"bot_id": bot_id, "lines": filtered}

    async def _tool_get_positions(self) -> Dict:
        if not self.db:
            return {"error": "Database not available"}
        trades = await self.db.get_open_trades()
        return {"positions": trades}

    async def _tool_get_decisions(self, date: str) -> Dict:
        from pathlib import Path
        decisions_dir = Path("logs/decisions") / date
        if not decisions_dir.exists():
            return {"decisions": [], "message": f"No decisions for {date}"}
        files = sorted(decisions_dir.glob("*.json"))
        decisions = []
        for f in files[-10:]:  # Last 10 decisions
            try:
                data = json.loads(f.read_text())
                decisions.append({
                    "file": f.name,
                    "outcome": data.get("outcome", "unknown"),
                    "started_at": data.get("started_at", ""),
                    "mode": data.get("mode", ""),
                })
            except Exception:
                continue
        return {"decisions": decisions}

    async def _tool_get_account_info(self) -> Dict:
        if not self.api_client:
            return {"error": "TastyTrade client not available"}
        return await self.api_client.get_account_info()

    # --- Action tools ---

    async def _tool_start_bot(self, config_path: str, mode: str, name: str = "") -> Dict:
        bot_id = await self.bot_manager.start_bot(
            config_path=config_path, mode=mode, name=name or None,
        )
        return {"started": True, "bot_id": bot_id}

    async def _tool_stop_bot(self, bot_id: str) -> Dict:
        result = await self.bot_manager.stop_bot(bot_id)
        return {"stopped": result, "bot_id": bot_id}

    async def _tool_restart_bot(self, bot_id: str) -> Dict:
        new_id = await self.bot_manager.restart_bot(bot_id)
        return {"restarted": True, "old_id": bot_id, "new_id": new_id}

    async def _tool_close_position(self, trade_id: str) -> Dict:
        # Delegate to a bot's tool_handlers or direct API call
        return {"error": "Position closing via chat not yet wired to TastyTrade API"}

    async def _tool_change_mode(self, bot_id: str, new_mode: str) -> Dict:
        bot = self.bot_manager.get_bot(bot_id)
        if not bot:
            return {"error": f"Bot {bot_id} not found"}
        await self.bot_manager.stop_bot(bot_id)
        new_id = await self.bot_manager.start_bot(
            config_path=bot.config_path, mode=new_mode, name=bot.name,
        )
        return {"changed": True, "old_id": bot_id, "new_id": new_id, "new_mode": new_mode}

    # --- Chat loop ---

    async def chat(
        self,
        user_message: str,
        session_id: str = "default",
        on_token: Optional[Any] = None,
    ) -> str:
        """
        Run a chat turn: send user message to Claude with tools,
        execute any tool calls, return final text response.

        on_token: async callback for streaming tokens (optional).
        """
        from src_v2.services.llm_client import LLMClient

        if session_id not in self._conversations:
            self._conversations[session_id] = [{
                "role": "system",
                "content": (
                    "You are an AI assistant for the 0DTE options trading bot dashboard. "
                    "You can inspect bot status, read logs and decisions, check positions and account info, "
                    "and take management actions (start/stop/restart bots, close positions, change modes). "
                    "Action tools require user confirmation before executing. "
                    "Be concise and direct. Use the tools available to answer questions."
                ),
            }]

        messages = self._conversations[session_id]
        messages.append({"role": "user", "content": user_message})

        llm = LLMClient(model="anthropic/claude-haiku-4-5", max_tokens=2048)
        tools = self.get_tool_definitions()

        # Tool loop (max 5 iterations)
        for _ in range(5):
            resp = await llm.chat(messages=messages, tools=tools)

            if resp.get("tool_calls"):
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in resp["tool_calls"]
                    ],
                    "content": resp.get("content"),
                })
                # Execute each tool
                for tc in resp["tool_calls"]:
                    args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                    result = await self.execute_tool(tc["name"], args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, default=str),
                    })
                continue

            # No tool calls — we have a final response
            content = resp.get("content", "")
            messages.append({"role": "assistant", "content": content})
            return content

        return "I wasn't able to complete that request within the allowed steps."
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests_web/test_chat_agent.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Implement chat.js**

Create `web/static/chat.js`:

```javascript
/**
 * AI Chat UI — sends messages via WebSocket, renders responses and tool calls.
 */
(function () {
  const messagesEl = document.getElementById("chat-messages");
  const inputEl = document.getElementById("chat-input");
  let pendingConfirmation = null;

  function addMessage(role, content, extra) {
    const div = document.createElement("div");
    div.className = "mb-3";

    if (role === "user") {
      div.innerHTML = `<div class="text-cyan-400 text-xs mb-1">You</div>
        <div class="text-sm">${escapeHtml(content)}</div>`;
    } else if (role === "assistant") {
      div.innerHTML = `<div class="text-green-400 text-xs mb-1">Assistant</div>
        <div class="text-sm whitespace-pre-wrap">${escapeHtml(content)}</div>`;
    } else if (role === "tool") {
      div.innerHTML = `<div class="text-yellow-400 text-xs mb-1">Tool: ${escapeHtml(extra.tool)}</div>
        <pre class="text-xs text-gray-400 bg-slate-800 rounded p-2 overflow-x-auto">${escapeHtml(content)}</pre>`;
    } else if (role === "confirm") {
      div.innerHTML = `<div class="text-yellow-400 text-xs mb-1">Confirmation Required</div>
        <div class="text-sm text-yellow-200">${escapeHtml(content)}</div>
        <div class="mt-2">
          <button onclick="confirmAction(true)" class="bg-green-700 hover:bg-green-600 text-white text-xs px-3 py-1 rounded mr-2">Yes, proceed</button>
          <button onclick="confirmAction(false)" class="bg-slate-600 hover:bg-slate-500 text-white text-xs px-3 py-1 rounded">Cancel</button>
        </div>`;
    }

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escapeHtml(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
  }

  // Handle incoming chat messages from WebSocket
  wsOn("chat_response", function (data) {
    if (data.type === "tool_call") {
      addMessage("tool", JSON.stringify(data.result, null, 2), { tool: data.tool_name });
    } else if (data.type === "confirmation") {
      pendingConfirmation = data;
      addMessage("confirm", data.message);
    } else if (data.type === "text") {
      addMessage("assistant", data.content);
    }
  });

  window.sendChat = function (e) {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    addMessage("user", text);
    wsSend("chat_message", { message: text });
    inputEl.value = "";
  };

  window.confirmAction = function (yes) {
    if (!pendingConfirmation) return;
    wsSend("chat_confirm", { confirmed: yes, ...pendingConfirmation });
    pendingConfirmation = null;
    // Remove confirm buttons
    const buttons = messagesEl.querySelectorAll("button[onclick^='confirmAction']");
    buttons.forEach(b => b.parentElement.remove());
    addMessage("user", yes ? "Confirmed." : "Cancelled.");
  };
})();
```

- [ ] **Step 6: Wire chat into WebSocket handler in app.py**

Update the WebSocket handler in `web/app.py`:

```python
    from web.chat_agent import ChatAgent
    _chat_agent = ChatAgent(bot_manager=_bot_manager)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        import hmac
        session = ws.cookies.get("session", "")
        expected = _get_session_token(_get_token())
        if not hmac.compare_digest(session, expected):
            await ws.close(code=4001)
            return

        await ws.accept()
        await _ws_hub.connect(ws)
        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "chat_message":
                    user_msg = data.get("data", {}).get("message", "")
                    response = await _chat_agent.chat(user_msg)
                    await _ws_hub.send_to(ws, "chat_response", {
                        "type": "text",
                        "content": response,
                    })

                elif msg_type == "chat_confirm":
                    confirmed = data.get("data", {}).get("confirmed", False)
                    tool_name = data.get("data", {}).get("tool")
                    args = data.get("data", {}).get("args", {})
                    if confirmed and tool_name:
                        result = await _chat_agent.execute_tool(tool_name, args, confirmed=True)
                        await _ws_hub.send_to(ws, "chat_response", {
                            "type": "tool_call",
                            "tool_name": tool_name,
                            "result": result,
                        })
        except WebSocketDisconnect:
            _ws_hub.disconnect(ws)
```

- [ ] **Step 7: Run all tests**

```bash
.venv/bin/python -m pytest tests_web/ -v
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add web/chat_agent.py web/static/chat.js tests_web/test_chat_agent.py web/app.py
git commit -m "feat(web): add AI chat agent with bot management tools and WebSocket integration"
```

---

## Task 9: CLI Entry Point

**Files:**
- Create: `web/__main__.py`

- [ ] **Step 1: Create CLI entry point**

Create `web/__main__.py`:

```python
"""CLI entry point: python -m web"""
from __future__ import annotations

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="0DTE Bot Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
timeout 5 .venv/bin/python -m web --port 18080 || true
```

Expected: Server starts, then times out after 5s. No import errors.

- [ ] **Step 3: Commit**

```bash
git add web/__main__.py
git commit -m "feat(web): add CLI entry point (python -m web)"
```

---

## Task 10: Final Integration Test and .gitignore

**Files:**
- Modify: `.gitignore` — add `.superpowers/`
- Create: `tests_web/test_integration.py`

- [ ] **Step 1: Add .superpowers to .gitignore**

Append to `.gitignore`:

```
# Superpowers brainstorming sessions
.superpowers/
```

- [ ] **Step 2: Write integration test**

Create `tests_web/test_integration.py`:

```python
"""Integration test: full app startup and page navigation."""
from __future__ import annotations

import pytest


def test_full_navigation_flow(authed_client):
    """Visit every page and verify no 500 errors."""
    pages = ["/", "/bots", "/positions", "/decisions", "/logs", "/chat"]
    for page in pages:
        resp = authed_client.get(page)
        assert resp.status_code == 200, f"Page {page} returned {resp.status_code}"


def test_start_and_list_bot_flow(authed_client, monkeypatch):
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
```

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest tests_web/ -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add .gitignore tests_web/test_integration.py
git commit -m "feat(web): add integration tests and .gitignore for .superpowers/"
```

---

## Summary

| Task | What it builds | Tests |
|------|---------------|-------|
| 1 | FastAPI app, auth, login, base template | 5 auth tests |
| 2 | BotManager subprocess lifecycle | 7 bot manager tests |
| 3 | WebSocket hub + JS client | 3 hub tests |
| 4 | Log streamer + filtering | 8 log tests |
| 5 | Market poller (read-only TastyTrade) | — |
| 6 | All page routes + templates | 8 route tests |
| 7 | Wire real-time updates (logs + polling → WS) | — |
| 8 | AI chat agent with tools | 5 chat tests |
| 9 | CLI entry point (`python -m web`) | smoke test |
| 10 | Integration test + gitignore | 2 integration tests |
