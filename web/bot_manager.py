"""Bot subprocess lifecycle manager.

Persists bot metadata to a JSON state file so that running bots can be
re-adopted after a web server restart.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import signal
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

log = logging.getLogger(__name__)

_STATE_FILE = Path("logs/web_bots/.bot_state.json")


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


@dataclass(frozen=True)
class BotInfo:
    """Immutable snapshot of a bot's state."""

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


@dataclass
class _BotEntry:
    """Mutable internal state for a managed bot."""

    bot_id: str
    name: str
    config_path: str
    mode: str
    process: Optional[asyncio.subprocess.Process] = None
    status: str = "running"
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    error: Optional[str] = None
    log_file: Optional[str] = None
    _log_fh: Optional[IO] = field(default=None, repr=False)

    def to_info(self) -> BotInfo:
        return BotInfo(
            bot_id=self.bot_id,
            name=self.name,
            config_path=self.config_path,
            mode=self.mode,
            status=self.status,
            pid=self.pid,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            error=self.error,
            log_file=self.log_file,
        )


# Run modes the dashboard may request, mapped to zerodte.main's CLI flags.
# dry_run is the CLI default and needs no flag.
_MODE_FLAGS = {
    "dry_run": [],
    "manual": ["--manual"],
    "sandbox": ["--sandbox"],
    "alpaca": ["--alpaca"],
    "live": ["--live"],
}


class BotManager:
    """Manages bot subprocesses.

    Persists bot metadata to a JSON file so running bots survive web server
    restarts.  On init, ``recover()`` checks for still-alive PIDs and
    re-adopts them.
    """

    def __init__(self) -> None:
        self._bots: dict[str, _BotEntry] = {}

    async def recover(self) -> None:
        """Re-adopt bots from a previous server session.

        Reads the state file, checks which PIDs are still alive, and
        re-creates entries for them (without a Process handle — we can
        only signal them, not wait on them via asyncio).
        """
        if not _STATE_FILE.exists():
            return
        try:
            data = json.loads(_STATE_FILE.read_text())
        except Exception:
            log.warning("Could not read bot state file, starting fresh")
            return

        recovered = 0
        for entry_data in data:
            pid = entry_data.get("pid")
            if pid and _pid_alive(pid):
                entry = _BotEntry(
                    bot_id=entry_data["bot_id"],
                    name=entry_data["name"],
                    config_path=entry_data["config_path"],
                    mode=entry_data["mode"],
                    process=None,  # can't re-attach to subprocess
                    status="running",
                    pid=pid,
                    started_at=datetime.fromisoformat(entry_data["started_at"]) if entry_data.get("started_at") else None,
                    log_file=entry_data.get("log_file"),
                )
                self._bots[entry.bot_id] = entry
                recovered += 1
                log.info("Recovered bot %s (pid=%s, name=%s)", entry.bot_id, pid, entry.name)
            else:
                log.info("Stale bot %s (pid=%s) — process no longer running", entry_data.get("bot_id"), pid)

        if recovered:
            log.info("Recovered %d bot(s) from previous session", recovered)

    def _save_state(self) -> None:
        """Persist current bot metadata to disk."""
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for e in self._bots.values():
            if e.status != "running":
                continue
            entries.append({
                "bot_id": e.bot_id,
                "name": e.name,
                "config_path": e.config_path,
                "mode": e.mode,
                "pid": e.pid,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "log_file": e.log_file,
            })
        _STATE_FILE.write_text(json.dumps(entries, indent=2))

    # -- queries --

    def list_bots(self) -> list[BotInfo]:
        return [entry.to_info() for entry in self._bots.values()]

    def get_bot(self, bot_id: str) -> Optional[BotInfo]:
        entry = self._bots.get(bot_id)
        return entry.to_info() if entry else None

    @staticmethod
    def list_configs(config_dir: str = "config") -> list[str]:
        pattern = os.path.join(config_dir, "config*.yaml")
        return sorted(glob.glob(pattern))

    @staticmethod
    def _derive_name(config_path: str) -> str:
        stem = Path(config_path).stem  # e.g. "config_iron_fly_spx"
        if stem.startswith("config_") and len(stem) > len("config_"):
            return stem[len("config_"):]
        return stem

    # -- mutations --

    async def start_bot(
        self,
        config_path: str,
        mode: str = "dry_run",
        name: Optional[str] = None,
    ) -> str:
        # Both values end up on a command line, so validate before anything runs.
        if mode not in _MODE_FLAGS:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(_MODE_FLAGS)}")
        cfg_file = Path(config_path)
        if cfg_file.suffix not in (".yaml", ".yml") or not cfg_file.is_file():
            raise ValueError(f"Not a bot config file: {config_path!r}")

        bot_id = uuid.uuid4().hex[:8]
        bot_name = name or self._derive_name(config_path)

        # Prepare log directory and file
        log_dir = Path("logs/web_bots")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{bot_name}_{bot_id}.log"

        # Build command
        cmd = [sys.executable, "-m", "zerodte.main", f"--config={config_path}"]
        cmd.extend(_MODE_FLAGS[mode])

        log_fh = open(log_path, "w")

        # PYTHONUNBUFFERED ensures subprocess output is flushed immediately
        # so the log tailer can read lines in real time.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        entry = _BotEntry(
            bot_id=bot_id,
            name=bot_name,
            config_path=config_path,
            mode=mode,
            process=proc,
            status="running",
            pid=proc.pid,
            started_at=datetime.now(timezone.utc),
            log_file=str(log_path),
            _log_fh=log_fh,
        )
        self._bots[bot_id] = entry
        self._save_state()
        log.info("Started bot %s (pid=%s, config=%s)", bot_id, proc.pid, config_path)
        return bot_id

    async def stop_bot(self, bot_id: str) -> bool:
        entry = self._bots.get(bot_id)
        if entry is None:
            return False
        if entry.status != "running":
            return False

        proc = entry.process
        if proc is not None and proc.returncode is None:
            # Normal subprocess — we have a handle
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        elif proc is None and entry.pid and _pid_alive(entry.pid):
            # Recovered bot — no subprocess handle, use os.kill
            try:
                os.kill(entry.pid, signal.SIGTERM)
                # Wait briefly for process to exit
                for _ in range(20):
                    await asyncio.sleep(0.5)
                    if not _pid_alive(entry.pid):
                        break
                else:
                    os.kill(entry.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        entry.status = "stopped"
        entry.stopped_at = datetime.now(timezone.utc)
        if entry._log_fh:
            entry._log_fh.close()
            entry._log_fh = None

        self._save_state()
        log.info("Stopped bot %s", bot_id)
        return True

    async def restart_bot(self, bot_id: str) -> Optional[str]:
        entry = self._bots.get(bot_id)
        if entry is None:
            return None
        config_path = entry.config_path
        mode = entry.mode
        name = entry.name

        await self.stop_bot(bot_id)
        return await self.start_bot(config_path, mode=mode, name=name)

    def check_health(self) -> None:
        changed = False
        for entry in self._bots.values():
            if entry.status != "running":
                continue
            proc = entry.process
            if proc is not None and proc.returncode is not None:
                # Normal subprocess exited
                if proc.returncode == 0:
                    entry.status = "stopped"
                else:
                    entry.status = "error"
                    entry.error = f"Process exited with code {proc.returncode}"
                entry.stopped_at = datetime.now(timezone.utc)
                if entry._log_fh:
                    entry._log_fh.close()
                    entry._log_fh = None
                changed = True
            elif proc is None and entry.pid and not _pid_alive(entry.pid):
                # Recovered bot whose process has since died
                entry.status = "stopped"
                entry.stopped_at = datetime.now(timezone.utc)
                changed = True
        if changed:
            self._save_state()

    async def stop_all(self) -> None:
        running = [bid for bid, e in self._bots.items() if e.status == "running"]
        for bot_id in running:
            await self.stop_bot(bot_id)
