"""Bot subprocess lifecycle manager."""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

log = logging.getLogger(__name__)


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


class BotManager:
    """Manages bot subprocesses."""

    def __init__(self) -> None:
        self._bots: dict[str, _BotEntry] = {}

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
        bot_id = uuid.uuid4().hex[:8]
        bot_name = name or self._derive_name(config_path)

        # Prepare log directory and file
        log_dir = Path("logs/web_bots")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{bot_name}_{bot_id}.log"

        # Build command
        cmd = [sys.executable, "-m", "src_v2.main", f"--config={config_path}"]
        if mode and mode != "dry_run":
            cmd.append(f"--{mode}")

        log_fh = open(log_path, "w")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
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
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

        entry.status = "stopped"
        entry.stopped_at = datetime.now(timezone.utc)
        if entry._log_fh:
            entry._log_fh.close()
            entry._log_fh = None

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
        for entry in self._bots.values():
            if entry.status != "running":
                continue
            proc = entry.process
            if proc is not None and proc.returncode is not None:
                entry.status = "error"
                entry.error = f"Process exited with code {proc.returncode}"
                entry.stopped_at = datetime.now(timezone.utc)
                if entry._log_fh:
                    entry._log_fh.close()
                    entry._log_fh = None

    async def stop_all(self) -> None:
        running = [bid for bid, e in self._bots.items() if e.status == "running"]
        for bot_id in running:
            await self.stop_bot(bot_id)
