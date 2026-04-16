"""Log streamer: tail bot logs and filter to meaningful lines."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Callable, List, Optional

# Meaningful INFO patterns — cycle outcomes, trades, positions, lifecycle events
_MEANINGFUL_RE = re.compile(
    r"(cycle complete|outcome=|trade filled|order.*(filled|placed|cancel)"
    r"|position.*(open|close|expir|start)|bot init|initialized|shutdown"
    r"|session skip|entry watch|breakout|stop.*(trigger|hit|loss)"
    r"|strategy.*(created|loaded)|authenticated|monitor.*start|listener"
    r"|mode[=:]|schedule|market.*(open|close|hour)|no.*(market|session)"
    r"|connect|disconnect|error|fail|retry|timeout)",
    re.IGNORECASE,
)


class LogFilter:
    """Filter log lines to ERROR, WARNING, and meaningful INFO."""

    def accept(self, line: str) -> bool:
        """Return True if the line should be shown to the user."""
        if "[ERROR]" in line or "[WARNING]" in line:
            return True
        if "[DEBUG]" in line:
            return False
        if "[INFO]" in line:
            return bool(_MEANINGFUL_RE.search(line))
        # Unknown format — accept to be safe
        return True

    def classify(self, line: str) -> str:
        """Return 'error', 'warning', or 'info'."""
        if "[ERROR]" in line:
            return "error"
        if "[WARNING]" in line:
            return "warning"
        return "info"


def tail_lines(file_path: str, n: int = 50) -> List[str]:
    """Read the last *n* lines from *file_path*. Returns [] if missing."""
    try:
        with open(file_path, "r") as fh:
            lines = fh.readlines()
    except (FileNotFoundError, OSError):
        return []
    # Strip trailing newlines, take last n non-empty-after-strip
    stripped = [l.rstrip("\n") for l in lines if l.strip()]
    return stripped[-n:]


class LogTailer:
    """Continuously tail a log file, invoking a callback for accepted lines."""

    def __init__(
        self,
        file_path: str,
        log_filter: Optional[LogFilter] = None,
    ) -> None:
        self.file_path = file_path
        self.filter = log_filter or LogFilter()
        self._running = False

    async def tail(self, callback: Callable) -> None:
        """Tail the file, calling ``await callback(line, level)`` for each accepted line.

        Polls every 0.5 s. Handles file truncation / rotation by resetting
        the read offset when the file shrinks.
        """
        self._running = True
        pos = 0
        inode: Optional[int] = None

        # Start from beginning so we catch output from fast-exiting bots
        try:
            stat = os.stat(self.file_path)
            inode = stat.st_ino
        except OSError:
            pass

        while self._running:
            try:
                stat = os.stat(self.file_path)
            except OSError:
                await asyncio.sleep(0.5)
                continue

            # Detect rotation (new inode) or truncation (smaller size)
            if stat.st_ino != inode or stat.st_size < pos:
                pos = 0
                inode = stat.st_ino

            if stat.st_size > pos:
                with open(self.file_path, "r") as fh:
                    fh.seek(pos)
                    new_data = fh.read()
                    pos = fh.tell()

                for line in new_data.splitlines():
                    if self.filter.accept(line):
                        level = self.filter.classify(line)
                        await callback(line, level)

            await asyncio.sleep(0.5)

    def stop(self) -> None:
        """Signal the tail loop to exit."""
        self._running = False
