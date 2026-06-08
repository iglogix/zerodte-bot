# src_v2/flux/cache.py
"""SQLite cache for EOD chain snapshots (bounds marketdata.app credit spend)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from src_v2.flux.models import ChainSnapshot, OptionRow


class ChainCache:
    def __init__(self, db_path: str | Path = "data/flux_cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock → safe from the parallel fetch pool.
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS chains "
                            "(underlying TEXT, expiry TEXT, on_date TEXT, payload TEXT, "
                            "PRIMARY KEY(underlying, expiry, on_date))")

    def get(self, underlying: str, expiry: date, on_date: date) -> Optional[ChainSnapshot]:
        with self._lock:
            row = self.db.execute(
                "SELECT payload FROM chains WHERE underlying=? AND expiry=? AND on_date=?",
                (underlying, expiry.isoformat(), on_date.isoformat())).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        rows = tuple(OptionRow(**r) for r in d["rows"])
        return ChainSnapshot(d["underlying"], date.fromisoformat(d["expiry"]),
                             datetime.fromisoformat(d["as_of"]), rows)

    def put(self, snap: ChainSnapshot, on_date: date) -> None:
        payload = json.dumps({
            "underlying": snap.underlying, "expiry": snap.expiry.isoformat(),
            "as_of": snap.as_of.astimezone(timezone.utc).isoformat(),
            "rows": [r.__dict__ for r in snap.rows]})
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO chains VALUES (?,?,?,?)",
                            (snap.underlying, snap.expiry.isoformat(), on_date.isoformat(), payload))
            self.db.commit()
