"""Cost estimator + spend ledger for Databento backfill.

Wraps metadata.get_cost (free to call) so the driver can dry-run the entire
manifest before fetching anything, and tracks per-call spend during live runs
to a JSONL ledger that survives across restarts.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, List, Optional


def estimate_request_cost(
    client: Any,
    symbols: List[str],
    start: str,
    end: str,
    dataset: str = "OPRA.PILLAR",
    schema: str = "cbbo-1m",
    stype_in: str = "raw_symbol",
) -> float:
    """Returns the expected cost in USD for an identical-shape get_range request.

    Calling metadata.get_cost is free per Databento docs.
    """
    return float(client.metadata.get_cost(
        dataset=dataset,
        schema=schema,
        symbols=symbols,
        start=start,
        end=end,
        stype_in=stype_in,
    ))


class CostLedger:
    """Append-only JSONL log of per-day API costs.

    Records "live" and "dry_run" entries separately so dry-run costs never
    inflate the live total.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._records: List[dict] = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def record(
        self,
        day: date,
        mode: str,
        cost_usd: float,
        n_symbols: int,
        n_quotes: int,
    ) -> None:
        rec = {
            "date": day.isoformat(),
            "mode": mode,
            "cost_usd": float(cost_usd),
            "n_symbols": int(n_symbols),
            "n_quotes": int(n_quotes),
        }
        self._records.append(rec)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def total_usd(self, mode: Optional[str] = "live") -> float:
        if mode is None:
            return sum(r["cost_usd"] for r in self._records)
        return sum(r["cost_usd"] for r in self._records if r["mode"] == mode)

    def n_records(self, mode: Optional[str] = None) -> int:
        if mode is None:
            return len(self._records)
        return sum(1 for r in self._records if r["mode"] == mode)
