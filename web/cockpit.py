"""Live P&L Cockpit routes.

Serves the cockpit page and three data endpoints:

* ``GET  /cockpit``            — the single-page app (auth-protected like the rest)
* ``GET  /api/positions``      — open trades grouped by strategy (+ legs, credits)
* ``POST /api/quotes``         — proxy to options_api for live marks; computes and
                                 persists per-position open P&L on every poll
* ``GET  /api/pnl_history``    — persisted intraday P&L samples (seeds the curve)

Quote source and position source are switched by ``COCKPIT_SIM=1`` (the offline
harness) vs. the live bot DB + options_api.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Body
from fastapi.responses import FileResponse, JSONResponse

from web.cockpit_data import (
    group_positions,
    open_pnl_for_position,
    snapshot_request_body,
)

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static" / "cockpit"


def _sim_enabled() -> bool:
    return os.environ.get("COCKPIT_SIM", "").lower() in ("1", "true", "yes")


def _options_api_url() -> str:
    return os.environ.get("OPTIONS_API_URL", "http://127.0.0.1:8101").rstrip("/")


def _options_api_token() -> str:
    return os.environ.get("OPTIONS_API_TOKEN", "")


def _bot_db_path() -> str:
    return os.environ.get("COCKPIT_DB", "data/meic_bot.db")


def _pnl_db_path() -> str:
    return os.environ.get("COCKPIT_PNL_DB", "data/cockpit_pnl.db")


class _State:
    """Lazily-built singletons for the cockpit (sim world, db service, store)."""

    def __init__(self) -> None:
        self._sim = None
        self._db_service = None
        self._store = None

    @property
    def sim(self):
        if self._sim is None:
            from web.cockpit_sim import CockpitSim

            self._sim = CockpitSim()
        return self._sim

    @property
    def store(self):
        if self._store is None:
            from zerodte.services.pnl_snapshot import PnlSnapshotStore

            self._store = PnlSnapshotStore(_pnl_db_path())
        return self._store

    async def open_trades(self) -> List[Dict[str, Any]]:
        if _sim_enabled():
            return self.sim.trades()
        if self._db_service is None:
            from trading_core.services.db_service import DatabaseService

            svc = DatabaseService(_bot_db_path())
            svc.initialize()
            self._db_service = svc
        return await self._db_service.get_open_trades()

    async def fetch_snapshot(self, req_body: Dict[str, Any]) -> Dict[str, Any]:
        if _sim_enabled():
            return self.sim.snapshot(req_body)
        token = _options_api_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"{_options_api_url()}/market-data/options-snapshot"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(url, json=req_body, headers=headers)
            resp.raise_for_status()
            return resp.json()


_state = _State()


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/cockpit")
    async def cockpit_page():
        index = _STATIC_DIR / "index.html"
        if not index.is_file():
            return JSONResponse({"error": "cockpit not built"}, status_code=500)
        # Never cache the HTML, so the browser always loads the current scripts
        # (a stale index.html pinned to old JS was making the "now" payoff
        # curve appear frozen). The JS/CSS are versioned with ?v= for busting.
        return FileResponse(index, headers={"Cache-Control": "no-store, max-age=0"})

    @router.get("/api/positions")
    async def api_positions():
        trades = await _state.open_trades()
        positions = group_positions(trades)
        return {
            "as_of": datetime.now().isoformat(),
            "sim": _sim_enabled(),
            "positions": positions,
        }

    @router.post("/api/quotes")
    async def api_quotes(payload: Dict[str, Any] = Body(...)):
        positions: List[Dict[str, Any]] = payload.get("positions") or []
        if not positions:
            return {"as_of": datetime.now().isoformat(), "marks": {}, "underlying": None, "pnl": []}

        req_body = snapshot_request_body(positions)
        try:
            snapshot = await _state.fetch_snapshot(req_body)
        except httpx.HTTPError as exc:
            log.warning("options_api snapshot failed: %s", exc)
            return JSONResponse({"error": f"quote source unavailable: {exc}"}, status_code=502)

        # Index legs by symbol -> full quote record.
        marks: Dict[str, Dict[str, Any]] = {}
        for leg in snapshot.get("legs", []):
            marks[leg["symbol"]] = leg
        mark_prices = {
            sym: (q.get("mark") if q.get("mark") is not None else q.get("mid"))
            for sym, q in marks.items()
        }

        underlyings = snapshot.get("underlyings", {})
        und = underlyings.get("SPX") or (next(iter(underlyings.values()), None))
        und_price = (und or {}).get("last") or (und or {}).get("mid")

        # A position contributes P&L only once it has entered. In sim mode the
        # snapshot reports which trades are active at the current session time;
        # in live mode every open trade is active.
        sim_block = snapshot.get("sim") or {}
        active_ids = sim_block.get("active_trade_ids")
        active_set = set(active_ids) if active_ids is not None else None

        # Per-position open P&L (authoritative; persisted for the curve).
        pnl: List[Dict[str, Any]] = []
        samples: List[Dict[str, Any]] = []
        for pos in positions:
            if active_set is not None and pos["trade_id"] not in active_set:
                val = None
            else:
                val = open_pnl_for_position(pos, mark_prices)
            entry = {
                "trade_id": pos["trade_id"],
                "strategy": pos.get("strategy", "Other"),
                "open_pnl": val,
                "max_profit": pos.get("max_profit"),
            }
            pnl.append(entry)
            if val is not None:
                samples.append(
                    {
                        "trade_id": pos["trade_id"],
                        "strategy": pos.get("strategy", "Other"),
                        "open_pnl": val,
                        "underlying_price": und_price,
                    }
                )
        # Persist in the snapshot's own time (simulated session time in sim mode)
        # so the seeded curve and the live curve share one timeline.
        ts = None
        try:
            ts = datetime.fromisoformat(snapshot["as_of"])
        except (KeyError, TypeError, ValueError):
            ts = None
        if samples:
            try:
                _state.store.record_many(samples, ts=ts)
            except Exception:
                log.exception("Failed to persist P&L snapshot")

        return {
            "as_of": snapshot.get("as_of", datetime.now().isoformat()),
            "underlying": und,
            "marks": marks,
            "pnl": pnl,
            "sim": snapshot.get("sim"),
            "market_regime": snapshot.get("market_regime"),
        }

    @router.post("/api/sim/reset")
    async def api_sim_reset():
        """Restart the simulated session from the open and clear today's history."""
        if not _sim_enabled():
            return JSONResponse({"error": "not in sim mode"}, status_code=400)
        _state.sim.reset()
        try:
            _state.store.clear_for_date(date.today())
        except Exception:
            log.exception("Failed to clear P&L history on reset")
        return {"ok": True, "minute": _state.sim.sim_minute}

    @router.get("/api/pnl_history")
    async def api_pnl_history(d: Optional[str] = None):
        day = date.fromisoformat(d) if d else date.today()
        try:
            rows = _state.store.series_for_date(day)
        except Exception:
            log.exception("Failed to read P&L history")
            rows = []
        return {"date": day.isoformat(), "samples": rows}

    return router
