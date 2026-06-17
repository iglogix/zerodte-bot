"""Intraday P&L snapshot store for the Live P&L Cockpit.

Persists a time series of per-trade open P&L so the cockpit's intraday curve
survives a page refresh and can be seeded on load. One row per (trade, sample).

The table lives on the shared SQLAlchemy ``Base`` so it is created alongside the
rest of the schema, but the store opens its own engine so the web app can record
and query snapshots without depending on the running bot process.
"""

from __future__ import annotations

from datetime import date, datetime
from datetime import time as dt_time
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String

from src_v2.models.database import Base, get_engine, get_session


class PnlSnapshot(Base):
    """A single open-P&L sample for one trade at one instant."""

    __tablename__ = "pnl_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, nullable=False, index=True)
    trade_id = Column(String(50), nullable=False, index=True)
    strategy = Column(String(40), nullable=False, index=True)
    open_pnl = Column(Float, nullable=False)
    underlying_price = Column(Float, nullable=True)


class PnlSnapshotStore:
    """Record and query intraday P&L samples in a SQLite file."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._engine = get_engine(db_path)
        # create_all is a no-op for tables that already exist, so this only
        # creates pnl_snapshots without touching the bot's other tables.
        PnlSnapshot.__table__.create(self._engine, checkfirst=True)

    def record(
        self,
        trade_id: str,
        strategy: str,
        open_pnl: float,
        underlying_price: Optional[float] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        """Persist one P&L sample."""
        with get_session(self._engine) as session:
            session.add(
                PnlSnapshot(
                    ts=ts or datetime.now(),
                    trade_id=trade_id,
                    strategy=strategy,
                    open_pnl=float(open_pnl),
                    underlying_price=(
                        None if underlying_price is None else float(underlying_price)
                    ),
                )
            )

    def record_many(self, samples: List[Dict[str, Any]], ts: Optional[datetime] = None) -> int:
        """Persist a batch of samples sharing one timestamp. Returns count written."""
        stamp = ts or datetime.now()
        rows = [
            PnlSnapshot(
                ts=stamp,
                trade_id=str(s["trade_id"]),
                strategy=str(s.get("strategy", "")),
                open_pnl=float(s["open_pnl"]),
                underlying_price=(
                    None if s.get("underlying_price") is None else float(s["underlying_price"])
                ),
            )
            for s in samples
        ]
        if not rows:
            return 0
        with get_session(self._engine) as session:
            session.add_all(rows)
        return len(rows)

    def series_for_date(self, day: Optional[date] = None) -> List[Dict[str, Any]]:
        """Return all samples for ``day`` (default today), ordered by timestamp.

        Each item: ``{ts (iso), trade_id, strategy, open_pnl, underlying_price}``.
        The cockpit sums these per timestamp (optionally filtered by strategy) to
        seed the intraday curve.
        """
        day = day or date.today()
        start = datetime.combine(day, dt_time(0, 0, 0))
        end = datetime.combine(day, dt_time(23, 59, 59, 999999))
        with get_session(self._engine) as session:
            rows = (
                session.query(PnlSnapshot)
                .filter(PnlSnapshot.ts >= start, PnlSnapshot.ts <= end)
                .order_by(PnlSnapshot.ts.asc(), PnlSnapshot.id.asc())
                .all()
            )
            return [
                {
                    "ts": r.ts.isoformat(),
                    "trade_id": r.trade_id,
                    "strategy": r.strategy,
                    "open_pnl": r.open_pnl,
                    "underlying_price": r.underlying_price,
                }
                for r in rows
            ]
