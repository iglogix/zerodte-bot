"""Execution-comparison log: records assumed-vs-actual fills for paper trading.

Two producers append JSON lines to logs/execution_comparison/<date>.jsonl:
  - entry executor → "assumed" records (the mids/credit we expected, and the
    short-leg stop trigger prices) keyed by OCC leg symbol.
  - OrderFillListener → "fill" records (actual TT fill price + order type).

The report (`compare`) joins them by symbol so we can quantify how real fills
deviate from the backtest's assumptions — especially the stop fill vs the
credit-relative trigger that the backtest modeled.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ExecutionLog:
    def __init__(self, base_dir: str = "logs"):
        self._dir = Path(base_dir) / "execution_comparison"

    def _path(self) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{date.today().isoformat()}.jsonl"

    def _append(self, record: Dict[str, Any]) -> None:
        record.setdefault("ts", datetime.now().isoformat())
        try:
            with open(self._path(), "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("Failed to append execution-comparison record")

    def log_assumed_entry(self, trade_id: str, payload: Dict[str, Any],
                          put_short_sym: str, call_short_sym: str) -> None:
        """Record the assumed entry credits and short-leg stop triggers.

        `payload` is the meic_two_spreads order payload (build_order output).
        """
        put = payload.get("put_spread", {})
        call = payload.get("call_spread", {})
        self._append({
            "type": "assumed", "trade_id": trade_id, "kind": "entry",
            "quantity": payload.get("quantity"),
            "total_credit": payload.get("total_credit"),
            "stop_amount": payload.get("stop_amount"),
            "legs": {
                "short_put": {"symbol": put_short_sym,
                              "assumed_credit": put.get("credit"),
                              "short_mid": put.get("short_mid"),
                              "stop_trigger": put.get("stop_trigger")},
                "short_call": {"symbol": call_short_sym,
                               "assumed_credit": call.get("credit"),
                               "short_mid": call.get("short_mid"),
                               "stop_trigger": call.get("stop_trigger")},
            },
        })

    def log_fill(self, symbol: str, fill_price: float, order_type: str,
                 is_stop_fill: bool, quantity: int = 1) -> None:
        self._append({
            "type": "fill", "symbol": symbol, "fill_price": float(fill_price),
            "order_type": order_type, "is_stop_fill": bool(is_stop_fill),
            "quantity": int(quantity),
        })


def compare(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join assumed short-leg records to their stop fills by symbol.

    Returns one row per matched stop fill with the slippage vs the assumed
    trigger and which backtest stop model that implies.
    """
    # symbol -> assumed short-leg context
    assumed: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if r.get("type") == "assumed":
            for role, leg in r.get("legs", {}).items():
                sym = leg.get("symbol")
                if sym:
                    assumed[sym] = {"trade_id": r.get("trade_id"), "role": role, **leg}

    rows: List[Dict[str, Any]] = []
    for r in records:
        if r.get("type") != "fill" or not r.get("is_stop_fill"):
            continue
        ctx = assumed.get(r["symbol"])
        if ctx is None:
            rows.append({"symbol": r["symbol"], "match": "unmatched",
                         "actual_fill": r["fill_price"]})
            continue
        trigger = ctx.get("stop_trigger")
        actual = r["fill_price"]
        slip = (actual - trigger) if trigger is not None else None
        rows.append({
            "trade_id": ctx.get("trade_id"), "role": ctx.get("role"),
            "symbol": r["symbol"], "match": "matched",
            "stop_trigger": trigger, "actual_fill": actual,
            "slippage": None if slip is None else round(slip, 2),
            # The backtest's "trigger"/"short_trigger" models assume fill ~= trigger;
            # a large positive slippage means reality looked like the "combo" model.
            "implied_model": _classify(slip),
        })
    return rows


def _classify(slip: float | None) -> str:
    if slip is None:
        return "unknown"
    if slip <= 0.05:
        return "trigger/short_trigger (fill ~at trigger)"
    if slip <= 0.50:
        return "mild slippage past trigger"
    return "combo-like (deep past trigger)"
