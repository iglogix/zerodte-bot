"""Cockpit data layer — turn bot DB trades into cockpit positions + quote requests.

Strategy grouping lives only in the bot DB: the ``trade_id`` prefix names the
strategy and the ``strategy_data`` JSON (merged onto the trade dict by
``DatabaseService._trade_to_dict``) holds the leg symbols, per-side credits and
open quantities. This module is the single place that knows those conventions.

Pure functions only (no I/O), so they are trivially unit-testable.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

# trade_id prefix -> human strategy name
STRATEGY_LABELS: Dict[str, str] = {
    "meic": "MEIC",
    "ph": "Power Hour",
    "orbdir": "ORB Directional",
    "ic": "Iron Condor",
}

_MULTIPLIER = 100  # SPX/SPXW option contract multiplier


def strategy_label(trade_id: str) -> str:
    """Map a trade_id to its strategy name via the leading prefix."""
    tid = (trade_id or "").lower()
    for prefix, label in STRATEGY_LABELS.items():
        if tid.startswith(prefix + "-"):
            return label
    return "Other"


def parse_osi_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Parse a 21-char OSI symbol into root/expiration/right/strike, else None."""
    if not symbol or len(symbol) != 21:
        return None
    try:
        root = symbol[:6].strip()
        yy, mm, dd = int(symbol[6:8]), int(symbol[8:10]), int(symbol[10:12])
        expiration = date(2000 + yy, mm, dd)
        right = "call" if symbol[12] == "C" else "put"
        strike = int(symbol[13:21]) / 1000.0
        return {"root": root, "expiration": expiration, "right": right, "strike": strike}
    except (ValueError, IndexError):
        return None


def _leg_from_symbol(symbol: str, side: str, right: str, qty: int) -> Dict[str, Any]:
    """Build one leg dict, parsing strike/expiration/right from the OSI symbol."""
    parsed = parse_osi_symbol(symbol) or {}
    return {
        "symbol": symbol,
        "side": side,  # "short" (sold) | "long" (bought)
        "right": parsed.get("right", right),
        "strike": parsed.get("strike"),
        "expiration": (parsed.get("expiration") or None),
        "qty": int(qty),
        "multiplier": _MULTIPLIER,
    }


def build_position(trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn one open-trade dict into a cockpit position (spreads + flat legs).

    Returns None if the trade exposes no recognisable leg symbols.
    """
    trade_id = trade.get("trade_id", "")
    sd = trade  # strategy_data is already merged onto the trade dict

    spreads: List[Dict[str, Any]] = []
    legs: List[Dict[str, Any]] = []

    for right in ("put", "call"):
        short_sym = sd.get(f"short_{right}_symbol")
        long_sym = sd.get(f"long_{right}_symbol")
        if not short_sym and not long_sym:
            continue
        qty = int(sd.get(f"{right}_open_qty") or sd.get("quantity") or 1)
        credit = sd.get(f"{right}_credit")
        if credit is None:
            credit = trade.get(f"{right}_spread_premium")
        spread: Dict[str, Any] = {
            "right": right,
            "credit": float(credit or 0.0),
            "qty": qty,
            "short": None,
            "long": None,
        }
        if short_sym:
            leg = _leg_from_symbol(short_sym, "short", right, qty)
            spread["short"] = leg
            legs.append(leg)
        if long_sym:
            leg = _leg_from_symbol(long_sym, "long", right, qty)
            spread["long"] = leg
            legs.append(leg)
        spreads.append(spread)

    if not legs:
        return None

    entry_credit = float(
        sd.get("strangle_credit") or trade.get("total_premium") or sum(s["credit"] for s in spreads)
    )
    expiration = trade.get("expiration_date")
    if isinstance(expiration, (date, datetime)):
        expiration = expiration.isoformat()[:10]

    # Max profit = full credit kept (per side credit * qty * multiplier).
    max_profit = sum(s["credit"] * s["qty"] * _MULTIPLIER for s in spreads)

    return {
        "trade_id": trade_id,
        "strategy": strategy_label(trade_id),
        "underlying": trade.get("underlying", "SPX"),
        "expiration": expiration,
        "entry_time": (
            trade["entry_time"].isoformat()
            if isinstance(trade.get("entry_time"), (date, datetime))
            else trade.get("entry_time")
        ),
        "entry_credit": entry_credit,
        "max_profit": max_profit,
        "stop_dollars": sd.get("stop_dollars") or trade.get("stop_loss_amount"),
        "spreads": spreads,
        "legs": legs,
    }


def group_positions(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build cockpit positions from open trades, skipping unrecognisable ones."""
    out: List[Dict[str, Any]] = []
    for t in trades:
        pos = build_position(t)
        if pos is not None:
            out.append(pos)
    return out


def all_legs(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten every leg across positions (for a quote request)."""
    legs: List[Dict[str, Any]] = []
    for p in positions:
        legs.extend(p.get("legs", []))
    return legs


def snapshot_request_body(
    positions: List[Dict[str, Any]],
    account_id: str = "cockpit",
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the options_api /market-data/options-snapshot request body.

    Maps position side -> order side (short=sell, long=buy) as the snapshot API
    expects, and dedupes underlyings.
    """
    legs_out: List[Dict[str, Any]] = []
    underlyings: List[str] = []
    for p in positions:
        und = p.get("underlying", "SPX")
        if und not in underlyings:
            underlyings.append(und)
        for leg in p.get("legs", []):
            exp = leg.get("expiration")
            if isinstance(exp, (date, datetime)):
                exp = exp.isoformat()[:10]
            legs_out.append(
                {
                    "symbol": leg["symbol"],
                    "asset_type": "option",
                    "underlying": und,
                    "expiration": exp,
                    "right": leg["right"],
                    "strike": leg.get("strike") or 0.0,
                    "quantity": leg.get("qty", 1),
                    "side": "sell" if leg["side"] == "short" else "buy",
                    "multiplier": leg.get("multiplier", _MULTIPLIER),
                }
            )
    return {
        "as_of": (as_of or datetime.now()).isoformat(),
        "account": {"broker": "tastytrade", "account_id": account_id},
        "underlyings": underlyings or ["SPX"],
        "legs": legs_out,
        "include": {
            "quotes": True,
            "greeks": True,
            "open_interest": False,
            "volume": False,
            "underlying_quotes": True,
            "market_regime": True,
        },
    }


def open_pnl_for_position(position: Dict[str, Any], marks: Dict[str, float]) -> Optional[float]:
    """Compute one position's open P&L in dollars from per-symbol marks.

    For each credit spread: pnl = (credit - (short_mark - long_mark)) * qty * mult.
    A long-only or short-only leg is handled by treating the missing leg as 0.
    Returns None if no required mark is available (so the caller can flag stale).
    """
    total = 0.0
    seen = False
    for spread in position.get("spreads", []):
        short_leg = spread.get("short")
        long_leg = spread.get("long")
        short_mark = marks.get(short_leg["symbol"]) if short_leg else 0.0
        long_mark = marks.get(long_leg["symbol"]) if long_leg else 0.0
        if short_leg and short_mark is None:
            continue
        if long_leg and long_mark is None:
            continue
        seen = True
        cost_to_close = (short_mark or 0.0) - (long_mark or 0.0)
        total += (spread["credit"] - cost_to_close) * spread["qty"] * _MULTIPLIER
    return total if seen else None
