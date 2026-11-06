"""
ORB tool definitions — JSON schemas for tools available to the LLM agent.
"""

from __future__ import annotations

from typing import List, Dict, Any

ORB_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_orb_levels",
        "description": (
            "Fetch the 08:00–08:15 ET 15-min opening range candle and compute "
            "ORB high, low, midline, and range size. Cached for the session day."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_candles",
        "description": (
            "Fetch the last N 5-minute candles for the ES futures symbol. "
            "Used by the regime classifier to assess post-09:30 momentum."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of 5-min candles to return (default 10).",
                    "default": 10,
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_breakout_status",
        "description": (
            "Check whether price has broken above the ORB high (long bias) "
            "or below the ORB low (short bias). Returns bias and current price."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_account_info",
        "description": "Return account balance, net liquidating value, and buying power.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "place_futures_bracket",
        "description": (
            "Place a directional ES/MES futures bracket order (market entry with "
            "embedded TP and SL). `direction` is 'long' or 'short'. Returns order ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "Trade direction: long or short.",
                },
                "entry_price": {
                    "type": "number",
                    "description": "Expected entry price (for logging; actual fill may differ).",
                },
                "tp_price": {
                    "type": "number",
                    "description": "Take-profit price.",
                },
                "sl_price": {
                    "type": "number",
                    "description": "Stop-loss price.",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of contracts (default 1).",
                    "default": 1,
                },
            },
            "required": ["direction", "entry_price", "tp_price", "sl_price"],
        },
    },
    {
        "name": "close_position",
        "description": "Close an open futures position by trade ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "trade_id": {
                    "type": "string",
                    "description": "The trade ID to close.",
                }
            },
            "required": ["trade_id"],
        },
    },
]
