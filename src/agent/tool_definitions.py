"""
Tool definitions for the LLM agent.
These are passed to the LLM as available tools it can call.
"""

from typing import Any, Dict, List

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_spx_price",
            "description": "Get the current SPX index price. Returns cached value if fresh (< 60s old), otherwise fetches live.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_option_chain",
            "description": "Get the 0DTE SPX option chain with Greeks (delta, gamma, theta, vega), bid/ask, volume. Optionally filter by delta range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expiration": {
                        "type": "string",
                        "description": "Expiration filter. Use '0DTE' for today's expiration.",
                        "default": "0DTE",
                    },
                    "min_delta": {
                        "type": "number",
                        "description": "Minimum absolute delta value (e.g. 0.10 for 10-delta).",
                    },
                    "max_delta": {
                        "type": "number",
                        "description": "Maximum absolute delta value (e.g. 0.15 for 15-delta).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_5min_candles",
            "description": "Get the last N 5-minute SPX candles (OHLCV). Used to assess market stability before entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of candles to return (most recent N candles).",
                        "default": 6,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_info",
            "description": "Get current account equity, buying power, and cash balance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_open_positions",
            "description": "Get all currently open iron condor positions with their P&L and status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_buying_power",
            "description": "Check if sufficient buying power is available for a new trade. Enforces 50% account limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "required_amount": {
                        "type": "number",
                        "description": "Buying power required for the new trade in dollars.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_iron_condor",
            "description": "Place a full iron condor (4 legs: short call, long call, short put, long put). Validates strike order before execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short_call_strike": {
                        "type": "number",
                        "description": "Strike price of the short call.",
                    },
                    "long_call_strike": {
                        "type": "number",
                        "description": "Strike price of the long call (must be > short_call_strike).",
                    },
                    "short_put_strike": {
                        "type": "number",
                        "description": "Strike price of the short put.",
                    },
                    "long_put_strike": {
                        "type": "number",
                        "description": "Strike price of the long put (must be < short_put_strike).",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of contracts.",
                        "default": 1,
                    },
                },
                "required": ["short_call_strike", "long_call_strike", "short_put_strike", "long_put_strike"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_credit_spread",
            "description": "Place a single credit spread (call or put side).",
            "parameters": {
                "type": "object",
                "properties": {
                    "spread_type": {
                        "type": "string",
                        "enum": ["call", "put"],
                        "description": "Type of spread.",
                    },
                    "short_strike": {
                        "type": "number",
                        "description": "Strike price of the short leg.",
                    },
                    "long_strike": {
                        "type": "number",
                        "description": "Strike price of the long leg.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of contracts.",
                        "default": 1,
                    },
                },
                "required": ["spread_type", "short_strike", "long_strike"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_stop_order",
            "description": "Place an OCO stop-loss order for a short option position. Creates stop-limit (40pt gap) + stop-market (30pt further) pair.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Option symbol to protect (e.g. SPXW240222C5890).",
                    },
                    "stop_loss_amount": {
                        "type": "number",
                        "description": "Total premium collected (stop triggers when position reaches this loss).",
                    },
                    "stop_limit_gap": {
                        "type": "integer",
                        "description": "Gap in points between stop trigger and limit price.",
                        "default": 40,
                    },
                    "stop_market_gap": {
                        "type": "integer",
                        "description": "Additional gap in points for the backup stop-market order.",
                        "default": 30,
                    },
                },
                "required": ["symbol", "stop_loss_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_position",
            "description": "Close an open position or specific leg. Use trade_id + close_all=true to close entire iron condor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Option symbol to close.",
                    },
                    "trade_id": {
                        "type": "string",
                        "description": "Trade ID to close entirely.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of contracts to close.",
                        "default": 1,
                    },
                    "order_type": {
                        "type": "string",
                        "enum": ["market", "limit"],
                        "description": "Order type for closing.",
                        "default": "market",
                    },
                    "close_all": {
                        "type": "boolean",
                        "description": "Close all legs of the trade.",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a pending order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to cancel.",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
]
