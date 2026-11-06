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
    {
        "type": "function",
        "function": {
            "name": "validate_ic_params",
            "description": (
                "Pre-flight validation of proposed iron condor parameters against live chain data "
                "and strategy rules. Call this BEFORE place_iron_condor to catch rule violations "
                "early and avoid wasting a validator round-trip. Returns pass/fail with per-check "
                "detail: delta range, wing width symmetry, per-side credit, and credit equality "
                "(within the configured threshold). Does NOT place any order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "short_call_strike": {"type": "number", "description": "Short call strike."},
                    "long_call_strike": {"type": "number", "description": "Long call strike (> short_call_strike)."},
                    "short_put_strike": {"type": "number", "description": "Short put strike."},
                    "long_put_strike": {"type": "number", "description": "Long put strike (< short_put_strike)."},
                },
                "required": ["short_call_strike", "long_call_strike", "short_put_strike", "long_put_strike"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_premium_balance_candidates",
            "description": (
                "Searches the live 0DTE chain for iron condor configurations where call and put "
                "credits are balanced within the credit equality threshold. Tries wing widths of "
                "25, 30, and 35 points automatically. Returns top_candidate with the best-balanced "
                "strikes ready to place, plus a place_instruction string. "
                "Use this whenever you need to enter a trade: call this tool, then pass "
                "top_candidate strikes directly to place_iron_condor. No manual strike selection "
                "or wing width reasoning needed — the math is done for you. "
                "If candidates_found is 0, use the call-side strikes from get_option_chain and "
                "accept a lower put credit per strategy rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_credit_per_side": {
                        "type": "number",
                        "description": "Desired credit per side in dollars (e.g. 1.50 for $150). Defaults to midpoint of strategy range.",
                    },
                    "wing_width": {
                        "type": "integer",
                        "description": "Wing width in points to start searching (default 30). The tool also tries ±5pt adjustments.",
                        "default": 30,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of candidates to return (default 5).",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_skew_snapshot",
            "description": (
                "Return a concise put/call skew snapshot for the 0DTE chain: implied-vol ratio "
                "(put-wing IV ÷ call-wing IV) and credit asymmetry ratio (put credit ÷ call credit) "
                "at the 10–15 delta band. Use this as CONTEXT ONLY — high skew explains why premiums "
                "are unequal between sides and tells you how much wing-width adjustment is needed. "
                "It is NEVER a reason to skip a trade. When skew is high, use "
                "get_premium_balance_candidates to find wing widths that bring both sides within the "
                "credit equality threshold, or accept a lower put premium per the strategy rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "delta_target": {
                        "type": "number",
                        "description": "Absolute delta to sample for skew measurement (default 0.12).",
                        "default": 0.12,
                    },
                },
                "required": [],
            },
        },
    },
]
