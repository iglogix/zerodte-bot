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
    """Per-strategy execution records.

    `strategy` namespaces the output as
    `<base_dir>/execution_comparison/<strategy>/<date>.jsonl`. Without it, two
    bots sharing a `log_dir` (MEIC and ORB both default to "logs") interleave
    into one file, and the per-strategy reports then read each other's rows.
    Separation is enforced here rather than by giving every config a distinct
    log_dir, so a config that forgets cannot silently re-merge them.

    Passing no strategy keeps the flat legacy path, so historical files and
    anything reading them still work.
    """

    def __init__(self, base_dir: str = "logs", strategy: str = ""):
        self._strategy = strategy or ""
        root = Path(base_dir) / "execution_comparison"
        self._dir = root / self._strategy if self._strategy else root

    def _path(self) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{date.today().isoformat()}.jsonl"

    def _append(self, record: Dict[str, Any]) -> None:
        record.setdefault("ts", datetime.now().isoformat())
        if self._strategy:
            record.setdefault("strategy", self._strategy)
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
            "capreq_usd": payload.get("capreq_usd"),
            "stop_amount": payload.get("stop_amount"),
            "legs": {
                "short_put": {"symbol": put_short_sym,
                              "assumed_credit": put.get("credit"),
                              "short_mid": put.get("short_mid"),
                              "short_strike": put.get("short_strike"),
                              "long_strike": put.get("long_strike"),
                              "stop_trigger": put.get("stop_trigger")},
                "short_call": {"symbol": call_short_sym,
                               "assumed_credit": call.get("credit"),
                               "short_mid": call.get("short_mid"),
                               "short_strike": call.get("short_strike"),
                               "long_strike": call.get("long_strike"),
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

    def read_today_assumed(self) -> List[Dict[str, Any]]:
        """Return today's logged "assumed" entry records (for restart rehydration).

        Reading back our own log lets the QuoteComparator rebuild its in-memory
        tracking after a restart instead of losing open tranches.
        """
        path = self._dir / f"{date.today().isoformat()}.jsonl"
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "assumed":
                out.append(rec)
        return out

    def log_entry_fill(
        self,
        trade_id: str,
        side: str,
        strikes: Dict[str, Any],
        short_quote: Dict[str, Any],
        long_quote: Dict[str, Any],
        modelled_credit: float,
        result: Dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        """Counterfactual record for one entry attempt — filled or not.

        This is what turns the paper run into evidence about the open question
        ("can this fill at <=$0.25/spread?") instead of just a P&L curve. It
        pins, at the moment of the retest touch: the raw leg quotes, the three
        reference prices the decision sits between, what the backtest assumed,
        which ladder rung filled, and what that cost.

        `natural` is the immediate-execution price (sell the short at its bid,
        buy the long at its ask) — the credit obtainable by crossing outright.
        Recording no-fills matters as much as fills: a strategy that cannot get
        filled at the modelled price is a finding, not a gap in the data.
        """
        def _q(q):
            b, a = (q or {}).get("bid"), (q or {}).get("ask")
            b = None if b is None else float(b)
            a = None if a is None else float(a)
            mid = None if (b is None or a is None) else round((b + a) / 2, 3)
            spread = None if (b is None or a is None) else round(a - b, 3)
            return {"bid": b, "ask": a, "mid": mid, "spread": spread}

        s, l = _q(short_quote), _q(long_quote)
        mid_credit = (None if (s["mid"] is None or l["mid"] is None)
                      else round(s["mid"] - l["mid"], 3))
        natural = (None if (s["bid"] is None or l["ask"] is None)
                   else round(s["bid"] - l["ask"], 3))

        self._append({
            "type": "entry_fill",
            "trade_id": trade_id,
            "side": side,
            "strikes": strikes,
            "short_quote": s,
            "long_quote": l,
            # Reference prices the entry decision sits between.
            "mid_credit": mid_credit,
            "natural_credit": natural,
            "modelled_credit": modelled_credit,   # what the backtest assumed
            # Outcome.
            "filled": bool(result.get("filled")),
            "fill_credit": result.get("fill_credit"),
            "rung": result.get("rung"),
            "requested_credit": result.get("requested_credit"),
            "slippage_per_spread": result.get("slippage_per_spread"),
            "time_to_fill_seconds": round(elapsed_seconds, 2),
            "attempts": result.get("attempts"),
            "order_id": result.get("order_id"),
        })

    def log_exit_fill(
        self,
        trade_id: str,
        symbol: str,
        reason: str,
        quantity: int,
        exit_quote: Optional[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> None:
        """What one leg actually cost to close, against its quote at that moment.

        The counterpart to log_entry_fill. Exit realism is the load-bearing
        unknown for this strategy — the WF study priced exits at MID and its own
        notes flag breach-exit fills as needing a live study — so a fill with no
        record of the quote it crossed answers nothing.

        Emitted for every close attempt including no-ops (a leg already flat),
        because a close that did not happen is itself a finding.
        """
        bid = (exit_quote or {}).get("bid")
        ask = (exit_quote or {}).get("ask")
        bid = None if bid is None else float(bid)
        ask = None if ask is None else float(ask)
        mid = None if (bid is None or ask is None) else round((bid + ask) / 2, 4)

        fill = result.get("fill_price")
        fill = None if fill in (None, "") else float(fill)
        # Closing a SHORT leg means buying it back: paying above mid is the cost.
        # Sign is left to the reader — role is not known at this layer — so the
        # raw distance is recorded and interpreted per leg.
        slip = None if (fill is None or mid is None) else round(fill - mid, 4)

        self._append({
            "type": "exit_fill",
            "trade_id": trade_id,
            "symbol": symbol,
            "reason": reason,
            "quantity": quantity,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": None if (bid is None or ask is None) else round(ask - bid, 4),
            "fill_price": fill,
            "fill_vs_mid": slip,
            "filled": bool(result.get("filled")),
            "skipped": bool(result.get("skipped")),
            "status": result.get("status"),
            "order_id": result.get("order_id"),
        })

    def log_quote(self, trade_id: str, leg: Dict[str, Any],
                  bid: float, ask: float) -> None:
        """Record a short leg's real bid/ask vs its stop trigger (manual mode).

        `leg` is a registered leg dict: {symbol, role, trigger, ...}.
        Used to measure where the short's real ask is when it crosses the
        credit-relative trigger — i.e. where a stop would actually fill.
        """
        trigger = leg.get("trigger")
        ask_f = None if ask is None else float(ask)
        self._append({
            "type": "quote", "trade_id": trade_id,
            "symbol": leg.get("symbol"), "role": leg.get("role"),
            "real_bid": None if bid is None else float(bid),
            "real_ask": ask_f,
            "trigger": trigger,
            "crossed": bool(trigger is not None and ask_f is not None and ask_f >= trigger),
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


def compare_quotes(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (trade_id, role) short leg: did its real ask cross the stop trigger,
    and by how much at the first crossing (the real overshoot a stop-market
    would suffer vs the assumed trigger). One row per short leg seen.

    Pure function over "quote" records logged by the QuoteComparator.
    """
    legs: Dict[tuple, Dict[str, Any]] = {}
    for r in sorted((x for x in records if x.get("type") == "quote"),
                    key=lambda x: x.get("ts", "")):
        key = (r.get("trade_id"), r.get("role"))
        leg = legs.setdefault(key, {
            "trade_id": r.get("trade_id"), "role": r.get("role"),
            "symbol": r.get("symbol"), "trigger": r.get("trigger"),
            "polls": 0, "crossed": False,
            "first_cross_ask": None, "first_cross_ts": None, "max_ask": None,
        })
        leg["polls"] += 1
        ask = r.get("real_ask")
        if ask is not None:
            leg["max_ask"] = ask if leg["max_ask"] is None else max(leg["max_ask"], ask)
        if r.get("crossed") and not leg["crossed"]:
            leg["crossed"] = True
            leg["first_cross_ask"] = ask
            leg["first_cross_ts"] = r.get("ts")

    rows = []
    for leg in legs.values():
        trig = leg["trigger"]
        overshoot = None
        if leg["crossed"] and leg["first_cross_ask"] is not None and trig is not None:
            overshoot = round(leg["first_cross_ask"] - trig, 2)
        rows.append({**leg, "overshoot_at_cross": overshoot})
    return rows


def _classify(slip: float | None) -> str:
    if slip is None:
        return "unknown"
    if slip <= 0.05:
        return "trigger/short_trigger (fill ~at trigger)"
    if slip <= 0.50:
        return "mild slippage past trigger"
    return "combo-like (deep past trigger)"
