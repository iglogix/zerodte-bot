"""Order submission for Flux — combo (default) or leg-in execution, mode-gated.

Safety: `manual` mode NEVER submits (ticket only). Submission is an explicit user action
in `sandbox` (cert) or `live` (real money). The arrangement logic (which legs, what order)
is pure/testable; TastyTradeSubmitter is the only thing that talks to the broker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

from src_v2.flux.orders import OrderLeg, OrderTicket

LegSpec = Tuple[str, str, int]   # (osi, action, qty)


@dataclass
class SubmitResult:
    ok: bool
    order_ids: List[str] = field(default_factory=list)
    detail: str = ""


def combo_legs(ticket: OrderTicket) -> List[LegSpec]:
    """All legs for ONE complex order (atomic, no legging risk)."""
    return [(l.osi, l.action, l.quantity) for l in ticket.legs]


def legin_groups(ticket: OrderTicket) -> List[List[OrderLeg]]:
    """Leg-in order: BUY the longs first (never naked-short), THEN SELL the shorts.
    Each leg carries its own mid (`limit`) so it's worked at mid."""
    longs = [l for l in ticket.legs if l.is_long]
    shorts = [l for l in ticket.legs if not l.is_long]
    return [longs, shorts]


class OrderSubmitter(ABC):
    def __init__(self, mode: str = "manual"):
        self.mode = mode

    @abstractmethod
    def submit(self, ticket: OrderTicket, execution: str = "combo") -> SubmitResult:
        ...


class MockSubmitter(OrderSubmitter):
    """Records submissions instead of routing them — for tests and a non-broker preview."""
    def __init__(self, mode: str = "sandbox"):
        super().__init__(mode)
        self.calls: List[Tuple[OrderTicket, str]] = []

    def submit(self, ticket: OrderTicket, execution: str = "combo") -> SubmitResult:
        if self.mode == "manual":
            return SubmitResult(False, [], "manual mode — submission disabled")
        self.calls.append((ticket, execution))
        n = len(ticket.legs) if execution == "legin" else 1
        return SubmitResult(True, [f"MOCK-{i+1}" for i in range(n)],
                            f"mock {execution} order(s) in {self.mode}")


class TastyTradeSubmitter(OrderSubmitter):
    """Real broker submission via the bot's TastyTrade client. Only constructed in
    sandbox/live (serve --mode). NOTE: needs an authenticated client; the async/session
    integration must be validated live (it has never placed a real order in tests)."""
    def __init__(self, mode: str, client, loop=None, proxy: str | None = None):
        super().__init__(mode)
        self.client = client          # an authenticated TastyTradeClient for `mode`
        self.loop = loop              # persistent background loop the client lives on
        self.proxy = proxy            # "spy" → submit a SPY proxy (cert has no SPXW)

    def submit(self, ticket: OrderTicket, execution: str = "combo") -> SubmitResult:
        if self.mode == "manual":
            return SubmitResult(False, [], "manual mode — submission disabled")
        import asyncio

        proxied = ""
        if self.proxy == "spy":
            from src_v2.flux.orders import spy_proxy
            ticket = spy_proxy(ticket)
            proxied = " [SPY proxy]"

        async def _do() -> List[str]:
            if execution == "legin":
                ids: List[str] = []
                for group in legin_groups(ticket):         # buy longs first, then sell shorts
                    for leg in group:
                        r = await self.client.place_option_limit(
                            leg.osi, leg.action, leg.quantity, leg.limit)
                        ids.append(r["order_id"])
                return ids
            r = await self.client.place_complex_order(
                combo_legs(ticket), ticket.price, ticket.price_effect)
            return [r["order_id"]]

        try:
            if self.loop is not None:
                ids = asyncio.run_coroutine_threadsafe(_do(), self.loop).result(timeout=60)
            else:
                ids = asyncio.run(_do())
            return SubmitResult(True, ids, f"{execution}: {len(ids)} order(s) in {self.mode}{proxied}")
        except Exception as e:
            return SubmitResult(False, [], f"broker error: {e}")


def make_tt_submitter(mode: str) -> "TastyTradeSubmitter":
    """Build a live submitter: spin a background event loop, connect an authenticated
    TastyTradeClient on it, and route submissions there. Raises if connect fails."""
    import asyncio
    import threading

    from src_v2.services.tastytrade_client import TastyTradeClient

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    client = TastyTradeClient(mode=mode)
    asyncio.run_coroutine_threadsafe(client.connect(), loop).result(timeout=30)
    proxy = "spy" if mode == "sandbox" else None   # cert has no SPXW → submit a SPY proxy
    return TastyTradeSubmitter(mode, client, loop, proxy=proxy)
