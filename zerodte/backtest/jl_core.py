"""Pure logic for the 0DTE SPX Jade Lizard backtest.

No I/O, no Databento, no bars — everything operates on an in-memory quote dict
{(minute_et, strike, is_call): (bid, ask)} so it is fully unit-testable.

Pricing is chain-derived: ATM = argmin|call_mid - put_mid|, forward = atm +
(call_mid - put_mid), EM = ATM straddle mid. Exit value = combined bid/ask mid
at the exit anchor (TastyTrade-comparable "exercise at mid").
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

QuoteKey = Tuple[str, float, bool]
Quotes = Dict[QuoteKey, Tuple[float, float]]


def make_anchors(start: str = "09:30", end: str = "16:00", step_min: int = 10) -> List[str]:
    t = int(start[:2]) * 60 + int(start[3:5])
    e = int(end[:2]) * 60 + int(end[3:5])
    out: List[str] = []
    while t <= e:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += step_min
    return out


def _mid(bid: float, ask: float) -> Optional[float]:
    if bid <= 0 and ask <= 0:
        return None
    if bid <= 0:
        return ask
    if ask <= 0:
        return bid
    return (bid + ask) / 2


def quote_mid(quotes: Quotes, minute: str, strike: float, is_call: bool,
              fallback_min: int = 5) -> Optional[float]:
    q = quotes.get((minute, strike, is_call))
    if q is None and fallback_min > 0:
        hh, mm = int(minute[:2]), int(minute[3:5])
        offsets: List[int] = []
        for d in range(1, fallback_min + 1):
            offsets += [-d, d]
        for off in offsets:
            nmm, nhh = mm + off, hh
            while nmm < 0:
                nmm += 60
                nhh -= 1
            while nmm >= 60:
                nmm -= 60
                nhh += 1
            if nhh < 0 or nhh > 23:
                continue
            q = quotes.get((f"{nhh:02d}:{nmm:02d}", strike, is_call))
            if q is not None:
                break
    if q is None:
        return None
    return _mid(*q)


def atm_from_chain(quotes: Quotes, minute: str) -> Optional[float]:
    """ATM strike = the one minimizing |call_mid - put_mid| (no external spot)."""
    strikes = sorted({k[1] for k in quotes if k[0] == minute})
    best: Optional[Tuple[float, float]] = None
    for s in strikes:
        c = quote_mid(quotes, minute, s, True, 0)
        p = quote_mid(quotes, minute, s, False, 0)
        if c is None or p is None:
            continue
        diff = abs(c - p)
        if best is None or diff < best[1]:
            best = (s, diff)
    return best[0] if best else None


def compute_em(quotes: Quotes, minute: str, atm: float) -> Optional[float]:
    c = quote_mid(quotes, minute, atm, True)
    p = quote_mid(quotes, minute, atm, False)
    if c is None or p is None:
        return None
    return c + p


def forward_from_chain(quotes: Quotes, minute: str, atm: float) -> Optional[float]:
    c = quote_mid(quotes, minute, atm, True)
    p = quote_mid(quotes, minute, atm, False)
    if c is None or p is None:
        return None
    return atm + (c - p)


@dataclass
class JlConfig:
    entry_anchor: str = "09:30"
    anchors: List[str] = field(default_factory=lambda: make_anchors("09:30", "16:00", 10))
    em_mult_put: float = 0.25
    em_mult_call: float = 0.25
    put_width: float = 30.0
    call_width: float = 10.0
    tp_pct: float = 0.50          # exit when combined mark <= tp_pct * total_credit
    step: float = 5.0
    multiplier: int = 100
    slippage_per_spread: float = 0.0          # headline = 0
    commission_per_contract_per_leg: float = 0.0
    exit_mode: str = "mid"        # "mid" (mark at exit anchor) or "intrinsic"


def select_strikes(forward: float, em: float, cfg: JlConfig
                   ) -> Tuple[float, float, float, float]:
    step = cfg.step
    short_put = math.floor((forward - cfg.em_mult_put * em) / step) * step
    long_put = short_put - cfg.put_width
    short_call = math.ceil((forward + cfg.em_mult_call * em) / step) * step
    long_call = short_call + cfg.call_width
    return short_put, long_put, short_call, long_call


def entry_credit(quotes: Quotes, minute: str,
                 strikes: Tuple[float, float, float, float],
                 cfg: JlConfig) -> Optional[Dict]:
    sp, lp, sc, lc = strikes
    spm = quote_mid(quotes, minute, sp, False)
    lpm = quote_mid(quotes, minute, lp, False)
    scm = quote_mid(quotes, minute, sc, True)
    lcm = quote_mid(quotes, minute, lc, True)
    if None in (spm, lpm, scm, lcm):
        return None
    put_credit = spm - lpm
    call_credit = scm - lcm
    if put_credit <= 0 or call_credit <= 0:
        return None
    return {
        "put_credit": put_credit, "call_credit": call_credit,
        "total_credit": put_credit + call_credit,
        "sp": spm, "lp": lpm, "sc": scm, "lc": lcm,
    }


def combined_mark(quotes: Quotes, minute: str,
                  strikes: Tuple[float, float, float, float]) -> Optional[float]:
    sp, lp, sc, lc = strikes
    spm = quote_mid(quotes, minute, sp, False)
    lpm = quote_mid(quotes, minute, lp, False)
    scm = quote_mid(quotes, minute, sc, True)
    lcm = quote_mid(quotes, minute, lc, True)
    if None in (spm, lpm, scm, lcm):
        return None
    return max(spm - lpm, 0) + max(scm - lcm, 0)


def _intrinsic_debit(strikes: Tuple[float, float, float, float], S: float) -> float:
    sp, lp, sc, lc = strikes
    put_debit = max(sp - S, 0) - max(lp - S, 0)
    call_debit = max(S - sc, 0) - max(S - lc, 0)
    return put_debit + call_debit


def settlement_forward(quotes: Quotes, anchors: List[str]) -> Optional[float]:
    """Chain-implied SPX forward at the FINAL anchor — the settlement reference.

    Uses the same put-call-parity forward as entry so settlement stays on the
    SAME basis as strike placement. (SPY*10 closes run ~18 pts below the SPX
    chain forward — using them to settle chain-placed strikes manufactures
    phantom put-side losses.) Returns None if the final anchor has no ATM chain;
    the caller then falls back to an explicit spot_close.
    """
    last = anchors[-1]
    atm = atm_from_chain(quotes, last)
    if atm is None:
        return None
    return forward_from_chain(quotes, last, atm)


def evaluate_day(quotes: Quotes, cfg: JlConfig,
                 spot_close: Optional[float] = None) -> Dict:
    """Evaluate one trading day. Returns {'skip': reason} or a trade-result dict."""
    ea = cfg.entry_anchor
    atm = atm_from_chain(quotes, ea)
    if atm is None:
        return {"skip": "no_atm"}
    em = compute_em(quotes, ea, atm)
    if em is None or em <= 0:
        return {"skip": "no_em"}
    fwd = forward_from_chain(quotes, ea, atm)
    if fwd is None:
        return {"skip": "no_forward"}
    strikes = select_strikes(fwd, em, cfg)
    cred = entry_credit(quotes, ea, strikes, cfg)
    if cred is None:
        return {"skip": "no_credit"}
    total = cred["total_credit"]

    kind: Optional[str] = None
    exit_anchor: Optional[str] = None
    exit_mark: Optional[float] = None
    last_avail: Optional[Tuple[str, float]] = None
    for a in cfg.anchors:
        if a == ea:
            continue
        m = combined_mark(quotes, a, strikes)
        if m is None:
            continue
        last_avail = (a, m)
        if m <= cfg.tp_pct * total:
            kind, exit_anchor, exit_mark = "tp", a, m
            break

    if kind is None:
        if cfg.exit_mode == "intrinsic":
            S = settlement_forward(quotes, cfg.anchors)
            if S is None:
                S = spot_close  # fallback only if the final anchor's chain is missing
            if S is None:
                return {"skip": "no_settle"}
            exit_anchor, exit_mark, kind = cfg.anchors[-1], _intrinsic_debit(strikes, S), "expiry"
        else:
            if last_avail is None:
                return {"skip": "no_marks"}
            exit_anchor, exit_mark, kind = last_avail[0], last_avail[1], "expiry"

    mult = cfg.multiplier
    gross = (total - exit_mark) * mult
    slip = cfg.slippage_per_spread
    # entry always pays slip on 2 spreads; exit pays slip on 2 spreads UNLESS the
    # losers cash-settle at intrinsic (no closing trade).
    exit_slip = 0.0 if (kind == "expiry" and cfg.exit_mode == "intrinsic") else slip * 2
    cost = (slip * 2 + exit_slip) * mult + cfg.commission_per_contract_per_leg * 4
    net = gross - cost

    return {
        "skip": None, "kind": kind, "exit_anchor": exit_anchor,
        "total_credit": total, "exit_mark": exit_mark,
        "gross": gross, "pnl": net, "strikes": strikes,
        "em": em, "forward": fwd, "put_credit": cred["put_credit"],
        "call_credit": cred["call_credit"],
        "upside_riskfree": total >= cfg.call_width,
    }


def _max_consec_losses(pnls: List[float]) -> int:
    best = cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _max_drawdown(pnls: List[float]) -> float:
    cum = peak = worst = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return worst


def evaluate(by_date: Dict, cfg: JlConfig) -> Dict:
    """Run evaluate_day over a {date: {quotes, spot_close}} map; aggregate metrics."""
    results: List[Dict] = []
    skips: Dict[str, int] = defaultdict(int)
    exits: Dict[str, int] = defaultdict(int)
    for d in sorted(by_date.keys()):
        info = by_date[d]
        r = evaluate_day(info["quotes"], cfg, info.get("spot_close"))
        if r.get("skip"):
            skips[r["skip"]] += 1
            continue
        r["date"] = d
        exits[r["kind"]] += 1
        results.append(r)

    pnls = [r["pnl"] for r in results]
    if not pnls:
        return {"n": 0, "skips": dict(skips), "results": []}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / max(1, len(pnls) - 1)
    std = math.sqrt(var)
    credits = [r["total_credit"] for r in results]
    return {
        "n": len(pnls),
        "wr": 100 * len(wins) / len(pnls),
        "total": sum(pnls),
        "mean": mean,
        "sharpe_ann": math.sqrt(252) * mean / std if std > 0 else 0.0,
        "max_consec_L": _max_consec_losses(pnls),
        "max_dd": _max_drawdown(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "worst": min(pnls),
        "best": max(pnls),
        "exits": dict(exits),
        "skips": dict(skips),
        "avg_credit": sum(credits) / len(credits),
        "pct_upside_riskfree": 100 * sum(1 for r in results if r["upside_riskfree"]) / len(results),
        "results": results,
    }
