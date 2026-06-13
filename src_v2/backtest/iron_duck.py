"""Iron Duck backtest engine — SPX broken-skewed iron condor (multi-day).

Strategy (docs/The-Iron-Duck-Options-Spread/The-Iron-Duck-Options-Spread-Strategy.md):
  - SELL ~65d call ITM, BUY 1 strike higher (or ~62d)   -> 1-strike-wide ITM call vertical
  - SELL ~15d put OTM,  BUY ~5d (OO) or ~10d (cheat sheet) put -> wider OTM put vertical
  - net credit; "no upside risk" when credit >= call-spread width (flat "beak")
  - entry constraints (cheat sheet): POP >= 85%; credit >= call width;
    theoretical T+0 loss at the downside breakeven < credit
  - DTE 1-21 (instructor's OO sweep: 5/7/10/14/21), sequential entries
    (enter when the prior trade closes, next trading day)
  - exits: stop loss at 100% of credit (manual, evaluated on closes);
    otherwise hold to expiration (cash settle); optional early "beak" exit
    when P(return to max-profit zone) < 10%

Data: EOD NBBO chains per expiration from data/iron_duck_eod/SPXW/<exp>.pkl
(built by scripts/backfill_iron_duck_eod.py via ThetaData). Greeks are derived
from the chain itself: implied forward via put-call parity, then Black-76
(discount ~ 1) IV / delta — same approach as the Hedge Hog engine, which this
module borrows its put-side greeks and store from.

All option prices are per-share (SPX points); P&L uses the 100 multiplier.
Entries and all marks are at the 16:00 EOD NBBO — the only data available —
so the "manual" stop triggers on a daily close, matching the instructor's
alert-based (not resting order) stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.optimize import brentq
from scipy.stats import norm

from src_v2.backtest.hedgehog import (
    Chain,
    HedgeHogStore,
    Quote,
    _to_date,
    black_put,
    black_put_iv,
    parity_forward,
)

DuckStore = HedgeHogStore  # same pickle layout, different cache dir


# ---------------------------------------------------------------- greeks


def black_call(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sq
    d2 = d1 - sq
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def black_call_delta(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if F > K else 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sq
    return norm.cdf(d1)


def black_call_iv(price: float, F: float, K: float, T: float) -> float:
    intrinsic = max(F - K, 0.0)
    if T <= 0 or price <= intrinsic + 1e-6:
        return float("nan")
    try:
        return brentq(lambda s: black_call(F, K, T, s) - price, 1e-4, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return float("nan")


def _mid(q: Quote) -> float:
    return (q[0] + q[1]) / 2


def _fill(q: Quote, side: str, k: float) -> float:
    """Fill price: mid +/- k * half-spread. side='buy' pays up, 'sell' gives up."""
    bid, ask, _ = q
    mid = (bid + ask) / 2
    hs = (ask - bid) / 2
    return mid + k * hs if side == "buy" else mid - k * hs


# ---------------------------------------------------------------- smile


class Smile:
    """Per-(date, expiry) vol-by-strike from OTM quotes, linear interpolation.

    ITM mids can sit below intrinsic when EOD spreads blow out, so every
    strike's vol is implied from its OTM side (puts below F, calls above),
    then shared across rights via parity.
    """

    def __init__(self, chain: Chain, F: float, T: float):
        self.F, self.T = F, T
        pts: List[Tuple[float, float]] = []
        for (k, right), q in chain.items():
            if q[0] <= 0:
                continue
            mid = _mid(q)
            if right == "P" and k <= F:
                iv = black_put_iv(mid, F, k, T)
            elif right == "C" and k > F:
                iv = black_call_iv(mid, F, k, T)
            else:
                continue
            if iv > 0:
                pts.append((k, iv))
        pts.sort()
        self.ks = [k for k, _ in pts]
        self.vols = [v for _, v in pts]

    def ok(self) -> bool:
        return len(self.ks) >= 4

    def vol(self, k: float) -> Optional[float]:
        ks, vols = self.ks, self.vols
        if not ks:
            return None
        if k <= ks[0]:
            return vols[0]
        if k >= ks[-1]:
            return vols[-1]
        import bisect

        i = bisect.bisect_left(ks, k)
        k0, k1 = ks[i - 1], ks[i]
        w = (k - k0) / (k1 - k0)
        return vols[i - 1] * (1 - w) + vols[i] * w

    def call_delta(self, k: float) -> Optional[float]:
        v = self.vol(k)
        return black_call_delta(self.F, k, self.T, v) if v else None

    def put_delta_abs(self, k: float) -> Optional[float]:
        d = self.call_delta(k)
        return 1.0 - d if d is not None else None


# ---------------------------------------------------------------- params


@dataclass
class DuckParams:
    # construction (deltas are call-delta for calls, abs put-delta for puts)
    dte_target: int = 7
    dte_min: int = 3
    dte_max: int = 24
    call_short_delta: float = 0.65
    call_long_mode: str = "next_strike"  # "next_strike" (cheat sheet) | "delta" (OO 62d)
    call_long_delta: float = 0.62
    put_short_delta: float = 0.15
    put_long_delta: float = 0.05  # OO config; cheat sheet says 0.10
    # entry constraints (cheat sheet); None disables a gate
    require_no_upside_risk: bool = False  # credit >= call width
    pop_min: Optional[float] = None  # e.g. 0.85
    require_be_loss_lt_credit: bool = False
    min_credit: Optional[float] = None  # SPX points
    # exits
    stop_frac: Optional[float] = 1.0  # stop at frac * credit loss; None = no stop
    beak_exit_prob: Optional[float] = None  # e.g. 0.10 -> close when P(back in head) < 10%
    # fills & costs (per contract)
    fill_k: float = 0.0  # 0 = mid, 1 = cross full half-spread (entry)
    exit_fill_k: Optional[float] = None  # stop/beak exits; None = same as fill_k
    commission_per_leg: float = 1.0  # each way
    fees_per_leg: float = 0.15  # regulatory/exchange, each way
    min_bid: float = 0.05  # short legs must have a real bid
    qty: int = 1


# ---------------------------------------------------------------- records


@dataclass
class DuckTrade:
    entry_date: int
    expiry: int
    dte: int
    spot_entry: float
    k_call_short: float
    k_call_long: float
    k_put_short: float
    k_put_long: float
    credit: float  # net credit, SPX points (after fill model)
    credit_mid: float  # net credit at pure mid (stop reference)
    call_width: float
    put_width: float
    pop_entry: float
    be_down: float
    be_loss: float  # theoretical T+0 loss if spot -> BE now (points)
    atm_iv_entry: float
    margin_usd: float
    qty: int = 1
    exit_date: Optional[int] = None
    exit_reason: Optional[str] = None
    spot_exit: Optional[float] = None
    pnl_usd: Optional[float] = None
    days_held: Optional[int] = None
    daily_marks: List[float] = field(default_factory=list)


@dataclass
class SkipRecord:
    date: int
    reason: str


# ---------------------------------------------------------------- engine

LEGS = 4


class IronDuckEngine:
    def __init__(self, store: DuckStore, params: DuckParams | None = None):
        self.store = store
        self.p = params or DuckParams()
        self.trades: List[DuckTrade] = []
        self.skips: List[SkipRecord] = []
        self.daily_equity: Dict[int, float] = {}  # date -> total MTM P&L (open+closed)

    # -------------------------------------------------- helpers

    def _smile(self, date: int, expiry: int) -> Optional[Smile]:
        chain = self.store.chain(date, expiry)
        if not chain:
            return None
        F = parity_forward(chain)
        if not F:
            return None
        T = max((_to_date(expiry) - _to_date(date)).days, 0) / 365.0
        sm = Smile(chain, F, T)
        return sm if sm.ok() else None

    @staticmethod
    def payoff(tr: DuckTrade, s: float) -> float:
        """Expiration payoff per share at settlement price s."""
        call_sp = max(s - tr.k_call_short, 0.0) - max(s - tr.k_call_long, 0.0)
        put_sp = max(tr.k_put_short - s, 0.0) - max(tr.k_put_long - s, 0.0)
        return tr.credit - call_sp - put_sp

    def _position_value(self, tr: DuckTrade, chain: Chain, fill_k: float = 0.0) -> Optional[float]:
        """Cost to close (debit, per share). fill_k>0 pays the spread."""
        legs = [
            (tr.k_call_short, "C", "buy"),
            (tr.k_call_long, "C", "sell"),
            (tr.k_put_short, "P", "buy"),
            (tr.k_put_long, "P", "sell"),
        ]
        total = 0.0
        for k, right, side in legs:
            q = chain.get((k, right))
            if q is None or q[1] <= 0:
                return None
            px = _fill(q, side, fill_k)
            total += px if side == "buy" else -px
        return total

    def _model_value(self, tr: DuckTrade, sm: Smile, F: float) -> float:
        """Black-76 value of the position (cost to close) at forward F,
        sticky-strike vols from the smile."""

        def call(k: float) -> float:
            return black_call(F, k, sm.T, sm.vol(k) or 0.0)

        def put(k: float) -> float:
            return black_put(F, k, sm.T, sm.vol(k) or 0.0)

        return (
            call(tr.k_call_short) - call(tr.k_call_long) + put(tr.k_put_short) - put(tr.k_put_long)
        )

    # -------------------------------------------------- entry

    def _pick_expiry(self, date: int) -> Optional[Tuple[int, int]]:
        cands = self.store.expiries_in_dte(date, self.p.dte_min, self.p.dte_max)
        if not cands:
            return None
        return min(cands, key=lambda e: (abs(e[1] - self.p.dte_target), e[1]))

    def _strike_by_call_delta(
        self, sm: Smile, chain: Chain, target: float, right: str
    ) -> Optional[float]:
        """Listed strike whose call-delta is nearest target; short legs need a bid."""
        best, best_err = None, 1e9
        for (k, r), q in chain.items():
            if r != right:
                continue
            d = sm.call_delta(k)
            if d is None:
                continue
            err = abs(d - target)
            if err < best_err:
                best, best_err = k, err
        return best

    def _next_strike_up(self, chain: Chain, k: float, right: str) -> Optional[float]:
        ks = sorted({kk for (kk, r) in chain if r == right and kk > k})
        return ks[0] if ks else None

    def try_enter(self, date: int) -> Optional[DuckTrade]:
        p = self.p
        picked = self._pick_expiry(date)
        if not picked:
            self.skips.append(SkipRecord(date, "no_expiry"))
            return None
        expiry, dte = picked
        chain = self.store.chain(date, expiry)
        sm = self._smile(date, expiry)
        if not sm:
            self.skips.append(SkipRecord(date, "no_smile"))
            return None
        F = sm.F

        # --- strikes
        k_cs = self._strike_by_call_delta(sm, chain, p.call_short_delta, "C")
        if k_cs is None:
            self.skips.append(SkipRecord(date, "no_call_short"))
            return None
        if p.call_long_mode == "next_strike":
            k_cl = self._next_strike_up(chain, k_cs, "C")
        else:
            k_cl = self._strike_by_call_delta(sm, chain, p.call_long_delta, "C")
        if k_cl is None or k_cl <= k_cs:
            self.skips.append(SkipRecord(date, "no_call_long"))
            return None
        # puts: target abs put delta == 1 - call delta
        k_ps = self._strike_by_call_delta(sm, chain, 1.0 - p.put_short_delta, "P")
        k_pl = self._strike_by_call_delta(sm, chain, 1.0 - p.put_long_delta, "P")
        if k_ps is None or k_pl is None or k_pl >= k_ps:
            self.skips.append(SkipRecord(date, "no_put_legs"))
            return None

        legs = {
            "cs": chain.get((k_cs, "C")),
            "cl": chain.get((k_cl, "C")),
            "ps": chain.get((k_ps, "P")),
            "pl": chain.get((k_pl, "P")),
        }
        if any(q is None or q[1] <= 0 for q in legs.values()):
            self.skips.append(SkipRecord(date, "missing_quote"))
            return None
        if legs["cs"][0] < p.min_bid or legs["ps"][0] < p.min_bid:
            self.skips.append(SkipRecord(date, "no_short_bid"))
            return None

        credit = (
            _fill(legs["cs"], "sell", p.fill_k)
            - _fill(legs["cl"], "buy", p.fill_k)
            + _fill(legs["ps"], "sell", p.fill_k)
            - _fill(legs["pl"], "buy", p.fill_k)
        )
        credit_mid = _mid(legs["cs"]) - _mid(legs["cl"]) + _mid(legs["ps"]) - _mid(legs["pl"])
        if credit <= 0:
            self.skips.append(SkipRecord(date, "no_credit"))
            return None

        call_width = k_cl - k_cs
        put_width = k_ps - k_pl

        # --- breakevens & POP (lognormal at the BE strike's vol)
        be_down = k_ps - credit
        T = sm.T if sm.T > 0 else 1e-6
        vol_be = sm.vol(be_down) or sm.vol(k_ps) or 0.2
        # P(S_T <= be_down)
        p_below = norm.cdf((math.log(be_down / F) + 0.5 * vol_be**2 * T) / (vol_be * math.sqrt(T)))
        if credit >= call_width:
            p_above = 0.0  # flat beak, no upside loss
        else:
            be_up = k_cs + credit
            vol_up = sm.vol(be_up) or 0.2
            p_above = 1.0 - norm.cdf(
                (math.log(be_up / F) + 0.5 * vol_up**2 * T) / (vol_up * math.sqrt(T))
            )
        pop = 1.0 - p_below - p_above

        atm_iv = sm.vol(F) or 0.0

        tr = DuckTrade(
            entry_date=date,
            expiry=expiry,
            dte=dte,
            spot_entry=F,
            k_call_short=k_cs,
            k_call_long=k_cl,
            k_put_short=k_ps,
            k_put_long=k_pl,
            credit=credit,
            credit_mid=credit_mid,
            call_width=call_width,
            put_width=put_width,
            pop_entry=pop,
            be_down=be_down,
            be_loss=0.0,
            atm_iv_entry=atm_iv,
            margin_usd=(max(call_width, put_width) - credit) * 100 * p.qty,
            qty=p.qty,
        )
        # theoretical T+0 loss if spot gapped to the downside BE right now
        tr.be_loss = max(0.0, self._model_value(tr, sm, be_down) - credit)

        # --- gates
        if p.require_no_upside_risk and credit < call_width:
            self.skips.append(SkipRecord(date, "upside_risk"))
            return None
        if p.pop_min is not None and pop < p.pop_min:
            self.skips.append(SkipRecord(date, "pop"))
            return None
        if p.require_be_loss_lt_credit and tr.be_loss >= credit:
            self.skips.append(SkipRecord(date, "be_loss"))
            return None
        if p.min_credit is not None and credit < p.min_credit:
            self.skips.append(SkipRecord(date, "min_credit"))
            return None
        return tr

    # -------------------------------------------------- exits

    def _check_stop(self, tr: DuckTrade, date: int) -> bool:
        if self.p.stop_frac is None:
            return False
        chain = self.store.chain(date, tr.expiry)
        if not chain:
            return False
        val = self._position_value(tr, chain, fill_k=0.0)  # alert on mid mark
        if val is None:
            return False
        return (val - tr.credit) >= self.p.stop_frac * tr.credit

    def _check_beak(self, tr: DuckTrade, date: int) -> bool:
        if self.p.beak_exit_prob is None:
            return False
        sm = self._smile(date, tr.expiry)
        if not sm or sm.T <= 0:
            return False
        if sm.F <= tr.k_call_long:
            return False  # not in the beak
        v = sm.vol(tr.k_call_short) or 0.0
        if v <= 0:
            return False
        # P(S_T < short call) = chance of returning to the max-profit "head"
        p_back = norm.cdf(
            (math.log(tr.k_call_short / sm.F) + 0.5 * v * v * sm.T) / (v * math.sqrt(sm.T))
        )
        return p_back < self.p.beak_exit_prob

    def close(self, tr: DuckTrade, date: int, reason: str) -> bool:
        chain = self.store.chain(date, tr.expiry)
        if reason == "expiry":
            # cash settle at the expiry-day parity forward (PM settlement)
            settle = parity_forward(chain) if chain else None
            if settle is None:
                settle = self.store.spot(date)
            if settle is None:
                return False
            pnl_pts = self.payoff(tr, settle)
            spot_exit = settle
            fees = (
                (self.p.commission_per_leg + self.p.fees_per_leg) * LEGS * tr.qty
            )  # entry side only; expiration has no closing commission
        else:
            if not chain:
                return False
            ek = self.p.exit_fill_k if self.p.exit_fill_k is not None else self.p.fill_k
            val = self._position_value(tr, chain, fill_k=ek)
            if val is None:
                sm = self._smile(date, tr.expiry)
                if not sm:
                    return False
                val = self._model_value(tr, sm, sm.F)
            pnl_pts = tr.credit - val
            sm = self._smile(date, tr.expiry)
            spot_exit = sm.F if sm else None
            fees = (self.p.commission_per_leg + self.p.fees_per_leg) * 2 * LEGS * tr.qty
        tr.exit_date = date
        tr.exit_reason = reason
        tr.spot_exit = spot_exit
        tr.pnl_usd = pnl_pts * 100 * tr.qty - fees
        tr.days_held = (_to_date(date) - _to_date(tr.entry_date)).days
        return True

    # -------------------------------------------------- run

    def run(self, start: int = 0, end: int = 99999999) -> List[DuckTrade]:
        dates = [d for d in self.store.quote_dates() if start <= d <= end]
        open_tr: Optional[DuckTrade] = None
        realized = 0.0
        for date in dates:
            if open_tr is not None and date > open_tr.entry_date:
                reason = None
                if date >= open_tr.expiry:
                    reason = "expiry"
                elif self._check_stop(open_tr, date):
                    reason = "stop"
                elif self._check_beak(open_tr, date):
                    reason = "beak"
                if reason and self.close(open_tr, date, reason):
                    realized += open_tr.pnl_usd
                    self.trades.append(open_tr)
                    open_tr = None
            if open_tr is None:
                # sequential: re-establish on the next trading date after a close;
                # entering the same day a trade closed is allowed only at start
                last_exit = self.trades[-1].exit_date if self.trades else None
                if last_exit is None or date > last_exit:
                    open_tr = self.try_enter(date)
            # daily MTM
            mtm = realized
            if open_tr is not None:
                chain = self.store.chain(date, open_tr.expiry)
                val = self._position_value(open_tr, chain) if chain else None
                if val is not None:
                    mtm += (open_tr.credit - val) * 100 * open_tr.qty
            self.daily_equity[date] = mtm
        if open_tr is not None and open_tr.exit_date is None:
            last = dates[-1] if dates else None
            if last and self.close(open_tr, last, "eod_force"):
                self.trades.append(open_tr)
        return self.trades


# ---------------------------------------------------------------- report


def summarize(trades: List[DuckTrade], daily_equity: Dict[int, float]) -> Dict:
    closed = [t for t in trades if t.pnl_usd is not None]
    if not closed:
        return {"trades": 0}
    pnls = [t.pnl_usd for t in closed]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    eq = [daily_equity[d] for d in sorted(daily_equity)]
    peak, max_dd = -1e18, 0.0
    for v in eq:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    margin = max((t.margin_usd for t in closed), default=0.0)
    days = (_to_date(sorted(daily_equity)[-1]) - _to_date(sorted(daily_equity)[0])).days
    years = max(days / 365.25, 1e-9)
    net = sum(pnls)
    by_reason: Dict[str, int] = {}
    for t in closed:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
    return {
        "trades": len(closed),
        "net_pnl": round(net, 2),
        "avg_pnl": round(net / len(closed), 2),
        "win_rate": round(len(wins) / len(closed), 4),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "max_dd_usd": round(max_dd, 2),
        "max_margin_usd": round(margin, 2),
        "roc_ann": round(net / margin / years, 4) if margin else None,
        "years": round(years, 2),
        "exit_reasons": by_reason,
        "avg_credit": round(sum(t.credit for t in closed) / len(closed), 2),
        "avg_pop": round(sum(t.pop_entry for t in closed) / len(closed), 4),
        "avg_dte": round(sum(t.dte for t in closed) / len(closed), 1),
        "avg_days_held": round(sum(t.days_held for t in closed) / len(closed), 1),
    }
