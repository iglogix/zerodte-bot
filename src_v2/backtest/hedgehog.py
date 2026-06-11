"""Hedge Hog backtest engine — SPY swing skew-harvest structure.

Strategy (docs/Hedge-Hog-Options-Spread-Strategy.md):
  - "hedge": long put vertical ~30 DTE (25-35), long ~30-delta / short ~20-delta
  - "hog":   short naked put ~100 DTE (75-115), ~5-10-delta (~7), with the hard
             constraint credit(hog) > debit(hedge)  => package is net credit
  - quality gate: modeled PoP at the downside loss point > 65%
  - exits (any one, base case = close whole package & recycle):
      1) hog >= 50% max profit
      2) vertical >= 80% max profit
      3) vertical < 7 DTE
      4) spot below vertical short strike within 10 days of entry

Data: EOD NBBO chains per expiration from data/hedgehog_eod/SPY/<exp>.pkl
(built by scripts/backfill_hedgehog_eod.py via ThetaData). Greeks are derived
from the chain itself: implied forward via put-call parity, then Black-76
(discount ~ 1) IV / delta — no external rate or dividend inputs.

All option prices are per-share; P&L in dollars uses the 100 multiplier.
"""

from __future__ import annotations

import datetime as dt
import math
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.optimize import brentq
from scipy.stats import norm

Quote = Tuple[float, float, int]  # bid, ask, volume
Chain = Dict[Tuple[float, str], Quote]


# ---------------------------------------------------------------- greeks

def black_put(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - F, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sq
    d2 = d1 - sq
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def black_put_delta(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return -1.0 if F < K else 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sq
    return norm.cdf(d1) - 1.0


def black_put_iv(price: float, F: float, K: float, T: float) -> float:
    intrinsic = max(K - F, 0.0)
    if T <= 0 or price <= intrinsic + 1e-6:
        return float("nan")
    try:
        return brentq(lambda s: black_put(F, K, T, s) - price, 1e-4, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return float("nan")


def parity_forward(chain: Chain) -> Optional[float]:
    """Implied forward from put-call parity at the most ATM strikes."""
    cands = []
    for (k, right), (bid, ask, _v) in chain.items():
        if right != "C" or bid <= 0:
            continue
        p = chain.get((k, "P"))
        if not p or p[0] <= 0:
            continue
        c_mid = (bid + ask) / 2
        p_mid = (p[0] + p[1]) / 2
        cands.append((abs(c_mid - p_mid), k + c_mid - p_mid))
    if not cands:
        return None
    cands.sort()
    best = cands[:3]
    return sum(f for _, f in best) / len(best)


# ---------------------------------------------------------------- store

def _to_date(d: int) -> dt.date:
    return dt.date(d // 10000, d // 100 % 100, d % 100)


def _to_int(d: dt.date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


class HedgeHogStore:
    """Lazy per-expiry loader over data/hedgehog_eod/<root>/ pickles."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.expiries: List[int] = sorted(
            int(p.stem)
            for p in self.cache_dir.glob("*.pkl")
            if not p.stem.startswith("_")
        )
        if not self.expiries:
            raise FileNotFoundError(f"no expiry pickles in {self.cache_dir}")
        self._quote_dates: Optional[List[int]] = None

    @lru_cache(maxsize=48)
    def _load(self, expiry: int) -> Dict[int, Chain]:
        with open(self.cache_dir / f"{expiry}.pkl", "rb") as f:
            return pickle.load(f)["days"]

    def quote_dates(self) -> List[int]:
        """Master trading calendar = union of quote dates (cached on disk scan)."""
        if self._quote_dates is None:
            idx_path = self.cache_dir / "_calendar.pkl"
            sig = (len(self.expiries), self.expiries[-1])
            if idx_path.exists():
                with open(idx_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("sig") == sig:
                    self._quote_dates = cached["dates"]
                    return self._quote_dates
            dates: set[int] = set()
            for exp in self.expiries:
                dates.update(self._load(exp).keys())
                self._load.cache_clear()
            self._quote_dates = sorted(dates)
            with open(idx_path, "wb") as f:
                pickle.dump({"sig": sig, "dates": self._quote_dates}, f)
        return self._quote_dates

    def chain(self, quote_date: int, expiry: int) -> Optional[Chain]:
        if expiry not in self.expiries:
            return None
        return self._load(expiry).get(quote_date)

    def expiries_in_dte(
        self, quote_date: int, dte_min: int, dte_max: int
    ) -> List[Tuple[int, int]]:
        """[(expiry, dte)] with any data on quote_date, dte in [dte_min, dte_max]."""
        qd = _to_date(quote_date)
        out = []
        for exp in self.expiries:
            dte = (_to_date(exp) - qd).days
            if dte_min <= dte <= dte_max and self.chain(quote_date, exp):
                out.append((exp, dte))
        return out

    def spot(self, quote_date: int) -> Optional[float]:
        """Parity forward of the nearest listed expiry (3-30 days out)."""
        for exp, _dte in self.expiries_in_dte(quote_date, 1, 30):
            ch = self.chain(quote_date, exp)
            f = parity_forward(ch) if ch else None
            if f:
                return f
        return None


# ---------------------------------------------------------------- params

@dataclass
class HhParams:
    # hedge: long put vertical
    hedge_dte_target: int = 30
    hedge_dte_min: int = 25
    hedge_dte_max: int = 35
    hedge_long_delta: float = 0.30  # absolute put delta
    hedge_short_delta: float = 0.20
    # hog: far short put
    hog_dte_target: int = 100
    hog_dte_min: int = 75
    hog_dte_max: int = 115
    hog_delta: float = 0.07
    hog_delta_max: float = 0.12  # furthest we'll step toward ATM chasing credit
    hog_delta_min: float = 0.04
    # quality gate
    pop_min: float = 0.65
    # exits
    hog_tp_frac: float = 0.50
    vert_tp_frac: float = 0.80
    vert_close_dte: int = 7
    fast_breach_days: int = 10
    # portfolio
    max_concurrent: int = 1
    ladder_every_days: int = 1  # min days between entries when laddering
    qty: int = 1
    # fills & costs (per contract)
    fill_k: float = 0.5          # 0 = mid, 1 = cross the full half-spread
    commission_open: float = 1.0
    commission_close: float = 0.0
    fees_per_leg: float = 0.15   # regulatory/exchange, each way
    # liquidity sanity
    min_bid: float = 0.05        # short legs must have a real bid
    # ablation: keep entry selection + triggers identical but only fill the
    # hog leg (isolates the hedge's contribution to the package)
    trade_hedge: bool = True


# ---------------------------------------------------------------- records

@dataclass
class HedgeHogTrade:
    entry_date: int
    spot_entry: float
    hedge_expiry: int
    k_long: float
    k_short: float
    debit: float
    hog_expiry: int
    k_hog: float
    hog_credit: float
    hog_delta: float
    pop_entry: float
    atm_iv_entry: float
    margin_usd: float
    qty: int = 1
    # filled at exit
    exit_date: Optional[int] = None
    exit_reason: Optional[str] = None
    spot_exit: Optional[float] = None
    pnl_usd: Optional[float] = None
    days_held: Optional[int] = None
    # diagnostics
    net_credit: float = 0.0
    rr_ratio: float = 0.0
    daily_marks: List[float] = field(default_factory=list)


@dataclass
class SkipRecord:
    date: int
    reason: str


# ---------------------------------------------------------------- engine

def _fill(q: Quote, side: str, k: float) -> float:
    """Fill price: mid +/- k * half-spread. side='buy' pays up, 'sell' gives up."""
    bid, ask, _ = q
    mid = (bid + ask) / 2
    hs = (ask - bid) / 2
    return mid + k * hs if side == "buy" else mid - k * hs


def _mid(q: Quote) -> float:
    return (q[0] + q[1]) / 2


class HedgeHogEngine:
    def __init__(self, store: HedgeHogStore, params: HhParams | None = None):
        self.store = store
        self.p = params or HhParams()
        self.trades: List[HedgeHogTrade] = []
        self.skips: List[SkipRecord] = []
        self.daily_equity: Dict[int, float] = {}  # date -> total MTM P&L (open+closed)

    # -------------------------------------------------- chain analytics

    def _put_deltas(
        self, chain: Chain, quote_date: int, expiry: int
    ) -> Optional[Dict[float, Tuple[float, float, Quote]]]:
        """{strike: (delta_abs, iv, quote)} for puts with a live bid."""
        F = parity_forward(chain)
        if not F:
            return None
        T = (_to_date(expiry) - _to_date(quote_date)).days / 365.0
        out = {}
        for (k, right), q in chain.items():
            if right != "P" or q[0] <= 0 or k >= F:
                continue
            iv = black_put_iv(_mid(q), F, k, T)
            if not (iv > 0):
                continue
            d = abs(black_put_delta(F, k, T, iv))
            out[k] = (d, iv, q)
        return out or None

    def _atm_iv(self, chain: Chain, quote_date: int, expiry: int) -> Optional[float]:
        F = parity_forward(chain)
        if not F:
            return None
        T = (_to_date(expiry) - _to_date(quote_date)).days / 365.0
        # nearest strikes to the forward, OTM puts first (ITM mids can sit
        # below intrinsic when spreads blow out, killing the inversion)
        puts = sorted(
            (k for (k, right), q in chain.items() if right == "P" and q[0] > 0),
            key=lambda k: (k > F, abs(k - F)),
        )
        for k in puts[:6]:
            iv = black_put_iv(_mid(chain[(k, "P")]), F, k, T)
            if iv > 0:
                return iv
        return None

    # -------------------------------------------------- entry

    def try_enter(self, date: int) -> Optional[HedgeHogTrade]:
        p = self.p
        spot = self.store.spot(date)
        if not spot:
            self.skips.append(SkipRecord(date, "no_spot"))
            return None

        hedge_cands = self.store.expiries_in_dte(date, p.hedge_dte_min, p.hedge_dte_max)
        if not hedge_cands:
            self.skips.append(SkipRecord(date, "no_hedge_expiry"))
            return None
        hedge_exp, _ = min(hedge_cands, key=lambda e: abs(e[1] - p.hedge_dte_target))

        hog_cands = self.store.expiries_in_dte(date, p.hog_dte_min, p.hog_dte_max)
        if not hog_cands:
            self.skips.append(SkipRecord(date, "no_hog_expiry"))
            return None
        hog_exp, _ = min(hog_cands, key=lambda e: abs(e[1] - p.hog_dte_target))

        hedge_chain = self.store.chain(date, hedge_exp)
        hog_chain = self.store.chain(date, hog_exp)
        hd = self._put_deltas(hedge_chain, date, hedge_exp)
        gd = self._put_deltas(hog_chain, date, hog_exp)
        if not hd or not gd:
            self.skips.append(SkipRecord(date, "no_greeks"))
            return None

        # hedge: long ~30d / short ~20d put vertical
        k_long = min(hd, key=lambda k: abs(hd[k][0] - p.hedge_long_delta))
        k_short = min(hd, key=lambda k: abs(hd[k][0] - p.hedge_short_delta))
        if k_long <= k_short or hd[k_short][2][0] < p.min_bid:
            self.skips.append(SkipRecord(date, "bad_vertical"))
            return None
        debit = _fill(hd[k_long][2], "buy", p.fill_k) - _fill(
            hd[k_short][2], "sell", p.fill_k
        )
        width = k_long - k_short
        if debit <= 0 or debit >= width:
            self.skips.append(SkipRecord(date, "bad_vertical"))
            return None
        rr = (width - debit) / debit

        # hog: ~7-delta short put, stepped toward ATM until credit > debit
        cands = sorted(
            (k for k in gd if p.hog_delta_min <= gd[k][0] <= p.hog_delta_max),
            key=lambda k: gd[k][0],
        )
        pick = None
        for k in cands:
            if gd[k][0] < p.hog_delta - 0.02 or gd[k][2][0] < p.min_bid:
                continue
            credit = _fill(gd[k][2], "sell", p.fill_k)
            if credit > debit:
                pick = (k, credit)
                break
        if pick is None:
            self.skips.append(SkipRecord(date, "credit_le_debit"))
            return None
        k_hog, hog_credit = pick

        # quality gate: PoP above downside loss point at hedge expiry
        atm_iv = self._atm_iv(hedge_chain, date, hedge_exp)
        pop = self._pop(date, spot, hedge_exp, k_long, k_short, debit,
                        hog_exp, k_hog, gd[k_hog][1], hog_credit, atm_iv)
        if pop is not None and pop < p.pop_min:
            self.skips.append(SkipRecord(date, "pop_below_min"))
            return None

        margin = self._naked_put_margin(spot, k_hog, hog_credit)
        if p.trade_hedge:
            margin += debit * 100
        tr = HedgeHogTrade(
            entry_date=date, spot_entry=spot,
            hedge_expiry=hedge_exp, k_long=k_long, k_short=k_short, debit=debit,
            hog_expiry=hog_exp, k_hog=k_hog, hog_credit=hog_credit,
            hog_delta=gd[k_hog][0], pop_entry=pop if pop is not None else float("nan"),
            atm_iv_entry=atm_iv if atm_iv is not None else float("nan"),
            margin_usd=margin * self.p.qty, qty=self.p.qty,
            net_credit=hog_credit - (debit if p.trade_hedge else 0.0), rr_ratio=rr,
        )
        return tr

    def _pop(self, date, spot, hedge_exp, k_long, k_short, debit,
             hog_exp, k_hog, hog_iv, hog_credit, atm_iv) -> Optional[float]:
        """P(S_T1 above the package's downside breakeven at hedge expiry)."""
        t1 = (_to_date(hedge_exp) - _to_date(date)).days / 365.0
        t_rem = (_to_date(hog_exp) - _to_date(hedge_exp)).days / 365.0
        if not atm_iv or t1 <= 0:
            return None

        def pkg_pnl(s1: float) -> float:
            vert = max(k_long - s1, 0.0) - max(k_short - s1, 0.0) - debit
            hog_val = black_put(s1, k_hog, t_rem, hog_iv)  # sticky-strike IV
            return vert + (hog_credit - hog_val)

        # "downside price where the trade would START to lose money" = lower
        # edge of the profit zone (the deep-tail breakeven). The tiny droop
        # near flat spot is expected ("flatten the small upside droop", §5)
        # and is NOT the loss point this gate measures.
        breakeven = None
        s = spot * 0.5
        while s < spot:
            if pkg_pnl(s) >= 0:
                breakeven = s
                break
            s += spot * 0.0025
        if breakeven is None:
            return 0.0  # no profit zone below spot at all
        # lognormal, zero drift, ATM vol of the hedge expiry
        z = (math.log(spot / breakeven) - 0.5 * atm_iv * atm_iv * t1) / (
            atm_iv * math.sqrt(t1)
        )
        return float(norm.cdf(z))

    @staticmethod
    def _naked_put_margin(spot: float, k: float, credit: float) -> float:
        """Reg-T style: max(20% spot - OTM, 10% strike) + premium, per contract."""
        otm = max(spot - k, 0.0)
        return (max(0.20 * spot - otm, 0.10 * k) + credit) * 100

    # -------------------------------------------------- marks & exits

    # quote assumed for an OTM strike that has dropped off the tape
    _WORTHLESS: Quote = (0.0, 0.05, 0)

    def _marks(self, tr: HedgeHogTrade, date: int):
        """(vert_mark_mid, hog_mark_mid) or None if quotes missing.

        A leg absent from the chain while spot is above its strike is treated
        as nearly worthless (deep-OTM strikes drop off the tape); if spot is
        at/below the strike the quote is genuinely missing and we punt.
        """
        hc = self.store.chain(date, tr.hedge_expiry)
        gc = self.store.chain(date, tr.hog_expiry)
        if not hc or not gc:
            return None
        spot = self.store.spot(date)
        ql, qs = hc.get((tr.k_long, "P")), hc.get((tr.k_short, "P"))
        qh = gc.get((tr.k_hog, "P"))
        legs = []
        for q, k in ((ql, tr.k_long), (qs, tr.k_short), (qh, tr.k_hog)):
            if q is None:
                if spot is not None and spot > k:
                    q = self._WORTHLESS
                else:
                    return None
            legs.append(q)
        ql, qs, qh = legs
        return _mid(ql) - _mid(qs), _mid(qh), (ql, qs, qh)

    def check_exit(self, tr: HedgeHogTrade, date: int) -> Optional[str]:
        p = self.p
        held = (_to_date(date) - _to_date(tr.entry_date)).days
        hedge_dte = (_to_date(tr.hedge_expiry) - _to_date(date)).days

        spot = self.store.spot(date)
        if spot and spot < tr.k_short and held < p.fast_breach_days:
            return "fast_breach"

        m = self._marks(tr, date)
        if m is not None:
            vert_mark, hog_mark, _ = m
            if tr.hog_credit - hog_mark >= p.hog_tp_frac * tr.hog_credit:
                return "hog_50pct"
            width = tr.k_long - tr.k_short
            if vert_mark - tr.debit >= p.vert_tp_frac * (width - tr.debit):
                return "vertical_80pct"
        if hedge_dte < p.vert_close_dte:
            return "vertical_lt7dte"
        return None

    def close(self, tr: HedgeHogTrade, date: int, reason: str) -> bool:
        p = self.p
        m = self._marks(tr, date)
        if m is None:
            return False
        _, _, (ql, qs, qh) = m
        # unwind: sell long put, buy back short put, buy back hog
        proceeds = -_fill(qh, "buy", p.fill_k)
        legs = 1
        if p.trade_hedge:
            proceeds += _fill(ql, "sell", p.fill_k) - _fill(qs, "buy", p.fill_k)
            legs = 3
        per_contract = (tr.net_credit + proceeds) * 100
        costs = legs * (
            p.commission_open + p.commission_close + 2 * p.fees_per_leg
        )
        tr.exit_date = date
        tr.exit_reason = reason
        tr.spot_exit = self.store.spot(date)
        tr.pnl_usd = (per_contract - costs) * tr.qty
        tr.days_held = (_to_date(date) - _to_date(tr.entry_date)).days
        return True

    # -------------------------------------------------- main loop

    def run(self, start: int = 0, end: int = 99999999) -> List[HedgeHogTrade]:
        p = self.p
        dates = [d for d in self.store.quote_dates() if start <= d <= end]
        open_pos: List[HedgeHogTrade] = []
        last_entry: Optional[int] = None
        realized = 0.0

        for date in dates:
            # manage open positions
            still_open: List[HedgeHogTrade] = []
            for tr in open_pos:
                hedge_dte = (_to_date(tr.hedge_expiry) - _to_date(date)).days
                reason = self.check_exit(tr, date)
                if reason is None and hedge_dte <= 1:
                    reason = "vertical_lt7dte"  # hard fallback near expiry
                if reason and self.close(tr, date, reason):
                    realized += tr.pnl_usd
                    self.trades.append(tr)
                else:
                    if reason:  # quotes missing; retry next day
                        tr.daily_marks.append(float("nan"))
                    still_open.append(tr)
            open_pos = still_open

            # ladder entries
            if len(open_pos) < p.max_concurrent and (
                last_entry is None
                or (_to_date(date) - _to_date(last_entry)).days >= p.ladder_every_days
            ):
                tr = self.try_enter(date)
                if tr:
                    open_pos.append(tr)
                    last_entry = date

            # daily portfolio MTM
            mtm = realized
            for tr in open_pos:
                m = self._marks(tr, date)
                if m:
                    vert_mark, hog_mark, _ = m
                    open_val = tr.net_credit - hog_mark
                    if p.trade_hedge:
                        open_val += vert_mark
                    mtm += open_val * 100 * tr.qty
            self.daily_equity[date] = mtm

        # force-close anything left at the last date
        for tr in open_pos:
            for date in reversed(dates):
                if self.close(tr, date, "end_of_data"):
                    self.trades.append(tr)
                    break
        self.trades.sort(key=lambda t: t.entry_date)
        return self.trades


# ---------------------------------------------------------------- stats

def summarize(trades: List[HedgeHogTrade], daily_equity: Dict[int, float]) -> Dict:
    closed = [t for t in trades if t.pnl_usd is not None]
    if not closed:
        return {"trades": 0}
    pnls = [t.pnl_usd for t in closed]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    gross_w = sum(wins)
    gross_l = abs(sum(losses))

    eq = sorted(daily_equity.items())
    rets = [b - a for (_, a), (_, b) in zip(eq, eq[1:])]
    sharpe = 0.0
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
        if var > 0:
            sharpe = mean / math.sqrt(var) * math.sqrt(252)

    peak = dd = 0.0
    for _, v in eq:
        peak = max(peak, v)
        dd = max(dd, peak - v)

    margins = [t.margin_usd for t in closed if t.margin_usd]
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    years = max((_to_date(eq[-1][0]) - _to_date(eq[0][0])).days / 365.25, 1e-9)

    by_reason: Dict[str, Dict] = {}
    for t in closed:
        r = by_reason.setdefault(t.exit_reason, {"n": 0, "pnl": 0.0, "wins": 0})
        r["n"] += 1
        r["pnl"] += t.pnl_usd
        r["wins"] += 1 if t.pnl_usd > 0 else 0
    by_year: Dict[int, Dict] = {}
    for t in closed:
        y = t.exit_date // 10000
        r = by_year.setdefault(y, {"n": 0, "pnl": 0.0, "wins": 0})
        r["n"] += 1
        r["pnl"] += t.pnl_usd
        r["wins"] += 1 if t.pnl_usd > 0 else 0

    return {
        "trades": len(closed),
        "win_rate": len(wins) / len(closed),
        "total_pnl_usd": sum(pnls),
        "avg_pnl_usd": sum(pnls) / len(closed),
        "avg_win_usd": gross_w / len(wins) if wins else 0.0,
        "avg_loss_usd": -gross_l / len(losses) if losses else 0.0,
        "worst_trade_usd": min(pnls),
        "best_trade_usd": max(pnls),
        "profit_factor": gross_w / gross_l if gross_l > 0 else float("inf"),
        "avg_days_held": sum(t.days_held for t in closed) / len(closed),
        "avg_margin_usd": avg_margin,
        "ann_return_on_avg_margin": (
            (sum(pnls) / years) / avg_margin if avg_margin else 0.0
        ),
        "sharpe_annualised": sharpe,
        "max_drawdown_usd": dd,
        "trades_per_year": len(closed) / years,
        "by_exit_reason": by_reason,
        "by_year": by_year,
    }
